"""
Dice rolling engine for the Personal Dungeon Master (Phase 7).

Provides cryptographically seeded random rolls, LLM tag parsing, and
result formatting.  The tag syntax understood by the LLM is::

    [ROLL: <label> <N>d<X>[+<modifier>]]
    [ROLL: <label> <N>d<X>[-<modifier>]]

Examples::

    [ROLL: attack 1d20+5]
    [ROLL: damage 2d6]
    [ROLL: save 1d20-2]
"""

from __future__ import annotations

import re
import secrets

from random import Random

from src.dice.die import Die, RollRequest, RollResult

# ---------------------------------------------------------------------------
# Module-level PRNG seeded from the OS cryptographic source.
# Using a seeded Random (not the global random module) means:
#   1. Rolls are unpredictable to the LLM.
#   2. Tests can inject a deterministic instance if needed.
# ---------------------------------------------------------------------------
_rng: Random = Random(secrets.randbits(128))

# Regex that matches [ROLL: <label> <N>d<X>[+/-<mod>]]
# Label may contain letters, digits, spaces (captured lazily).
_TAG_RE = re.compile(
    r"\[ROLL:\s*(?P<label>[^\]]+?)\s+(?P<count>\d+)d(?P<faces>\d+)"
    r"(?P<sign>[+-])(?P<mod>\d+)?\s*\]",
    re.IGNORECASE,
)

# Faces value → Die member
_FACE_TO_DIE: dict[int, Die] = {d.value: d for d in Die}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def roll(req: RollRequest, *, _rng: Random = _rng) -> RollResult:
    """
    Execute a :class:`~src.dice.die.RollRequest` and return a
    :class:`~src.dice.die.RollResult`.

    Advantage / disadvantage applies to *single-die* rolls only
    (``count == 1``).  When multiple dice are requested with advantage the
    flag is ignored — the standard rule is that adv/disadv only applies to
    d20 checks which are always a single die.

    Parameters
    ----------
    req:
        The roll specification.
    _rng:
        Injected ``random.Random`` instance (used by tests for seeded rolls).

    Returns
    -------
    RollResult
        Complete result including all individual rolls and the final total.
    """
    faces = req.die.value

    if (req.advantage or req.disadvantage) and req.count == 1:
        # Roll twice and keep the relevant result
        a = _rng.randint(1, faces)
        b = _rng.randint(1, faces)
        all_rolls = [a, b]
        kept = max(a, b) if req.advantage else min(a, b)
        total = max(1, kept + req.modifier)
    else:
        all_rolls = [_rng.randint(1, faces) for _ in range(req.count)]
        total = max(1, sum(all_rolls) + req.modifier)

    return RollResult(
        die=req.die,
        rolls=all_rolls,
        modifier=req.modifier,
        total=total,
        label=req.label,
    )


def parse_roll_tags(text: str) -> list[RollRequest]:
    """
    Extract every ``[ROLL: ...]`` tag from *text* and return a list of
    :class:`~src.dice.die.RollRequest` objects in order of appearance.

    Tags that reference an unknown die type (e.g. ``d7``) are silently
    skipped — the raw tag is left in the text for ``substitute_rolls`` to
    leave unchanged.

    Parameters
    ----------
    text:
        Raw LLM response string.

    Returns
    -------
    list[RollRequest]
        Parsed roll requests.  May be empty.
    """
    requests: list[RollRequest] = []
    for m in _TAG_RE.finditer(text):
        faces = int(m.group("faces"))
        if faces not in _FACE_TO_DIE:
            continue
        sign = m.group("sign") or "+"
        mod_str = m.group("mod")
        modifier = int(mod_str) if mod_str else 0
        if sign == "-":
            modifier = -modifier
        requests.append(
            RollRequest(
                label=m.group("label").strip(),
                die=_FACE_TO_DIE[faces],
                count=int(m.group("count")),
                modifier=modifier,
            )
        )
    return requests


def substitute_rolls(text: str, results: list[RollResult]) -> str:
    """
    Replace each ``[ROLL: ...]`` tag in *text* with its formatted result.

    Tags are replaced in order of appearance.  If *results* contains fewer
    entries than tags, any remaining tags are left verbatim.

    Parameters
    ----------
    text:
        Raw LLM response string containing ``[ROLL: ...]`` tags.
    results:
        Roll results from :func:`roll`, in the same order as the tags.

    Returns
    -------
    str
        Text with all matching tags replaced by human-readable results.
    """
    result_iter = iter(results)
    known_faces = {d.value for d in Die}

    def _replacer(m: re.Match) -> str:
        faces = int(m.group("faces"))
        if faces not in known_faces:
            return m.group(0)  # leave unknown die tags verbatim
        result = next(result_iter, None)
        if result is None:
            return m.group(0)
        return format_result(result)

    return _TAG_RE.sub(_replacer, text)


def format_result(result: RollResult) -> str:
    """
    Produce a human-readable string for a single :class:`~src.dice.die.RollResult`.

    Examples::

        ✨ Attack Roll [d20+5]: rolled 14 + 5 = **19**
        ✨ Damage [2d6]: rolled [3, 5] = **8**
        ✨ Save [d20-2]: rolled 9 - 2 = **7**

    Parameters
    ----------
    result:
        The completed roll result.

    Returns
    -------
    str
        Formatted result suitable for terminal / chat display.
    """
    label = result.label.title()
    die_name = f"d{result.die.value}"
    count = len(result.rolls)

    if count == 1:
        roll_str = str(result.rolls[0])
    else:
        # For multi-die rolls show the individual results
        roll_str = "[" + ", ".join(str(r) for r in result.rolls) + "]"

    if result.modifier > 0:
        dice_expr = f"{count}{die_name}+{result.modifier}"
        calc = f"{roll_str} + {result.modifier}"
    elif result.modifier < 0:
        dice_expr = f"{count}{die_name}{result.modifier}"
        calc = f"{roll_str} - {abs(result.modifier)}"
    else:
        dice_expr = f"{count}{die_name}"
        calc = roll_str

    return f"✨ {label} [{dice_expr}]: rolled {calc} = **{result.total}**"


# Player-expression regex: [N]d<faces>[+/-mod]
_EXPR_RE = re.compile(
    r"^(?P<count>\d+)?d(?P<faces>\d+)(?:(?P<sign>[+-])(?P<mod>\d+))?$",
    re.IGNORECASE,
)


def parse_player_expression(expr: str, label: str = "roll") -> "RollRequest | None":
    """
    Parse a player-typed dice expression into a :class:`~src.dice.die.RollRequest`.

    Understands expressions like ``d20``, ``2d6``, ``d20+3``, ``1d8-1``.
    Returns ``None`` if the expression is not recognised or uses an unknown die.

    Parameters
    ----------
    expr:
        Raw expression string from the player (e.g. ``"d20+3"``).
    label:
        Label to attach to the roll (default ``"roll"``).

    Returns
    -------
    RollRequest | None
    """
    m = _EXPR_RE.match(expr.strip())
    if not m:
        return None
    faces = int(m.group("faces"))
    if faces not in _FACE_TO_DIE:
        return None
    count = int(m.group("count") or 1)
    sign = m.group("sign") or "+"
    mod_str = m.group("mod")
    modifier = int(mod_str) if mod_str else 0
    if sign == "-":
        modifier = -modifier
    return RollRequest(label=label, die=_FACE_TO_DIE[faces], count=count, modifier=modifier)
