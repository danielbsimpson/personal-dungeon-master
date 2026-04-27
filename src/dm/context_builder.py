"""
Context builder — assembles the system prompt injected into the DM's LLM call
each turn.

The system prompt is built from these sections (in order):
  1. DM persona and behavioural instructions
  2. Relevant rules for the current narrative state
  3. Campaign summary
  4. Player character sheet
  5. Creature reference
  6. Revealed campaign book (via spoiler guard)
  7. Retrieved graph memory (via MemoryManager.get_context)

The session window (short-term messages) is NOT part of the system prompt —
it is passed separately as the ``messages`` list in the LLM call.

Public API
----------
detect_narrative_state(text) -> NarrativeState
build_system_prompt(campaign, rules, memory, state, current_text, token_budget) -> str
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.campaign.parser import Character, Creature, ParsedCampaign
from src.dm.memory.manager import MemoryManager
from src.dm.spoiler_guard import revealed_text
from src.rules.loader import RulesReference
from src.rules.reference import NarrativeState, get_relevant_rules

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Token budget helpers
# ─────────────────────────────────────────────────────────────────────────────

# Rough approximation: 1 token ≈ 4 characters for English prose.
_CHARS_PER_TOKEN: int = 4

# Default system-prompt token budget.  Leaves room for the session window and
# the model's output within a typical 8 K context window.
# Phase 9 replaces this with model-aware token counting.
_DEFAULT_TOKEN_BUDGET: int = 6_000

# ─────────────────────────────────────────────────────────────────────────────
# Narrative state detection
# ─────────────────────────────────────────────────────────────────────────────

_COMBAT_KEYWORDS: frozenset[str] = frozenset({
    "attack", "initiative", "damage", "strike", "combat", "fight", "battle",
    "wound", "kill", "slay", "dodge", "ambush", "i draw", "i swing",
    "charge", "retreat", "flanking",
})

_REST_KEYWORDS: frozenset[str] = frozenset({
    "long rest", "short rest", "take a rest", "make camp", "set up camp",
    "sleep", "we rest", "i rest",
})

_SOCIAL_KEYWORDS: frozenset[str] = frozenset({
    "persuade", "deceive", "intimidate", "negotiate", "i talk", "i speak",
    "convince", "tavern", "inn", "merchant",
})


def detect_narrative_state(text: str) -> NarrativeState:
    """
    Infer the current :class:`~src.rules.reference.NarrativeState` from the
    most recent turn text (player input or DM response).

    Uses lightweight keyword matching — the DM agent may override this at any
    time by passing an explicit *state* to :func:`build_system_prompt`.

    Priority: COMBAT > REST > SOCIAL > EXPLORATION.
    """
    lower = text.lower()
    if any(kw in lower for kw in _COMBAT_KEYWORDS):
        return NarrativeState.COMBAT
    if any(kw in lower for kw in _REST_KEYWORDS):
        return NarrativeState.REST
    if any(kw in lower for kw in _SOCIAL_KEYWORDS):
        return NarrativeState.SOCIAL
    return NarrativeState.EXPLORATION


# ─────────────────────────────────────────────────────────────────────────────
# Prompt section formatters
# ─────────────────────────────────────────────────────────────────────────────

_DM_PERSONA: str = """\
You are the Dungeon Master for a tabletop RPG adventure. Your responsibilities:

- Narrate scenes, locations, and NPC dialogue in vivid, immersive prose.
- Stay faithful to the campaign source material. Do not invent locations, NPCs, \
or plot events that contradict the campaign book.
- NEVER reveal future events, locations, enemies, or plot details that the player \
has not yet encountered. Only reference content up to the current scene.
- Apply the correct game rules for all mechanical actions (attack rolls, saving \
throws, ability checks, conditions, spell effects). Use the rules reference provided.
- When a dice roll is required, output a roll tag inline: \
[ROLL: <label> <NdX+modifier>]. Examples: [ROLL: attack d20+5], [ROLL: damage 2d6].
- Keep responses immersive and appropriately paced. Match the campaign's tone.
- Address the player in second person ("You see...", "You hear...").
- When NPCs speak, use dialogue with quotation marks and in-character voice.
- Never break the fourth wall or refer to yourself as an AI.\
"""


def _format_character(char: Character) -> str:
    """Format a :class:`Character` into a concise text block for the system prompt."""
    ab = char.ability_scores
    class_line = char.character_class
    if char.subclass:
        class_line += f" ({char.subclass})"
    lines = [
        f"**{char.name}** — {class_line}, Level {char.level}",
        (
            f"Race: {char.race or '—'}  |  Background: {char.background or '—'}  "
            f"|  Alignment: {char.alignment or '—'}"
        ),
        (
            f"HP: {char.hit_points}  |  AC: {char.armor_class}  "
            f"|  Speed: {char.speed} ft  |  Proficiency: +{char.proficiency_bonus}  "
            f"|  Passive Perception: {char.passive_perception}"
        ),
        "",
        (
            f"STR {ab.strength}  DEX {ab.dexterity}  CON {ab.constitution}  "
            f"INT {ab.intelligence}  WIS {ab.wisdom}  CHA {ab.charisma}"
        ),
    ]
    if char.equipment:
        lines.append(
            "**Equipment:** " + ", ".join(e["item"] for e in char.equipment)
        )
    if char.features:
        lines.append(
            "**Features:** " + ", ".join(f["name"] for f in char.features)
        )
    if char.spells_prepared:
        lines.append("**Spells Prepared:** " + ", ".join(char.spells_prepared))
    return "\n".join(lines)


def _format_creatures(creatures: list[Creature]) -> str:
    """Format the creature list into a concise reference block."""
    if not creatures:
        return "(No creatures defined for this campaign.)"
    parts: list[str] = []
    for c in creatures:
        header = f"### {c.name}"
        if c.creature_type:
            header += f" ({c.creature_type})"
        parts.append(header)
        if c.flavor_text:
            parts.append(c.flavor_text.strip())
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────


async def build_system_prompt(
    campaign: ParsedCampaign,
    rules: RulesReference,
    memory: MemoryManager,
    state: NarrativeState = NarrativeState.EXPLORATION,
    current_text: str = "",
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
) -> str:
    """
    Assemble the full system prompt for a DM turn.

    Parameters
    ----------
    campaign:
        Parsed campaign data (summary, character, creatures, scenes).
    rules:
        Loaded rules reference for the configured edition.
    memory:
        Initialised :class:`MemoryManager` — ``load()`` must have been called.
    state:
        Current narrative state — drives which rules sections are included.
    current_text:
        The player's current input, used for keyword-based rules selection and
        as the graph search query.
    token_budget:
        Approximate maximum token count for the system prompt.  When the
        assembled prompt exceeds this, the revealed campaign text is truncated
        to fit within budget (keeping the most recent content).

    Returns
    -------
    str
        The complete system prompt string.
    """
    # 1. DM persona
    persona_section = _DM_PERSONA.strip()

    # 2. Relevant rules
    rules_text = get_relevant_rules(rules, state, current_text)
    rules_section = f"## RULES REFERENCE\n\n{rules_text}"

    # 3. Campaign summary
    summary_section = f"## CAMPAIGN SUMMARY\n\n{campaign.summary.strip()}"

    # 4. Character sheet
    char_section = f"## PLAYER CHARACTER\n\n{_format_character(campaign.character)}"

    # 5. Creature reference
    creatures_section = (
        f"## CREATURE REFERENCE\n\n{_format_creatures(campaign.creatures)}"
    )

    # 6. Revealed campaign book (spoiler-guarded)
    book_text = revealed_text(campaign.scenes, memory.campaign_progress)
    book_section = f"## CAMPAIGN BOOK (revealed scenes)\n\n{book_text.strip()}"

    # 7. Graph memory context (async retrieval)
    query = current_text or campaign.summary
    memory_ctx = await memory.get_context(query, group_id="")
    memory_section = f"## MEMORY\n\n{memory_ctx}" if memory_ctx else ""

    # Assemble — memory section is omitted when empty
    static_sections = [
        persona_section,
        rules_section,
        summary_section,
        char_section,
        creatures_section,
        book_section,
    ]
    if memory_section:
        static_sections.append(memory_section)

    prompt = "\n\n---\n\n".join(static_sections)

    # Token budget guard: truncate the book section if over budget
    estimated_tokens = len(prompt) // _CHARS_PER_TOKEN
    if estimated_tokens > token_budget:
        log.warning(
            "System prompt ~%d tokens exceeds budget of %d. "
            "Truncating revealed campaign text.",
            estimated_tokens,
            token_budget,
        )
        static_chars = sum(
            len(s)
            for s in [
                persona_section,
                rules_section,
                summary_section,
                char_section,
                creatures_section,
            ]
        )
        sep_chars = len("\n\n---\n\n") * len(static_sections)
        budget_chars = token_budget * _CHARS_PER_TOKEN
        allowed_book_chars = max(200, budget_chars - static_chars - sep_chars)
        truncated_book = book_text[-allowed_book_chars:]
        book_section = (
            f"## CAMPAIGN BOOK (revealed scenes — truncated)\n\n"
            f"[...earlier content omitted...]\n\n{truncated_book.strip()}"
        )
        static_sections = [
            persona_section,
            rules_section,
            summary_section,
            char_section,
            creatures_section,
            book_section,
        ]
        if memory_section:
            static_sections.append(memory_section)
        prompt = "\n\n---\n\n".join(static_sections)

    return prompt
