"""
Rules reference helpers — retrieve relevant rules sections to inject into the
DM system prompt based on the current narrative state.

Public API
----------
NarrativeState          Enum driving section selection
get_all_rules(ref)      Full rules text — use for small context windows
get_relevant_rules(ref, state, context)  State-aware subset — use for large campaigns
search_rules(ref, topic)  Keyword search across all rule sections
"""

from __future__ import annotations

import re
from enum import Enum

from src.rules.loader import RulesReference


# ─────────────────────────────────────────────────────────────────────────────
# Narrative state
# ─────────────────────────────────────────────────────────────────────────────

class NarrativeState(Enum):
    """Current mode of play, used to select which rules sections are relevant."""

    EXPLORATION = "exploration"
    COMBAT = "combat"
    SOCIAL = "social"
    REST = "rest"


# ─────────────────────────────────────────────────────────────────────────────
# Section selection map
#
# core is ALWAYS included — it is the baseline for every state.
# Additional sections are added per state.
# ─────────────────────────────────────────────────────────────────────────────

_ALWAYS_INCLUDE: frozenset[str] = frozenset({"core"})

_STATE_SECTIONS: dict[NarrativeState, frozenset[str]] = {
    NarrativeState.EXPLORATION: frozenset({"equipment"}),
    NarrativeState.COMBAT: frozenset({"combat", "conditions", "equipment"}),
    NarrativeState.SOCIAL: frozenset({}),
    NarrativeState.REST: frozenset({}),
}

# Sections that are included when the word "spell" or "magic" appears in the
# context, regardless of narrative state.
_SPELLCASTING_KEYWORDS: frozenset[str] = frozenset({
    "spell", "magic", "cast", "cantrip", "concentration", "ritual",
    "arcane", "divine", "sorcery", "warlock", "wizard", "cleric", "druid",
    "bard", "paladin", "ranger",
})


def _sections_for_state(
    ref: RulesReference,
    state: NarrativeState,
    context: str = "",
) -> list[str]:
    """
    Return section names that should be included for *state*, filtered to those
    actually present in *ref*.  ``core`` is always first.

    If *context* contains spellcasting keywords, ``spellcasting`` is added.
    """
    wanted = _ALWAYS_INCLUDE | _STATE_SECTIONS.get(state, frozenset())

    if context:
        lower = context.lower()
        if any(kw in lower for kw in _SPELLCASTING_KEYWORDS):
            wanted = wanted | frozenset({"spellcasting"})

    # Preserve "core" first, then alphabetical for the rest
    present = [s for s in wanted if s in ref.sections]
    present.sort(key=lambda s: (0 if s == "core" else 1, s))
    return present


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────


def get_all_rules(ref: RulesReference) -> str:
    """
    Return all loaded rules sections concatenated as a single string, each
    preceded by a labelled header.

    Suitable for smaller models or short campaigns where fitting everything in
    context is feasible.
    """
    parts: list[str] = []
    for name in ref.section_names:
        parts.append(f"## Rules: {name.upper()}\n\n{ref.sections[name].strip()}")
    return "\n\n---\n\n".join(parts)


def get_relevant_rules(
    ref: RulesReference,
    state: NarrativeState,
    context: str = "",
) -> str:
    """
    Return only the rules sections relevant to the current *state*, plus any
    triggered by keywords in *context*.

    ``core`` is always included as the first section.

    Parameters
    ----------
    ref:
        The loaded :class:`RulesReference`.
    state:
        The current :class:`NarrativeState` (EXPLORATION / COMBAT / SOCIAL / REST).
    context:
        Optional text from the current turn (player input + last DM response).
        Used to detect spellcasting keywords and include ``spellcasting.md``.
    """
    section_names = _sections_for_state(ref, state, context)
    parts: list[str] = []
    for name in section_names:
        parts.append(f"## Rules: {name.upper()}\n\n{ref.sections[name].strip()}")
    return "\n\n---\n\n".join(parts)


def search_rules(ref: RulesReference, topic: str) -> str:
    """
    Search all rules sections for content matching *topic*.

    Splits each section into chunks at heading boundaries and returns all
    chunks that mention the topic keyword (case-insensitive).  Used by the
    ``/rules`` CLI command so the player can look up a specific mechanic
    (e.g. ``/rules grapple``) mid-session.

    Parameters
    ----------
    ref:
        The loaded :class:`RulesReference`.
    topic:
        The keyword or short phrase to search for.

    Returns
    -------
    str
        Matching rule excerpts, each preceded by its section name, separated
        by dividers.  Returns a "not found" message if no matches exist.
    """
    if not topic.strip():
        return "No topic provided. Usage: /rules <topic>  e.g. /rules grapple"

    topic_lower = topic.strip().lower()
    matches: list[str] = []

    for section_name in ref.section_names:
        text = ref.sections[section_name]
        # Split at headings (##, ###, ####) while keeping the heading itself.
        chunks = re.split(r"(?=\n#{2,4} )", text)
        for chunk in chunks:
            if topic_lower in chunk.lower():
                header = f"[{section_name.upper()}]"
                matches.append(f"{header}\n{chunk.strip()}")

    if not matches:
        return (
            f"No rules found for '{topic}'.\n"
            f"Available sections: {', '.join(ref.section_names)}"
        )

    return "\n\n---\n\n".join(matches)
