"""
Campaign parser — turns raw markdown and text files into structured Pydantic models.

Public API
----------
parse_character(markdown: str) -> Character
parse_creatures(markdown: str) -> list[Creature]
parse_campaign(campaign: Campaign) -> ParsedCampaign
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

# Keys that appear as header rows in markdown tables and should be skipped
# when parsing tables as key-value pairs.
_TABLE_HEADER_KEYS: frozenset[str] = frozenset({
    "field", "save", "skill", "item", "stat",
    "element", "details", "bonus", "value", "notes",
    # ability score column headers in creature stat tables
    "str", "dex", "con", "int", "wis", "cha",
})


def _strip_bold(text: str) -> str:
    """Remove ``**…**`` markdown bold markers."""
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", text).strip()


def _parse_int(text: str, default: int = 0) -> int:
    """Extract the first signed integer from *text* (handles '+2', '14 (desc)', '30 ft')."""
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else default


def _parse_kv_table(block: str) -> dict[str, str]:
    """
    Parse a two-column markdown table (``| Key | Value |``) into a plain dict.

    - Strips ``**…**`` bold markers from both key and value.
    - Skips separator rows (``|---|``) and known header-row keys.
    """
    result: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"\|\s*[-:]+", line):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 2:
            key = _strip_bold(parts[0])
            val = _strip_bold(parts[1])
            if key and key.lower() not in _TABLE_HEADER_KEYS:
                result[key] = val
    return result


def _parse_bonus_table(block: str) -> dict[str, int]:
    """Parse a two-column markdown table of ``| Name | +N |`` into ``{name: n}``."""
    result: dict[str, int] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"\|\s*[-:]+", line):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 2:
            key = _strip_bold(parts[0])
            if key and key.lower() not in _TABLE_HEADER_KEYS:
                result[key] = _parse_int(parts[1])
    return result


def _parse_bullet_list(block: str) -> list[str]:
    """Extract items from a markdown bullet list (``- item`` or ``* item``)."""
    return [
        m.group(1).strip()
        for line in block.splitlines()
        if (m := re.match(r"^\s*[-*]\s+(.+)", line))
    ]


def _section(text: str, heading: str, level: int = 2) -> str:
    """
    Extract content beneath a markdown heading (at the given level), up to
    (but not including) the next heading of the same or higher level.

    Uses a prefix match so ``_section(text, "Skills", 2)`` matches
    ``## Skills (Proficient)`` and ``_section(text, "Features", 2)`` matches
    ``## Features & Traits``.

    Returns an empty string if the heading is not found.
    """
    hashes = "#" * level
    pattern = re.compile(
        rf"^{re.escape(hashes)}\s+{re.escape(heading)}.*?$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return ""
    start = m.end()
    next_h = re.compile(r"^#{1," + str(level) + r"}\s", re.MULTILINE)
    nm = next_h.search(text, start)
    return text[start : nm.start() if nm else len(text)].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────


class AbilityScores(BaseModel):
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10


class Character(BaseModel):
    name: str
    character_class: str
    subclass: str = ""
    level: int
    race: str = ""
    background: str = ""
    alignment: str = ""
    hit_points: int
    armor_class: int
    speed: int
    proficiency_bonus: int
    passive_perception: int
    ability_scores: AbilityScores
    saving_throws: dict[str, int] = {}
    skills: dict[str, int] = {}
    spellcasting_ability: str = ""
    spell_save_dc: int = 0
    spell_attack_bonus: int = 0
    cantrips: list[str] = []
    spells_prepared: list[str] = []
    spell_slots: dict[str, int] = {}   # e.g. {"1st": 3}
    equipment: list[dict[str, str]] = []
    features: list[dict[str, str]] = []  # [{name, description}]
    raw_markdown: str


class Creature(BaseModel):
    name: str
    encounter_location: str = ""
    flavor_text: str = ""
    creature_type: str = ""
    size: str = ""
    ac: int = 0
    hit_points_raw: str = ""
    hit_points: int = 0
    speed: str = ""
    challenge: str = ""
    ability_scores: AbilityScores | None = None
    saving_throws: list[str] = []
    damage_immunities: list[str] = []
    condition_immunities: list[str] = []
    senses: list[str] = []
    traits: list[dict[str, str]] = []    # [{name, description}]
    actions: list[dict[str, str]] = []   # [{name, description}]
    encounter_notes: str = ""
    raw_markdown: str


@dataclass
class ParsedCampaign:
    summary: str
    character: Character
    creatures: list[Creature]
    scenes: list[str]        # each element is one scene's full text (including its header)
    scene_titles: list[str]  # e.g. ["Arrival in Millhaven", ...]
    raw_book: str


# ─────────────────────────────────────────────────────────────────────────────
# Character parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_class_field(raw: str) -> tuple[str, str]:
    """Split ``'Wizard (School of Divination)'`` → ``('Wizard', 'School of Divination')``."""
    m = re.match(r"^([^(]+?)\s*(?:\((.+)\))?$", raw.strip())
    if not m:
        return raw.strip(), ""
    return m.group(1).strip(), (m.group(2) or "").strip()


def _parse_core_stats(block: str) -> AbilityScores:
    """Parse the Core Stats table (``Stat | Score | Modifier`` columns)."""
    name_map = {
        "Strength": "strength",
        "Dexterity": "dexterity",
        "Constitution": "constitution",
        "Intelligence": "intelligence",
        "Wisdom": "wisdom",
        "Charisma": "charisma",
    }
    stats: dict[str, int] = {v: 10 for v in name_map.values()}
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"\|\s*[-:]+", line):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 2:
            key = _strip_bold(parts[0])
            if key in name_map:
                stats[name_map[key]] = _parse_int(parts[1])
    return AbilityScores(**stats)


def _parse_features(block: str) -> list[dict[str, str]]:
    """
    Parse ``**Name** — Description`` paragraphs (one feature per paragraph)
    into ``[{name, description}]``.
    """
    features: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for line in block.splitlines():
        m = re.match(r"^\*\*(.+?)\*\*\s*[—\-]+\s*(.*)", line)
        if m:
            if current:
                features.append(current)
            current = {
                "name": m.group(1).strip(),
                "description": m.group(2).strip(),
            }
        elif current:
            stripped = line.strip()
            if stripped:
                current["description"] += " " + stripped
            else:
                # Blank line ends the current feature
                features.append(current)
                current = None

    if current:
        features.append(current)
    return features


def parse_character(markdown: str) -> Character:
    """Parse a ``character.md`` file into a :class:`Character` model."""
    # Character name from first ## heading
    name_m = re.search(r"^##\s+(.+?)$", markdown, re.MULTILINE)
    name = name_m.group(1).strip() if name_m else "Unknown"

    # Character Overview (class, level, HP, AC, speed, etc.)
    overview = _parse_kv_table(_section(markdown, "Character Overview", 2))
    char_class, subclass = _parse_class_field(overview.get("Class", ""))

    # Core Stats
    ability_scores = _parse_core_stats(_section(markdown, "Core Stats", 2))

    # Saving Throws
    saving_throws = _parse_bonus_table(_section(markdown, "Saving Throws", 2))

    # Skills — heading may be "Skills (Proficient)"; _section uses prefix match
    skills = _parse_bonus_table(_section(markdown, "Skills", 2))

    # Spellcasting
    sc_block = _section(markdown, "Spellcasting", 2)
    sc_kv = _parse_kv_table(sc_block)
    spell_slots_raw = _parse_kv_table(_section(sc_block, "Spell Slots", 3))
    spell_slots: dict[str, int] = {}
    for k, v in spell_slots_raw.items():
        try:
            spell_slots[k] = int(v)
        except ValueError:
            pass
    cantrips = _parse_bullet_list(_section(sc_block, "Cantrips Known", 3))
    # Heading may be "Spells Prepared (4)"; _section uses prefix match
    spells_prepared = _parse_bullet_list(_section(sc_block, "Spells Prepared", 3))

    # Equipment — custom two-column parser (Item | Notes)
    equipment: list[dict[str, str]] = []
    for line in _section(markdown, "Equipment", 2).splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"\|\s*[-:]+", line):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 2 and parts[0].lower() not in {"item", "field"}:
            equipment.append({
                "item": _strip_bold(parts[0]),
                "notes": _strip_bold(parts[1]),
            })

    # Features & Traits — heading may be "Features & Traits"; _section uses prefix match
    features = _parse_features(_section(markdown, "Features", 2))

    return Character(
        name=name,
        character_class=char_class,
        subclass=subclass,
        level=_parse_int(overview.get("Level", "1"), default=1),
        race=overview.get("Race", ""),
        background=overview.get("Background", ""),
        alignment=overview.get("Alignment", ""),
        hit_points=_parse_int(overview.get("Hit Points", "0")),
        armor_class=_parse_int(overview.get("Armor Class", "10")),
        speed=_parse_int(overview.get("Speed", "30")),
        proficiency_bonus=_parse_int(overview.get("Proficiency Bonus", "2")),
        passive_perception=_parse_int(overview.get("Passive Perception", "10")),
        ability_scores=ability_scores,
        saving_throws=saving_throws,
        skills=skills,
        spellcasting_ability=sc_kv.get("Spellcasting Ability", ""),
        spell_save_dc=_parse_int(sc_kv.get("Spell Save DC", "0")),
        spell_attack_bonus=_parse_int(sc_kv.get("Spell Attack Bonus", "0")),
        cantrips=cantrips,
        spells_prepared=spells_prepared,
        spell_slots=spell_slots,
        equipment=equipment,
        features=features,
        raw_markdown=markdown,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Creature parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_creature_stats_table(block: str) -> AbilityScores | None:
    """
    Parse the ``| STR | DEX | CON | INT | WIS | CHA |`` table that appears in
    creature stat blocks.  Returns ``None`` if the table is not present.
    """
    stat_keys = {"STR", "DEX", "CON", "INT", "WIS", "CHA"}
    in_stats = False
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"\|\s*[-:]+", line):
            continue
        parts = [_strip_bold(p.strip()) for p in line.strip("|").split("|")]
        # Detect the header row
        if all(p in stat_keys for p in parts if p):
            in_stats = True
            continue
        if in_stats:
            vals = [_parse_int(p) for p in parts if p]
            if len(vals) >= 6:
                return AbilityScores(
                    strength=vals[0],
                    dexterity=vals[1],
                    constitution=vals[2],
                    intelligence=vals[3],
                    wisdom=vals[4],
                    charisma=vals[5],
                )
            in_stats = False
    return None


def _parse_named_abilities(block: str) -> list[dict[str, str]]:
    """
    Parse bullet items of the form ``- **Name** — description`` or
    ``- **Name.** description`` into ``[{name, description}]``.
    """
    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for line in block.splitlines():
        m = re.match(r"^\s*[-*]\s+\*\*(.+?)\*\*[.\s]*(?:[—\-]\s*)?(.*)", line)
        if m:
            if current:
                items.append(current)
            current = {
                "name": m.group(1).strip().rstrip("."),
                "description": m.group(2).strip(),
            }
        elif current and line.strip() and not line.strip().startswith("-"):
            current["description"] += " " + line.strip()

    if current:
        items.append(current)
    return items


def _flavor_text(block: str) -> str:
    """
    Extract the flavor text paragraphs that precede the first markdown table
    (``|``) or sub-section (``###``) in a creature block.
    """
    # Skip the ## header line
    newline = block.find("\n")
    rest = block[newline + 1:].strip() if newline != -1 else ""
    boundary = re.search(r"^(\||#{3})", rest, re.MULTILINE)
    return rest[: boundary.start()].strip() if boundary else rest.strip()


def _parse_creature_block(block: str) -> Creature:
    """Parse a single creature block (from its ``##`` header to end of block)."""
    # Name and optional location from ``## Name (Location)``
    header_m = re.match(r"^##\s+(.+?)(?:\s+\((.+?)\))?\s*$", block, re.MULTILINE)
    if header_m:
        name = header_m.group(1).strip()
        location = (header_m.group(2) or "").strip()
    else:
        name, location = "Unknown", ""

    flavor = _flavor_text(block)

    # Main key-value table (Type, Size, AC, Hit Points, Speed, Challenge)
    kv = _parse_kv_table(block)

    # Ability scores — look in ### Stats first, then full block
    stats_section = _section(block, "Stats", 3)
    ability_scores = (
        _parse_creature_stats_table(stats_section)
        if stats_section
        else _parse_creature_stats_table(block)
    )

    # Optional subsections
    saving_throws = _parse_bullet_list(_section(block, "Saving Throws", 3))
    damage_immunities = _parse_bullet_list(_section(block, "Damage Immunities", 3))
    condition_immunities = _parse_bullet_list(_section(block, "Condition Immunities", 3))
    senses = _parse_bullet_list(_section(block, "Senses", 3))
    traits = _parse_named_abilities(_section(block, "Traits", 3))
    actions = _parse_named_abilities(_section(block, "Actions", 3))
    encounter_notes = (
        _section(block, "Encounter Notes", 3)
        or _section(block, "Non-Combat Resolution", 3)
    )

    hp_raw = kv.get("Hit Points", "")

    return Creature(
        name=name,
        encounter_location=location,
        flavor_text=flavor,
        creature_type=kv.get("Type", ""),
        size=kv.get("Size", ""),
        ac=_parse_int(kv.get("AC", "0")),
        hit_points_raw=hp_raw,
        hit_points=_parse_int(hp_raw),
        speed=kv.get("Speed", ""),
        challenge=kv.get("Challenge", ""),
        ability_scores=ability_scores,
        saving_throws=saving_throws,
        damage_immunities=damage_immunities,
        condition_immunities=condition_immunities,
        senses=senses,
        traits=traits,
        actions=actions,
        encounter_notes=encounter_notes,
        raw_markdown=block,
    )


def parse_creatures(markdown: str) -> list[Creature]:
    """Parse a ``creature.md`` file into a list of :class:`Creature` models."""
    # Each creature starts at a level-2 heading (##)
    boundaries = [
        m.start()
        for m in re.finditer(r"^##\s+\S", markdown, re.MULTILINE)
    ]
    if not boundaries:
        return []
    blocks = [
        markdown[start : (boundaries[i + 1] if i + 1 < len(boundaries) else len(markdown))].strip()
        for i, start in enumerate(boundaries)
    ]
    return [_parse_creature_block(b) for b in blocks]


# ─────────────────────────────────────────────────────────────────────────────
# Scene splitting
# ─────────────────────────────────────────────────────────────────────────────


def _split_scenes(raw_book: str) -> tuple[list[str], list[str]]:
    """
    Split the campaign book text into ``(scene_titles, scene_texts)``.

    Scene boundaries are detected by lines matching ``## SCENE N — Title``.
    If no ``SCENE`` markers are found, the entire book is returned as a single scene.
    """
    pattern = re.compile(
        r"^(##\s+SCENE\s+\d+\s*[—\-]\s*(.+?)\s*)$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(raw_book))
    if not matches:
        return ["Scene 1"], [raw_book]

    titles: list[str] = []
    scenes: list[str] = []
    for i, m in enumerate(matches):
        titles.append(m.group(2).strip())
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_book)
        scenes.append(raw_book[start:end].strip())

    return titles, scenes


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────


def parse_campaign(campaign: "Campaign") -> ParsedCampaign:  # noqa: F821
    """Parse all files for *campaign* into a :class:`ParsedCampaign`."""
    from src.campaign.loader import Campaign  # local import avoids any future circular deps

    summary = campaign.readme_path.read_text(encoding="utf-8")
    character = parse_character(campaign.character_path.read_text(encoding="utf-8"))
    creatures = parse_creatures(campaign.creature_path.read_text(encoding="utf-8"))
    raw_book = campaign.book_path.read_text(encoding="utf-8")
    scene_titles, scenes = _split_scenes(raw_book)

    return ParsedCampaign(
        summary=summary,
        character=character,
        creatures=creatures,
        scenes=scenes,
        scene_titles=scene_titles,
        raw_book=raw_book,
    )
