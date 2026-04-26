"""Tests for the rules loader and reference system (Phase 4)."""

import pytest
from pathlib import Path
from dataclasses import dataclass

from src.rules.loader import RulesReference, load_rules
from src.rules.reference import NarrativeState, get_all_rules, get_relevant_rules


# ─────────────────────────────────────────────────────────────────────────────
# Minimal Settings stub — mirrors the fields load_rules reads
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _MockSettings:
    game_edition: str
    rules_edition_dir: Path


# ─────────────────────────────────────────────────────────────────────────────
# Loader tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadRules:
    def test_loads_all_five_5e_sections(self):
        """The real rules/5e/ directory must contain all five expected sections."""
        ref = load_rules()
        assert set(ref.section_names) >= {"core", "combat", "conditions", "spellcasting", "equipment"}

    def test_edition_is_set(self):
        ref = load_rules()
        assert ref.edition == "5e"

    def test_sections_are_non_empty(self):
        ref = load_rules()
        for name, text in ref.sections.items():
            assert len(text.strip()) > 0, f"Section '{name}' is empty"

    def test_section_names_are_lowercase_stems(self):
        ref = load_rules()
        for name in ref.section_names:
            assert name == name.lower(), f"Section name '{name}' is not lowercase"
            assert "." not in name, f"Section name '{name}' should not include extension"

    def test_unknown_edition_raises_file_not_found(self, tmp_path):
        settings = _MockSettings(
            game_edition="pathfinder2e",
            rules_edition_dir=tmp_path / "rules" / "pathfinder2e",
        )
        with pytest.raises(FileNotFoundError, match="pathfinder2e"):
            load_rules(settings)

    def test_empty_edition_dir_raises_value_error(self, tmp_path):
        edition_dir = tmp_path / "rules" / "5e"
        edition_dir.mkdir(parents=True)
        settings = _MockSettings(game_edition="5e", rules_edition_dir=edition_dir)
        with pytest.raises(ValueError, match="No rule files"):
            load_rules(settings)

    def test_custom_rules_dir_loads_correctly(self, tmp_path):
        edition_dir = tmp_path / "rules" / "homebrew"
        edition_dir.mkdir(parents=True)
        (edition_dir / "core.md").write_text("# Homebrew Core\nSome rules.", encoding="utf-8")
        (edition_dir / "combat.md").write_text("# Homebrew Combat\nFight rules.", encoding="utf-8")
        settings = _MockSettings(game_edition="homebrew", rules_edition_dir=edition_dir)
        ref = load_rules(settings)
        assert ref.edition == "homebrew"
        assert "core" in ref.sections
        assert "combat" in ref.sections
        assert "Homebrew Core" in ref.sections["core"]

    def test_section_names_property_is_sorted(self):
        ref = load_rules()
        assert ref.section_names == sorted(ref.section_names)


# ─────────────────────────────────────────────────────────────────────────────
# Reference tests
# ─────────────────────────────────────────────────────────────────────────────


def _make_ref(sections: dict[str, str]) -> RulesReference:
    return RulesReference(edition="test", sections=sections)


class TestGetAllRules:
    def test_contains_all_sections(self):
        ref = _make_ref({"core": "core text", "combat": "combat text"})
        result = get_all_rules(ref)
        assert "core text" in result
        assert "combat text" in result

    def test_includes_section_labels(self):
        ref = _make_ref({"core": "core text"})
        result = get_all_rules(ref)
        assert "CORE" in result

    def test_real_rules_all_sections_present(self):
        ref = load_rules()
        result = get_all_rules(ref)
        for name in ref.section_names:
            assert name.upper() in result


class TestGetRelevantRules:
    def test_core_always_included_exploration(self):
        ref = _make_ref({"core": "core text", "combat": "combat text"})
        result = get_relevant_rules(ref, NarrativeState.EXPLORATION)
        assert "core text" in result

    def test_core_always_included_combat(self):
        ref = _make_ref({"core": "core text", "combat": "combat text"})
        result = get_relevant_rules(ref, NarrativeState.COMBAT)
        assert "core text" in result

    def test_core_always_included_social(self):
        ref = _make_ref({"core": "core text"})
        result = get_relevant_rules(ref, NarrativeState.SOCIAL)
        assert "core text" in result

    def test_core_always_included_rest(self):
        ref = _make_ref({"core": "core text"})
        result = get_relevant_rules(ref, NarrativeState.REST)
        assert "core text" in result

    def test_combat_state_includes_combat_section(self):
        ref = _make_ref({"core": "c", "combat": "combat text", "conditions": "cond text"})
        result = get_relevant_rules(ref, NarrativeState.COMBAT)
        assert "combat text" in result

    def test_combat_state_includes_conditions(self):
        ref = _make_ref({"core": "c", "combat": "combat text", "conditions": "cond text"})
        result = get_relevant_rules(ref, NarrativeState.COMBAT)
        assert "cond text" in result

    def test_social_state_excludes_combat(self):
        ref = _make_ref({"core": "c", "combat": "combat text"})
        result = get_relevant_rules(ref, NarrativeState.SOCIAL)
        assert "combat text" not in result

    def test_exploration_excludes_combat(self):
        ref = _make_ref({"core": "c", "combat": "combat text"})
        result = get_relevant_rules(ref, NarrativeState.EXPLORATION)
        assert "combat text" not in result

    def test_spellcasting_added_when_keyword_in_context(self):
        ref = _make_ref({"core": "c", "spellcasting": "spell text"})
        result = get_relevant_rules(ref, NarrativeState.EXPLORATION, context="I cast a spell")
        assert "spell text" in result

    def test_spellcasting_not_added_without_keyword(self):
        ref = _make_ref({"core": "c", "spellcasting": "spell text"})
        result = get_relevant_rules(ref, NarrativeState.EXPLORATION, context="I look around the room")
        assert "spell text" not in result

    def test_missing_section_gracefully_skipped(self):
        """If a state requests a section not in the ref, no KeyError is raised."""
        ref = _make_ref({"core": "c"})  # no combat or conditions
        result = get_relevant_rules(ref, NarrativeState.COMBAT)
        assert "c" in result  # core still present

    def test_core_appears_first_in_output(self):
        ref = _make_ref({"core": "CORE_TEXT", "combat": "COMBAT_TEXT", "conditions": "COND_TEXT"})
        result = get_relevant_rules(ref, NarrativeState.COMBAT)
        assert result.index("CORE_TEXT") < result.index("COMBAT_TEXT")

    def test_real_rules_combat_state(self):
        ref = load_rules()
        result = get_relevant_rules(ref, NarrativeState.COMBAT)
        assert "CORE" in result
        assert "COMBAT" in result
        assert "CONDITIONS" in result

    def test_real_rules_social_state_has_core_only(self):
        ref = load_rules()
        result = get_relevant_rules(ref, NarrativeState.SOCIAL)
        assert "CORE" in result
        # combat section should not be present in social state without keywords
        assert "COMBAT" not in result
