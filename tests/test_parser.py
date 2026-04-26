"""Tests for the campaign parser (Phase 3)."""

import pytest

from src.campaign.loader import load_campaigns
from src.campaign.parser import (
    ParsedCampaign,
    _split_scenes,
    parse_campaign,
    parse_character,
    parse_creatures,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — parse the example campaign once per test session
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def example_campaign():
    from src.config import settings

    campaigns = load_campaigns(settings.campaigns_dir)
    return next(c for c in campaigns if c.name == "example-campaign")


@pytest.fixture(scope="module")
def parsed(example_campaign) -> ParsedCampaign:
    return parse_campaign(example_campaign)


# ─────────────────────────────────────────────────────────────────────────────
# Character tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParseCharacter:
    def test_name(self, parsed):
        assert parsed.character.name == "Torben Ashford"

    def test_level(self, parsed):
        assert parsed.character.level == 2

    def test_class(self, parsed):
        assert parsed.character.character_class == "Wizard"

    def test_subclass_contains_divination(self, parsed):
        assert "Divination" in parsed.character.subclass

    def test_hit_points(self, parsed):
        assert parsed.character.hit_points == 14

    def test_armor_class(self, parsed):
        assert parsed.character.armor_class == 12

    def test_speed(self, parsed):
        assert parsed.character.speed == 30

    def test_proficiency_bonus(self, parsed):
        assert parsed.character.proficiency_bonus == 2

    def test_ability_scores(self, parsed):
        scores = parsed.character.ability_scores
        assert scores.strength == 10
        assert scores.dexterity == 14
        assert scores.constitution == 12
        assert scores.intelligence == 16
        assert scores.wisdom == 13
        assert scores.charisma == 11

    def test_saving_throws_present(self, parsed):
        st = parsed.character.saving_throws
        assert "Intelligence" in st
        assert st["Intelligence"] == 5

    def test_skills_present(self, parsed):
        assert "Arcana" in parsed.character.skills
        assert parsed.character.skills["Arcana"] == 5

    def test_spellcasting_ability(self, parsed):
        assert parsed.character.spellcasting_ability == "Intelligence"

    def test_spell_save_dc(self, parsed):
        assert parsed.character.spell_save_dc == 13

    def test_spell_attack_bonus(self, parsed):
        assert parsed.character.spell_attack_bonus == 5

    def test_cantrips(self, parsed):
        assert "Mage Hand" in parsed.character.cantrips
        assert "Minor Illusion" in parsed.character.cantrips

    def test_spells_prepared(self, parsed):
        assert "Detect Magic" in parsed.character.spells_prepared
        assert "Shield" in parsed.character.spells_prepared

    def test_spell_slots(self, parsed):
        assert parsed.character.spell_slots.get("1st") == 3

    def test_equipment_present(self, parsed):
        items = [e["item"] for e in parsed.character.equipment]
        assert any("Quarterstaff" in i for i in items)

    def test_features_present(self, parsed):
        names = [f["name"] for f in parsed.character.features]
        assert any("Portent" in n for n in names)
        assert any("Researcher" in n for n in names)

    def test_raw_markdown_non_empty(self, parsed):
        assert len(parsed.character.raw_markdown) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Creature tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParseCreatures:
    def test_creature_count(self, parsed):
        # Vault Ooze, Coldstone's Warden, Millhaven Watch Guard
        assert len(parsed.creatures) == 3

    def test_vault_ooze_name(self, parsed):
        names = [c.name for c in parsed.creatures]
        assert "Vault Ooze" in names

    def test_warden_name(self, parsed):
        names = [c.name for c in parsed.creatures]
        assert any("Warden" in n for n in names)

    def test_vault_ooze_ac(self, parsed):
        ooze = next(c for c in parsed.creatures if c.name == "Vault Ooze")
        assert ooze.ac == 8

    def test_vault_ooze_hp(self, parsed):
        ooze = next(c for c in parsed.creatures if c.name == "Vault Ooze")
        assert ooze.hit_points == 22

    def test_warden_hp(self, parsed):
        warden = next(c for c in parsed.creatures if "Warden" in c.name)
        assert warden.hit_points == 60

    def test_warden_ac(self, parsed):
        warden = next(c for c in parsed.creatures if "Warden" in c.name)
        assert warden.ac == 20

    def test_creature_type_parsed(self, parsed):
        ooze = next(c for c in parsed.creatures if c.name == "Vault Ooze")
        assert "Ooze" in ooze.creature_type

    def test_vault_ooze_traits(self, parsed):
        ooze = next(c for c in parsed.creatures if c.name == "Vault Ooze")
        assert len(ooze.traits) >= 2
        trait_names = [t["name"] for t in ooze.traits]
        assert any("Amorphous" in n for n in trait_names)

    def test_vault_ooze_actions(self, parsed):
        ooze = next(c for c in parsed.creatures if c.name == "Vault Ooze")
        assert len(ooze.actions) >= 1

    def test_warden_ability_scores(self, parsed):
        warden = next(c for c in parsed.creatures if "Warden" in c.name)
        assert warden.ability_scores is not None
        assert warden.ability_scores.strength == 18

    def test_warden_condition_immunities(self, parsed):
        warden = next(c for c in parsed.creatures if "Warden" in c.name)
        assert len(warden.condition_immunities) > 0

    def test_warden_encounter_notes_non_empty(self, parsed):
        warden = next(c for c in parsed.creatures if "Warden" in c.name)
        assert len(warden.encounter_notes) > 0

    def test_vault_ooze_encounter_notes_non_empty(self, parsed):
        ooze = next(c for c in parsed.creatures if c.name == "Vault Ooze")
        assert len(ooze.encounter_notes) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Scene splitting tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSplitScenes:
    def test_scene_count(self, parsed):
        assert len(parsed.scenes) == 3

    def test_scene_title_content(self, parsed):
        assert "Millhaven" in parsed.scene_titles[0]

    def test_all_scenes_non_empty(self, parsed):
        for scene in parsed.scenes:
            assert len(scene.strip()) > 0

    def test_scene_text_contains_header(self, parsed):
        # Each scene text should start with its ## SCENE header
        assert parsed.scenes[0].startswith("## SCENE 1")

    def test_no_scene_markers_returns_single_scene(self):
        titles, scenes = _split_scenes("No scene markers here.\nJust plain text.")
        assert len(scenes) == 1
        assert titles == ["Scene 1"]


# ─────────────────────────────────────────────────────────────────────────────
# Integration test
# ─────────────────────────────────────────────────────────────────────────────


class TestParseCampaignIntegration:
    def test_summary_non_empty(self, parsed):
        assert len(parsed.summary) > 0

    def test_summary_contains_campaign_title(self, parsed):
        assert "Mira Coldstone" in parsed.summary

    def test_raw_book_non_empty(self, parsed):
        assert len(parsed.raw_book) > 0

    def test_returns_parsed_campaign_type(self, parsed):
        assert isinstance(parsed, ParsedCampaign)
