"""Tests for the DM agent core (Phase 6)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.campaign.parser import AbilityScores, Character, Creature, ParsedCampaign
from src.dm.context_builder import build_system_prompt, detect_narrative_state
from src.dm.dungeon_master import DungeonMaster
from src.dm.spoiler_guard import revealed_text
from src.rules.loader import RulesReference
from src.rules.reference import NarrativeState


# ─────────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_character() -> Character:
    return Character(
        name="Aldric",
        character_class="Fighter",
        level=1,
        hit_points=12,
        armor_class=16,
        speed=30,
        proficiency_bonus=2,
        passive_perception=11,
        ability_scores=AbilityScores(strength=16, dexterity=12, constitution=14),
        raw_markdown="",
    )


def _make_campaign() -> ParsedCampaign:
    return ParsedCampaign(
        summary="A brave hero ventures into the Dungeon of Dread.",
        character=_make_character(),
        creatures=[
            Creature(name="Goblin", creature_type="Humanoid", flavor_text="Small and cunning.", raw_markdown=""),
        ],
        scenes=[
            "## Scene One: The Village Gate\nYou arrive at the village gate.",
            "## Scene Two: The Forest Path\nA goblin steps out of the shadows.",
            "## Scene Three: The Dungeon\nThe ancient door creaks open.",
        ],
        scene_titles=["Scene One: The Village Gate", "Scene Two: The Forest Path", "Scene Three: The Dungeon"],
        raw_book=(
            "## Scene One: The Village Gate\nYou arrive at the village gate.\n\n"
            "## Scene Two: The Forest Path\nA goblin steps out of the shadows.\n\n"
            "## Scene Three: The Dungeon\nThe ancient door creaks open."
        ),
    )


def _make_rules() -> RulesReference:
    """Minimal RulesReference that satisfies get_relevant_rules without real files."""
    return RulesReference(
        edition="5e",
        sections={
            "core": "## Core Rules\nProficiency bonus: +2. Ability checks: d20 + modifier.",
            "combat": "## Combat\nInitiative: d20 + DEX modifier. Attack roll: d20 + mod.",
            "conditions": "## Conditions\nBlinded: can't see.",
            "equipment": "## Equipment\nLongsword: 1d8 slashing.",
        },
    )


def _make_memory(progress: int = 0, search_result: str = "") -> MagicMock:
    """Return a MagicMock MemoryManager with sensible async defaults."""
    memory = MagicMock()
    memory.campaign_progress = progress
    memory.get_context = AsyncMock(return_value=search_result)
    memory.record_turn = AsyncMock()
    memory.session_messages = MagicMock(return_value=[])
    return memory


# ─────────────────────────────────────────────────────────────────────────────
# TestSpoilerGuard
# ─────────────────────────────────────────────────────────────────────────────


class TestSpoilerGuard:
    def test_progress_zero_returns_only_first_scene(self):
        scenes = ["Scene A", "Scene B", "Scene C"]
        assert revealed_text(scenes, 0) == "Scene A"

    def test_returns_scenes_up_to_and_including_progress(self):
        scenes = ["Scene A", "Scene B", "Scene C"]
        result = revealed_text(scenes, 1)
        assert result == "Scene A\n\nScene B"
        assert "Scene C" not in result

    def test_clamps_progress_beyond_last_scene(self):
        scenes = ["Scene A", "Scene B"]
        assert revealed_text(scenes, 99) == "Scene A\n\nScene B"

    def test_empty_scenes_returns_empty_string(self):
        assert revealed_text([], 5) == ""

    def test_all_scenes_returned_at_last_index(self):
        scenes = ["A", "B", "C"]
        assert revealed_text(scenes, 2) == "A\n\nB\n\nC"


# ─────────────────────────────────────────────────────────────────────────────
# TestDetectNarrativeState
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectNarrativeState:
    def test_combat_keywords_return_combat(self):
        assert detect_narrative_state("I attack the goblin with my sword") == NarrativeState.COMBAT

    def test_damage_keyword_returns_combat(self):
        assert detect_narrative_state("How much damage does it deal?") == NarrativeState.COMBAT

    def test_rest_keywords_return_rest(self):
        assert detect_narrative_state("We take a long rest in the cave") == NarrativeState.REST

    def test_social_keywords_return_social(self):
        assert detect_narrative_state("I try to persuade the innkeeper") == NarrativeState.SOCIAL

    def test_neutral_text_defaults_to_exploration(self):
        assert detect_narrative_state("I look around the room carefully") == NarrativeState.EXPLORATION

    def test_empty_string_defaults_to_exploration(self):
        assert detect_narrative_state("") == NarrativeState.EXPLORATION


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildSystemPrompt
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSystemPrompt:
    async def test_prompt_contains_persona_and_campaign_sections(self):
        campaign = _make_campaign()
        rules = _make_rules()
        memory = _make_memory()

        prompt = await build_system_prompt(campaign, rules, memory)

        assert "Dungeon Master" in prompt
        assert "A brave hero ventures" in prompt
        assert "Aldric" in prompt
        assert "Goblin" in prompt
        assert "RULES REFERENCE" in prompt

    def test_spoiler_guard_hides_future_scenes_at_progress_zero(self):
        """Sync wrapper so we can test the spoiler guard logic directly."""
        campaign = _make_campaign()
        # revealed_text at progress=0 should only have Scene One
        text = revealed_text(campaign.scenes, 0)
        assert "Scene One" in text
        assert "Scene Two" not in text
        assert "Scene Three" not in text

    async def test_spoiler_guard_reveals_scenes_up_to_current_progress(self):
        campaign = _make_campaign()
        rules = _make_rules()
        memory = _make_memory(progress=1)

        prompt = await build_system_prompt(campaign, rules, memory)

        assert "Scene One" in prompt
        assert "Scene Two" in prompt
        assert "Scene Three" not in prompt

    async def test_memory_context_included_when_non_empty(self):
        campaign = _make_campaign()
        rules = _make_rules()
        memory = _make_memory(search_result="## Remembered Context\n- The village elder is friendly.")

        prompt = await build_system_prompt(campaign, rules, memory)

        assert "village elder is friendly" in prompt
        assert "MEMORY" in prompt

    async def test_memory_context_omitted_when_empty(self):
        campaign = _make_campaign()
        rules = _make_rules()
        memory = _make_memory(search_result="")

        prompt = await build_system_prompt(campaign, rules, memory)

        assert "MEMORY" not in prompt

    async def test_combat_state_includes_combat_rules(self):
        campaign = _make_campaign()
        rules = _make_rules()
        memory = _make_memory()

        prompt = await build_system_prompt(
            campaign, rules, memory, state=NarrativeState.COMBAT
        )

        assert "COMBAT" in prompt.upper()


# ─────────────────────────────────────────────────────────────────────────────
# TestDungeonMaster
# ─────────────────────────────────────────────────────────────────────────────


class TestDungeonMaster:
    def _make_dm(
        self,
        llm_response: str = "You enter the dungeon.",
        progress: int = 0,
        search_result: str = "",
    ) -> DungeonMaster:
        llm = MagicMock()
        llm.complete = MagicMock(return_value=llm_response)
        campaign = _make_campaign()
        rules = _make_rules()
        memory = _make_memory(progress=progress, search_result=search_result)
        return DungeonMaster(llm, campaign, rules, memory)

    async def test_respond_returns_llm_response(self):
        dm = self._make_dm(llm_response="The torches flicker.")
        result = await dm.respond("I look around.")
        assert result == "The torches flicker."

    async def test_respond_calls_llm_with_system_prompt_and_player_input(self):
        dm = self._make_dm()
        await dm.respond("I search the chest.")
        call_args = dm._llm.complete.call_args
        messages = call_args[0][0]
        # First message must be the system prompt
        assert messages[0]["role"] == "system"
        assert "Dungeon Master" in messages[0]["content"]
        # Last message must be the player input
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "I search the chest."

    async def test_respond_records_turn_in_memory(self):
        dm = self._make_dm(llm_response="A goblin snarls.")
        await dm.respond("I attack!")
        dm._memory.record_turn.assert_called_once_with(
            "I attack!", "A goblin snarls.", 1
        )

    async def test_respond_increments_turn_counter(self):
        dm = self._make_dm()
        assert dm._turn == 0
        await dm.respond("First action.")
        assert dm._turn == 1
        await dm.respond("Second action.")
        assert dm._turn == 2

    async def test_start_campaign_returns_string_and_records_turn(self):
        dm = self._make_dm(llm_response="Welcome, adventurer!")
        result = await dm.start_campaign()
        assert result == "Welcome, adventurer!"
        dm._memory.record_turn.assert_called_once()

    async def test_respond_advances_progress_when_next_scene_title_in_response(self):
        dm = self._make_dm(
            llm_response="You step onto the Forest Path. Scene Two: The Forest Path begins.",
            progress=0,
        )
        await dm.respond("I leave the village.")
        dm._memory.advance_progress.assert_called_once_with(1)

    async def test_respond_does_not_advance_progress_without_next_scene_title(self):
        dm = self._make_dm(
            llm_response="Nothing interesting happens. You wait.",
            progress=0,
        )
        await dm.respond("I look around.")
        dm._memory.advance_progress.assert_not_called()

    async def test_respond_does_not_advance_past_last_scene(self):
        # progress=2 is the last scene (index 2 of 3)
        dm = self._make_dm(
            llm_response="The dungeon swallows you whole.",
            progress=2,
        )
        await dm.respond("I enter the dungeon.")
        dm._memory.advance_progress.assert_not_called()

    async def test_respond_uses_session_window_messages(self):
        dm = self._make_dm()
        prior = [{"role": "assistant", "content": "You see a door."}]
        dm._memory.session_messages = MagicMock(return_value=prior)
        await dm.respond("I open the door.")
        messages = dm._llm.complete.call_args[0][0]
        # session messages appear between system prompt and new user message
        assert messages[1] == prior[0]
