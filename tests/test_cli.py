"""Tests for CLI speaker attribution (Phase 15)."""

from __future__ import annotations

from src.campaign.parser import AbilityScores, Character
from src.interface.cli import _match_party_member, parse_speaker


def _char(name: str) -> Character:
    return Character(
        name=name,
        character_class="Fighter",
        level=1,
        hit_points=10,
        armor_class=14,
        speed=30,
        proficiency_bonus=2,
        passive_perception=10,
        ability_scores=AbilityScores(),
        raw_markdown="",
    )


_PARTY = [_char("Aldric Stormblade"), _char("Mira")]


class TestParseSpeaker:
    def test_no_prefix_returns_none(self):
        speaker, action = parse_speaker("I attack the goblin.", _PARTY)
        assert speaker is None
        assert action == "I attack the goblin."

    def test_exact_name_prefix_matches(self):
        speaker, action = parse_speaker("Mira: I cast fireball.", _PARTY)
        assert speaker is not None
        assert speaker.name == "Mira"
        assert action == "I cast fireball."

    def test_first_name_prefix_matches(self):
        speaker, action = parse_speaker("Aldric: I raise my shield.", _PARTY)
        assert speaker is not None
        assert speaker.name == "Aldric Stormblade"
        assert action == "I raise my shield."

    def test_unknown_name_falls_back_to_raw(self):
        speaker, action = parse_speaker("Gandalf: you shall not pass.", _PARTY)
        assert speaker is None
        assert action == "Gandalf: you shall not pass."

    def test_sentence_with_colon_is_not_treated_as_speaker(self):
        raw = "I say to the guard: halt there."
        speaker, action = parse_speaker(raw, _PARTY)
        assert speaker is None
        assert action == raw

    def test_case_insensitive_match(self):
        speaker, _ = parse_speaker("mira: I duck.", _PARTY)
        assert speaker is not None
        assert speaker.name == "Mira"


class TestMatchPartyMember:
    def test_match_by_number_is_not_this_helper(self):
        # _match_party_member is name-based only; digits do not match names.
        assert _match_party_member(_PARTY, "1") is None

    def test_prefix_match(self):
        member = _match_party_member(_PARTY, "Ald")
        assert member is not None
        assert member.name == "Aldric Stormblade"

    def test_no_match_returns_none(self):
        assert _match_party_member(_PARTY, "Zephyr") is None
