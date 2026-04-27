"""Tests for the dice engine (Phase 7)."""

from __future__ import annotations

from random import Random

import pytest

from src.dice.die import Die, RollRequest, RollResult
from src.dice.roller import format_result, parse_roll_tags, roll, substitute_rolls


# ---------------------------------------------------------------------------
# Die enum
# ---------------------------------------------------------------------------


def test_die_face_values():
    assert Die.D4.value == 4
    assert Die.D6.value == 6
    assert Die.D8.value == 8
    assert Die.D10.value == 10
    assert Die.D12.value == 12
    assert Die.D20.value == 20
    assert Die.D100.value == 100


def test_die_members_count():
    """All seven standard die types are present."""
    assert len(Die) == 7


# ---------------------------------------------------------------------------
# roll() — basic behaviour
# ---------------------------------------------------------------------------


def test_roll_within_range():
    """roll() result is within [1, die.value] (no modifier)."""
    for die in Die:
        req = RollRequest(label="test", die=die)
        result = roll(req)
        assert 1 <= result.total <= die.value, f"Out of range for {die}"


def test_roll_with_positive_modifier():
    """Modifier is added to the rolled value."""
    req = RollRequest(label="attack", die=Die.D20, modifier=5)
    rng = Random(42)
    result = roll(req, _rng=rng)
    expected_raw = sum(result.rolls)
    assert result.total == max(1, expected_raw + 5)


def test_roll_with_negative_modifier():
    """Negative modifier is subtracted; total is floored at 1."""
    req = RollRequest(label="damage", die=Die.D4, modifier=-10)
    result = roll(req)
    assert result.total == 1  # floor


def test_roll_total_floor_is_one():
    """Total is always at least 1 regardless of modifier."""
    req = RollRequest(label="penalty", die=Die.D4, modifier=-100)
    result = roll(req)
    assert result.total == 1


def test_roll_multiple_dice_count():
    """Rolling multiple dice returns the expected number of individual rolls."""
    req = RollRequest(label="fireball", die=Die.D6, count=2)
    result = roll(req)
    assert len(result.rolls) == 2


def test_roll_multiple_dice_total():
    """Total equals sum of rolls plus modifier."""
    req = RollRequest(label="damage", die=Die.D6, count=3, modifier=2)
    result = roll(req)
    assert result.total == max(1, sum(result.rolls) + 2)


# ---------------------------------------------------------------------------
# roll() — advantage and disadvantage
# ---------------------------------------------------------------------------


def test_advantage_returns_two_rolls():
    req = RollRequest(label="check", die=Die.D20, advantage=True)
    result = roll(req)
    assert len(result.rolls) == 2


def test_disadvantage_returns_two_rolls():
    req = RollRequest(label="check", die=Die.D20, disadvantage=True)
    result = roll(req)
    assert len(result.rolls) == 2


def test_advantage_uses_higher_roll():
    req = RollRequest(label="check", die=Die.D20, advantage=True)
    result = roll(req)
    assert result.total == max(1, max(result.rolls))


def test_disadvantage_uses_lower_roll():
    req = RollRequest(label="check", die=Die.D20, disadvantage=True)
    result = roll(req)
    assert result.total == max(1, min(result.rolls))


def test_advantage_gte_disadvantage_same_seed():
    """With the same seed, advantage total >= disadvantage total."""
    seed = 99
    req_adv = RollRequest(label="check", die=Die.D20, advantage=True)
    req_dis = RollRequest(label="check", die=Die.D20, disadvantage=True)

    result_adv = roll(req_adv, _rng=Random(seed))
    result_dis = roll(req_dis, _rng=Random(seed))

    assert result_adv.total >= result_dis.total


# ---------------------------------------------------------------------------
# parse_roll_tags
# ---------------------------------------------------------------------------


def test_parse_single_tag_no_modifier():
    tags = parse_roll_tags("[ROLL: attack 1d20+0]")
    assert len(tags) == 1
    assert tags[0].die == Die.D20
    assert tags[0].count == 1
    assert tags[0].modifier == 0
    assert tags[0].label == "attack"


def test_parse_single_tag_with_positive_modifier():
    tags = parse_roll_tags("[ROLL: attack 1d20+5]")
    assert len(tags) == 1
    assert tags[0].modifier == 5


def test_parse_single_tag_with_negative_modifier():
    tags = parse_roll_tags("[ROLL: save 1d20-3]")
    assert len(tags) == 1
    assert tags[0].modifier == -3


def test_parse_multi_die_tag():
    tags = parse_roll_tags("[ROLL: damage 2d6+0]")
    assert tags[0].count == 2
    assert tags[0].die == Die.D6


def test_parse_multiple_tags():
    text = "Roll [ROLL: attack 1d20+4] then [ROLL: damage 2d8+3] for damage."
    tags = parse_roll_tags(text)
    assert len(tags) == 2
    assert tags[0].label == "attack"
    assert tags[1].label == "damage"
    assert tags[1].die == Die.D8


def test_parse_no_tags():
    assert parse_roll_tags("No dice here.") == []


def test_parse_unknown_die_skipped():
    """Unknown die faces (e.g. d7) are silently skipped."""
    tags = parse_roll_tags("[ROLL: test 1d7+0]")
    assert tags == []


def test_parse_tag_case_insensitive():
    tags = parse_roll_tags("[roll: check 1d20+0]")
    assert len(tags) == 1


# ---------------------------------------------------------------------------
# substitute_rolls
# ---------------------------------------------------------------------------


def test_substitute_replaces_tag():
    result = RollResult(die=Die.D20, rolls=[14], modifier=5, total=19, label="attack")
    text = "You swing at the goblin [ROLL: attack 1d20+5]!"
    out = substitute_rolls(text, [result])
    assert "[ROLL:" not in out
    assert "19" in out


def test_substitute_non_tag_content_unchanged():
    result = RollResult(die=Die.D20, rolls=[10], modifier=0, total=10, label="check")
    original_prefix = "The door creaks open. "
    text = original_prefix + "[ROLL: check 1d20+0]"
    out = substitute_rolls(text, [result])
    assert out.startswith(original_prefix)


def test_substitute_multiple_tags():
    r1 = RollResult(die=Die.D20, rolls=[15], modifier=4, total=19, label="attack")
    r2 = RollResult(die=Die.D8, rolls=[5, 3], modifier=3, total=11, label="damage")
    text = "[ROLL: attack 1d20+4] hits for [ROLL: damage 2d8+3]."
    out = substitute_rolls(text, [r1, r2])
    assert "[ROLL:" not in out
    assert "19" in out
    assert "11" in out


def test_substitute_fewer_results_than_tags():
    """Extra tags are left verbatim when results run out."""
    r1 = RollResult(die=Die.D20, rolls=[10], modifier=0, total=10, label="a")
    text = "[ROLL: a 1d20+0] and [ROLL: b 1d20+0]"
    out = substitute_rolls(text, [r1])
    assert "[ROLL: b" in out


def test_substitute_unknown_die_left_verbatim():
    """Tags with unknown die types are not removed."""
    text = "[ROLL: weird 1d7+0]"
    out = substitute_rolls(text, [])
    assert text == out


# ---------------------------------------------------------------------------
# format_result
# ---------------------------------------------------------------------------


def test_format_result_single_die_positive_mod():
    result = RollResult(die=Die.D20, rolls=[14], modifier=5, total=19, label="attack")
    formatted = format_result(result)
    assert "Attack" in formatted
    assert "d20" in formatted
    assert "19" in formatted
    assert "14" in formatted
    assert "+ 5" in formatted


def test_format_result_single_die_negative_mod():
    result = RollResult(die=Die.D20, rolls=[8], modifier=-2, total=6, label="save")
    formatted = format_result(result)
    assert "- 2" in formatted
    assert "6" in formatted


def test_format_result_no_modifier():
    result = RollResult(die=Die.D6, rolls=[4], modifier=0, total=4, label="damage")
    formatted = format_result(result)
    # No modifier means no sign character should appear in the dice expression
    assert "+0" not in formatted
    assert "-0" not in formatted


def test_format_result_multi_die():
    result = RollResult(die=Die.D6, rolls=[3, 5], modifier=0, total=8, label="damage")
    formatted = format_result(result)
    assert "[3, 5]" in formatted
    assert "8" in formatted


# ---------------------------------------------------------------------------
# Statistical distribution (1 000 rolls per die type)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("die", list(Die))
def test_distribution_within_bounds(die: Die):
    """1 000 rolls stay within [1, die.value]; rough coverage of min and max."""
    results = [roll(RollRequest(label="stat", die=die)).total for _ in range(1000)]
    assert min(results) >= 1
    assert max(results) <= die.value
    # With 1 000 rolls we expect to hit 1 and the max at least once for small dice;
    # for D100 the probability of hitting exactly 1 in 1 000 is only ~1%, so we
    # only assert the bounds, not that they're achieved.
    if die.value <= 12:
        assert 1 in results, f"Never rolled 1 on {die}"
        assert die.value in results, f"Never rolled max ({die.value}) on {die}"
