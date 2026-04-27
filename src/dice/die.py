"""
Die types and roll data structures for the dice engine (Phase 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Die(Enum):
    """Standard tabletop die types.  The value is the number of faces."""

    D4 = 4
    D6 = 6
    D8 = 8
    D10 = 10
    D12 = 12
    D20 = 20
    D100 = 100


@dataclass
class RollRequest:
    """
    Parameters describing a single dice roll.

    Attributes
    ----------
    label:
        Human-readable label for the roll (e.g. ``"attack"``, ``"damage"``).
    die:
        The :class:`Die` to roll.
    count:
        Number of dice to roll and sum.
    modifier:
        Flat integer bonus/penalty added to the sum of rolls.
    advantage:
        When ``True`` the die is rolled twice and the *higher* result is kept.
        Mutually exclusive with ``disadvantage``.
    disadvantage:
        When ``True`` the die is rolled twice and the *lower* result is kept.
        Mutually exclusive with ``advantage``.
    """

    label: str
    die: Die
    count: int = 1
    modifier: int = 0
    advantage: bool = False
    disadvantage: bool = False


@dataclass
class RollResult:
    """
    The outcome of a :class:`RollRequest`.

    Attributes
    ----------
    die:
        The die that was rolled.
    rolls:
        The individual die results (before modifier).  When advantage or
        disadvantage is in play this contains *all* raw results (both sets),
        not just the kept values.
    modifier:
        The flat modifier that was applied.
    total:
        Final total (sum of kept rolls + modifier, floored at 1).
    label:
        Copied from the originating :class:`RollRequest`.
    """

    die: Die
    rolls: list[int] = field(default_factory=list)
    modifier: int = 0
    total: int = 0
    label: str = ""
