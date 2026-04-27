"""
Campaign validator — checks a campaign folder for structural correctness and
common authoring issues before running a session.

Usage
-----
    python scripts/validate_campaign.py campaigns/my-campaign
    python scripts/validate_campaign.py          # validates all campaigns/

Exit codes
----------
0   All checks passed (warnings are non-fatal).
1   One or more errors found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OK = "[OK]"
_WARN = "[WARN]"
_ERR = "[ERR]"

# Accepted filenames for each required file (singular/plural variants).
_REQUIRED_ALTERNATIVES: list[tuple[str, ...]] = [
    ("README.md",),
    ("character.md", "characters.md"),
    ("creature.md", "creatures.md"),
]

# Section-header pattern used by the spoiler guard (lines starting with ##).
_SCENE_HEADER_RE = re.compile(r"^#{2}\s+.+", re.MULTILINE)

# Reasonable ability-score range for D&D 5e (includes racial bonuses & magic).
_SCORE_MIN = 1
_SCORE_MAX = 30


def _print(tag: str, msg: str) -> None:
    print(f"  {tag}  {msg}")


# ---------------------------------------------------------------------------
# Per-file checks
# ---------------------------------------------------------------------------


def _check_required_files(folder: Path) -> list[str]:
    """Return error messages for any required file that is absent."""
    errors: list[str] = []
    for alternatives in _REQUIRED_ALTERNATIVES:
        if not any((folder / name).exists() for name in alternatives):
            # Report the canonical (first) name in the error message.
            errors.append(f"Missing required file: {alternatives[0]}")
    # Campaign book: must have a .txt whose stem matches the folder name.
    txt = folder / f"{folder.name}.txt"
    if not txt.exists():
        errors.append(
            f"Missing campaign book: {folder.name}.txt  "
            "(must match the folder name)"
        )
    return errors


def _check_readme(folder: Path) -> list[str]:
    """Check README.md is non-empty and has at least a heading."""
    path = folder / "README.md"
    if not path.exists():
        return []  # Already caught by required-files check.
    text = path.read_text(encoding="utf-8").strip()
    errors: list[str] = []
    if not text:
        errors.append("README.md is empty.")
    elif not re.search(r"^#\s+", text, re.MULTILINE):
        errors.append("README.md has no top-level heading (# ...).")
    return errors


def _check_character(folder: Path) -> tuple[list[str], list[str]]:
    """
    Check character.md (or characters.md) for required fields.

    Returns (errors, warnings).
    """
    path = folder / "character.md"
    if not path.exists():
        path = folder / "characters.md"
    if not path.exists():
        return [], []
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []

    # Required fields (case-insensitive key search)
    for required in ("name", "class", "level", "hp", "hit points", "ac", "armor class"):
        if required.lower() not in text.lower():
            warnings.append(f"character.md: could not find '{required}' field.")

    # Ability scores: extract integers from STR/DEX/CON/INT/WIS/CHA rows
    score_pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    )
    for match in score_pattern.finditer(text):
        scores = [int(v) for v in match.groups()]
        for score in scores:
            if not (_SCORE_MIN <= score <= _SCORE_MAX):
                warnings.append(
                    f"character.md: ability score {score} is outside the "
                    f"valid D&D 5e range ({_SCORE_MIN}–{_SCORE_MAX})."
                )

    return errors, warnings


def _check_creatures(folder: Path) -> tuple[list[str], list[str]]:
    """
    Check creatures.md for minimum content.

    Returns (errors, warnings).
    """
    # Accept both 'creature.md' and 'creatures.md'.
    path = folder / "creatures.md"
    if not path.exists():
        path = folder / "creature.md"
    if not path.exists():
        return [], []

    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []

    if not text.strip():
        errors.append("creatures.md is empty.")
        return errors, warnings

    # Each creature entry should start with a ## heading.
    creature_headers = re.findall(r"^##\s+(.+)", text, re.MULTILINE)
    if not creature_headers:
        warnings.append(
            "creatures.md: no '## Creature Name' headings found. "
            "Each creature should start with a level-2 heading."
        )
    else:
        warnings += [
            f"creatures.md: creature '{h}' has no HP value."
            for h in creature_headers
            if "hp" not in _section_below(text, h).lower()
            and "hit point" not in _section_below(text, h).lower()
        ]

    return errors, warnings


def _section_below(text: str, heading: str) -> str:
    """Return the text beneath ``## heading`` up to the next ## heading."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}.*?$(.+?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(1) if m else ""


def _check_campaign_book(folder: Path) -> tuple[list[str], list[str]]:
    """
    Check the campaign .txt book for scene headers and creature name parity.

    Returns (errors, warnings).
    """
    txt = folder / f"{folder.name}.txt"
    if not txt.exists():
        return [], []

    text = txt.read_text(encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []

    if not text.strip():
        errors.append(f"{folder.name}.txt is empty.")
        return errors, warnings

    # Scene headers
    scene_headers = _SCENE_HEADER_RE.findall(text)
    if not scene_headers:
        warnings.append(
            f"{folder.name}.txt: no '## Scene Title' headings found.\n"
            "    The spoiler guard uses '## headings' as scene boundaries.\n"
            "    Add at least one '## Scene Name' heading."
        )
    else:
        _print(_OK, f"{folder.name}.txt: {len(scene_headers)} scene heading(s) found.")

    # Creature name parity: names in creatures.md should appear in the book.
    creatures_path = folder / "creatures.md"
    if not creatures_path.exists():
        creatures_path = folder / "creature.md"
    if creatures_path.exists():
        creature_text = creatures_path.read_text(encoding="utf-8")
        creature_names = re.findall(r"^##\s+(.+)", creature_text, re.MULTILINE)
        book_lower = text.lower()
        for name in creature_names:
            if name.lower() not in book_lower:
                warnings.append(
                    f"Creature '{name}' is defined in creatures.md but "
                    f"not mentioned in {folder.name}.txt."
                )

    return errors, warnings


# ---------------------------------------------------------------------------
# Campaign validator
# ---------------------------------------------------------------------------


def validate_campaign(folder: Path) -> bool:
    """
    Run all checks against a single campaign folder.

    Returns ``True`` if no errors were found (warnings are non-fatal).
    """
    print(f"\n{'─' * 60}")
    print(f"  Campaign: {folder.name}")
    print(f"  Path:     {folder}")
    print(f"{'─' * 60}")

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Required files
    file_errors = _check_required_files(folder)
    if file_errors:
        all_errors.extend(file_errors)
    else:
        _print(_OK, "All required files present.")

    # README
    readme_errors = _check_readme(folder)
    all_errors.extend(readme_errors)
    if not readme_errors:
        _print(_OK, "README.md looks good.")

    # Character
    char_errors, char_warnings = _check_character(folder)
    all_errors.extend(char_errors)
    all_warnings.extend(char_warnings)
    if not char_errors and not char_warnings:
        _print(_OK, "character.md looks good.")

    # Creatures
    creature_errors, creature_warnings = _check_creatures(folder)
    all_errors.extend(creature_errors)
    all_warnings.extend(creature_warnings)
    if not creature_errors and not creature_warnings:
        _print(_OK, "creatures.md looks good.")

    # Campaign book
    book_errors, book_warnings = _check_campaign_book(folder)
    all_errors.extend(book_errors)
    all_warnings.extend(book_warnings)

    # Report
    for w in all_warnings:
        _print(_WARN, w)
    for e in all_errors:
        _print(_ERR, e)

    if not all_errors and not all_warnings:
        print(f"\n  All checks passed for '{folder.name}'.\n")
    elif not all_errors:
        print(
            f"\n  '{folder.name}' passed with {len(all_warnings)} warning(s). "
            "Review warnings above.\n"
        )
    else:
        print(
            f"\n  '{folder.name}' failed with {len(all_errors)} error(s) "
            f"and {len(all_warnings)} warning(s).\n"
        )

    return len(all_errors) == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = sys.argv[1:]

    # Determine which folders to validate.
    if args:
        folders = [Path(a) for a in args]
    else:
        # Default: validate all subfolders of campaigns/
        campaigns_root = Path(__file__).resolve().parent.parent / "campaigns"
        if not campaigns_root.exists():
            print(f"[ERR]  campaigns/ directory not found at {campaigns_root}")
            sys.exit(1)
        folders = [p for p in sorted(campaigns_root.iterdir()) if p.is_dir()]
        if not folders:
            print(f"[WARN]  No campaign folders found in {campaigns_root}")
            sys.exit(0)

    passed = 0
    failed = 0
    for folder in folders:
        if not folder.is_dir():
            print(f"[ERR]  Not a directory: {folder}")
            failed += 1
            continue
        ok = validate_campaign(folder)
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
