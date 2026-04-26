"""
Campaign loader — scans the campaigns directory and returns validated Campaign metadata.

A valid campaign folder must contain:
  - README.md
  - character.md
  - creature.md
  - <folder-name>.txt  (the campaign book)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_REQUIRED_FILES: frozenset[str] = frozenset({"README.md", "character.md", "creature.md"})


@dataclass(frozen=True)
class Campaign:
    """Lightweight metadata for a campaign folder."""

    name: str
    path: Path
    readme_path: Path
    character_path: Path
    creature_path: Path
    book_path: Path


def _find_book_file(folder: Path) -> Path | None:
    """Return the campaign book .txt file if it exists (must match the folder name)."""
    candidate = folder / f"{folder.name}.txt"
    return candidate if candidate.is_file() else None


def load_campaigns(campaigns_dir: Path | None = None) -> list[Campaign]:
    """
    Scan *campaigns_dir* for valid campaign folders.

    Returns a sorted list of :class:`Campaign` metadata objects.

    Raises
    ------
    FileNotFoundError
        If *campaigns_dir* does not exist.
    ValueError
        If no valid campaigns are found (includes per-folder error details).
    """
    from src.config import settings  # deferred to avoid circular imports at module level

    base = Path(campaigns_dir) if campaigns_dir is not None else settings.campaigns_dir

    if not base.exists():
        raise FileNotFoundError(f"Campaigns directory not found: {base}")

    campaigns: list[Campaign] = []
    errors: list[str] = []

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue

        missing: list[str] = []

        for required in sorted(_REQUIRED_FILES):
            if not (entry / required).is_file():
                missing.append(required)

        book = _find_book_file(entry)
        if book is None:
            missing.append(f"{entry.name}.txt")

        if missing:
            errors.append(
                f"  '{entry.name}' is missing: {', '.join(missing)}"
            )
            continue

        campaigns.append(
            Campaign(
                name=entry.name,
                path=entry,
                readme_path=entry / "README.md",
                character_path=entry / "character.md",
                creature_path=entry / "creature.md",
                book_path=book,  # type: ignore[arg-type]  # never None here
            )
        )

    if not campaigns:
        hint = ("\n" + "\n".join(errors)) if errors else ""
        raise ValueError(
            f"No valid campaigns found in '{base}'.{hint}"
        )

    return campaigns
