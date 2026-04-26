"""
Rules loader — scans the rules directory for the configured game edition and
returns a RulesReference holding all section texts keyed by file stem.

Usage
-----
    from src.rules.loader import load_rules
    from src.config import settings

    rules = load_rules(settings)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RulesReference:
    """All loaded rules sections for a single game edition."""

    edition: str
    """The edition identifier, e.g. ``'5e'``."""

    sections: dict[str, str] = field(default_factory=dict)
    """
    Mapping of section name → full text.
    The section name is the lowercase file stem, e.g. ``'combat'``, ``'core'``.
    """

    @property
    def section_names(self) -> list[str]:
        """Return a sorted list of loaded section names."""
        return sorted(self.sections)


def load_rules(settings: "Settings | None" = None) -> RulesReference:  # type: ignore[name-defined]
    """
    Load all ``.md`` files from ``RULES_DIR/<edition>/`` into a
    :class:`RulesReference`.

    Parameters
    ----------
    settings:
        If *None*, the module-level singleton from ``src.config`` is used.

    Raises
    ------
    FileNotFoundError
        If the edition directory does not exist.
    ValueError
        If the edition directory exists but contains no ``.md`` files.
    """
    if settings is None:
        from src.config import settings as _settings  # deferred to avoid circular imports

        settings = _settings

    edition_dir: Path = settings.rules_edition_dir

    if not edition_dir.exists():
        raise FileNotFoundError(
            f"Rules directory for edition '{settings.game_edition}' not found: {edition_dir}"
        )

    md_files = sorted(edition_dir.glob("*.md"))
    if not md_files:
        raise ValueError(
            f"No rule files found in '{edition_dir}'. "
            f"Add at least one .md file for edition '{settings.game_edition}'."
        )

    sections: dict[str, str] = {}
    for path in md_files:
        sections[path.stem.lower()] = path.read_text(encoding="utf-8")

    return RulesReference(edition=settings.game_edition, sections=sections)
