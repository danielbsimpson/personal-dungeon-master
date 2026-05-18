"""
Campaign loader — scans the campaigns directory and returns validated Campaign metadata.

A valid campaign folder must contain:
  - README.md
  - character.md
  - creature.md
  - <folder-name>.txt  (the campaign book)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from src.campaign.parser import ParsedCampaign

log = logging.getLogger(__name__)


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


async def enrich_campaign(
    parsed: "ParsedCampaign",
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
    summarise_fn: Callable[[str], Awaitable[str]],
    memory_dir: Path,
) -> None:
    """
    Build (or load from cache) the semantic chunks and hierarchical index for
    *parsed* and attach them in place.

    This is called once per session startup, after the LLM and embedder are
    available.  Results are cached to disk so subsequent starts are fast.

    Args:
        parsed: A ParsedCampaign returned by parse_campaign().
        embed_fn: Async embedder — returns a vector per input string.
        summarise_fn: Async LLM function — returns a summary string.
        memory_dir: Base memory directory (e.g. settings.memory_dir).
    """
    from src.campaign.chunker import semantic_chunk
    from src.campaign.index import build_or_load_index
    from src.config import settings as _settings

    campaign_name = parsed.summary[:40].split("\n")[0].strip() or "campaign"

    # Derive cache paths
    cache_dir = memory_dir / campaign_name
    chunks_cache = cache_dir / "campaign_chunks.pkl"
    index_cache = cache_dir / "campaign_index.pkl"

    import pickle

    # ── Chunks (Phase A) ─────────────────────────────────────────────────────
    if chunks_cache.exists():
        log.info("Loading cached campaign chunks from '%s'.", chunks_cache)
        try:
            with open(chunks_cache, "rb") as f:
                parsed.chunks = pickle.load(f)  # noqa: S301 — trusted local file
        except Exception as exc:
            log.warning("Failed to load cached chunks (%s); rebuilding.", exc)
            parsed.chunks = []

    if not parsed.chunks:
        log.info("Building semantic chunks for campaign...")
        parsed.chunks = await semantic_chunk(
            parsed.raw_book,
            embed_fn,
            breakpoint_threshold=_settings.chunk_breakpoint_threshold,
            min_chunk_sentences=_settings.chunk_min_sentences,
            max_chunk_sentences=_settings.chunk_max_sentences,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(chunks_cache, "wb") as f:
            pickle.dump(parsed.chunks, f)
        log.info("Saved %d chunks to '%s'.", len(parsed.chunks), chunks_cache)

    if not parsed.chunks:
        log.warning("No chunks produced for campaign — hierarchical index skipped.")
        return

    # ── Hierarchical index (Phase C) ─────────────────────────────────────────
    parsed.index = await build_or_load_index(
        chunks=parsed.chunks,
        embed_fn=embed_fn,
        summarise_fn=summarise_fn,
        campaign_name=campaign_name,
        cache_path=index_cache,
    )
