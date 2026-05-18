"""
BM25 keyword index over campaign chunks (Phase E — Fusion retrieval).

Provides exact and near-exact token matching for proper nouns, NPC names,
and location-specific vocabulary that semantic search may miss.  Results
are fused with the semantic index via Reciprocal Rank Fusion in ``fusion.py``.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.campaign.chunker import CampaignChunk


def _tokenise(text: str) -> list[str]:
    """Lowercase word-level tokenisation."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25CampaignIndex:
    """BM25 keyword index over a fixed corpus of campaign chunks.

    Build once alongside the embedding index and pickle with it.
    ``BM25Okapi`` is fully picklable, so the index survives session
    restarts at zero extra cost.
    """

    def __init__(self, chunks: list[CampaignChunk]) -> None:
        self.chunks = chunks
        corpus = [_tokenise(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
        progress_index: int | None = None,
    ) -> list[tuple[float, CampaignChunk]]:
        """Return (score, chunk) pairs ranked by BM25.

        Args:
            query: The player's current input.
            top_k: Maximum number of results to return.
            progress_index: Spoiler guard — if set, only chunks whose
                ``chunk.index <= progress_index`` are returned.

        Returns:
            List of (score, chunk) tuples in descending BM25 score order.
            Chunks with a score of exactly 0 are excluded.
        """
        tokens = _tokenise(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[tuple[float, CampaignChunk]] = []
        for idx, score in ranked:
            if score <= 0.0:
                # BM25Okapi returns 0 for non-matching chunks; skip them.
                break
            chunk = self.chunks[idx]
            if progress_index is not None and chunk.index > progress_index:
                continue
            results.append((float(score), chunk))
            if len(results) >= top_k:
                break

        return results
