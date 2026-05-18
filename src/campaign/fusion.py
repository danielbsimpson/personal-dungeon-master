"""
Reciprocal Rank Fusion (RRF) for combining semantic and BM25 results.

Phase E — Fusion retrieval.

RRF uses rank position rather than raw scores, making it robust to the
different scales produced by cosine similarity (bounded 0–1) and BM25
(unbounded, corpus-dependent).  A chunk that ranks highly in *both* lists
receives a larger combined score than one that ranks well in only one.
"""

from __future__ import annotations

from src.campaign.chunker import CampaignChunk


def rrf(
    semantic_results: list[CampaignChunk],
    keyword_results: list[tuple[float, CampaignChunk]],
    k: int = 60,
    top_n: int = 5,
) -> list[CampaignChunk]:
    """Combine two ranked lists using Reciprocal Rank Fusion.

    Args:
        semantic_results: Chunks from the embedding search, ordered
            best-first.  Typically the output of
            ``CampaignIndex._semantic_retrieve()``.
        keyword_results: ``(score, chunk)`` pairs from BM25, ordered
            best-first.  Typically the output of
            ``BM25CampaignIndex.search()``.
        k: RRF smoothing constant.  Larger values reduce the impact of
            top-rank positions, preventing any single list from
            dominating.  The literature recommends ``k=60``.
        top_n: Number of final results to return.

    Returns:
        Re-ranked, deduplicated list of CampaignChunks, ordered by
        descending fused RRF score.
    """
    scores: dict[int, float] = {}

    for rank, chunk in enumerate(semantic_results):
        scores[chunk.index] = scores.get(chunk.index, 0.0) + 1.0 / (k + rank + 1)

    for rank, (_, chunk) in enumerate(keyword_results):
        scores[chunk.index] = scores.get(chunk.index, 0.0) + 1.0 / (k + rank + 1)

    # Build a unified index → chunk mapping from both lists.
    chunk_map: dict[int, CampaignChunk] = {c.index: c for c in semantic_results}
    chunk_map.update({c.index: c for _, c in keyword_results})

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[idx] for idx, _ in ranked[:top_n] if idx in chunk_map]
