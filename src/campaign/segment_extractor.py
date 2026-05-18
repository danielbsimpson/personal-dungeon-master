"""
Relevant Segment Extraction (RSE) for campaign chunks (Phase D).

Given a list of retrieved chunks and the full ordered chunk list,
expand each retrieved chunk to include its immediate neighbours when
they are thematically adjacent (cosine similarity above threshold).
Overlapping expanded windows are merged into a single segment.

This ensures the DM receives full narrative context around a retrieved hit,
rather than isolated sentences torn from their surrounding scene.
"""

from __future__ import annotations

import numpy as np

from src.campaign.chunker import CampaignChunk


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom else 0.0


def extract_segments(
    retrieved: list[CampaignChunk],
    all_chunks: list[CampaignChunk],
    all_embeddings: list[list[float]],
    adjacency_threshold: float = 0.72,
    max_window: int = 3,
) -> list[str]:
    """
    Expand each retrieved chunk into a contiguous narrative segment.

    For each retrieved chunk, looks at neighbouring chunks (up to max_window
    in each direction). A neighbour is included when its cosine similarity to
    the retrieved chunk exceeds adjacency_threshold. Overlapping windows are
    merged into a single segment.

    Args:
        retrieved: Chunks returned by the hierarchical index.
        all_chunks: Complete ordered list of campaign chunks.
        all_embeddings: Embedding vectors aligned with all_chunks by position.
        adjacency_threshold: Minimum cosine similarity to include a neighbour.
        max_window: Maximum number of chunks to expand on each side.

    Returns:
        List of merged segment texts, deduplicated and in narrative order.
    """
    if not retrieved or not all_chunks:
        return []

    # Build fast lookup: chunk.index → (chunk, embedding)
    chunk_by_idx: dict[int, tuple[CampaignChunk, list[float]]] = {
        c.index: (c, all_embeddings[i]) for i, c in enumerate(all_chunks)
    }

    covered: set[int] = set()
    segments: list[list[int]] = []  # [start_index, end_index] inclusive

    for hit in retrieved:
        if hit.index in covered:
            continue
        if hit.index not in chunk_by_idx:
            continue

        _, hit_emb = chunk_by_idx[hit.index]
        start = hit.index
        end = hit.index

        # Expand backwards
        for step in range(1, max_window + 1):
            prev_idx = hit.index - step
            if prev_idx < 0 or prev_idx not in chunk_by_idx:
                break
            _, prev_emb = chunk_by_idx[prev_idx]
            if _cosine(hit_emb, prev_emb) < adjacency_threshold:
                break
            start = prev_idx

        # Expand forwards
        for step in range(1, max_window + 1):
            next_idx = hit.index + step
            if next_idx not in chunk_by_idx:
                break
            _, next_emb = chunk_by_idx[next_idx]
            if _cosine(hit_emb, next_emb) < adjacency_threshold:
                break
            end = next_idx

        segments.append([start, end])
        covered.update(range(start, end + 1))

    # Merge overlapping segments (sort by start)
    segments.sort(key=lambda s: s[0])
    merged: list[list[int]] = []
    for seg in segments:
        if merged and seg[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], seg[1])
        else:
            merged.append([seg[0], seg[1]])

    # Assemble final text segments
    result: list[str] = []
    for start, end in merged:
        texts = [
            chunk_by_idx[i][0].text
            for i in range(start, end + 1)
            if i in chunk_by_idx
        ]
        if texts:
            result.append("\n".join(texts))

    return result
