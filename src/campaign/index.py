"""
Two-tier campaign book index (Phase C — Hierarchical indices).

Tier 1 (coarse): act/chapter summaries — one entry per act.
Tier 2 (fine):   semantic chunks from Phase A — many entries per act.

Both tiers embed with contextual headers from Phase B.
Both are stored in memory at session start; the index is persisted to disk
at ``memory/<campaign_name>/campaign_index.pkl`` so it does not rebuild on
every session start.

Phase E (Fusion retrieval) will extend CampaignIndex.retrieve with BM25.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

from src.campaign.chunker import CampaignChunk
from src.campaign.header import with_header

log = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


@dataclass
class ActSummary:
    act: str
    summary_text: str
    chunk_indices: list[int]    # which fine-grained chunks belong to this act
    embedding: list[float] = field(default_factory=list, repr=False)


@dataclass
class CampaignIndex:
    act_summaries: list[ActSummary]
    chunks: list[CampaignChunk]
    chunk_embeddings: list[list[float]] = field(default_factory=list, repr=False)

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    async def build(
        cls,
        chunks: list[CampaignChunk],
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        summarise_fn: Callable[[str], Awaitable[str]],
        campaign_name: str,
    ) -> "CampaignIndex":
        """
        Build the two-tier hierarchical index from a list of CampaignChunks.

        Args:
            chunks: Ordered semantic chunks from Phase A.
            embed_fn: Async function returning float vectors for a list of texts.
            summarise_fn: Async function that calls the LLM to produce a summary.
            campaign_name: Used for logging and future multi-campaign support.

        Returns:
            Fully constructed CampaignIndex.
        """
        log.info("Building campaign index for '%s' (%d chunks)...", campaign_name, len(chunks))

        # 1. Group chunks by act
        acts: dict[str, list[CampaignChunk]] = {}
        for chunk in chunks:
            acts.setdefault(chunk.act, []).append(chunk)

        # 2. Build act summaries (one LLM call per act)
        act_summaries: list[ActSummary] = []
        for act_name, act_chunks in acts.items():
            act_text = "\n\n".join(c.text for c in act_chunks)
            prompt = (
                "Summarise the following campaign act in 3-5 sentences "
                f"for a dungeon master's reference:\n\n{act_text}"
            )
            try:
                summary = await summarise_fn(prompt)
            except Exception:
                log.warning("Summary LLM call failed for act '%s'; using first chunk.", act_name)
                summary = act_chunks[0].text[:500] if act_chunks else act_name

            act_summaries.append(
                ActSummary(
                    act=act_name,
                    summary_text=summary,
                    chunk_indices=[c.index for c in act_chunks],
                )
            )

        # 3. Embed act summaries
        act_texts = [f"[{s.act}]\n{s.summary_text}" for s in act_summaries]
        act_embeddings = await embed_fn(act_texts)
        for summary, emb in zip(act_summaries, act_embeddings):
            summary.embedding = emb

        # 4. Embed fine-grained chunks with contextual headers (Phase B)
        chunk_texts = [with_header(c, campaign_name) for c in chunks]
        # Embed in batches to avoid memory issues with very long campaigns
        chunk_embeddings = await embed_fn(chunk_texts)

        log.info(
            "Campaign index built: %d acts, %d chunks.", len(act_summaries), len(chunks)
        )
        return cls(
            act_summaries=act_summaries,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: list[float],
        progress_index: int,
        top_acts: int = 2,
        top_chunks: int = 5,
    ) -> list[CampaignChunk]:
        """
        Two-pass hierarchical retrieval.

        Pass 1: Find the most relevant acts from the coarse tier.
        Pass 2: Retrieve fine-grained chunks within those acts, respecting
                the spoiler guard (only chunks whose index <= progress_index).

        Args:
            query_embedding: Embedding vector for the player's current input.
            progress_index: Spoiler guard — only chunks up to this index.
            top_acts: Number of acts to consider in pass 1.
            top_chunks: Maximum number of chunks to return.

        Returns:
            Ordered list of the most relevant CampaignChunks.
        """
        if not self.act_summaries or not self.chunks:
            return []

        q = np.array(query_embedding, dtype=np.float32)

        # Pass 1: find the most relevant acts
        act_scores = [
            _cosine(q, np.array(s.embedding, dtype=np.float32))
            for s in self.act_summaries
        ]
        best_act_indices = sorted(
            range(len(act_scores)), key=lambda i: act_scores[i], reverse=True
        )[:top_acts]
        relevant_acts = {self.act_summaries[i].act for i in best_act_indices}

        # Pass 2: retrieve fine-grained chunks within relevant acts
        # Apply spoiler guard: only chunks whose index <= progress_index
        candidate_chunks = [
            (i, c)
            for i, c in enumerate(self.chunks)
            if c.act in relevant_acts and c.index <= progress_index
        ]

        if not candidate_chunks:
            # Fallback: ignore act filter but still respect spoiler guard
            candidate_chunks = [
                (i, c) for i, c in enumerate(self.chunks) if c.index <= progress_index
            ]

        chunk_scores = [
            (
                _cosine(q, np.array(self.chunk_embeddings[i], dtype=np.float32)),
                c,
            )
            for i, c in candidate_chunks
        ]
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in chunk_scores[:top_chunks]]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Pickle the index to *path* for fast reload on next session start."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Campaign index saved to '%s'.", path)

    @classmethod
    def load(cls, path: Path) -> "CampaignIndex":
        """Load a previously pickled CampaignIndex."""
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301 — trusted local file


async def build_or_load_index(
    chunks: list[CampaignChunk],
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
    summarise_fn: Callable[[str], Awaitable[str]],
    campaign_name: str,
    cache_path: Path,
) -> CampaignIndex:
    """
    Load a cached CampaignIndex if it exists, otherwise build and cache it.

    Args:
        chunks: Ordered semantic chunks (used only if cache miss).
        embed_fn: Async embedder (used only if cache miss).
        summarise_fn: Async LLM summariser (used only if cache miss).
        campaign_name: Campaign name for logging.
        cache_path: Path to the pickle cache file.

    Returns:
        Ready-to-use CampaignIndex.
    """
    if cache_path.exists():
        log.info("Loading cached campaign index from '%s'.", cache_path)
        try:
            return CampaignIndex.load(cache_path)
        except Exception as exc:
            log.warning("Failed to load cached index (%s); rebuilding.", exc)

    index = await CampaignIndex.build(chunks, embed_fn, summarise_fn, campaign_name)
    index.save(cache_path)
    return index
