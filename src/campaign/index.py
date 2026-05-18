"""
Two-tier campaign book index (Phase C — Hierarchical indices).

Tier 1 (coarse): act/chapter summaries — one entry per act.
Tier 2 (fine):   semantic chunks from Phase A — many entries per act.

Both tiers embed with contextual headers from Phase B.
Both are stored in memory at session start; the index is persisted to disk
at ``memory/<campaign_name>/campaign_index.pkl`` so it does not rebuild on
every session start.

Phase E adds a BM25 keyword index (``bm25_index``) that is fused with the
semantic results via Reciprocal Rank Fusion when ``query_text`` is supplied
to ``retrieve()``.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

from src.campaign.bm25_index import BM25CampaignIndex
from src.campaign.chunker import CampaignChunk
from src.campaign.fusion import rrf
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
    # Phase E: BM25 keyword index built alongside the embedding index.
    # ``None`` on old pickled instances — handled gracefully in retrieve().
    bm25_index: BM25CampaignIndex | None = field(default=None, repr=False)

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

        # 5. Build BM25 keyword index (Phase E)
        bm25_idx = BM25CampaignIndex(chunks)

        log.info(
            "Campaign index built: %d acts, %d chunks.", len(act_summaries), len(chunks)
        )
        return cls(
            act_summaries=act_summaries,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            bm25_index=bm25_idx,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: list[float],
        progress_index: int,
        top_acts: int = 2,
        top_chunks: int = 5,
        query_text: str = "",
    ) -> list[CampaignChunk]:
        """
        Two-pass hierarchical retrieval, optionally fused with BM25 (Phase E).

        Pass 1: Find the most relevant acts from the coarse tier.
        Pass 2: Retrieve fine-grained chunks within those acts, respecting
                the spoiler guard (only chunks whose index <= progress_index).
        Fusion: When ``query_text`` is non-empty and a BM25 index is present,
                semantic and keyword results are merged via Reciprocal Rank
                Fusion before returning.

        Args:
            query_embedding: Embedding vector for the player's current input.
            progress_index: Spoiler guard — only chunks up to this index.
            top_acts: Number of acts to consider in pass 1.
            top_chunks: Maximum number of chunks to return.
            query_text: Raw player input string.  When non-empty, BM25
                retrieval is performed and results are fused with the
                semantic results via RRF.

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
        semantic_results = [c for _, c in chunk_scores[: top_chunks * 2]]

        # Phase E: fuse with BM25 when query_text is provided.
        bm25_idx: BM25CampaignIndex | None = getattr(self, "bm25_index", None)
        if query_text and bm25_idx is not None:
            keyword_results = bm25_idx.search(
                query_text,
                top_k=top_chunks * 2,
                progress_index=progress_index,
            )
            return rrf(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                top_n=top_chunks,
            )

        return semantic_results[:top_chunks]

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
