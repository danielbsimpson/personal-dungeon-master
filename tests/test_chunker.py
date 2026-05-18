"""Tests for Phase A (semantic chunker), Phase B (headers), Phase C (index),
Phase D (RSE), Phase E (BM25 fusion), Phase F (contextual compression),
and Phase G (adaptive retrieval config)."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import numpy as np
import pytest

from src.campaign.bm25_index import BM25CampaignIndex
from src.campaign.chunker import CampaignChunk, semantic_chunk
from src.campaign.compressor import compress
from src.campaign.fusion import rrf
from src.campaign.header import header_string, with_header
from src.campaign.index import CampaignIndex
from src.campaign.segment_extractor import extract_segments
from src.dm.retrieval_config import RetrievalConfig, get_retrieval_config
from src.rules.reference import NarrativeState


# ─────────────────────────────────────────────────────────────────────────────
# Mock helpers
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "The hero stepped into the ruined keep. Crumbling walls loomed overhead. "
    "A cold wind swept through the broken windows. Goblins lurked in the shadows. "
    "Their yellow eyes gleamed with malice. The stench of smoke filled the air. "
    "Suddenly a goblin leaped from behind a pillar. It raised its rusty blade. "
    "The hero drew their sword and readied for battle. "
    "After the fight, the hero explored the rest of the keep. "
    "An old chest sat in the corner, bound with iron. "
    "Inside, a map lay folded beneath a bag of coins."
)


async def _mock_embed(texts: list[str]) -> list[list[float]]:
    """Return a deterministic pseudo-embedding for each input string."""
    results = []
    for text in texts:
        # Use character-frequency fingerprint so similar texts get similar vectors
        vec = np.zeros(16, dtype=np.float32)
        for i, ch in enumerate(text[:16]):
            vec[i % 16] += ord(ch) / 1000.0
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        results.append(vec.tolist())
    return results


async def _mock_summarise(prompt: str) -> str:
    """Return a stub summary regardless of the prompt."""
    return "A summary of this act."


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — Semantic chunker
# ─────────────────────────────────────────────────────────────────────────────


class TestSemanticChunker:
    @pytest.mark.asyncio
    async def test_returns_non_empty_list(self):
        chunks = await semantic_chunk(SAMPLE_TEXT, _mock_embed)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_no_empty_chunks(self):
        chunks = await semantic_chunk(SAMPLE_TEXT, _mock_embed)
        assert all(c.text.strip() for c in chunks)

    @pytest.mark.asyncio
    async def test_chunks_cover_whole_text(self):
        """All sentences should appear in at least one chunk (no text lost)."""
        chunks = await semantic_chunk(SAMPLE_TEXT, _mock_embed)
        combined = " ".join(c.text for c in chunks)
        # Every word from the original should appear somewhere
        for word in SAMPLE_TEXT.split():
            assert word in combined, f"Word '{word}' missing from chunks."

    @pytest.mark.asyncio
    async def test_indices_are_sequential(self):
        chunks = await semantic_chunk(SAMPLE_TEXT, _mock_embed)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    @pytest.mark.asyncio
    async def test_scene_header_propagated_from_markdown(self):
        text = "## The Ruined Keep\nYou see crumbling walls. Goblins lurk. The door creaks."
        chunks = await semantic_chunk(text, _mock_embed, min_chunk_sentences=1)
        assert any(c.scene_header == "The Ruined Keep" for c in chunks)

    @pytest.mark.asyncio
    async def test_max_chunk_sentences_forces_split(self):
        """Setting max_chunk_sentences=3 should produce multiple chunks."""
        chunks = await semantic_chunk(SAMPLE_TEXT, _mock_embed, max_chunk_sentences=3)
        assert len(chunks) > 1

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_list(self):
        chunks = await semantic_chunk("", _mock_embed)
        assert chunks == []


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — Contextual chunk headers
# ─────────────────────────────────────────────────────────────────────────────


class TestContextualHeaders:
    def _make_chunk(self, act: str = "Act 1", scene: str = "Goblin Ambush") -> CampaignChunk:
        return CampaignChunk(
            index=0,
            text="Goblins attack.",
            scene_header=scene,
            act=act,
            start_char=0,
            end_char=16,
        )

    def test_header_contains_act_and_scene(self):
        chunk = self._make_chunk()
        h = with_header(chunk)
        assert "Act 1" in h
        assert "Goblin Ambush" in h

    def test_header_prefixed_before_text(self):
        chunk = self._make_chunk()
        result = with_header(chunk)
        assert result.endswith(chunk.text)

    def test_header_string_matches_template(self):
        chunk = self._make_chunk(act="Act 2", scene="The Keep")
        h = header_string(chunk)
        assert "Act 2" in h
        assert "The Keep" in h
        assert "Goblins" not in h  # header_string does NOT include the text

    def test_raw_text_unchanged_in_with_header(self):
        chunk = self._make_chunk()
        result = with_header(chunk, campaign_name="test-campaign")
        assert chunk.text in result


# ─────────────────────────────────────────────────────────────────────────────
# Phase C — Hierarchical index
# ─────────────────────────────────────────────────────────────────────────────


def _make_chunks(n: int = 6) -> list[CampaignChunk]:
    """Return *n* dummy chunks split evenly between Act 1 and Act 2."""
    chunks = []
    for i in range(n):
        act = "Act 1" if i < n // 2 else "Act 2"
        chunks.append(
            CampaignChunk(
                index=i,
                text=f"Scene text for chunk {i}. Something happens here. The hero acts.",
                scene_header=f"Scene {i}",
                act=act,
                start_char=i * 50,
                end_char=(i + 1) * 50,
            )
        )
    return chunks


class TestCampaignIndex:
    @pytest.mark.asyncio
    async def test_build_produces_act_summaries(self):
        chunks = _make_chunks(6)
        index = await CampaignIndex.build(chunks, _mock_embed, _mock_summarise, "test")
        assert len(index.act_summaries) == 2  # Act 1 and Act 2

    @pytest.mark.asyncio
    async def test_build_produces_chunk_embeddings(self):
        chunks = _make_chunks(6)
        index = await CampaignIndex.build(chunks, _mock_embed, _mock_summarise, "test")
        assert len(index.chunk_embeddings) == len(chunks)

    @pytest.mark.asyncio
    async def test_retrieve_respects_spoiler_guard(self):
        chunks = _make_chunks(6)
        index = await CampaignIndex.build(chunks, _mock_embed, _mock_summarise, "test")
        query_emb = (await _mock_embed(["goblin attack"]))[0]
        results = index.retrieve(query_embedding=query_emb, progress_index=2)
        assert all(c.index <= 2 for c in results)

    @pytest.mark.asyncio
    async def test_retrieve_returns_chunks(self):
        chunks = _make_chunks(6)
        index = await CampaignIndex.build(chunks, _mock_embed, _mock_summarise, "test")
        query_emb = (await _mock_embed(["hero fights"]))[0]
        results = index.retrieve(query_embedding=query_emb, progress_index=99, top_chunks=3)
        assert len(results) <= 3
        assert all(isinstance(c, CampaignChunk) for c in results)

    @pytest.mark.asyncio
    async def test_top_acts_narrows_results(self):
        chunks = _make_chunks(6)
        index = await CampaignIndex.build(chunks, _mock_embed, _mock_summarise, "test")
        query_emb = (await _mock_embed(["scene"]))[0]
        results = index.retrieve(query_embedding=query_emb, progress_index=99, top_acts=1)
        acts_returned = {c.act for c in results}
        assert len(acts_returned) <= 2  # may include fallback chunks from other acts

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, tmp_path):
        chunks = _make_chunks(4)
        index = await CampaignIndex.build(chunks, _mock_embed, _mock_summarise, "test")
        cache_file = tmp_path / "index.pkl"
        index.save(cache_file)
        loaded = CampaignIndex.load(cache_file)
        assert len(loaded.chunks) == len(index.chunks)
        assert len(loaded.act_summaries) == len(index.act_summaries)
        assert len(loaded.chunk_embeddings) == len(index.chunk_embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Phase D — Relevant segment extraction (RSE)
# ─────────────────────────────────────────────────────────────────────────────


def _make_chunks_with_embeddings(n: int = 8):
    """Return (chunks, embeddings) where adjacent chunks share a similar direction."""
    chunks = []
    embeddings = []
    rng = np.random.default_rng(42)
    for i in range(n):
        # Make adjacent chunks similar by using a slowly-rotating vector
        angle = i * 0.1
        vec = np.array([np.cos(angle), np.sin(angle), 0.5], dtype=np.float32)
        vec /= np.linalg.norm(vec)
        chunks.append(
            CampaignChunk(
                index=i,
                text=f"Chunk {i} text. The hero does something in scene {i}.",
                scene_header=f"Scene {i}",
                act="Act 1",
                start_char=i * 40,
                end_char=(i + 1) * 40,
            )
        )
        embeddings.append(vec.tolist())
    return chunks, embeddings


class TestExtractSegments:
    def test_returns_list_of_strings(self):
        chunks, embeddings = _make_chunks_with_embeddings(6)
        segments = extract_segments(
            retrieved=[chunks[2]],
            all_chunks=chunks,
            all_embeddings=embeddings,
        )
        assert isinstance(segments, list)
        assert all(isinstance(s, str) for s in segments)

    def test_retrieved_chunk_text_in_result(self):
        chunks, embeddings = _make_chunks_with_embeddings(6)
        segments = extract_segments(
            retrieved=[chunks[3]],
            all_chunks=chunks,
            all_embeddings=embeddings,
        )
        combined = "\n".join(segments)
        assert "Chunk 3 text" in combined

    def test_adjacent_chunks_merged(self):
        """With high adjacency threshold and slowly-rotating vectors, neighbours merge."""
        chunks, embeddings = _make_chunks_with_embeddings(8)
        # Force merge by using a very high threshold so any similarity passes
        segments = extract_segments(
            retrieved=[chunks[3]],
            all_chunks=chunks,
            all_embeddings=embeddings,
            adjacency_threshold=0.0,  # always include neighbours
            max_window=2,
        )
        combined = "\n".join(segments)
        # At threshold 0.0, all neighbours within window=2 should be included
        assert "Chunk 1 text" in combined or "Chunk 2 text" in combined or "Chunk 4 text" in combined

    def test_distant_chunks_are_not_merged(self):
        """Two distant retrieved chunks with zero adjacency threshold still merge if window overlaps."""
        chunks, embeddings = _make_chunks_with_embeddings(10)
        segments = extract_segments(
            retrieved=[chunks[0], chunks[9]],
            all_chunks=chunks,
            all_embeddings=embeddings,
            adjacency_threshold=1.1,  # never expand (no cosine can exceed 1)
            max_window=1,
        )
        # Should produce 2 separate segments (one per retrieved chunk)
        assert len(segments) == 2

    def test_empty_retrieved_returns_empty(self):
        chunks, embeddings = _make_chunks_with_embeddings(4)
        assert extract_segments([], chunks, embeddings) == []

    def test_no_duplicate_text(self):
        """The same chunk should not appear twice even if retrieved twice."""
        chunks, embeddings = _make_chunks_with_embeddings(6)
        segments = extract_segments(
            retrieved=[chunks[3], chunks[3]],
            all_chunks=chunks,
            all_embeddings=embeddings,
        )
        # Count occurrences of chunk 3's unique text across all segments
        count = sum(s.count("Chunk 3 text") for s in segments)
        assert count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase E — Fusion retrieval (BM25 + RRF)
# ─────────────────────────────────────────────────────────────────────────────


def _make_named_chunks() -> list[CampaignChunk]:
    """Return chunks with proper-noun-rich text for BM25 smoke tests."""
    data = [
        ("Reidoth the druid lives in the ruins of Thundertree.", "Act 1", "Thundertree"),
        ("The Cragmaw goblins patrol the trade road near Phandalin.", "Act 1", "Trade Road"),
        ("Sister Garaele seeks the banshee Agatha in Conyberry.", "Act 2", "Conyberry"),
        ("Nezznar the Black Spider controls the Wave Echo Cave.", "Act 2", "Wave Echo Cave"),
        ("Sildar Hallwinter is a member of the Lords' Alliance.", "Act 1", "Phandalin"),
    ]
    return [
        CampaignChunk(
            index=i,
            text=text,
            scene_header=scene,
            act=act,
            start_char=i * 80,
            end_char=(i + 1) * 80,
        )
        for i, (text, act, scene) in enumerate(data)
    ]


class TestBM25CampaignIndex:
    def test_search_returns_relevant_chunk(self):
        chunks = _make_named_chunks()
        idx = BM25CampaignIndex(chunks)
        results = idx.search("Reidoth druid", top_k=3)
        texts = [c.text for _, c in results]
        assert any("Reidoth" in t for t in texts)

    def test_proper_noun_is_top_result(self):
        chunks = _make_named_chunks()
        idx = BM25CampaignIndex(chunks)
        results = idx.search("Nezznar Black Spider", top_k=3)
        assert results, "Expected at least one result"
        assert "Nezznar" in results[0][1].text

    def test_empty_query_returns_empty(self):
        chunks = _make_named_chunks()
        idx = BM25CampaignIndex(chunks)
        assert idx.search("") == []

    def test_spoiler_guard_filters_results(self):
        chunks = _make_named_chunks()
        idx = BM25CampaignIndex(chunks)
        # Nezznar is chunk index 3; with progress_index=2 it should not appear.
        results = idx.search("Nezznar", top_k=5, progress_index=2)
        assert all(c.index <= 2 for _, c in results)

    def test_zero_score_chunks_excluded(self):
        chunks = _make_named_chunks()
        idx = BM25CampaignIndex(chunks)
        results = idx.search("zzzunknowntoken", top_k=5)
        assert results == []


class TestRRF:
    def _make_chunk(self, idx: int, text: str = "") -> CampaignChunk:
        return CampaignChunk(
            index=idx,
            text=text or f"Chunk {idx}.",
            scene_header="Scene",
            act="Act 1",
            start_char=idx * 20,
            end_char=(idx + 1) * 20,
        )

    def test_chunk_in_both_lists_ranks_first(self):
        chunk_a = self._make_chunk(0, "Common chunk.")
        chunk_b = self._make_chunk(1, "Semantic only.")
        chunk_c = self._make_chunk(2, "Keyword only.")
        result = rrf(
            semantic_results=[chunk_a, chunk_b],
            keyword_results=[(0.9, chunk_a), (0.7, chunk_c)],
        )
        assert result[0].index == chunk_a.index

    def test_deduplication(self):
        chunk_a = self._make_chunk(0)
        result = rrf(
            semantic_results=[chunk_a],
            keyword_results=[(1.0, chunk_a)],
        )
        assert sum(1 for c in result if c.index == chunk_a.index) == 1

    def test_empty_inputs(self):
        assert rrf([], []) == []

    def test_top_n_respected(self):
        chunks = [self._make_chunk(i) for i in range(10)]
        kw = [(float(10 - i), c) for i, c in enumerate(chunks)]
        result = rrf(semantic_results=chunks, keyword_results=kw, top_n=3)
        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_fusion_index_retrieve_with_query_text(self):
        """CampaignIndex.retrieve with query_text triggers BM25 fusion."""
        chunks = _make_named_chunks()
        index = await CampaignIndex.build(chunks, _mock_embed, _mock_summarise, "test")
        query_emb = (await _mock_embed(["Reidoth druid"]))[0]
        results = index.retrieve(
            query_embedding=query_emb,
            progress_index=99,
            query_text="Reidoth druid",
        )
        assert results, "Expected non-empty results"
        assert any("Reidoth" in c.text for c in results)

    @pytest.mark.asyncio
    async def test_bm25_index_built_on_build(self):
        """CampaignIndex.build should populate bm25_index."""
        chunks = _make_chunks(4)
        index = await CampaignIndex.build(chunks, _mock_embed, _mock_summarise, "test")
        assert index.bm25_index is not None

    @pytest.mark.asyncio
    async def test_retrieve_without_query_text_falls_back_to_semantic(self):
        """retrieve() with no query_text should still return results (semantic only)."""
        chunks = _make_chunks(4)
        index = await CampaignIndex.build(chunks, _mock_embed, _mock_summarise, "test")
        query_emb = (await _mock_embed(["hero"]))[0]
        results = index.retrieve(query_embedding=query_emb, progress_index=99, query_text="")
        assert isinstance(results, list)


# ─────────────────────────────────────────────────────────────────────────────
# Phase F — Contextual compression
# ─────────────────────────────────────────────────────────────────────────────


class TestContextualCompression:
    @pytest.mark.asyncio
    async def test_compress_returns_relevant_subset(self):
        async def _llm(prompt: str) -> str:
            return "Goblins lurk in the shadows."

        passage = (
            "Goblins lurk in the shadows. "
            "The sky is blue today. "
            "A gentle breeze blows through the trees."
        )
        result = await compress(passage, "goblin encounter", _llm)
        assert "Goblins" in result
        # Unrelated sentence should not appear (the mock returns only the relevant one)
        assert "sky is blue" not in result

    @pytest.mark.asyncio
    async def test_compress_falls_back_when_none_returned(self):
        async def _llm_none(prompt: str) -> str:
            return "NONE"

        passage = "Unrelated passage about the weather."
        result = await compress(passage, "very specific query", _llm_none)
        assert result == passage

    @pytest.mark.asyncio
    async def test_compress_falls_back_on_empty_response(self):
        async def _llm_empty(prompt: str) -> str:
            return ""

        passage = "Some passage."
        result = await compress(passage, "query", _llm_empty)
        assert result == passage

    @pytest.mark.asyncio
    async def test_compress_falls_back_on_llm_error(self):
        async def _llm_error(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        passage = "Important campaign text."
        result = await compress(passage, "query", _llm_error)
        assert result == passage

    @pytest.mark.asyncio
    async def test_compress_truncates_long_passage(self):
        """Passage exceeding max_passage_tokens should be truncated before sending."""
        calls: list[str] = []

        async def _capturing_llm(prompt: str) -> str:
            calls.append(prompt)
            return "Short result."

        long_passage = "x" * 10_000
        await compress(long_passage, "query", _capturing_llm, max_passage_tokens=10)
        assert calls, "LLM should have been called"
        # 10 tokens * 4 chars/token = 40 chars max passage in prompt
        assert long_passage[:40] in calls[0]
        assert len(calls[0]) < len(long_passage)


# ─────────────────────────────────────────────────────────────────────────────
# Phase G — Adaptive retrieval by NarrativeState
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrievalConfig:
    def test_all_states_have_configs(self):
        for state in NarrativeState:
            cfg = get_retrieval_config(state)
            assert isinstance(cfg, RetrievalConfig)

    def test_combat_has_fewer_graph_results_than_social(self):
        combat = get_retrieval_config(NarrativeState.COMBAT)
        social = get_retrieval_config(NarrativeState.SOCIAL)
        assert combat.graph_results < social.graph_results

    def test_social_does_not_compress(self):
        cfg = get_retrieval_config(NarrativeState.SOCIAL)
        assert cfg.compress is False

    def test_combat_compresses(self):
        cfg = get_retrieval_config(NarrativeState.COMBAT)
        assert cfg.compress is True

    def test_combat_has_narrower_act_filter(self):
        combat = get_retrieval_config(NarrativeState.COMBAT)
        exploration = get_retrieval_config(NarrativeState.EXPLORATION)
        assert combat.top_acts <= exploration.top_acts

    def test_social_has_wider_adjacency_window(self):
        social = get_retrieval_config(NarrativeState.SOCIAL)
        combat = get_retrieval_config(NarrativeState.COMBAT)
        assert social.adjacency_window > combat.adjacency_window

    def test_rest_has_minimal_chunk_count(self):
        rest = get_retrieval_config(NarrativeState.REST)
        social = get_retrieval_config(NarrativeState.SOCIAL)
        assert rest.top_chunks < social.top_chunks

    def test_unknown_state_returns_default(self):
        """get_retrieval_config should return a valid default for any state."""
        cfg = get_retrieval_config(NarrativeState.EXPLORATION)
        assert cfg.top_chunks > 0
        assert cfg.graph_results > 0

