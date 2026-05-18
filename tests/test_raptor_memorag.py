"""Tests for Phase H (RAPTOR) and Phase I (MemoRAG).

Test classes
------------
TestRaptorTree               — build, search, save/load
TestRaptorTreeEdgeCases      — empty episodes, single episode
TestMemoRAGStore             — add_clues, retrieve_clues, augment_query, save/load
TestMemoRAGEdgeCases         — empty episodes, clue generation failure
TestMemoryManagerRaptor      — rebuild_raptor integration
TestMemoryManagerMemoRAG     — end_of_session integration
TestMemoryManagerGetContext  — RAPTOR context injected, MemoRAG query augmentation
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.dm.memory.memo_rag import MemoryClue, MemoRAGStore
from src.dm.memory.raptor import RaptorNode, RaptorTree


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

DIM = 8  # small embedding dimension for tests


def _fixed_embed_fn(texts: list[str]):
    """Deterministic embed_fn: each text gets a unique unit vector."""
    embeddings = []
    for i, _ in enumerate(texts):
        v = np.zeros(DIM, dtype=np.float32)
        v[i % DIM] = 1.0
        embeddings.append(v.tolist())
    return embeddings


async def _async_embed(texts: list[str]) -> list[list[float]]:
    return _fixed_embed_fn(texts)


async def _async_summarise(prompt: str) -> str:
    return "A brief summary of the provided episodes."


def _make_manager(tmp_path: Path):
    """Create a MemoryManager with mocked Graphiti internals."""
    from src.dm.memory.manager import MemoryManager

    mock_graph = AsyncMock()
    mock_graph.search = AsyncMock(return_value="")

    with (
        patch(
            "src.dm.memory.manager.build_graphiti_clients",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "src.dm.memory.manager.GraphitiStore",
            return_value=mock_graph,
        ),
    ):
        manager = MemoryManager(memory_dir=tmp_path)
        manager._campaign_name = "test_campaign"
        manager._progress = 0
        campaign_path = tmp_path / "test_campaign"
        campaign_path.mkdir(parents=True, exist_ok=True)
        from src.dm.memory.session_store import SessionStore
        manager._session = SessionStore(campaign_path, window=20)
        manager._session.load()
        manager._graph = mock_graph

    return manager, mock_graph


# ─────────────────────────────────────────────────────────────────────────────
# RaptorTree
# ─────────────────────────────────────────────────────────────────────────────


class TestRaptorTree:
    async def test_build_creates_leaf_nodes_for_each_episode(self) -> None:
        """Level-0 nodes correspond 1-to-1 with input episodes."""
        episodes = ["Episode one.", "Episode two.", "Episode three."]
        tree = await RaptorTree.build(
            episodes=episodes,
            embed_fn=_async_embed,
            summarise_fn=_async_summarise,
            max_clusters=10,
        )
        leaf_nodes = [n for n in tree.nodes if n.level == 0]
        assert len(leaf_nodes) == 3
        assert leaf_nodes[0].text == "Episode one."
        assert leaf_nodes[1].text == "Episode two."

    async def test_build_produces_summary_nodes_above_leaves(self) -> None:
        """With enough episodes, levels > 0 must be present."""
        # 9 episodes → n_clusters=min(10, max(1, 9//3))=3 → 3 summaries (level 1)
        episodes = [f"Episode {i}." for i in range(9)]
        tree = await RaptorTree.build(
            episodes=episodes,
            embed_fn=_async_embed,
            summarise_fn=_async_summarise,
            max_clusters=10,
        )
        assert any(n.level > 0 for n in tree.nodes)

    async def test_build_summary_nodes_reference_children(self) -> None:
        """Every level-1+ node must have a non-empty children list."""
        episodes = [f"Episode {i}." for i in range(6)]
        tree = await RaptorTree.build(
            episodes=episodes,
            embed_fn=_async_embed,
            summarise_fn=_async_summarise,
            max_clusters=10,
        )
        for node in tree.nodes:
            if node.level > 0:
                assert node.children, f"Level-{node.level} node {node.node_id} has no children"

    async def test_search_returns_top_k_nodes(self) -> None:
        """search() returns at most top_k nodes sorted by similarity."""
        episodes = [f"Episode {i}." for i in range(6)]
        tree = await RaptorTree.build(
            episodes=episodes,
            embed_fn=_async_embed,
            summarise_fn=_async_summarise,
        )
        query_emb = [1.0] + [0.0] * (DIM - 1)
        results = tree.search(query_emb, top_k=3)
        assert len(results) <= 3

    async def test_search_returns_highest_similarity_first(self) -> None:
        """The most similar node to the query must appear first."""
        tree = RaptorTree()
        target = [1.0] + [0.0] * (DIM - 1)
        other = [0.0, 1.0] + [0.0] * (DIM - 2)
        tree.nodes = [
            RaptorNode(node_id=0, level=0, text="target", embedding=target, children=[]),
            RaptorNode(node_id=1, level=0, text="other", embedding=other, children=[]),
        ]
        results = tree.search(target, top_k=2)
        assert results[0].text == "target"

    async def test_save_and_load_round_trips(self, tmp_path: Path) -> None:
        """save() followed by load() reconstructs identical nodes."""
        episodes = ["Alpha.", "Beta.", "Gamma.", "Delta.", "Epsilon.", "Zeta."]
        original = await RaptorTree.build(
            episodes=episodes,
            embed_fn=_async_embed,
            summarise_fn=_async_summarise,
        )
        path = tmp_path / "raptor_tree.json"
        original.save(path)
        loaded = RaptorTree.load(path)

        assert len(loaded.nodes) == len(original.nodes)
        for orig, load in zip(original.nodes, loaded.nodes):
            assert orig.node_id == load.node_id
            assert orig.level == load.level
            assert orig.text == load.text
            assert orig.children == load.children


class TestRaptorTreeEdgeCases:
    async def test_build_with_empty_episodes_returns_empty_tree(self) -> None:
        tree = await RaptorTree.build(
            episodes=[],
            embed_fn=_async_embed,
            summarise_fn=_async_summarise,
        )
        assert tree.nodes == []

    async def test_build_with_single_episode_returns_leaf_only(self) -> None:
        """One episode → no clustering → just one leaf node."""
        tree = await RaptorTree.build(
            episodes=["Lone episode."],
            embed_fn=_async_embed,
            summarise_fn=_async_summarise,
        )
        assert len(tree.nodes) == 1
        assert tree.nodes[0].level == 0

    async def test_search_on_empty_tree_returns_empty_list(self) -> None:
        tree = RaptorTree()
        result = tree.search([1.0] + [0.0] * (DIM - 1), top_k=5)
        assert result == []

    async def test_build_handles_summarise_failure_gracefully(self) -> None:
        """If summarise_fn raises, build should fall back to truncated text."""

        async def _bad_summarise(prompt: str) -> str:
            raise RuntimeError("LLM offline")

        tree = await RaptorTree.build(
            episodes=["A.", "B.", "C.", "D.", "E.", "F."],
            embed_fn=_async_embed,
            summarise_fn=_bad_summarise,
        )
        # Tree should still be built (with truncated text as summaries)
        assert len(tree.nodes) > 6


# ─────────────────────────────────────────────────────────────────────────────
# MemoRAGStore
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoRAGStore:
    async def test_add_clues_stores_clues_with_correct_session(self) -> None:
        """Clues generated by add_clues are stored with the given session number."""
        store = MemoRAGStore()

        async def _summarise(prompt: str) -> str:
            return "Torben found a key\nThe dragon was defeated\nA secret passage opened"

        await store.add_clues(
            session_episodes=["DM says something."],
            session_number=1,
            embed_fn=_async_embed,
            summarise_fn=_summarise,
            clues_per_session=3,
        )

        assert len(store.clues) == 3
        assert all(c.session == 1 for c in store.clues)

    async def test_add_clues_respects_clues_per_session_limit(self) -> None:
        """add_clues never stores more clues than clues_per_session."""

        async def _many_lines(prompt: str) -> str:
            return "\n".join(f"Clue {i}" for i in range(20))

        store = MemoRAGStore()
        await store.add_clues(
            session_episodes=["Long session narration."],
            session_number=1,
            embed_fn=_async_embed,
            summarise_fn=_many_lines,
            clues_per_session=5,
        )
        assert len(store.clues) <= 5

    def test_retrieve_clues_returns_top_k_by_cosine(self) -> None:
        """retrieve_clues returns the most similar clues to the query."""
        store = MemoRAGStore()
        target = [1.0] + [0.0] * (DIM - 1)
        other = [0.0, 1.0] + [0.0] * (DIM - 2)
        store.clues = [
            MemoryClue(clue_id=0, text="match", session=1, embedding=target),
            MemoryClue(clue_id=1, text="no-match", session=1, embedding=other),
        ]
        results = store.retrieve_clues(target, top_k=1)
        assert len(results) == 1
        assert results[0].text == "match"

    def test_augment_query_prepends_clue_text(self) -> None:
        """augment_query returns the original query extended with clue text."""
        store = MemoRAGStore()
        emb = [1.0] + [0.0] * (DIM - 1)
        store.clues = [
            MemoryClue(clue_id=0, text="Torben found a key", session=1, embedding=emb),
        ]
        result = store.augment_query("What happened?", emb, top_k=1)
        assert "What happened?" in result
        assert "Torben found a key" in result

    def test_augment_query_unchanged_when_no_clues(self) -> None:
        """augment_query returns the original query when the store is empty."""
        store = MemoRAGStore()
        emb = [1.0] + [0.0] * (DIM - 1)
        result = store.augment_query("What happened?", emb)
        assert result == "What happened?"

    async def test_save_and_load_round_trips(self, tmp_path: Path) -> None:
        """save() + load() reconstructs identical clues."""
        store = MemoRAGStore()

        async def _summarise(prompt: str) -> str:
            return "A clue about the dungeon"

        await store.add_clues(
            session_episodes=["The party descended into the dungeon."],
            session_number=2,
            embed_fn=_async_embed,
            summarise_fn=_summarise,
            clues_per_session=1,
        )
        path = tmp_path / "memo_rag.json"
        store.save(path)
        loaded = MemoRAGStore.load(path)

        assert len(loaded.clues) == len(store.clues)
        assert loaded.clues[0].text == store.clues[0].text
        assert loaded.clues[0].session == store.clues[0].session


class TestMemoRAGEdgeCases:
    async def test_add_clues_skips_empty_episodes(self) -> None:
        """No clues are added when session_episodes is empty."""
        store = MemoRAGStore()
        await store.add_clues(
            session_episodes=[],
            session_number=1,
            embed_fn=_async_embed,
            summarise_fn=_async_summarise,
        )
        assert store.clues == []

    async def test_add_clues_skips_on_summarise_failure(self) -> None:
        """add_clues handles a failing summarise_fn gracefully."""

        async def _bad(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        store = MemoRAGStore()
        # Should not raise
        await store.add_clues(
            session_episodes=["Some narration."],
            session_number=1,
            embed_fn=_async_embed,
            summarise_fn=_bad,
        )
        assert store.clues == []

    def test_retrieve_clues_returns_empty_when_store_empty(self) -> None:
        store = MemoRAGStore()
        assert store.retrieve_clues([1.0] + [0.0] * (DIM - 1)) == []


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager — RAPTOR integration (Phase H)
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryManagerRaptor:
    async def test_rebuild_raptor_builds_tree_from_episode_texts(
        self, tmp_path: Path
    ) -> None:
        """After recording turns, rebuild_raptor populates manager.raptor_tree."""
        manager, mock_graph = _make_manager(tmp_path)
        manager._embedder = MagicMock()
        manager._embedder.create = AsyncMock(
            side_effect=lambda texts: _fixed_embed_fn(texts)
        )
        # Simulate episodes accumulated over several turns
        manager._episode_texts = [f"DM narration {i}." for i in range(6)]

        await manager.rebuild_raptor(llm_fn=_async_summarise)

        assert manager.raptor_tree is not None
        assert len(manager.raptor_tree.nodes) > 0

    async def test_rebuild_raptor_saves_tree_to_disk(self, tmp_path: Path) -> None:
        """rebuild_raptor writes raptor_tree.json when _raptor_path is set."""
        manager, _ = _make_manager(tmp_path)
        manager._embedder = MagicMock()
        manager._embedder.create = AsyncMock(
            side_effect=lambda texts: _fixed_embed_fn(texts)
        )
        manager._episode_texts = [f"Episode {i}." for i in range(6)]
        raptor_path = tmp_path / "test_campaign" / "raptor_tree.json"
        manager._raptor_path = raptor_path

        await manager.rebuild_raptor(llm_fn=_async_summarise)

        assert raptor_path.exists()
        raw = json.loads(raptor_path.read_text())
        assert "nodes" in raw

    async def test_rebuild_raptor_noop_when_no_episodes(self, tmp_path: Path) -> None:
        """rebuild_raptor does nothing when _episode_texts is empty."""
        manager, _ = _make_manager(tmp_path)
        manager._episode_texts = []

        await manager.rebuild_raptor(llm_fn=_async_summarise)

        assert manager.raptor_tree is None

    async def test_record_turn_accumulates_episode_texts(
        self, tmp_path: Path
    ) -> None:
        """Each record_turn call appends the DM response to _episode_texts."""
        manager, _ = _make_manager(tmp_path)

        await manager.record_turn("player input", "dm response one", turn=1)
        await manager.record_turn("player input 2", "dm response two", turn=2)

        assert manager._episode_texts == ["dm response one", "dm response two"]


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager — MemoRAG integration (Phase I)
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryManagerMemoRAG:
    async def test_end_of_session_generates_clues(self, tmp_path: Path) -> None:
        """end_of_session calls add_clues and populates the memo_store."""
        manager, _ = _make_manager(tmp_path)
        manager._embedder = MagicMock()
        manager._embedder.create = AsyncMock(
            side_effect=lambda texts: _fixed_embed_fn(texts)
        )
        manager.memo_store = MemoRAGStore()  # simulate load() creating a fresh store
        manager._episode_texts = ["Big battle happened.", "Dragon was slain."]

        async def _summarise(prompt: str) -> str:
            return "Dragon was slain in the big battle"

        await manager.end_of_session(llm_fn=_summarise)

        assert manager.memo_store is not None
        assert len(manager.memo_store.clues) > 0

    async def test_end_of_session_increments_session_number(
        self, tmp_path: Path
    ) -> None:
        """end_of_session increments _session_number each call."""
        manager, _ = _make_manager(tmp_path)
        manager._embedder = MagicMock()
        manager._embedder.create = AsyncMock(
            side_effect=lambda texts: _fixed_embed_fn(texts)
        )
        manager._episode_texts = ["Some narration."]
        initial = manager._session_number

        async def _summarise(prompt: str) -> str:
            return "A memorable event"

        await manager.end_of_session(llm_fn=_summarise)

        assert manager._session_number == initial + 1

    async def test_end_of_session_noop_when_no_episodes(
        self, tmp_path: Path
    ) -> None:
        """end_of_session does nothing if no episodes were recorded."""
        manager, _ = _make_manager(tmp_path)
        manager._episode_texts = []
        initial_session = manager._session_number

        await manager.end_of_session(llm_fn=_async_summarise)

        assert manager._session_number == initial_session


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager.get_context — Phase H/I integration
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryManagerGetContextPhaseHI:
    async def test_get_context_prepends_raptor_context(
        self, tmp_path: Path
    ) -> None:
        """When raptor_tree is set and embedder is available, RAPTOR context
        is prepended to the graph result with a --- separator."""
        manager, mock_graph = _make_manager(tmp_path)
        mock_graph.search.return_value = "## Graph Context\n- Some fact"
        manager._embedder = MagicMock()
        manager._embedder.create = AsyncMock(
            side_effect=lambda texts: _fixed_embed_fn(texts)
        )
        # Inject a minimal RAPTOR tree with one summary node
        emb = [1.0] + [0.0] * (DIM - 1)
        manager.raptor_tree = RaptorTree(
            nodes=[
                RaptorNode(
                    node_id=0, level=1, text="Big battle summary.", embedding=emb, children=[]
                )
            ]
        )

        result = await manager.get_context("battle", group_id="test_campaign")

        assert "Big battle summary." in result
        assert "## Graph Context" in result
        assert "---" in result

    async def test_get_context_augments_query_with_memo_clues(
        self, tmp_path: Path
    ) -> None:
        """When memo_store has clues, the query passed to GraphitiStore.search
        includes clue text as context."""
        manager, mock_graph = _make_manager(tmp_path)
        mock_graph.search.return_value = ""
        manager._embedder = MagicMock()
        manager._embedder.create = AsyncMock(
            side_effect=lambda texts: _fixed_embed_fn(texts)
        )
        # Inject a clue directly (bypass add_clues for speed)
        emb = [1.0] + [0.0] * (DIM - 1)
        manager.memo_store = MemoRAGStore(
            clues=[
                MemoryClue(clue_id=0, text="Torben found a key", session=1, embedding=emb)
            ]
        )

        await manager.get_context("What happened?", group_id="test_campaign")

        # The query passed to graph.search should contain the clue
        call_args = mock_graph.search.call_args
        assert "Torben found a key" in call_args.args[0]

    async def test_get_context_no_raptor_no_memo_behaves_as_before(
        self, tmp_path: Path
    ) -> None:
        """When raptor_tree=None and memo_store is empty, behaviour is
        identical to pre-Phase H/I: graph result returned directly."""
        manager, mock_graph = _make_manager(tmp_path)
        mock_graph.search.return_value = "## Remembered Context\n- Torben found a key"

        result = await manager.get_context(
            "What did Torben find?", group_id="test_campaign"
        )

        assert result == "## Remembered Context\n- Torben found a key"
        mock_graph.search.assert_called_once_with(
            "What did Torben find?", "test_campaign", num_results=10
        )
