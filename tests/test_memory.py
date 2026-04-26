"""Tests for the memory subsystem (Phase 5).

Test classes
------------
TestSessionStore          — file-based session window (no graph needed)
TestMemoryManagerProgress — advance_progress monotonicity & persistence
TestMemoryManagerRecordTurn  — record_turn wires graph + session correctly
TestMemoryManagerGetContext  — get_context returns the graph search result
TestMemoryManagerResetSession — reset_session clears window, graph intact
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.dm.memory.session_store import SessionStore


# ─────────────────────────────────────────────────────────────────────────────
# SessionStore
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionStore:
    def test_trims_to_window_size(self, tmp_path: Path) -> None:
        """Messages beyond the window are dropped from the front."""
        store = SessionStore(tmp_path, window=3)
        store.append("user", "a")
        store.append("assistant", "b")
        store.append("user", "c")
        store.append("assistant", "d")  # pushes "a" out

        msgs = store.messages()
        assert len(msgs) == 3
        assert msgs[0]["content"] == "b"
        assert msgs[-1]["content"] == "d"

    def test_persists_and_reloads_across_instances(self, tmp_path: Path) -> None:
        """A second SessionStore pointed at the same path sees the saved messages."""
        store1 = SessionStore(tmp_path, window=10)
        store1.append("user", "hello")
        store1.append("assistant", "hi there")

        store2 = SessionStore(tmp_path, window=10)
        store2.load()

        assert store2.messages() == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

    def test_clear_empties_window_without_touching_graph_files(
        self, tmp_path: Path
    ) -> None:
        """clear() empties the message list and leaves no graphiti.kuzu dir."""
        store = SessionStore(tmp_path, window=10)
        store.append("user", "hello")
        store.clear()

        assert store.messages() == []
        # File should exist but contain an empty list
        assert (tmp_path / "session.json").read_text(encoding="utf-8") == "[]"
        # No graph files were created
        assert not (tmp_path / "graphiti.kuzu").exists()


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager helpers
# ─────────────────────────────────────────────────────────────────────────────


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
        manager._session = SessionStore(campaign_path, window=20)
        manager._session.load()
        manager._graph = mock_graph

    return manager, mock_graph


# ─────────────────────────────────────────────────────────────────────────────
# advance_progress
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryManagerProgress:
    def test_advance_progress_is_monotonically_increasing(
        self, tmp_path: Path
    ) -> None:
        """Progress cannot go backwards."""
        manager, _ = _make_manager(tmp_path)
        manager._progress = 3

        manager.advance_progress(1)
        assert manager.campaign_progress == 3  # no-op

        manager.advance_progress(3)
        assert manager.campaign_progress == 3  # equal — also no-op

        manager.advance_progress(5)
        assert manager.campaign_progress == 5

    def test_advance_progress_persists_to_disk(self, tmp_path: Path) -> None:
        """Calling advance_progress writes progress.json."""
        manager, _ = _make_manager(tmp_path)
        manager.advance_progress(2)

        progress_file = tmp_path / "test_campaign" / "progress.json"
        assert progress_file.exists()
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        assert data["section"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# record_turn
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryManagerRecordTurn:
    async def test_record_turn_appends_to_session_and_calls_graph(
        self, tmp_path: Path
    ) -> None:
        """record_turn must append both messages to the session window and
        call GraphitiStore.add_episode exactly once."""
        manager, mock_graph = _make_manager(tmp_path)

        await manager.record_turn("player said this", "dm replied this", turn=1)

        msgs = manager._session.messages()  # type: ignore[union-attr]
        assert {"role": "user", "content": "player said this"} in msgs
        assert {"role": "assistant", "content": "dm replied this"} in msgs

        mock_graph.add_episode.assert_called_once()
        call_kwargs = mock_graph.add_episode.call_args
        assert call_kwargs.kwargs["name"] == "turn_1"
        assert call_kwargs.kwargs["content"] == "dm replied this"


# ─────────────────────────────────────────────────────────────────────────────
# get_context
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryManagerGetContext:
    async def test_get_context_returns_graph_search_result(
        self, tmp_path: Path
    ) -> None:
        """get_context passes the query to GraphitiStore.search and returns its
        result unchanged."""
        manager, mock_graph = _make_manager(tmp_path)
        mock_graph.search.return_value = "## Remembered Context\n- Torben found a key"

        result = await manager.get_context(
            "What did Torben find?", group_id="test_campaign"
        )

        assert result == "## Remembered Context\n- Torben found a key"
        mock_graph.search.assert_called_once_with(
            "What did Torben find?", "test_campaign"
        )


# ─────────────────────────────────────────────────────────────────────────────
# reset_session
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryManagerResetSession:
    def test_reset_session_clears_window_but_preserves_graph_dir(
        self, tmp_path: Path
    ) -> None:
        """reset_session clears the session window.  The graph directory is
        not deleted."""
        manager, _ = _make_manager(tmp_path)
        manager._session.append("user", "hello")  # type: ignore[union-attr]

        # Simulate a graph dir existing
        graph_dir = tmp_path / "test_campaign" / "graphiti.kuzu"
        graph_dir.mkdir(parents=True, exist_ok=True)

        manager.reset_session()

        assert manager._session.messages() == []  # type: ignore[union-attr]
        assert graph_dir.exists()  # graph untouched

