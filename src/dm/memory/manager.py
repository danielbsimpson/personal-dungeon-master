"""
Memory manager — composes GraphitiStore (long-term graph), SessionStore
(short-term window), and a progress pointer (spoiler guard state).

Usage
-----
    manager = MemoryManager(memory_dir=settings.memory_dir, settings=settings)
    await manager.load("example-campaign")

    # After each DM turn:
    await manager.record_turn(player_input, dm_response, turn=n)

    # Before building the system prompt:
    ctx = await manager.get_context(current_input, group_id="example-campaign")

    # Advance the spoiler guard after the player reaches a new scene:
    manager.advance_progress(to_section=2)

    # Player-triggered resets via /reset and /fullreset CLI commands:
    manager.reset_session()
    await manager.full_reset()
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from src.config import Settings
from src.config import settings as _default_settings

from .graphiti_factory import build_graphiti_clients
from .graphiti_store import GraphitiStore
from .memo_rag import MemoRAGStore
from .raptor import RaptorTree
from .session_store import SessionStore

log = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates long-term graph memory, short-term session window, and
    the campaign progress pointer."""

    def __init__(
        self,
        memory_dir: Path,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings: Settings = settings or _default_settings
        self._memory_dir = memory_dir
        self._graph: Optional[GraphitiStore] = None
        self._session: Optional[SessionStore] = None
        self._progress: int = 0
        self._campaign_name: str = ""
        self._embedder: Optional[Any] = None  # OpenAIEmbedder, set in load()
        # Phase H — RAPTOR
        self.raptor_tree: Optional[RaptorTree] = None
        self._raptor_path: Optional[Path] = None
        # Phase I — MemoRAG
        self.memo_store: Optional[MemoRAGStore] = None
        self._memo_path: Optional[Path] = None
        # Accumulated episode texts for RAPTOR / MemoRAG (current session)
        self._episode_texts: list[str] = []
        self._session_number: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def load(self, campaign_name: str) -> None:
        """
        Set up all memory components for *campaign_name*.

        Creates ``memory/<campaign_name>/`` if it does not exist, opens (or
        creates) the Kuzu graph database, loads the session window from disk,
        and reads the progress pointer.

        Parameters
        ----------
        campaign_name:
            The campaign folder name; used as the subdirectory name under
            ``memory_dir``.
        """
        self._campaign_name = campaign_name
        campaign_path = self._memory_dir / campaign_name
        campaign_path.mkdir(parents=True, exist_ok=True)

        # Session window
        self._session = SessionStore(
            campaign_path, window=self._settings.session_window
        )
        self._session.load()

        # Progress pointer + session number
        progress_path = campaign_path / "progress.json"
        if progress_path.exists():
            data = json.loads(progress_path.read_text(encoding="utf-8"))
            self._progress = int(data.get("section", 0))
            self._session_number = int(data.get("session_number", 0))

        # Phase H — load persisted RAPTOR tree
        self._raptor_path = campaign_path / "raptor_tree.json"
        if self._raptor_path.exists():
            try:
                self.raptor_tree = RaptorTree.load(self._raptor_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("RAPTOR: failed to load tree (%s); starting fresh.", exc)

        # Phase I — load or create MemoRAG store
        self._memo_path = campaign_path / "memo_rag.json"
        if self._memo_path.exists():
            try:
                self.memo_store = MemoRAGStore.load(self._memo_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("MemoRAG: failed to load store (%s); starting fresh.", exc)
                self.memo_store = MemoRAGStore()
        else:
            self.memo_store = MemoRAGStore()

        # Graphiti knowledge graph
        db_path = campaign_path / "graphiti.kuzu"
        llm_client, embedder = build_graphiti_clients(self._settings)
        self._embedder = embedder
        self._graph = GraphitiStore(db_path, llm_client, embedder)
        await self._graph.setup()

    # ------------------------------------------------------------------
    # Turn recording
    # ------------------------------------------------------------------

    async def record_turn(
        self,
        player_input: str,
        dm_response: str,
        turn: int,
        speaker: str | None = None,
    ) -> None:
        """
        Persist a completed conversation turn.

        Appends both messages to the session window and ingests the DM
        response as a Graphiti episode for long-term graph extraction.

        Parameters
        ----------
        player_input:
            The player's raw input text.
        dm_response:
            The final DM narration (after dice substitution).
        turn:
            Monotonically increasing turn counter (used as the episode name).
        speaker:
            Optional name of the party member who acted this turn.  When
            supplied, the persisted player message is attributed as
            ``[speaker] ...`` and the episode source records the speaker.
        """
        assert self._session is not None, "Call load() before record_turn()"
        assert self._graph is not None, "Call load() before record_turn()"

        stored_input = f"[{speaker}] {player_input}" if speaker else player_input
        source_description = (
            f"DM narration (player: {speaker})" if speaker else "DM narration"
        )

        self._session.append("user", stored_input)
        self._session.append("assistant", dm_response)
        self._episode_texts.append(dm_response)  # Phase H/I accumulation

        try:
            await self._graph.add_episode(
                name=f"turn_{turn}",
                content=dm_response,
                source_description=source_description,
                turn=turn,
                group_id=self._campaign_name,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Graph memory add_episode failed (%s); skipping.", exc)

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------

    async def get_context(
        self,
        current_text: str,
        group_id: str,
        num_results: int = 10,
    ) -> str:
        """
        Retrieve the most relevant remembered facts for the current turn.

        Parameters
        ----------
        current_text:
            The player's latest input (or a summary of the current scene),
            used as the search query against the knowledge graph.
        group_id:
            Campaign identifier (passed through to ``GraphitiStore.search``).
        num_results:
            Maximum number of graph fact edges to retrieve.  Phase G passes
            a per-NarrativeState value via ``RetrievalConfig.graph_results``.

        Returns
        -------
        str
            A Markdown context block for injection into the system prompt, or
            ``""`` if the graph is empty or no relevant facts were found.
        """
        assert self._graph is not None, "Call load() before get_context()"

        query = current_text
        raptor_context = ""

        # Phase I: MemoRAG — augment query with distilled session clues
        if (
            self.memo_store is not None
            and self.memo_store.clues
            and self._embedder is not None
        ):
            try:
                q_emb = (await self.embed([current_text]))[0]
                query = self.memo_store.augment_query(
                    current_text, q_emb, top_k=self._settings.memorag_top_k
                )
                # Phase H: RAPTOR — reuse the same query embedding
                if self.raptor_tree is not None:
                    nodes = self.raptor_tree.search(
                        q_emb, top_k=self._settings.raptor_top_k_retrieval
                    )
                    raptor_context = "\n".join(n.text for n in nodes)
            except Exception as exc:  # noqa: BLE001
                log.warning("RAPTOR/MemoRAG context failed (%s); skipping.", exc)
        elif self.raptor_tree is not None and self._embedder is not None:
            try:
                q_emb = (await self.embed([current_text]))[0]
                nodes = self.raptor_tree.search(
                    q_emb, top_k=self._settings.raptor_top_k_retrieval
                )
                raptor_context = "\n".join(n.text for n in nodes)
            except Exception as exc:  # noqa: BLE001
                log.warning("RAPTOR context failed (%s); skipping.", exc)

        try:
            graph_context = await self._graph.search(query, group_id, num_results=num_results)
        except Exception as exc:  # noqa: BLE001
            log.warning("Graph memory search failed (%s); skipping.", exc)
            graph_context = ""

        parts = [p for p in [raptor_context, graph_context] if p]
        return "\n\n---\n\n".join(parts) if parts else ""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed *texts* using the configured Ollama embedder.

        Used by context_builder to obtain query embeddings for hierarchical
        campaign index retrieval (Phase C).

        Parameters
        ----------
        texts:
            One or more strings to embed.

        Returns
        -------
        list[list[float]]
            One float vector per input string.
        """
        assert self._embedder is not None, "Call load() before embed()"
        return await self._embedder.create(texts)

    def session_messages(self) -> list[dict]:
        """
        Return the current short-term session window as a list of messages.

        Returns
        -------
        list[dict]
            List of ``{"role": ..., "content": ...}`` dicts representing the
            last N messages (bounded by ``SESSION_WINDOW``).
        """
        assert self._session is not None, "Call load() before session_messages()"
        return self._session.messages()

    # ------------------------------------------------------------------
    # Progress pointer
    # ------------------------------------------------------------------

    def advance_progress(self, to_section: int) -> None:
        """
        Advance the spoiler-guard progress pointer to *to_section*.

        The pointer is monotonically increasing: calling this with a value
        less than or equal to the current progress is a no-op.

        Parameters
        ----------
        to_section:
            The scene/section index the player has now reached.
        """
        if to_section <= self._progress:
            return
        self._progress = to_section
        if self._campaign_name:
            progress_path = (
                self._memory_dir / self._campaign_name / "progress.json"
            )
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                json.dumps({"section": self._progress}, ensure_ascii=False),
                encoding="utf-8",
            )

    @property
    def campaign_progress(self) -> int:
        """Current scene/section index (0 = before the adventure begins)."""
        return self._progress

    # ------------------------------------------------------------------
    # Phase H — RAPTOR
    # ------------------------------------------------------------------

    async def rebuild_raptor(
        self,
        llm_fn: Callable[[str], Awaitable[str]],
    ) -> None:
        """Rebuild the RAPTOR tree from all episode texts accumulated this session.

        Saves the resulting tree to disk.  No-op when there are no episodes or
        RAPTOR is disabled via :attr:`~src.config.Settings.raptor_enabled`.
        """
        if not self._episode_texts or not self._settings.raptor_enabled:
            return
        if self._embedder is None:
            log.warning("RAPTOR: embedder not ready; skipping rebuild.")
            return

        log.info("RAPTOR: rebuilding tree from %d episodes.", len(self._episode_texts))
        self.raptor_tree = await RaptorTree.build(
            episodes=self._episode_texts,
            embed_fn=self.embed,
            summarise_fn=llm_fn,
            max_clusters=self._settings.raptor_max_clusters,
        )
        if self._raptor_path is not None:
            self.raptor_tree.save(self._raptor_path)
        log.info("RAPTOR: rebuild complete (%d nodes).", len(self.raptor_tree.nodes))

    # ------------------------------------------------------------------
    # Phase I — MemoRAG
    # ------------------------------------------------------------------

    async def end_of_session(
        self,
        llm_fn: Callable[[str], Awaitable[str]],
    ) -> None:
        """Generate MemoRAG clues and rebuild the RAPTOR tree at session end.

        Call this when the player quits or between sessions.  Clues are saved
        to disk; the session episode list is cleared afterwards.

        Parameters
        ----------
        llm_fn:
            Async callable wrapping the LLM (prompt → response text).
        """
        if not self._episode_texts:
            return

        self._session_number += 1

        # Generate and persist MemoRAG clues
        if self._settings.memorag_enabled and self.memo_store is not None:
            await self.memo_store.add_clues(
                self._episode_texts,
                self._session_number,
                embed_fn=self.embed,
                summarise_fn=llm_fn,
                clues_per_session=self._settings.memorag_clues_per_session,
            )
            if self._memo_path is not None:
                self.memo_store.save(self._memo_path)

        # Rebuild RAPTOR tree
        await self.rebuild_raptor(llm_fn)

        # Persist the updated session number alongside the progress pointer
        if self._campaign_name:
            progress_path = self._memory_dir / self._campaign_name / "progress.json"
            existing: dict = {}
            if progress_path.exists():
                existing = json.loads(progress_path.read_text(encoding="utf-8"))
            existing["session_number"] = self._session_number
            progress_path.write_text(
                json.dumps(existing, ensure_ascii=False), encoding="utf-8"
            )

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Clear the short-term session window.

        The knowledge graph is preserved.  Use this for the ``/reset`` CLI
        command.
        """
        assert self._session is not None, "Call load() before reset_session()"
        self._session.clear()

    async def full_reset(self) -> None:
        """Wipe both the session window and the Kuzu knowledge graph.

        Drops and recreates the Kuzu database directory, then reinitialises
        the graph.  Use this for the ``/fullreset`` CLI command.

        .. note::
            On Windows, the Kuzu file connection is released by the garbage
            collector (``KuzuDriver.close()`` is a no-op).  ``gc.collect()``
            is called before deletion, but file locks may occasionally persist
            on heavily loaded systems.  If ``shutil.rmtree`` raises a
            ``PermissionError``, retry after a short delay or restart the
            process.
        """
        self.reset_session()
        self._episode_texts = []

        if self._graph is not None:
            await self._graph.close()
            self._graph = None

        db_path = self._memory_dir / self._campaign_name / "graphiti.kuzu"
        if db_path.exists():
            gc.collect()  # encourage GC to release Kuzu file handles
            shutil.rmtree(db_path)

        # Clear RAPTOR and MemoRAG artefacts
        self.raptor_tree = None
        self.memo_store = MemoRAGStore()
        if self._raptor_path is not None and self._raptor_path.exists():
            self._raptor_path.unlink()
        if self._memo_path is not None and self._memo_path.exists():
            self._memo_path.unlink()

        llm_client, embedder = build_graphiti_clients(self._settings)
        self._graph = GraphitiStore(db_path, llm_client, embedder)
        await self._graph.setup()

    # ------------------------------------------------------------------
    # Entity inspection (used by CLI /journal and /graph commands)
    # ------------------------------------------------------------------

    async def get_entities(self) -> list[dict]:
        """Return all stored Entity nodes for the ``/journal`` command."""
        assert self._graph is not None, "Call load() before get_entities()"
        return await self._graph.get_entities()

    async def search_entity(self, name: str) -> list[dict]:
        """Return edges related to *name* for the ``/graph`` command."""
        assert self._graph is not None, "Call load() before search_entity()"
        return await self._graph.search_entity(name)
