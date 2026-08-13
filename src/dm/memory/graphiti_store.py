"""
Graphiti store — thin wrapper around a Graphiti instance backed by Kuzu.

Each campaign gets its own Kuzu database file under
``memory/<campaign_name>/graphiti.kuzu/``.  Data isolation is therefore
achieved at the file level: we always call Graphiti with ``group_id=None``
(which uses the Kuzu default of ``''``), avoiding the ``driver.clone()``
branch that is optimised for Neo4j/FalkorDB multi-tenant setups.

Search returns a formatted context block suitable for direct injection
into an LLM system prompt.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import kuzu
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.nodes import EpisodeType

log = logging.getLogger(__name__)

# FTS indexes required by graphiti-core's Kuzu search queries.
# Each entry is (table_name, index_name, columns).
_KUZU_FTS_INDEXES: list[tuple[str, str, list[str]]] = [
    ("Entity",         "node_name_and_summary", ["name", "summary"]),
    ("Community",      "community_name",        ["name"]),
    ("Episodic",       "episode_content",       ["content"]),
    ("RelatesToNode_", "edge_name_and_fact",    ["name", "fact"]),
]


class GraphitiStore:
    """Wraps a Graphiti instance backed by a local Kuzu graph database."""

    def __init__(self, db_path: Path, llm_client: Any, embedder: Any) -> None:
        """
        Parameters
        ----------
        db_path:
            Directory where the Kuzu database files are stored.
            Created automatically on first use.
        llm_client:
            A Graphiti-compatible LLM client (e.g. ``OpenAIGenericClient``
            pointed at Ollama).
        embedder:
            A Graphiti-compatible embedder (e.g. ``OpenAIEmbedder`` pointed
            at Ollama's ``nomic-embed-text``).
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        driver = KuzuDriver(db=str(db_path))
        self._graphiti = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Initialise graph indices and constraints.

        Calls ``build_indices_and_constraints()`` (a no-op for Kuzu), then
        installs the Kuzu FTS extension and creates the four full-text search
        indexes that graphiti-core 0.28+ requires.  Silently skips any index
        that already exists.
        """
        await self._graphiti.build_indices_and_constraints()
        self._create_fts_indexes()

    def _create_fts_indexes(self) -> None:
        """Install the Kuzu FTS extension and create required FTS indexes.

        Uses a temporary synchronous connection — the same pattern used by
        ``KuzuDriver.setup_schema()``.  Each ``CREATE_FTS_INDEX`` call is
        guarded against the "already exists" error so that subsequent
        startups are idempotent.
        """
        driver: KuzuDriver = self._graphiti.driver  # type: ignore[attr-defined]
        conn = kuzu.Connection(driver.db)
        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD EXTENSION fts")
            for table, index_name, columns in _KUZU_FTS_INDEXES:
                col_list = ", ".join(f"'{c}'" for c in columns)
                query = f"CALL CREATE_FTS_INDEX('{table}', '{index_name}', [{col_list}])"
                try:
                    conn.execute(query)
                    log.debug("Created FTS index %s on %s", index_name, table)
                except RuntimeError as exc:
                    if "already exists" in str(exc):
                        log.debug("FTS index %s already exists — skipping", index_name)
                    else:
                        raise
        finally:
            conn.close()

    async def close(self) -> None:
        """Release the graph connection."""
        await self._graphiti.close()

    # ------------------------------------------------------------------
    # Episode ingestion
    # ------------------------------------------------------------------

    async def add_episode(
        self,
        name: str,
        content: str,
        source_description: str,
        turn: int,
        group_id: str,  # accepted for API consistency; isolation is at DB-file level
    ) -> None:
        """
        Ingest a DM response as a Graphiti episode.

        Graphiti automatically extracts named entities (NPCs, locations,
        items, events) and their relationships from *content*, storing them
        as temporally-tagged nodes and edges.

        Parameters
        ----------
        name:
            Human-readable episode label, e.g. ``"turn_7"``.
        content:
            The DM narration text to be ingested.
        source_description:
            Brief description of the source, e.g. ``"DM narration"``.
        turn:
            Turn number; used only for the episode name.
        group_id:
            Campaign identifier.  Stored for documentation purposes; Graphiti
            is called with ``group_id=None`` so that the Kuzu default is used
            and the ``driver.clone()`` path (optimised for Neo4j) is avoided.
        """
        await self._graphiti.add_episode(
            name=name,
            episode_body=content,
            source_description=source_description,
            reference_time=datetime.now(),
            source=EpisodeType.message,
            group_id=None,  # use Kuzu default (''); isolation is at DB-file level
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        group_id: str,  # accepted for API consistency
        num_results: int = 10,
    ) -> str:
        """
        Hybrid search (semantic + keyword + graph traversal) over stored facts.

        Parameters
        ----------
        query:
            The current player/DM context to search against.
        group_id:
            Campaign identifier (accepted for API consistency; not passed to
            Graphiti — the entire DB is scoped to this campaign already).
        num_results:
            Maximum number of fact edges to retrieve.

        Returns
        -------
        str
            A Markdown-formatted context block, or ``""`` if nothing was found.
        """
        edges = await self._graphiti.search(
            query,
            group_ids=None,  # search entire DB (already campaign-scoped)
            num_results=num_results,
        )
        if not edges:
            return ""

        lines = ["## Remembered Context"]
        for edge in edges:
            lines.append(f"- {edge.fact}")
        return "\n".join(lines)

    async def get_entities(self) -> list[dict]:
        """
        Return all stored Entity nodes as a list of dicts.

        Each dict has ``name`` and ``summary`` keys.  Used by the ``/journal``
        CLI command.

        Returns
        -------
        list[dict]
            Entities sorted by name, or an empty list if the DB is empty.
        """
        driver: KuzuDriver = self._graphiti.driver  # type: ignore[attr-defined]
        conn = kuzu.Connection(driver.db)
        try:
            result = conn.execute("MATCH (e:Entity) RETURN e.name, e.summary")
            entities = []
            while result.has_next():
                row = result.get_next()
                entities.append({"name": row[0] or "", "summary": row[1] or ""})
            return sorted(entities, key=lambda e: e["name"].lower())
        except Exception:  # noqa: BLE001  — graph may be empty on first run
            return []
        finally:
            conn.close()

    async def search_entity(self, name: str) -> list[dict]:
        """
        Look up edges related to a named entity.

        Used by the ``/graph <entity>`` CLI command.

        Parameters
        ----------
        name:
            Entity name to search for (case-insensitive substring match).

        Returns
        -------
        list[dict]
            List of ``{"source": ..., "relation": ..., "target": ..., "fact": ...}``
            dicts, or an empty list if nothing is found.
        """
        edges = await self._graphiti.search(
            name,
            group_ids=None,
            num_results=20,
        )
        results = []
        for edge in edges:
            results.append({
                "source": getattr(edge, "source_node_uuid", ""),
                "fact": edge.fact,
                "name": getattr(edge, "name", ""),
            })
        return results
