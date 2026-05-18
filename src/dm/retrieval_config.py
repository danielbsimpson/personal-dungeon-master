"""
Per-NarrativeState retrieval parameters.

Phase G — Adaptive retrieval by NarrativeState.

Retrieval parameters are tuned to the current mode of play:

  COMBAT      — tight, fast.  Fewer chunks, narrow act filter, less graph
                history (combat is real-time and fact-dense).  Always compress.
  SOCIAL      — wide, thorough.  More graph results so NPC backstory and prior
                conversation history surface reliably.  No compression (full
                NPC context reads better uncut).
  EXPLORATION — balanced defaults that work for most scene types.
  REST        — minimal.  The player is pausing; small retrieval footprint is
                sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rules.reference import NarrativeState


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval parameters for a single DM turn."""

    # ── Campaign index (Phases C / E) ─────────────────────────────────────────
    top_acts: int = 2
    """Number of acts to narrow to in the first pass of hierarchical retrieval."""

    top_chunks: int = 5
    """Number of fine-grained chunks to retrieve from the selected acts."""

    # ── RSE expansion window (Phase D) ───────────────────────────────────────
    adjacency_window: int = 2
    """Maximum chunks to expand on each side of a retrieved hit in RSE."""

    # ── Graphiti graph (Phase G) ──────────────────────────────────────────────
    graph_results: int = 10
    """Maximum number of graph fact edges to retrieve from Graphiti."""

    # ── Contextual compression (Phase F) ─────────────────────────────────────
    compress: bool = True
    """Whether to apply contextual compression to retrieved segments."""


RETRIEVAL_CONFIGS: dict[NarrativeState, RetrievalConfig] = {
    NarrativeState.COMBAT: RetrievalConfig(
        top_acts=1,
        top_chunks=4,
        adjacency_window=1,
        graph_results=6,   # combat is fast-paced; fewer facts, lower latency
        compress=True,
    ),
    NarrativeState.SOCIAL: RetrievalConfig(
        top_acts=2,
        top_chunks=6,
        adjacency_window=3,
        graph_results=15,  # NPC history matters; pull more graph context
        compress=False,    # full NPC segments read better uncut
    ),
    NarrativeState.EXPLORATION: RetrievalConfig(
        top_acts=2,
        top_chunks=5,
        adjacency_window=2,
        graph_results=10,
        compress=True,
    ),
    NarrativeState.REST: RetrievalConfig(
        top_acts=1,
        top_chunks=3,
        adjacency_window=1,
        graph_results=8,
        compress=False,
    ),
}


def get_retrieval_config(state: NarrativeState) -> RetrievalConfig:
    """Return the :class:`RetrievalConfig` for *state*.

    Falls back to the default ``RetrievalConfig()`` for any state not
    explicitly listed in ``RETRIEVAL_CONFIGS``.
    """
    return RETRIEVAL_CONFIGS.get(state, RetrievalConfig())
