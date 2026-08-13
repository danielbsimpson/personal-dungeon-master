"""
Phase H — RAPTOR (Recursive Abstractive Processing for Tree-Organised Retrieval)
over the DM's episode history.

Builds a recursive tree of summarised episode clusters so that high-level
narrative context is always retrievable even after many sessions.

Algorithm (simplified from Sarthi et al., 2024):
  Level 0: individual DM-turn episodes (leaves)
  Level 1+: LLM summaries of GMM-clustered node groups
  Root:     single summary of all top-level nodes

Retrieval: cosine similarity across *all* levels, with optional per-level
weight boost to favour broader or more specific context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

log = logging.getLogger(__name__)

_SUMMARISE_PROMPT = """\
You are summarising key events and revelations from a tabletop RPG campaign.

Episodes to summarise:
{episodes}

Write a concise 2-3 sentence summary of the most important events, \
characters encountered, and decisions made. Focus on information that will \
be useful for future context."""


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


@dataclass
class RaptorNode:
    """A single node in the RAPTOR tree."""

    node_id: int
    level: int           # 0 = leaf episode; 1+ = summary
    text: str            # episode text or LLM summary
    embedding: list[float]
    children: list[int]  # indices into RaptorTree.nodes (empty for leaves)


class RaptorTree:
    """Recursive tree of summarised episode clusters."""

    def __init__(self, nodes: list[RaptorNode] | None = None) -> None:
        self.nodes: list[RaptorNode] = nodes or []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    async def build(
        cls,
        episodes: list[str],
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        summarise_fn: Callable[[str], Awaitable[str]],
        max_clusters: int = 10,
        max_levels: int = 6,
    ) -> "RaptorTree":
        """
        Build a RAPTOR tree from *episodes*.

        Parameters
        ----------
        episodes:
            Raw text of each DM turn (ordered chronologically).
        embed_fn:
            Async callable: ``list[str] -> list[list[float]]``.
        summarise_fn:
            Async callable: ``str -> str`` (LLM summarisation).
        max_clusters:
            Maximum number of clusters per level.
        max_levels:
            Maximum tree depth (excludes leaf level 0).

        Returns
        -------
        RaptorTree
            Populated tree ready for :meth:`search`.
        """
        if not episodes:
            return cls()

        tree = cls()

        # Level 0: embed leaves
        log.info("RAPTOR: embedding %d episodes (level 0).", len(episodes))
        embeddings = await embed_fn(episodes)
        for i, (text, emb) in enumerate(zip(episodes, embeddings)):
            tree.nodes.append(
                RaptorNode(node_id=i, level=0, text=text, embedding=emb, children=[])
            )

        current_level_indices = list(range(len(tree.nodes)))

        for level in range(1, max_levels + 1):
            n = len(current_level_indices)
            if n <= 1:
                break  # already at root

            n_clusters = min(max_clusters, max(1, n // 3))

            if n_clusters >= n:
                # Merge everything into one summary
                log.info(
                    "RAPTOR: level %d — single summary from %d nodes.", level, n
                )
                new_indices = await cls._summarise_clusters(
                    tree, [current_level_indices], level, embed_fn, summarise_fn
                )
                current_level_indices = new_indices
                break

            log.info(
                "RAPTOR: level %d — clustering %d nodes into %d clusters.",
                level,
                n,
                n_clusters,
            )
            labels = cls._gmm_cluster(tree, current_level_indices, n_clusters)

            clusters: dict[int, list[int]] = {}
            for idx, label in zip(current_level_indices, labels):
                clusters.setdefault(label, []).append(idx)

            new_indices = await cls._summarise_clusters(
                tree, list(clusters.values()), level, embed_fn, summarise_fn
            )
            current_level_indices = new_indices

        return tree

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gmm_cluster(
        tree: "RaptorTree",
        node_indices: list[int],
        n_clusters: int,
    ) -> list[int]:
        """Cluster *node_indices* using GaussianMixture.

        Falls back to a single cluster on import error or fitting failure.
        """
        try:
            from sklearn.mixture import GaussianMixture  # type: ignore[import]
        except ImportError:
            log.warning("scikit-learn not installed; RAPTOR clustering skipped.")
            return [0] * len(node_indices)

        vectors = np.array(
            [tree.nodes[i].embedding for i in node_indices], dtype=np.float32
        )
        # Clamp to available samples
        n_clusters = min(n_clusters, len(node_indices))
        if n_clusters < 2:
            return [0] * len(node_indices)

        try:
            gm = GaussianMixture(
                n_components=n_clusters,
                covariance_type="diag",
                n_init=3,
                random_state=42,
            )
            return gm.fit_predict(vectors).tolist()
        except Exception as exc:  # noqa: BLE001
            log.warning("RAPTOR GMM failed (%s); grouping all into one cluster.", exc)
            return [0] * len(node_indices)

    @staticmethod
    async def _summarise_clusters(
        tree: "RaptorTree",
        clusters: list[list[int]],
        level: int,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        summarise_fn: Callable[[str], Awaitable[str]],
    ) -> list[int]:
        """Summarise each cluster and append the resulting nodes to *tree*.

        Returns the indices of the newly created summary nodes.
        """
        summaries: list[str] = []
        children_per_cluster: list[list[int]] = []

        for cluster_indices in clusters:
            texts = [tree.nodes[i].text for i in cluster_indices]
            episodes_text = "\n\n".join(texts)
            prompt = _SUMMARISE_PROMPT.format(episodes=episodes_text)
            try:
                summary = await summarise_fn(prompt)
                if not summary or summary.strip().upper() == "NONE":
                    summary = episodes_text[:500]
            except Exception as exc:  # noqa: BLE001
                log.warning("RAPTOR summarise failed (%s); truncating text.", exc)
                summary = episodes_text[:500]
            summaries.append(summary.strip())
            children_per_cluster.append(cluster_indices)

        # Batch-embed all summaries in one call
        embeddings = await embed_fn(summaries)

        new_indices: list[int] = []
        for summary, children, emb in zip(summaries, children_per_cluster, embeddings):
            node_id = len(tree.nodes)
            tree.nodes.append(
                RaptorNode(
                    node_id=node_id,
                    level=level,
                    text=summary,
                    embedding=emb,
                    children=children,
                )
            )
            new_indices.append(node_id)

        return new_indices

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        level_weights: dict[int, float] | None = None,
    ) -> list[RaptorNode]:
        """Retrieve the *top_k* most relevant nodes across all tree levels.

        Parameters
        ----------
        query_embedding:
            The query vector.
        top_k:
            Maximum nodes to return.
        level_weights:
            Optional per-level score multiplier (default 1.0 for all levels).
            Use ``{0: 0.5, 1: 1.0}`` to down-weight raw episodes relative
            to summaries.

        Returns
        -------
        list[RaptorNode]
            Up to *top_k* nodes sorted by descending similarity × weight.
        """
        if not self.nodes:
            return []

        weights = level_weights or {}
        scored: list[tuple[float, RaptorNode]] = []
        for node in self.nodes:
            sim = _cosine(query_embedding, node.embedding)
            w = weights.get(node.level, 1.0)
            scored.append((sim * w, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise the tree to a JSON file at *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "level": n.level,
                    "text": n.text,
                    "embedding": n.embedding,
                    "children": n.children,
                }
                for n in self.nodes
            ]
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        log.info("RAPTOR: saved %d nodes to %s.", len(self.nodes), path)

    @classmethod
    def load(cls, path: Path) -> "RaptorTree":
        """Load a previously saved tree from *path*."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        nodes = [
            RaptorNode(
                node_id=n["node_id"],
                level=n["level"],
                text=n["text"],
                embedding=n["embedding"],
                children=n["children"],
            )
            for n in raw["nodes"]
        ]
        tree = cls(nodes=nodes)
        log.info("RAPTOR: loaded %d nodes from %s.", len(nodes), path)
        return tree
