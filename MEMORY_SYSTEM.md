# RAG Implementation Guide — Personal Dungeon Master

A phased implementation guide for upgrading the RAG pipeline in
[`personal-dungeon-master`](https://github.com/danielbsimpson/personal-dungeon-master),
based on techniques catalogued in
[`NirDiamant/RAG_Techniques`](https://github.com/NirDiamant/RAG_Techniques).

The guide is ordered by implementation priority. Each section covers what to
build, where it fits in the existing codebase, the code changes required, and
how to verify correctness.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Phase A — Semantic chunking of the campaign book](#2-phase-a--semantic-chunking-of-the-campaign-book) ✅
3. [Phase B — Contextual chunk headers](#3-phase-b--contextual-chunk-headers) ✅
4. [Phase C — Hierarchical indices](#4-phase-c--hierarchical-indices) ✅
5. [Phase D — Relevant segment extraction](#5-phase-d--relevant-segment-extraction) ✅
6. [Phase E — Fusion retrieval for the campaign index](#6-phase-e--fusion-retrieval-for-the-campaign-index) ⬜
7. [Phase F — Contextual compression](#7-phase-f--contextual-compression) ⬜
8. [Phase G — Adaptive retrieval by NarrativeState](#8-phase-g--adaptive-retrieval-by-narrativestate) ⬜
9. [Phase H — RAPTOR over player history](#9-phase-h--raptor-over-player-history) ⬜
10. [Phase I — MemoRAG for long campaigns](#10-phase-i--memorag-for-long-campaigns) ⬜
11. [Other techniques to explore if necessary](#11-other-techniques-to-explore-if-necessary)

---

## 1. Architecture overview

### Dual knowledge bases

The project retrieves from two fundamentally different knowledge bases on every
turn. Understanding this distinction drives all architectural decisions.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Per-turn context assembly                │
│                        (dm/context_builder.py)                   │
│                                                                   │
│  ┌──────────────────────────┐   ┌──────────────────────────────┐ │
│  │   DM world knowledge     │   │   Player journey knowledge   │ │
│  │  (static, pre-authored)  │   │  (dynamic, session-generated)│ │
│  │                          │   │                              │ │
│  │  • Campaign .txt         │   │  • Graphiti temporal graph   │ │
│  │  • 5e rules/*.md         │   │  • Session window (last N)   │ │
│  │  • creature.md           │   │  • Progress pointer          │ │
│  │  • character.md          │   │                              │ │
│  │                          │   │  ← Phases H, I target here   │ │
│  │  ← Phases A–F target here│   │                              │ │
│  └──────────────────────────┘   └──────────────────────────────┘ │
│                          │                │                       │
│              ┌───────────▼────────────────▼──────────┐           │
│              │   Shared retrieval pipeline            │           │
│              │   Fusion → Rerank → Compress           │           │
│              │   ← Phases E, F, G target here         │           │
│              └───────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                               │
                      Ollama LLM (local)
```

### Files touched by this guide

| File | Role |
|---|---|
| `src/campaign/parser.py` | Chunking logic for the campaign book |
| `src/campaign/loader.py` | Index construction on campaign load |
| `src/dm/context_builder.py` | Retrieval orchestration and prompt assembly |
| `src/dm/memory/manager.py` | Player journey retrieval (Graphiti wrapper) |
| `src/dm/memory/graphiti_store.py` | Low-level Graphiti search calls |
| `src/rules/reference.py` | Rules section selection (adaptive retrieval) |
| `src/config.py` | New settings added per phase |
| `requirements.txt` | New dependencies per phase |

### Dependency additions summary

```
# Phase A
rank-bm25>=0.2.2

# Phase C
sentence-transformers>=2.7.0   # for summary embeddings

# Phase F
# no new deps — uses existing Ollama provider

# Phase H
scipy>=1.13.0                  # for RAPTOR clustering
scikit-learn>=1.5.0
umap-learn>=0.5.6
```

---

## 2. Phase A — Semantic chunking of the campaign book ✅ COMPLETE

> **Status:** Implemented in `src/campaign/chunker.py`. Wired into `src/campaign/loader.py` via `enrich_campaign()`. Cached at `memory/<campaign>/campaign_chunks.pkl`. Tested in `tests/test_chunker.py`.

### What it is

Currently the campaign book (`[campaign_name].txt`) is split on fixed-size
windows or scene-header markers (`##`, `SCENE:`). Semantic chunking instead
finds natural topic boundaries in the text by measuring embedding similarity
between adjacent sentences: when similarity drops sharply, a new chunk begins.

### Why it matters here

Campaign books mix description, dialogue, mechanical instructions, and lore
within a single scene. A fixed-size splitter regularly cuts mid-encounter, mid
NPC-introduction, or mid-lore-reveal — forcing the retriever to piece together
a coherent scene from two separate chunks. Semantic chunking keeps narrative
units together.

### Reference technique

Technique #12 in RAG_Techniques:
`all_rag_techniques/semantic_chunking.ipynb`

### Implementation

#### 1. Add dependency

```
# requirements.txt
rank-bm25>=0.2.2         # needed later for fusion retrieval — install now
```

No new embedding dependency is needed because Graphiti already uses
`nomic-embed-text` via Ollama. Reuse that embedder.

#### 2. New file: `src/campaign/chunker.py`

```python
"""
Semantic chunker for campaign book text.

Splits the campaign book into topic-coherent chunks by measuring cosine
similarity between adjacent sentence embeddings and inserting a boundary
wherever similarity falls below a configurable threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class CampaignChunk:
    index: int
    text: str
    scene_header: str          # nearest preceding scene/chapter header
    act: str                   # top-level act or chapter label if present
    start_char: int            # character offset in original document
    end_char: int


def _sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex heuristic."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def semantic_chunk(
    text: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    breakpoint_threshold: float = 0.75,
    min_chunk_sentences: int = 3,
    max_chunk_sentences: int = 40,
) -> list[CampaignChunk]:
    """
    Segment *text* into semantically coherent chunks.

    Args:
        text: Full campaign book text.
        embed_fn: Function that accepts a list of strings and returns a list
            of float vectors (same length).  Uses the Ollama embedder already
            configured for Graphiti.
        breakpoint_threshold: Cosine similarity below this value between
            adjacent sentences triggers a new chunk boundary.
        min_chunk_sentences: Never split before this many sentences.
        max_chunk_sentences: Force a split after this many sentences.

    Returns:
        Ordered list of CampaignChunk objects.
    """
    sentences = _sentences(text)
    if not sentences:
        return []

    # Embed all sentences in one batch (Ollama handles batching internally)
    vectors = np.array(embed_fn(sentences), dtype=np.float32)

    # Identify candidate breakpoints
    boundaries: list[int] = [0]
    consecutive = 0

    for i in range(1, len(sentences)):
        sim = _cosine(vectors[i - 1], vectors[i])
        consecutive += 1

        force_split = consecutive >= max_chunk_sentences
        semantic_split = (
            sim < breakpoint_threshold and consecutive >= min_chunk_sentences
        )

        if force_split or semantic_split:
            boundaries.append(i)
            consecutive = 0

    boundaries.append(len(sentences))

    # Build CampaignChunk objects
    chunks: list[CampaignChunk] = []
    current_act = "Introduction"
    current_scene = "Prologue"
    char_offset = 0

    header_re = re.compile(
        r"^(#{1,3}|SCENE:|ACT:|CHAPTER:)\s*(.+)$", re.IGNORECASE | re.MULTILINE
    )

    for idx in range(len(boundaries) - 1):
        start_sent = boundaries[idx]
        end_sent = boundaries[idx + 1]
        chunk_text = " ".join(sentences[start_sent:end_sent])

        # Update act/scene from any headers appearing in this chunk
        for m in header_re.finditer(chunk_text):
            marker = m.group(1).upper().rstrip(":")
            label = m.group(2).strip()
            if marker in ("ACT", "CHAPTER", "###"):
                current_act = label
            else:
                current_scene = label

        start_char = text.find(sentences[start_sent], char_offset)
        end_char = (
            text.find(sentences[end_sent - 1], start_char) + len(sentences[end_sent - 1])
            if end_sent <= len(sentences)
            else len(text)
        )
        char_offset = max(char_offset, end_char)

        chunks.append(
            CampaignChunk(
                index=idx,
                text=chunk_text,
                scene_header=current_scene,
                act=current_act,
                start_char=start_char,
                end_char=end_char,
            )
        )

    return chunks
```

#### 3. Wire into `src/campaign/parser.py`

```python
# In ParsedCampaign, replace the raw campaign_text string with:
from src.campaign.chunker import CampaignChunk, semantic_chunk

@dataclass
class ParsedCampaign:
    summary: str
    character: Character
    creatures: list[Creature]
    campaign_text: str              # raw text kept for spoiler guard
    chunks: list[CampaignChunk]     # NEW: semantic chunks


# In the parsing function, after loading campaign_text:
def parse_campaign(campaign: Campaign, embed_fn) -> ParsedCampaign:
    # ... existing parsing ...
    chunks = semantic_chunk(campaign_text, embed_fn)
    return ParsedCampaign(..., chunks=chunks)
```

#### 4. New setting in `src/config.py`

```python
CHUNK_BREAKPOINT_THRESHOLD: float = 0.75   # cosine similarity floor
CHUNK_MIN_SENTENCES: int = 3
CHUNK_MAX_SENTENCES: int = 40
```

### Verification

```python
# tests/test_chunker.py
def test_chunks_respect_min_length():
    chunks = semantic_chunk(SAMPLE_TEXT, mock_embed_fn, min_chunk_sentences=3)
    assert all(len(c.text.split(".")) >= 2 for c in chunks)

def test_scene_header_propagates():
    text = "## The Ruined Keep\nYou see crumbling walls. Goblins lurk."
    chunks = semantic_chunk(text, mock_embed_fn)
    assert chunks[0].scene_header == "The Ruined Keep"

def test_no_empty_chunks():
    chunks = semantic_chunk(SAMPLE_TEXT, mock_embed_fn)
    assert all(c.text.strip() for c in chunks)
```

---

## 3. Phase B — Contextual chunk headers ✅ COMPLETE

> **Status:** Implemented in `src/campaign/header.py`. `with_header()` is called during `CampaignIndex.build()` to enrich chunk embeddings with act/scene context. Tested in `tests/test_chunker.py`.

### What it is

Before embedding each campaign chunk, prepend a short metadata header
describing the document context: act, chapter, and scene. This metadata is
baked into the embedding so that retrieval naturally filters by narrative
location, not just semantic content.

### Why it matters here

The same enemy type (goblins, cultists, bandits) appears in multiple acts.
Without context headers, "goblin encounter" retrieval returns chunks from
different points in the story indiscriminately. A header like
`[Act 1 | Scene: Goblin Ambush on the Trade Road]` disambiguates.

### Reference technique

Technique #9 in RAG_Techniques:
`all_rag_techniques/contextual_chunk_headers.ipynb`

### Implementation

#### 1. New file: `src/campaign/header.py`

```python
"""
Prepends a context header to each CampaignChunk before embedding.

The header is stored separately from the chunk text so that:
  - The retrieval embedding includes the context signal.
  - The text injected into the LLM prompt is the raw chunk only
    (avoiding redundant metadata in the generated response).
"""

from __future__ import annotations
from src.campaign.chunker import CampaignChunk


HEADER_TEMPLATE = "[{act} | {scene}]\n"


def with_header(chunk: CampaignChunk, campaign_name: str) -> str:
    """
    Return the chunk text prefixed with a context header.
    This string is what gets embedded — NOT what gets injected into
    the LLM prompt.
    """
    header = HEADER_TEMPLATE.format(
        act=chunk.act,
        scene=chunk.scene_header,
    )
    return header + chunk.text


def header_string(chunk: CampaignChunk) -> str:
    """Return just the header, for display or debugging."""
    return HEADER_TEMPLATE.format(act=chunk.act, scene=chunk.scene_header)
```

#### 2. Update the index builder (Phase C will formalise this)

When embedding chunks during index construction, pass `with_header(chunk)`
to the embedder, but store `chunk.text` as the retrieval payload:

```python
# Pseudocode in the index builder
for chunk in parsed_campaign.chunks:
    embedding = embed(with_header(chunk, campaign_name))   # context-enriched vector
    index.add(embedding=embedding, payload=chunk.text, metadata=chunk)
```

This ensures the embedding space captures narrative location while the text
returned to the LLM is clean campaign prose.

### Verification

```python
def test_header_contains_act_and_scene():
    chunk = CampaignChunk(index=0, text="Goblins attack.", scene_header="Ambush", act="Act 1", ...)
    h = with_header(chunk, "lost-mines")
    assert "[Act 1 | Ambush]" in h
    assert "Goblins attack." in h

def test_raw_text_unchanged():
    chunk = CampaignChunk(index=0, text="Goblins attack.", ...)
    assert with_header(chunk, "lost-mines").endswith(chunk.text)
```

---

## 4. Phase C — Hierarchical indices ✅ COMPLETE

> **Status:** Implemented in `src/campaign/index.py`. `CampaignIndex.build()` creates act summaries (tier 1) and fine-grained chunk embeddings (tier 2). Cached at `memory/<campaign>/campaign_index.pkl`. `context_builder.py` uses two-pass hierarchical retrieval when the index is present, falling back to the scene window when not. Tested in `tests/test_chunker.py`.

### What it is

Two-tier retrieval over the campaign book. The first tier contains
act/chapter-level summary chunks (high-level: "what happens in this act").
The second tier contains the fine-grained semantic chunks from Phase A.
A query first retrieves the most relevant act summaries, then retrieves
fine-grained chunks filtered to those acts.

### Why it matters here

A player asking "what do I know about this dungeon?" should not return
isolated sentences from 15 different encounters scattered across acts. It
should first narrow to "we are in Act 2 – The Cragmaw Hideout" and then
retrieve the specific rooms and encounters within that act. This two-pass
approach dramatically reduces retrieval noise.

### Reference technique

Technique #18 in RAG_Techniques:
`all_rag_techniques/hierarchical_indices.ipynb`

### Implementation

#### 1. Add dependency

```
# requirements.txt
sentence-transformers>=2.7.0   # for offline summary embedding if Ollama is busy
```

Using the existing Ollama embedder is preferred; `sentence-transformers`
is a fallback for batch preprocessing.

#### 2. New file: `src/campaign/index.py`

```python
"""
Two-tier campaign book index.

Tier 1 (coarse): act/chapter summaries — one entry per act.
Tier 2 (fine):   semantic chunks from Phase A — many entries per act.

Both tiers embed with contextual headers from Phase B.
Both are stored in memory at session start; for very long campaigns
they should be persisted to disk alongside the Kuzu graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.campaign.chunker import CampaignChunk
from src.campaign.header import with_header


@dataclass
class ActSummary:
    act: str
    summary_text: str
    chunk_indices: list[int]       # which fine-grained chunks belong to this act
    embedding: list[float] = field(default_factory=list, repr=False)


@dataclass
class CampaignIndex:
    act_summaries: list[ActSummary]
    chunks: list[CampaignChunk]
    chunk_embeddings: list[list[float]] = field(default_factory=list, repr=False)

    # ── Construction ──────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        chunks: list[CampaignChunk],
        embed_fn: Callable[[list[str]], list[list[float]]],
        summarise_fn: Callable[[str], str],   # LLM call to produce act summary
        campaign_name: str,
    ) -> "CampaignIndex":
        # 1. Group chunks by act
        acts: dict[str, list[CampaignChunk]] = {}
        for chunk in chunks:
            acts.setdefault(chunk.act, []).append(chunk)

        # 2. Build act summaries (one LLM call per act)
        act_summaries: list[ActSummary] = []
        for act_name, act_chunks in acts.items():
            act_text = "\n\n".join(c.text for c in act_chunks)
            summary = summarise_fn(
                f"Summarise the following campaign act in 3-5 sentences "
                f"for a dungeon master's reference:\n\n{act_text}"
            )
            act_summaries.append(
                ActSummary(
                    act=act_name,
                    summary_text=summary,
                    chunk_indices=[c.index for c in act_chunks],
                )
            )

        # 3. Embed act summaries
        act_texts = [f"[{s.act}]\n{s.summary_text}" for s in act_summaries]
        act_embeddings = embed_fn(act_texts)
        for summary, emb in zip(act_summaries, act_embeddings):
            summary.embedding = emb

        # 4. Embed fine-grained chunks (with contextual headers)
        chunk_texts = [with_header(c, campaign_name) for c in chunks]
        chunk_embeddings = embed_fn(chunk_texts)

        return cls(
            act_summaries=act_summaries,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
        )

    # ── Retrieval ─────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: list[float],
        progress_index: int,           # spoiler guard: only chunks up to this index
        top_acts: int = 2,
        top_chunks: int = 5,
    ) -> list[CampaignChunk]:
        """Two-pass hierarchical retrieval."""
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

        # Pass 2: retrieve fine-grained chunks within those acts
        # Enforce spoiler guard: only chunks whose index <= progress_index
        candidate_chunks = [
            (i, c)
            for i, c in enumerate(self.chunks)
            if c.act in relevant_acts and c.index <= progress_index
        ]

        if not candidate_chunks:
            # Fallback: ignore act filter, still respect spoiler guard
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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0
```

#### 3. Integrate into `src/campaign/loader.py`

```python
# After parsing the campaign, build the index:
from src.campaign.index import CampaignIndex

parsed = parse_campaign(campaign, embed_fn=graphiti_embed)
index = CampaignIndex.build(
    chunks=parsed.chunks,
    embed_fn=graphiti_embed,
    summarise_fn=llm_provider.complete_single,
    campaign_name=campaign.name,
)
# Attach to parsed campaign for downstream use
parsed.index = index
```

#### 4. Update `src/dm/context_builder.py`

```python
# Replace the spoiler-guard text slice with hierarchical retrieval:
relevant_chunks = parsed_campaign.index.retrieve(
    query_embedding=embed(player_input),
    progress_index=memory.campaign_progress,
    top_acts=2,
    top_chunks=5,
)
campaign_context = "\n\n---\n\n".join(c.text for c in relevant_chunks)
```

### Persistence

For campaigns with many acts, build the index once and pickle it to
`memory/<campaign_name>/campaign_index.pkl` so it does not rebuild on every
session start.

```python
import pickle
cache_path = memory_dir / campaign_name / "campaign_index.pkl"
if cache_path.exists():
    with open(cache_path, "rb") as f:
        index = pickle.load(f)
else:
    index = CampaignIndex.build(...)
    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
```

### Verification

```python
def test_retrieval_respects_spoiler_guard():
    index = build_test_index(chunks=make_chunks(acts=["Act 1", "Act 2"]))
    results = index.retrieve(query_embedding=..., progress_index=5)
    assert all(c.index <= 5 for c in results)

def test_top_acts_narrows_results():
    index = build_test_index(...)
    results = index.retrieve(..., top_acts=1)
    acts_returned = {c.act for c in results}
    assert len(acts_returned) <= 1
```

---

## 5. Phase D — Relevant segment extraction ✅ COMPLETE

> **Status:** Implemented in `src/campaign/segment_extractor.py`. `extract_segments()` expands each retrieved chunk to adjacent neighbours and merges overlapping windows before prompt injection. Wired into `context_builder.py`. Tested in `tests/test_chunker.py`.

### What it is

After the hierarchical index returns its top chunks (Phase C), a
post-processing step assembles contiguous multi-chunk segments by looking at
neighbouring chunks around each retrieved hit. If chunk 7 is retrieved and
chunks 6 and 8 are thematically adjacent, they are merged into a single segment
before injection into the prompt.

### Why it matters here

Campaign encounters are not atomic. A retrieved chunk about a goblin archer may
need the preceding chunk (the room description) and the following chunk (the
trap mechanic) to give the DM enough context to narrate accurately. RSE
reconstructs these natural segments automatically.

### Reference technique

Technique #10 in RAG_Techniques:
`all_rag_techniques/relevant_segment_extraction.ipynb`

### Implementation

#### New file: `src/campaign/segment_extractor.py`

```python
"""
Relevant Segment Extraction (RSE) for campaign chunks.

Given a list of retrieved chunks and the full ordered chunk list,
expand each retrieved chunk to include its immediate neighbours when
they are thematically adjacent (cosine similarity above threshold).
Overlapping expanded windows are merged.
"""

from __future__ import annotations

from src.campaign.chunker import CampaignChunk
import numpy as np


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def extract_segments(
    retrieved: list[CampaignChunk],
    all_chunks: list[CampaignChunk],
    all_embeddings: list[list[float]],
    adjacency_threshold: float = 0.72,
    max_window: int = 3,
) -> list[str]:
    """
    Expand each retrieved chunk into a contiguous segment.

    Args:
        retrieved: Chunks returned by the hierarchical index.
        all_chunks: Complete ordered list of campaign chunks.
        all_embeddings: Embeddings aligned with all_chunks.
        adjacency_threshold: Minimum cosine similarity for a neighbour
            to be included in the same segment.
        max_window: Maximum number of chunks to expand on each side.

    Returns:
        List of merged segment texts, deduplicated and ordered.
    """
    chunk_by_idx = {c.index: (c, all_embeddings[i]) for i, c in enumerate(all_chunks)}
    covered: set[int] = set()
    segments: list[tuple[int, int]] = []   # (start_index, end_index) inclusive

    for hit in retrieved:
        if hit.index in covered:
            continue

        start = hit.index
        end = hit.index

        # Expand backwards
        for step in range(1, max_window + 1):
            prev_idx = hit.index - step
            if prev_idx < 0 or prev_idx not in chunk_by_idx:
                break
            prev_chunk, prev_emb = chunk_by_idx[prev_idx]
            _, hit_emb = chunk_by_idx[hit.index]
            if _cosine(hit_emb, prev_emb) < adjacency_threshold:
                break
            start = prev_idx

        # Expand forwards
        for step in range(1, max_window + 1):
            next_idx = hit.index + step
            if next_idx not in chunk_by_idx:
                break
            next_chunk, next_emb = chunk_by_idx[next_idx]
            _, hit_emb = chunk_by_idx[hit.index]
            if _cosine(hit_emb, next_emb) < adjacency_threshold:
                break
            end = next_idx

        segments.append((start, end))
        covered.update(range(start, end + 1))

    # Merge overlapping segments
    segments.sort()
    merged: list[tuple[int, int]] = []
    for seg in segments:
        if merged and seg[0] <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(list(seg))

    # Assemble text
    result: list[str] = []
    for start, end in merged:
        texts = [
            chunk_by_idx[i][0].text
            for i in range(start, end + 1)
            if i in chunk_by_idx
        ]
        result.append("\n".join(texts))

    return result
```

#### Wire into `context_builder.py`

```python
from src.campaign.segment_extractor import extract_segments

retrieved_chunks = parsed_campaign.index.retrieve(...)
segments = extract_segments(
    retrieved=retrieved_chunks,
    all_chunks=parsed_campaign.chunks,
    all_embeddings=parsed_campaign.index.chunk_embeddings,
)
campaign_context = "\n\n---\n\n".join(segments)
```

### Verification

```python
def test_adjacent_chunks_are_merged():
    # chunks 3, 4, 5 have high mutual similarity; retrieval hits chunk 4
    segments = extract_segments(retrieved=[chunk_4], all_chunks=all_chunks, ...)
    assert "chunk 3 text" in segments[0]
    assert "chunk 5 text" in segments[0]

def test_distant_chunks_are_not_merged():
    segments = extract_segments(retrieved=[chunk_1, chunk_9], ...)
    assert len(segments) == 2
```

---

## 6. Phase E — Fusion retrieval for the campaign index ⬜ TODO

> **Status:** Not yet implemented. `rank-bm25` is already in `requirements.txt`. When implemented, `CampaignIndex.retrieve()` will be extended to accept `query_text` and fuse BM25 results with semantic results via RRF.

### What it is

Augment the semantic (embedding) search in Phase C with a BM25 keyword search
over the same chunk corpus. The two result sets are combined using Reciprocal
Rank Fusion (RRF), which averages ranks rather than scores and handles
different score scales gracefully.

### Why it matters here

TTRPG content is heavily proper-noun-driven. Semantic search on "Reidoth the
druid" may not surface the exact chunk because it hasn't built strong semantic
associations with that name. BM25 finds it by exact or near-exact token match.
Fusion retrieval captures both signals.

### Reference technique

Technique #15 in RAG_Techniques:
`all_rag_techniques/fusion_retrieval.ipynb`

### Implementation

#### New file: `src/campaign/bm25_index.py`

```python
"""BM25 keyword index over campaign chunks."""

from __future__ import annotations

import re
from rank_bm25 import BM25Okapi

from src.campaign.chunker import CampaignChunk


def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25CampaignIndex:
    def __init__(self, chunks: list[CampaignChunk]) -> None:
        self.chunks = chunks
        corpus = [_tokenise(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, CampaignChunk]]:
        tokens = _tokenise(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [(score, self.chunks[idx]) for idx, score in ranked]
```

#### New file: `src/campaign/fusion.py`

```python
"""
Reciprocal Rank Fusion (RRF) for combining semantic and BM25 results.
"""

from __future__ import annotations

from src.campaign.chunker import CampaignChunk


def rrf(
    semantic_results: list[CampaignChunk],
    keyword_results: list[tuple[float, CampaignChunk]],
    k: int = 60,
    top_n: int = 5,
) -> list[CampaignChunk]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    Args:
        semantic_results: Ordered list from embedding search (best first).
        keyword_results:  List of (score, chunk) from BM25 (best first).
        k: RRF constant — larger k reduces the impact of top rankings.
        top_n: Number of final results to return.

    Returns:
        Re-ranked list of chunks.
    """
    scores: dict[int, float] = {}

    for rank, chunk in enumerate(semantic_results):
        scores[chunk.index] = scores.get(chunk.index, 0.0) + 1.0 / (k + rank + 1)

    for rank, (_, chunk) in enumerate(keyword_results):
        scores[chunk.index] = scores.get(chunk.index, 0.0) + 1.0 / (k + rank + 1)

    # Map index → chunk object
    chunk_map = {c.index: c for c in semantic_results}
    chunk_map.update({c.index: c for _, c in keyword_results})

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[idx] for idx, _ in ranked[:top_n] if idx in chunk_map]
```

#### Update `CampaignIndex.build` (Phase C) to also build BM25

```python
# In CampaignIndex.build():
from src.campaign.bm25_index import BM25CampaignIndex

self.bm25_index = BM25CampaignIndex(chunks)
```

#### Update `CampaignIndex.retrieve` to use fusion

```python
from src.campaign.fusion import rrf

def retrieve(self, query_embedding, query_text, progress_index, top_acts=2, top_chunks=5):
    # ... existing semantic retrieval ...
    semantic_results = [c for _, c in chunk_scores[:top_chunks * 2]]

    # BM25 retrieval
    bm25_results = self.bm25_index.search(query_text, top_k=top_chunks * 2)
    # Apply spoiler guard to BM25 results
    bm25_results = [(s, c) for s, c in bm25_results if c.index <= progress_index]

    # Fuse
    return rrf(semantic_results, bm25_results, top_n=top_chunks)
```

### Verification

```python
def test_proper_noun_retrieval():
    # BM25 should surface "Reidoth the druid" even with weak semantic signal
    results = index.retrieve(
        query_embedding=embed("who is Reidoth"),
        query_text="who is Reidoth",
        progress_index=99,
    )
    assert any("Reidoth" in c.text for c in results)

def test_rrf_combines_ranks():
    merged = rrf(semantic=[chunk_a, chunk_b], keyword=[(0.9, chunk_b), (0.7, chunk_c)])
    # chunk_b appears in both lists and should rank highest
    assert merged[0].index == chunk_b.index
```

---

## 7. Phase F — Contextual compression ⬜ TODO

> **Status:** Not yet implemented. Will add `src/campaign/compressor.py` and a `COMPRESSION_ENABLED` config flag. No new dependencies required.

### What it is

After fusion retrieval assembles segments (Phase D), pass each segment through
a compression step: prompt the LLM to extract only the sentences relevant to
the current player action. The compressed output is what gets injected into the
system prompt, not the full segment.

### Why it matters here

On an 8B parameter model with an 8K–16K context window, every token saved is a
token that can be spent on session history, rules, and character context.
Campaign segments can be 400–800 tokens; compressing to the 50–150 most
relevant tokens dramatically extends effective context without losing fidelity.

### Reference technique

Technique #13 in RAG_Techniques:
`all_rag_techniques/contextual_compression.ipynb`

### Implementation

#### New file: `src/campaign/compressor.py`

```python
"""
Contextual compression: given a retrieved segment and the player's current
action, extract only the sentences relevant to that action.
"""

from __future__ import annotations

from typing import Callable


COMPRESSION_PROMPT = """\
You are a dungeon master assistant. Extract only the sentences from the passage
below that are directly relevant to the player's current action. Return the
extracted sentences verbatim, separated by spaces. If nothing is relevant,
return the single word NONE.

Player action: {query}

Passage:
{passage}

Relevant sentences:"""


def compress(
    passage: str,
    query: str,
    llm_fn: Callable[[str], str],
    max_passage_tokens: int = 800,
) -> str:
    """
    Compress *passage* to the sentences most relevant to *query*.

    Args:
        passage: Retrieved campaign segment text.
        query: Player's current input.
        llm_fn: Function that accepts a prompt string and returns a string.
        max_passage_tokens: Truncate passage before sending to LLM.

    Returns:
        Compressed passage, or the original passage if compression fails
        or returns NONE.
    """
    # Rough token estimate: 1 token ≈ 4 characters
    truncated = passage[: max_passage_tokens * 4]

    prompt = COMPRESSION_PROMPT.format(query=query, passage=truncated)
    result = llm_fn(prompt).strip()

    if not result or result.upper() == "NONE":
        return passage   # fallback: return full passage

    return result
```

#### Wire into `context_builder.py`

```python
from src.campaign.compressor import compress

segments = extract_segments(...)
compressed_segments = [
    compress(seg, player_input, llm_provider.complete_single)
    for seg in segments
]
campaign_context = "\n\n---\n\n".join(compressed_segments)
```

#### New setting

```python
# src/config.py
COMPRESSION_MAX_PASSAGE_TOKENS: int = 800
COMPRESSION_ENABLED: bool = True   # disable for debugging
```

### Verification

```python
def test_compression_reduces_length():
    long_segment = "Relevant sentence. Irrelevant content. " * 50
    result = compress(long_segment, "player attacks goblin", mock_llm)
    assert len(result) < len(long_segment)

def test_compression_falls_back_on_none():
    result = compress("Unrelated passage.", "very specific query", lambda _: "NONE")
    assert result == "Unrelated passage."
```

---

## 8. Phase G — Adaptive retrieval by NarrativeState ⬜ TODO

> **Status:** Not yet implemented. Will add `src/dm/retrieval_config.py` with per-`NarrativeState` retrieval parameters, then thread `config` into `context_builder.py`'s retrieval calls.

### What it is

Use the existing `NarrativeState` enum (`EXPLORATION`, `COMBAT`, `SOCIAL`,
`REST`) as a signal to change retrieval parameters — which knowledge bases to
query, how many chunks to retrieve, and from which tiers — rather than using
fixed parameters on every turn.

### Why it matters here

The DM's information needs change dramatically by scene type. In combat, the
system needs creature stat blocks and initiative facts quickly, with high
precision. In social scenes, it needs NPC backstory and prior conversation
history, tolerating more tokens for richer personality context. In exploration,
it needs location descriptions and environmental hazards. Adaptive retrieval
tailors the query strategy to these different modes automatically.

### Reference technique

Technique #22 in RAG_Techniques:
`all_rag_techniques/adaptive_retrieval.ipynb`

### Implementation

#### New file: `src/dm/retrieval_config.py`

```python
"""
Per-NarrativeState retrieval parameters.

These control what gets retrieved and from which source on each turn,
allowing the system to focus on the most relevant knowledge for the
current scene type.
"""

from __future__ import annotations

from dataclasses import dataclass
from src.rules.reference import NarrativeState


@dataclass(frozen=True)
class RetrievalConfig:
    # Campaign index
    top_acts: int = 2
    top_chunks: int = 5
    adjacency_window: int = 2          # RSE expansion window
    # Graphiti graph
    graph_results: int = 10
    # Rules
    include_rules_sections: tuple[str, ...] = ("core",)
    # Compression
    compress: bool = True


RETRIEVAL_CONFIGS: dict[NarrativeState, RetrievalConfig] = {
    NarrativeState.COMBAT: RetrievalConfig(
        top_acts=1,
        top_chunks=4,
        adjacency_window=1,
        graph_results=6,             # fewer graph results — combat is fast-paced
        include_rules_sections=("core", "combat", "conditions"),
        compress=True,
    ),
    NarrativeState.SOCIAL: RetrievalConfig(
        top_acts=2,
        top_chunks=6,
        adjacency_window=3,
        graph_results=15,            # more graph results — NPC history matters
        include_rules_sections=("core",),
        compress=False,              # NPC context benefits from full segments
    ),
    NarrativeState.EXPLORATION: RetrievalConfig(
        top_acts=2,
        top_chunks=5,
        adjacency_window=2,
        graph_results=10,
        include_rules_sections=("core", "equipment"),
        compress=True,
    ),
    NarrativeState.REST: RetrievalConfig(
        top_acts=1,
        top_chunks=3,
        adjacency_window=1,
        graph_results=8,
        include_rules_sections=("core",),
        compress=False,
    ),
}


def get_retrieval_config(state: NarrativeState) -> RetrievalConfig:
    return RETRIEVAL_CONFIGS.get(state, RetrievalConfig())
```

#### Update `context_builder.py`

```python
from src.dm.retrieval_config import get_retrieval_config

config = get_retrieval_config(narrative_state)

retrieved_chunks = parsed_campaign.index.retrieve(
    query_embedding=embed(player_input),
    query_text=player_input,
    progress_index=memory.campaign_progress,
    top_acts=config.top_acts,
    top_chunks=config.top_chunks,
)
segments = extract_segments(
    retrieved=retrieved_chunks,
    all_chunks=parsed_campaign.chunks,
    all_embeddings=parsed_campaign.index.chunk_embeddings,
    max_window=config.adjacency_window,
)
if config.compress:
    segments = [compress(s, player_input, llm_provider.complete_single) for s in segments]

graph_context = await memory.get_context(
    player_input, group_id=campaign_name, num_results=config.graph_results
)
rules_text = get_relevant_rules(rules_ref, narrative_state)
```

### Verification

```python
def test_combat_config_has_fewer_graph_results():
    combat = get_retrieval_config(NarrativeState.COMBAT)
    social = get_retrieval_config(NarrativeState.SOCIAL)
    assert combat.graph_results < social.graph_results

def test_all_states_have_configs():
    for state in NarrativeState:
        config = get_retrieval_config(state)
        assert config is not None
```

---

## 9. Phase H — RAPTOR over player history ⬜ TODO

> **Status:** Not yet implemented. Requires `scipy`, `scikit-learn`, and `umap-learn`. Will add `src/dm/memory/raptor.py` and integrate into `MemoryManager.get_context()`.

### What it is

RAPTOR (Recursive Abstractive Processing for Tree-Organised Retrieval) builds a
tree of summaries over the Graphiti episode corpus. Leaf nodes are individual
turn episodes. Parent nodes are LLM-generated summaries of clusters of related
episodes. Top nodes are high-level session or arc summaries.

At retrieval time, the tree is searched at multiple levels simultaneously,
returning both high-level narrative arc context and specific detailed facts.

### Why it matters here

After 5+ sessions, the Graphiti graph accumulates hundreds of episodes.
Searching all of them on every turn becomes slow and noisy. RAPTOR's tree
structure allows the DM to answer "what has happened with the thieves' guild
arc overall?" (top-level summary node) and "what did the player say to Gundren
last session?" (leaf node) within a single retrieval pass.

### Reference technique

Technique #28 in RAG_Techniques:
`all_rag_techniques/raptor.ipynb`

### Dependencies

```
# requirements.txt
scipy>=1.13.0
scikit-learn>=1.5.0
umap-learn>=0.5.6
```

### Implementation

#### New file: `src/dm/memory/raptor.py`

```python
"""
RAPTOR: Recursive Abstractive Processing for Tree-Organised Retrieval.

Applied to the Graphiti episode corpus (player journey knowledge base).
Builds a summary tree that is persisted alongside the Kuzu graph and
updated periodically (e.g. every N turns or on session end).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class RaptorNode:
    level: int              # 0 = leaf episode, 1+ = summary levels
    text: str               # episode text or generated summary
    embedding: list[float] = field(default_factory=list, repr=False)
    children: list[int] = field(default_factory=list)   # child node indices
    node_id: int = 0


class RaptorTree:
    """
    A multi-level summary tree over episode texts.

    Construction:
        1. Embed all episode texts (leaves).
        2. Cluster leaves using Gaussian Mixture Model.
        3. Summarise each cluster with the LLM → level-1 nodes.
        4. Repeat on level-1 nodes until 1 root node remains.

    Retrieval:
        Embed the query. Search all nodes at all levels simultaneously.
        Return top-k nodes, mixing levels for multi-granularity context.
    """

    def __init__(self) -> None:
        self.nodes: list[RaptorNode] = []

    @classmethod
    def build(
        cls,
        episodes: list[str],
        embed_fn: Callable[[list[str]], list[list[float]]],
        summarise_fn: Callable[[str], str],
        max_clusters: int = 10,
    ) -> "RaptorTree":
        tree = cls()
        if not episodes:
            return tree

        # Level 0: embed all episodes
        embeddings = embed_fn(episodes)
        for i, (text, emb) in enumerate(zip(episodes, embeddings)):
            node = RaptorNode(level=0, text=text, embedding=emb, node_id=i)
            tree.nodes.append(node)

        current_level_nodes = list(tree.nodes)
        level = 1

        while len(current_level_nodes) > 1:
            n_clusters = min(max_clusters, max(2, len(current_level_nodes) // 3))
            vectors = np.array([n.embedding for n in current_level_nodes], dtype=np.float32)

            # Cluster using Gaussian Mixture Model (soft clustering)
            gm = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = gm.fit_predict(vectors)

            # Build summary nodes
            next_level_nodes: list[RaptorNode] = []
            for cluster_id in range(n_clusters):
                members = [
                    current_level_nodes[i]
                    for i, lbl in enumerate(labels)
                    if lbl == cluster_id
                ]
                if not members:
                    continue

                combined = "\n\n".join(m.text for m in members)
                summary = summarise_fn(
                    f"Summarise the following events from a D&D campaign "
                    f"in 3-5 sentences for a dungeon master's memory:\n\n{combined}"
                )
                node_id = len(tree.nodes)
                summary_emb = embed_fn([summary])[0]
                summary_node = RaptorNode(
                    level=level,
                    text=summary,
                    embedding=summary_emb,
                    children=[m.node_id for m in members],
                    node_id=node_id,
                )
                tree.nodes.append(summary_node)
                next_level_nodes.append(summary_node)

            current_level_nodes = next_level_nodes
            level += 1
            if level > 6:   # guard against degenerate input
                break

        return tree

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        level_weights: dict[int, float] | None = None,
    ) -> list[RaptorNode]:
        """
        Retrieve top-k nodes across all levels.

        level_weights: Optional per-level boost. Defaults to uniform.
        """
        q = np.array(query_embedding, dtype=np.float32)
        level_weights = level_weights or {}

        scored = []
        for node in self.nodes:
            emb = np.array(node.embedding, dtype=np.float32)
            denom = np.linalg.norm(q) * np.linalg.norm(emb)
            sim = float(np.dot(q, emb) / denom) if denom else 0.0
            boost = level_weights.get(node.level, 1.0)
            scored.append((sim * boost, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: Path) -> None:
        data = [
            {
                "node_id": n.node_id,
                "level": n.level,
                "text": n.text,
                "embedding": n.embedding,
                "children": n.children,
            }
            for n in self.nodes
        ]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "RaptorTree":
        tree = cls()
        for item in json.loads(path.read_text()):
            tree.nodes.append(RaptorNode(**item))
        return tree
```

#### Integrate into `MemoryManager`

```python
# src/dm/memory/manager.py
from src.dm.memory.raptor import RaptorTree

class MemoryManager:
    def __init__(self, ...):
        ...
        self.raptor_tree: RaptorTree | None = None
        self._raptor_path: Path | None = None

    async def load(self, campaign_name: str):
        ...
        raptor_path = memory_dir / campaign_name / "raptor_tree.json"
        self._raptor_path = raptor_path
        if raptor_path.exists():
            self.raptor_tree = RaptorTree.load(raptor_path)

    async def rebuild_raptor(self, episodes: list[str]):
        """
        Rebuild the RAPTOR tree from recent episodes.
        Call at session end or every RAPTOR_REBUILD_EVERY turns.
        """
        self.raptor_tree = RaptorTree.build(
            episodes=episodes,
            embed_fn=self._embed_fn,
            summarise_fn=self._llm_fn,
        )
        if self._raptor_path:
            self.raptor_tree.save(self._raptor_path)

    async def get_context(self, query: str, group_id: str) -> str:
        graph_context = await self._graphiti_store.search(query, group_id)

        raptor_context = ""
        if self.raptor_tree:
            q_emb = self._embed_fn([query])[0]
            nodes = self.raptor_tree.search(q_emb, top_k=3)
            raptor_context = "\n".join(n.text for n in nodes)

        parts = [p for p in [raptor_context, graph_context] if p]
        return "\n\n---\n\n".join(parts)
```

#### New settings

```python
# src/config.py
RAPTOR_ENABLED: bool = True
RAPTOR_REBUILD_EVERY: int = 10   # turns between rebuilds
RAPTOR_MAX_CLUSTERS: int = 10
```

### Verification

```python
def test_raptor_builds_multi_level_tree():
    tree = RaptorTree.build(episodes=["ep1", "ep2", "ep3", "ep4"], ...)
    levels = {n.level for n in tree.nodes}
    assert max(levels) >= 1   # at least one summary level

def test_raptor_search_returns_mixed_levels():
    tree = RaptorTree.build(episodes=[...] * 12, ...)
    results = tree.search(query_embedding=embed("goblin"), top_k=5)
    result_levels = {n.level for n in results}
    assert len(result_levels) > 1   # not all from the same level

def test_raptor_persists_and_reloads(tmp_path):
    tree = RaptorTree.build(["ep1", "ep2", "ep3"], ...)
    tree.save(tmp_path / "tree.json")
    loaded = RaptorTree.load(tmp_path / "tree.json")
    assert len(loaded.nodes) == len(tree.nodes)
```

---

## 10. Phase I — MemoRAG for long campaigns ⬜ TODO

> **Status:** Not yet implemented. Will add `src/dm/memory/memo_rag.py` and integrate into `MemoryManager.get_context()` and `MemoryManager.end_of_session()`.

### What it is

MemoRAG adds a lightweight global memory layer that pre-generates "clues" —
short natural-language signals — from the full episode corpus before retrieval
runs. These clues guide the retrieval system toward the right part of the
knowledge base, effectively acting as a query router for long campaigns where
the DM might not know which session an important fact lives in.

### Why it matters here

After a long campaign, the player might say "I want to revisit the merchant I
met early on." Without MemoRAG, the retrieval system has no signal about which
session or graph cluster the merchant appears in. MemoRAG generates clues like
"The merchant Toblen was introduced in session 2 and has a complicated
relationship with the Redbrands" and surfaces them before the main retrieval
pass, steering it correctly.

### Reference technique

Technique #34 in RAG_Techniques:
`all_rag_techniques/memorag.ipynb`

### Implementation

#### New file: `src/dm/memory/memo_rag.py`

```python
"""
MemoRAG: global memory clue store for long campaigns.

Clues are short (1-3 sentence) natural-language summaries generated from
session episodes. On each turn, the most relevant clues are retrieved first
and used to augment the main retrieval query.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class MemoryClue:
    clue_id: int
    text: str
    session: int               # which session this clue was generated from
    embedding: list[float] = field(default_factory=list, repr=False)


class MemoRAGStore:
    """Stores and retrieves global campaign memory clues."""

    def __init__(self) -> None:
        self.clues: list[MemoryClue] = []

    def add_clues(
        self,
        session_episodes: list[str],
        session_number: int,
        embed_fn: Callable[[list[str]], list[list[float]]],
        summarise_fn: Callable[[str], str],
        clues_per_session: int = 5,
    ) -> None:
        """
        Generate and store memory clues from a completed session.
        Called at session end.
        """
        combined = "\n\n".join(session_episodes)
        prompt = (
            f"You are a dungeon master assistant. Generate exactly "
            f"{clues_per_session} short memory clues (1-2 sentences each) "
            f"from the following session summary that would help a dungeon "
            f"master recall important facts in future sessions. Format as a "
            f"numbered list.\n\n{combined}"
        )
        raw = summarise_fn(prompt)
        clue_texts = [
            line.split(".", 1)[-1].strip()
            for line in raw.splitlines()
            if line.strip() and line[0].isdigit()
        ][:clues_per_session]

        embeddings = embed_fn(clue_texts)
        start_id = len(self.clues)
        for i, (text, emb) in enumerate(zip(clue_texts, embeddings)):
            self.clues.append(
                MemoryClue(
                    clue_id=start_id + i,
                    text=text,
                    session=session_number,
                    embedding=emb,
                )
            )

    def retrieve_clues(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[MemoryClue]:
        if not self.clues:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        scored = []
        for clue in self.clues:
            emb = np.array(clue.embedding, dtype=np.float32)
            denom = np.linalg.norm(q) * np.linalg.norm(emb)
            sim = float(np.dot(q, emb) / denom) if denom else 0.0
            scored.append((sim, clue))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [clue for _, clue in scored[:top_k]]

    def augment_query(
        self,
        original_query: str,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> str:
        """
        Return an augmented query string: original query + relevant clues.
        The augmented query is used for the main retrieval pass.
        """
        clues = self.retrieve_clues(query_embedding, top_k=top_k)
        if not clues:
            return original_query

        clue_text = " ".join(c.text for c in clues)
        return f"{original_query} Context: {clue_text}"

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: Path) -> None:
        data = [
            {
                "clue_id": c.clue_id,
                "text": c.text,
                "session": c.session,
                "embedding": c.embedding,
            }
            for c in self.clues
        ]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "MemoRAGStore":
        store = cls()
        for item in json.loads(path.read_text()):
            store.clues.append(MemoryClue(**item))
        return store
```

#### Integrate into `MemoryManager`

```python
# src/dm/memory/manager.py
from src.dm.memory.memo_rag import MemoRAGStore

class MemoryManager:
    async def load(self, campaign_name: str):
        ...
        memo_path = memory_dir / campaign_name / "memo_rag.json"
        self._memo_path = memo_path
        self.memo_store = MemoRAGStore.load(memo_path) if memo_path.exists() else MemoRAGStore()

    async def get_context(self, query: str, group_id: str) -> str:
        q_emb = self._embed_fn([query])[0]

        # MemoRAG: augment query with global clues before main retrieval
        augmented_query = self.memo_store.augment_query(query, q_emb)

        # Graphiti search on augmented query
        graph_context = await self._graphiti_store.search(augmented_query, group_id)

        # RAPTOR search on original query embedding
        raptor_context = ""
        if self.raptor_tree:
            nodes = self.raptor_tree.search(q_emb, top_k=3)
            raptor_context = "\n".join(n.text for n in nodes)

        parts = [p for p in [raptor_context, graph_context] if p]
        return "\n\n---\n\n".join(parts)

    def end_of_session(self, session_episodes: list[str], session_number: int):
        """Call at session end to update the MemoRAG store and RAPTOR tree."""
        self.memo_store.add_clues(
            session_episodes, session_number,
            embed_fn=self._embed_fn,
            summarise_fn=self._llm_fn,
        )
        if self._memo_path:
            self.memo_store.save(self._memo_path)
```

#### New settings

```python
# src/config.py
MEMORAG_ENABLED: bool = True
MEMORAG_CLUES_PER_SESSION: int = 5
MEMORAG_TOP_K: int = 3
```

### Verification

```python
def test_clues_are_generated_per_session():
    store = MemoRAGStore()
    store.add_clues(["ep1", "ep2", "ep3"], session_number=1,
                    embed_fn=mock_embed, summarise_fn=mock_llm, clues_per_session=3)
    assert len(store.clues) == 3
    assert all(c.session == 1 for c in store.clues)

def test_augmented_query_contains_clues():
    store = MemoRAGStore()
    store.clues = [MemoryClue(clue_id=0, text="Toblen is a merchant.", session=1,
                               embedding=[0.1, 0.2, 0.3])]
    augmented = store.augment_query("who is the merchant", [0.1, 0.2, 0.3])
    assert "Toblen" in augmented

def test_memo_store_persists(tmp_path):
    store = MemoRAGStore()
    store.clues = [MemoryClue(0, "Clue.", 1, [0.1])]
    store.save(tmp_path / "memo.json")
    loaded = MemoRAGStore.load(tmp_path / "memo.json")
    assert loaded.clues[0].text == "Clue."
```

---

## 11. Other techniques to explore if necessary

The following techniques from RAG_Techniques were evaluated and determined to be
low-priority for this project's current architecture, interactive performance
constraints, and local model setup. They are documented here for reference and
may become valuable under specific conditions described below.

---

### HyDE — Hypothetical Document Embedding (technique #7)

**What it is:** Generate a hypothetical answer document from the query using the
LLM, embed the hypothetical document instead of the raw query, and use that
embedding for retrieval. The idea is that the embedding of "what a relevant
answer looks like" is closer to the actual document embedding than the embedding
of the question itself.

**Why it is lower priority here:** The player's action ("I attack the goblin
with my longsword") already carries enough semantic signal to retrieve the
correct encounter chunk. The campaign book is structured narrative, not a Q&A
corpus — the query–document embedding gap that HyDE addresses is smaller here
than in open-domain retrieval tasks.

**When to revisit:** If retrieval recall is measurably poor on vague player
inputs ("I look around"), HyDE could improve results by grounding the search
in what a relevant scene description looks like. Also relevant if the rules
RAG is implemented for very large rulesets (Pathfinder 2e) where the query–rule
semantic gap is larger.

**Implementation reference:**
`all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb`

---

### HyPE — Hypothetical Prompt Embeddings (technique #8)

**What it is:** At index construction time, generate multiple hypothetical
questions that each chunk could answer, then embed those questions (not the
chunk itself). At query time, match the player's input against the stored
question embeddings. This shifts retrieval to question–question matching.

**Why it is lower priority here:** HyPE is most valuable when the query is a
natural-language question and the document is a long answer. Campaign book
chunks are not formatted as answers — they are narration, dialogue, and
encounter descriptions. Generating hypothetical questions from narration prose
tends to produce misaligned questions. Additionally, the preprocessing cost
(one LLM call per chunk) at index build time is significant for local models.

**When to revisit:** If a structured Q&A rules index is added (e.g., a
searchable FAQ over the 5e SRD), HyPE would excel because rules are naturally
answer-shaped and the player's rules questions are naturally question-shaped.

**Implementation reference:**
`all_rag_techniques/HyPE_Hypothetical_Prompt_Embeddings.ipynb`

---

### Self-RAG (technique #30)

**What it is:** The LLM evaluates its own retrieval results during generation,
deciding whether to retrieve, whether the retrieved documents are relevant, and
whether the generated response is grounded in the retrieved context. It adds
special tokens (Retrieve, ISREL, ISSUP, ISUSE) to guide this self-evaluation.

**Why it is lower priority here:** Self-RAG requires multiple sequential LLM
calls per turn (retrieve → self-evaluate → optionally re-retrieve → generate).
On a local 8B model this significantly increases turn latency, which breaks
the conversational feel of the game. The spoiler guard and Graphiti
deduplication already handle the most critical retrieval quality problems.

**When to revisit:** If the project moves to a faster backend (external cloud
provider, larger quantised model, or GPU-accelerated inference at high token
rates), the latency penalty becomes acceptable. Self-RAG would then reduce
factual errors in rules adjudication and NPC consistency.

**Implementation reference:**
`all_rag_techniques/self_rag.ipynb`

---

### CRAG — Corrective RAG (technique #31)

**What it is:** Adds a retrieval evaluator that scores retrieved documents as
"correct", "incorrect", or "ambiguous". If documents score as incorrect or
ambiguous, CRAG triggers a web search or alternative knowledge source to
supplement retrieval before generating the response.

**Why it is lower priority here:** The project runs entirely offline by design
(no web search). The alternative of re-retrieving from the campaign index on a
low-confidence result is partially achieved by the fusion retrieval in Phase E.
A full CRAG implementation would require an additional classification LLM call
per turn.

**When to revisit:** If the project adds support for external LLM providers
(Phase 17) and web search becomes available, CRAG could be used to supplement
campaign retrieval with real-world lore for open-world campaign elements.
Also worth exploring as a rules accuracy guardrail if the LLM frequently
misapplies 5e mechanics.

**Implementation reference:**
`all_rag_techniques/crag.ipynb`

---

### Multi-modal RAG with captioning (technique #20)

**What it is:** Embed images from the campaign `images/` directory using a
vision model, generate text captions, and include those captions in the
retrieval index. Enables image-driven scene surfacing.

**Why it is lower priority here:** This is already planned as Phase 13
(DM avatar and campaign image display) using a keyword-matching approach via
`image_resolver.py`. The full multi-modal RAG approach would significantly
increase setup complexity and requires a vision-capable model (e.g., `llava`)
running alongside the DM model in Ollama.

**When to revisit:** Once Phase 13's keyword-matching approach proves
insufficient (e.g., scenes without exact filename matches), multi-modal
RAG with captioning is the natural upgrade. The `image_resolver.py`
module already provides the integration hook.

**Implementation reference:**
`all_rag_techniques/multi_model_rag_with_captioning.ipynb`

---

### Intelligent reranking (technique #16)

**What it is:** After initial retrieval, pass all candidate chunks along with
the query to a cross-encoder model or LLM-as-judge scorer. Re-score and
re-order by the cross-encoder's relevance signal rather than the original
cosine similarity or BM25 rank.

**Why it is lower priority here:** Cross-encoder reranking requires either a
dedicated reranking model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) or
an additional LLM scoring call per candidate chunk. On local hardware with 8B
parameter models, the latency cost is high relative to the marginal benefit
over the fusion retrieval in Phase E. The RRF fusion already provides a
meaningful reranking signal.

**When to revisit:** If fusion retrieval (Phase E) still surfaces irrelevant
chunks in practice — particularly for campaigns with many similar-sounding
locations or NPC names — adding a lightweight cross-encoder reranker as a
post-processing step after fusion is the right move. The
`sentence-transformers` library (already added in Phase C) includes compatible
cross-encoder models that run locally.

**Implementation reference:**
`all_rag_techniques/reranking.ipynb`

---

*End of implementation guide.*