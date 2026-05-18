"""
Semantic chunker for campaign book text.

Splits the campaign book into topic-coherent chunks by measuring cosine
similarity between adjacent sentence embeddings and inserting a boundary
wherever similarity falls below a configurable threshold.

The embed_fn is async to match the existing Ollama/Graphiti embedder interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np


@dataclass
class CampaignChunk:
    index: int
    text: str
    scene_header: str   # nearest preceding scene/chapter header
    act: str            # top-level act or chapter label if present
    start_char: int     # character offset in original document
    end_char: int


def _sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex heuristic."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


async def semantic_chunk(
    text: str,
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
    breakpoint_threshold: float = 0.75,
    min_chunk_sentences: int = 3,
    max_chunk_sentences: int = 40,
) -> list[CampaignChunk]:
    """
    Segment *text* into semantically coherent chunks.

    Args:
        text: Full campaign book text.
        embed_fn: Async function that accepts a list of strings and returns a
            list of float vectors (same length). Uses the Ollama embedder
            already configured for Graphiti.
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

    # Embed all sentences in one batch
    raw_vectors = await embed_fn(sentences)
    vectors = np.array(raw_vectors, dtype=np.float32)

    # Identify candidate breakpoints
    boundaries: list[int] = [0]
    consecutive = 0

    for i in range(1, len(sentences)):
        sim = _cosine(vectors[i - 1], vectors[i])
        consecutive += 1

        force_split = consecutive >= max_chunk_sentences
        semantic_split = sim < breakpoint_threshold and consecutive >= min_chunk_sentences

        if force_split or semantic_split:
            boundaries.append(i)
            consecutive = 0

    boundaries.append(len(sentences))

    # Regex for act/scene headers
    header_re = re.compile(
        r"^(#{1,3}|SCENE:|ACT:|CHAPTER:)\s*(.+)$", re.IGNORECASE | re.MULTILINE
    )

    # Build CampaignChunk objects
    chunks: list[CampaignChunk] = []
    current_act = "Introduction"
    current_scene = "Prologue"
    char_offset = 0

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
        if start_char == -1:
            start_char = char_offset

        last_sentence = sentences[end_sent - 1] if end_sent <= len(sentences) else sentences[-1]
        end_char_pos = text.find(last_sentence, start_char)
        end_char = (end_char_pos + len(last_sentence)) if end_char_pos != -1 else len(text)
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
