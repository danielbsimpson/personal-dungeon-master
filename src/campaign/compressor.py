"""
Contextual compression for retrieved campaign segments.

Phase F — Contextual compression.

Passes each retrieved segment through the LLM to extract only the sentences
directly relevant to the current player action.  The compressed output is
injected into the system prompt instead of the full segment, saving tokens
on models with limited context windows.

Compression is optional: set ``COMPRESSION_ENABLED=false`` in ``.env`` or
pass ``llm_fn=None`` to ``build_system_prompt`` to skip it entirely.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

_COMPRESSION_PROMPT = """\
You are a dungeon master assistant. Extract only the sentences from the passage \
below that are directly relevant to the player's current action. Return the \
extracted sentences verbatim, separated by spaces. If nothing is relevant, \
return the single word NONE.

Player action: {query}

Passage:
{passage}

Relevant sentences:"""


async def compress(
    passage: str,
    query: str,
    llm_fn: Callable[[str], Awaitable[str]],
    max_passage_tokens: int = 800,
) -> str:
    """Compress *passage* to the sentences most relevant to *query*.

    Uses the configured LLM to extract relevant sentences from the retrieved
    campaign segment, reducing the number of tokens injected into the system
    prompt.

    Args:
        passage: Retrieved campaign segment text.
        query: The player's current input — the action or question that
            triggered this retrieval.
        llm_fn: Async function that accepts a prompt string and returns the
            model's reply as a string.
        max_passage_tokens: Approximate upper bound on passage length before
            sending to the LLM.  Uses a 4-chars-per-token heuristic to avoid
            overwhelming the compressor with very long segments.

    Returns:
        Compressed passage string.  Falls back to the original *passage* if:
          - the LLM call raises an exception,
          - the model returns an empty string, or
          - the model returns ``"NONE"`` (no relevant sentences found).
    """
    # Rough token estimate: 1 token ≈ 4 characters.
    truncated = passage[: max_passage_tokens * 4]
    prompt = _COMPRESSION_PROMPT.format(query=query, passage=truncated)

    try:
        result = (await llm_fn(prompt)).strip()
    except Exception as exc:
        log.warning("Compression LLM call failed (%s); returning full passage.", exc)
        return passage

    if not result or result.upper() == "NONE":
        return passage  # fallback: full segment is better than nothing

    return result
