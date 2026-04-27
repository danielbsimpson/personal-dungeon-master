"""
Abstract base for all LLM providers.

New providers (e.g. OpenAI in Phase 13) must subclass LLMProvider and
implement complete().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Metadata about a single model available on a provider."""

    name: str
    context_window: int
    provider: str


class LLMProvider(ABC):
    """Common interface for every LLM backend."""

    context_window: int = 4_096
    """
    The maximum number of tokens this model supports in a single request
    (input + output combined).  Subclasses should update this after the model
    is configured (e.g. by querying the provider API).  Used by the DM agent
    to set an accurate system-prompt token budget.
    """

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        """
        Send a list of chat messages and return the assistant's reply text.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts in
                OpenAI chat format.
            **kwargs: Optional overrides (e.g. ``temperature``, ``max_tokens``).

        Returns:
            The assistant's reply as a plain string.
        """
        ...
