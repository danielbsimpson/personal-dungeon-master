"""
Ollama LLM provider.

Uses Ollama's OpenAI-compatible /v1/chat/completions endpoint via the openai
SDK, and the native Ollama REST API for model listing and context-window
discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from openai import OpenAI

from src.llm.base import LLMProvider, ModelInfo

if TYPE_CHECKING:
    from src.config import Settings

_DEFAULT_CONTEXT_WINDOW = 4_096


class OllamaProvider(LLMProvider):
    """LLM provider backed by a locally running Ollama instance."""

    def __init__(self, settings: "Settings") -> None:
        self._base_url: str = settings.ollama_base_url.rstrip("/")
        self._temperature: float = settings.dm_temperature
        self._max_tokens: int = settings.max_tokens
        self._model: str = ""
        self.context_window: int = _DEFAULT_CONTEXT_WINDOW

        # The openai SDK accepts a custom base_url so it speaks to Ollama's
        # OpenAI-compatible endpoint.  The api_key value is arbitrary —
        # Ollama requires the field to be present but ignores its value.
        self._client = OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="ollama",
        )

        self._check_reachable()

    # ── Reachability ──────────────────────────────────────────────────────────

    def _check_reachable(self) -> None:
        """Raise RuntimeError with an actionable message if Ollama is not up."""
        try:
            with httpx.Client(timeout=5.0) as client:
                client.get(f"{self._base_url}/api/tags")
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._base_url}.\n"
                "Make sure Ollama is installed and running:\n"
                "  https://ollama.com\n"
                "  ollama serve"
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama at {self._base_url} did not respond within 5 seconds.\n"
                "Check that the service is running and not overloaded."
            )

    # ── Model discovery ───────────────────────────────────────────────────────

    def list_models(self) -> list[ModelInfo]:
        """Return all models currently available in the local Ollama instance."""
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self._base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [
            ModelInfo(
                name=entry["name"],
                context_window=_DEFAULT_CONTEXT_WINDOW,
                provider="ollama",
            )
            for entry in data.get("models", [])
        ]

    # ── Model configuration ───────────────────────────────────────────────────

    def configure_model(self, model: str) -> None:
        """
        Set the active model and fetch its context window from Ollama.

        Call this once after construction (the factory handles it).
        """
        self._model = model
        self.context_window = self._fetch_context_window(model)

    def _fetch_context_window(self, model: str) -> int:
        """
        Query Ollama for the model's native context length.

        Tries model_info first (Ollama ≥ 0.3), then falls back to the
        ``num_ctx`` entry in the parameters string, then to the default.
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f"{self._base_url}/api/show",
                    json={"name": model},
                )
            response.raise_for_status()
            data = response.json()

            # Ollama ≥ 0.3 exposes a model_info dict.  Keys vary by
            # architecture (e.g. "llama.context_length", "phi3.context_length")
            # so we match any key ending with the suffix.
            for key, value in data.get("model_info", {}).items():
                if key.endswith("context_length") and isinstance(value, int):
                    return value

            # Older versions surface context size in the parameters string.
            for line in data.get("parameters", "").splitlines():
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == "num_ctx":
                    return int(parts[1])

        except Exception:
            pass  # Non-fatal — fall through to the safe default.

        return _DEFAULT_CONTEXT_WINDOW

    # ── Inference ─────────────────────────────────────────────────────────────

    def complete(self, messages: list[dict], **kwargs) -> str:
        """Send chat messages to Ollama and return the reply text."""
        if not self._model:
            raise RuntimeError(
                "No model configured on OllamaProvider. "
                "Call configure_model() before complete()."
            )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        return response.choices[0].message.content or ""
