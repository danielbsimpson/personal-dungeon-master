"""
llama.cpp LLM provider.

Uses the llama.cpp server's OpenAI-compatible /v1/chat/completions endpoint
via the openai SDK.  The server's native REST API is used for reachability
checks and context-window discovery.

Start the server with:
    llama-server -m path/to/model.gguf -c 4096
    https://github.com/ggml-org/llama.cpp
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

import httpx
from openai import OpenAI

from src.llm.base import LLMProvider, ModelInfo

if TYPE_CHECKING:
    from src.config import Settings

log = logging.getLogger(__name__)

_DEFAULT_CONTEXT_WINDOW = 4_096


class LlamaCppProvider(LLMProvider):
    """LLM provider backed by a locally running llama.cpp server."""

    def __init__(self, settings: "Settings") -> None:
        self._base_url: str = settings.llamacpp_base_url.rstrip("/")
        self._temperature: float = settings.dm_temperature
        self._max_tokens: int = settings.max_tokens
        self._model: str = ""
        self.context_window: int = _DEFAULT_CONTEXT_WINDOW

        # The openai SDK accepts a custom base_url so it speaks to llama.cpp's
        # OpenAI-compatible endpoint.  The api_key value is arbitrary —
        # llama.cpp requires the field to be present but ignores its value
        # unless --api-key is configured on the server.
        self._client = OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="no-key-required",
        )

        self._check_reachable()

    # ── Reachability ──────────────────────────────────────────────────────────

    def _check_reachable(self) -> None:
        """Raise RuntimeError with an actionable message if the server is not up."""
        try:
            with httpx.Client(timeout=5.0) as client:
                client.get(f"{self._base_url}/health")
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot reach llama.cpp server at {self._base_url}.\n"
                "Make sure llama-server is installed and running:\n"
                "  https://github.com/ggml-org/llama.cpp\n"
                "  llama-server -m path/to/model.gguf"
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"llama.cpp server at {self._base_url} did not respond within 5 seconds.\n"
                "Check that the service is running and not overloaded."
            )

    # ── Model discovery ───────────────────────────────────────────────────────

    def list_models(self) -> list[ModelInfo]:
        """Return models currently loaded in the llama.cpp server."""
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self._base_url}/v1/models")
        response.raise_for_status()
        data = response.json()
        return [
            ModelInfo(
                name=entry["id"],
                context_window=entry.get("meta", {}).get(
                    "n_ctx_train", _DEFAULT_CONTEXT_WINDOW
                ),
                provider="llamacpp",
            )
            for entry in data.get("data", [])
        ]

    # ── Model configuration ───────────────────────────────────────────────────

    def configure_model(self, model: str) -> None:
        """
        Set the active model and discover its context window from the server.

        Call this once after construction (the factory handles it).
        """
        self._model = model
        self.context_window = self._fetch_context_window()
        log.info(
            "LlamaCppProvider configured: model='%s' context_window=%d",
            model,
            self.context_window,
        )

    def _fetch_context_window(self) -> int:
        """
        Query the server for the active context size.

        Tries GET /props first — this reflects the value configured with ``-c``
        at server start.  Falls back to the ``n_ctx_train`` value from
        GET /v1/models (the model's maximum training context), then to the
        safe default.
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self._base_url}/props")
            response.raise_for_status()
            data = response.json()
            n_ctx = data.get("default_generation_settings", {}).get("n_ctx")
            if isinstance(n_ctx, int) and n_ctx > 0:
                return n_ctx
        except Exception:
            pass

        try:
            models = self.list_models()
            if models:
                return models[0].context_window
        except Exception:
            pass

        return _DEFAULT_CONTEXT_WINDOW

    # ── Inference ─────────────────────────────────────────────────────────────

    def complete(self, messages: list[dict], **kwargs) -> str:
        """Send chat messages to the llama.cpp server and return the reply text."""
        if not self._model:
            raise RuntimeError(
                "No model configured on LlamaCppProvider. "
                "Call configure_model() before complete()."
            )
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        log.debug(
            "LlamaCppProvider.complete: model=%s messages=%d temperature=%s max_tokens=%d",
            self._model,
            len(messages),
            temperature,
            max_tokens,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reply = response.choices[0].message.content or ""
        log.debug(
            "LlamaCppProvider.complete: reply length=%d chars", len(reply)
        )
        return reply

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """Stream tokens from the llama.cpp server using the OpenAI-compatible API."""
        if not self._model:
            raise RuntimeError(
                "No model configured on LlamaCppProvider. "
                "Call configure_model() before stream()."
            )
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        log.debug(
            "LlamaCppProvider.stream: model=%s messages=%d temperature=%s max_tokens=%d",
            self._model,
            len(messages),
            temperature,
            max_tokens,
        )
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
