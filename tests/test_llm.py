"""Tests for the LLM provider layer (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.llm.factory import create_provider
from src.llm.ollama_provider import OllamaProvider


# ---------------------------------------------------------------------------
# Minimal settings stand-in for tests — avoids loading .env or triggering
# the pydantic-settings singleton in src.config.
# ---------------------------------------------------------------------------

@dataclass
class _MockSettings:
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    dm_model: str = "llama3"
    dm_temperature: float = 0.8
    max_tokens: int = 1024


# ---------------------------------------------------------------------------
# Helper: construct OllamaProvider with _check_reachable bypassed.
# ---------------------------------------------------------------------------

def _make_provider(settings: _MockSettings | None = None) -> OllamaProvider:
    """Create an OllamaProvider without actually connecting to Ollama."""
    if settings is None:
        settings = _MockSettings()
    with patch.object(OllamaProvider, "_check_reachable"), \
         patch("src.llm.ollama_provider.OpenAI"):
        return OllamaProvider(settings)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_returns_ollama_provider(self) -> None:
        """create_provider returns an OllamaProvider when LLM_PROVIDER=ollama."""
        settings = _MockSettings(dm_model="llama3")

        with patch("src.llm.factory.OllamaProvider") as MockProvider:
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance

            provider = create_provider(settings)

            MockProvider.assert_called_once_with(settings)
            mock_instance.configure_model.assert_called_once_with("llama3")
            assert provider is mock_instance

    def test_raises_for_unsupported_provider(self) -> None:
        """create_provider raises NotImplementedError for any non-ollama value."""
        settings = _MockSettings(llm_provider="openai")

        with pytest.raises(NotImplementedError, match="not yet supported"):
            create_provider(settings)


class TestOllamaProviderListModels:
    def test_parses_tags_response(self) -> None:
        """list_models correctly parses the /api/tags JSON response."""
        provider = _make_provider()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "mistral:7b"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with patch("src.llm.ollama_provider.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = lambda _self: mock_client
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            models = provider.list_models()

        assert len(models) == 2
        assert models[0].name == "llama3:latest"
        assert models[1].name == "mistral:7b"
        assert all(m.provider == "ollama" for m in models)

    def test_returns_empty_list_when_no_models(self) -> None:
        """list_models returns an empty list when Ollama has no pulled models."""
        provider = _make_provider()

        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with patch("src.llm.ollama_provider.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = lambda _self: mock_client
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            assert provider.list_models() == []


class TestOllamaProviderReachability:
    def test_missing_ollama_raises_runtime_error(self) -> None:
        """OllamaProvider.__init__ raises RuntimeError when Ollama is not running."""
        settings = _MockSettings()

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        with patch("src.llm.ollama_provider.OpenAI"), \
             patch("src.llm.ollama_provider.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = lambda _self: mock_client
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
                OllamaProvider(settings)

    def test_timeout_raises_runtime_error(self) -> None:
        """OllamaProvider.__init__ raises RuntimeError when Ollama times out."""
        settings = _MockSettings()

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("Timeout")

        with patch("src.llm.ollama_provider.OpenAI"), \
             patch("src.llm.ollama_provider.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = lambda _self: mock_client
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(RuntimeError, match="did not respond within"):
                OllamaProvider(settings)


class TestOllamaProviderStream:
    def test_stream_yields_multiple_chunks(self) -> None:
        """stream() yields incremental content from each SSE chunk."""
        provider = _make_provider()
        provider._model = "llama3"

        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = ", world"
        chunk3 = MagicMock()
        chunk3.choices[0].delta.content = "!"
        # Chunks with no content (None) must be silently skipped.
        empty_chunk = MagicMock()
        empty_chunk.choices[0].delta.content = None

        provider._client.chat.completions.create.return_value = iter(
            [chunk1, empty_chunk, chunk2, chunk3]
        )

        result = list(provider.stream([{"role": "user", "content": "hi"}]))

        assert result == ["Hello", ", world", "!"]

    def test_stream_uses_stream_equals_true(self) -> None:
        """stream() passes stream=True to the OpenAI client."""
        provider = _make_provider()
        provider._model = "llama3"
        provider._client.chat.completions.create.return_value = iter([])

        list(provider.stream([{"role": "user", "content": "hi"}]))

        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_stream_raises_when_model_not_configured(self) -> None:
        """stream() raises RuntimeError if no model has been configured."""
        provider = _make_provider()
        # _model is empty string by default

        with pytest.raises(RuntimeError, match="No model configured"):
            list(provider.stream([]))


class TestLLMProviderStreamDefault:
    def test_default_stream_delegates_to_complete(self) -> None:
        """The default stream() implementation yields complete() as one chunk."""
        from src.llm.base import LLMProvider

        class _ConcreteProvider(LLMProvider):
            def complete(self, messages: list[dict], **kwargs) -> str:
                return "full response"

        provider = _ConcreteProvider()
        chunks = list(provider.stream([]))
        assert chunks == ["full response"]
