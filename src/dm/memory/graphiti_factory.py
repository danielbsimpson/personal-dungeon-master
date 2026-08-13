"""
Graphiti factory — builds the LLM client and embedder for Graphiti,
both pointed at the local llama.cpp server's OpenAI-compatible endpoint.

Graphiti requires:
  - An LLM client for entity/relationship extraction (structured output)
  - An embedder for vector similarity search

Both are implemented here using ``OpenAIGenericClient`` and
``OpenAIEmbedder`` from graphiti-core, configured to call the llama.cpp
server instead of the real OpenAI API.

Recommended llama.cpp models for entity extraction (need strong tool-calling):
    llama3.1:8b, qwen2.5:7b, mistral-nemo

For embeddings, start a separate llama.cpp embedding server with:
    llama-server -m path/to/embedding-model.gguf --embedding

Recommended embedding model:
    nomic-embed-text (768-dim)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

if TYPE_CHECKING:
    from src.config import Settings


def build_graphiti_clients(
    settings: "Settings",
) -> tuple[OpenAIGenericClient, OpenAIEmbedder]:
    """
    Construct the Graphiti LLM client and embedder, both using llama.cpp.

    Parameters
    ----------
    settings:
        The application ``Settings`` singleton.

    Returns
    -------
    tuple[OpenAIGenericClient, OpenAIEmbedder]
        ``(llm_client, embedder)`` ready to pass into ``GraphitiStore``.
    """
    base_url = f"{settings.llamacpp_base_url}/v1"

    llm_config = LLMConfig(
        api_key="no-key-required",
        model=settings.dm_model or "llama3.1:8b",
        base_url=base_url,
        temperature=settings.dm_temperature,
    )
    llm_client = OpenAIGenericClient(config=llm_config)

    embedder_config = OpenAIEmbedderConfig(
        api_key="no-key-required",
        base_url=base_url,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    return llm_client, embedder
