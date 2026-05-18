"""
LLM provider factory.

Reads LLM_PROVIDER from settings and constructs the appropriate LLMProvider
implementation.  Only llama.cpp is supported until Phase 13 adds OpenAI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

from src.llm.base import LLMProvider, ModelInfo
from src.llm.llamacpp_provider import LlamaCppProvider

if TYPE_CHECKING:
    from src.config import Settings

_console = Console()


def create_provider(settings: "Settings") -> LLMProvider:
    """
    Construct and return the correct LLMProvider for the current settings.

    Raises:
        NotImplementedError: If LLM_PROVIDER is anything other than 'llamacpp'.
        RuntimeError: If the server is unreachable or no model can be determined.
    """
    if settings.llm_provider != "llamacpp":
        raise NotImplementedError(
            f"LLM provider '{settings.llm_provider}' is not yet supported.\n"
            "Only 'llamacpp' is available at this stage.\n"
            "OpenAI support will be added in Phase 13.\n"
            "Set LLM_PROVIDER=llamacpp in your .env file."
        )

    provider = LlamaCppProvider(settings)

    model = settings.dm_model or _resolve_model(provider.list_models())
    provider.configure_model(model)

    return provider


def _resolve_model(models: list[ModelInfo]) -> str:
    """
    Return the model to use from the server's loaded model list.

    If exactly one model is loaded (the typical llama.cpp single-server case)
    it is selected automatically.  When multiple models are available (router
    mode) the user is prompted to choose.
    """
    if not models:
        raise RuntimeError(
            "No models found on the llama.cpp server.\n"
            "Start the server with a model, e.g.:\n"
            "  llama-server -m path/to/model.gguf"
        )

    if len(models) == 1:
        _console.print(
            f"[bold cyan]Using model:[/bold cyan] {models[0].name}"
        )
        return models[0].name

    return _pick_model_interactively(models)


def _pick_model_interactively(models: list[ModelInfo]) -> str:
    """Display a numbered model list and return the user's chosen model name."""
    table = Table(show_header=True, header_style="bold cyan", show_edge=False)
    table.add_column("#", style="cyan", justify="right", width=4)
    table.add_column("Model", style="white")

    for i, m in enumerate(models, 1):
        table.add_row(str(i), m.name)

    _console.print()
    _console.print("[bold cyan]Available llama.cpp models[/bold cyan]")
    _console.print(table)

    choice = IntPrompt.ask(
        "Select a model",
        choices=[str(i) for i in range(1, len(models) + 1)],
    )
    return models[choice - 1].name
