"""
Application configuration.

Loads all environment variables from .env (via python-dotenv), applies defaults,
and validates that the selected provider's required fields are present at startup.

Usage:
    from src.config import settings
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root (two levels up from this file: src/config.py → root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root.  load_dotenv is a no-op if the file is absent.
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """All runtime configuration for Personal Dungeon Master."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM provider ─────────────────────────────────────────────────────────
    llm_provider: str = "openai"
    """Which LLM backend to use: 'openai' or 'ollama'."""

    openai_api_key: str = ""
    """Required when llm_provider='openai'."""

    ollama_base_url: str = "http://localhost:11434"
    """Base URL for the Ollama service. Required when llm_provider='ollama'."""

    dm_model: str = ""
    """
    Model name to use for the DM agent.
    Defaults to empty — when provider is 'ollama' and this is unset, an
    interactive model picker will be presented at startup.
    """

    # ── Rules ─────────────────────────────────────────────────────────────────
    game_edition: str = "5e"
    """Game edition whose rules folder to load (e.g. '5e')."""

    # ── Directories ───────────────────────────────────────────────────────────
    campaigns_dir: Path = _PROJECT_ROOT / "campaigns"
    memory_dir: Path = _PROJECT_ROOT / "memory"
    rules_dir: Path = _PROJECT_ROOT / "rules"

    # ── LLM behaviour ─────────────────────────────────────────────────────────
    dm_temperature: float = 0.8
    max_tokens: int = 1024

    # ── Derived paths (set by validators) ─────────────────────────────────────
    # Not read from env; computed from rules_dir + game_edition.
    rules_edition_dir: Path = _PROJECT_ROOT / "rules" / "5e"

    # ─────────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────────

    @field_validator("llm_provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"openai", "ollama"}:
            raise ValueError(
                f"LLM_PROVIDER must be 'openai' or 'ollama', got '{v}'."
            )
        return v

    @field_validator("campaigns_dir", "memory_dir", "rules_dir", mode="before")
    @classmethod
    def _resolve_path(cls, v: object) -> Path:
        p = Path(str(v))
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p.resolve()

    @model_validator(mode="after")
    def _validate_provider_fields(self) -> "Settings":
        """Raise an informative error if provider-specific required fields are missing."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when LLM_PROVIDER=openai. "
                "Set it in your .env file or as an environment variable."
            )
        # Derive the edition-specific rules directory.
        self.rules_edition_dir = (self.rules_dir / self.game_edition).resolve()
        return self


# Module-level singleton — import this everywhere.
settings = Settings()
