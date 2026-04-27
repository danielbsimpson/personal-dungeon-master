"""
Personal Dungeon Master — entry point.

Wires together the full pipeline:

1. Load campaigns from ``campaigns/``
2. Let the player select a campaign interactively (or use ``--campaign``)
3. Load the D&D rules for the configured edition (or use ``--edition``)
4. Create the LLM provider (verifies Ollama connectivity, picks a model
   interactively if ``DM_MODEL`` is not set in ``.env``)
5. Initialise and load :class:`~src.dm.memory.manager.MemoryManager`
6. Build the :class:`~src.dm.dungeon_master.DungeonMaster`
7. Hand control to :func:`~src.interface.cli.run_session`

Run with::

    python -m src.main
    python -m src.main --campaign example-campaign --model llama3.1:8b
    # or, after a ``pip install -e .`` that wires the console_scripts entry point:
    pdm
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from src.campaign.loader import load_campaigns
from src.campaign.parser import parse_campaign
from src.campaign.selector import select_campaign
from src.config import settings
from src.dm.dungeon_master import DungeonMaster
from src.dm.memory.manager import MemoryManager
from src.interface.cli import print_banner, print_campaign_header, run_session
from src.llm.factory import create_provider
from src.rules.loader import load_rules

console = Console()
app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging() -> None:
    """
    Configure file-based logging to ``logs/``.

    - ``logs/session_<date>.log`` — INFO and above (session events, warnings)
    - ``logs/debug_<date>.log``   — DEBUG and above (LLM calls, token counts)

    Console output is kept at WARNING so debug noise does not clutter the
    terminal during normal play.
    """
    logs_dir = _PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Session log — INFO+
    session_handler = logging.FileHandler(
        logs_dir / f"session_{date_str}.log", encoding="utf-8"
    )
    session_handler.setLevel(logging.INFO)
    session_handler.setFormatter(fmt)
    root.addHandler(session_handler)

    # Debug log — DEBUG+
    debug_handler = logging.FileHandler(
        logs_dir / f"debug_{date_str}.log", encoding="utf-8"
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(fmt)
    root.addHandler(debug_handler)

    # Suppress noisy third-party loggers at WARNING in logs
    for noisy in ("httpx", "httpcore", "openai", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Core async session logic
# ---------------------------------------------------------------------------


async def _run_session(
    campaign_name: Optional[str],
    edition: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    reset: bool,
) -> None:
    """Bootstrap and run a DM session with the given CLI overrides."""
    log = logging.getLogger(__name__)

    # Apply CLI overrides to settings before anything else uses them.
    if edition:
        settings.game_edition = edition
        settings.rules_edition_dir = (settings.rules_dir / edition).resolve()
        log.info("Edition overridden via CLI: %s", edition)
    if provider:
        settings.llm_provider = provider
        log.info("Provider overridden via CLI: %s", provider)
    if model:
        settings.dm_model = model
        log.info("Model overridden via CLI: %s", model)

    print_banner()

    # ------------------------------------------------------------------
    # 1. Load available campaigns
    # ------------------------------------------------------------------
    try:
        campaigns = load_campaigns(settings.campaigns_dir)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Error loading campaigns:[/bold red] {exc}")
        sys.exit(1)

    if not campaigns:
        console.print(
            "[bold red]No campaigns found in "
            f"'{settings.campaigns_dir}'.[/bold red]\n"
            "Add a campaign folder with a .txt story file to get started."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Select campaign (skip menu if --campaign was given)
    # ------------------------------------------------------------------
    if campaign_name:
        matches = [c for c in campaigns if c.name.lower() == campaign_name.lower()]
        if not matches:
            available = ", ".join(c.name for c in campaigns)
            console.print(
                f"[bold red]Campaign '[bold]{campaign_name}[/bold]' not found.[/bold red]\n"
                f"Available campaigns: {available}"
            )
            sys.exit(1)
        selected = matches[0]
        log.info("Campaign selected via CLI: %s", selected.name)
    else:
        selected = select_campaign(campaigns)

    log.info("Starting session: campaign='%s'", selected.name)

    # ------------------------------------------------------------------
    # 3. Parse campaign content
    # ------------------------------------------------------------------
    try:
        parsed = parse_campaign(selected)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Failed to parse campaign:[/bold red] {exc}")
        log.exception("Campaign parse error for '%s'", selected.name)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Load rules
    # ------------------------------------------------------------------
    try:
        rules = load_rules(settings)
    except FileNotFoundError as exc:
        console.print(
            f"[bold red]Rules directory not found:[/bold red] {exc}\n"
            f"Make sure 'rules/{settings.game_edition}/' exists."
        )
        sys.exit(1)
    except ValueError as exc:
        console.print(f"[bold red]Rules error:[/bold red] {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Create LLM provider
    # ------------------------------------------------------------------
    try:
        llm = create_provider(settings)
    except NotImplementedError as exc:
        console.print(f"[bold red]Provider not supported:[/bold red] {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        console.print(f"[bold red]LLM provider error:[/bold red] {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Initialise memory
    # ------------------------------------------------------------------
    memory = MemoryManager(memory_dir=settings.memory_dir, settings=settings)
    try:
        await memory.load(selected.name)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Memory initialisation failed:[/bold red] {exc}")
        log.exception("Memory init error for campaign '%s'", selected.name)
        sys.exit(1)

    # Apply --reset before the session starts.
    if reset:
        memory.reset_session()
        console.print("[yellow]Session window cleared (--reset).[/yellow]")
        log.info("Session window reset via CLI flag.")

    # ------------------------------------------------------------------
    # 7. Build DM and start the session
    # ------------------------------------------------------------------
    dm = DungeonMaster(
        llm=llm,
        campaign=parsed,
        rules=rules,
        memory=memory,
        settings=settings,
    )

    print_campaign_header(parsed)
    log.info(
        "Session ready: model='%s' context_window=%d",
        settings.dm_model,
        llm.context_window,
    )
    await run_session(dm, memory, rules)


# ---------------------------------------------------------------------------
# CLI entry point (typer)
# ---------------------------------------------------------------------------


@app.command()
def run(
    campaign: Optional[str] = typer.Option(
        None,
        "--campaign",
        "-c",
        help="Skip the selection menu and load a named campaign directly.",
        metavar="NAME",
    ),
    edition: Optional[str] = typer.Option(
        None,
        "--edition",
        "-e",
        help="Override GAME_EDITION (e.g. '5e').",
        metavar="EDITION",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Override LLM_PROVIDER (e.g. 'ollama').",
        metavar="PROVIDER",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override DM_MODEL (e.g. 'llama3.1:8b').",
        metavar="MODEL",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Wipe the session window before starting (graph is preserved).",
    ),
) -> None:
    """Personal Dungeon Master — locally-run AI tabletop RPG."""
    setup_logging()
    try:
        asyncio.run(_run_session(campaign, edition, provider, model, reset))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app()
