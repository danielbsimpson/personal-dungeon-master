"""
Personal Dungeon Master — entry point.

Wires together the full pipeline:

1. Load campaigns from ``campaigns/``
2. Let the player select a campaign interactively
3. Load the D&D rules for the configured edition
4. Create the LLM provider (verifies Ollama connectivity, picks a model
   interactively if ``DM_MODEL`` is not set in ``.env``)
5. Initialise and load :class:`~src.dm.memory.manager.MemoryManager`
6. Build the :class:`~src.dm.dungeon_master.DungeonMaster`
7. Hand control to :func:`~src.interface.cli.run_session`

Run with::

    python -m src.main
    # or, after a ``pip install -e .`` that wires the console_scripts entry point:
    pdm
"""

from __future__ import annotations

import asyncio
import sys

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


async def main() -> None:
    """Bootstrap and run a DM session."""
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
    # 2. Select campaign
    # ------------------------------------------------------------------
    selected = select_campaign(campaigns)

    # ------------------------------------------------------------------
    # 3. Parse campaign content
    # ------------------------------------------------------------------
    try:
        parsed = parse_campaign(selected)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Failed to parse campaign:[/bold red] {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Load rules
    # ------------------------------------------------------------------
    try:
        rules = load_rules(settings)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Failed to load rules:[/bold red] {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Create LLM provider
    # ------------------------------------------------------------------
    try:
        llm = create_provider(settings)
    except (NotImplementedError, RuntimeError) as exc:
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
        sys.exit(1)

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
    await run_session(dm, memory)


def run() -> None:
    """Synchronous entry point (used by console_scripts in pyproject.toml)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
