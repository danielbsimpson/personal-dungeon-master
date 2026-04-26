"""
Campaign selector — displays a numbered list of available campaigns and returns
the user's selection.
"""

from __future__ import annotations

from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

from src.campaign.loader import Campaign

_console = Console()


def select_campaign(campaigns: list[Campaign]) -> Campaign:
    """
    Display a numbered table of *campaigns* using Rich and return the selected one.

    Re-prompts on invalid input until a valid choice is made.
    """
    if not campaigns:
        raise ValueError("No campaigns to select from.")

    table = Table(title="Available Campaigns", show_header=True, header_style="bold cyan")
    table.add_column("#", style="bold", width=4, justify="right")
    table.add_column("Campaign Name", style="bold")
    table.add_column("Path", style="dim")

    for i, campaign in enumerate(campaigns, 1):
        table.add_row(str(i), campaign.name, str(campaign.path))

    _console.print()
    _console.print(table)
    _console.print()

    while True:
        choice = IntPrompt.ask(
            "[bold cyan]Select a campaign[/bold cyan]",
            default=1,
        )
        if 1 <= choice <= len(campaigns):
            return campaigns[choice - 1]
        _console.print(
            f"[red]Please enter a number between 1 and {len(campaigns)}.[/red]"
        )
