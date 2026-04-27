"""
Text-based CLI for Personal Dungeon Master (Phase 8).

Handles all terminal I/O: welcome banner, campaign display, DM narration
rendering, dice panels, special commands, and graceful exit.

Special commands
----------------
/help           List available commands
/quit, /exit    Save and exit gracefully
/journal        Render a human-readable view of the knowledge graph
/status         Display current character stats and inventory
/save           Explicitly trigger a session save
/reset          Wipe the short-term session window (graph preserved)
/fullreset      Wipe session window AND knowledge graph
/roll <expr>    Roll dice (e.g. /roll d20+3, /roll 2d6)
/graph <entity> Look up a named entity in the knowledge graph
"""

from __future__ import annotations

import asyncio
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.campaign.parser import Character, ParsedCampaign
from src.dice.die import RollResult
from src.dice.roller import format_result, parse_player_expression, roll
from src.dm.dungeon_master import DungeonMaster
from src.dm.memory.manager import MemoryManager
from src.rules.loader import RulesReference
from src.rules.reference import search_rules

console = Console()

_BANNER = r"""
 ____                                     _   ____  __  __
|  _ \ ___ _ __ ___  ___  _ __   __ _  | | |  _ \|  \/  |
| |_) / _ \ '__/ __|/ _ \| '_ \ / _` | | | | | | | |\/| |
|  __/  __/ |  \__ \ (_) | | | | (_| | | | | |_| | |  | |
|_|   \___|_|  |___/\___/|_| |_|\__,_| |_| |____/|_|  |_|
"""

_COMMANDS: dict[str, str] = {
    "/help": "List available commands",
    "/quit | /exit": "Save and exit gracefully",
    "/journal": "Show knowledge graph entities and relationships",
    "/status": "Show character stats and inventory",
    "/save": "Explicitly save the session state",
    "/reset": "Wipe the session window (graph preserved)",
    "/fullreset": "Wipe session window AND knowledge graph (restart campaign)",
    "/roll <expr>": "Roll dice  e.g. /roll d20+3  /roll 2d6",
    "/graph <entity>": "Look up an entity in the knowledge graph",
    "/rules <topic>": "Look up a rules topic  e.g. /rules grapple  /rules concentration",
}


# ---------------------------------------------------------------------------
# Banner & startup
# ---------------------------------------------------------------------------


def print_banner() -> None:
    """Print the styled welcome banner."""
    console.print(Text(_BANNER, style="bold dark_orange"), justify="center")
    console.print(
        Rule(style="dim orange3"),
    )


def print_campaign_header(campaign: ParsedCampaign) -> None:
    """Print campaign name and summary before the adventure starts."""
    console.print()
    panel = Panel(
        Markdown(campaign.summary),
        title=f"[bold gold1]Campaign[/bold gold1]",
        border_style="gold1",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


# ---------------------------------------------------------------------------
# DM narration rendering
# ---------------------------------------------------------------------------


def print_dm_response(text: str) -> None:
    """Render the DM's narration as a rich Markdown panel."""
    console.print()
    console.print(
        Panel(
            Markdown(text),
            title="[bold green]Dungeon Master[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# Dice panels
# ---------------------------------------------------------------------------


def print_roll_results(results: list[RollResult]) -> None:
    """
    Render a styled dice panel for every roll result.

    Shown between the player's input and the DM's narration.
    """
    if not results:
        return

    console.print()
    for result in results:
        die_badge = f"[bold white on dark_red] d{result.die.value} [/bold white on dark_red]"
        rolls_str = ", ".join(str(r) for r in result.rolls)
        mod_str = ""
        if result.modifier > 0:
            mod_str = f"  +{result.modifier}"
        elif result.modifier < 0:
            mod_str = f"  {result.modifier}"

        body = (
            f"{die_badge}  "
            f"[dim]Rolls:[/dim] [cyan]{rolls_str}[/cyan]{mod_str}  "
            f"[dim]Total:[/dim] [bold white]{result.total}[/bold white]"
        )
        console.print(
            Panel(
                body,
                title=f"[bold yellow]🎲 {result.label.title()}[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
                expand=False,
            )
        )


# ---------------------------------------------------------------------------
# Special commands
# ---------------------------------------------------------------------------


def _cmd_help() -> None:
    table = Table(
        title="Available Commands",
        box=box.ROUNDED,
        border_style="dim",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("Description", style="white")
    for cmd, desc in _COMMANDS.items():
        table.add_row(cmd, desc)
    console.print()
    console.print(table)
    console.print()


def _cmd_status(character: Character) -> None:
    # Ability scores table
    scores = character.ability_scores
    score_table = Table(box=box.MINIMAL_DOUBLE_HEAD, show_header=True, header_style="bold")
    for attr in ("STR", "DEX", "CON", "INT", "WIS", "CHA"):
        score_table.add_column(attr, justify="center", style="cyan", width=6)
    score_table.add_row(
        str(scores.strength),
        str(scores.dexterity),
        str(scores.constitution),
        str(scores.intelligence),
        str(scores.wisdom),
        str(scores.charisma),
    )

    # Main stats
    stats_text = (
        f"[bold]{character.name}[/bold]  "
        f"[dim]{character.race} {character.character_class}"
        + (f" ({character.subclass})" if character.subclass else "")
        + f" — Level {character.level}[/dim]\n\n"
        f"[green]HP[/green] {character.hit_points}  "
        f"[blue]AC[/blue] {character.armor_class}  "
        f"[yellow]Speed[/yellow] {character.speed} ft  "
        f"[magenta]Proficiency[/magenta] +{character.proficiency_bonus}  "
        f"[dim]Passive Perception[/dim] {character.passive_perception}"
    )

    equipment_lines = []
    for item in character.equipment[:10]:
        name = item.get("item") or item.get("name") or next(iter(item.values()), "?")
        equipment_lines.append(f"• {name}")
    equip_text = "\n".join(equipment_lines) if equipment_lines else "[dim]None[/dim]"

    console.print()
    console.print(
        Panel(
            stats_text,
            title="[bold cyan]Character Status[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print(score_table)
    console.print()
    console.print(
        Panel(
            equip_text,
            title="[bold cyan]Equipment[/bold cyan]",
            border_style="cyan",
            expand=False,
            padding=(0, 2),
        )
    )
    console.print()


async def _cmd_journal(memory: MemoryManager) -> None:
    entities = await memory.get_entities()
    if not entities:
        console.print("[dim]No entities in the knowledge graph yet.[/dim]")
        return

    table = Table(
        title="Knowledge Graph — Entities",
        box=box.ROUNDED,
        border_style="magenta",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Name", style="bold white", no_wrap=True)
    table.add_column("Summary", style="dim white", overflow="fold")
    for entity in entities:
        table.add_row(entity["name"], entity["summary"] or "[dim]—[/dim]")
    console.print()
    console.print(table)
    console.print()


async def _cmd_graph(memory: MemoryManager, entity_name: str) -> None:
    if not entity_name.strip():
        console.print("[red]Usage: /graph <entity name>[/red]")
        return

    results = await memory.search_entity(entity_name)
    if not results:
        console.print(f"[dim]No facts found for '[bold]{entity_name}[/bold]'.[/dim]")
        return

    console.print()
    console.print(
        Panel(
            "\n".join(f"• {r['fact']}" for r in results),
            title=f"[bold magenta]Graph: {entity_name}[/bold magenta]",
            border_style="magenta",
            padding=(0, 2),
        )
    )
    console.print()


def _cmd_roll(expr: str) -> None:
    if not expr.strip():
        console.print("[red]Usage: /roll <expression>  e.g.  /roll d20+3  /roll 2d6[/red]")
        return

    req = parse_player_expression(expr.strip(), label="player roll")
    if req is None:
        console.print(
            f"[red]Unknown dice expression '[bold]{expr}[/bold]'. "
            "Try formats like d20, 2d6, d20+3, d8-1.[/red]"
        )
        return

    result = roll(req)
    print_roll_results([result])


def _cmd_rules(rules: RulesReference, topic: str) -> None:
    if not topic.strip():
        console.print(
            "[red]Usage: /rules <topic>  e.g.  /rules grapple  /rules concentration[/red]"
        )
        return

    result = search_rules(rules, topic.strip())
    console.print()
    console.print(
        Panel(
            Markdown(result),
            title=f"[bold cyan]Rules: {topic.title()}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def _confirm(prompt: str) -> bool:
    return Confirm.ask(f"[bold yellow]{prompt}[/bold yellow]", default=False)


# ---------------------------------------------------------------------------
# Main session loop
# ---------------------------------------------------------------------------


async def run_session(
    dm: DungeonMaster,
    memory: MemoryManager,
    rules: Optional[RulesReference] = None,
) -> None:
    """
    Run the interactive DM session until the player quits.

    Parameters
    ----------
    dm:
        Fully initialised :class:`~src.dm.dungeon_master.DungeonMaster`.
    memory:
        Loaded :class:`~src.dm.memory.manager.MemoryManager` (same instance
        used by the DM).
    rules:
        Loaded :class:`~src.rules.loader.RulesReference` — used by the
        ``/rules`` command.  Falls back to ``dm.rules`` when not supplied.
    """
    rules = rules or dm.rules
    character = dm.campaign.character

    # Opening narration
    console.print()
    console.print(Rule("[dim]The adventure begins…[/dim]", style="dim green"))
    with console.status("[green]The DM is setting the scene…[/green]", spinner="dots"):
        opening = await dm.start_campaign()
    print_roll_results(dm.last_roll_results)
    print_dm_response(opening)

    try:
        while True:
            console.print()
            raw = Prompt.ask("[bold cyan]You[/bold cyan]").strip()

            if not raw:
                continue

            # --- Special commands -----------------------------------------
            lower = raw.lower()

            if lower in ("/quit", "/exit"):
                console.print("[dim]Saving and exiting…[/dim]")
                break

            if lower == "/help":
                _cmd_help()
                continue

            if lower == "/status":
                _cmd_status(character)
                continue

            if lower == "/journal":
                with console.status("[magenta]Querying knowledge graph…[/magenta]", spinner="dots"):
                    await _cmd_journal(memory)
                continue

            if lower.startswith("/graph "):
                entity = raw[7:].strip()
                with console.status("[magenta]Searching graph…[/magenta]", spinner="dots"):
                    await _cmd_graph(memory, entity)
                continue

            if lower.startswith("/rules "):
                _cmd_rules(rules, raw[7:].strip())
                continue

            if lower == "/rules":
                _cmd_rules(rules, "")
                continue

            if lower.startswith("/roll "):
                _cmd_roll(raw[6:].strip())
                continue

            if lower == "/save":
                # Session auto-saves on every turn; this is an explicit confirm.
                console.print("[green]Session saved.[/green]")
                continue

            if lower == "/reset":
                if _confirm("Wipe the session window? (Knowledge graph is preserved)"):
                    memory.reset_session()
                    console.print("[yellow]Session window cleared.[/yellow]")
                continue

            if lower == "/fullreset":
                if _confirm(
                    "Wipe BOTH the session window and the knowledge graph? "
                    "This restarts the campaign from scratch."
                ):
                    with console.status("[red]Resetting…[/red]", spinner="dots"):
                        await memory.full_reset()
                    console.print("[red]Full reset complete.[/red]")
                continue

            # --- Normal player input --------------------------------------
            with console.status("[green]The DM is thinking…[/green]", spinner="dots"):
                response = await dm.respond(raw)

            print_roll_results(dm.last_roll_results)
            print_dm_response(response)

    except KeyboardInterrupt:
        console.print()
        console.print("[dim]Interrupted — saving session and exiting.[/dim]")

    console.print()
    console.print(Rule("[dim]Session ended.[/dim]", style="dim"))
