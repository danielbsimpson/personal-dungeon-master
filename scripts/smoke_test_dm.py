"""
DM agent smoke test — runs one full turn against a real Ollama instance.

Usage
-----
    python scripts/smoke_test_dm.py
    python scripts/smoke_test_dm.py --campaign "example-campaign" --model llama3.1:8b

Prerequisites
-------------
1. Ollama running locally (default http://localhost:11434).
2. A model pulled, e.g.:  ollama pull llama3.1:8b
3. (Optional) Copy .env.example to .env and set DM_MODEL.

What this script tests
----------------------
- Campaign loading and parsing
- Rules loading
- MemoryManager.load() (creates Kuzu DB under memory/)
- build_system_prompt() — full prompt assembly including spoiler guard
- LLM call via OllamaProvider
- MemoryManager.record_turn() — session window + Graphiti episode ingestion
- (Optional) Second turn — verifies session window is threaded correctly
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import textwrap
from pathlib import Path

# Ensure the project root is on the path when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DM agent smoke test")
    parser.add_argument(
        "--campaign",
        default="example-campaign",
        help="Name of the campaign folder under campaigns/ (default: example-campaign)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Ollama model name to use (overrides DM_MODEL in .env)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=1,
        help="Number of interactive turns to run (default: 1)",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete and recreate the campaign memory directory before starting",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    # Late imports so sys.path is already patched
    from src.campaign.loader import load_campaigns
    from src.campaign.parser import parse_campaign
    from src.config import settings
    from src.dm.context_builder import build_system_prompt, detect_narrative_state
    from src.dm.dungeon_master import DungeonMaster
    from src.dm.memory.manager import MemoryManager
    from src.llm.factory import create_provider
    from src.rules.loader import load_rules

    # Allow --model to override .env
    if args.model:
        settings.dm_model = args.model

    # Guard: this script only supports Ollama
    if settings.llm_provider != "ollama":
        print(
            f"\nERROR: LLM_PROVIDER is set to '{settings.llm_provider}' in your .env.\n"
            "This smoke test requires Ollama (local inference only).\n"
            "Set LLM_PROVIDER=ollama in your .env file and re-run."
        )
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Personal Dungeon Master — Smoke Test")
    print(f"{'='*60}")
    print(f"  Campaign : {args.campaign}")
    print(f"  Model    : {settings.dm_model or '(auto-pick)'}")
    print(f"  Provider : {settings.llm_provider}")
    print(f"  Memory   : {settings.memory_dir}")
    print(f"{'='*60}\n")

    # ── 1. Load campaign ──────────────────────────────────────────────
    print("[1/5] Loading campaigns...")
    campaigns = load_campaigns(settings.campaigns_dir)
    campaign_meta = next(
        (c for c in campaigns if c.name == args.campaign), None
    )
    if campaign_meta is None:
        names = [c.name for c in campaigns]
        print(f"ERROR: Campaign '{args.campaign}' not found. Available: {names}")
        sys.exit(1)
    campaign = parse_campaign(campaign_meta)
    print(f"      OK — {len(campaign.scenes)} scenes, {len(campaign.creatures)} creatures")

    # ── 2. Load rules ─────────────────────────────────────────────────
    print("[2/5] Loading rules...")
    rules = load_rules(settings)
    print(f"      OK — sections: {rules.section_names}")

    # ── 3. Initialise memory ──────────────────────────────────────────
    print("[3/5] Initialising memory (Kuzu graph)...")
    if args.reset_memory:
        import shutil
        campaign_mem = settings.memory_dir / args.campaign
        if campaign_mem.exists():
            shutil.rmtree(campaign_mem)
            print(f"      Reset: removed {campaign_mem}")
    memory = MemoryManager(memory_dir=settings.memory_dir, settings=settings)
    await memory.load(args.campaign)
    print(f"      OK — progress at scene {memory.campaign_progress}")

    # ── 4. Create LLM provider ────────────────────────────────────────
    print("[4/5] Connecting to Ollama...")
    provider = create_provider(settings)
    print(f"      OK — model: {settings.dm_model}")

    # ── 5. Build DM ───────────────────────────────────────────────────
    dm = DungeonMaster(
        llm=provider,
        campaign=campaign,
        rules=rules,
        memory=memory,
        settings=settings,
    )

    # ── Opening narration ─────────────────────────────────────────────
    print("\n[5/5] Requesting opening narration...\n")
    print("-" * 60)
    opening = await dm.start_campaign()
    print(_wrap(opening))
    print("-" * 60)

    # ── Interactive turns ─────────────────────────────────────────────
    for i in range(args.turns):
        print(f"\n[Turn {i + 1}]")
        try:
            player_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not player_input:
            continue

        state = detect_narrative_state(player_input)
        print(f"(narrative state: {state.value})\n")

        print("DM:")
        print("-" * 60)
        response = await dm.respond(player_input)
        print(_wrap(response))
        print("-" * 60)
        print(f"(progress: scene {memory.campaign_progress} / {len(campaign.scenes) - 1})")

    print("\nSmoke test complete.")


def _wrap(text: str, width: int = 80) -> str:
    """Wrap long DM output for readability in a terminal."""
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            lines.append(textwrap.fill(paragraph, width=width))
        else:
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
