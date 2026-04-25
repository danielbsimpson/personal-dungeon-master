# Personal Dungeon Master

An AI-powered Dungeon Master that reads from structured campaign files and guides a player through a tabletop RPG adventure via a conversational text interface — with voice interaction planned for a future phase.

Runs against **OpenAI API models** or **local models via Ollama** — your choice at startup. Built for players who want full privacy and offline play on their own hardware.

---

## Overview

Personal Dungeon Master is a locally-run AI assistant that acts as your Dungeon Master (DM). You select a campaign, and the DM reads the full campaign book, character sheet, creature roster, and campaign summary to build a rich, contextual understanding of the adventure. It then guides you through the story turn by turn — narrating scenes, voicing NPCs, resolving actions, and tracking everything that has happened so far using persistent memory.

The DM is designed to be faithful to the campaign source material while never revealing future events prematurely. It adapts to player choices within the scope of the written adventure, maintaining narrative coherence at all times.

The system supports a pluggable LLM provider layer. You can point it at the OpenAI API for cloud-hosted models, or switch to a fully local setup using [Ollama](https://ollama.com/) — ideal for offline play, privacy, or running on your own GPU hardware.

---

## Features

### Core
- **Campaign selection** — choose from any campaign folder in your local `campaigns/` directory
- **AI Dungeon Master** — an LLM-powered DM that narrates, responds, and adjudicates in natural language
- **Text-based chat interface** — type your actions and responses; the DM replies in character
- **Persistent memory** — the DM remembers every encounter, decision, and event from the current session and across sessions
- **Campaign-aware context** — the DM reads all campaign files at startup and uses them to stay accurate and immersive
- **Chronological spoiler protection** — the DM will not reveal future events, locations, or enemies before they are reached in the story
- **Character sheet awareness** — the DM tracks player stats, inventory, abilities, and progression
- **Creature reference** — the DM uses creature data for accurate combat narration and encounter descriptions
- **Pluggable LLM providers** — run against OpenAI API models or local models via Ollama; switch with a single config value
- **Local model support** — works with any model pulled into Ollama (e.g., `llama3`, `mistral`, `gemma3`, `deepseek-r2`); recommended for offline play or GPU-local inference
- **Full 5e rules knowledge** — the DM is grounded in the complete D&D 5th Edition System Reference Document (SRD); it applies rules correctly for combat, spellcasting, ability checks, conditions, and more
- **Rules-accurate adjudication** — when a player attempts an action, the DM applies the correct 5e mechanics (attack rolls, saving throws, skill checks, spell effects) without the player needing to look anything up
- **Integrated dice engine** — the DM rolls all dice using a Python RNG engine with cryptographically seeded randomness; every attack roll, damage roll, saving throw, and skill check uses the correct die type (d4, d6, d8, d10, d12, d20, d100) with appropriate modifiers applied automatically
- **Transparent dice results** — every roll is displayed to the player in the terminal before the DM narrates the outcome, showing the die type, raw roll, modifier, and total

### Planned
- **Voice input** — speak your actions to the DM using speech-to-text
- **Voice output** — the DM narrates back using text-to-speech with a consistent voice persona
- **Save & resume** — save the exact state of an adventure and resume from any point
- **Multiple players** — support for a party of characters in a single campaign
- **Custom campaign creation** — tooling to help author new campaigns in the required format
- **Additional ruleset editions** — extend the rules system beyond 5e to support D&D 3.5e, D&D 4e, Pathfinder 1e/2e, and other TTRPGs
- **DM avatar** — a visual character portrait representing the Dungeon Master, displayed in a GUI or web UI alongside narration
- **Animated visual dice** — a rendered dice-roll animation shown to the player when the DM rolls, replacing the plain text result with a visual display

---

## Repository Structure

```
personal-dungeon-master/
├── README.md
├── TODO.md
├── rules/
│   └── 5e/
│       ├── core.md                # Core mechanics (ability scores, proficiency, skills, checks)
│       ├── combat.md              # Combat rules (initiative, action economy, attacks, damage, death)
│       ├── conditions.md          # All condition definitions (blinded, charmed, frightened, etc.)
│       ├── spellcasting.md        # Spellcasting rules, spell slots, concentration, ritual casting
│       └── equipment.md           # Weapons, armor, adventuring gear, encumbrance rules
├── campaigns/
│   └── example-campaign/
│       ├── README.md              # Campaign summary and lore overview
│       ├── character.md           # Player character sheet (stats, inventory, backstory)
│       ├── creature.md            # Bestiary for this campaign (all enemies and NPCs)
│       └── example-campaign.txt   # The full campaign book (encounters, locations, story)
├── src/
│   ├── main.py                    # Entry point — campaign selection and session loop
│   ├── dm/
│   │   ├── dungeon_master.py      # Core DM agent (LLM orchestration)
│   │   ├── context_builder.py     # Loads and structures campaign files into LLM context
│   │   ├── memory.py              # Persistent memory: session logs, encounter history
│   │   └── spoiler_guard.py       # Ensures the DM does not reference future campaign events
│   ├── llm/
│   │   ├── base.py                # Abstract LLMProvider interface
│   │   ├── openai_provider.py     # OpenAI API implementation
│   │   ├── ollama_provider.py     # Ollama local model implementation
│   │   └── factory.py             # Instantiate the correct provider from config
│   ├── rules/
│   │   ├── loader.py              # Loads rule files for the configured game edition
│   │   └── reference.py           # Retrieves relevant rules sections for a given context
│   ├── dice/
│   │   ├── die.py                 # Die enum (d4, d6, d8, d10, d12, d20, d100) and RollResult dataclass
│   │   └── roller.py              # Core dice engine: roll(), roll_multiple(), advantage/disadvantage
│   ├── campaign/
│   │   ├── loader.py              # Reads and validates campaign folder structure
│   │   ├── parser.py              # Parses campaign book, character sheet, and creature file
│   │   └── selector.py            # Interactive campaign selection at startup
│   ├── interface/
│   │   ├── cli.py                 # Text-based terminal chat interface
│   │   └── voice.py               # (Future) Voice I/O interface
│   └── config.py                  # Configuration (model selection, paths, settings)
├── memory/
│   └── <campaign_name>/
│       ├── session.json           # Current session state and conversation history
│       └── journal.md             # Human-readable log of the player's journey
├── tests/
│   ├── test_loader.py
│   ├── test_parser.py
│   ├── test_memory.py
│   ├── test_rules.py
│   ├── test_dice.py
│   └── test_dm.py
├── .env.example                   # Template for API keys and environment variables
├── requirements.txt               # Python dependencies
└── pyproject.toml                 # Project metadata and tooling config
```

---

## Campaign Folder Format

Each campaign lives in its own folder under `campaigns/`. The folder name becomes the campaign identifier.

### `README.md` — Campaign Summary
A high-level overview of the campaign: setting, tone, main plot hooks, and any player-facing lore that is safe to know before the adventure begins.

```markdown
# The Lost Mines of Phandelver

**Setting:** The Sword Coast, Faerûn  
**Recommended Level:** 1–5  
**Tone:** Classic heroic fantasy, dungeon exploration  

## Summary
The players are hired to escort a wagon to Phandalin...
```

### `character.md` — Character Sheet
Full character details for the player character(s). This includes stats, class, race, background, abilities, spells, inventory, and any backstory relevant to the campaign.

```markdown
# Aldric Stoneforge

**Race:** Dwarf  
**Class:** Fighter (Level 1)  
**Background:** Soldier  

## Ability Scores
| STR | DEX | CON | INT | WIS | CHA |
|-----|-----|-----|-----|-----|-----|
| 16  | 12  | 15  | 10  | 13  | 8   |

## Equipment
- Chain mail
- Longsword
- Shield
...
```

### `creature.md` — Bestiary
A catalogue of all creatures (enemies, beasts, and key NPCs) the player may encounter. Each entry includes stat blocks, abilities, and flavor text the DM can use for narration.

```markdown
# Creature Reference

## Goblin
**Type:** Humanoid (Goblinoid)  
**CR:** 1/4  
**HP:** 7 | **AC:** 15  
**Speed:** 30 ft.

**Attacks:** Scimitar +4 (1d6+2 slashing), Shortbow +4 (1d6+2 piercing)  
**Traits:** Nimble Escape — can Disengage or Hide as a bonus action.

*Flavor:* Goblins are small, black-hearted humanoids that lair in...
```

### `[campaign_name].txt` — Campaign Book
The full campaign text, written sequentially. This is the authoritative source for locations, encounters, NPCs, story beats, and outcomes. The DM reads this file in full at startup and uses it as its primary reference. The DM will only surface information from sections that have been chronologically reached in the current adventure.

---

## Getting Started

### Prerequisites
- Python 3.11+
- **One of the following LLM backends:**
  - **OpenAI API** — an API key from [platform.openai.com](https://platform.openai.com)
  - **Ollama (local)** — [Ollama](https://ollama.com/) installed and running with at least one model pulled (e.g., `ollama pull llama3`). A mid-range GPU (e.g., RTX 4060 8 GB) can run 8B–13B parameter models comfortably.
- A campaign folder set up under `campaigns/`

### Installation

```bash
git clone https://github.com/your-username/personal-dungeon-master.git
cd personal-dungeon-master
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and configure your chosen backend:

```bash
cp .env.example .env
```

**Option A — OpenAI API:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
DM_MODEL=gpt-4o
CAMPAIGNS_DIR=./campaigns
MEMORY_DIR=./memory
GAME_EDITION=5e
RULES_DIR=./rules
```

**Option B — Local Ollama:**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
DM_MODEL=llama3
CAMPAIGNS_DIR=./campaigns
MEMORY_DIR=./memory
GAME_EDITION=5e
RULES_DIR=./rules
```

For Ollama, make sure the Ollama service is running (`ollama serve`) and the model is pulled (`ollama pull llama3`) before starting the DM. You can also let the startup menu list your available local models and pick one interactively.

### Running

```bash
python src/main.py
```

You will be prompted to select a campaign. Once selected, the DM will load all campaign files, initialize memory, and begin the adventure.

---

## How It Works

### 1. Campaign Loading
At startup, the `campaign/loader.py` module scans the `campaigns/` directory, validates each campaign folder, and presents a selection menu. Once chosen, `parser.py` reads all four campaign files into structured data objects.

### 2. LLM Provider Selection
`llm/factory.py` reads `LLM_PROVIDER` from config and instantiates the correct provider:
- `openai` → `OpenAIProvider` — calls the OpenAI Chat Completions API using the `openai` Python SDK
- `ollama` → `OllamaProvider` — calls the local Ollama REST API (`/api/chat`), which is OpenAI-compatible, so the same message format is used for both

Both providers implement the same `LLMProvider` abstract interface: `complete(messages, **kwargs) -> str`. The DM agent never knows or cares which backend is active.

If the provider is `ollama`, the startup sequence also queries `ollama list` to display available local models, allowing the user to pick one before the adventure begins.

### 3. Rules Loading
`rules/loader.py` reads the edition set in `GAME_EDITION` (default `5e`) and loads all rule files from `rules/5e/` into a structured `RulesReference` object. This object is passed to the context builder and is available for the DM agent to draw from throughout the session.

`rules/reference.py` exposes a `get_relevant_rules(context: str) -> str` helper that returns the most pertinent rules sections given a narrative context (e.g., combat started → return combat rules; a spell is cast → return spellcasting rules). This keeps the rules portion of the system prompt focused and within token budget.

### 4. Dice Engine
All dice rolls are performed by `dice/roller.py` using Python's `random` module seeded from `secrets.randbits()` at session start, guaranteeing real randomness the LLM cannot predict or influence.

The LLM **never** generates roll results itself — instead it emits a structured roll tag (e.g., `[ROLL: attack d20+5]`) wherever a roll is required. The DM agent intercepts every tag in the raw response, calls the dice engine, and substitutes real results before the narrative is rendered. The resolved results are also injected as a system message so the LLM narrates from the actual outcome on its next turn.

The `Die` enum (`d4`, `d6`, `d8`, `d10`, `d12`, `d20`, `d100`) constrains every roll to the correct physical die. The engine supports:
- `roll(die, modifier)` — single die with modifier
- `roll_multiple(n, die)` — multi-dice expressions (e.g., 2d6 for a greatsword)
- `roll_advantage(die)` / `roll_disadvantage(die)` — roll twice, keep highest/lowest

Every roll result is displayed to the player in the terminal (die type, each individual roll, modifier, total) before the DM narrates the outcome.

### 5. Context Building
`dm/context_builder.py` assembles a system prompt for the LLM. This prompt includes:
- The DM persona and behavioral rules
- The game edition and active rules reference (relevant sections from `rules/5e/`)
- The campaign summary and tone
- The full character sheet
- The full creature reference
- The portion of the campaign book that has been reached so far (the "revealed window")
- A summary of everything that has happened in the adventure (from memory)

### 6. Spoiler Guard
`dm/spoiler_guard.py` maintains a pointer into the campaign book representing how far the player has progressed. Only content up to and including the current narrative position is included in the LLM context. Future encounters, plot twists, and locations are withheld until the player reaches them.

### 7. Memory
`dm/memory.py` maintains two forms of memory:
- **Session history** — the full conversation log for the current session, stored in `memory/<campaign_name>/session.json`
- **Journey journal** — a summarized, human-readable log of key events, decisions, NPCs met, and items found, stored in `memory/<campaign_name>/journal.md`

At the start of each session, the journal is included in the LLM context so the DM is fully caught up on the player's history, even after the conversation window has been truncated.

### 8. The Chat Loop
The main loop in `interface/cli.py` accepts text input from the user, passes it to the DM agent along with the full context window, and prints the DM's response. The conversation continues until the user quits.

---

## DM Behavior & Persona

The DM is instructed to:
- Narrate in an immersive, engaging second-person voice ("You step into the dimly lit tavern...")
- Voice NPCs distinctly, using their names and personalities from the campaign book
- Signal dice rolls using a structured tag (e.g., `[ROLL: attack d20+5]`) rather than narrating a result directly — the dice engine intercepts these tags, performs the real roll, and the DM narrates the outcome from the actual result
- Use the correct die for every situation: d20 for attack rolls and skill checks, weapon-specific dice for damage (d6 for shortsword, d8 for longsword, 2d6 for greatsword, etc.), d8 for healing spells, and so on
- Apply 5e rules correctly and consistently: attack rolls, saving throws, ability checks, spell effects, conditions, and action economy
- Reference the character sheet for class features, spell slots, proficiencies, and modifiers when adjudicating player actions
- Apply creature stat blocks accurately during combat (AC, HP, attacks, special traits)
- Handle edge cases (contested rolls, cover, concentration, opportunity attacks) according to RAW (Rules As Written) by default
- Never break the fourth wall or acknowledge being an AI unless the user explicitly asks
- Advance the story only as fast as the player drives it
- Summarize key events in memory after each significant encounter

---

## Rules System

The DM's rules knowledge is stored in the `rules/` directory as plain Markdown files, organized by game edition. The initial edition is **D&D 5th Edition**, sourced from the [5e SRD (Systems Reference Document)](https://dnd.wizards.com/resources/systems-reference-document), which is released under Creative Commons CC-BY-4.0 and free to use.

### Rule Files (`rules/5e/`)

| File | Contents |
|---|---|
| `core.md` | Ability scores, modifiers, proficiency bonus, advantage/disadvantage, skill checks, saving throws, passive scores |
| `combat.md` | Initiative, turn structure, action economy (action/bonus action/reaction/movement), attack rolls, damage, critical hits, death saving throws, conditions applied in combat |
| `conditions.md` | Full definitions of all 15 conditions (blinded, charmed, deafened, exhaustion, frightened, grappled, incapacitated, invisible, paralyzed, petrified, poisoned, prone, restrained, stunned, unconscious) |
| `spellcasting.md` | Spell slots, casting time, range, components, concentration, ritual casting, spell attack rolls, saving throw DCs, spell schools |
| `equipment.md` | Weapon properties, armor categories, encumbrance, improvised weapons, silvered/magical weapons |

### How Rules Are Used

At session startup, all rule files for the configured edition are loaded into a `RulesReference` object. The context builder injects relevant sections into the system prompt based on the current narrative state (e.g., if combat begins, the full combat and conditions sections are included). This keeps the DM accurate without consuming the entire context window with rules on every turn.

For future scalability, the rules reference is designed to support RAG (Retrieval-Augmented Generation) — querying only the most relevant rules passages per turn rather than including the full ruleset each time.

---

## Local Model Recommendations

When using Ollama, model choice has a significant impact on narration quality and response speed. The following models have been tested or are recommended for this use case (all runnable on an RTX 4060 8 GB):

| Model | Size | Notes |
|---|---|---|
| `llama3.1:8b` | ~5 GB | Good balance of speed and quality for narration |
| `mistral:7b` | ~4 GB | Fast, strong instruction following |
| `gemma3:9b` | ~6 GB | Strong at creative writing and roleplay |
| `deepseek-r2:8b` | ~5 GB | Good reasoning for complex encounter adjudication |
| `llama3.1:70b-q4` | ~40 GB | Requires high VRAM; best quality if hardware allows |

Larger models (30B+) will require a GPU with more VRAM or CPU offloading (slower). The context window of the chosen model is automatically respected — the system will warn if the campaign + memory content exceeds the model's context length.

---

## Voice Interface (Future)

The planned voice interface will use:
- **Speech-to-text**: capture the player's spoken input and transcribe it — `faster-whisper` (runs locally, no API needed) is the preferred option for Ollama users who want a fully offline setup
- **Text-to-speech**: read the DM's response aloud using a consistent, atmospheric voice
- A push-to-talk or voice activity detection (VAD) mode

The text interface and voice interface will share the same underlying DM agent, differing only in the I/O layer.

---

## Contributing

This is a personal project, but contributions and ideas are welcome. Please open an issue before submitting a pull request.

---

## License

MIT
