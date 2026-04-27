# TODO — Personal Dungeon Master

A phased development plan for building the Personal Dungeon Master system. Tasks are ordered by priority and dependency within each phase.

---

## Phase 1 — Project Scaffolding

Get the repository structure, tooling, and configuration in place before writing any logic.

- [x] Initialize Python project with `pyproject.toml` (package name, version, Python version constraint)
- [x] Create `requirements.txt` with initial dependencies:
  - `openai` (OpenAI API client — also used for the Ollama OpenAI-compatible endpoint)
  - `httpx` (direct HTTP calls to Ollama REST API where needed)
  - `python-dotenv` (environment variable loading)
  - `rich` (terminal formatting and styling)
  - `typer` (CLI argument parsing)
  - `pydantic` (data validation for campaign file parsing)
- [x] Create `.env.example` with all required environment variables:
  - `LLM_PROVIDER` (`openai` or `ollama`)
  - `OPENAI_API_KEY` (required when `LLM_PROVIDER=openai`)
  - `OLLAMA_BASE_URL` (default `http://localhost:11434`, required when `LLM_PROVIDER=ollama`)
  - `DM_MODEL` (e.g., `gpt-4o` for OpenAI, `llama3` for Ollama)
  - `GAME_EDITION` (default `5e`; controls which rules folder is loaded)
  - `CAMPAIGNS_DIR`, `MEMORY_DIR`, `RULES_DIR`
- [x] Add `.gitignore` (exclude `.env`, `__pycache__`, `.venv`, `memory/`, `*.pyc`)
- [x] Create `src/` directory with `__init__.py` files for each subpackage
- [x] Create placeholder `tests/` directory with a `conftest.py` and stub test files:
  - `test_llm.py`, `test_loader.py`, `test_parser.py`, `test_memory.py`, `test_rules.py`, `test_dice.py`, `test_dm.py`
- [x] Create the `campaigns/` directory with an `example-campaign/` folder containing sample files:
  - `README.md` (short sample campaign summary)
  - `character.md` (sample character sheet)
  - `creature.md` (2–3 sample creature entries)
  - `example-campaign.txt` (short sample campaign narrative — 3–5 scenes)
- [x] Create the `rules/5e/` directory with initial rule files sourced from the D&D 5e SRD (CC-BY-4.0):
  - `core.md` (ability scores, proficiency, skills, checks, saving throws, advantage/disadvantage)
  - `combat.md` (initiative, action economy, attack rolls, damage, critical hits, death saves)
  - `conditions.md` (all 15 condition definitions)
  - `spellcasting.md` (spell slots, concentration, ritual casting, spell attack rolls, save DCs)
  - `equipment.md` (weapon properties, armor categories, encumbrance)
- [x] Create `src/config.py` — load all `.env` values, define default paths, expose a `Settings` object; validate that the chosen provider's required fields are present on startup; include `GAME_EDITION` (default `5e`) and `RULES_DIR`

---

## Phase 2 — Ollama LLM Provider

Build the provider layer for local inference via Ollama. The entire system runs locally before any external API is introduced. OpenAI support is added later in Phase 13.

- [x] `src/llm/base.py`
  - [x] Define abstract `LLMProvider` base class with a single required method: `complete(messages: list[dict], **kwargs) -> str`
  - [x] Define a `ModelInfo` dataclass (`name`, `context_window`, `provider`)
- [x] `src/llm/ollama_provider.py`
  - [x] Implement `OllamaProvider(LLMProvider)` using the OpenAI-compatible Ollama endpoint (`{OLLAMA_BASE_URL}/v1/chat/completions`) via the `openai` SDK with a custom `base_url`
  - [x] Implement `list_models() -> list[ModelInfo]` — calls `GET {OLLAMA_BASE_URL}/api/tags` and returns available local models
  - [x] Check that the Ollama service is reachable at startup; print a helpful error if not
  - [x] Respect the context window reported by `ollama show <model>` (query at startup and store in settings)
- [x] `src/llm/factory.py`
  - [x] `create_provider(settings: Settings) -> LLMProvider` — read `LLM_PROVIDER` and return the correct implementation; raise a clear not-yet-supported error if the provider is anything other than `ollama`
  - [x] If `DM_MODEL` is not set, call `list_models()` and present an interactive selection menu using `rich`
- [x] Write unit tests in `tests/test_llm.py`
  - [x] Test that `factory.create_provider` returns `OllamaProvider` when `LLM_PROVIDER=ollama`
  - [x] Test that an unsupported provider value raises a clear, user-friendly error
  - [x] Mock the HTTP response and test `OllamaProvider.list_models` parses the tags response correctly
  - [x] Test that a missing Ollama service raises a clear, user-friendly error

---

## Phase 3 — Campaign Loading & Parsing

Build the layer that reads campaign folders and turns them into structured data the DM can use.

- [x] `src/campaign/loader.py`
  - [x] Scan `CAMPAIGNS_DIR` for valid campaign folders (must contain all four required files)
  - [x] Return a list of `Campaign` metadata objects (name, path, file paths)
  - [x] Raise a clear error if a required file is missing or the directory is empty
- [x] `src/campaign/parser.py`
  - [x] Parse `README.md` → campaign summary string
  - [x] Parse `character.md` → structured `Character` Pydantic model (name, stats, class, inventory, etc.)
  - [x] Parse `creature.md` → list of `Creature` Pydantic models (name, stats, abilities, flavor)
  - [x] Read `[campaign_name].txt` → raw campaign book text, split into logical sections or scenes
  - [x] Define a `ParsedCampaign` dataclass/model that holds all four parsed components
- [x] `src/campaign/selector.py`
  - [x] Display a numbered list of available campaigns using `rich`
  - [x] Accept user input and return the selected `Campaign`
  - [x] Handle invalid input gracefully (re-prompt)
- [x] Write unit tests in `tests/test_loader.py` and `tests/test_parser.py`
  - [x] Test that valid campaigns load correctly
  - [x] Test that missing files raise the appropriate errors
  - [x] Test that the example campaign parses without errors

---

## Phase 4 — Rules System

Load the full ruleset for the configured game edition and make it available to the DM agent.

- [x] `src/rules/loader.py`
  - [x] Read `GAME_EDITION` from settings (default `5e`)
  - [x] Scan `RULES_DIR/<edition>/` and load all `.md` files into a `RulesReference` dataclass
  - [x] Each rule file maps to a named section (e.g., `combat`, `spellcasting`, `conditions`)
  - [x] Raise a clear error if the edition folder does not exist or contains no rule files
- [x] `src/rules/reference.py`
  - [x] `get_all_rules(ref: RulesReference) -> str` — return the full rules text concatenated (for smaller models or short campaigns)
  - [x] `get_relevant_rules(ref: RulesReference, context: str) -> str` — return only the sections most relevant to the current narrative context (e.g., if combat is active return `combat.md` + `conditions.md`; if a spell is being cast return `spellcasting.md`)
  - [x] Define a `NarrativeState` enum (`EXPLORATION`, `COMBAT`, `SOCIAL`, `REST`) to drive section selection
  - [x] Always include `core.md` as a baseline in every prompt
- [x] Populate `rules/5e/` rule files with accurate 5e SRD content:
  - [x] `core.md` — ability scores & modifiers, proficiency bonus progression, skill list & governing abilities, ability checks, saving throws, passive scores, advantage/disadvantage, hidden rules
  - [x] `combat.md` — initiative, surprise, turn structure, action/bonus action/reaction/free action/movement, attack rolls, damage rolls, critical hits, two-weapon fighting, grapple/shove, ranged attack rules, cover, death saving throws, stabilisation, mounted combat, underwater combat
  - [x] `conditions.md` — full definitions for all 15 conditions with mechanical effects
  - [x] `spellcasting.md` — spell slots by class, cantrips, concentration rules, ritual casting, verbal/somatic/material components, spell attack rolls, spellcasting ability modifier & save DC formula, counterspell/dispel rules, spell school descriptions
  - [x] `equipment.md` — weapon properties (finesse, versatile, thrown, reach, etc.), weapon tables, armor categories & AC, shields, donning/doffing armor, encumbrance (standard & variant), silvered weapons, improvised weapons
- [x] Write unit tests in `tests/test_rules.py`
  - [x] Test that the loader correctly reads all files in `rules/5e/`
  - [x] Test that `get_relevant_rules` returns the combat section when `NarrativeState` is `COMBAT`
  - [x] Test that `core.md` is always included regardless of state
  - [x] Test that an unknown edition raises a clear error

---

## Phase 5 — Memory System (Graphiti Graph RAG)

Build the memory system using [Graphiti](https://github.com/getzep/graphiti) — a temporal context graph engine that automatically extracts entities and relationships from each DM response and stores them in a queryable knowledge graph. Graphiti handles entity extraction, deduplication, temporal fact management, and hybrid retrieval; we only need to wire it to our turn loop and provide a local graph database.

### Architecture

- **Graph engine**: `graphiti-core` backed by **Kuzu** — an embeddable, file-based graph database (no separate server, stores under `memory/<campaign_name>/graphiti.kuzu/`). Kuzu is the local-first equivalent of Neo4j.
- **Entity extraction**: Graphiti calls the configured LLM with a structured-output prompt to extract entities and relationships from each ingested episode. No custom extraction code needed.
- **Episodes**: Each DM response (and optionally player input) is ingested as a Graphiti episode (`add_episode()`). Graphiti derives entities, relationships, and temporal facts automatically.
- **Retrieval**: `graphiti.search(query)` performs hybrid retrieval (semantic + keyword + graph traversal) to return the most relevant facts for the current turn.
- **Short-term context**: `SessionStore` continues to manage the last N raw `{role, content}` messages — Graphiti does not manage the chat window.
- **Progress pointer**: `memory/<campaign_name>/progress.json` — current scene index (unchanged).

> ⚠️ **Structured output requirement**: Graphiti's entity extraction relies on LLM structured output (function/tool calling). Ollama support is via `OpenAIGenericClient`. Models with strong structured output support are recommended — e.g., `llama3.1:8b`, `mistral-nemo`, `qwen2.5:7b`, or `deepseek-r1:7b`. Smaller models without tool-calling support may produce malformed extractions that Graphiti will silently skip. Document the recommended model in `.env.example`.

### Tasks

- [x] Add `graphiti-core[kuzu]` to `requirements.txt` and `pyproject.toml` (replaces `networkx`)
- [x] `src/dm/memory/graphiti_store.py`
  - [x] `GraphitiStore` class wrapping a `Graphiti` instance configured with `KuzuDriver`
    - [x] `__init__(db_path: Path, llm_client, embedder)` — create `KuzuDriver(db=str(db_path))`, instantiate `Graphiti(graph_driver=driver, llm_client=llm_client, embedder=embedder)`
    - [x] `async setup()` — call `graphiti.build_indices_and_constraints()` once on first run
    - [x] `async add_episode(name: str, content: str, source_description: str, turn: int, group_id: str)` — thin wrapper around `graphiti.add_episode(...)` with `reference_time=datetime.now()`
    - [x] `async search(query: str, group_id: str, num_results: int = 10) -> str` — call `graphiti.search(query, group_ids=[group_id], num_results=num_results)`, format results as a structured text block for the system prompt
    - [x] `async close()` — call `graphiti.close()`
- [x] `src/dm/memory/graphiti_factory.py`
  - [x] `build_graphiti_clients(settings) -> tuple[llm_client, embedder]` — constructs `OpenAIGenericClient` and `OpenAIEmbedder` both pointed at `{OLLAMA_BASE_URL}/v1` with `api_key="ollama"`, using `settings.dm_model` and a configured embedding model (default `nomic-embed-text`)
  - [x] Pulls embedding model name from a new `EMBEDDING_MODEL` setting (default `nomic-embed-text`, dim `768`)
- [x] `src/dm/memory/session_store.py`
  - [x] `SessionStore` class scoped to a campaign path
  - [x] `load()` — read `session.json` from disk on startup; return empty list if absent
  - [x] `append(role: str, content: str)` — add message and persist; trim to the last N messages (default 20, configurable via `SESSION_WINDOW` setting)
  - [x] `messages() -> list[dict]` — return the current window as `[{role, content}, ...]`
  - [x] `clear()` — wipe the session window (graph is preserved)
- [x] `src/dm/memory/manager.py`
  - [x] `MemoryManager` class — composes `GraphitiStore`, `SessionStore`, and progress pointer
  - [x] `async load(campaign_name: str)` — set up `GraphitiStore` (run `setup()` if first use), load session and progress from disk
  - [x] `async record_turn(player_input: str, dm_response: str, turn: int)` — append both messages to `SessionStore`; ingest `dm_response` as a Graphiti episode
  - [x] `async get_context(current_text: str, group_id: str) -> str` — call `GraphitiStore.search(current_text, group_id)` and return the formatted block for injection into the system prompt
  - [x] `advance_progress(to_section: int)` — update and persist the scene pointer (monotonically increasing)
  - [x] `campaign_progress` property — current section index
  - [x] `reset_session()` — wipe session window only; graph is preserved
  - [x] `full_reset()` — wipe session window; drop and recreate the Kuzu DB directory
- [x] Add to `src/config.py` and `.env.example`:
  - [x] `SESSION_WINDOW` (default `20`) — short-term message window size
  - [x] `EMBEDDING_MODEL` (default `nomic-embed-text`) — Ollama embedding model for Graphiti
  - [x] `GRAPHITI_TELEMETRY_ENABLED=false` — disable anonymous telemetry in `.env.example`
- [x] Write unit tests in `tests/test_memory.py`
  - [x] Test `SessionStore` trims to the configured window size
  - [x] Test `SessionStore` persists and reloads correctly across instances
  - [x] Test `SessionStore.clear()` empties the window without touching graph files
  - [x] Test `MemoryManager.advance_progress` is monotonically increasing (cannot go backwards)
  - [x] Mock `GraphitiStore` and test `MemoryManager.record_turn` calls `add_episode` and `SessionStore.append` correctly
  - [x] Mock `GraphitiStore` and test `MemoryManager.get_context` returns the formatted search result string

---

## Phase 6 — DM Agent Core

Build the LLM-powered Dungeon Master that reads context and generates responses.

- [x] `src/dm/spoiler_guard.py`
  - [x] Accept the full campaign book text and a current progress pointer
  - [x] Return only the portion of the campaign book up to and including the current section
  - [x] Implement section detection (scenes, chapters, or encounter headers in the `.txt` file)
  - [x] Ensure the revealed window grows as progress advances but never shrinks
- [x] `src/dm/context_builder.py`
  - [x] `build_system_prompt(campaign: ParsedCampaign, rules: RulesReference, memory: MemoryManager) -> str`
    - [x] Include DM persona instructions and behavioral rules
    - [x] Include game edition identifier and relevant rules sections (via `rules/reference.py`)
    - [x] Include campaign summary
    - [x] Include character sheet (formatted clearly for the LLM)
    - [x] Include creature reference
    - [x] Include the revealed portion of the campaign book (via `spoiler_guard`)
    - [x] Include the retrieved graph memory context (via `memory.get_context()`) — entities and relationships relevant to the current turn, not a full history dump
    - [x] Append the short-term session window (last N messages) as prior conversation context
  - [x] Keep system prompt within a configurable token budget (avoid context overflow)
  - [x] Log a warning if the context approaches the model's context window limit
  - [x] Update `NarrativeState` passed to `get_relevant_rules` based on current session state (e.g., swap to `COMBAT` when combat begins)
- [x] `src/dm/dungeon_master.py`
  - [x] `DungeonMaster` class — wraps the `LLMProvider` (injected via factory), context builder, and memory manager
  - [x] `start_campaign()` — deliver the opening narration (introduction to the adventure)
  - [x] `respond(player_input: str) -> str`:
    1. Infer narrative state from player input
    2. Build context (system prompt + session window) and call LLM
    3. Record turn in memory (session window + Graphiti episode)
    4. Advance progress pointer if next scene title appears in DM response
    5. Return the final DM response
    - Note: Dice tag substitution (`[ROLL: ...]`) wired in Phase 7 ✅
  - [ ] Implement retry logic with exponential backoff for API errors (delegates to provider implementation)
- [x] Write unit tests in `tests/test_dm.py`
  - [x] Mock the `LLMProvider` interface and test that the system prompt is built correctly
  - [x] Test that spoiler guard correctly limits campaign content
  - [x] Test that progress advances after appropriate responses

---

## Phase 7 — Dice Engine

Build the dice rolling engine that gives the DM real, unbiased randomness for every roll in the game.

- [x] `src/dice/die.py`
  - [x] Define `Die` enum with all standard tabletop die types: `D4`, `D6`, `D8`, `D10`, `D12`, `D20`, `D100`
  - [x] Each member stores its number of faces as an integer value (e.g., `D20 = 20`)
  - [x] Define `RollResult` dataclass: `die`, `rolls: list[int]`, `modifier: int`, `total: int`, `label: str` (e.g., `"attack"`)
  - [x] Define `RollRequest` dataclass: `label`, `die`, `count` (default 1), `modifier` (default 0), `advantage: bool`, `disadvantage: bool`
- [x] `src/dice/roller.py`
  - [x] Seed `random.Random` from `secrets.randbits(128)` at module load — ensures cryptographically seeded randomness that the LLM cannot predict
  - [x] `roll(req: RollRequest) -> RollResult`:
    - [x] Roll `req.count` dice, each constrained to `[1, req.die.value]`
    - [x] If `advantage=True`: roll twice, keep the highest single die result
    - [x] If `disadvantage=True`: roll twice, keep the lowest single die result
    - [x] Apply `modifier` to the sum of rolls to produce `total`
    - [x] `total` is always at minimum 1 (per 5e rules for damage rolls)
  - [x] `parse_roll_tags(text: str) -> list[RollRequest]` — extract all `[ROLL: <label> <NdX>[+/-modifier>]]` tags from an LLM response string using regex
  - [x] `substitute_rolls(text: str, results: list[RollResult]) -> str` — replace each `[ROLL: ...]` tag in the text with its formatted result string
  - [x] `format_result(result: RollResult) -> str` — produce a human-readable result string for terminal display (e.g., `✨ Attack Roll [d20+5]: rolled 14 + 5 = **19**`)
- [x] Write unit tests in `tests/test_dice.py`
  - [x] Test that `Die` enum values match their face counts (D6.value == 6, D20.value == 20, etc.)
  - [x] Test that `roll()` always returns a value within the valid range `[1 + modifier, die.value + modifier]` (allowing for negative modifiers flooring at 1)
  - [x] Test that rolling multiple dice returns the expected number of individual rolls
  - [x] Test that advantage always returns a total ≥ disadvantage given the same seed
  - [x] Test that `parse_roll_tags` correctly extracts die type, count, and modifier from tag strings
  - [x] Test that `substitute_rolls` replaces all tags in a string without mutating non-tag content
  - [x] Test 1000 rolls on each die type — assert min/max are within bounds and distribution is roughly uniform

---

## Phase 8 — Text Interface & Main Loop

Wire everything together into a working text-based adventure session.

- [x] `src/interface/cli.py`
  - [x] Print a styled welcome banner using `rich`
  - [x] Display the selected campaign name and a brief summary before the adventure begins
  - [x] Render DM responses with atmospheric formatting (markdown-aware, colored output)
  - [x] Accept player input via `input()` prompt styled with `rich`
  - [x] Handle special commands:
    - [x] `/quit` or `/exit` — save and exit gracefully
    - [x] `/journal` — render a human-readable view of the knowledge graph (entities grouped by type, with their relationships) using `rich`
    - [x] `/status` — display current character stats and inventory
    - [x] `/save` — explicitly save session state (auto-save also happens after every message)
    - [x] `/reset` — confirm and wipe the session window; knowledge graph is preserved
    - [x] `/fullreset` — confirm and wipe both session window and knowledge graph (restart the campaign from scratch)
    - [x] `/roll <expression>` — let the player roll their own dice at any time (e.g., `/roll d20+3`, `/roll 2d6`); display the result in the same styled format as DM rolls
    - [x] `/graph <entity>` — look up a specific entity in the knowledge graph and display its relationships
    - [x] `/help` — list available commands
  - [x] After each DM response, if dice were rolled, render each `RollResult` in a styled panel (using `rich`) showing: die type badge, individual roll values, modifier, and bold total — displayed between the player's input and the DM's narration
  - [x] Handle `KeyboardInterrupt` (Ctrl+C) gracefully — save and exit
- [x] `src/main.py`
  - [x] Entry point: load config, run campaign selector, load rules for configured edition, initialize LLM provider via factory, initialize `DungeonMaster`, start chat loop
  - [x] Verify Ollama is running at startup; if `DM_MODEL` is not set, show model picker
  - [x] Check that all required environment variables are set before proceeding
  - [x] Print a helpful error message if the `campaigns/` directory is empty or missing
  - [x] Print a helpful error message if the `rules/<edition>/` directory is missing
- [ ] Manual end-to-end test with the example campaign:
  - [ ] Play through at least 3 scenes of the example campaign
  - [ ] Verify spoiler guard does not reveal future scenes
  - [ ] Verify knowledge graph accumulates entities and relationships correctly after each turn
  - [ ] Run `/journal` and confirm entities render correctly
  - [ ] Quit and resume — verify the DM remembers NPCs, locations, and events from the previous session via graph retrieval

---

## Phase 9 — Polish & Reliability

Harden the system before adding features.

- [ ] Add a `--edition` CLI flag to override `GAME_EDITION` from the command line (e.g., `python src/main.py --edition 5e`)
- [ ] Add a `/rules` CLI command to look up a specific rules topic mid-session (e.g., `/rules grapple`)
- [ ] Add input validation for all campaign file parsing (graceful error messages, not stack traces)
- [ ] Add a `--campaign` CLI flag to skip the selection menu (`python src/main.py --campaign lost-mines`)
- [ ] Add a `--reset` CLI flag to reset session state at startup
- [ ] Add a `--provider` CLI flag to override `LLM_PROVIDER` from the command line (Ollama-only until Phase 13; passing `openai` will raise a clear not-yet-supported error)
- [ ] Add a `--model` CLI flag to override `DM_MODEL` from the command line (e.g., quickly switch local models)
- [ ] Implement token counting before sending to the LLM — the graph retrieval step naturally bounds long-term memory size; also truncate the oldest session window messages if the total context still exceeds the limit, preserving the system prompt and the retrieved graph context; use model context window from `ModelInfo`
- [ ] At Ollama startup, query the selected model's context length via `ollama show <model>` and store it in settings
- [ ] Add a configurable `DM_TEMPERATURE` setting (default `0.8`) for response creativity
- [ ] Add a configurable `MAX_TOKENS` setting for DM response length
- [ ] Write a `validate-campaign` helper script that checks a campaign folder for correctness and reports any issues
- [ ] Add logging to a `logs/` directory (info-level for session events, debug-level for LLM calls)

---

## Phase 10 — Streaming Output & Player Interrupt

Replace the “Thinking...” spinner with live word-by-word output from the LLM. Allow the player to interrupt the DM mid-response, just as you might cut off a speaker at the table.

- [ ] `src/llm/base.py`
  - [ ] Add optional `stream(messages: list[dict], **kwargs) -> Iterator[str]` method to `LLMProvider`; provide a default implementation that delegates to `complete()` for providers that do not implement streaming
- [ ] `src/llm/ollama_provider.py`
  - [ ] Implement `stream()` using `stream=True` on the OpenAI-compatible client; yield string tokens as they arrive from the model
- [ ] `src/dm/dungeon_master.py`
  - [ ] Add `respond_stream(player_input: str) -> Iterator[str]` — same pipeline as `respond()` but yields tokens incrementally
  - [ ] Buffer the incoming stream to detect and resolve complete `[ROLL: ...]` tags before yielding downstream; accumulate partial tags at chunk boundaries until the closing `]` is received, then perform the roll and emit the resolved text
  - [ ] Record the full (or partial, if interrupted) response in memory regardless of whether the stream was cut short
- [ ] `src/interface/cli.py`
  - [ ] Replace the “Thinking...” spinner with a live streaming display using `rich.live`; print each token as it arrives
  - [ ] Implement player interrupt: spawn a background thread that waits for any keyboard input while the DM is streaming; when triggered, send a cancel signal to the stream iterator and display a `[interrupted]` indicator
  - [ ] Ensure partial DM responses are formatted and stored correctly after interruption
  - [ ] Ensure dice roll panels still render correctly when rolls occur mid-stream
- [ ] Write unit tests
  - [ ] Test that `OllamaProvider.stream()` yields multiple incremental string chunks
  - [ ] Test that dice tags split across chunk boundaries are buffered and resolved correctly
  - [ ] Test that an interrupted response is still passed to `MemoryManager.record_turn`

---

## Phase 11 — DM Personality System

Give the player six distinct Dungeon Master personalities to choose from at startup. Each personality changes the DM’s narrative voice, verbosity, and disposition through a targeted system prompt directive.

### Personalities

| Name | Tone | Verbosity | Description |
|---|---|---|---|
| **The Sage** | Kind | Balanced | Measured, wise, and balanced. Thoughtful pacing. Kind but honest about consequences. The reliable default. |
| **The Chronicler** | Kind | Verbose | Literary narrator. Richly detailed scenes with evocative prose — every shadow and scent accounted for. Never rushes. |
| **The Bard** | Neutral | Verbose | Theatrical and charismatic. Every NPC voiced with dramatic flair. Leans into humor, unexpected twists, and memorable moments. |
| **The Tactician** | Neutral | Concise | Precise and rules-focused. Efficient narration, strict mechanical accuracy, fair challenge above all. |
| **The Warden** | Harsh | Concise | Austere and unforgiving. Terse narration, permanent consequences, real danger. The world does not forgive mistakes. |
| **The Mentor** | Kind | Balanced | Patient, encouraging, and beginner-friendly. Explains rules clearly, celebrates good decisions, and guides gently. |

### Tasks

- [ ] `src/dm/personality.py`
  - [ ] Define `DMPersonality` dataclass: `name: str`, `description: str`, `system_prompt_directive: str`, `verbosity: Literal["concise", "balanced", "verbose"]`, `tone: Literal["harsh", "neutral", "kind"]`
  - [ ] Define `PERSONALITIES: list[DMPersonality]` — the six named personalities above, each with a full `system_prompt_directive` paragraph describing the expected DM behaviour
  - [ ] Implement `get_personality(name: str) -> DMPersonality` — case-insensitive lookup; raise a clear, user-friendly error for unknown names
  - [ ] Default personality is **The Sage**
- [ ] `src/dm/context_builder.py`
  - [ ] Accept an optional `personality: DMPersonality | None` parameter
  - [ ] Inject `personality.system_prompt_directive` as a `## DM Personality` block immediately after the core DM persona section in the system prompt
- [ ] `src/main.py`
  - [ ] After campaign selection, display the personality menu (name, tone, verbosity, one-line description) using `rich`
  - [ ] Prompt the player to choose; default to **The Sage** on empty input
  - [ ] Pass the selected personality through to `DungeonMaster` and `context_builder`
- [ ] `src/interface/cli.py`
  - [ ] Add `/personality` command — display current personality, list all six options, prompt to switch; new personality takes effect from the next DM response
  - [ ] Display the active personality name in the session header banner
- [ ] Write unit tests in `tests/test_dm.py`
  - [ ] Test that each personality’s `system_prompt_directive` appears in the built system prompt
  - [ ] Test that `get_personality` raises a clear error for unknown names
  - [ ] Test that omitting a personality defaults to **The Sage**

---

## Phase 12 — Web Interface

A browser-based interface served locally via FastAPI. Supports all session commands, streams DM output word-by-word via WebSocket, and displays the DM avatar and campaign scene images.

- [ ] Add `fastapi`, `uvicorn[standard]`, `websockets` to `requirements.txt`
- [ ] `src/interface/web.py`
  - [ ] FastAPI app instance; serve static frontend at `GET /`
  - [ ] `GET /campaigns` — return available campaigns as JSON
  - [ ] `GET /personalities` — return all DM personalities as JSON
  - [ ] `POST /session/start` — accept `{campaign, personality}`, initialise a `DungeonMaster` session, return a `session_id`
  - [ ] `WS /session/{id}/chat` — WebSocket endpoint for the chat loop
    - [ ] Accept `{type: "message", content: "..."}` — player turn
    - [ ] Accept `{type: "interrupt"}` — cancel the current stream
    - [ ] Send `{type: "token", content: "..."}` — one streamed token
    - [ ] Send `{type: "roll", result: {...}}` — dice roll panel data
    - [ ] Send `{type: "scene_image", url: "..."}` — campaign image to display
    - [ ] Send `{type: "done"}` — stream complete
  - [ ] `POST /session/{id}/command` — handle `/status`, `/journal`, `/graph`, `/roll`, `/reset`, `/fullreset`, `/save`; return structured JSON
  - [ ] `GET /session/{id}/avatar` — return the avatar image for the current personality
  - [ ] `GET /campaign/{name}/image/{filename}` — serve images from `images/<campaign_name>/`; validate the resolved path stays within the images directory to prevent directory traversal
- [ ] `src/interface/static/`
  - [ ] `index.html` — single-page layout: chat panel (centre), DM avatar (left sidebar), campaign image display (right sidebar), collapsible character status panel
  - [ ] `style.css` — dark fantasy-themed styling with atmospheric colour palette
  - [ ] `app.js` — WebSocket client; renders streaming tokens word by word; handles interrupt button; displays dice roll panels; loads and fades in avatar and campaign images
- [ ] `src/main.py`
  - [ ] Add `--interface` CLI flag (`cli` / `web`, default `cli`)
  - [ ] When `--interface web`: start `uvicorn` on `localhost:8000` and open the browser automatically
- [ ] Write integration tests
  - [ ] Test `GET /campaigns` returns the expected list
  - [ ] Test `GET /personalities` returns all six personalities
  - [ ] Test `POST /session/start` returns a valid session ID with a mocked `DungeonMaster`
  - [ ] Test WebSocket message/response cycle with a mocked `DungeonMaster`
  - [ ] Test that `GET /campaign/{name}/image/{filename}` rejects path traversal attempts

---

## Phase 13 — DM Avatar & Campaign Image Display

Give the Dungeon Master a face. Each personality has a distinct avatar image. A display panel in the web interface shows scene images from the campaign’s `images/` directory at key narrative moments.

### DM Avatar

- [ ] Create `images/avatars/` directory; add placeholder avatar images for each personality (`the-sage.png`, `the-chronicler.png`, `the-bard.png`, `the-tactician.png`, `the-warden.png`, `the-mentor.png`) and a `default.png` fallback
- [ ] `src/dm/avatar.py`
  - [ ] `get_avatar_path(personality: DMPersonality) -> Path` — resolve `images/avatars/<personality-slug>.png`; fall back to `images/avatars/default.png` if not present
- [ ] Web interface: display the active avatar in the left sidebar; swap the image via a `{type: "avatar_updated"}` WebSocket message when the player changes personality mid-session
- [ ] CLI (stretch): detect terminal graphics protocol support (Sixel / Kitty) and render the avatar inline using `rich-pixels` or equivalent; fall back gracefully to displaying only the personality name

### Campaign Image Display

- [ ] Establish naming convention: campaign images stored under `images/<campaign_name>/`, filenames are lowercase slugs matching location or scene names (e.g., `tavern.png`, `forest-clearing.png`, `goblin-king.png`)
- [ ] `src/dm/image_resolver.py`
  - [ ] `resolve_scene_image(dm_response: str, campaign_name: str) -> Path | None` — tokenise the DM response, compare lowercase words against image filename stems in `images/<campaign_name>/`; return the best-matching path or `None`
  - [ ] Validate the resolved path is within `images/<campaign_name>/` to prevent directory traversal
- [ ] Web interface: when `resolve_scene_image` returns a path, send `{type: "scene_image", url: "/campaign/{name}/image/{filename}"}` over the WebSocket; the frontend fades in the image in the right display panel
- [ ] CLI: print a subtle hint line below the DM response (e.g., `  📷  forest-clearing.png`) when a matching image is found
- [ ] Write unit tests
  - [ ] Test `get_avatar_path` returns the correct path for each personality
  - [ ] Test `get_avatar_path` falls back to `default.png` for an unknown personality
  - [ ] Test `resolve_scene_image` matches a keyword from the DM response to an image filename
  - [ ] Test `resolve_scene_image` returns `None` when no image matches
  - [ ] Test `resolve_scene_image` rejects path traversal in the `campaign_name` parameter

---

## Phase 14 — Voice Interface

Add speech-to-text and text-to-speech so the player can speak to the DM and hear responses.

- [ ] Research and select STT library (candidates: `faster-whisper` (preferred — runs locally on GPU, no API needed), `openai-whisper`, `SpeechRecognition` + Google)
- [ ] Research and select TTS library (candidates: `edge-tts` (free, high quality, offline-friendly), `pyttsx3`, `openai` TTS API, `elevenlabs`)
- [ ] `src/interface/voice.py`
  - [ ] Implement push-to-talk mode (press a key to start recording, release to transcribe)
  - [ ] Implement voice activity detection (VAD) mode as an alternative
  - [ ] Transcribe audio to text and pass to the DM agent (reuse same `DungeonMaster.respond()` logic)
  - [ ] Synthesize DM response text to audio and play it back
  - [ ] Allow a configurable TTS voice persona (e.g., a deep, gravelly narrator voice)
- [ ] Add `--voice` CLI flag to launch in voice mode (`python src/main.py --voice`)
- [ ] Ensure voice and text modes are interchangeable — same session state, same memory
- [ ] Add `voice` dependencies to a separate `requirements-voice.txt` to keep the base install lightweight

---

## Phase 15 — Multi-Character & Party Support

Extend the system to support a full party of adventurers.

- [ ] Update `character.md` format to support multiple characters (list of character blocks)
- [ ] Update `context_builder.py` to include all party member sheets in the system prompt
- [ ] Update the CLI to let the player specify which character is taking an action (e.g., `Aldric: I search the room`)
- [ ] Update memory/journal to track individual character events
- [ ] Consider: multiplayer mode where different humans control different characters (out of scope for v1)

---

## Phase 16 — Campaign Authoring Tooling

Make it easy to create new campaigns in the required format.

- [ ] Write a `new-campaign` script that scaffolds a new campaign folder with template files
- [ ] Document the campaign file format in detail in `campaigns/README.md`
- [ ] Provide a complete worked example campaign (more than just a stub)
- [ ] Add a campaign linter that checks for:
  - [ ] Consistent creature names between `creature.md` and the campaign book
  - [ ] Character stats that are within valid D&D 5e ranges
  - [ ] Section headers in the campaign book that are parseable by the spoiler guard
- [ ] Consider: edition-agnostic linting where character stat validation is driven by the loaded rules edition

---

## Phase 17 — External LLM Provider Support

Add support for cloud-hosted LLM providers as an optional alternative to Ollama. The system must be fully functional with Ollama before this phase begins. All providers share the same `LLMProvider` interface — the DM agent requires no changes.

### Supported Providers

| Provider | Setting value | Notes |
|---|---|---|
| **OpenAI** | `openai` | GPT-4o, GPT-4o-mini, o1, o3 series; requires `OPENAI_API_KEY` |
| **Anthropic** | `anthropic` | Claude 3.5 Sonnet, Claude 3 Haiku; requires `ANTHROPIC_API_KEY` |
| **Google Gemini** | `gemini` | Gemini 1.5 Pro, Gemini 1.5 Flash; requires `GEMINI_API_KEY` |

Any provider that exposes an OpenAI-compatible `/v1/chat/completions` endpoint (e.g., Together AI, Groq, Fireworks) can be added with minimal effort by subclassing the OpenAI provider and pointing it at a custom `base_url`.

### Tasks

- [ ] `src/llm/openai_provider.py`
  - [ ] Implement `OpenAIProvider(LLMProvider)` using the `openai` Python SDK
  - [ ] Read `OPENAI_API_KEY` and `DM_MODEL` from settings
  - [ ] Pass `temperature`, `max_tokens` from settings
  - [ ] Implement `stream()` using `stream=True` on the OpenAI client (Phase 10 streaming contract)
  - [ ] Implement retry with exponential backoff for rate limit and transient API errors
- [ ] `src/llm/anthropic_provider.py`
  - [ ] Implement `AnthropicProvider(LLMProvider)` using the `anthropic` Python SDK
  - [ ] Read `ANTHROPIC_API_KEY` and `DM_MODEL` from settings
  - [ ] Map the `messages` list format to Anthropic's API (system message → `system` param; user/assistant turns → `messages`)
  - [ ] Implement `stream()` using the Anthropic streaming API
  - [ ] Implement retry with exponential backoff for rate limit and transient API errors
- [ ] `src/llm/gemini_provider.py`
  - [ ] Implement `GeminiProvider(LLMProvider)` using the `google-generativeai` Python SDK
  - [ ] Read `GEMINI_API_KEY` and `DM_MODEL` from settings
  - [ ] Map the `messages` list to Gemini's `ChatSession` / `generate_content` format
  - [ ] Implement `stream()` using `stream=True` on `generate_content`
  - [ ] Implement retry with exponential backoff for quota and transient API errors
- [ ] `src/llm/factory.py`
  - [ ] Update `create_provider()` to return the correct implementation for `openai`, `anthropic`, and `gemini`; raise a clear not-yet-supported error for any other value
- [ ] `src/config.py`
  - [ ] Add `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` — each validated only when the corresponding provider is selected
  - [ ] Add provider-specific model defaults (e.g., `gpt-4o` for OpenAI, `claude-3-5-sonnet-20241022` for Anthropic, `gemini-1.5-pro` for Gemini)
- [ ] Update `.env.example` with all three API key fields and their required provider values
- [ ] Update `src/main.py` to remove the Ollama-only guard; validate that the required API key for the selected provider is present before proceeding
- [ ] Update the `--provider` CLI flag (Phase 9) to accept `openai`, `anthropic`, and `gemini` as valid values
- [ ] Add optional provider SDK dependencies to `requirements.txt` with comments indicating they are only needed for the respective provider (`anthropic`, `google-generativeai`)
- [ ] Write unit tests in `tests/test_llm.py`
  - [ ] Test that `factory.create_provider` returns the correct class for each supported provider value
  - [ ] Mock the OpenAI SDK and test `OpenAIProvider.complete` and `stream` format messages correctly
  - [ ] Mock the Anthropic SDK and test `AnthropicProvider.complete` maps messages to the Anthropic format correctly
  - [ ] Mock the Gemini SDK and test `GeminiProvider.complete` maps messages to the Gemini format correctly
  - [ ] Test that a missing API key raises a clear, user-friendly error for each provider
  - [ ] Test that an unsupported provider value raises a clear, user-friendly error
- [ ] End-to-end smoke test: play the opening scene of the example campaign via each provider; confirm output parity with Ollama behaviour

---



## Stretch Goals & Future Ideas

- [ ] ~~Web UI~~ — now Phase 12
- [ ] Campaign marketplace — a way to share and download community campaigns
- [ ] ~~DM personality modes~~ — now Phase 11
- [ ] Visual dice rolling — replace the plain-text `rich` dice panel with an animated dice roll rendered in the terminal (ASCII art spinning die) or in a GUI window; the animation plays during the roll and settles on the final value
- [ ] ~~DM avatar~~ — now Phase 13
- [ ] Image generation — generate scene illustrations using an image model at key story moments
- [ ] Ambient audio — play background music and sound effects that match the current scene
- [ ] Export session as a story — convert the journey journal into a formatted narrative document
- [ ] Additional ruleset editions — extend the `rules/` system to support other TTRPG systems by adding an edition folder alongside `5e/`:
  - [ ] D&D 3.5e
  - [ ] D&D 4e
  - [ ] Pathfinder 1e
  - [ ] Pathfinder 2e
  - [ ] Other d20 systems (OSR, 13th Age, etc.)
- [ ] Rules RAG (Retrieval-Augmented Generation) — for very large rulesets (e.g., full Pathfinder), embed rule chunks and retrieve only the most relevant passages per turn rather than loading the full ruleset into context

---

## Known Decisions to Revisit

- **LLM provider abstraction**: ~~Decided~~ — implement a `src/llm/` layer with an abstract `LLMProvider` interface and an `OllamaProvider` (Phase 2). External cloud providers (OpenAI, Anthropic, Gemini) are added in Phase 17 once the system is proven locally. The factory reads `LLM_PROVIDER` from config. The DM agent always works through the interface and never references a specific backend.
- **Ollama model selection**: If `DM_MODEL` is unset and provider is `ollama`, query `ollama list` at startup and present an interactive picker. Store the selected model name in the running config (not written back to `.env`).
- **Rules in context strategy**: For 5e, the full SRD rules text is modest enough to fit in a large context window alongside a campaign. Start with full-rules inclusion per turn. For larger rulesets (Pathfinder, etc.) or smaller context window models, implement RAG over the rules directory as a follow-up.
- **NarrativeState tracking**: The context builder needs to know the current state (combat/exploration/social) to select relevant rules sections. Decide whether this is tracked explicitly by the DM agent or inferred from the LLM's last response.
- **Dice roll interception approach**: The LLM signals rolls via `[ROLL: <label> <NdX>[+modifier]]` tags embedded in its raw response. The DM agent strips the tags, performs real rolls via the dice engine, and re-prompts the LLM with the resolved results so it can narrate from actual outcomes. This two-pass approach keeps roll results honest while preserving narrative quality. Decide whether the second LLM call is always made or only when rolls occurred.
- **Rules authority vs. narrative flexibility**: Decide the default enforcement level — strict RAW, RAW with common sense exceptions, or narrative-first. Expose this as a `DM_RULES_MODE` config setting (`strict` / `flexible`).
- **Campaign book sectioning**: The spoiler guard needs a consistent way to detect section boundaries in the `.txt` file. Decide on a required format (e.g., lines starting with `##` or `SCENE:`) before writing any real campaigns.
- **Context window management**: For long campaigns, the full campaign book will exceed the model's context window. Decide whether to use chunking + retrieval (RAG) or summarization. RAG is likely the right long-term answer.
- **Memory system**: ~~Decided~~ — [Graphiti](https://github.com/getzep/graphiti) + Kuzu (Phase 5). Graphiti is a temporal context graph engine that automatically extracts entities and relationships from each DM response episode. Kuzu is the local-first embeddable graph backend (no server). Short-term context is the last N session messages managed by `SessionStore`. At each turn, `MemoryManager.get_context()` calls `graphiti.search()` (hybrid semantic + keyword + graph traversal) and injects the result into the system prompt. The main risk is structured-output quality with smaller Ollama models — recommend a model with strong tool-calling support (e.g. `llama3.1:8b` or `qwen2.5:7b`).
- **Progress tracking granularity**: Decide whether "progress" is tracked by scene number, by a keyword in the campaign book, or by a running percentage. Scene headers are the cleanest option.
