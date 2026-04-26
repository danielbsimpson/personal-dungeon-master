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

- [ ] `src/rules/loader.py`
  - [ ] Read `GAME_EDITION` from settings (default `5e`)
  - [ ] Scan `RULES_DIR/<edition>/` and load all `.md` files into a `RulesReference` dataclass
  - [ ] Each rule file maps to a named section (e.g., `combat`, `spellcasting`, `conditions`)
  - [ ] Raise a clear error if the edition folder does not exist or contains no rule files
- [ ] `src/rules/reference.py`
  - [ ] `get_all_rules(ref: RulesReference) -> str` — return the full rules text concatenated (for smaller models or short campaigns)
  - [ ] `get_relevant_rules(ref: RulesReference, context: str) -> str` — return only the sections most relevant to the current narrative context (e.g., if combat is active return `combat.md` + `conditions.md`; if a spell is being cast return `spellcasting.md`)
  - [ ] Define a `NarrativeState` enum (`EXPLORATION`, `COMBAT`, `SOCIAL`, `REST`) to drive section selection
  - [ ] Always include `core.md` as a baseline in every prompt
- [ ] Populate `rules/5e/` rule files with accurate 5e SRD content:
  - [ ] `core.md` — ability scores & modifiers, proficiency bonus progression, skill list & governing abilities, ability checks, saving throws, passive scores, advantage/disadvantage, hidden rules
  - [ ] `combat.md` — initiative, surprise, turn structure, action/bonus action/reaction/free action/movement, attack rolls, damage rolls, critical hits, two-weapon fighting, grapple/shove, ranged attack rules, cover, death saving throws, stabilisation, mounted combat, underwater combat
  - [ ] `conditions.md` — full definitions for all 15 conditions with mechanical effects
  - [ ] `spellcasting.md` — spell slots by class, cantrips, concentration rules, ritual casting, verbal/somatic/material components, spell attack rolls, spellcasting ability modifier & save DC formula, counterspell/dispel rules, spell school descriptions
  - [ ] `equipment.md` — weapon properties (finesse, versatile, thrown, reach, etc.), weapon tables, armor categories & AC, shields, donning/doffing armor, encumbrance (standard & variant), silvered weapons, improvised weapons
- [ ] Write unit tests in `tests/test_rules.py`
  - [ ] Test that the loader correctly reads all files in `rules/5e/`
  - [ ] Test that `get_relevant_rules` returns the combat section when `NarrativeState` is `COMBAT`
  - [ ] Test that `core.md` is always included regardless of state
  - [ ] Test that an unknown edition raises a clear error

---

## Phase 5 — Memory System (Graph RAG)

Build a Graph RAG memory system so the DM retains rich, structured knowledge of the player's journey across sessions and retrieves only the most relevant context per turn — keeping prompts compact regardless of campaign length.

### Data model

- **Nodes** represent named entities in the world: `Character`, `Location`, `Creature`, `Item`, `Faction`, `Event`, `Revelation`
- **Edges** represent directed relationships between entities: `MET`, `VISITED`, `DEFEATED`, `OWNS`, `KNOWS`, `IS_PART_OF`, `CAUSED`, `DISCOVERED`, `ALLIED_WITH`, `HOSTILE_TO`
- Every node and edge carries a `turn` timestamp (integer message index) and an optional `notes` string
- Storage: `memory/<campaign_name>/graph.json` — a plain JSON adjacency list; swappable to SQLite in a later phase
- Short-term context: `memory/<campaign_name>/session.json` — the last N raw `{role, content}` messages (configurable window; default 20)
- Progress pointer: `memory/<campaign_name>/progress.json` — current scene index

### Tasks

- [ ] Add `networkx` to `requirements.txt` and `pyproject.toml`
- [ ] `src/dm/memory/graph_store.py`
  - [ ] Define `EntityType` and `RelationType` enums
  - [ ] Define `Entity` dataclass: `id` (slug), `type: EntityType`, `label: str`, `notes: str`, `first_seen_turn: int`
  - [ ] Define `Relationship` dataclass: `source_id`, `relation: RelationType`, `target_id`, `notes: str`, `turn: int`
  - [ ] `GraphStore` class backed by a `networkx.DiGraph`
    - [ ] `add_entity(entity: Entity)` — upsert; update `notes` if the entity already exists
    - [ ] `add_relationship(rel: Relationship)` — add directed edge; skip duplicate relation on same turn
    - [ ] `get_subgraph(seed_ids: list[str], depth: int = 2) -> str` — BFS from seed nodes up to `depth` hops; serialise the result as a structured text block suitable for LLM context
    - [ ] `save(path: Path)` / `load(path: Path)` — persist/restore the full graph as JSON
- [ ] `src/dm/memory/extractor.py`
  - [ ] `extract_entities_and_relations(text: str, provider: LLMProvider, turn: int) -> tuple[list[Entity], list[Relationship]]`
    - [ ] Build a terse extraction prompt asking the LLM to return JSON listing new entities and relationships found in `text`
    - [ ] Parse and validate the JSON response; silently skip malformed entries
    - [ ] Called once per DM response — cheap single-pass extraction
- [ ] `src/dm/memory/session_store.py`
  - [ ] `SessionStore` class scoped to a campaign path
  - [ ] `load()` — read `session.json` from disk on startup; return empty list if absent
  - [ ] `append(role: str, content: str)` — add message and persist; trim to the last N messages (default 20, configurable via `SESSION_WINDOW` setting)
  - [ ] `messages() -> list[dict]` — return the current window as `[{role, content}, ...]`
  - [ ] `clear()` — wipe the session window (keeps graph)
- [ ] `src/dm/memory/manager.py`
  - [ ] `MemoryManager` class — composes `GraphStore`, `SessionStore`, and progress pointer
  - [ ] `load(campaign_name: str)` — load graph, session, and progress from disk
  - [ ] `record_turn(player_input: str, dm_response: str, provider: LLMProvider)` — append both messages to session, run extractor on dm_response, upsert graph
  - [ ] `get_context(current_text: str) -> str` — extract seed entity names from `current_text`, retrieve the relevant subgraph, return a formatted block for the system prompt
  - [ ] `advance_progress(to_section: int)` — update and persist the scene pointer
  - [ ] `campaign_progress` property — current section index
  - [ ] `reset_session()` — wipe session window only; graph is preserved
  - [ ] `full_reset()` — wipe session window and graph (new run of the campaign)
- [ ] Add `SESSION_WINDOW` (default `20`) to `src/config.py` and `.env.example`
- [ ] Write unit tests in `tests/test_memory.py`
  - [ ] Test `GraphStore.add_entity` upserts correctly (duplicate id updates notes)
  - [ ] Test `GraphStore.get_subgraph` returns nodes within the specified depth and no further
  - [ ] Test `SessionStore` trims to the configured window size
  - [ ] Test `SessionStore` persists and reloads correctly across instances
  - [ ] Test `MemoryManager.advance_progress` is monotonically increasing (cannot go backwards)
  - [ ] Mock the `LLMProvider` and test `extract_entities_and_relations` handles malformed JSON without raising

---

## Phase 6 — DM Agent Core

Build the LLM-powered Dungeon Master that reads context and generates responses.

- [ ] `src/dm/spoiler_guard.py`
  - [ ] Accept the full campaign book text and a current progress pointer
  - [ ] Return only the portion of the campaign book up to and including the current section
  - [ ] Implement section detection (scenes, chapters, or encounter headers in the `.txt` file)
  - [ ] Ensure the revealed window grows as progress advances but never shrinks
- [ ] `src/dm/context_builder.py`
  - [ ] `build_system_prompt(campaign: ParsedCampaign, rules: RulesReference, memory: MemoryManager) -> str`
    - [ ] Include DM persona instructions and behavioral rules
    - [ ] Include game edition identifier and relevant rules sections (via `rules/reference.py`)
    - [ ] Include campaign summary
    - [ ] Include character sheet (formatted clearly for the LLM)
    - [ ] Include creature reference
    - [ ] Include the revealed portion of the campaign book (via `spoiler_guard`)
    - [ ] Include the retrieved graph memory context (via `memory.get_context()`) — entities and relationships relevant to the current turn, not a full history dump
    - [ ] Append the short-term session window (last N messages) as prior conversation context
  - [ ] Keep system prompt within a configurable token budget (avoid context overflow)
  - [ ] Log a warning if the context approaches the model's context window limit
  - [ ] Update `NarrativeState` passed to `get_relevant_rules` based on current session state (e.g., swap to `COMBAT` when combat begins)
- [ ] `src/dm/dungeon_master.py`
  - [ ] `DungeonMaster` class — wraps the `LLMProvider` (injected via factory), `DiceRoller`, context builder, and memory manager
  - [ ] `start_campaign()` — deliver the opening narration (introduction to the adventure)
  - [ ] `respond(player_input: str) -> str`:
    1. Build context and send to LLM to get a raw response
    2. Scan raw response for `[ROLL: <type> <die>[+modifier]]` tags using regex
    3. For each tag, call the dice engine with the correct `Die` and modifier; replace the tag with the real result inline
    4. Inject resolved roll results as a system message and call the LLM once more to generate the narrative from actual results
    5. Return the final narrated response
  - [ ] After each response, call `memory.record_turn(player_input, dm_response, provider)` — this appends to the session window and updates the knowledge graph in one call
  - [ ] After each response, check if the response implies a new section has been reached and advance the progress pointer
  - [ ] Implement retry logic with exponential backoff for API errors (delegates to provider implementation)
- [ ] Write unit tests in `tests/test_dm.py`
  - [ ] Mock the `LLMProvider` interface and test that the system prompt is built correctly
  - [ ] Test that spoiler guard correctly limits campaign content
  - [ ] Test that progress advances after appropriate responses

---

## Phase 7 — Dice Engine

Build the dice rolling engine that gives the DM real, unbiased randomness for every roll in the game.

- [ ] `src/dice/die.py`
  - [ ] Define `Die` enum with all standard tabletop die types: `D4`, `D6`, `D8`, `D10`, `D12`, `D20`, `D100`
  - [ ] Each member stores its number of faces as an integer value (e.g., `D20 = 20`)
  - [ ] Define `RollResult` dataclass: `die`, `rolls: list[int]`, `modifier: int`, `total: int`, `label: str` (e.g., `"attack"`)
  - [ ] Define `RollRequest` dataclass: `label`, `die`, `count` (default 1), `modifier` (default 0), `advantage: bool`, `disadvantage: bool`
- [ ] `src/dice/roller.py`
  - [ ] Seed `random.Random` from `secrets.randbits(128)` at module load — ensures cryptographically seeded randomness that the LLM cannot predict
  - [ ] `roll(req: RollRequest) -> RollResult`:
    - [ ] Roll `req.count` dice, each constrained to `[1, req.die.value]`
    - [ ] If `advantage=True`: roll twice, keep the highest single die result
    - [ ] If `disadvantage=True`: roll twice, keep the lowest single die result
    - [ ] Apply `modifier` to the sum of rolls to produce `total`
    - [ ] `total` is always at minimum 1 (per 5e rules for damage rolls)
  - [ ] `parse_roll_tags(text: str) -> list[RollRequest]` — extract all `[ROLL: <label> <NdX>[+/-modifier>]]` tags from an LLM response string using regex
  - [ ] `substitute_rolls(text: str, results: list[RollResult]) -> str` — replace each `[ROLL: ...]` tag in the text with its formatted result string
  - [ ] `format_result(result: RollResult) -> str` — produce a human-readable result string for terminal display (e.g., `✨ Attack Roll [d20+5]: rolled 14 + 5 = **19**`)
- [ ] Write unit tests in `tests/test_dice.py`
  - [ ] Test that `Die` enum values match their face counts (D6.value == 6, D20.value == 20, etc.)
  - [ ] Test that `roll()` always returns a value within the valid range `[1 + modifier, die.value + modifier]` (allowing for negative modifiers flooring at 1)
  - [ ] Test that `roll_multiple(2, Die.D6)` returns exactly 2 individual rolls
  - [ ] Test that advantage always returns a total ≥ disadvantage given the same seed
  - [ ] Test that `parse_roll_tags` correctly extracts die type, count, and modifier from tag strings
  - [ ] Test that `substitute_rolls` replaces all tags in a string without mutating non-tag content
  - [ ] Test 1000 rolls on each die type — assert min/max are within bounds and distribution is roughly uniform

---

## Phase 8 — Text Interface & Main Loop

Wire everything together into a working text-based adventure session.

- [ ] `src/interface/cli.py`
  - [ ] Print a styled welcome banner using `rich`
  - [ ] Display the selected campaign name and a brief summary before the adventure begins
  - [ ] Render DM responses with atmospheric formatting (markdown-aware, colored output)
  - [ ] Accept player input via `input()` prompt styled with `rich`
  - [ ] Handle special commands:
    - [ ] `/quit` or `/exit` — save and exit gracefully
    - [ ] `/journal` — render a human-readable view of the knowledge graph (entities grouped by type, with their relationships) using `rich`
    - [ ] `/status` — display current character stats and inventory
    - [ ] `/save` — explicitly save session state (auto-save also happens after every message)
    - [ ] `/reset` — confirm and wipe the session window; knowledge graph is preserved
    - [ ] `/fullreset` — confirm and wipe both session window and knowledge graph (restart the campaign from scratch)
    - [ ] `/roll <expression>` — let the player roll their own dice at any time (e.g., `/roll d20+3`, `/roll 2d6`); display the result in the same styled format as DM rolls
    - [ ] `/graph <entity>` — look up a specific entity in the knowledge graph and display its relationships
    - [ ] `/help` — list available commands
  - [ ] After each DM response, if dice were rolled, render each `RollResult` in a styled panel (using `rich`) showing: die type badge, individual roll values, modifier, and bold total — displayed between the player's input and the DM's narration
  - [ ] Handle `KeyboardInterrupt` (Ctrl+C) gracefully — save and exit
- [ ] `src/main.py`
  - [ ] Entry point: load config, run campaign selector, load rules for configured edition, initialize LLM provider via factory, initialize `DungeonMaster`, start chat loop
  - [ ] Verify Ollama is running at startup; if `DM_MODEL` is not set, show model picker
  - [ ] Check that all required environment variables are set before proceeding
  - [ ] Print a helpful error message if the `campaigns/` directory is empty or missing
  - [ ] Print a helpful error message if the `rules/<edition>/` directory is missing
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

## Phase 10 — Voice Interface

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

## Phase 11 — Multi-Character & Party Support

Extend the system to support a full party of adventurers.

- [ ] Update `character.md` format to support multiple characters (list of character blocks)
- [ ] Update `context_builder.py` to include all party member sheets in the system prompt
- [ ] Update the CLI to let the player specify which character is taking an action (e.g., `Aldric: I search the room`)
- [ ] Update memory/journal to track individual character events
- [ ] Consider: multiplayer mode where different humans control different characters (out of scope for v1)

---

## Phase 12 — Campaign Authoring Tooling

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

## Phase 13 — OpenAI Provider Integration

Add support for cloud-hosted OpenAI models. The system must be fully functional with Ollama before this phase begins.

- [ ] `src/llm/openai_provider.py`
  - [ ] Implement `OpenAIProvider(LLMProvider)` using the `openai` Python SDK
  - [ ] Read `OPENAI_API_KEY` and `DM_MODEL` from settings
  - [ ] Pass `temperature`, `max_tokens` from settings
  - [ ] Implement retry with exponential backoff for rate limit and transient API errors
- [ ] Update `src/llm/factory.py` to support `LLM_PROVIDER=openai` — return `OpenAIProvider`
- [ ] Update `src/main.py` to handle OpenAI at startup: verify `OPENAI_API_KEY` is present; remove the Ollama-only guard
- [ ] Confirm `src/config.py` `OPENAI_API_KEY` validation triggers correctly when `LLM_PROVIDER=openai`
- [ ] Update the `--provider` CLI flag (Phase 9) to accept `openai` as a valid value
- [ ] Write unit tests in `tests/test_llm.py`
  - [ ] Test that `factory.create_provider` returns `OpenAIProvider` when `LLM_PROVIDER=openai`
  - [ ] Mock the OpenAI SDK and test `OpenAIProvider.complete` formats messages correctly
  - [ ] Test that a missing `OPENAI_API_KEY` raises a clear, user-friendly error
- [ ] End-to-end smoke test: play the opening scene of the example campaign via OpenAI; confirm output parity with Ollama behaviour

---

## Stretch Goals & Future Ideas

- [ ] Web UI — a browser-based chat interface instead of the terminal
- [ ] Campaign marketplace — a way to share and download community campaigns
- [ ] DM personality modes — choose between strict RAW (Rules As Written), narrative-first, or beginner-friendly
- [ ] Visual dice rolling — replace the plain-text `rich` dice panel with an animated dice roll rendered in the terminal (ASCII art spinning die) or in a GUI window; the animation plays during the roll and settles on the final value
- [ ] DM avatar — a character portrait representing the Dungeon Master displayed alongside narration in a GUI or web UI; could be static art or a lightly animated idle loop; the avatar reacts to narrative state (combat = menacing expression, calm scene = neutral/inviting)
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

- **LLM provider abstraction**: ~~Decided~~ — implement a `src/llm/` layer with an abstract `LLMProvider` interface and an `OllamaProvider` (Phase 2). `OpenAIProvider` is added in Phase 13 once the system is proven locally. The factory reads `LLM_PROVIDER` from config. The DM agent always works through the interface and never references a specific backend.
- **Ollama model selection**: If `DM_MODEL` is unset and provider is `ollama`, query `ollama list` at startup and present an interactive picker. Store the selected model name in the running config (not written back to `.env`).
- **Rules in context strategy**: For 5e, the full SRD rules text is modest enough to fit in a large context window alongside a campaign. Start with full-rules inclusion per turn. For larger rulesets (Pathfinder, etc.) or smaller context window models, implement RAG over the rules directory as a follow-up.
- **NarrativeState tracking**: The context builder needs to know the current state (combat/exploration/social) to select relevant rules sections. Decide whether this is tracked explicitly by the DM agent or inferred from the LLM's last response.
- **Dice roll interception approach**: The LLM signals rolls via `[ROLL: <label> <NdX>[+modifier]]` tags embedded in its raw response. The DM agent strips the tags, performs real rolls via the dice engine, and re-prompts the LLM with the resolved results so it can narrate from actual outcomes. This two-pass approach keeps roll results honest while preserving narrative quality. Decide whether the second LLM call is always made or only when rolls occurred.
- **Rules authority vs. narrative flexibility**: Decide the default enforcement level — strict RAW, RAW with common sense exceptions, or narrative-first. Expose this as a `DM_RULES_MODE` config setting (`strict` / `flexible`).
- **Campaign book sectioning**: The spoiler guard needs a consistent way to detect section boundaries in the `.txt` file. Decide on a required format (e.g., lines starting with `##` or `SCENE:`) before writing any real campaigns.
- **Context window management**: For long campaigns, the full campaign book will exceed the model's context window. Decide whether to use chunking + retrieval (RAG) or summarization. RAG is likely the right long-term answer.
- **Memory system**: ~~Decided~~ — Graph RAG approach (Phase 5). Long-term memory is stored as a knowledge graph of entities and relationships (`graph.json`). Short-term context is the last N session messages (`session.json`). At each turn, `MemoryManager.get_context()` retrieves the relevant subgraph via BFS from seed entities in the current text and injects it into the system prompt — no full-history dump, no summarisation needed.
- **Progress tracking granularity**: Decide whether "progress" is tracked by scene number, by a keyword in the campaign book, or by a running percentage. Scene headers are the cleanest option.
