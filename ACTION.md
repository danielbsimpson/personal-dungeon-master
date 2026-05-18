# ACTION PLAN: Player-Driven Dice Roll Integration

## Overview

The dice engine (`die.py`, `roller.py`) is already implemented and partially wired into
the system, but the current flow is **DM-resolved**: when the LLM produces a `[ROLL: ...]`
tag, the server rolls the dice automatically and substitutes the result before the player
ever sees it. The goal of this plan is to **invert control** — the LLM requests a roll,
the player is shown what to roll and presses Enter to roll (or rolls physically), the
result is fed back into a follow-up LLM call, and the LLM then narrates the outcome based
on the actual value. Different DM personalities will interpret the same roll number
differently.

---

## Current State (Baseline)

| Component | What it does today |
|---|---|
| `src/dice/die.py` | Defines `Die`, `RollRequest`, `RollResult` data classes |
| `src/dice/roller.py` | `roll()`, `parse_roll_tags()`, `substitute_rolls()`, `format_result()` |
| `src/dm/dungeon_master.py` `respond()` | Calls `parse_roll_tags()` + `substitute_rolls()` **after** the LLM responds — player never sees the roll |
| `src/interface/cli.py` | `print_roll_results()` renders rolls already substituted; `/roll` command is standalone only |
| `src/dm/context_builder.py` | DM persona instructs LLM to emit `[ROLL: label NdX+mod]` tags |

**Gap**: rolls are resolved silently server-side. The player has no agency and no
information about what was rolled or why.

---

## Desired Flow (Per Turn)

```
Player types action
        │
        ▼
DM LLM responds with narrative + [ROLL: attack 1d20+5] tag(s)
        │
        ▼
CLI detects pending roll(s) — pauses narration
        │
        ▼
CLI displays: "🎲 Roll for Attack — d20+5. Press Enter to roll..."
        │
        ▼
Player presses Enter (or types a manual value)
        │
        ▼
die.py executes the roll; result shown in styled panel
        │
        ▼
Result injected back into LLM as a follow-up user message
  (e.g. "Roll result — Attack: rolled 14 + 5 = 19")
        │
        ▼
DM LLM narrates outcome based on roll value + personality + context
        │
        ▼
Full narration rendered; turn complete
```

---

## Implementation Plan

### Step 1 — Split the `respond()` Turn Into Two Phases

**File:** `src/dm/dungeon_master.py`

Currently `respond()` is a single LLM call. It needs to become a two-phase call
when the LLM's first response contains `[ROLL: ...]` tags.

**Changes:**

1. Extract a new private method `_first_pass(player_input)` that calls the LLM and
   returns the raw response string (before roll substitution). This is essentially the
   existing `respond()` body up to the roll-detection block.

2. Extract a new private method `_second_pass(roll_results, first_response)` that
   assembles a follow-up message containing:
   - The DM's first response (with roll tags replaced by placeholder labels, not yet
     narrated outcomes).
   - A synthetic "user" message listing all roll results:
     ```
     [Roll results]
     • Attack: rolled 14 + 5 = 19
     • Damage: rolled [3, 4] = 7
     ```
   - Instructs the LLM to now narrate what those results mean in the story.

3. Update the public `respond()` to orchestrate both phases:
   ```
   first_response = await _first_pass(player_input)
   roll_requests = parse_roll_tags(first_response)
   if roll_requests:
       # yield/return roll_requests to caller so CLI can prompt the player
       # (see Step 2 for the handoff mechanism)
       roll_results = <provided by CLI after player interaction>
       final_response = await _second_pass(roll_results, first_response)
   else:
       final_response = first_response
   ```

4. Add a new public method `respond_with_rolls(player_input, roll_results)` that
   takes externally-supplied `RollResult` objects (from the CLI after player
   interaction) and calls `_second_pass()`. This keeps `respond()` backward-compatible
   for tests.

5. Also update `respond_stream()` (Phase 10 method) in the same way so streaming mode
   also supports the two-phase flow.

---

### Step 2 — New CLI Handoff: Detect Rolls and Prompt the Player

**File:** `src/interface/cli.py`

The CLI's main game loop calls `_stream_dm_response()` then `print_roll_results()`.
The loop must be updated so that when the DM's first-pass response contains roll tags,
the session pauses and prompts the player.

**Changes:**

1. Add a new function `prompt_player_rolls(roll_requests) -> list[RollResult]`:
   - Iterates over each `RollRequest`.
   - Prints a styled panel:
     ```
     ┌─────────────────────────────────────┐
     │  🎲  Roll for Attack — 1d20+5       │
     │  Press Enter to roll, or type value │
     └─────────────────────────────────────┘
     ```
   - Reads player input: if blank → `roll(req)` (auto-roll); if a digit → build a
     `RollResult` with `total=int(input)` (manual/physical dice support); if invalid
     → re-prompt once.
   - Calls `print_roll_results([result])` to display immediately after each roll.
   - Returns the list of all `RollResult` objects in tag order.

2. Add a new async function `_stream_dm_response_two_phase(dm, player_input)`:
   - Calls `dm.first_pass(player_input)` to get the first LLM response.
   - If the response has no roll tags → renders it normally (same as today).
   - If the response has roll tags:
     - Displays the DM's partial narration up to the first roll request (everything
       before the first `[ROLL:` tag) so the player has context.
     - Calls `prompt_player_rolls(roll_requests)` to get player-executed results.
     - Calls `dm.respond_with_rolls(player_input, roll_results)` for the second pass.
     - Streams and renders the final narration.

3. Replace the call to `_stream_dm_response(dm, raw)` in `run_session()` with the new
   `_stream_dm_response_two_phase(dm, raw)`.

---

### Step 3 — Update the System Prompt Directive for Two-Phase Rolls

**File:** `src/dm/context_builder.py`

The DM persona section already instructs the LLM to emit `[ROLL: ...]` tags. Update it
to communicate the new two-phase contract so the LLM knows its first response is a
"setup" pass and the second is the "resolution" pass.

**Changes to `_DM_PERSONA`:**

Replace the existing roll tag instruction with a more explicit two-phase description:

```
- When a player action warrants a dice check (attack, skill check, saving throw,
  persuasion, deception, stealth, etc.), you MUST request the roll in your response
  by emitting exactly one [ROLL: <label> <NdX+modifier>] tag per check.
  Examples: [ROLL: attack 1d20+5]  [ROLL: persuasion 1d20+3]  [ROLL: damage 2d6+2]
- After the roll tag, STOP narrating the outcome. Do not describe whether the action
  succeeds or fails — that narration comes in your NEXT response, once the roll result
  is provided.
- In your follow-up response (when given roll results), narrate what the number means
  in the story. A high roll is a success; a low roll is a failure or complication.
  The threshold for success depends on the situation and the opposing difficulty.
  React to the result in character — your personality determines how you describe
  triumph and failure.
```

---

### Step 4 — Personality-Aware Roll Outcome Narration

**File:** `src/dm/personality.py` and `src/dm/context_builder.py`

Each personality should have an explicit instruction for how it narrates roll outcomes.
This is the mechanism that makes different personalities react differently to the same
number.

**Changes:**

1. Add a `roll_outcome_directive` field to `DMPersonality`:
   ```python
   roll_outcome_directive: str
   ```

2. Populate it for each of the existing six personalities. Examples:

   | Personality | Low roll (1–7) | High roll (18–20) |
   |---|---|---|
   | The Sage | Describes setback plainly, suggests path forward | Praises cleanly, advances narrative |
   | The Chronicler | Vivid prose of failure, sensory detail of pain | Lyrical triumph, extended description |
   | The Bard | Rhymes the fumble humorously, lightens mood | Celebrates with theatrical flourish |
   | The Warlord | Blunt: "You failed. Adapt." | Brief commendation, focus on next threat |
   | The Trickster | Mocks gently, adds a twist complication | Reluctant praise with a sting in the tail |
   | The Oracle | Frames failure as fate, hints at deeper meaning | Frames success as destiny unfolding |

3. In `build_system_prompt()`, append `personality.roll_outcome_directive` after the
   existing `personality.system_prompt_directive` block.

---

### Step 5 — Second-Pass Context Injection

**File:** `src/dm/dungeon_master.py` (`_second_pass()` method)

The second LLM call needs full context to interpret the roll. The message list must be:

```
[system]  Full system prompt (same as first pass — includes personality, campaign,
          character sheet, creatures, rules, memory context)
[user]    Original player action  ("I try to persuade the guard")
[assistant]  First-pass DM response with roll tag replaced by a neutral placeholder
             e.g. "[Awaiting roll result for: Persuasion 1d20+3]"
[user]    Roll resolution message:
          "Roll results:\n• Persuasion: rolled 11 + 3 = 14\n\nPlease narrate the outcome."
```

The `[Awaiting roll result]` substitution prevents the model from seeing a
pre-resolved result in its own prior turn while still conveying what was requested.

**Helper needed in `roller.py`:**
- `placeholder_rolls(text, roll_requests) -> str`: replaces each `[ROLL: ...]` tag
  with `[Awaiting roll: <label>]` — used to sanitise the first-pass response before
  it is fed back as the assistant turn.

---

### Step 6 — Manual Roll Support (Physical Dice)

**File:** `src/interface/cli.py` (`prompt_player_rolls()`)

Players using physical dice should be able to type their roll total instead of pressing
Enter for an auto-roll. The prompt for each roll should make this explicit:

```
🎲 Roll for Persuasion — 1d20+3
   Press Enter to roll digitally, or type your result:  _
```

Behaviour:
- Blank input → auto-roll via `roll(req)`.
- Integer input → construct a `RollResult` with the provided `total`, `modifier=req.modifier`,
  `rolls=[]` (empty, since no individual dice values are known), `label=req.label`,
  `die=req.die`. The system still feeds the total into the LLM correctly.
- Non-integer, non-blank input → print a one-line error and re-prompt.

---

### Step 7 — Session Memory: Record Roll Context

**File:** `src/dm/dungeon_master.py` and `src/dm/memory/session_store.py`

Currently `record_turn()` saves the player input and DM response. After this change,
turns with dice rolls should also persist the roll context so memory retrieval gives
the LLM useful history.

**Changes:**

1. Extend the turn recording in `respond_with_rolls()` to include roll results as part
   of the assistant message stored in the session window:
   ```
   dm_response_with_rolls_footer = dm_response + "\n\n---\n" + roll_summary
   ```
   where `roll_summary` is e.g. `"[Rolls this turn: Attack 19, Damage 7]"`.

2. This ensures that when the session window is included in future turns, the LLM has
   context about what checks were made and their outcomes.

---

### Step 8 — Tests

**File:** `tests/test_dice.py` and `tests/test_dm.py`

**New tests needed:**

1. `test_dice.py`:
   - `test_parse_roll_tags_multiple()` — verify multiple tags in one response are all
     parsed in order.
   - `test_placeholder_rolls()` — verify `placeholder_rolls()` replaces tags correctly
     and does not corrupt surrounding text.

2. `test_dm.py`:
   - `test_respond_two_phase_with_rolls()` — mock LLM to return a first-pass response
     with a `[ROLL: ...]` tag; supply a `RollResult`; assert the second-pass message
     list has the correct structure (system → user → assistant with placeholder →
     user with roll result).
   - `test_respond_no_rolls_unchanged()` — verify that when no roll tags are present,
     the existing single-pass path is used and the output is identical to current
     behaviour.
   - `test_personality_directive_in_second_pass()` — assert that the
     `roll_outcome_directive` for the active personality appears in the system prompt
     sent in the second pass.

---

## File Change Summary

| File | Change type | Description |
|---|---|---|
| `src/dm/dungeon_master.py` | Modify | Split `respond()` into `_first_pass()` + `_second_pass()`; add `respond_with_rolls()` |
| `src/dm/context_builder.py` | Modify | Update `_DM_PERSONA` with two-phase roll instructions; inject `roll_outcome_directive` |
| `src/dm/personality.py` | Modify | Add `roll_outcome_directive` field to `DMPersonality`; populate for all six personalities |
| `src/dice/roller.py` | Modify | Add `placeholder_rolls()` helper |
| `src/interface/cli.py` | Modify | Add `prompt_player_rolls()`; add `_stream_dm_response_two_phase()`; update `run_session()` |
| `tests/test_dice.py` | Modify | Add `test_parse_roll_tags_multiple`, `test_placeholder_rolls` |
| `tests/test_dm.py` | Modify | Add two-phase turn tests and personality directive tests |

---

## Out of Scope for This Plan

- Changing the dice engine itself (`die.py`, the core `roll()` function) — it is
  already correct and sufficient.
- Changing how `/roll` (manual command) works — it is independent of the game loop.
- Automatic difficulty class (DC) tracking — the LLM determines success/failure
  narratively based on the roll total; no numeric DC lookup system is added here.
- Initiative order / turn-based combat sequencing — that is a separate feature.
