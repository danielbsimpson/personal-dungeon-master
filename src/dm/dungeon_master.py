"""
Dungeon Master agent — orchestrates LLM calls, context building, memory
recording, and (from Phase 7) dice substitution.

Usage
-----
    dm = DungeonMaster(
        llm=create_provider(settings),
        campaign=parse_campaign(campaign),
        rules=load_rules(settings),
        memory=memory_manager,   # after manager.load(campaign_name)
        settings=settings,
    )
    opening = await dm.start_campaign()
    # ...main loop...
    response = await dm.respond(player_input)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from src.campaign.parser import ParsedCampaign
from src.config import Settings
from src.config import settings as _default_settings
from src.dice.roller import parse_roll_tags, roll, substitute_rolls
from src.dm.context_builder import build_system_prompt, detect_narrative_state
from src.dm.memory.manager import MemoryManager
from src.dm.personality import DEFAULT_PERSONALITY, DMPersonality
from src.llm.base import LLMProvider
from src.rules.loader import RulesReference
from src.rules.reference import NarrativeState

log = logging.getLogger(__name__)


class DungeonMaster:
    """
    The LLM-powered Dungeon Master agent.

    Orchestrates context building, LLM inference, memory recording, and
    spoiler-guard advancement.  Dice tag substitution is wired in Phase 7 —
    until then, any ``[ROLL: ...]`` tags produced by the LLM are passed
    through verbatim in the response string.

    Parameters
    ----------
    llm:
        Initialised :class:`~src.llm.base.LLMProvider`.
    campaign:
        Fully parsed campaign data.
    rules:
        Loaded rules reference for the configured edition.
    memory:
        Initialised :class:`~src.dm.memory.manager.MemoryManager` —
        ``load()`` must be called before the first ``respond()`` call.
    settings:
        Application settings.  Defaults to the module-level singleton.
    personality:
        Active :class:`~src.dm.personality.DMPersonality`.  Defaults to
        :data:`~src.dm.personality.DEFAULT_PERSONALITY` (The Sage) when
        ``None`` is passed.
    """

    def __init__(
        self,
        llm: LLMProvider,
        campaign: ParsedCampaign,
        rules: RulesReference,
        memory: MemoryManager,
        settings: Optional[Settings] = None,
        personality: Optional[DMPersonality] = None,
    ) -> None:
        self._llm = llm
        self._campaign = campaign
        self._rules = rules
        self._memory = memory
        self._settings = settings or _default_settings
        self._personality: DMPersonality = personality or DEFAULT_PERSONALITY
        self._turn: int = 0
        self._state: NarrativeState = NarrativeState.EXPLORATION
        self._last_roll_results: list = []

    # ------------------------------------------------------------------
    # Token budget helpers
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Approximate token count: 1 token ≈ 4 characters."""
        return max(1, len(text) // 4)

    def _system_prompt_budget(self) -> int:
        """
        Compute the token budget for the system prompt.

        Reserves ``max_tokens`` for the model's reply and a 256-token safety
        margin, leaving the rest for the system prompt.  Falls back to a
        conservative default if the provider does not expose a numeric
        context_window (e.g. in tests with mock providers).
        """
        try:
            cw = int(self._llm.context_window)
        except (TypeError, ValueError):
            cw = 4_096
        return max(1_000, cw - self._settings.max_tokens - 256)

    def _trim_session_to_budget(
        self,
        session_msgs: list[dict],
        system_prompt: str,
        current_input: str,
    ) -> list[dict]:
        """
        Drop the oldest session messages until the full message list fits
        within the model's context window.

        Preserves the system prompt and the current player input; trims
        user/assistant pairs from the front of the session window.
        """
        reserved_output = self._settings.max_tokens
        margin = 256
        try:
            cw = int(self._llm.context_window)
        except (TypeError, ValueError):
            cw = 4_096
        total_budget = max(1_000, cw - reserved_output - margin)

        fixed_tokens = self._estimate_tokens(system_prompt) + self._estimate_tokens(current_input)

        msgs = list(session_msgs)
        original_len = len(msgs)

        while msgs:
            session_tokens = sum(self._estimate_tokens(m["content"]) for m in msgs)
            if fixed_tokens + session_tokens <= total_budget:
                break
            # Drop the oldest pair (user + assistant) to preserve turn coherence.
            msgs = msgs[2:] if len(msgs) >= 2 else msgs[1:]

        if len(msgs) < original_len:
            log.info(
                "Trimmed session window from %d to %d messages to fit context budget "
                "(context_window=%d, estimated usage=%d tokens).",
                original_len,
                len(msgs),
                self._llm.context_window,
                fixed_tokens + sum(self._estimate_tokens(m["content"]) for m in msgs),
            )

        return msgs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_campaign(self) -> str:
        """
        Deliver the opening narration for the campaign.

        Builds the full system prompt (without player input) and asks the LLM
        to introduce the adventure.  Records the opening as turn 0 so the
        graph starts accumulating facts from the first scene.

        Returns
        -------
        str
            The DM's opening narration.
        """
        system_prompt = await build_system_prompt(
            campaign=self._campaign,
            rules=self._rules,
            memory=self._memory,
            state=self._state,
            current_text="",
            token_budget=self._system_prompt_budget(),
            personality=self._personality,
            llm_fn=self._make_llm_fn(),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Please begin the adventure. Describe the opening scene "
                    "and set the stage for the player."
                ),
            },
        ]
        response = self._llm.complete(
            messages, temperature=self._settings.dm_temperature
        )
        self._turn += 1
        await self._memory.record_turn("(campaign start)", response, self._turn)
        return response

    async def respond(self, player_input: str) -> str:
        """
        Generate a DM response to *player_input*.

        Steps
        -----
        1. Infer narrative state from the player's input.
        2. Build the system prompt (includes spoiler-guarded book + graph
           memory context).
        3. Assemble the full message list: system prompt + session window +
           new player message.
        4. Call the LLM.
        5. Record the turn in memory (session window + Graphiti episode).
        6. Check whether the response implies the player has reached a new
           scene and advance the progress pointer accordingly.
        7. Return the DM response.

        Note
        ----
        Dice tag substitution (``[ROLL: ...]``) is added in Phase 7.  Until
        then, roll tags produced by the LLM appear verbatim in the output.

        Parameters
        ----------
        player_input:
            The player's raw input text for this turn.

        Returns
        -------
        str
            The DM's narrated response.
        """
        self._state = detect_narrative_state(player_input)

        system_prompt = await build_system_prompt(
            campaign=self._campaign,
            rules=self._rules,
            memory=self._memory,
            state=self._state,
            current_text=player_input,
            token_budget=self._system_prompt_budget(),
            personality=self._personality,
            llm_fn=self._make_llm_fn(),
        )

        session_msgs = self._trim_session_to_budget(
            self._memory.session_messages(), system_prompt, player_input
        )
        messages = (
            [{"role": "system", "content": system_prompt}]
            + session_msgs
            + [{"role": "user", "content": player_input}]
        )

        dm_response = self._llm.complete(
            messages,
            temperature=self._settings.dm_temperature,
            max_tokens=self._settings.max_tokens,
        )

        # --- Phase 7: resolve any [ROLL: ...] tags in the DM response -------
        roll_requests = parse_roll_tags(dm_response)
        if roll_requests:
            roll_results = [roll(req) for req in roll_requests]
            dm_response = substitute_rolls(dm_response, roll_results)
            self._last_roll_results = roll_results
        else:
            self._last_roll_results = []
        # ---------------------------------------------------------------------

        self._turn += 1
        await self._memory.record_turn(player_input, dm_response, self._turn)
        self._maybe_advance_progress(dm_response)

        # Phase H: periodic RAPTOR rebuild
        if (
            self._settings.raptor_enabled
            and self._turn > 0
            and self._turn % self._settings.raptor_rebuild_every == 0
        ):
            try:
                await self._memory.rebuild_raptor(self._make_llm_fn())
            except Exception as exc:  # noqa: BLE001
                log.warning("RAPTOR background rebuild failed: %s", exc)

        return dm_response

    async def end_session(self) -> None:
        """Generate session-end MemoRAG clues and rebuild the RAPTOR tree.

        Call this when the player quits or between play sessions.  It is safe
        to skip — the game continues normally without it.
        """
        try:
            await self._memory.end_of_session(self._make_llm_fn())
        except Exception as exc:  # noqa: BLE001
            log.warning("Session-end memory update failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_llm_fn(self):
        """Return an async callable ``(prompt: str) -> str`` wrapping the
        synchronous LLM provider.  Used for Phase F contextual compression
        and Phase H/I RAPTOR/MemoRAG summarisation."""
        llm = self._llm

        async def _llm_fn(prompt: str) -> str:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: llm.complete([{"role": "user", "content": prompt}]),
            )

        return _llm_fn

    def _maybe_advance_progress(self, dm_response: str) -> None:
        """
        Advance the spoiler-guard progress pointer if the DM response
        references the title of the next unvisited scene.

        This is a lightweight heuristic — it searches for the next scene's
        title (case-insensitive) in the response text.  Correct behaviour
        relies on scene titles being reasonably unique within the campaign.
        """
        next_section = self._memory.campaign_progress + 1
        if next_section >= len(self._campaign.scene_titles):
            return

        next_title = self._campaign.scene_titles[next_section].lower()
        if next_title and next_title in dm_response.lower():
            log.info(
                "Progress advancing to scene %d ('%s')",
                next_section,
                self._campaign.scene_titles[next_section],
            )
            self._memory.advance_progress(next_section)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_roll_results(self) -> list:
        """
        Roll results from the most recent :meth:`respond` call.

        Returns an empty list if no dice were rolled in the last turn.
        The CLI uses this to render styled dice panels after each DM response.
        """
        return self._last_roll_results

    @property
    def personality(self) -> DMPersonality:
        """The currently active :class:`~src.dm.personality.DMPersonality`."""
        return self._personality

    def set_personality(self, personality: DMPersonality) -> None:
        """
        Switch to a new personality.

        Takes effect from the next :meth:`respond` or :meth:`respond_stream`
        call — the current turn is not affected.
        """
        log.info("Personality changed: '%s' → '%s'", self._personality.name, personality.name)
        self._personality = personality

    @property
    def campaign(self) -> "ParsedCampaign":
        """The parsed campaign data for this session."""
        return self._campaign

    @property
    def rules(self) -> "RulesReference":
        """The loaded rules reference for this session."""
        return self._rules

    # ------------------------------------------------------------------
    # Streaming support (Phase 10)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_safe_text(
        buf: str,
        roll_results: list,
    ) -> tuple[str, str]:
        """
        Split *buf* into ``(safe_to_yield, remaining_buffer)``.

        Yields text that contains no open, unresolved ``[ROLL: ...]`` tags.
        Keeps partial tags in *remaining_buffer* until the closing ``]``
        arrives in a future chunk.  When a complete tag is detected it is
        resolved immediately; the roll result is appended to *roll_results*.
        """
        safe_parts: list[str] = []

        while True:
            open_idx = buf.find("[ROLL:")
            if open_idx == -1:
                # No roll tag anywhere — everything is safe.
                safe_parts.append(buf)
                buf = ""
                break

            close_idx = buf.find("]", open_idx)
            if close_idx == -1:
                # Incomplete tag — yield text before it; keep tag fragment.
                safe_parts.append(buf[:open_idx])
                buf = buf[open_idx:]
                break

            # Complete tag — resolve it.
            before = buf[:open_idx]
            tag = buf[open_idx : close_idx + 1]
            buf = buf[close_idx + 1 :]

            tags = parse_roll_tags(tag)
            if tags:
                results = [roll(req) for req in tags]
                roll_results.extend(results)
                resolved = substitute_rolls(tag, results)
                safe_parts.append(before + resolved)
            else:
                safe_parts.append(before + tag)

        return "".join(safe_parts), buf

    async def respond_stream(
        self,
        player_input: str,
        cancel_event: threading.Event | None = None,
    ):  # -> AsyncGenerator[str, None]
        """
        Async generator that streams the DM's response token by token.

        Follows the same pipeline as :meth:`respond` but yields incremental
        text chunks as they arrive from the LLM.  ``[ROLL: ...]`` tags are
        buffered across chunk boundaries and resolved before being yielded,
        so callers always receive clean, resolved text.

        If *cancel_event* is set (by an interrupt thread) the stream is
        cut short; the partial response is still recorded in memory so
        the knowledge graph always reflects what the player saw.

        Parameters
        ----------
        player_input:
            The player's raw input text for this turn.
        cancel_event:
            Optional :class:`threading.Event`.  When set, the stream yields
            a ``[interrupted]`` marker and stops early.

        Yields
        ------
        str
            Incremental text chunks of the DM's narration.
        """
        self._state = detect_narrative_state(player_input)

        system_prompt = await build_system_prompt(
            campaign=self._campaign,
            rules=self._rules,
            memory=self._memory,
            state=self._state,
            current_text=player_input,
            token_budget=self._system_prompt_budget(),
            personality=self._personality,
            llm_fn=self._make_llm_fn(),
        )

        session_msgs = self._trim_session_to_budget(
            self._memory.session_messages(), system_prompt, player_input
        )
        messages = (
            [{"role": "system", "content": system_prompt}]
            + session_msgs
            + [{"role": "user", "content": player_input}]
        )

        roll_results: list = []
        response_parts: list[str] = []
        pending = ""
        interrupted = False

        try:
            for chunk in self._llm.stream(
                messages,
                temperature=self._settings.dm_temperature,
                max_tokens=self._settings.max_tokens,
            ):
                if cancel_event is not None and cancel_event.is_set():
                    interrupted = True
                    break

                pending += chunk
                safe, pending = self._split_safe_text(pending, roll_results)
                if safe:
                    response_parts.append(safe)
                    yield safe
        finally:
            # Flush any remaining buffered text (including incomplete tags).
            if pending:
                tags = parse_roll_tags(pending)
                if tags:
                    results = [roll(req) for req in tags]
                    roll_results.extend(results)
                    resolved = substitute_rolls(pending, results)
                    response_parts.append(resolved)
                    if not interrupted:
                        yield resolved
                else:
                    response_parts.append(pending)
                    if not interrupted:
                        yield pending

            if interrupted:
                yield "\n\n*[interrupted]*"

            full_response = "".join(response_parts)
            self._last_roll_results = roll_results
            self._turn += 1
            await self._memory.record_turn(player_input, full_response, self._turn)
            self._maybe_advance_progress(full_response)
