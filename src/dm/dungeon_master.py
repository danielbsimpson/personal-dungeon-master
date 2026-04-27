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

import logging
from typing import Optional

from src.campaign.parser import ParsedCampaign
from src.config import Settings
from src.config import settings as _default_settings
from src.dice.roller import parse_roll_tags, roll, substitute_rolls
from src.dm.context_builder import build_system_prompt, detect_narrative_state
from src.dm.memory.manager import MemoryManager
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
    """

    def __init__(
        self,
        llm: LLMProvider,
        campaign: ParsedCampaign,
        rules: RulesReference,
        memory: MemoryManager,
        settings: Optional[Settings] = None,
    ) -> None:
        self._llm = llm
        self._campaign = campaign
        self._rules = rules
        self._memory = memory
        self._settings = settings or _default_settings
        self._turn: int = 0
        self._state: NarrativeState = NarrativeState.EXPLORATION
        self._last_roll_results: list = []

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
        )

        messages = (
            [{"role": "system", "content": system_prompt}]
            + self._memory.session_messages()
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

        return dm_response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
    def campaign(self) -> "ParsedCampaign":
        """The parsed campaign data for this session."""
        return self._campaign
