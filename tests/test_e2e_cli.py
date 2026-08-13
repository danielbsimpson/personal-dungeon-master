"""
Scripted end-to-end smoke test for the CLI session loop (Phase 8).

Drives :func:`src.interface.cli.run_session` through a full mini-session using
a fake LLM provider and a faked Graphiti backend, so it runs in CI with no live
model server, network access, or embeddings.  It exercises the real pipeline:
campaign parsing, rules loading, system-prompt assembly, the two-phase
player-driven dice flow, in-session commands, session persistence, and
graceful quit.
"""

from __future__ import annotations

import types
from typing import Iterator

import pytest

from src.campaign.loader import load_campaigns
from src.campaign.parser import parse_campaign
from src.config import Settings
from src.dm.dungeon_master import DungeonMaster
from src.dm.memory import manager as manager_mod
from src.dm.memory.manager import MemoryManager
from src.dm.personality import DEFAULT_PERSONALITY
from src.interface import cli
from src.llm.base import LLMProvider
from src.rules.loader import load_rules


class _FakeLLM(LLMProvider):
    """Deterministic provider that returns scripted replies in order."""

    context_window = 8192

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self._i = 0
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], **kwargs) -> str:
        self.calls.append(messages)
        if self._i < len(self._replies):
            reply = self._replies[self._i]
            self._i += 1
            return reply
        return "The tale continues quietly."

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        yield self.complete(messages, **kwargs)


class _FakeGraph:
    """Async no-op stand-in for GraphitiStore (no Kuzu, no network)."""

    def __init__(self, *args, **kwargs) -> None:
        self.episodes: list[str] = []

    async def setup(self) -> None:  # noqa: D401
        return None

    async def add_episode(self, *, name, content, source_description, turn, group_id) -> None:
        self.episodes.append(content)

    async def search(self, query, group_id, num_results: int = 10) -> str:
        return ""

    async def close(self) -> None:
        return None


@pytest.fixture
def _example_campaign():
    campaigns = load_campaigns()
    meta = next(c for c in campaigns if c.name == "example-campaign")
    return parse_campaign(meta)


async def test_scripted_cli_session_end_to_end(tmp_path, monkeypatch, _example_campaign):
    # Hermetic settings: temp memory dir, advanced memory phases off so the
    # session needs no embeddings/clustering.
    settings = Settings(
        memory_dir=tmp_path / "memory",
        raptor_enabled=False,
        memorag_enabled=False,
        compression_enabled=False,
        session_window=10,
    )
    rules = load_rules(settings)

    # Fake the graph backend and its client construction inside MemoryManager.
    monkeypatch.setattr(manager_mod, "GraphitiStore", _FakeGraph)
    monkeypatch.setattr(manager_mod, "build_graphiti_clients", lambda _s: (None, None))

    memory = MemoryManager(memory_dir=settings.memory_dir, settings=settings)
    await memory.load("example-campaign")

    llm = _FakeLLM(
        replies=[
            "The village of Barrowmere stirs to life beneath a grey sky.",  # opening
            "You survey the quiet square; nothing stirs but the wind.",     # look around
            "You raise your blade at the goblin. [ROLL: attack 1d20+5]",     # attack (first pass)
            "Your strike lands true and the goblin crumples.",              # attack (second pass)
        ]
    )

    dm = DungeonMaster(
        llm=llm,
        campaign=_example_campaign,
        rules=rules,
        memory=memory,
        settings=settings,
        personality=DEFAULT_PERSONALITY,
    )

    # Scripted player turns; the loop ends on /quit.
    player_inputs = iter(
        [
            "I look around the village",
            "I attack the goblin",
            "/status",
            "/roll d20+3",
            "/quit",
        ]
    )

    def fake_ask(prompt: str = "", default=None, **kwargs):
        # Roll prompts auto-roll (blank input); all other prompts take the
        # next scripted player action.
        if "Your roll" in str(prompt):
            return ""
        return next(player_inputs)

    monkeypatch.setattr(cli, "Prompt", types.SimpleNamespace(ask=fake_ask))

    # Should run to completion without raising.
    await cli.run_session(dm, memory, rules)

    # The opening plus both action turns were sent to the LLM (2 for the
    # dice turn: first pass + outcome narration).
    assert llm._i >= 4

    # The two narrative turns were recorded to the session window and graph.
    messages = memory.session_messages()
    joined = " ".join(m["content"] for m in messages)
    assert "I look around the village" in joined
    assert "I attack the goblin" in joined

    # The graph received an episode per recorded turn.
    assert len(memory._graph.episodes) >= 2  # type: ignore[attr-defined]

    # Session state was persisted to disk.
    session_file = settings.memory_dir / "example-campaign" / "session.json"
    assert session_file.exists()

    # The auto-rolled result was fed back into the outcome-narration call.
    second_pass_msgs = llm.calls[-1]
    assert any("Roll results" in m["content"] for m in second_pass_msgs)
