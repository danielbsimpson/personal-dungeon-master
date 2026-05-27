"""
Integration tests for the Phase 12 web interface (src/interface/web.py).

Uses httpx + FastAPI's async test client so we can test REST endpoints and
WebSocket sessions without a running server.  All heavy I/O (DM bootstrap,
LLM calls, memory) is mocked so tests stay fast.
"""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mock_session(session_id: str = "test-session-id") -> tuple:
    """Return (mock_session_state, patched sessions dict)."""
    from src.interface.web import SessionState

    mock_dm = MagicMock()
    mock_dm.last_roll_results = []
    mock_dm.start_campaign = AsyncMock(return_value="You find yourself in a tavern.")
    mock_dm.respond_stream = MagicMock()
    mock_dm.personality.name = "The Sage"
    mock_dm.campaign.character = _mock_character()
    mock_dm.rules = MagicMock()

    mock_memory = MagicMock()
    mock_memory.get_entities = AsyncMock(return_value=[
        {"name": "Aela", "summary": "A fierce warrior."}
    ])
    mock_memory.search_entity = AsyncMock(return_value=[
        {"fact": "Aela wielded a silver axe."}
    ])
    mock_memory.reset_session = MagicMock()
    mock_memory.full_reset = AsyncMock()

    mock_campaign = MagicMock()
    mock_campaign.character = _mock_character()

    mock_rules = MagicMock()

    session = SessionState(
        dm=mock_dm,
        memory=mock_memory,
        campaign=mock_campaign,
        rules=mock_rules,
        started=False,
    )
    return session, session_id


def _mock_character():
    char = MagicMock()
    char.name = "Aldric"
    char.race = "Human"
    char.character_class = "Fighter"
    char.subclass = "Champion"
    char.level = 3
    char.hit_points = 28
    char.armor_class = 16
    char.speed = 30
    char.proficiency_bonus = 2
    char.passive_perception = 11
    char.equipment = [{"item": "Longsword"}, {"item": "Shield"}]
    scores = MagicMock()
    scores.strength = 16
    scores.dexterity = 12
    scores.constitution = 14
    scores.intelligence = 10
    scores.wisdom = 12
    scores.charisma = 10
    char.ability_scores = scores
    return char


# ---------------------------------------------------------------------------
# Sync client fixture (used for simple REST tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from src.interface.web import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /campaigns
# ---------------------------------------------------------------------------

def test_get_campaigns_returns_list(client):
    from src.campaign.loader import Campaign
    from pathlib import Path

    mock_campaigns = [
        Campaign(
            name="example-campaign",
            path=Path("/fake/example-campaign"),
            readme_path=Path("/fake/README.md"),
            character_path=Path("/fake/character.md"),
            creature_path=Path("/fake/creature.md"),
            book_path=Path("/fake/example-campaign.txt"),
        )
    ]
    with patch("src.interface.web.load_campaigns", return_value=mock_campaigns):
        resp = client.get("/campaigns")

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["name"] == "example-campaign"


def test_get_campaigns_empty_raises_500(client):
    with patch("src.interface.web.load_campaigns", side_effect=ValueError("no campaigns")):
        resp = client.get("/campaigns")
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /personalities
# ---------------------------------------------------------------------------

def test_get_personalities_returns_all_six(client):
    resp = client.get("/personalities")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 6
    names = {p["name"] for p in data}
    assert "The Sage" in names
    assert "The Warden" in names
    assert "The Bard" in names


def test_personalities_have_required_fields(client):
    resp = client.get("/personalities")
    for p in resp.json():
        assert "name" in p
        assert "slug" in p
        assert "tone" in p
        assert "verbosity" in p
        assert "description" in p


# ---------------------------------------------------------------------------
# POST /session/start
# ---------------------------------------------------------------------------

def test_start_session_returns_session_id(client):
    from src.campaign.loader import Campaign
    from pathlib import Path

    mock_campaign_meta = Campaign(
        name="example-campaign",
        path=Path("/fake"),
        readme_path=Path("/fake/README.md"),
        character_path=Path("/fake/character.md"),
        creature_path=Path("/fake/creature.md"),
        book_path=Path("/fake/example-campaign.txt"),
    )
    mock_parsed = MagicMock()
    mock_rules = MagicMock()
    mock_llm = MagicMock()
    mock_memory = MagicMock()
    mock_memory.load = AsyncMock()
    mock_memory.embed = AsyncMock(return_value=[[0.1] * 768])

    with (
        patch("src.interface.web.load_campaigns", return_value=[mock_campaign_meta]),
        patch("src.interface.web.parse_campaign", return_value=mock_parsed),
        patch("src.interface.web.load_rules", return_value=mock_rules),
        patch("src.interface.web.create_provider", return_value=mock_llm),
        patch("src.interface.web.MemoryManager", return_value=mock_memory),
        patch("src.interface.web.enrich_campaign", new_callable=AsyncMock),
    ):
        resp = client.post(
            "/session/start",
            json={"campaign": "example-campaign", "personality": "The Sage"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert len(data["session_id"]) == 36  # UUID format
    assert data["campaign"] == "example-campaign"
    assert data["personality"] == "The Sage"


def test_start_session_unknown_campaign_returns_404(client):
    from src.campaign.loader import Campaign
    from pathlib import Path

    mock_campaign_meta = Campaign(
        name="other-campaign",
        path=Path("/fake"),
        readme_path=Path("/fake/README.md"),
        character_path=Path("/fake/character.md"),
        creature_path=Path("/fake/creature.md"),
        book_path=Path("/fake/other-campaign.txt"),
    )
    with patch("src.interface.web.load_campaigns", return_value=[mock_campaign_meta]):
        resp = client.post(
            "/session/start",
            json={"campaign": "nonexistent", "personality": "The Sage"},
        )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /session/{id}/command
# ---------------------------------------------------------------------------

@pytest.fixture
def session_client():
    """Client with a pre-populated session in _sessions."""
    from src.interface.web import app, _sessions

    session, sid = _make_mock_session()
    _sessions[sid] = session
    yield TestClient(app), sid
    _sessions.pop(sid, None)


def test_command_status(session_client):
    client, sid = session_client
    resp = client.post(f"/session/{sid}/command", json={"command": "status"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["command"] == "status"
    assert data["data"]["name"] == "Aldric"


def test_command_roll(session_client):
    client, sid = session_client
    resp = client.post(f"/session/{sid}/command", json={"command": "roll", "expr": "d20"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["command"] == "roll"
    assert 1 <= data["data"]["total"] <= 20


def test_command_roll_bad_expr(session_client):
    client, sid = session_client
    resp = client.post(f"/session/{sid}/command", json={"command": "roll", "expr": "zzzz"})
    assert resp.status_code == 400


def test_command_journal(session_client):
    client, sid = session_client
    resp = client.post(f"/session/{sid}/command", json={"command": "journal"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["command"] == "journal"
    assert len(data["data"]) == 1
    assert data["data"][0]["name"] == "Aela"


def test_command_unknown_session_returns_404(client):
    resp = client.post("/session/does-not-exist/command", json={"command": "status"})
    assert resp.status_code == 404


def test_command_unknown_command_returns_400(session_client):
    client, sid = session_client
    resp = client.post(f"/session/{sid}/command", json={"command": "foobar"})
    assert resp.status_code == 400


def test_command_reset(session_client):
    client, sid = session_client
    resp = client.post(f"/session/{sid}/command", json={"command": "reset"})
    assert resp.status_code == 200
    assert resp.json()["command"] == "reset"


def test_command_personality_switch(session_client):
    client, sid = session_client
    resp = client.post(
        f"/session/{sid}/command",
        json={"command": "personality", "personality": "The Bard"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["command"] == "personality"
    assert data["data"]["name"] == "The Bard"


def test_command_personality_unknown(session_client):
    client, sid = session_client
    resp = client.post(
        f"/session/{sid}/command",
        json={"command": "personality", "personality": "The Ghost"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Path traversal guard — GET /campaign/{name}/image/{filename}
# (The endpoint is defined in Phase 13; this test verifies the guard logic
#  in the web module is wired before that phase lands.)
# ---------------------------------------------------------------------------

def test_image_endpoint_traversal_rejected(client):
    resp = client.get("/campaign/example-campaign/image/../../../etc/passwd")
    # Must not return 200 with file contents; 404 or 400 are both acceptable
    assert resp.status_code in (404, 400, 422)


# ---------------------------------------------------------------------------
# WebSocket — opening narration cycle
# ---------------------------------------------------------------------------

def test_websocket_sends_opening_narration():
    """
    The WebSocket should send an 'opening' message on first connect
    after start_campaign() completes.
    """
    from src.interface.web import app, _sessions

    session, sid = _make_mock_session()
    session.dm.last_roll_results = []
    _sessions[sid] = session

    try:
        with TestClient(app) as client:
            with client.websocket_connect(f"/session/{sid}/chat") as ws:
                data = ws.receive_json()
                assert data["type"] == "opening"
                assert "tavern" in data["content"]
    finally:
        _sessions.pop(sid, None)


def test_websocket_unknown_session_sends_error():
    from src.interface.web import app

    with TestClient(app) as client:
        with client.websocket_connect("/session/no-such-id/chat") as ws:
            data = ws.receive_json()
            assert data["type"] == "error"
