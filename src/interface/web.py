"""
Web interface for Personal Dungeon Master (Phase 12).

A FastAPI application that serves the browser-based UI and exposes:

- ``GET  /``                              — single-page frontend
- ``GET  /campaigns``                     — available campaign list
- ``GET  /personalities``                 — all six DM personalities
- ``POST /session/start``                 — bootstrap a DM session
- ``POST /session/{id}/command``          — execute a non-streaming command
- ``WS   /session/{id}/chat``             — WebSocket streaming chat loop

WebSocket message contract
--------------------------
Client → Server::

    {"type": "message",   "content": "..."}   # player turn
    {"type": "interrupt"}                      # cancel current stream

Server → Client::

    {"type": "opening",    "content": "..."}   # first DM narration (on connect)
    {"type": "token",      "content": "..."}   # one streamed token
    {"type": "roll",       "result":  {...}}   # dice roll panel data
    {"type": "interrupted"}                    # stream was cancelled
    {"type": "done"}                           # stream complete
    {"type": "error",      "message": "..."}   # something went wrong

Run via ``src/main.py --interface web`` (adds ``--host``/``--port`` flags).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.campaign.loader import enrich_campaign, load_campaigns
from src.campaign.parser import ParsedCampaign, parse_campaign
from src.config import Settings
from src.config import settings as _default_settings
from src.dm.dungeon_master import DungeonMaster
from src.dm.memory.manager import MemoryManager
from src.dm.personality import DEFAULT_PERSONALITY, PERSONALITIES, get_personality
from src.llm.factory import create_provider
from src.rules.loader import RulesReference, load_rules

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """All runtime state for one browser session."""

    dm: DungeonMaster
    memory: MemoryManager
    campaign: ParsedCampaign
    rules: RulesReference
    started: bool = False  # True after start_campaign() has been called


# Global in-memory session store (local single-user app).
_sessions: dict[str, SessionState] = {}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Personal Dungeon Master", docs_url=None, redoc_url=None)

# Serve static files (HTML/CSS/JS) if the directory exists.
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class StartRequest(BaseModel):
    campaign: str
    personality: str = "The Sage"


class CommandRequest(BaseModel):
    command: str
    entity: Optional[str] = None   # /graph <entity>
    expr: Optional[str] = None     # /roll <expr>
    personality: Optional[str] = None  # /personality switch
    topic: Optional[str] = None    # /rules <topic>


# ---------------------------------------------------------------------------
# Routes — static frontend
# ---------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    html = _STATIC_DIR / "index.html"
    if not html.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(str(html))


# ---------------------------------------------------------------------------
# Routes — metadata
# ---------------------------------------------------------------------------


@app.get("/campaigns")
async def get_campaigns(settings: Settings = None) -> list[dict]:
    cfg = settings or _default_settings
    try:
        campaigns = load_campaigns(cfg.campaigns_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [{"name": c.name} for c in campaigns]


@app.get("/personalities")
async def get_personalities() -> list[dict]:
    return [
        {
            "name": p.name,
            "slug": p.slug,
            "description": p.description,
            "tone": p.tone,
            "verbosity": p.verbosity,
        }
        for p in PERSONALITIES
    ]


# ---------------------------------------------------------------------------
# Routes — session lifecycle
# ---------------------------------------------------------------------------


@app.post("/session/start")
async def start_session(body: StartRequest) -> dict:
    """
    Bootstrap a full DM session.

    Loads the campaign, rules, LLM provider, and memory in exactly the same
    order as ``src/main.py``.  Returns a ``session_id`` the frontend uses
    for all subsequent requests.

    The opening narration is **not** sent here — it is sent as a
    ``{"type": "opening", ...}`` message on the first WebSocket connect so
    the client can display it inside the chat panel.
    """
    cfg = _default_settings

    # -- Campaign ----------------------------------------------------------
    try:
        campaigns = load_campaigns(cfg.campaigns_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Cannot load campaigns: {exc}") from exc

    matches = [c for c in campaigns if c.name.lower() == body.campaign.lower()]
    if not matches:
        available = ", ".join(c.name for c in campaigns)
        raise HTTPException(
            status_code=404,
            detail=f"Campaign '{body.campaign}' not found. Available: {available}",
        )
    selected = matches[0]

    try:
        parsed = parse_campaign(selected)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Campaign parse error: {exc}") from exc

    # -- Rules -------------------------------------------------------------
    try:
        rules = load_rules(cfg)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Rules error: {exc}") from exc

    # -- LLM ---------------------------------------------------------------
    try:
        llm = create_provider(cfg)
    except (NotImplementedError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

    # -- Personality -------------------------------------------------------
    try:
        personality = get_personality(body.personality)
    except ValueError:
        personality = DEFAULT_PERSONALITY

    # -- Memory ------------------------------------------------------------
    memory = MemoryManager(memory_dir=cfg.memory_dir, settings=cfg)
    try:
        await memory.load(selected.name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Memory init error: {exc}") from exc

    # -- Campaign index / enrichment (best-effort) -------------------------
    async def _summarise(prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: llm.complete([{"role": "user", "content": prompt}]),
        )

    try:
        await enrich_campaign(
            parsed=parsed,
            embed_fn=memory.embed,
            summarise_fn=_summarise,
            memory_dir=cfg.memory_dir,
        )
    except Exception as exc:
        log.warning("enrich_campaign skipped: %s", exc)

    # -- DungeonMaster -----------------------------------------------------
    dm = DungeonMaster(
        llm=llm,
        campaign=parsed,
        rules=rules,
        memory=memory,
        settings=cfg,
        personality=personality,
    )

    session_id = str(uuid.uuid4())
    _sessions[session_id] = SessionState(
        dm=dm,
        memory=memory,
        campaign=parsed,
        rules=rules,
    )
    log.info("Session created: id=%s campaign='%s' personality='%s'",
             session_id, selected.name, personality.name)

    return {
        "session_id": session_id,
        "campaign": selected.name,
        "personality": personality.name,
        "personality_slug": personality.slug,
    }


# ---------------------------------------------------------------------------
# Routes — commands (non-streaming)
# ---------------------------------------------------------------------------


@app.post("/session/{session_id}/command")
async def handle_command(session_id: str, body: CommandRequest) -> dict:
    """Execute a non-streaming player command and return structured JSON."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    cmd = body.command

    if cmd == "status":
        return {"command": "status", "data": _character_to_dict(session.campaign.character)}

    if cmd == "journal":
        entities = await session.memory.get_entities()
        return {"command": "journal", "data": entities}

    if cmd == "graph":
        if not body.entity:
            raise HTTPException(status_code=400, detail="entity is required for /graph")
        results = await session.memory.search_entity(body.entity)
        return {"command": "graph", "entity": body.entity, "data": results}

    if cmd == "roll":
        if not body.expr:
            raise HTTPException(status_code=400, detail="expr is required for /roll")
        from src.dice.roller import parse_player_expression, roll  # noqa: PLC0415
        req = parse_player_expression(body.expr, label="player roll")
        if req is None:
            raise HTTPException(status_code=400, detail=f"Unknown dice expression: {body.expr}")
        result = roll(req)
        return {"command": "roll", "data": _roll_to_dict(result)}

    if cmd == "rules":
        if not body.topic:
            raise HTTPException(status_code=400, detail="topic is required for /rules")
        from src.rules.reference import search_rules  # noqa: PLC0415
        text = search_rules(session.rules, body.topic)
        return {"command": "rules", "topic": body.topic, "data": text}

    if cmd == "reset":
        session.memory.reset_session()
        return {"command": "reset"}

    if cmd == "fullreset":
        await session.memory.full_reset()
        return {"command": "fullreset"}

    if cmd == "save":
        return {"command": "save"}

    if cmd == "personality":
        if not body.personality:
            raise HTTPException(status_code=400, detail="personality is required")
        try:
            new_p = get_personality(body.personality)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.dm.set_personality(new_p)
        return {
            "command": "personality",
            "data": {"name": new_p.name, "slug": new_p.slug, "description": new_p.description},
        }

    raise HTTPException(status_code=400, detail=f"Unknown command: {cmd}")


# ---------------------------------------------------------------------------
# WebSocket — streaming chat
# ---------------------------------------------------------------------------


@app.websocket("/session/{session_id}/chat")
async def ws_chat(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket endpoint for the streaming chat loop.

    On first connect, sends the opening narration as
    ``{"type": "opening", "content": "..."}``.

    Subsequently handles ``{"type": "message"}`` turns and
    ``{"type": "interrupt"}`` signals.
    """
    await websocket.accept()

    session = _sessions.get(session_id)
    if session is None:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    # ---- Single receiver task → feeds a queue ----------------------------
    # FastAPI/Starlette only allows one active receive at a time, so we
    # funnel all incoming messages through one background task and a queue.
    msg_queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def _receiver() -> None:
        try:
            while True:
                data = await websocket.receive_json()
                await msg_queue.put(data)
        except WebSocketDisconnect:
            await msg_queue.put(None)
        except Exception as exc:
            log.warning("WebSocket receive error: %s", exc)
            await msg_queue.put(None)

    recv_task = asyncio.create_task(_receiver())

    try:
        # -- Opening narration (sent only once) ----------------------------
        if not session.started:
            try:
                opening = await session.dm.start_campaign()
                session.started = True
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": f"Start error: {exc}"})
                return

            for result in session.dm.last_roll_results:
                await websocket.send_json({"type": "roll", "result": _roll_to_dict(result)})
            await websocket.send_json({"type": "opening", "content": opening})
        else:
            await websocket.send_json({"type": "reconnected"})

        # -- Main message loop ---------------------------------------------
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break  # client disconnected

            msg_type = msg.get("type")

            if msg_type == "message":
                player_input = msg.get("content", "").strip()
                if player_input:
                    await _stream_response(websocket, session, player_input, msg_queue)

    finally:
        recv_task.cancel()
        # Phase H/I: generate MemoRAG clues and rebuild RAPTOR tree on disconnect.
        try:
            await session.dm.end_session()
        except Exception as exc:  # noqa: BLE001
            log.warning("Session-end memory update failed: %s", exc)


async def _stream_response(
    websocket: WebSocket,
    session: SessionState,
    player_input: str,
    msg_queue: asyncio.Queue[dict | None],
) -> None:
    """
    Stream a single DM response over the WebSocket.

    Runs ``respond_stream()`` while a background task watches for an
    ``{"type": "interrupt"}`` message in the queue.  Roll results are
    sent after the stream finishes, followed by ``{"type": "done"}``.
    """
    cancel_event = threading.Event()

    async def _watch_interrupts() -> None:
        while not cancel_event.is_set():
            try:
                msg = await asyncio.wait_for(msg_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            if msg is None or msg.get("type") == "interrupt":
                cancel_event.set()
                return
            # Re-queue non-interrupt messages (shouldn't normally arrive mid-stream).
            await msg_queue.put(msg)

    interrupt_task = asyncio.create_task(_watch_interrupts())

    try:
        async for chunk in session.dm.respond_stream(player_input, cancel_event=cancel_event):
            if chunk == "\n\n*[interrupted]*":
                await websocket.send_json({"type": "interrupted"})
            else:
                await websocket.send_json({"type": "token", "content": chunk})
    except Exception as exc:
        log.error("respond_stream error: %s", exc)
        await websocket.send_json({"type": "error", "message": "Stream error occurred"})
    finally:
        cancel_event.set()
        interrupt_task.cancel()

        for result in session.dm.last_roll_results:
            await websocket.send_json({"type": "roll", "result": _roll_to_dict(result)})
        await websocket.send_json({"type": "done"})


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _roll_to_dict(result) -> dict:
    return {
        "label": result.label,
        "die": result.die.value,
        "rolls": result.rolls,
        "modifier": result.modifier,
        "total": result.total,
    }


def _character_to_dict(char) -> dict:
    scores = char.ability_scores
    return {
        "name": char.name,
        "race": char.race,
        "character_class": char.character_class,
        "subclass": char.subclass,
        "level": char.level,
        "hit_points": char.hit_points,
        "armor_class": char.armor_class,
        "speed": char.speed,
        "proficiency_bonus": char.proficiency_bonus,
        "passive_perception": char.passive_perception,
        "ability_scores": {
            "strength": scores.strength,
            "dexterity": scores.dexterity,
            "constitution": scores.constitution,
            "intelligence": scores.intelligence,
            "wisdom": scores.wisdom,
            "charisma": scores.charisma,
        },
        "equipment": char.equipment[:10],
    }
