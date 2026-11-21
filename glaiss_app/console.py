"""GlAIss console application: the web surface, never the request path.

Hard rule (Spec, 2026-11-21): this repo contains **exactly one proxy** — the llpc
waitroom door (``llpc/waitroom.py``, listening on :8080). Everything clients send
goes to that door, and it alone decides which model serves them.

This module must therefore never grow:
  * a catch-all/passthrough route that forwards client traffic to a model server,
  * a provider host/port/scheme setting to forward *to*,
  * a tap that reads prompts or responses (that is the door's and the engine's job).

What lives here instead: a static page, a websocket fan-out of monitor snapshots,
and (via ``monitor.install``) the read-only collectors plus the deliberate
model-control endpoints, which go through ``swap-to.sh`` — never a direct
start/stop of a serving unit.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketState

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
STATE_DIR = Path(os.environ.get("GLAISS_STATE_DIR", BASE_DIR))
CONFIG_PATH = STATE_DIR / "glaiss_config.json"
STATIC_DIR = PACKAGE_DIR / "static"

DEFAULT_CONFIG: Dict[str, Any] = {"listen_port": 8899}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            return {**DEFAULT_CONFIG, **{k: v for k, v in cfg.items() if k in DEFAULT_CONFIG}}
        except Exception:
            pass
    save_config(DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def apply_env_overrides(cfg: dict) -> dict:
    """Deployment overrides used by systemd without mutating the saved config."""
    if os.environ.get("GLAISS_LISTEN_PORT"):
        cfg["listen_port"] = int(os.environ["GLAISS_LISTEN_PORT"])
    return cfg


CONFIG = apply_env_overrides(load_config())


class LoopLock:
    """asyncio.Lock bound to the loop that is actually running (uvicorn workers)."""

    def __init__(self):
        self._locks: Dict[int, asyncio.Lock] = {}

    def _get(self) -> asyncio.Lock:
        loop_id = id(asyncio.get_event_loop())
        lock = self._locks.get(loop_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop_id] = lock
        return lock

    async def __aenter__(self):
        return await self._get().__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        self._get().release()


class Hub:
    """Fan-out of console messages (monitor snapshots, control jobs, door events).

    No replay buffer on purpose: the durable event feed is
    ``journal.JournalRing`` (served by ``/api/logs``, survives a page reload), and
    a second in-memory copy here would just be a second source of truth.
    """

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.lock = LoopLock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self.lock:
            self.clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(message)
        async with self.lock:
            targets = list(self.clients)
        dead = []
        for ws in targets:
            try:
                if ws.application_state == WebSocketState.CONNECTED:
                    await ws.send_text(payload)
                else:
                    dead.append(ws)
            except Exception:
                dead.append(ws)
        if dead:
            async with self.lock:
                for ws in dead:
                    self.clients.discard(ws)

    # monitor.ControlManager and the door-event route call emit(); with no request
    # log to keep, emitting is just a broadcast.
    async def emit(self, message: dict) -> None:
        await self.broadcast(message)


hub = Hub()

app = FastAPI(title="GlAIss console", docs_url=None, redoc_url=None)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def console_ws(ws: WebSocket):
    """Live console channel: the client only listens; the monitor pushes."""
    await hub.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        await hub.disconnect(ws)
