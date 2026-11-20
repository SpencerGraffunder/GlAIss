#!/usr/bin/env python3
"""
Glaiss — a passthrough LLM proxy that shows you what your local LLMs are doing.

Sits between your clients and your real OpenAI-compatible backend (vLLM,
llama.cpp server, LM Studio, etc). Forwards requests/responses unmodified,
while tapping the traffic to compute live stats (PP estimate, decode t/s)
and streaming prompt/response text to a web dashboard over WebSocket.

Run:
    python3 glaiss.py
Then point your LLM clients at Glaiss instead of the real backend:
    http://<this-box>:8899/v1/chat/completions
Dashboard:
    http://<this-box>:8899/

Config lives in glaiss_config.json next to this file (created on first run).
"""

import asyncio
import json
import base64
import mimetypes
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import httpx
import tiktoken
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from starlette.websockets import WebSocketState

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
STATE_DIR = Path(os.environ.get("GLAISS_STATE_DIR", BASE_DIR))
CONFIG_PATH = STATE_DIR / "glaiss_config.json"
STATIC_DIR = PACKAGE_DIR / "static"

DEFAULT_CONFIG = {
    "listen_port": 8899,
    "provider_host": "127.0.0.1",
    "provider_port": 8080,
    "provider_scheme": "http",
    "models": [],  # list of model name strings shown as configured/passed-through
    # rolling calibration of tokens/sec prefill speed, keyed by model name.
    # "_default" is used when a model has no calibration yet or model name is unknown.
    "pp_tokens_per_sec": {"_default": 1500.0},
}

# ---------------------------------------------------------------------------
# Config load/save
# ---------------------------------------------------------------------------

class LoopLock:
    """An asyncio.Lock that lazily binds to the *running* event loop on first use.

    Constructing ``asyncio.Lock()`` at import time binds it to the default event
    loop, which is NOT the loop uvicorn runs the app on. Any time the lock is
    actually contended, the waiting coroutine creates a Future on the import-time
    loop and awaits it on the server loop -> ``RuntimeError: got Future attached
    to a different loop``. Creating the real lock lazily (inside the running
    loop) avoids that. Interface-compatible with asyncio.Lock for acquire/release
    and ``async with``.
    """

    def __init__(self):
        self._lock = None

    def _get(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self):
        return await self._get().acquire()

    def release(self):
        self._get().release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.release()


_cfg_lock = LoopLock()


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            # merge in any new default keys without clobbering user values
            merged = {**DEFAULT_CONFIG, **cfg}
            merged["pp_tokens_per_sec"] = {
                **DEFAULT_CONFIG["pp_tokens_per_sec"],
                **cfg.get("pp_tokens_per_sec", {}),
            }
            return merged
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
    if os.environ.get("GLAISS_PROVIDER_HOST"):
        cfg["provider_host"] = os.environ["GLAISS_PROVIDER_HOST"]
    if os.environ.get("GLAISS_PROVIDER_PORT"):
        cfg["provider_port"] = int(os.environ["GLAISS_PROVIDER_PORT"])
    if os.environ.get("GLAISS_PROVIDER_SCHEME"):
        cfg["provider_scheme"] = os.environ["GLAISS_PROVIDER_SCHEME"]
    if os.environ.get("GLAISS_LISTEN_PORT"):
        cfg["listen_port"] = int(os.environ["GLAISS_LISTEN_PORT"])
    return cfg


CONFIG = apply_env_overrides(load_config())


def provider_base_url() -> str:
    return f"{CONFIG['provider_scheme']}://{CONFIG['provider_host']}:{CONFIG['provider_port']}"


# ---------------------------------------------------------------------------
# Tokenizer (approximate — used only for the PP-progress ESTIMATE, not for
# anything sent to the backend). cl100k is a reasonable universal stand-in;
# exact counts will vary per real model tokenizer.
# ---------------------------------------------------------------------------

_ENC = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_ENC.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# WebSocket hub — fan-out broadcast to all connected dashboard clients
# ---------------------------------------------------------------------------


# Message types that are live state (sent fresh on connect) rather than
# log entries — excluded from the replayable history buffer.
_NON_LOG_TYPES = {"stats", "config"}

# How many log entries to keep for replay. Bounded so a long-running server
# doesn't grow unbounded; old entries drop off the front.
LOG_BUFFER_MAX = 500


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.lock = LoopLock()
        # Replayable log of request-level events (prompt/queued/started/response/error).
        # Survives page reloads; reset on server restart (in-memory only).
        # Per-token streaming deltas are NOT logged — only the final response is.
        self.log: deque = deque(maxlen=LOG_BUFFER_MAX)

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self.lock:
            self.clients.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self.lock:
            self.clients.discard(ws)

    def history(self) -> list:
        return list(self.log)

    async def broadcast(self, message: dict):
        """Live fan-out only — used for stats pushes and per-token deltas."""
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

    async def emit(self, message: dict):
        """Broadcast a request-level event AND record it in the replay log.

        Record before sending so a client that connects mid-broadcast still
        sees this entry in its replayed history.
        """
        self.log.append(message)
        await self.broadcast(message)


hub = Hub()

# ---------------------------------------------------------------------------
# Generation queue — serializes requests that actually consume VRAM.
#
# The upstream model can only hold one context at a time, so generation
# requests (chat/completions with messages/prompt) are sent one at a time.
# asyncio.Lock wakes waiters in FIFO order, which gives us exactly the
# "accept everything, then drain in arrival order" behavior we want.
# Lightweight endpoints (/v1/models, /health, ...) never touch this lock.
# ---------------------------------------------------------------------------


class GenQueue:
    def __init__(self):
        self._lock = LoopLock()
        self._waiting = 0  # requests currently blocked on the lock

    @property
    def waiting(self) -> int:
        return self._waiting

    async def acquire(self):
        """Block until we're first in line. asyncio.Lock wakes waiters FIFO."""
        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            # Decrement on BOTH the success and cancellation paths so a request
            # that is cancelled while waiting can't leak the counter.
            self._waiting -= 1

    def release(self):
        """Hand the slot to the next waiter. Must be called exactly once per acquire."""
        self._lock.release()


gen_queue = GenQueue()

# ---------------------------------------------------------------------------
# Prompt dedup — repeated agent calls resend most of the same context.
#
# We keep a short rolling window of recent prompt texts. For each new prompt
# we find the most recent one it shares a long common prefix with (the agent
# loop case: same conversation, a few new messages appended). If a good match
# exists we report only the newly-appended tail as the "added" part and let
# the dashboard collapse the repeated prefix. If nothing recent is similar we
# treat it as a fresh conversation and show the full text.
# ---------------------------------------------------------------------------

# How many recent prompts to consider when looking for a match.
PROMPT_HISTORY_MAX = 8
# A match only counts if the shared prefix is at least this fraction of the
# shorter prompt — otherwise it's a coincidence, not a repeated context.
DIFF_MIN_PREFIX_RATIO = 0.5


class PromptHistory:
    def __init__(self):
        self._recent: deque = deque(maxlen=PROMPT_HISTORY_MAX)

    def add(self, text: str):
        if text:
            self._recent.append(text)

    def clear(self):
        self._recent.clear()

    def best_match(self, text: str) -> Optional[tuple]:
        """Return (matched_text, common_prefix_len) for the most recent prompt
        that shares a long common prefix with `text`, or None if none qualify.

        Iterates most-recent-first so the first hit is the closest prior call —
        exactly the "diff against the most recent similar request" behavior."""
        if not text:
            return None
        for prev in reversed(self._recent):
            prefix_len = _common_prefix_len(text, prev)
            shorter = min(len(text), len(prev))
            if shorter <= 0:
                continue
            if prefix_len / shorter >= DIFF_MIN_PREFIX_RATIO:
                return (prev, prefix_len)
        return None


prompt_history = PromptHistory()


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

# ---------------------------------------------------------------------------
# Live aggregate stats (top bar). Kept small and rolling.
# ---------------------------------------------------------------------------


class Stats:
    def __init__(self):
        self.active_requests: dict[str, dict] = {}
        self.recent_decode_tps = deque(maxlen=20)
        self.recent_pp_tps = deque(maxlen=20)
        self.total_requests = 0
        self.total_tokens_generated = 0

    def snapshot(self) -> dict:
        return {
            "type": "stats",
            "active_requests": list(self.active_requests.values()),
            "avg_decode_tps": round(
                sum(self.recent_decode_tps) / len(self.recent_decode_tps), 1
            )
            if self.recent_decode_tps
            else None,
            "avg_pp_tps": round(sum(self.recent_pp_tps) / len(self.recent_pp_tps), 1)
            if self.recent_pp_tps
            else None,
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "queued_count": gen_queue.waiting,
            "provider": provider_base_url(),
            "models": CONFIG["models"],
        }


stats = Stats()
# Per-request asyncio.Task registry, kept OUT of the stats.active_requests
# entries (which are JSON-serialized for the dashboard broadcast — a Task is
# not serializable). Maps req_id -> the task running that request, so the
# /api/cancel endpoint can abort it. Entries are dropped on completion/cancel.
request_tasks: dict = {}


STATS_PUSH_INTERVAL = 0.3  # seconds


async def stats_pusher():
    while True:
        await asyncio.sleep(STATS_PUSH_INTERVAL)
        if hub.clients:
            await hub.broadcast(stats.snapshot())


# ---------------------------------------------------------------------------
# Prompt text extraction — text-only, no images/tool schemas/etc.
# ---------------------------------------------------------------------------


def extract_prompt_text(body: dict) -> str:
    # chat completions: messages[] with role/content (content may be str or list of parts)
    if "messages" in body and isinstance(body["messages"], list):
        parts = []
        for m in body["messages"]:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # multimodal content array — keep only text parts
                text = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                text = str(content)
            if text:
                parts.append(f"[{role}] {text}")
        return "\n".join(parts)
    # legacy completions: prompt field (str or list of str)
    if "prompt" in body:
        p = body["prompt"]
        if isinstance(p, list):
            return "\n".join(str(x) for x in p)
        return str(p)
    return ""


def extract_settings(body: dict) -> dict:
    """Pull the model-relevant sampling/thinking settings out of a request
    body so they can be shown in the log. Only keys that are actually present
    are included (thinking level, temperature, etc.)."""
    settings = {}
    # thinking / reasoning controls (naming varies by client)
    for key in ("enable_thinking", "thinking", "reasoning_effort"):
        if key in body:
            settings[key] = body[key]
    # common sampling params
    for key in ("temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens"):
        if body.get(key) is not None:
            settings[key] = body[key]
    return settings


async def _resolve_image_url(url: str) -> str:
    """Convert an image URL to a base64 data URL (LM Studio requires base64).

    - data: URLs pass through unchanged.
    - file paths / file:// URLs are read from disk.
    - http(s) URLs are downloaded.
    - Unknown formats pass through unchanged.
    """
    if not url or url.startswith("data:"):
        return url
    # Diagnostic: log every non-data image URL so we can see what the client sends.
    _short = url[:80] + ("…" if len(url) > 80 else "")
    print(f"[glaiss] image url seen: {_short!r} (len={len(url)})", flush=True)

    # file path (absolute, relative, ~, or file:// URL)
    path = None
    if url.startswith("file://"):
        path = url[len("file://") :]
    elif os.path.isabs(url) or url.startswith("~") or not url.startswith(("http://", "https://", "data:")):
        # treat as a filesystem path (absolute, ~, or relative to CWD)
        path = os.path.expanduser(url)

    if path is not None:
        if not os.path.isfile(path):
            print(f"[glaiss] image file not found: {path!r}", flush=True)
            return url
        with open(path, "rb") as f:
            data = f.read()
        mime = mimetypes.guess_type(path)[0] or "image/png"
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")

    # http(s) URL — download
    if url.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url, follow_redirects=True)
                r.raise_for_status()
                mime = r.headers.get("content-type", "image/png").split(";")[0].strip()
                return f"data:{mime};base64," + base64.b64encode(r.content).decode("ascii")
        except Exception:
            return url  # leave as-is if download fails

    return url


async def _convert_webp_to_png(data_url: str) -> str:
    """Convert a WebP data URL to a PNG data URL via macOS `sips`.

    LM Studio rejects WebP images ('url' field must be a base64 encoded image),
    so we transcode them to PNG before forwarding. Returns the original data
    URL unchanged if conversion is unavailable or fails.
    """
    import subprocess
    import tempfile
    try:
        b64 = data_url.split(",", 1)[1]
        raw = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as wf:
            wf.write(raw)
            webp_path = wf.name
        png_path = webp_path + ".png"
        subprocess.run(
            ["sips", "-s", "format", "png", webp_path, "--out", png_path],
            check=True,
            capture_output=True,
            timeout=30,
        )
        png_data = open(png_path, "rb").read()
        os.unlink(webp_path)
        os.unlink(png_path)
        return "data:image/png;base64," + base64.b64encode(png_data).decode("ascii")
    except Exception as e:
        print(f"[glaiss] WebP→PNG conversion failed: {e}", flush=True)
        return data_url


async def _resolve_images_in_body(body_json: dict) -> dict:
    """Convert every image_url in the request body to a base64 data URL,
    so LM Studio accepts the request. Returns the (possibly modified) body."""
    if not isinstance(body_json, dict):
        return body_json
    for msg in body_json.get("messages", []) or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                iu = part.get("image_url")
                if isinstance(iu, dict) and iu.get("url"):
                    url = iu["url"]
                    if url.startswith("data:image/webp"):
                        iu["url"] = await _convert_webp_to_png(url)
                    else:
                        iu["url"] = await _resolve_image_url(url)
    return body_json


def calibrate_pp(model: str, prompt_tokens: int, pp_seconds: float):
    if prompt_tokens <= 0 or pp_seconds <= 0:
        return
    observed_tps = prompt_tokens / pp_seconds
    key = model or "_default"
    prev = CONFIG["pp_tokens_per_sec"].get(key, CONFIG["pp_tokens_per_sec"]["_default"])
    # exponential moving average so estimates settle over time without
    # jerking wildly on one-off outlier requests
    CONFIG["pp_tokens_per_sec"][key] = round(prev * 0.7 + observed_tps * 0.3, 1)
    save_config(CONFIG)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Glaiss")


@app.on_event("startup")
async def _startup():
    asyncio.create_task(stats_pusher())


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
async def get_config():
    return CONFIG


@app.post("/api/config")
async def update_config(request: Request):
    global CONFIG
    updates = await request.json()
    async with _cfg_lock:
        for key in ("provider_host", "provider_port", "provider_scheme", "models"):
            if key in updates:
                CONFIG[key] = updates[key]
        save_config(CONFIG)
    await hub.broadcast({"type": "config", "config": CONFIG})
    return CONFIG


@app.post("/api/cancel/{req_id}")
async def cancel_request(req_id: str):
    """Abort an in-flight or queued request (dashboard "x" button).

    Cancels the request's own task: if it is blocked on the queue lock the
    waiter is dropped (it never reaches the upstream); if it is streaming, the
    generator's finally releases the slot and closes the upstream connection.
    """
    entry = stats.active_requests.get(req_id)
    if entry is None:
        return JSONResponse(
            {"error": "no such active request"}, status_code=404
        )
    task = request_tasks.get(req_id)
    if task is None or task.done():
        # Already finished — nothing left to cancel.
        stats.active_requests.pop(req_id, None)
        request_tasks.pop(req_id, None)
        return JSONResponse(
            {"error": "already completed"}, status_code=409
        )
    entry["cancelled"] = True
    task.cancel()
    await hub.emit({"type": "cancelled", "id": req_id, "ts": time.time()})
    return {"ok": True, "id": req_id}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        # send an immediate snapshot + config so the page has state right away
        await ws.send_text(json.dumps(stats.snapshot()))
        await ws.send_text(json.dumps({"type": "config", "config": CONFIG}))
        # replay the log history so a page reload restores the full log.
        # the marker tells the client to clear its panes first so the replay
        # doesn't duplicate entries or double the counters.
        await ws.send_text(json.dumps({"type": "history_start"}))
        for entry in hub.history():
            await ws.send_text(json.dumps(entry))
        # only now join the live broadcast fan-out, so the replay above can't
        # be interleaved with a concurrent live event
        async with hub.lock:
            hub.clients.add(ws)
        while True:
            # dashboard doesn't need to send us anything; just keep the socket open
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(ws)


# ---------------------------------------------------------------------------
# The actual proxy. Catches any path so it works for /v1/chat/completions,
# /v1/completions, /v1/models, etc — anything the backend exposes.
# ---------------------------------------------------------------------------


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(full_path: str, request: Request):
    if full_path == "" or full_path == "ws":
        return JSONResponse({"error": "not found"}, status_code=404)

    body_bytes = await request.body()
    target_url = f"{provider_base_url()}/{full_path}"

    # try to parse body as JSON to extract prompt text + streaming flag;
    # if it's not JSON (e.g. GET /v1/models) just pass through untouched.
    body_json = None
    if body_bytes:
        try:
            body_json = json.loads(body_bytes)
        except Exception:
            body_json = None

    is_chatlike = body_json is not None and (
        "messages" in body_json or "prompt" in body_json
    )

    if not is_chatlike:
        # plain passthrough, no stats/logging (e.g. /v1/models, /health)
        return await _plain_passthrough(request, target_url, body_bytes)

    # Resolve image URLs to base64 data URLs (LM Studio requires base64).
    body_json = await _resolve_images_in_body(body_json)
    body_bytes = json.dumps(body_json).encode("utf-8")

    req_id = str(uuid.uuid4())[:8]
    model = body_json.get("model", "unknown")
    prompt_text = extract_prompt_text(body_json)
    prompt_tokens_est = estimate_tokens(prompt_text)
    wants_stream = bool(body_json.get("stream", False))
    settings = extract_settings(body_json)

    t_request_start = time.monotonic()

    pp_tps_est = CONFIG["pp_tokens_per_sec"].get(
        model, CONFIG["pp_tokens_per_sec"]["_default"]
    )
    est_pp_seconds = (prompt_tokens_est / pp_tps_est) if pp_tps_est > 0 else None

    stats.total_requests += 1
    stats.active_requests[req_id] = {
        "id": req_id,
        "model": model,
        "phase": "queued",
        "prompt_tokens_est": prompt_tokens_est,
        "started_at": t_request_start,
        "est_pp_seconds": est_pp_seconds,
        "pp_progress_pct": 0,
        "decode_tps": None,
        "tokens_generated": 0,
        "cancelled": False,
    }
    # Keep the running task separately (not in the entry — entries are
    # JSON-serialized for the dashboard, a Task is not serializable).
    request_tasks[req_id] = asyncio.current_task()

    # Dedup: find the most recent similar prompt and report only the added tail.
    match = prompt_history.best_match(prompt_text)
    if match:
        matched_text, prefix_len = match
        is_diff = True
        repeated_len = prefix_len
        # drop the leading separator so the added tail renders cleanly
        added_text = prompt_text[prefix_len:].lstrip("\n")
    else:
        is_diff = False
        repeated_len = 0
        added_text = prompt_text

    await hub.emit(
        {
            "type": "prompt",
            "id": req_id,
            "model": model,
            "settings": settings,
            "text": prompt_text,
            "prompt_tokens_est": prompt_tokens_est,
            "est_pp_seconds": est_pp_seconds,
            "is_diff": is_diff,
            "repeated_len": repeated_len,
            "added_text": added_text,
            "ts": time.time(),
        }
    )
    prompt_history.add(prompt_text)

    # ---- FIFO queue: wait our turn before touching the upstream ----
    # Position in line = requests already waiting + 1 (ourselves).
    queue_position = gen_queue.waiting + 1
    await hub.emit(
        {
            "type": "queued",
            "id": req_id,
            "model": model,
            "position": queue_position,
            "ts": time.time(),
        }
    )

    t_queued_start = time.monotonic()
    try:
        await gen_queue.acquire()
    except asyncio.CancelledError:
        # Cancelled while waiting (the user hit the x button). Drop both
        # registries and let the cancellation propagate so this request never
        # reaches the upstream.
        stats.active_requests.pop(req_id, None)
        request_tasks.pop(req_id, None)
        raise
    wait_seconds = time.monotonic() - t_queued_start

    # We're first in line — the upstream is ours until we release below.
    entry = stats.active_requests.get(req_id)
    if entry:
        entry["phase"] = "prefill"
    await hub.emit(
        {
            "type": "started",
            "id": req_id,
            "model": model,
            "wait_seconds": round(wait_seconds, 3),
            "ts": time.time(),
        }
    )

    # kick off a small background task that fakes smooth PP-progress ticks
    # between "request sent" and "first token received", based on the
    # estimate. purely cosmetic — cancelled the instant real data arrives.
    progress_task = asyncio.create_task(
        _tick_pp_progress(req_id, est_pp_seconds)
    )

    client = httpx.AsyncClient(timeout=None)

    try:
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        upstream_req = client.build_request(
            request.method, target_url, headers=headers, content=body_bytes
        )
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as e:
        progress_task.cancel()
        gen_queue.release()  # don't hold the slot for a request that never ran
        stats.active_requests.pop(req_id, None)
        request_tasks.pop(req_id, None)
        await hub.emit(
            {"type": "error", "id": req_id, "text": f"Could not reach provider: {e}"}
        )
        await client.aclose()
        return JSONResponse(
            {"error": f"Glaiss: could not reach provider at {target_url}: {e}"},
            status_code=502,
        )

    if wants_stream:
        return StreamingResponse(
            _stream_and_tap(
                upstream_resp, client, req_id, model, prompt_tokens_est, progress_task
            ),
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
        )
    else:
        # non-streaming response: read it whole, then report as one shot
        try:
            content = await upstream_resp.aread()
        except asyncio.CancelledError:
            # Cancelled while reading the response: do the critical cleanup
            # (release the slot, drop both registries), offload the close so it
            # is safe under cancellation, then let the cancellation propagate.
            progress_task.cancel()
            gen_queue.release()
            stats.active_requests.pop(req_id, None)
            request_tasks.pop(req_id, None)
            asyncio.create_task(_close_stream(upstream_resp, client))
            raise

        await upstream_resp.aclose()
        await client.aclose()
        progress_task.cancel()
        gen_queue.release()  # upstream is done generating — hand the slot on

        t_first = time.monotonic()
        pp_seconds = t_first - t_request_start
        response_text = ""
        completion_tokens = None
        try:
            parsed = json.loads(content)
            response_text = _extract_completion_text(parsed)
            usage = parsed.get("usage", {})
            completion_tokens = usage.get("completion_tokens")
            prompt_tokens_actual = usage.get("prompt_tokens", prompt_tokens_est)
        except Exception:
            prompt_tokens_actual = prompt_tokens_est

        calibrate_pp(model, prompt_tokens_actual, pp_seconds)
        stats.recent_pp_tps.append(
            prompt_tokens_actual / pp_seconds if pp_seconds > 0 else 0
        )
        if completion_tokens:
            stats.total_tokens_generated += completion_tokens

        await hub.emit(
            {
                "type": "response",
                "id": req_id,
                "model": model,
                "text": response_text,
                "done": True,
                "pp_seconds": round(pp_seconds, 3),
                "completion_tokens": completion_tokens,
                "tokens_generated": completion_tokens,
                "ts": time.time(),
            }
        )
        stats.active_requests.pop(req_id, None)
        request_tasks.pop(req_id, None)
        return StreamingResponse(
            iter([content]),
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type", "application/json"),
        )


async def _plain_passthrough(request: Request, target_url: str, body_bytes: bytes):
    client = httpx.AsyncClient(timeout=None)
    try:
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        upstream_req = client.build_request(
            request.method, target_url, headers=headers, content=body_bytes
        )
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as e:
        await client.aclose()
        return JSONResponse(
            {"error": f"Glaiss: could not reach provider at {target_url}: {e}"},
            status_code=502,
        )

    async def gen():
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
        finally:
            await upstream_resp.aclose()
            await client.aclose()

    return StreamingResponse(
        gen(),
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type"),
    )


async def _tick_pp_progress(req_id: str, est_pp_seconds: Optional[float]):
    """Cosmetic smooth progress ticking while waiting for the first token."""
    if not est_pp_seconds or est_pp_seconds <= 0:
        return
    start = time.monotonic()
    try:
        while True:
            await asyncio.sleep(0.1)
            elapsed = time.monotonic() - start
            pct = min(99, int((elapsed / est_pp_seconds) * 100))
            entry = stats.active_requests.get(req_id)
            if entry is None:
                return
            entry["pp_progress_pct"] = pct
    except asyncio.CancelledError:
        return


def _sse_extract_delta(line: str):
    """Parse one OpenAI-style SSE 'data: {...}' line.

    Returns a dict with:
      content:   str        — response text delta (may be empty)
      reasoning: str        — thinking/reasoning delta (may be empty)
      usage:     dict|None  — the usage object (present on the final chunk)
      is_done:   bool       — True for the [DONE] marker
    Returns None if the line is not an SSE data line we care about.
    """
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return {"is_done": True}
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    result = {"content": "", "reasoning": "", "usage": obj.get("usage")}
    choices = obj.get("choices", [])
    if choices:
        delta = choices[0].get("delta") or choices[0].get("text")
        if isinstance(delta, dict):
            result["content"] = delta.get("content") or ""
            result["reasoning"] = delta.get("reasoning_content") or ""
        elif isinstance(delta, str):
            result["content"] = delta
    return result


def _extract_completion_text(parsed: dict) -> str:
    choices = parsed.get("choices", [])
    if not choices:
        return ""
    c = choices[0]
    if "message" in c:
        return c["message"].get("content", "") or ""
    return c.get("text", "") or ""


async def _close_stream(upstream_resp, client: httpx.AsyncClient):
    """Best-effort close of an upstream response + client.

    Swallows errors so it is safe to run off the critical path — in particular
    when the owning request is being cancelled and the loop may interrupt the
    awaits. Never raises.
    """
    try:
        await upstream_resp.aclose()
    except Exception:
        pass
    try:
        await client.aclose()
    except Exception:
        pass


async def _stream_and_tap(
    upstream_resp,
    client: httpx.AsyncClient,
    req_id: str,
    model: str,
    prompt_tokens_est: int,
    progress_task: asyncio.Task,
):
    t_request_start = stats.active_requests[req_id]["started_at"]
    t_first_token: Optional[float] = None
    thinking_tokens = 0  # reasoning/thinking tokens generated
    content_tokens = 0   # response (non-thinking) tokens generated
    buf = ""
    full_text = ""       # accumulated response (content) text
    full_thinking = ""   # accumulated thinking text
    usage: Optional[dict] = None  # provider's final usage chunk (accurate counts)

    try:
        async for raw_chunk in upstream_resp.aiter_bytes():
            yield raw_chunk  # forward untouched to the real client, immediately

            buf += raw_chunk.decode("utf-8", errors="ignore")
            while "\n\n" in buf:
                line, buf = buf.split("\n\n", 1)
                line = line.strip()
                if not line:
                    continue
                d = _sse_extract_delta(line)
                if d is None or d.get("is_done"):
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                    continue
                content = d.get("content", "")
                reasoning = d.get("reasoning", "")
                if not content and not reasoning:
                    continue

                now = time.monotonic()
                # The first token of ANY kind (thinking or response) marks the
                # end of prefill — so pp_seconds is the true time-to-first-token,
                # not time-to-first-response-token (which would include thinking).
                if t_first_token is None:
                    t_first_token = now
                    progress_task.cancel()
                    pp_seconds = t_first_token - t_request_start
                    calibrate_pp(model, prompt_tokens_est, pp_seconds)
                    stats.recent_pp_tps.append(
                        prompt_tokens_est / pp_seconds if pp_seconds > 0 else 0
                    )
                    entry = stats.active_requests.get(req_id)
                    if entry:
                        entry["phase"] = "decode"
                        entry["pp_progress_pct"] = 100
                        entry["pp_seconds"] = round(pp_seconds, 3)

                # Count + stream each kind. Thinking and response are separate
                # streams, both forwarded to the log UI in real time.
                if reasoning:
                    thinking_tokens += 1
                    full_thinking += reasoning
                    await hub.broadcast(
                        {
                            "type": "response",
                            "id": req_id,
                            "model": model,
                            "kind": "thinking",
                            "text": reasoning,
                            "done": False,
                            "ts": time.time(),
                        }
                    )
                if content:
                    content_tokens += 1
                    full_text += content
                    await hub.broadcast(
                        {
                            "type": "response",
                            "id": req_id,
                            "model": model,
                            "kind": "content",
                            "text": content,
                            "done": False,
                            "ts": time.time(),
                        }
                    )

                # Decode speed = all generated tokens (thinking + response) over
                # the total generation time, so it is not thrown off by thinking.
                total = thinking_tokens + content_tokens
                entry = stats.active_requests.get(req_id)
                if entry:
                    entry["tokens_generated"] = total
                    elapsed_decode = now - (t_first_token or now)
                    if elapsed_decode > 0:
                        entry["decode_tps"] = round(total / elapsed_decode, 1)
    finally:
        progress_task.cancel()
        gen_queue.release()  # sync — always runs, hands the slot to next in line
        entry = stats.active_requests.pop(req_id, None)  # sync — always runs
        request_tasks.pop(req_id, None)  # sync — always runs
        if entry and entry.get("decode_tps"):
            stats.recent_decode_tps.append(entry["decode_tps"])
        total = thinking_tokens + content_tokens
        # Prefer the provider's authoritative usage chunk for the final counts:
        # completion_tokens = thinking + response; reasoning_tokens = thinking.
        if usage:
            total = usage.get("completion_tokens", total)
            reasoning_tokens = (
                (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                )
                or thinking_tokens
            )
            thinking_tokens = reasoning_tokens
            content_tokens = max(0, total - reasoning_tokens)
        stats.total_tokens_generated += total
        was_cancelled = bool(entry and entry.get("cancelled"))
        # Best-effort close, offloaded so it is safe even while we are being
        # cancelled (no awaits in this finally -> CancelledError propagates).
        asyncio.create_task(_close_stream(upstream_resp, client))
        if was_cancelled:
            # No "done" message — the request was aborted. The CancelledError
            # (if this is the cancel path) continues to propagate.
            return
        await hub.emit(
            {
                "type": "response",
                "id": req_id,
                "model": model,
                "kind": "content",
                "text": full_text,  # full response so a replay can reconstruct it
                "reasoning": full_thinking,  # full thinking text
                "done": True,
                "ts": time.time(),
                "final_decode_tps": entry.get("decode_tps") if entry else None,
                "tokens_generated": total,
                "thinking_tokens": thinking_tokens,
                "content_tokens": content_tokens,
                "pp_seconds": entry.get("pp_seconds") if entry else None,
            }
        )


if __name__ == "__main__":
    print(f"Glaiss listening on 0.0.0.0:{CONFIG['listen_port']}")
    print(f"Proxying to provider at {provider_base_url()}")
    print(f"Point your clients at: http://<this-host>:{CONFIG['listen_port']}/v1/...")
    uvicorn.run(app, host="0.0.0.0", port=CONFIG["listen_port"])
