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

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "glaiss_config.json"
STATIC_DIR = BASE_DIR / "static"

DEFAULT_CONFIG = {
    "listen_port": 8899,
    "provider_host": "127.0.0.1",
    "provider_port": 8000,
    "provider_scheme": "http",
    "models": [],  # list of model name strings shown as configured/passed-through
    # rolling calibration of tokens/sec prefill speed, keyed by model name.
    # "_default" is used when a model has no calibration yet or model name is unknown.
    "pp_tokens_per_sec": {"_default": 1500.0},
}

# ---------------------------------------------------------------------------
# Config load/save
# ---------------------------------------------------------------------------

_cfg_lock = asyncio.Lock()


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
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


CONFIG = load_config()


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


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self.lock:
            self.clients.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self.lock:
            self.clients.discard(ws)

    async def broadcast(self, message: dict):
        dead = []
        payload = json.dumps(message)
        async with self.lock:
            targets = list(self.clients)
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


hub = Hub()

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
            "provider": provider_base_url(),
            "models": CONFIG["models"],
        }


stats = Stats()

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
    return FileResponse(BASE_DIR / "index.html")


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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        # send an immediate snapshot + config so the page has state right away
        await ws.send_text(json.dumps(stats.snapshot()))
        await ws.send_text(json.dumps({"type": "config", "config": CONFIG}))
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

    req_id = str(uuid.uuid4())[:8]
    model = body_json.get("model", "unknown")
    prompt_text = extract_prompt_text(body_json)
    prompt_tokens_est = estimate_tokens(prompt_text)
    wants_stream = bool(body_json.get("stream", False))

    t_request_start = time.monotonic()

    pp_tps_est = CONFIG["pp_tokens_per_sec"].get(
        model, CONFIG["pp_tokens_per_sec"]["_default"]
    )
    est_pp_seconds = (prompt_tokens_est / pp_tps_est) if pp_tps_est > 0 else None

    stats.total_requests += 1
    stats.active_requests[req_id] = {
        "id": req_id,
        "model": model,
        "phase": "prefill",
        "prompt_tokens_est": prompt_tokens_est,
        "started_at": t_request_start,
        "est_pp_seconds": est_pp_seconds,
        "pp_progress_pct": 0,
        "decode_tps": None,
        "tokens_generated": 0,
    }

    await hub.broadcast(
        {
            "type": "prompt",
            "id": req_id,
            "model": model,
            "text": prompt_text,
            "prompt_tokens_est": prompt_tokens_est,
            "est_pp_seconds": est_pp_seconds,
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
        stats.active_requests.pop(req_id, None)
        await hub.broadcast(
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
        content = await upstream_resp.aread()
        await upstream_resp.aclose()
        await client.aclose()
        progress_task.cancel()

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

        await hub.broadcast(
            {
                "type": "response",
                "id": req_id,
                "model": model,
                "text": response_text,
                "done": True,
                "pp_seconds": round(pp_seconds, 3),
                "completion_tokens": completion_tokens,
                "ts": time.time(),
            }
        )
        stats.active_requests.pop(req_id, None)
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


def _sse_extract_delta_text(line: str) -> Optional[str]:
    """Pull the text delta out of one OpenAI-style SSE 'data: {...}' line."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    choices = obj.get("choices", [])
    if not choices:
        return None
    delta = choices[0].get("delta") or choices[0].get("text")
    if isinstance(delta, dict):
        return delta.get("content") or ""
    if isinstance(delta, str):
        return delta
    return None


def _extract_completion_text(parsed: dict) -> str:
    choices = parsed.get("choices", [])
    if not choices:
        return ""
    c = choices[0]
    if "message" in c:
        return c["message"].get("content", "") or ""
    return c.get("text", "") or ""


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
    tokens_seen = 0
    buf = ""

    try:
        async for raw_chunk in upstream_resp.aiter_bytes():
            yield raw_chunk  # forward untouched to the real client, immediately

            buf += raw_chunk.decode("utf-8", errors="ignore")
            while "\n\n" in buf:
                line, buf = buf.split("\n\n", 1)
                line = line.strip()
                if not line:
                    continue
                delta_text = _sse_extract_delta_text(line)
                if delta_text is None:
                    continue
                now = time.monotonic()
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

                if delta_text:
                    tokens_seen += 1
                    entry = stats.active_requests.get(req_id)
                    if entry:
                        entry["tokens_generated"] = tokens_seen
                        elapsed_decode = now - (t_first_token or now)
                        if elapsed_decode > 0:
                            tps = tokens_seen / elapsed_decode
                            entry["decode_tps"] = round(tps, 1)

                    await hub.broadcast(
                        {
                            "type": "response",
                            "id": req_id,
                            "model": model,
                            "text": delta_text,
                            "done": False,
                            "ts": time.time(),
                        }
                    )
    finally:
        progress_task.cancel()
        await upstream_resp.aclose()
        await client.aclose()
        entry = stats.active_requests.pop(req_id, None)
        if entry and entry.get("decode_tps"):
            stats.recent_decode_tps.append(entry["decode_tps"])
        stats.total_tokens_generated += tokens_seen
        await hub.broadcast(
            {
                "type": "response",
                "id": req_id,
                "model": model,
                "text": "",
                "done": True,
                "ts": time.time(),
                "final_decode_tps": entry.get("decode_tps") if entry else None,
                "tokens_generated": tokens_seen,
            }
        )


if __name__ == "__main__":
    print(f"Glaiss listening on 0.0.0.0:{CONFIG['listen_port']}")
    print(f"Proxying to provider at {provider_base_url()}")
    print(f"Point your clients at: http://<this-host>:{CONFIG['listen_port']}/v1/...")
    uvicorn.run(app, host="0.0.0.0", port=CONFIG["listen_port"])
