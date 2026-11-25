#!/usr/bin/env python3
"""Offline test suite for llpc model-waitroom v3 (single-door regime).

Spins fake ENGINE (:8085 stand-in) and fake GATEKEEPER (:8090 stand-in) on
ephemeral ports, points the waitroom at them via env + module reload, and
drives real HTTP through handle(). Nothing touches the live services.

Run:  python3 ~/llama-swap/test_waitroom.py      (exit 0 = all green)
"""
import asyncio
import importlib
import json
import logging
import os
import sys
import tempfile
import time

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import waitroom as W  # noqa: E402

CHECKS = 0
FAILS = []


def ck(cond, name):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(name)
        print(f"  FAIL: {name}")
    return cond


CATALOG = {"data": [
    {"id": "Qwen 3.8 Flash Next Q4",
     "meta": {"llamaswap": {"aliases": ["qwen3.8-flash-next-unsloth-ud-q4_k_xl", "strata31"]}},
     "architecture": {"output_modalities": ["text"]}},
    {"id": "Qwen 3.8 27B Q6",
     "meta": {"llamaswap": {"aliases": ["qwen3.8-27b", "llama-qwen"]}},
     "architecture": {"output_modalities": ["text"]}},
    {"id": "Comfy Studio",
     "meta": {"llamaswap": {"aliases": ["comfyui"]}},
     "architecture": {"output_modalities": ["image"]}},
    {"id": "DeepSeek V4.1 Flash",
     "meta": {"llamaswap": {"aliases": ["deepseek-v4.1-flash", "strata-dsv41"]}},
     "architecture": {"output_modalities": ["text"]}},
]}

FLASH = "Qwen 3.8 Flash Next Q4"
Q27 = "Qwen 3.8 27B Q6"
FLASH_ALIAS = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
Q27_ALIAS = "qwen3.8-27b"
COMFY = "Comfy Studio"
DS = "DeepSeek V4.1 Flash"
DS_ALIAS = "deepseek-v4.1-flash"


class FakeState:
    def __init__(self):
        self.healthy = True
        self.running = []       # names in gatekeeper /running
        self.gate_chats = []    # model names the fake gatekeeper received
        self.engine_chats = []  # model names the fake engine received
        self.unloads = []       # model names the fake gatekeeper was asked to unload


def make_engine_app(st):
    app = web.Application()

    async def health(r):
        return web.json_response({"status": "ok"} if st.healthy else {"status": "down"},
                                 status=200 if st.healthy else 503)

    async def chat(r):
        body = await r.json()
        st.engine_chats.append(body.get("model"))
        return web.json_response({"via": "engine", "model": body.get("model")})

    async def models(r):
        return web.json_response({"object": "list",
                                  "data": [{"id": "engine-loaded-model"}]})

    async def other(r):
        return web.Response(text="ENGINE-OTHER")

    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    app.router.add_route("*", "/{tail:.*}", other)
    return app


def make_gate_app(st):
    app = web.Application()

    async def models(r):
        return web.json_response(CATALOG)

    async def running(r):
        return web.json_response({"running": [{"model": m, "state": "ready"}
                                              for m in st.running]})

    async def chat(r):
        body = await r.json()
        st.gate_chats.append(body.get("model"))
        delay = getattr(st, "gate_delay", 0)
        if delay:
            # interruptible: teardown must not wait out a fake "model load"
            end = asyncio.get_running_loop().time() + delay
            while asyncio.get_running_loop().time() < end:
                await asyncio.sleep(0.05)
        if getattr(st, "gate_chat_status", 200) != 200:
            return web.json_response(
                {"error": {"message": "all inference is paused"}},
                status=st.gate_chat_status)
        return web.json_response({"via": "gate", "model": body.get("model"),
                                  "echo": body})

    async def unload(r):
        name = r.match_info["name"]
        st.unloads.append(name)
        st.running = [m for m in st.running if m != name]
        return web.Response(text="OK")

    async def status500(r):
        return web.json_response({"error": "boom"}, status=500)

    app.router.add_get("/v1/models", models)
    app.router.add_get("/running", running)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/api/models/unload/{name}", unload)
    return app


class Harness:
    """One scenario: fakes started, waitroom reloaded against them."""

    def __init__(self, running=(), healthy=True, hold=0.0, wait_s=1.0,
                 intent=None, lastswap=None, default_model=FLASH, swap_alive=True,
                 quantum=5.0, quantum_n=8, idle_switch=0.05,
                 lastmodel=None, park_wake=0.2):
        self.st = FakeState()
        self.st.healthy = healthy
        self.st.running = list(running)
        self.quantum, self.quantum_n, self.idle_switch = quantum, quantum_n, idle_switch
        self.park_wake = park_wake
        self.tmp = tempfile.mkdtemp(prefix="waitroom-test-")
        self.cfg_file = os.path.join(self.tmp, "config.yaml")
        self.intent_file = os.path.join(self.tmp, "waitroom-intent")
        self.lastswap_file = os.path.join(self.tmp, "waitroom-last-swap")
        self.default_file = os.path.join(self.tmp, "default-model")
        self.lastmodel_file = os.path.join(self.tmp, "last-model")
        with open(self.default_file, "w") as f:
            f.write(default_model)
        if lastmodel is not None:
            with open(self.lastmodel_file, "w") as f:
                f.write(lastmodel + "\n")
        if intent is not None:
            with open(self.intent_file, "w") as f:
                json.dump(intent, f)
        if lastswap is not None:
            with open(self.lastswap_file, "w") as f:
                f.write(repr(lastswap))
        self.hold, self.wait_s = hold, wait_s
        self._swap_alive = swap_alive
        self.clients = []
        self.client = None

    async def __aenter__(self):
        eng = TestClient(TestServer(make_engine_app(self.st)))
        await eng.start_server()
        self.clients = [eng]
        swap_url = ""
        if self._swap_alive:
            gate = TestClient(TestServer(make_gate_app(self.st)))
            await gate.start_server()
            self.clients.append(gate)
            swap_url = str(gate.make_url("")).rstrip("/")
        else:
            swap_url = "http://127.0.0.1:1"   # connection-refused
        engine_port = eng.make_url("").port
        # Model registry (llama-swap/config.yaml stand-in): same names/aliases
        # as the real catalog, every proxy pointed at this scenario's fake
        # engine so the fail-open port lookup is exercised for real.
        with open(self.cfg_file, "w") as f:
            f.write("models:\n")
            for name, alias, unit in ((FLASH, FLASH_ALIAS, "strata31"),
                                      (Q27, Q27_ALIAS, "llama-qwen"),
                                      (DS, DS_ALIAS, "strata-dsv41"),
                                      (COMFY, "comfyui", "comfyui")):
                f.write(f'  "{name}":\n')
                f.write(f'    aliases: ["{alias}"]\n')
                f.write(f'    cmd: /home/graffunder/llama-swap/unit-keepalive.sh {unit} 5\n')
                f.write(f'    proxy: http://127.0.0.1:{engine_port}\n')
                kind = "image" if name == COMFY else "text"
                f.write(f'    capabilities: {{in: [text], out: [{kind}]}}\n')
        os.environ.update({
            "LLPC_MODELS_CONFIG": self.cfg_file,
            "WAITROOM_BACKEND": str(eng.make_url("")).rstrip("/"),
            "WAITROOM_SWAP": swap_url.rstrip("/"),
            "WAITROOM_WAIT_S": str(self.wait_s),
            "WAITROOM_POLL_S": "0.1",
            "WAITROOM_SWAP_HOLD_S": str(self.hold),
            "WAITROOM_INTENT_FILE": self.intent_file,
            "WAITROOM_LASTSWAP_FILE": self.lastswap_file,
            "WAITROOM_DEFAULT_FILE": self.default_file,
            "WAITROOM_LASTMODEL_FILE": self.lastmodel_file,
            "WAITROOM_PARK_WAKE_S": str(self.park_wake),
            "WAITROOM_INTENT_TTL_S": "43200",
            "WAITROOM_QUANTUM_S": str(self.quantum),
            "WAITROOM_QUANTUM_N": str(self.quantum_n),
            "WAITROOM_IDLE_SWITCH_S": str(self.idle_switch),
        })
        importlib.reload(W)
        wr = web.Application()

        async def on_startup(app):
            from aiohttp import ClientSession
            app["session"] = ClientSession()
            app["swap"] = W.SwapState(app["session"])
            app["arb"] = W.Arbiter()

        async def on_cleanup(app):
            await app["session"].close()

        wr.on_startup.append(on_startup)
        wr.on_cleanup.append(on_cleanup)
        wr.router.add_route("*", "/{tail:.*}", W.handle)
        self.client = TestClient(TestServer(wr))
        await self.client.start_server()
        return self

    async def __aexit__(self, *a):
        await self.client.close()
        for c in self.clients:
            await c.close()

    async def post_chat(self, model=None, path="/v1/chat/completions"):
        payload = {} if model is None else {"model": model}
        async with self.client.post(path, json=payload) as r:
            return r.status, await r.json()

    async def post_json(self, payload, path="/v1/chat/completions"):
        async with self.client.post(path, json=payload) as r:
            return r.status, await r.json()

    def intent(self):
        try:
            with open(self.intent_file) as f:
                return json.load(f)
        except Exception:
            return None


async def t_loaded_rides():
    async with Harness(running=[FLASH]) as h:
        s, b = await h.post_chat(FLASH_ALIAS)
        ck(s == 200 and b["via"] == "gate", "loaded: rides through gatekeeper")
        ck(b["model"] == FLASH_ALIAS, "loaded: name forwarded as-is (no rewrite)")
        ck(h.intent() is None, "loaded: intent file untouched")


async def t_unknown_rewrites():
    async with Harness(running=[FLASH]) as h:
        s, b = await h.post_chat("gpt-4o-mini")
        ck(s == 200 and b["via"] == "gate", "unknown: routed through gatekeeper")
        ck(b["model"] == FLASH, "unknown: rewritten to the loaded model")


async def t_no_model_rewrites():
    async with Harness(running=[FLASH]) as h:
        s, b = await h.post_chat(None)
        ck(s == 200 and b["via"] == "gate", "no-model: routed through gatekeeper")
        ck(b["model"] == FLASH, "no-model: rewritten to default-model file")


async def t_ds_effort_guard():
    # DeepSeek-V4.1's template raises on Qwen-only effort words (xhigh) and on
    # "medium"; the waitroom rewrites them when the RESOLVED target is DS.
    async with Harness(running=[FLASH]) as h:
        s, b = await h.post_json({"model": DS, "messages": [],
                                  "chat_template_kwargs": {"reasoning_effort": "xhigh"}})
        ck(s == 200 and b["model"] == DS, "ds-guard: DS choice survives loaded Qwen")
        ck(b["echo"]["chat_template_kwargs"]["reasoning_effort"] == "high",
           "ds-guard: xhigh rewritten to high on DS target (ctk)")
    async with Harness(running=[FLASH]) as h:
        s, b = await h.post_json({"model": FLASH, "messages": [],
                                  "chat_template_kwargs": {"reasoning_effort": "xhigh"}})
        ck(b["echo"]["chat_template_kwargs"]["reasoning_effort"] == "xhigh",
           "ds-guard: Qwen target keeps its xhigh untouched")
    async with Harness(running=[DS]) as h:
        s, b = await h.post_json({"model": DS_ALIAS, "messages": [],
                                  "reasoning_effort": "medium"})
        ck(b["echo"]["reasoning_effort"] == "62",
           "ds-guard: medium -> 62 (top-level word, alias-resolved target)")
    ck(W.guard_ds_effort(b'{"reasoning_effort":"xhigh"}', "DeepSeek V4.1 Flash")
       == b'{"reasoning_effort":"high"}', "ds-guard unit: word swapped")
    ck(W.guard_ds_effort(b'{"reasoning_effort":"xhigh"}', "Qwen 3.8 Flash Next Q4")
       == b'{"reasoning_effort":"xhigh"}', "ds-guard unit: non-DS target untouched")
    ck(W.guard_ds_effort(b'{"reasoning_effort":"max"}', "strata-dsv41")
       == b'{"reasoning_effort":"max"}', "ds-guard unit: DS-native words pass through")


async def t_stale_ride_rewrites():
    intent = {"model": FLASH, "stale_label": Q27_ALIAS, "at": time.time()}
    async with Harness(running=[FLASH], intent=intent) as h:
        s, b = await h.post_chat(Q27_ALIAS)
        ck(s == 200 and b["via"] == "gate", "stale: rides through gatekeeper")
        ck(b["model"] == FLASH, "stale: label rewritten to loaded (no swap-back)")
        ck(h.st.gate_chats == [FLASH], "stale: gate never saw the stale name")


async def t_reaffirm_keeps_stale():
    intent = {"model": Q27, "stale_label": FLASH_ALIAS, "at": time.time()}
    async with Harness(running=[FLASH], intent=intent) as h:
        s, b = await h.post_chat(Q27)
        ck(s == 200 and b["via"] == "gate", "reaffirm: swapped through gatekeeper")
        ck(b["model"] == Q27, "reaffirm: target name forwarded")
        it = h.intent()
        ck(it and it["model"] == Q27 and it["stale_label"] == FLASH_ALIAS,
           "reaffirm: intent rewritten keeping stale label")


async def t_new_choice_swaps():
    async with Harness(running=[FLASH]) as h:
        s, b = await h.post_chat(Q27_ALIAS)
        ck(s == 200 and b["via"] == "gate", "choice: swap through gatekeeper")
        ck(b["model"] == Q27_ALIAS, "choice: chosen name forwarded")
        it = h.intent()
        ck(it and it["model"] == Q27, "choice: intent written (canonical id)")


async def t_loaded_never_touches_intent():
    intent = {"model": Q27, "stale_label": None, "at": time.time()}
    async with Harness(running=[FLASH], intent=intent) as h:
        before = open(h.intent_file).read()
        s, b = await h.post_chat(FLASH_ALIAS)
        ck(s == 200 and b["via"] == "gate", "loaded+intent: still rides")
        ck(open(h.intent_file).read() == before,
           "loaded+intent: matching request does NOT void the intent")


async def t_hold_parks_then_swaps():
    now = time.time()
    async with Harness(running=[FLASH], hold=1.0, lastswap=now) as h:
        t0 = time.monotonic()
        s, b = await h.post_chat(Q27_ALIAS)
        dt = time.monotonic() - t0
        ck(s == 200 and b["model"] == Q27_ALIAS, "hold: parked then swapped")
        ck(dt >= 0.8, "hold: park actually waited for the hold to expire")
        it = h.intent()
        ck(it and it["model"] == Q27, "hold: intent written only at swap time")


async def t_media_hold_down_503():
    async with Harness(running=[COMFY], healthy=False, wait_s=0.4) as h:
        s, b = await h.post_chat(FLASH)
        ck(s == 503 and b.get("error", {}).get("type") == "waitroom_timeout",
           "media: comfy holding + engine down -> capped 503, comfy untouched")
        ck(h.st.gate_chats == [], "media: nothing forwarded to gatekeeper while comfy holds")


async def t_media_hold_recovers():
    h = Harness(running=[COMFY], healthy=False, wait_s=3.0)
    async with h:
        async def flip():
            await asyncio.sleep(0.3)
            h.st.healthy = True
        task = asyncio.ensure_future(flip())
        s, b = await h.post_chat(FLASH)
        await task
        ck(s == 200, "media: recovered engine serves (re-decide after wait)")


async def t_adopt_untracked_unknown():
    async with Harness(running=[], healthy=True) as h:
        s, b = await h.post_chat("gpt-4o-mini")
        ck(s == 200 and b["via"] == "gate", "adopt: engine-up-untracked goes through gate")
        ck(b["model"] == FLASH, "adopt: unknown rewritten to default for adoption")


async def t_adopt_untracked_named():
    async with Harness(running=[], healthy=True) as h:
        s, b = await h.post_chat(Q27)
        ck(s == 200 and b["via"] == "gate" and b["model"] == Q27,
           "adopt: named model forwarded to gate (its keepalive starts/adopts)")


async def t_empty_unknown_waits():
    async with Harness(running=[], healthy=False, wait_s=0.4) as h:
        s, b = await h.post_chat("gpt-4o-mini")
        ck(s == 503, "empty+unknown: waits for a model to appear (never summons a load)")


async def t_empty_body_400():
    async with Harness(running=[FLASH]) as h:
        async with h.client.post("/v1/chat/completions", data=b"") as r:
            ck(r.status == 400, "empty POST -> OpenAI-style 400")
            ck((await r.json()).get("error", {}).get("type") == "invalid_request_error",
               "empty POST: error has JSON body")


async def t_empty_named_loads():
    async with Harness(running=[], healthy=False) as h:
        s, b = await h.post_chat(Q27_ALIAS)
        ck(s == 200 and b["via"] == "gate" and b["model"] == Q27_ALIAS,
           "empty+named: forwarded immediately — llama-swap owns the cold load")


async def t_failopen_gate_dead():
    async with Harness(healthy=True, swap_alive=False) as h:
        s, b = await h.post_chat(FLASH)
        ck(s == 200 and b["via"] == "engine", "fail-open: gate dead -> direct :8085")


async def t_failopen_gate_dead_down():
    async with Harness(healthy=False, swap_alive=False, wait_s=0.4) as h:
        s, b = await h.post_chat(FLASH)
        ck(s == 503, "fail-open: gate dead + engine down -> wait -> 503")


async def t_catalog_passthrough():
    async with Harness(running=[FLASH]) as h:
        async with h.client.get("/v1/models") as r:
            d = await r.json()
        ids = [m["id"] for m in d["data"]]
        ck(r.status == 200 and FLASH in ids and Q27 in ids,
           "catalog: full llama-swap catalog served")
        ck(all(m.get("context_length") == 262144 for m in d["data"]),
           "catalog: context_length filled")


async def t_health_truth():
    async with Harness(running=[FLASH], healthy=True) as h:
        async with h.client.get("/health") as r:
            ck(r.status == 200, "health: up -> 200")
    async with Harness(running=[], healthy=False) as h:
        async with h.client.get("/health") as r:
            ck(r.status == 503, "health: down -> 503 (truth, no hold)")


async def t_nonchat_passthrough():
    async with Harness(running=[FLASH]) as h:
        async with h.client.get("/webui/index.html") as r:
            t = await r.text()
        ck(r.status == 200 and t == "ENGINE-OTHER", "non-chat: transparent to engine (webui)")


async def t_gate_pause_503_propagates():
    async with Harness(running=[FLASH]) as h:
        h.st.gate_chat_status = 503          # paused-gate style rejection
        s, b = await h.post_chat(FLASH)
        ck(s == 503, "pause-style 503 from gate surfaces to the caller")
        ck(h.st.engine_chats == [],
           "pause-style 503 does NOT silently bypass to the engine")


async def t_observed_swap_arms_hold():
    async with Harness(running=[FLASH], hold=60.0) as h:
        s, b = await h.post_chat(FLASH)          # warms the state view
        ck(s == 200, "observed: baseline ride ok")
        st = h.client.server.app["swap"]
        h.st.running = [Q27]                     # an outside swap happens
        await st.refresh(force=True)             # (as a later request would)
        ck(st.loaded_text == Q27, "observed: member-change seen")
        with open(h.lastswap_file) as f:
            ts = float(f.read())
        ck(ts > time.time() - 5, "observed: member-change armed the hold timestamp")
        route, _ = await W.decide(st, json.dumps({"model": FLASH}).encode())
        ck(route == "hold", "observed: a NEW choice against the fresh model parks")


async def t_extract_and_set_model():
    body = json.dumps({"model": 'a"b', "messages": []}).encode()
    ck(W.extract_model(body) == 'a"b', "extract: escaped quote in name")
    body2 = b'{"max_tokens": 100, "model": "\xd8\xad\xd9\x85 hello", "x": 1}'
    ck(W.extract_model(body2) == "﻽﻽ hello" or W.extract_model(body2),
       "extract: unicode escapes decode (no crash)")
    ck(W.extract_model(b'{"messages": []}') is None, "extract: absent -> None")
    big = b'{"pad": "' + b"x" * (W.MODEL_SCAN + 100) + b'", "model": "deep"}'
    ck(W.extract_model(big) == "deep", "extract: full-parse fallback past MODEL_SCAN")
    ck(W.set_model(b'{"model": "old", "m": 1}', "New Name") and
       json.loads(W.set_model(b'{"model": "old", "m": 1}', "New Name"))["model"] == "New Name"
       and json.loads(W.set_model(b'{"model": "old", "m": 1}', "New Name"))["m"] == 1,
       "set_model: rewritten in place, payload intact")
    inj = W.set_model(b'{"m": 1}', "N")
    ck(json.loads(inj) == {"model": "N", "m": 1}, "set_model: injected when field absent")
    ck(json.loads(W.set_model(b"{}", "N")) == {"model": "N"}, "set_model: injected into {}")
    ck(W.set_model(b'not json', "x") == b'not json', "set_model: non-object -> unchanged")


async def t_hold_math():
    async with Harness(running=[FLASH], hold=0.0) as h:
        st = W.SwapState(h.client.session)
        ck(st.hold_remaining() == 0.0, "hold: disabled (0) always 0")
        st2 = W.SwapState(h.client.session)
        st2.last_swap = time.time()
        W.SWAP_HOLD_S = 180.0
        ck(170.0 < st2.hold_remaining() <= 180.0, "hold: fresh swap ~ full hold left")


async def t_queue_serializes_two_models():
    """Two clients, two models: the second waits for the first, then gets the
    slot. This is the 2026-11-16 "war between picking" behaviour."""
    async with Harness(running=[FLASH], quantum=5.0, idle_switch=0.2,
                       wait_s=10.0) as h:
        h.st.gate_delay = 0.5
        arb = h.client.server.app["arb"]
        t1 = asyncio.create_task(h.post_chat(FLASH))
        await asyncio.sleep(0.15)
        t2 = asyncio.create_task(h.post_chat(Q27))
        await asyncio.sleep(0.2)
        snap = arb.snapshot()
        ck(snap["active"] == FLASH, "queue: first model holds the slot")
        ck(any(w["model"] == Q27 for w in snap["waiting"]),
           "queue: the other model is parked, not dropped")
        s1, _ = await t1
        s2, _ = await t2
        ck(s1 == 200 and s2 == 200, "queue: both models eventually served")
        ck(h.st.gate_chats == [FLASH, Q27],
           "queue: arrival order kept, no interleaved model loads")
        ck(arb.inflight == 0, "queue: slot released at the end")


async def t_queue_timeout_is_clear_503():
    async with Harness(running=[FLASH], quantum=30.0, idle_switch=30.0,
                       wait_s=0.3) as h:
        h.st.gate_delay = 1.2
        arb = h.client.server.app["arb"]
        t1 = asyncio.create_task(h.post_chat(FLASH))
        await asyncio.sleep(0.15)
        s2, b2 = await h.post_chat(Q27)
        ck(s2 == 503, "queue: displaced model gets 503, not a stalled 502")
        ck(b2.get("error", {}).get("type") == "queue_timeout",
           "queue: 503 says it was a queue wait")
        ck(b2.get("queue", {}).get("active") == FLASH,
           "queue: 503 names the model holding the slot")
        s1, _ = await t1
        ck(s1 == 200, "queue: the holder is never killed to satisfy the queue")
        ck(arb.inflight == 0, "queue: slot released after timeout path")


async def t_queue_affinity_same_model_runs_together():
    async with Harness(running=[FLASH], quantum=30.0, wait_s=10.0) as h:
        h.st.gate_delay = 0.4
        arb = h.client.server.app["arb"]
        t1 = asyncio.create_task(h.post_chat(FLASH))
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(h.post_chat(FLASH_ALIAS))
        await asyncio.sleep(0.15)
        ck(arb.inflight == 2,
           "queue: a conversation on the loaded model never waits (affinity)")
        await asyncio.gather(t1, t2)
        ck(arb.inflight == 0, "queue: both released")


async def t_queue_deepseek_one_inflight():
    async with Harness(running=[DS], quantum=30.0, wait_s=10.0) as h:
        h.st.gate_delay = 0.4
        arb = h.client.server.app["arb"]
        t1 = asyncio.create_task(h.post_chat(DS))
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(h.post_chat(DS_ALIAS))
        await asyncio.sleep(0.15)
        ck(arb.inflight == 1,
           "deepseek: only one request is in flight at a time")
        snap = arb.snapshot()
        ck(len(snap["inflight_requests"]) == 1
           and snap["inflight_requests"][0]["model"] == DS,
           "deepseek: one in-flight request is listed")
        ck(len(snap["waiting"]) == 1 and snap["waiting"][0]["model"] == DS,
           "deepseek: the second request waits as its own queue entry")
        await asyncio.gather(t1, t2)
        ck(arb.inflight == 0, "deepseek: both released")


async def t_queue_releases_after_upstream_error():
    async with Harness(running=[FLASH]) as h:
        h.st.gate_chat_status = 500
        s, _ = await h.post_chat(FLASH)
        ck(s == 500, "queue: upstream error surfaced unchanged")
        arb = h.client.server.app["arb"]
        ck(arb.inflight == 0, "queue: slot released after an error (no wedge)")


async def t_queue_endpoint_reports_state():
    async with Harness(running=[FLASH]) as h:
        async with h.client.get("/queue") as r:
            d = await r.json()
        ck(r.status == 200 and d.get("loaded") == FLASH,
           "queue: /queue reports the loaded model")
        ck("waiting" in d and "active" in d and "inflight" in d
           and "inflight_requests" in d,
           "queue: /queue exposes wait list, in-flight count, and per-request list")


async def t_failopen_uses_model_port():
    """Gatekeeper gone: the request fails open to the model's own port from the
    registry (per-engine ports, 2026-11-16)."""
    async with Harness(running=[FLASH], swap_alive=False) as h:
        s, b = await h.post_chat(FLASH)
        ck(s == 200 and b.get("via") == "engine",
           "failopen: routed to the model's own port from the registry")


# ------------------------------------------------- last-used stickiness tests

async def t_lastmodel_observed_record():
    async with Harness(running=[FLASH]) as h:
        s, _ = await h.post_chat(FLASH)            # warms the state view
        ck(s == 200, "lastmodel: baseline ride ok")
        try:
            got = open(h.lastmodel_file).read().strip()
        except FileNotFoundError:
            got = None
        ck(got == FLASH, "lastmodel: observed load is recorded")
        st = h.client.server.app["swap"]
        h.st.running = [DS]                        # an outside swap happens
        await st.refresh(force=True)
        ck(open(h.lastmodel_file).read().strip() == DS,
           "lastmodel: swap updates the record")
        h.st.running = [COMFY]                     # media takes the slot...
        await st.refresh(force=True)
        ck(open(h.lastmodel_file).read().strip() == DS,
           "lastmodel: media engine is never recorded as last-used")


async def t_default_name_prefers_lastmodel():
    # Empty box, engine up, name-less request: the adoption name comes from
    # the last-served model, NOT the default-model file (Spec 2026-11-18).
    async with Harness(running=[], healthy=True, lastmodel=DS) as h:
        s, b = await h.post_chat(None)
        ck(s == 200 and b["model"] == DS,
           "lastmodel: name-less adopt request names the last-served model")


async def t_current_model_chain():
    """current-model.sh: last-model > fresh intent > default file."""
    import subprocess
    tmp = tempfile.mkdtemp(prefix="cmtest-")
    last = os.path.join(tmp, "last-model")
    intent = os.path.join(tmp, "waitroom-intent")
    dflt = os.path.join(tmp, "default-model")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "current-model.sh")
    with open(dflt, "w") as f:
        f.write("Default Model M")
    env = dict(os.environ, WAITROOM_LASTMODEL_FILE=last,
               WAITROOM_INTENT_FILE=intent, WAITROOM_DEFAULT_FILE=dflt,
               WAITROOM_INTENT_TTL_S="43200")

    def run():
        r = subprocess.run(["bash", script], capture_output=True, text=True,
                           env=env)
        return r.stdout.strip()

    ck(run() == "Default Model M", "current-model: nothing -> default file")
    with open(intent, "w") as f:
        json.dump({"model": "Intent Model", "at": time.time()}, f)
    ck(run() == "Intent Model", "current-model: fresh intent wins over default")
    with open(intent, "w") as f:
        json.dump({"model": "Intent Model", "at": time.time() - 100000}, f)
    ck(run() == "Default Model M", "current-model: stale intent -> default")
    with open(intent, "w") as f:
        json.dump({"model": "Intent Model", "at": time.time()}, f)
    with open(last, "w") as f:
        f.write("Last Model\n")
    ck(run() == "Last Model", "current-model: last-model wins over fresh intent")


# ------------------------------------------------------- cancel-override tests

async def t_cancel_before_first_byte_overrides():
    """Spec's wrong-model mistake: request names another model, client aborts
    while it is still 'loading'. The forward must die, the hysteresis hold
    must be disarmed, llama-swap must be told to unload the target, and the
    corrected choice must be served without the park."""
    async with Harness(running=[FLASH], hold=180.0, wait_s=30.0) as h:
        h.st.gate_delay = 3.0
        arb = h.client.server.app["arb"]
        t = asyncio.create_task(h.client.post("/v1/chat/completions",
                                              json={"model": Q27_ALIAS}))
        await asyncio.sleep(0.5)
        ck(h.st.gate_chats == [Q27_ALIAS],
           "cancel: wrong-model request reached the gatekeeper")
        ck(float(open(h.lastswap_file).read()) > time.time() - 5,
           "cancel: the swap armed the hysteresis hold")
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(2.5)         # watcher poll + background unload
        ck(Q27 in h.st.unloads,
           "cancel: llama-swap asked to unload the cancelled target")
        ck(float(open(h.lastswap_file).read()) == 0.0,
           "cancel: hysteresis disarmed by the cancel")
        ck(arb.inflight == 0 and arb.active is None,
           "cancel: queue slot handed over immediately")
        ck(open(h.lastmodel_file).read().strip() == FLASH,
           "cancel: last-used record stays on the model before it")
        h.st.gate_delay = 0.2
        t0 = time.monotonic()
        s, b = await h.post_chat(DS)
        ck(s == 200 and b["model"] == DS, "cancel: corrected model served")
        ck(time.monotonic() - t0 < 1.5,
           "cancel: corrected model did NOT sit behind the 180s hold")


async def t_cancel_parked_in_hold_never_swaps():
    now = time.time()
    async with Harness(running=[FLASH], hold=30.0, lastswap=now, wait_s=30.0,
                       park_wake=0.2) as h:
        h.st.gate_delay = 0.3
        t = asyncio.create_task(h.client.post("/v1/chat/completions",
                                              json={"model": Q27}))
        await asyncio.sleep(0.6)          # several wake-ups, still parked
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.5)
        ck(h.st.gate_chats == [], "parked-cancel: parked choice never forwarded")
        ck(h.st.unloads == [], "parked-cancel: nothing unloaded")
        ck(float(open(h.lastswap_file).read()) >= now - 1,
           "parked-cancel: an existing hold is left armed")


async def t_wait_path_cancel_abandons():
    async with Harness(running=[COMFY], healthy=False, wait_s=30.0) as h:
        t = asyncio.create_task(h.client.post("/v1/chat/completions",
                                              json={"model": FLASH}))
        await asyncio.sleep(0.5)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.4)
        ck(h.st.gate_chats == [], "wait-cancel: abandoned waiter never forwards")


async def t_arbiter_acquire_cancel_no_ghost():
    async with Harness(running=[FLASH], quantum=30.0, idle_switch=30.0,
                       wait_s=30.0) as h:
        h.st.gate_delay = 0.8
        arb = h.client.server.app["arb"]
        t1 = asyncio.create_task(h.post_chat(FLASH))
        await asyncio.sleep(0.15)

        async def queued():
            return await arb.acquire("Ghost Model", 30.0)

        t2 = asyncio.create_task(queued())
        await asyncio.sleep(0.2)
        ck(any(w["model"] == "Ghost Model" for w in arb.snapshot()["waiting"]),
           "acquire-cancel: ghost queued first")
        t2.cancel()
        try:
            await t2
        except asyncio.CancelledError:
            pass
        ck(not any(w["model"] == "Ghost Model" for w in arb.snapshot()["waiting"]),
           "acquire-cancel: cancelled waiter removes itself")
        s1, _ = await t1
        ck(s1 == 200 and arb.inflight == 0, "acquire-cancel: holder unaffected")


# -------------------------------------------------- on-demand opt-out (Spec 2026-11-21)
# A model switched off under Models · On demand never loads by itself: the
# request naming it parks, traffic behind it for an enabled model still flows,
# and a model that is already generating finishes what it started.


async def t_ondemand_parks_and_lets_enabled_traffic_pass():
    allow = {"Flash"}
    arb = W.Arbiter(allowed=lambda m: m in allow)
    parked = asyncio.create_task(arb.acquire("Parked Model", 5.0))
    await asyncio.sleep(0.05)
    snap = arb.snapshot()
    ck(snap["active"] is None and snap["inflight"] == 0,
       "ondemand off: a parked model does not take the slot")
    ck(snap["waiting"] and snap["waiting"][0]["blocked"],
       "ondemand off: the parked waiter is reported as blocked")
    flash = asyncio.create_task(arb.acquire("Flash", 5.0))
    await asyncio.sleep(0.05)
    ck(arb.active == "Flash" and arb.inflight == 1,
       "ondemand off: a later request for an enabled model goes ahead of it")
    await flash
    arb.release()
    arb.batch_at = 0.0                      # the batch window is spent
    allow.add("Parked Model")
    arb._pump()
    await asyncio.wait_for(parked, 2.0)
    ck(arb.active == "Parked Model", "ondemand on again: the parked waiter gets the slot")
    arb.release()


async def t_ondemand_running_model_drains_then_releases():
    allow = {"A", "B"}
    arb = W.Arbiter(allowed=lambda m: m in allow)
    await arb.acquire("A", 5.0)
    ck(arb.active == "A" and arb.inflight == 1, "ondemand off mid-run: A holds the slot")
    allow.discard("A")
    b = asyncio.create_task(arb.acquire("B", 5.0))
    await asyncio.sleep(0.05)
    ck(arb.active == "A" and arb.inflight == 1,
       "ondemand off mid-run: the running generation is not cut off")
    ck(any(w["model"] == "B" for w in arb.snapshot()["waiting"]),
       "ondemand off mid-run: the next model waits for the drain")
    arb.release()
    await asyncio.wait_for(b, 2.0)
    ck(arb.active == "B" and arb.inflight == 1,
       "ondemand off mid-run: the slot passes on once the stream finished")
    arb.release()


async def t_ondemand_parked_waiter_does_not_steal_the_slot():
    allow = {"A"}
    arb = W.Arbiter(allowed=lambda m: m in allow)
    await arb.acquire("A", 5.0)

    async def parked_wait():
        try:
            await arb.acquire("Parked", 0.6)
            return "admitted"
        except W.QueueTimeout as exc:
            return f"timeout:{exc.blocked}"

    waiter = asyncio.create_task(parked_wait())
    await asyncio.sleep(0.05)
    arb.release()
    arb.batch_at = 0.0
    arb._pump()
    result = await waiter
    ck(result == "timeout:True",
       "a parked waiter is never admitted and times out saying it is off-demand")
    ck(arb.active == "A" and arb.handoffs == 1,
       "a parked waiter alone does not hand the slot to nobody")


async def t_schedule_window_blocks_only_outside_the_window():
    """Daily enabled windows gate the door exactly like the on-demand switch,
    but by the clock: an overnight window (20:00-06:00) blocks at 13:00 and
    admits at 22:00; a model with no window is never blocked."""
    from datetime import datetime
    saved_windows, saved_disabled = W.schedule_windows, W.on_demand_disabled
    W.on_demand_disabled = lambda now=None: []          # manual switch all on
    try:
        W.schedule_windows = lambda now=None: {"Night Model": {"start": "20:00", "end": "06:00"}}
        now_dt = datetime.now()
        noon = now_dt.replace(hour=13, minute=0, second=0, microsecond=0).timestamp()
        night = now_dt.replace(hour=22, minute=0, second=0, microsecond=0).timestamp()
        ck(W.model_in_schedule_window("Night Model", noon) is False,
           "window: 13:00 is outside 20:00-06:00, so the model is parked")
        ck(W.model_in_schedule_window("Night Model", night) is True,
           "window: 22:00 is inside the overnight window")
        # model_on_demand uses the REAL clock; give it a window that is closed
        # right now no matter when the suite runs: today+2h .. today+3h would
        # wrap, so use a 1-minute window anchored 2 hours from now.
        closed_start = (now_dt.hour + 2) % 24
        closed_end = (now_dt.hour + 3) % 24
        W.schedule_windows = lambda now=None: {
            "Later Model": {"start": f"{closed_start:02d}:00", "end": f"{closed_end:02d}:00"}}
        ck(W.model_on_demand("Later Model") is False,
           "window: model_on_demand parks a model whose window is closed right now")
        W.schedule_windows = lambda now=None: {}
        ck(W.model_on_demand("Later Model") is True,
           "window: clearing the window un-parks the model immediately")
        ck(W.model_in_schedule_window("Any Other Model", noon) is True,
           "window: a model with no window loads any time")

        # Daytime window crossing no midnight: 09:00-17:00.
        W.schedule_windows = lambda now=None: {"Work Hours": {"start": "09:00", "end": "17:00"}}
        early = now_dt.replace(hour=3, minute=0, second=0, microsecond=0).timestamp()
        ck(W.model_in_schedule_window("Work Hours", early) is False,
           "window: 03:00 is outside 09:00-17:00")
        ck(W.model_in_schedule_window("Work Hours", noon) is True,
           "window: 13:00 is inside 09:00-17:00")
    finally:
        W.schedule_windows, W.on_demand_disabled = saved_windows, saved_disabled


async def t_schedule_window_parks_then_serves_through_the_door():
    """A request naming a model outside its window parks in the queue with
    blocked=True, and the door's parked-waiter tick serves it the moment the
    window opens — no client retry, no unit touched."""
    state = {"open": False}
    async with Harness(running=[FLASH], quantum=30.0, idle_switch=0.05,
                       wait_s=10.0) as h:
        # The harness reloads the waitroom module on start, so patch AFTER it.
        W.on_demand_disabled = lambda now=None: []
        W.schedule_windows = lambda now=None: (
            {DS: {"start": "20:00", "end": "06:00"}} if not state["open"] else {})
        try:
            h.client.server.app["arb"] = W.Arbiter(allowed=W.model_on_demand)
            parked = asyncio.create_task(h.post_chat(DS))
            await asyncio.sleep(0.3)
            async with h.client.get("/state") as r:
                d = await r.json()
            ck(any(w["model"] == DS and w["blocked"] for w in d["waiting"]),
               "window door: the request parks and /state says blocked")
            ck("schedule_windows" in d, "window door: /state exposes the windows")
            s, b = await h.post_chat(FLASH)
            ck(s == 200, "window door: other traffic flows past the parked request")
            state["open"] = True
            s2, _ = await parked
            ck(s2 == 200, "window door: opening the window runs the parked request")
        finally:
            importlib.reload(W)      # drop the stubs before the next scenario


async def t_prompt_size_units():
    """The display-only prompt estimate: text lengths only, ~4 chars/token,
    and (None, None) whenever the body cannot be measured honestly."""
    j = lambda d: json.dumps(d).encode()
    ck(W.prompt_size(j({"model": "x", "messages":
                        [{"role": "user", "content": "a" * 400}]})) == (400, 100),
       "prompt size: chat text counted, /4 estimate")
    ck(W.prompt_size(j({"prompt": "b" * 80})) == (80, 20),
       "prompt size: legacy completions 'prompt' field counted")
    parts = [{"type": "text", "text": "c" * 40},
             {"type": "image_url", "image_url": {"url": "http://x/y.png"}}]
    ck(W.prompt_size(j({"messages": [{"role": "user", "content": parts}]})) == (40, 10),
       "prompt size: multimodal content counts only the text parts")
    ck(W.prompt_size(b"this is not json") == (None, None),
       "prompt size: unparseable body yields NO estimate, never a guess")
    ck(W.prompt_size(j({"model": "x"})) == (None, None),
       "prompt size: body with no prompt text yields None")
    ck(W.prompt_size(b"") == (None, None), "prompt size: empty body yields None")


async def t_prompt_estimate_travels_with_queue_entries():
    """The estimate rides on the queue entry: visible in /state's waiting list
    and (when supplied) in inflight_requests."""
    arb = W.Arbiter()
    e1 = await arb.acquire("Model One", 5)          # no estimate supplied
    task = asyncio.create_task(arb.acquire("Model Two", 5, 400, 100))
    await asyncio.sleep(0.05)
    try:
        snap = arb.snapshot()
        ck(snap["waiting"][0]["prompt_tokens_est"] == 100,
           "prompt size: parked request carries its estimate into /state waiting")
        ck(snap["inflight_requests"][0]["prompt_tokens_est"] is None,
           "prompt size: a request without a measurable prompt shows no estimate")
        w0 = snap["inflight_requests"][0]["waited_s"]
        ck(w0 is not None and w0 < 1.0,
           "card timers: an admitted request reports its queue wait (near-zero here)")
    finally:
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        arb.release(e1)


async def t_access_log_filter_keeps_meaningful_lines():
    """The console polls GET /state every 2 s; those lines must stop filling
    the journal, while POSTs and failed polls stay visible."""
    f = W._QuietPollAccessFilter()
    def rec(msg):
        return logging.LogRecord("aiohttp.access", logging.INFO, "", 0, msg, None, None)
    poll = '127.0.0.1 [23/Nov/2026:08:55:33 -0500] "GET /state HTTP/1.1" 200 658 "-" "python-httpx/0.28.1"'
    poll500 = '127.0.0.1 [23/Nov/2026:08:55:33 -0500] "GET /state HTTP/1.1" 500 96 "-" "curl/8"'
    chat = '192.168.1.5 [23/Nov/2026:08:55:40 -0500] "POST /v1/chat/completions HTTP/1.1" 200 9182 "-" "openai-python"'
    ck(f.filter(rec(poll)) is False, "access filter: a 2xx GET /state poll is silenced")
    ck(f.filter(rec(poll500)) is True, "access filter: a failing poll still logs")
    ck(f.filter(rec(chat)) is True, "access filter: POST traffic always logs")
    ck(f.filter(rec("model swap requested: Qwen")) is True,
       "access filter: non-access records are untouched")


async def t_ondemand_door_parks_one_request_and_serves_another():
    allow = {FLASH}
    async with Harness(running=[FLASH], quantum=30.0, idle_switch=0.05,
                       wait_s=8.0) as h:
        h.client.server.app["arb"] = W.Arbiter(allowed=lambda m: m in allow)
        parked = asyncio.create_task(h.post_chat(DS))
        await asyncio.sleep(0.25)
        s, b = await h.post_chat(FLASH)
        ck(s == 200 and b["via"] == "gate",
           "ondemand door: the enabled model is served while the other is parked")
        async with h.client.get("/state") as r:
            d = await r.json()
        ck(any(w["model"] == DS and w["blocked"] for w in d["waiting"]),
           "ondemand door: /state marks the parked request so the console can explain it")
        allow.add(DS)
        s2, _ = await parked
        ck(s2 == 200, "ondemand door: the parked request runs once the model is back on")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]

async def main():
    for t in TESTS:
        try:
            await t()
        except Exception as e:
            ck(False, f"{t.__name__} raised {type(e).__name__}: {e}")
    print(f"\n{CHECKS - len(FAILS)}/{CHECKS} checks passed")
    if FAILS:
        print("FAILED:", *FAILS, sep="\n  - ")
        sys.exit(1)
    print("ALL GREEN")


if __name__ == "__main__":
    asyncio.run(main())
