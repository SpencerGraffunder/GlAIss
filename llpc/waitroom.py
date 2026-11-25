"""llpc model-waitroom v3: :8080 smart front door for the llpc text models.

SINGLE-DOOR REGIME (v3, 2026-11-09, Spec's architecture call): every chat/
completion request is forwarded to llama-swap (:8090) — none bypasses to :8085
anymore. llama-swap's own in-flight accounting then covers ALL inference, so a
swap can never eject an engine under a live stream (the old :8085 bypass made
that impossible and is why the drain gate had to exist; it now guards manual
stops only). The waitroom keeps the POLICY layer llama-swap does not have:

1. VISIBILITY — /v1/models always serves llama-swap's full catalog
   (:8090/v1/models) with per-model loaded/unloaded status, immediately.

2. LOAD-WHICHEVER-IS-CALLED — a chat request naming an UNLOADED text model
   reaches llama-swap, which evicts the current text model, loads the target
   (healthCheckTimeout covers strata's minutes of load — the request parks
   there natively), and serves it.

3. STALE-LABEL PROTECTION (intent file) — a Hermes session cannot change its
   own model label after an agent-triggered swap (swap-to.sh <model>
   <stale_label> records both). Requests carrying the recorded stale label
   RIDE the loaded model: v3 REWRITES their "model" field to the loaded model
   before forwarding to :8090 (v2 let them slip to :8085 where the engine
   ignores the name; through the gate, an un-rewritten stale name would swap
   the machine back). Any other request naming a different catalog text model
   is a new explicit choice and swaps (after hysteresis). Requests matching
   the loaded model never touch the intent file (2026-11-08 live incident: a
   matching probe cleared the intent and the next stale-labeled request
   swapped the machine mid-test).

4. SWAP HYSTERESIS — a NEW model choice parks SWAP_HOLD_S seconds after the
   previous swap began (WAITROOM_LASTSWAP_FILE survives restarts; 0 disables).
   Explicit switches (swap-to.sh reaffirms the intent) are immune; stale-
   labeled sessions ride and never trigger a hold. The hold also arms when a
   text-member change is merely OBSERVED on /running.
   CANCEL OVERRIDE (Spec 2026-11-18): a request whose client disconnects
   BEFORE the first byte of the response (wrong-model mistake stopped in
   time) gets its forward cancelled, the hysteresis hold it armed is
   DISARMED, the queue slot is handed over immediately, and llama-swap is
   asked (in the background) to UNLOAD the target — cancelling an in-flight
   load. Nothing else unloads: the last loaded model still stands.

LAST-USED STICKINESS (Spec 2026-11-18): whatever text model was loaded last
is recorded in WAITROOM_LASTMODEL_FILE (no expiry — the OBSERVED loaded
model writes it, so it is only ever written when a load truly happened).
Every "which model does the box want" chain — this module's default naming
for name-less requests, current-model.sh, the keep-warm backstop and the
slot watchdog — prefers that file over the intent TTL and the default-model
file, so an idle box never drifts back to the default model on its own.
Nothing here unloads anything on a timer; a swap happens only when a request
nAMES a different model (ttl: 0 in config.yaml).

5. IMAGE-JOB PROTECTION (Spec 2026-11-06) — while a NON-text member (Comfy
   Studio) holds the llama-swap group, text requests WAIT (WAITROOM_WAIT_S
   cap) instead of forwarding: forwarded, llama-swap would swap-comfy-out
   (its unloadTimeout only covers in-flight :8090 traffic, and Comfy renders
   asynchronously with no request held).

6. ADOPTION — when the engine answers on its own port but llama-swap's /running
   is empty (fresh boot, or a manual systemd start), requests still go through
   :8090, which starts/adopts the running unit via its keepalive cmd. Unknown
   model names are rewritten to the intent/default text model so the gate can
   route them; an empty box only loads what a NAMED request asks for (unknown
   names wait for a model to appear, exactly like v2's waiting room).

7. FAIR SWAP QUEUE (2026-11-16, the "war between picking" fix) — requests are
   admitted through one FIFO arbiter keyed on the model they resolve to:
     * requests for the model that hold the slot run immediately (a
       conversation never waits behind an idle queue),
     * a request for a DIFFERENT model waits while the current model is
       working, then takes the slot as soon as that work drains — the caller
       gets a real answer instead of a 502 forty-five minutes later,
     * the serving model keeps the slot for WAITROOM_QUANTUM_S / N requests so
       two clients on two models cannot ping-pong full model loads at each
       other, and an idle slot is handed over after WAITROOM_IDLE_SWITCH_S,
     * order is arrival order, so no model can starve.
   The queue is serial by construction: only one model is ever in flight, so
   llama-swap never has to arbitrate two demands at once and the drain gate is
   never asked to kill a live stream to satisfy a queued model.
   GET /queue shows the active model, in-flight count and the wait list.

FAIL-OPEN: if llama-swap's view is unavailable (catalog + /running both
unreachable) routing degrades to v1 behavior — wait-if-down, then proxy
straight to the model's own port (from the model registry). If a forward to
:8090 dies with a connection error, the request is retried once against that
port directly. Rollback one-liner: point WAITROOM_SWAP at a dead port.

Non-chat paths are transparent to the loaded engine's port (engine webui,
/health truth-telling, ollama-style probes); every engine binds 127.0.0.1 —
:8080 is the only LAN door.

Env: WAITROOM_PORT (8080), WAITROOM_BACKEND (http://127.0.0.1:8085, fail-open
default when the requested model is unknown), WAITROOM_WAIT_S (1500),
WAITROOM_POLL_S (2), WAITROOM_SWAP (http://127.0.0.1:8090), WAITROOM_INTENT_FILE
(/home/graffunder/llama-swap/waitroom-intent), WAITROOM_INTENT_TTL_S (43200),
WAITROOM_SWAP_HOLD_S (600), WAITROOM_LASTSWAP_FILE
(/home/graffunder/llama-swap/waitroom-last-swap), WAITROOM_DEFAULT_FILE
(/home/graffunder/llama-swap/default-model), WAITROOM_LASTMODEL_FILE
(/home/graffunder/llama-swap/last-model), WAITROOM_CONTEXT (262144),
WAITROOM_QUANTUM_S (120), WAITROOM_QUANTUM_N (8), WAITROOM_IDLE_SWITCH_S (30).

Switching procedure: run ~/llama-swap/swap-to.sh TARGET [STALE_LABEL] — it
writes the intent then sends one probe chat request (max_tokens>=512, the
reasoning-model null-content gotcha) through this waitroom and blocks until
the swap serves it. NEVER stop a text unit with systemctl while a session is
live on it, and never use llama-swap's /api/models/pause (it 503-rejects all
:8090 inference until resumed — see README §8 gotchas).
"""
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import deque
from typing import Any
from urllib.parse import quote

from aiohttp import ClientSession, ClientTimeout, web

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:                          # the registry is optional: no registry, no ports
    import llpc_models        # noqa: E402
    REGISTRY = llpc_models.load()
except Exception as _reg_err:  # pragma: no cover - only on a broken config
    REGISTRY = None
    _REG_ERR = _reg_err

BACKEND = os.environ.get("WAITROOM_BACKEND", "http://127.0.0.1:8085").rstrip("/")
SWAP = os.environ.get("WAITROOM_SWAP", "http://127.0.0.1:8090").rstrip("/")
WAIT_S = float(os.environ.get("WAITROOM_WAIT_S", "1500"))
POLL_S = float(os.environ.get("WAITROOM_POLL_S", "2"))
INTENT_FILE = os.environ.get(
    "WAITROOM_INTENT_FILE", "/home/graffunder/llama-swap/waitroom-intent")
INTENT_TTL_S = float(os.environ.get("WAITROOM_INTENT_TTL_S", "43200"))
CONTEXT_LEN = int(os.environ.get("WAITROOM_CONTEXT", "262144"))
SWAP_HOLD_S = float(os.environ.get("WAITROOM_SWAP_HOLD_S", "600"))
QUANTUM_S = float(os.environ.get("WAITROOM_QUANTUM_S", "120"))
QUANTUM_N = int(os.environ.get("WAITROOM_QUANTUM_N", "8"))
IDLE_SWITCH_S = float(os.environ.get("WAITROOM_IDLE_SWITCH_S", "30"))
LASTSWAP_FILE = os.environ.get(
    "WAITROOM_LASTSWAP_FILE", "/home/graffunder/llama-swap/waitroom-last-swap")
DEFAULT_FILE = os.environ.get(
    "WAITROOM_DEFAULT_FILE", "/home/graffunder/llama-swap/default-model")
LASTMODEL_FILE = os.environ.get(
    "WAITROOM_LASTMODEL_FILE", "/home/graffunder/llama-swap/last-model")
SWAP_DESIRED_FILE = os.environ.get(
    "SWAP_DESIRED_FILE", "/home/graffunder/.llama-swap/swap-desired")
FALLBACK_MODEL = "Qwen 3.8 Flash Next Q4"
DS_CONCURRENCY_ONE = {
    "deepseek v4.1 flash", "deepseek-v4.1-flash", "strata-dsv41",
}
STATE_TTL_S = 10.0          # cache for catalog / running lookups
MODEL_SCAN = 262144         # bytes of body scanned for the "model" field
FULL_PARSE_LIMIT = 8 * 1024 * 1024  # bodies below this get a full json parse
DISCONNECT_POLL_S = 1.0     # how often a held-forward checks the client is gone
CANCEL_UNLOAD_MAX_S = 90.0  # how long the background load-cancel insists
PARK_WAKE_S = float(os.environ.get("WAITROOM_PARK_WAKE_S", "15"))  # hysteresis-park re-decision granularity (also the cancel-detection delay while parked)

logging.basicConfig(level=logging.INFO, format="%(asctime)s waitroom %(levelname)s %(message)s")
log = logging.getLogger("waitroom")


class _QuietPollAccessFilter(logging.Filter):
    """Silence the access-log spam from the console/watchdog GET polls.

    aiohttp logs every request through the `aiohttp.access` logger, and this
    service's log format stamps those lines `waitroom INFO` — the GlAIss
    monitor alone polls GET /state every couple of seconds. Meaningful events
    are already logged explicitly (swaps, parks, cancels, timeouts), so drop
    successful read-only poll lines and keep everything else: every POST, and
    any error response, even on a poll path.
    """

    QUIET_PATHS = ("/state", "/queue", "/metrics", "/health", "/models", "/v1/models")
    _RE = re.compile(r'"GET (/[^" ]*) HTTP/[^"]*"\s+(\d{3})')

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        m = self._RE.search(msg)
        if not m:
            return True
        path, status = m.group(1), m.group(2)
        if path in self.QUIET_PATHS and status.startswith("2"):
            return False
        return True


logging.getLogger("aiohttp.access").addFilter(_QuietPollAccessFilter())

GLAISS_EVENTS_URL = os.environ.get(
    "GLAISS_EVENTS_URL",
    "http://127.0.0.1:8899/internal/waitroom/events",
).rstrip("/")


def emit_glaiss_event(event_type: str, **fields) -> None:
    """Best-effort fire-and-forget event feed for GlAIss. Never delays routing."""
    if not GLAISS_EVENTS_URL or os.environ.get("WAITROOM_DISABLE_GLAISS_EVENTS"):
        return
    payload = {"type": event_type, "ts": time.time()}
    payload.update(fields)

    async def _send() -> None:
        try:
            timeout = ClientTimeout(total=1)
            async with ClientSession(timeout=timeout) as session:
                async with session.post(GLAISS_EVENTS_URL, json=payload):
                    pass
        except Exception:
            pass

    try:
        asyncio.get_running_loop().create_task(_send())
    except RuntimeError:
        pass

mctl: Any = None
try:
    import model_control
    mctl = model_control
except Exception:
    mctl = None


def read_swap_desired():
    """Unit currently holding/declaring the llama-swap slot claim."""
    try:
        with open(SWAP_DESIRED_FILE) as f:
            return f.read().strip() or None
    except Exception:
        return None


def control_lock_payload():
    if mctl is None:
        return None
    return mctl.read_lock()


def control_manual_payload():
    if mctl is None:
        return None
    return mctl.read_manual()


# --------------------------------------------------- on-demand opt-out gate
# The console (or `model-control.py on-demand`) writes ~/.llama-swap/
# model-ondemand.json; the arbiter reads it through this 1-second cache, so a
# checkbox flip needs no IPC and a pump storm costs at most one read per second.
# If model_control is unavailable the gate FAILS OPEN — routing must never
# depend on the optional control module (same rule as the manual lock).
_ONDEMAND_CACHE = {"at": 0.0, "disabled": []}
ONDEMAND_REFRESH_S = 1.0
# A parked waiter must notice a re-enabled model without needing new traffic:
# the arbiter re-pumps on this tick while anything is parked.
ONDEMAND_POLL_S = 2.0


def on_demand_disabled(now=None) -> list:
    now = time.time() if now is None else now
    if (now - _ONDEMAND_CACHE["at"]) >= ONDEMAND_REFRESH_S:
        _ONDEMAND_CACHE["at"] = now
        if mctl is not None:
            try:
                _ONDEMAND_CACHE["disabled"] = list(
                    mctl.read_on_demand().get("disabled") or [])
            except Exception as e:
                log.debug("on-demand list unreadable: %s", e)
    return _ONDEMAND_CACHE["disabled"]


# Daily enabled windows (model-schedule.json, same 1-second cache discipline):
# a model with a window is off the on-demand path OUTSIDE that window, so e.g.
# "DeepSeek only at night" parks daytime requests without anyone touching a
# unit. The parked-waiter tick (ONDEMAND_POLL_S) re-pumps, so a parked request
# is admitted the moment its window opens. Fail-open, same as the off list.
_SCHEDULE_CACHE = {"at": 0.0, "windows": {}}


def schedule_windows(now=None) -> dict:
    now = time.time() if now is None else now
    if (now - _SCHEDULE_CACHE["at"]) >= ONDEMAND_REFRESH_S:
        _SCHEDULE_CACHE["at"] = now
        if mctl is not None:
            try:
                _SCHEDULE_CACHE["windows"] = mctl.read_schedule()
            except Exception as e:
                log.debug("model schedule unreadable: %s", e)
    return _SCHEDULE_CACHE["windows"]


def model_in_schedule_window(model, now=None) -> bool:
    if mctl is None:
        return True
    try:
        return mctl.schedule_allows(model, schedule_windows(now), now)
    except Exception:
        return True


def model_on_demand(model) -> bool:
    """False means this model may not be auto-loaded right now: either the
    operator took it off the on-demand path, or the clock is outside its daily
    enabled window."""
    disabled = on_demand_disabled()
    if mctl is not None and disabled:
        try:
            if mctl.on_demand_blocked(model, disabled):
                return False
        except Exception:
            return True
    return model_in_schedule_window(model)


def lock_rejection_if_needed(st, model_name):
    """Return an aiohttp JSON rejection when a manual lock forbids `model_name`."""
    lock = control_lock_payload()
    if not lock:
        return None
    target = st.resolve(model_name) if model_name else None
    target = target or model_name or st.loaded_text
    if mctl.lock_conflict(target, locked_model=lock.get("model"), lock=lock):
        status, payload = mctl.lock_error_payload(lock)
        emit_glaiss_event("lock_rejection", requested=target, locked=lock.get("model"), status=status)
        try:
            return web.json_response(payload, status=status)
        except Exception:
            return None
    return None


CHAT_PATHS = ("/v1/chat/completions", "/chat/completions",
              "/v1/completions", "/completions")
MODELS_PATHS = ("/v1/models", "/models")

# "model" key with its value, unescaped-quoted; the value group is the raw JSON string.
MODEL_RE = re.compile(rb'"model"\s*:\s*"((?:[^"\\]|\\.)*)"')


# ---------------------------------------------------------------- intent file

def read_intent():
    """Fresh intent dict {"model", "stale_label"?} (the last explicit switch,
    plus the stale session label it protects when one was recorded), or None."""
    try:
        with open(INTENT_FILE) as f:
            d = json.load(f)
        if time.time() - float(d.get("at", 0)) > INTENT_TTL_S:
            return None
        m = d.get("model")
        if not isinstance(m, str) or not m:
            return None
        out = {"model": m}
        if isinstance(d.get("stale_label"), str) and d["stale_label"]:
            out["stale_label"] = d["stale_label"]
        return out
    except Exception:
        return None


def write_intent(model, stale_label=None) -> None:
    try:
        payload = json.dumps(
            {"model": model, "stale_label": stale_label, "at": time.time()})
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(INTENT_FILE) or ".", prefix=".waitroom-intent-")
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        os.replace(tmp, INTENT_FILE)
    except Exception as e:
        log.warning("intent write failed: %s", e)


def clear_intent() -> None:
    try:
        os.unlink(INTENT_FILE)
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("intent clear failed: %s", e)


# ---------------------------------------------------------------- last model
# The last text model OBSERVED loaded (written only from the real /running
# view, never from a request header). No TTL: this is the box's memory of
# what it was serving, and every idle-restoration backstop must prefer it
# (Spec 2026-11-18: "keep the last used model loaded ... never switch back
# to the default automatically").

def read_lastmodel():
    try:
        with open(LASTMODEL_FILE) as f:
            m = f.read().strip()
        return m or None
    except Exception:
        return None


def write_lastmodel(model) -> None:
    if not model:
        return
    try:
        payload = model + "\n"
        if read_lastmodel() == model:
            return
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(LASTMODEL_FILE) or ".", prefix=".last-model-")
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        os.replace(tmp, LASTMODEL_FILE)
    except Exception as e:
        log.warning("last-model write failed: %s", e)


# ---------------------------------------------------------------- last swap
def read_lastswap():
    """Epoch seconds of the last observed swap start, 0.0 when unknown."""
    try:
        with open(LASTSWAP_FILE) as f:
            return float(f.read().strip())
    except Exception:
        return 0.0


def write_lastswap(ts) -> None:
    try:
        d = os.path.dirname(LASTSWAP_FILE) or "."
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".waitroom-lastswap-")
        with os.fdopen(fd, "w") as f:
            f.write(repr(float(ts)))
        os.replace(tmp, LASTSWAP_FILE)
    except Exception as e:
        log.warning("last-swap write failed: %s", e)


# ------------------------------------------------------------------ swap state

class SwapState:
    """Cached view of llama-swap's catalog and running member. Fail-open: every
    fetch failure keeps the previous view (or None), and callers treat an empty
    view as 'unknown' -> legacy behavior."""

    def __init__(self, session: ClientSession):
        self.session = session
        self._swap_key = None
        self._key_tried = False
        self.catalog = None      # {canonical_id: {"id", "aliases": set, "text": bool}}
        self.catalog_at = 0.0
        self.loaded_raw = []     # running model names straight from /running
        self.loaded_at = 0.0
        self.last_swap = read_lastswap()

    def mark_swap(self) -> None:
        """A swap started NOW (triggered here, or observed as a loaded-member
        change): arm the hysteresis hold so a NEW model choice cannot evict
        the incoming model until SWAP_HOLD_S after this swap began."""
        self.last_swap = time.time()
        write_lastswap(self.last_swap)

    def hold_remaining(self) -> float:
        """Seconds a NEW model choice must still wait before it may swap."""
        if SWAP_HOLD_S <= 0 or self.last_swap <= 0:
            return 0.0
        return max(0.0, SWAP_HOLD_S - (time.time() - self.last_swap))

    @property
    def loaded_text(self):
        """Canonical id of the loaded TEXT model, or None."""
        if not self.catalog:
            return None
        for name in self.loaded_raw:
            m = self.resolve(name)
            if m and self.catalog[m].get("text"):
                return m
        return None

    def resolve(self, name):
        """Map a request model name (id or alias) to a canonical catalog id."""
        if not name or not self.catalog:
            return None
        if name in self.catalog:
            return name
        low = name.casefold()
        for cid, e in self.catalog.items():
            if cid.casefold() == low or low in {a.casefold() for a in e["aliases"]}:
                return cid
        return None

    def invalidate(self) -> None:
        self.catalog_at = 0.0
        self.loaded_at = 0.0

    async def _swap_get(self, path, timeout_s=5.0):
        """GET on the swap gatekeeper. Returns (status, parsed_json_or_None).
        Network failure -> (None, None). Retries once with an api key on 401/403."""
        async def _do(headers):
            try:
                async with self.session.get(SWAP + path, headers=headers or None,
                                            timeout=ClientTimeout(total=timeout_s)) as r:
                    status = r.status
                    try:
                        return status, await r.json()
                    except Exception:
                        return status, None
            except Exception:
                return None, None
        headers = {}
        if self._swap_key:
            headers["Authorization"] = f"Bearer {self._swap_key}"
        status, data = await _do(headers)
        if status in (401, 403) and not self._key_tried:
            self._key_tried = True
            self._swap_key = self._load_swap_key()
            if self._swap_key:
                headers["Authorization"] = f"Bearer {self._swap_key}"
                return await _do(headers)
        return status, data

    async def get_key(self):
        """Swap API key, lazily (forward path uses it when the client sent no
        Authorization header; probe-free local clients then work unchanged)."""
        if self._swap_key is None and not self._key_tried:
            self._key_tried = True
            self._swap_key = self._load_swap_key()
        return self._swap_key

    @staticmethod
    def _load_swap_key():
        try:
            with open("/home/graffunder/llama-swap/config.yaml") as f:
                m = re.search(r'^\s*-\s*"(strata-[^"]+)"', f.read(), re.M)
                return m.group(1) if m else None
        except Exception:
            return None

    async def refresh(self, force=False) -> None:
        now = time.monotonic()
        if force or now - self.catalog_at > STATE_TTL_S:
            try:
                status, data = await self._swap_get("/v1/models")
                if status == 200 and data:
                    cat = {}
                    for m in data.get("data", []):
                        mid = m.get("id")
                        if not mid:
                            continue
                        meta = (m.get("meta") or {}).get("llamaswap") or {}
                        out = (m.get("architecture") or {}).get("output_modalities") or []
                        cat[mid] = {
                            "id": mid,
                            "aliases": set(meta.get("aliases") or []),
                            "text": "text" in out,
                        }
                    if cat:
                        self.catalog = cat
                        self.catalog_at = now
            except Exception as e:
                log.debug("catalog refresh failed: %s", e)
        if force or now - self.loaded_at > STATE_TTL_S:
            try:
                status, data = await self._swap_get("/running")
                if status == 200 and data:
                    prev = self.loaded_text      # view computed on the OLD raw
                    new_raw = [e.get("model") for e in data.get("running", [])
                               if e.get("model")]
                    if new_raw != self.loaded_raw:
                        self.loaded_raw = new_raw
                        # X -> Y among text members is a swap, whoever asked
                        # for it (WebUI pipe warm, keep-warm, direct :8090).
                        # None -> X is a cold load or a restart of this
                        # proxy, not a swap: it must not arm the hold.
                        if prev and self.loaded_text and prev != self.loaded_text:
                            self.mark_swap()
                        # LAST-USED record: whatever text model is really
                        # loaded right now is the model the box "was using".
                        # Written only from this observed view, never from a
                        # request header — a failed or cancelled load never
                        # claims it.
                        if self.loaded_text and self.loaded_text != prev:
                            write_lastmodel(self.loaded_text)
                    self.loaded_at = now
            except Exception as e:
                log.debug("running refresh failed: %s", e)


# -------------------------------------------------------------------- arbiter

class QueueTimeout(Exception):
    """A queued request ran out of patience while another model held the slot,
    or while it sat parked because the operator took its model off the
    on-demand path."""

    def __init__(self, model, waited, active, blocked=False):
        self.model, self.waited, self.active, self.blocked = model, waited, active, blocked
        if blocked:
            msg = (f"{model!r} waited {waited:.0f}s but is switched OFF for "
                   f"on-demand loading, so nothing will load it; re-enable it in "
                   f"the GlAIss Models tab (On demand) or load it manually")
        else:
            msg = (f"{model!r} waited {waited:.0f}s for the model slot; the box "
                   f"is serving {active!r}")
        super().__init__(msg)


class Arbiter:
    """One GPU slot, every catalog model: FIFO admission with model batching.

    Rules (module docstring item 7):
      * the ACTIVE model's requests are admitted immediately while its quantum
        is fresh — a conversation never waits behind an idle queue;
      * a request for a DIFFERENT model waits until the quantum has expired and
        every in-flight request of the active model has drained, then becomes
        the active model. Arrival order decides which waiter is next.
      * an idle slot (nothing in flight for IDLE_SWITCH_S) is handed over early,
        so an occasional switch does not sit behind a stale quantum;
      * only one model is ever in flight, so two clients on two models cannot
        make llama-swap thrash model loads at each other, and the drain gate is
        never asked to kill a live stream to satisfy a queued model.

    The arbiter is pure admission control: it never loads or unloads anything
    itself. The request that wins the slot forwards to llama-swap, which owns
    the actual evict/load.
    """

    def __init__(self, allowed=None):
        # allowed(model) -> bool: the on-demand opt-out gate. None = allow all,
        # which keeps the arbiter testable in isolation and fail-open by default.
        self._allowed = allowed or (lambda model: True)
        self._q = []           # waiting entries, arrival order
        self._seq = 0
        self._timer = None
        self.active = None
        self.inflight = 0
        self._active = []       # admitted entries, one per in-flight request
        self.batch_at = 0.0
        self.batch_n = 0
        self.idle_since = 0.0
        self.handoffs = 0

    # -- visibility (GET /queue) --------------------------------------
    def snapshot(self):
        now = time.monotonic()
        left = 0.0
        if self.active:
            left = max(0.0, round(QUANTUM_S - (now - self.batch_at), 1))
        return {
            "active": self.active,
            "inflight": self.inflight,
            "inflight_requests": [{"req_id": e["seq"], "model": e["model"],
                                   "since_s": round(now - e.get("admitted_at", e["at"]), 1),
                                   "prompt_tokens_est": e.get("ptok")}
                                  for e in self._active],
            "quantum_left_s": left,
            "handoffs": self.handoffs,
            "waiting": [{"model": e["model"],
                         "waited_s": round(now - e["at"], 1),
                         "prompt_tokens_est": e.get("ptok"),
                         "blocked": not self._allowed(e["model"])}
                        for e in self._q],
        }

    # -- internals ----------------------------------------------------
    def _expired(self, now):
        if not self.active:
            return True
        return (now - self.batch_at) >= QUANTUM_S or self.batch_n >= QUANTUM_N

    def _others_waiting(self):
        # Parked (off-demand) waiters are not "somebody else waiting": they are
        # waiting on the operator, not on the slot, so they must not make the
        # active model give the slot up to nobody.
        return any(e["model"] != self.active and self._allowed(e["model"])
                   for e in self._q)

    def _blocked_waiting(self):
        """Waiters the operator took off the on-demand path. They hold their
        place in line for visibility but never hold up anyone else."""
        return [e for e in self._q if not self._allowed(e["model"])]

    def _model_limit(self, model):
        # DeepSeek V4.1 Flash: one in-flight request, matching the engine's
        # `concurrency.serving: 1`. Other models keep the arbiter's batch behavior.
        if model and str(model).strip().lower() in DS_CONCURRENCY_ONE:
            return 1
        return None

    def _admit(self, entry):
        self.inflight += 1
        self.batch_n += 1
        self.idle_since = 0.0
        entry["admitted"] = True
        entry["admitted_at"] = time.monotonic()
        self._active.append(entry)
        entry["ev"].set()
        emit_glaiss_event("queue_admit", model=entry.get("model"), req_id=entry.get("seq"), inflight=self.inflight)

    def _pump(self):
        now = time.monotonic()
        for _ in range(1000):          # bounded: each pass admits or switches
            # A model the operator took out of circulation keeps serving what is
            # already in flight, then gives the slot up; nothing new is admitted
            # to it while it is off the on-demand path.
            if self.active and not self._allowed(self.active):
                if self.inflight:
                    return
                self.active = None
                continue
            if self.active is None:
                # First waiter still on the on-demand path, not strictly the
                # head of the queue: a parked model must not block traffic.
                nxt = next((e for e in self._q if self._allowed(e["model"])), None)
                if nxt is None:
                    return
                self.active = nxt["model"]
                self.batch_at = now
                self.batch_n = 0
                self.idle_since = 0.0
                self.handoffs += 1
                emit_glaiss_event("arbiter_handoff", model=self.active, handoffs=self.handoffs)
            same = [e for e in self._q if e["model"] == self.active]
            # Serve the active model while its quantum is fresh, or always when
            # nobody else is waiting (a single-model box never stalls). Blocked
            # waiters do not count as "somebody else" — they are not next.
            if same and (not self._expired(now) or not self._others_waiting()):
                limit = self._model_limit(self.active)
                slots = len(same) if limit is None else max(0, limit - self.inflight)
                if slots == 0:
                    return
                for e in same[:slots]:
                    self._q.remove(e)
                    self._admit(e)
                continue
            # Hand the slot to another model only when no stream is running.
            idle_long = (self.inflight == 0 and self.idle_since
                         and now - self.idle_since >= IDLE_SWITCH_S)
            if (self.inflight == 0 and self._others_waiting()
                    and (self._expired(now) or idle_long)):
                self.active = None
                continue
            return
        log.warning("arbiter: pump gave up after 1000 passes (%s)", self.snapshot())

    def _arm(self):
        """Re-pump when the quantum expires or an idle slot matures: a waiter
        must not depend on another request arriving to be served."""
        if not self._q:
            return
        parked = bool(self._blocked_waiting())
        if not self.active and not parked:
            return
        now = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if not self.active:
            # Nothing runnable: only parked waiters. Tick slowly so flipping a
            # model back on gets them moving without waiting for new traffic.
            delay = ONDEMAND_POLL_S
        else:
            delay = max(0.2, (self.batch_at + QUANTUM_S) - now)
            if self.inflight == 0 and self.idle_since:
                delay = min(delay, max(0.2, IDLE_SWITCH_S - (now - self.idle_since)))
            if parked:
                delay = min(delay, ONDEMAND_POLL_S)
        # An earlier timer may be armed for the longer quantum: replace it so a
        # freed slot is handed over on the idle deadline, not minutes later.
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._timer = loop.call_later(delay, self._timer_fire)

    def _timer_fire(self):
        self._timer = None
        self._pump()
        self._arm()

    # -- admission ----------------------------------------------------
    async def acquire(self, model, timeout, pchars=None, ptok=None):
        """Wait for the slot. Returns the entry; raises QueueTimeout.
        Cancellation-safe: a client that goes away mid-wait removes itself
        (or gives the slot straight back if it was admitted in the moment
        of the cancel) so the FIFO never carries a ghost.
        pchars/ptok: display-only prompt-size estimate for the queue boxes."""
        entry = {"model": model, "ev": asyncio.Event(), "at": time.monotonic(),
                 "seq": self._seq, "admitted": False,
                 "pchars": pchars, "ptok": ptok}
        self._seq += 1
        self._q.append(entry)
        emit_glaiss_event("queue_arrive", model=model, req_id=entry["seq"])
        self._pump()
        if not entry["admitted"]:
            self._arm()
            try:
                await asyncio.wait_for(entry["ev"].wait(),
                                       timeout=max(0.05, timeout))
            except asyncio.TimeoutError:
                if entry["admitted"]:
                    self.release(entry)
                else:
                    try:
                        self._q.remove(entry)
                    except ValueError:
                        pass
                    self._pump()
                emit_glaiss_event("queue_timeout", model=model, req_id=entry["seq"])
                raise QueueTimeout(model, time.monotonic() - entry["at"],
                                   self.active,
                                   blocked=not self._allowed(model))
            except asyncio.CancelledError:
                if entry["admitted"]:
                    self.release(entry)
                else:
                    try:
                        self._q.remove(entry)
                    except ValueError:
                        pass
                    self._pump()
                emit_glaiss_event("request_cancel", model=model,
                                  req_id=entry["seq"], phase="queue_wait")
                raise
        return entry

    def force_idle(self):
        """Cancel override (Spec 2026-11-18): the request that held the slot
        was cancelled before it ever served — hand the slot over NOW instead
        of waiting out the quantum or the idle-switch delay, so the caller's
        corrected model choice is admitted on arrival."""
        if self.inflight == 0:
            self.active = None
            self._pump()
            self._arm()

    def release(self, entry=None):
        """Give the slot back. `entry` identifies the finished request."""
        self.inflight = max(0, self.inflight - 1)
        if entry is not None:
            try:
                self._active.remove(entry)
            except ValueError:
                pass
        elif self._active:
            self._active.pop()
        if self.inflight == 0:
            self.idle_since = time.monotonic()
        self._pump()
        self._arm()


# -------------------------------------------------------------------- helpers

async def backend_up(session: ClientSession) -> bool:
    try:
        async with session.get(f"{BACKEND}/health", timeout=ClientTimeout(total=5)) as r:
            return r.status == 200
    except Exception:
        return False


async def wait_for_backend(session: ClientSession, req: web.Request) -> bool:
    """Block until the backend answers /health, up to WAIT_S. Logs once per 30 s.
    Returns True on recovery, False on timeout, None when the CLIENT went away
    (the caller must not serve or swap for a caller that left — Spec's cancel
    rule 2026-11-18: a parked request the user stopped must never trigger a
    model change once they are gone)."""
    deadline = time.monotonic() + WAIT_S
    last_log = 0.0
    log.info("backend down; holding %s %s (cap %.0fs)", req.method, req.path, WAIT_S)
    while time.monotonic() < deadline:
        if req.transport is None:
            log.info("client went away while parked; abandoning %s %s",
                     req.method, req.path)
            return None
        if await backend_up(session):
            waited = WAIT_S - (deadline - time.monotonic())
            log.info("backend back after %.0fs; serving %s %s", waited, req.method, req.path)
            return True
        now = time.monotonic()
        if now - last_log > 30:
            log.info("still waiting for backend (%.0fs elapsed) for %s %s",
                     now - (deadline - WAIT_S), req.method, req.path)
            last_log = now
        await asyncio.sleep(POLL_S)
    log.warning("backend did not return within %.0fs; 503 for %s %s", WAIT_S, req.method, req.path)
    return False


def extract_model(body: bytes):
    """The request's model field (raw, possibly an alias), or None. Never raises."""
    if not body:
        return None
    m = MODEL_RE.search(body[:MODEL_SCAN])
    if m:
        try:
            return json.loads(b'"' + m.group(1) + b'"')
        except Exception:
            return m.group(1).decode("utf-8", "replace")
    if len(body) < FULL_PARSE_LIMIT:
        try:
            d = json.loads(body)
            if isinstance(d, dict) and isinstance(d.get("model"), str):
                return d["model"]
        except Exception:
            pass
    return None


def prompt_size(body: bytes):
    """(chars, est_tokens) for the prompt text of a chat/completions body.

    Display-only: the queue boxes show a rough prompt size while a request
    still sits in the queue (the engine reports exact numbers once it starts,
    and the cached share is unknowable before then). ~4 chars/token is the
    usual heuristic for mixed prose+code; we deliberately show "~" for it.
    Returns (None, None) for anything we cannot confidently measure — never
    raises, never puts a fake number on the page.
    """
    if not body:
        return None, None
    try:
        d = json.loads(body) if len(body) <= FULL_PARSE_LIMIT else None
    except Exception:
        d = None
    parts = []
    if isinstance(d, dict):
        if isinstance(d.get("prompt"), str):       # legacy completions shape
            parts.append(d["prompt"])
        msgs = d.get("messages")
        if isinstance(msgs, list):
            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                c = msg.get("content")
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, list):          # multimodal: text parts only
                    for p in c:
                        if (isinstance(p, dict) and p.get("type") in ("text", "input_text")
                                and isinstance(p.get("text"), str)):
                            parts.append(p["text"])
    if not parts:
        return None, None
    chars = sum(len(p) for p in parts)
    if not chars:
        return None, None
    return chars, max(1, round(chars / 4))


def set_model(body: bytes, model: str) -> bytes:
    """Rewrite the model field's value (used when the waitroom routes the
    request onto the loaded/intent model). When the body has NO model field
    (v2 let those slip through to the engine which ignores the name; the gate
    needs one), inject it after the opening brace. Falls back to the
    unmodified body on any mismatch. NOTE: json.dumps adds its own quotes, but
    the regex already captured the value BETWEEN the quotes — strip them, or
    the result is double-quoted and the upstream 400s the whole request."""
    m = MODEL_RE.search(body[:MODEL_SCAN])
    try:
        if not m:
            stripped = body.lstrip()
            if stripped.startswith(b"{") and len(body) < FULL_PARSE_LIMIT:
                val = json.dumps({"model": model})[1:-1]  # '"model": "..."'
                pos = body.index(b"{") + 1
                return body[:pos] + val.encode() + (b"," if stripped != b"{}" else b"") + body[pos:]
            return body
        val = json.dumps(model)[1:-1].encode()
        return body[:m.span(1)[0]] + val + body[m.span(1)[1]:]
    except Exception:
        return body


# DeepSeek-V4.1 effort guard (2026-11-17): the V4.1 chat template raises on any
# reasoning_effort word outside low/high/max (docs' "medium" only works as the
# number 62; "xhigh" is Qwen vocabulary and 400s the whole request). Clients
# show per-model ladders, but the provider-wide thinking setting in Hermes used
# to fire the same word at every llpc model — the gate is the last shared
# choke point, so the rewrite belongs here. Keyed on the RESOLVED target model,
# never on the word alone: a Qwen request keeps its xhigh untouched.
DS_EFFORT_TARGETS = {"deepseek v4.1 flash", "deepseek-v4.1-flash", "strata-dsv41"}
DS_EFFORT_REWRITES = (
    (re.compile(rb'("reasoning_effort"\s*:\s*)"xhigh"'), rb'\1"high"'),
    (re.compile(rb'("reasoning_effort"\s*:\s*)"medium"'), rb'\1"62"'),
)


def guard_ds_effort(body: bytes, target) -> bytes:
    """Neutralize Qwen-only effort words in requests routed to DeepSeek-V4.1."""
    if not body or not target or str(target).casefold() not in DS_EFFORT_TARGETS:
        return body
    for pat, rep in DS_EFFORT_REWRITES:
        body = pat.sub(rep, body)
    return body


def default_text_name(st: "SwapState") -> str:
    """Whom to name when the caller didn't pick a model: the last model
    OBSERVED loaded (sticky, Spec 2026-11-18), then fresh intent, then the
    default-model file, then Flash Next."""
    last = read_lastmodel()
    if last:
        return st.resolve(last) or last
    intent = read_intent()
    if intent:
        return st.resolve(intent["model"]) or intent["model"]
    try:
        with open(DEFAULT_FILE) as f:
            d = f.read().strip()
        if d:
            return d
    except Exception:
        pass
    return FALLBACK_MODEL


def backend_for(name) -> str:
    """Fail-open target for a request: the model's OWN port from the registry
    (2026-11-16 port split: every engine has its own), else WAITROOM_BACKEND."""
    port = None
    if REGISTRY is not None and name:
        try:
            port = REGISTRY.port(name)
        except Exception:
            port = None
    return f"http://127.0.0.1:{port}" if port else BACKEND


async def forward_to(session: ClientSession, req: web.Request, base: str,
                     body: bytes, auth_fallback=None, progress=None) -> web.StreamResponse:
    """Stream the backend's answer back (SSE-safe). `progress`, when given,
    gets progress['started'] = True the instant the upstream answers with
    status+headers — the boundary the cancel path keys on (client gone BEFORE
    this point => cancel the load; after this point => tokens were already
    flowing, keep the model loaded per the last-used rule)."""
    headers = {
        k: v for k, v in req.headers.items()
        if k.lower() not in ("host", "connection", "content-length", "transfer-encoding")
    }
    if auth_fallback is not None and not any(
            k.lower() == "authorization" for k in headers):
        key = await auth_fallback.get_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
    url = f"{base}{req.path}"
    if req.query_string:
        url += "?" + req.query_string
    timeout = ClientTimeout(total=3600, connect=10, sock_read=3600)
    async with session.request(
        req.method, url, headers=headers, data=body if body else None,
        timeout=timeout, allow_redirects=False,
    ) as up:
        if progress is not None:
            progress["started"] = True
        resp_headers = {
            k: v for k, v in up.headers.items()
            if k.lower() not in ("connection", "content-length", "transfer-encoding",
                                 "keep-alive", "date", "server")
        }
        resp = web.StreamResponse(status=up.status, headers=resp_headers)
        await resp.prepare(req)
        try:
            async for chunk in up.content.iter_any():
                if chunk:
                    await resp.write(chunk)
            await resp.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp


# ---------------------------------------------------------------------- routing

def comfy_claimed() -> bool:
    """True while the media cycle owns (or is loading onto) the slot.
    unit-keepalive.sh writes the target unit into ~/.llama-swap/swap-desired
    BEFORE ejecting anything, so this file covers the claim -> load ->
    job-submit window that /running misses (2026-11-09 D2: text requests
    arriving there woke strata twice within a second of ComfyUI's health
    pass / got-prompt and killed the video job). Any later load's declare —
    the post-job rewarm, a manual swap, the watchdog warm — clears it."""
    try:
        with open(SWAP_DESIRED_FILE) as f:
            return f.read().strip() == "comfyui"
    except Exception:
        return False


async def decide(st: SwapState, body: bytes, backend_up_now=False, force=False):
    """Policy for a chat request. Returns (route, body):

      ("fwd", b)    forward to llama-swap (:8090) with b — load-or-ride; the
                    request counts in llama-swap's in-flight table, so no swap
                    can eject the engine under it.
      ("hold", b)   hysteresis park (NEW model choice too soon after a swap).
      ("wait", b)   nothing serving text for this caller yet (image engine
                    holds the slot / empty box + unnamed model): wait, re-decide.
      ("legacy", b) llama-swap's view unavailable -> fail-open direct :8085.
    """
    name = extract_model(body)
    await st.refresh(force)
    if not st.catalog:
        return "legacy", body              # gatekeeper unreachable: v1 behavior
    cat = st.catalog
    if not name and not body.strip():
        return "legacy", body              # empty POST (probes): nothing to route
    m = st.resolve(name) if name else None
    if m and not cat[m].get("text"):
        m = None                           # media member named on a chat call
    loaded = st.loaded_text
    if loaded is None and st.loaded_raw:
        emit_glaiss_event("slot_claim_wait", reason="comfy_running", requested=m)
        return "wait", body                # Comfy holds the group: never evict it
    if loaded is None and comfy_claimed():
        # Comfy holds the CLAIM (unit loading, or job not yet submitted):
        # summoning strata in this window stops ComfyUI mid-load/mid-submit —
        # /running misses it because nothing is READY yet. Wait for the job to
        # start (then loaded_raw protects) or for the claim to die (watchdog
        # releases and its warm re-declares the slot for text).
        emit_glaiss_event("slot_claim_wait", reason="comfy_claimed", requested=m)
        return "wait", body
    if m and loaded and m == loaded:
        # Rides the loaded model; NEVER touches the intent file (2026-11-08
        # incident: a matching probe cleared the intent mid-test).
        return "fwd", body
    if loaded:
        intent = read_intent()
        im = st.resolve(intent["model"]) if intent else None
        if intent and im and cat.get(im, {}).get("text"):
            if m == im:
                # Reaffirmation of the last explicit switch (swap-to.sh probe):
                # swap to it now, keeping any recorded stale label.
                write_intent(im, stale_label=intent.get("stale_label"))
                return "fwd", set_model(body, im)
            stale = intent.get("stale_label")
            if stale and st.resolve(stale) == m:
                # The stale session label: ride the LOADED model — v3 rewrites
                # the name (through the gate an un-rewritten stale name would
                # swap the machine back).
                return "fwd", set_model(body, loaded)
        if not m:
            # Unknown/missing model name: the engine ignored it pre-v3; the
            # gate needs a routable name -> rewrite to the loaded model.
            return "fwd", set_model(body, loaded)
        # A different catalog text model = new explicit choice: it wins, but
        # only after the hysteresis hold expired (the intent is NOT written
        # now — a parked choice must not make itself immune).
        if st.hold_remaining() > 0:
            emit_glaiss_event("hysteresis_hold", requested=m, loaded=loaded, hold_remaining_s=round(st.hold_remaining(), 1))
            return "hold", body
        write_intent(m)
        return "fwd", body
    # here: no text model tracked by the gatekeeper
    if backend_up_now:
        # Engine serves on :8085 but :8090 does not track it yet (fresh boot /
        # manual systemd start): forward so llama-swap adopts the live unit
        # (keepalive sees the unit active and blocks; checkEndpoint adopts).
        if m:
            return "fwd", body
        return "fwd", set_model(body, default_text_name(st))
    # empty box
    if not m:
        return "wait", body                # unnamed requests never summon a load
    intent = read_intent()
    im = st.resolve(intent["model"]) if intent else None
    if intent and im == m:
        write_intent(im, stale_label=intent.get("stale_label"))
        return "fwd", body                 # explicit switch: load it straight away
    stale = intent.get("stale_label") if intent else None
    if stale and st.resolve(stale) == m:
        return "wait", body                # stale session waits for the slot to fill
    if st.hold_remaining() > 0:
        emit_glaiss_event("hysteresis_hold", requested=m, loaded=loaded, hold_remaining_s=round(st.hold_remaining(), 1))
        return "hold", body
    write_intent(m)
    return "fwd", body


async def serve_catalog(session: ClientSession, st: SwapState, req: web.Request):
    """/v1/models — the FULL text-model catalog (llama-swap's view), immediately.
    Never waits on the backend: a list call must not sit in the waiting room."""
    try:
        status, data = await st._swap_get("/v1/models")
        if status == 200 and data:
            models = data.get("data", [])
            if models:
                for m in models:
                    if not m.get("context_length"):
                        m["context_length"] = CONTEXT_LEN
                return web.json_response({"object": "list", "data": models})
    except Exception as e:
        log.warning("catalog fetch failed: %s", e)
    # Fail-open: the raw upstream's own list (only the loaded model) — or empty.
    try:
        async with session.get(f"{BACKEND}/v1/models",
                               timeout=ClientTimeout(total=5)) as r:
            if r.status == 200:
                return web.json_response(await r.json())
    except Exception:
        pass
    return web.json_response({"object": "list", "data": []})


async def handle(req: web.Request) -> web.StreamResponse:
    session: ClientSession = req.app["session"]
    st: SwapState = req.app["swap"]
    path = req.path.rstrip("/")
    if path in ("/health", "/v1/health"):
        # Truth-telling probe path: no waiting room for liveness checks.
        try:
            return await forward_to(session, req, BACKEND, b"")
        except Exception as e:
            log.warning("health probe failed: %s", e)
            return web.json_response({"status": "unavailable"}, status=503)
    if req.method == "GET" and path in MODELS_PATHS:
        return await serve_catalog(session, st, req)
    if req.method == "GET" and path in ("/queue", "/v1/queue"):
        arb = req.app.get("arb")
        try:
            await st.refresh()
        except Exception:
            pass
        body = {"loaded": st.loaded_text}
        if arb is not None:
            body.update(arb.snapshot())
        return web.json_response(body)

    if req.method == "GET" and path in ("/state", "/v1/state", "/waitroom/state"):
        arb = req.app.get("arb")
        try:
            await st.refresh()
        except Exception:
            pass
        desired = read_swap_desired()
        body = {
            "time": time.time(),
            "loaded_model": st.loaded_text,
            "loaded_raw": list(st.loaded_raw or []),
            "intent": read_intent(),
            "last_model": read_lastmodel(),
            "desired_unit": desired,
            "comfy_claimed": desired == "comfyui",
            "swap_hold_s": SWAP_HOLD_S,
            "hold_remaining_s": round(st.hold_remaining(), 1),
            "last_swap_unix": st.last_swap if st.last_swap and st.last_swap > 0 else None,
            "lock": control_lock_payload(),
            "manual_control": control_manual_payload(),
            "on_demand_disabled": on_demand_disabled(),
            "schedule_windows": schedule_windows(),
            "model_control_module": mctl is not None,
        }
        if arb is not None:
            body.update(arb.snapshot())
        return web.json_response(body)

    body = await req.read()
    route = "legacy"
    fwd_body = body
    if req.method in ("POST", "PUT") and path in CHAT_PATHS:
        if not body.strip():
            # empty POST has nothing to route on; speak the OpenAI dialect
            # instead of forwarding garbage (or 500-ing on the way)
            return web.json_response(
                {"error": {"message": "waitroom: request body must be a JSON "
                                       "object with a model field",
                           "type": "invalid_request_error"}},
                status=400)
        try:
            up = await backend_up(session)
            route, fwd_body = await decide(st, body, backend_up_now=up)
            if route == "legacy" and not up:
                route = "wait"             # fail-open path still waits if down
        except Exception as e:
            log.warning("route decision failed (%s); legacy proxy", e)
            route, fwd_body = "legacy", body

        # MANUAL MODEL LOCK (GlAIss): enforce after routing/rewrite decisions,
        # before hysteresis or the GPU-slot arbiter can start another model.
        if mctl is not None:
            rejection = lock_rejection_if_needed(st, extract_model(fwd_body or body))
            if rejection is not None:
                log.info("manual lock rejected request model=%s", extract_model(fwd_body or body))
                return rejection

    if route == "hold":
        # Hysteresis: park the caller until the hold on the loaded model
        # expires, re-deciding each wake-up (the machine may also have moved
        # meanwhile, and a client cancel simply ends the park — and must NOT
        # swap the box after the caller left, Spec 2026-11-18).
        while route == "hold":
            rem = st.hold_remaining()
            log.info("swap held by hysteresis (%s, %.0fs left)",
                     extract_model(body), rem)
            await asyncio.sleep(min(rem, PARK_WAKE_S) + 0.1)
            if req.transport is None:
                log.info("parked request abandoned by client (%s); no swap performed",
                         extract_model(body))
                emit_glaiss_event("request_cancel", model=extract_model(body), phase="hysteresis_park")
                return web.Response(status=499)
            try:
                up = await backend_up(session)
                route, fwd_body = await decide(st, body, backend_up_now=up, force=True)
            except Exception as e:
                log.warning("re-decision after hold failed (%s); legacy proxy", e)
                route, fwd_body = "legacy", body

    if route == "wait":
        ok = await wait_for_backend(session, req)
        if ok is None:
            emit_glaiss_event("request_cancel", model=extract_model(body), phase="parked_wait")
            return web.Response(status=499)        # client left while parked
        if not ok:
            return web.json_response(
                {"error": {"message": "text model is down and did not return within "
                                       f"{WAIT_S:.0f}s (image job in progress?)",
                           "type": "waitroom_timeout"}},
                status=503)
        try:
            route, fwd_body = await decide(st, body, backend_up_now=True, force=True)
            if route == "wait":            # backend back but route still parked:
                route, fwd_body = "legacy", body   # serve via legacy (v2 parity)
        except Exception as e:
            log.warning("re-route decision failed (%s); legacy proxy", e)
            route, fwd_body = "legacy", body

    try:
        if route == "fwd":
            if not fwd_body.strip():
                # no JSON object to route on (empty POST): speak the OpenAI dialect
                return web.json_response(
                    {"error": {"message": "waitroom: request body must be a JSON "
                                          "object with a model field",
                               "type": "invalid_request_error"}},
                    status=400)
            nm = st.resolve(extract_model(fwd_body))
            target = nm or extract_model(fwd_body) or st.loaded_text
            fwd_body = guard_ds_effort(fwd_body, target)
            # FAIR SWAP QUEUE: one model in flight at a time, arrival order,
            # batching while the quantum lasts. Queued callers get a real answer
            # or a clear queue_timeout — never llama-swap's 502 after a wait.
            arb: Arbiter = req.app.get("arb")
            progress = {"started": False}      # first byte from the upstream
            armed = {"at": None}               # hold this request armed (cancel resets it)
            last_before = read_lastmodel()     # last-used restore point for a cancel
            tcanon = st.resolve(target) or target

            async def _slot_and_forward():
                entry = None
                try:
                    if arb is not None and target:
                        try:
                            entry = await arb.acquire(target, WAIT_S,
                                                      *prompt_size(fwd_body or body))
                        except QueueTimeout as e:
                            log.info("queue timeout: %s", e)
                            return web.json_response(
                                {"error": {"message": f"waitroom: {e}",
                                           "type": "queue_timeout"},
                                 "queue": arb.snapshot()}, status=503)
                    if nm and st.loaded_text and nm != st.loaded_text:
                        log.info("model swap requested: %s", extract_model(fwd_body))
                        st.mark_swap()   # arm the hysteresis hold from the swap trigger
                        armed["at"] = st.last_swap
                    elif nm != st.loaded_text:
                        log.info("load/adopt requested: %s", extract_model(fwd_body))
                        st.mark_swap()
                        armed["at"] = st.last_swap
                    try:
                        resp = await forward_to(session, req, SWAP, fwd_body,
                                                auth_fallback=st, progress=progress)
                    except Exception as e:
                        # Gatekeeper died mid-request: fail-open once to the engine.
                        port_url = backend_for(target)
                        log.warning("forward to gatekeeper failed (%s); fail-open %s",
                                    e, port_url)
                        if not await backend_up(session):
                            raise
                        resp = await forward_to(session, req, port_url, fwd_body)
                    st.invalidate()
                    return resp
                finally:
                    if entry is not None:
                        arb.release(entry)

            # CANCEL OVERRIDE (Spec 2026-11-18): run the forward as a task and
            # watch the client. If it disconnects BEFORE the first byte of the
            # response (wrong-model mistake stopped in time), tear the forward
            # down instead of letting a model load finish for a caller who is
            # gone — and override the hysteresis this request armed so the
            # corrected choice is served on arrival. The disconnect arrives
            # two ways depending on the aiohttp runner (production run_app
            # polls: transport -> None; handler_cancellation=True raises a
            # CancelledError into the handler), so BOTH paths funnel into the
            # same background cleanup.
            fwd_task = asyncio.ensure_future(_slot_and_forward())

            async def _finish_cancel():
                """Reap the torn-down forward, then apply the override:
                disarm our hysteresis hold, hand the queue slot over, and ask
                llama-swap to cancel the target's load. A coroutine (not
                inline) because the arbiter release lands asynchronously and
                a cancelled handler task cannot await file writes."""
                try:
                    await fwd_task
                except (asyncio.CancelledError, Exception):
                    pass
                if armed["at"] is not None and st.last_swap >= armed["at"] and \
                        (st.last_swap == armed["at"] or st.loaded_text == tcanon):
                    st.last_swap = 0.0
                    write_lastswap(0.0)
                    log.info("cancel override: hysteresis disarmed")
                if arb is not None:
                    arb.force_idle()     # corrected choice must not queue
                if armed["at"] is not None:
                    log.info("client cancelled before first byte: '%s' load "
                             "cancelling, hysteresis overridden", target)
                    emit_glaiss_event("request_cancel", model=target,
                                      phase="before_first_byte")
                    await _cancel_load(session, st, arb, target,
                                       time.time(), last_before)
                else:
                    log.info("client cancelled before first byte: request "
                             "dropped (no swap was triggered by it)")
                    emit_glaiss_event("request_cancel", model=target,
                                      phase="before_first_byte")

            abandoned = False
            try:
                while True:
                    done, _ = await asyncio.wait({fwd_task},
                                                 timeout=DISCONNECT_POLL_S)
                    if done:
                        break
                    if req.transport is None and not progress["started"]:
                        abandoned = True
                        break
            except asyncio.CancelledError:
                # Handler cancelled (client-disconnect cancellation, or app
                # shutdown): same call as if we had polled it ourselves.
                fwd_task.cancel()
                if not progress["started"]:
                    asyncio.ensure_future(_finish_cancel())
                raise
            if abandoned:
                fwd_task.cancel()
                asyncio.ensure_future(_finish_cancel())
                return web.Response(status=499)
            return await fwd_task
        ok = await backend_up(session)
        if not ok:
            ok = await wait_for_backend(session, req)
            if ok is None:
                emit_glaiss_event("request_cancel", model=extract_model(body), phase="backend_wait")
                return web.Response(status=499)    # client left while parked
            if not ok:
                return web.json_response(
                    {"error": {"message": "text model is down and did not return within "
                                           f"{WAIT_S:.0f}s (image job in progress?)",
                               "type": "waitroom_timeout"}},
                    status=503)
        return await forward_to(
            session, req,
            backend_for(extract_model(body) or st.loaded_text), body)
    except Exception as e:
        log.warning("forward failed for %s %s: %s", req.method, req.path, e)
        return web.json_response({"error": {"message": f"waitroom: {e}", "type": "waitroom_error"}},
                                 status=502)


async def _cancel_load(session: ClientSession, st: SwapState, arb, target,
                       cancel_at: float, last_before) -> None:
    """Background half of the cancel override: the forward for `target` died
    with a client who never saw a token, so ask llama-swap to UNLOAD it —
    that kills a load still in flight, drops any queued waiter for it, and
    releases the slot even if the engine finished loading but served nothing
    (a model that never answered a token was never USED). llama-swap's unload
    blocks until the process stops, so one confirmed 200 means done; the loop
    re-checks /running because an unload landing between 'request dropped'
    and 'process starting' must not let the load sneak in.

    Stands down when the box moved on (newer swap to ANOTHER model) or when
    the SAME model is demanded again (re-pick after cancelling the message,
    not the model): fighting a live request is never the point."""
    url = f"{SWAP}/api/models/unload/{quote(str(target), safe='')}"
    key = await st.get_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    tcanon = st.resolve(target) or target
    deadline = time.monotonic() + CANCEL_UNLOAD_MAX_S
    while time.monotonic() < deadline:
        if arb is not None and arb.inflight > 0 and arb.active and \
                (st.resolve(arb.active) or arb.active) == tcanon:
            log.info("cancel-load: '%s' was demanded again mid-cancel; standing down",
                     target)
            return
        if st.last_swap > cancel_at and st.loaded_text and st.loaded_text != tcanon:
            log.info("cancel-load: box moved to '%s'; standing down ('%s' "
                     "needs no cancel)", st.loaded_text, target)
            return
        ok = False
        try:
            async with session.post(url, headers=headers, data=b"",
                                    timeout=ClientTimeout(total=20)) as r:
                await r.read()
                ok = r.status in (200, 404)   # 404: not in the catalog, nothing to stop
        except Exception:
            ok = False                        # llama-swap busy/stopping: retry
        await st.refresh(force=True)
        if ok and not any((st.resolve(n) or n) == tcanon for n in st.loaded_raw):
            # A cancelled load that DID complete raced its observation into
            # the last-used file; it never served a token, so restore the
            # pre-request record — the box's memory stays honest.
            if read_lastmodel() == tcanon:
                if last_before:
                    write_lastmodel(last_before)
                else:
                    try:
                        os.unlink(LASTMODEL_FILE)
                    except OSError:
                        pass
            log.info("cancel-load: '%s' released the slot (load cancelled)", target)
            return
        await asyncio.sleep(3.0)
    log.warning("cancel-load: could not confirm unload of '%s' within %.0fs",
                target, CANCEL_UNLOAD_MAX_S)


async def on_startup(app: web.Application) -> None:
    app["session"] = ClientSession()
    app["swap"] = SwapState(app["session"])
    app["arb"] = Arbiter(allowed=model_on_demand)
    log.info("waiting room open on :%s -> gatekeeper %s (fail-open per model "
             "port, cap %.0fs, quantum %.0fs/%d)",
             os.environ.get("WAITROOM_PORT", "8080"), SWAP, WAIT_S,
             QUANTUM_S, QUANTUM_N)


async def on_cleanup(app: web.Application) -> None:
    await app["session"].close()


def main() -> None:
    app = web.Application(client_max_size=200 * 1024 * 1024)  # aiohttp defaults to 1 MB; base64 vision payloads exceed it (413 'Content Too Large')
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)
    web.run_app(app, host="0.0.0.0", port=int(os.environ.get("WAITROOM_PORT", "8080")))


if __name__ == "__main__":
    main()
