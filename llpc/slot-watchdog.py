#!/usr/bin/env python3
"""llpc slot-watchdog — recover a STARVED text slot (cheap timer, no daemon).

Why (2026-11-09 incident): the OWUI pipe's video path swaps ComfyUI onto the
single GPU slot; if that swap sequence dies mid-flight (or a manual `systemctl
stop` lands the wrong way), ComfyUI keeps holding VRAM with `ttl: 0` while the
text port (:8085) is gone. Nothing brings text back: the keep-warm timer only
fires when VRAM is EMPTY, so the box sits hours-long with every waitroom
request parked until its 1500 s cap — that's what starved Hermes' context
compression to death after the drain victim. This watchdog is the missing
backstop: when text has been DOWN, ComfyUI IDLE, and clients parked, it asks
llama-swap (not systemd) to release ComfyUI and warm the wanted text model.

Second failure class (2026-11-09, 95 min starvation): llama-swap inference
PAUSED (POST /api/models/pause) — every :8090 warm answers 503, keep-warm and
this watchdog alike, and the pause survives llama-swap config reloads. warm()
now resumes on 503 and retries, and escalates to `systemctl start` of the
wanted unit directly if the llama-swap path cannot load at all.

Never touches busy work:
  * any text unit active/activating        -> a swap is loading, hands off
  * ComfyUI queue not drained              -> a render is running, hands off
  * cooldown SLOTWD_COOLDOWN_S between actions (default 600)

Runs from slot-watchdog.timer every minute. Disable:
`sudo systemctl stop slot-watchdog.timer` or GUARD_DISABLE=1 drop-in.
State: ~/.cache/slot-watchdog/state.json. All output lands in the journal as
"SLOT-WD ...".
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

SWAP = "http://127.0.0.1:8090"
CFG = "/home/graffunder/llama-swap/config.yaml"
INTENT_FILE = "/home/graffunder/llama-swap/waitroom-intent"
DEFAULT_MODEL_FILE = "/home/graffunder/llama-swap/default-model"
STATE = os.path.expanduser("~/.cache/slot-watchdog/state.json")
DOWN_REQUIRED_S = float(os.environ.get("SLOTWD_DOWN_REQUIRED_S", "300"))
COOLDOWN_S = float(os.environ.get("SLOTWD_COOLDOWN_S", "600"))

# Every model/unit/port comes from the model registry (llama-swap/config.yaml).
# The old hardcoded lists (TEXT_UNITS without llama-dsv4, MODEL_TO_UNIT without
# DeepSeek and defaulting to strata31) made this watchdog blind to any model
# added to the box: it then "warmed" a competing model over a healthy load,
# which is the 2026-11-16 self-switching failure. A model the registry cannot
# see now stops the watchdog instead of being guessed at.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llpc_models  # noqa: E402

try:
    REG = llpc_models.check(llpc_models.load())
except llpc_models.RegistryError as e:   # pragma: no cover - config broken
    print(f"SLOT-WD registry unusable, standing down: {e}", flush=True)
    raise SystemExit(0)

mctl: Any = None
try:
    import model_control
    mctl = model_control
except Exception:
    mctl = None

if mctl is not None and mctl.is_backstop_suppressed():
    print("SLOT-WD manual model-control active, standing down", flush=True)
    raise SystemExit(0)

TEXT_UNITS = REG.text_units()
_MEDIA = REG.media_models()
COMFY = _MEDIA[0].name if _MEDIA else None


def log(msg):
    print(f"SLOT-WD {msg}", flush=True)


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True, timeout=15).stdout


def unit_state(u):
    return sh("systemctl", "show", u, "-p", "ActiveState", "--value").strip()


def api_key():
    try:
        for line in open(CFG):
            line = line.strip()
            if line.startswith("- ") and "strata-" in line:
                # config entries are QUOTED: - "strata-XXXX"
                return line[2:].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def get_json(path, key, timeout=10):
    req = urllib.request.Request(f"{SWAP}{path}", headers={"Authorization": f"Bearer {key}"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def comfy_running_models(key):
    try:
        run = get_json("/running", key)
        return [str(r.get("model") or "") for r in run.get("running", [])]
    except Exception as e:
        log(f"/running probe failed: {e}")
        return None


def comfy_state(key):
    """('idle'|'busy'|'husk'|'absent', note). idle = tracked by llama-swap and
    queue drained; husk = systemd unit active but llama-swap no longer tracks
    it (incident leftover — safe to stop directly; nobody owns it)."""
    if not COMFY:
        return "absent", "no media model in the catalog"
    if unit_state("comfyui") not in ("active", "activating"):
        return "absent", "unit not up"
    models = comfy_running_models(key)
    if models is None:
        return "busy", "probe failed"        # conservative
    if COMFY in models:
        try:
            q = get_json(f"/upstream/{urllib.request.quote(COMFY)}/prompt", key)
            if int(q.get("exec_info", {}).get("queue_remaining", 1)) == 0:
                return "idle", "tracked, queue drained"
            return "busy", "render running"
        except Exception as e:
            return "busy", f"queue probe failed ({e})"
    return "husk", "unit active but llama-swap does not track it"


def parked():
    n = 0
    try:
        for line in sh("ss", "-H", "-tn", "state", "established",
                       "sport = :8080").splitlines():
            p = line.split()
            if len(p) >= 5 and int(p[2]) + int(p[3]) == 0:
                n += 1
    except Exception:
        pass
    return n


def wanted_model(key):
    """The model the box wants when we must act: the shared one-answer chain
    (last-loaded > fresh intent > default-model file > first catalog text
    model, Spec 2026-11-18). Falling back to the local copy of the chain only
    if current-model.sh itself is unusable."""
    try:
        out = sh("/home/graffunder/llama-swap/current-model.sh").strip()
        if out:
            return out
    except Exception:
        pass
    try:
        m = open("/home/graffunder/llama-swap/last-model").read().strip()
        if m:
            return m
    except Exception:
        pass
    try:
        d = json.load(open(INTENT_FILE))
        if time.time() - d.get("at", 0) < 43200 and d.get("model"):
            return d["model"]
    except Exception:
        pass
    try:
        m = open(DEFAULT_MODEL_FILE).read().strip()
        if m:
            return m
    except Exception:
        pass
    return "Qwen 3.8 Flash Next Q4"


def resume(key):
    """Undo llama-swap's pause (POST /api/models/pause unloads everything and
    503-rejects ALL :8090 inference until resumed — the flag even survives a
    config reload, so only a resume or a llama-swap restart clears it). A
    manual video cycle that paused and died mid-flight silently disarms every
    backstop that warms through :8090 — this exact hole starved the text slot
    95 min on 2026-11-09. Resume while NOT paused is a documented no-op."""
    try:
        req = urllib.request.Request(f"{SWAP}/api/models/resume", data=b"", method="POST",
                                     headers={"Authorization": f"Bearer {key}"})
        urllib.request.urlopen(req, timeout=10).read()
        log("inference resumed (llama-swap had been rejecting everything)")
    except Exception as e:
        log(f"resume call failed: {e}")


def warm(key, model, unit=None):
    """Tiny completion through llama-swap so IT owns the load (keepalive, FIFO,
    catalog all come along). Blocks until the model answers. A 503 means the
    stale-pause state: resume and retry once. If llama-swap cannot serve the
    load at all, start the wanted unit directly as last resort — waitroom
    proxies straight to :8085 for the loaded model, and llama-swap adopts a
    running proxy later (that's what a keepalive-started unit looks like)."""
    body = json.dumps({"model": model, "max_tokens": 512,
                       "messages": [{"role": "user", "content": "ping"}]}).encode()
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(f"{SWAP}/v1/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {key}",
                                                  "Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=1800).read()
            log(f"warm ok — '{model}' serving again")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt == 1:
                log("warm got 503 — llama-swap paused? resuming and retrying")
                resume(key)
                continue
            log(f"warm failed: {e}")
            break
        except Exception as e:
            log(f"warm failed: {e}")
            break
    if unit:
        log(f"escalation: starting '{unit}' directly (llama-swap warm path failed)")
        subprocess.run(["systemctl", "reset-failed", unit], timeout=15)
        subprocess.run(["systemctl", "start", "--no-block", unit], timeout=15)
    return False


def main():
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    st = {}
    try:
        st = json.load(open(STATE))
    except Exception:
        pass
    now = time.time()

    text_up = any(unit_state(u) in ("active", "activating") for u in TEXT_UNITS)
    if text_up:
        if st.get("down_since"):
            log(f"text slot back up after {now - st['down_since']:.0f}s dark")
        st.pop("down_since", None)
        st["last_check"] = now
        json.dump(st, open(STATE + ".tmp", "w"))
        os.replace(STATE + ".tmp", STATE)
        return

    down_since = st.get("down_since") or now
    st["down_since"] = down_since
    dark = now - down_since
    st["last_check"] = now
    json.dump(st, open(STATE + ".tmp", "w"))
    os.replace(STATE + ".tmp", STATE)
    if dark < DOWN_REQUIRED_S:
        return

    key = api_key()
    if not key:
        log("no llama-swap API key readable; cannot act")
        return
    npark = parked()
    cstate, cnote = comfy_state(key)
    if cstate == "busy":
        if npark:
            log(f"text dark {dark:.0f}s, {npark} parked client(s), but ComfyUI "
                f"is RENDERING ({cnote}) — hands off; investigate manually")
        return
    last_action = st.get("last_action", 0)
    if now - last_action < COOLDOWN_S:
        return
    model = wanted_model(key)
    unit = REG.unit(model)
    if not unit:
        log(f"wanted model {model!r} is not in the catalog — refusing to guess "
            f"a unit (fix {INTENT_FILE} or {DEFAULT_MODEL_FILE})")
        return

    if cstate == "absent":
        # Nothing holds the slot: keep-warm's job (fires ~4 min). Only act
        # ourselves when it has been conspicuously silent for another 5 min.
        if dark > DOWN_REQUIRED_S + 300:
            log(f"text dark {dark:.0f}s, nothing loaded, keep-warm silent — "
                f"warming '{model}' ({unit}) directly")
            st["last_action"] = now
            json.dump(st, open(STATE + ".tmp", "w"))
            os.replace(STATE + ".tmp", STATE)
            warm(key, model, unit)
        elif npark:
            log(f"text dark {dark:.0f}s, {npark} parked, nothing loaded — "
                f"leaving to keep-warm")
        return

    # idle (tracked) or husk (unit active, untracked): either way ComfyUI is
    # parking the slot with no work — release it, then warm the wanted model.
    log(f"STARVED: text dark {dark:.0f}s with {npark} parked client(s); "
        f"ComfyUI {cnote} — releasing slot, warming '{model}' ({unit})")
    st["last_action"] = now
    st["down_since"] = now        # restart the dark clock for the next attempt
    json.dump(st, open(STATE + ".tmp", "w"))
    os.replace(STATE + ".tmp", STATE)
    if cstate == "idle":
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{SWAP}/api/models/unload/{urllib.request.quote(COMFY)}",
                data=b"", method="POST",
                headers={"Authorization": f"Bearer {key}"}), timeout=900).read()
        except Exception as e:
            log(f"unload call failed ({e}); falling back to direct unit stop")
            subprocess.run(["systemctl", "stop", "comfyui"])
    else:  # husk: llama-swap has no idea — stop the unit directly
        subprocess.run(["systemctl", "stop", "comfyui"])
    warm(key, model, unit)


if __name__ == "__main__":
    if os.environ.get("GUARD_DISABLE") == "1":
        raise SystemExit(0)
    main()
