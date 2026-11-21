#!/usr/bin/env python3
"""LLPC model-control CLI with the same safety rails as the GlAIss dashboard.

Also imported as a MODULE by ``~/llama-swap/waitroom.py`` and
``~/llama-swap/slot-watchdog.py``, which call ``read_lock``, ``read_manual``,
``lock_conflict``, ``lock_error_payload`` and ``is_backstop_suppressed`` at
module level. Those are re-exported from ``glaiss_app.model_state`` below; keep
them in sync with what the waitroom calls or the routing door 500s on /state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
# Repo layout puts glaiss_app one level up; the installed layout
# (~/llama-swap/model_control.py) relies on GLAISS_ROOT or the unit's PYTHONPATH.
for _p in (os.path.join(_HERE, ".."), os.environ.get("GLAISS_ROOT", "/opt/glaiss")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

# dependency-free shared state: needed by the waitroom at import time
from glaiss_app import model_state as state

read_lock = state.read_lock
read_manual = state.read_manual
lock_conflict = state.lock_conflict
lock_error_payload = state.lock_error_payload
is_backstop_suppressed = state.is_backstop_suppressed
control_snapshot = state.control_snapshot
write_lock = state.write_lock
write_manual = state.write_manual
clear_lock = state.clear_lock
clear_manual = state.clear_manual
resume_auto = state.resume_auto
read_lastswap = state.read_lastswap

try:  # the CLI needs collectors; the waitroom does not
    from glaiss_app import collectors as _collectors
except Exception:  # pragma: no cover - only when the web install is absent
    _collectors = None

C: Any = _collectors


def print_json(obj):
    print(json.dumps(obj, indent=2, sort_keys=True))


def current_loaded_model():
    snap = C.waitroom_snapshot_sync()
    loaded = snap.get("loaded_model") or snap.get("active")
    if loaded:
        return loaded
    running, _ = C.llama_swap_running_sync()
    for item in (running or {}).get("running", []):
        if item.get("state") == "ready":
            return item.get("model")
    return None


def resolve(value: str):
    registry, err = C.registry_models_sync()
    if err and not registry:
        fail({"code": "registry_unavailable", "message": err})
    model = C.resolve_registry_model(registry or [], value)
    if not model:
        fail({"code": "unknown_model", "message": f"model not found: {value}", "registry_error": err})
    return model


def fail(err):
    if isinstance(err, dict) and "error" in err and set(err) <= {"error", "status"}:
        print_json({**err, "ok": False})
    else:
        print_json({"ok": False, "error": err})
    raise SystemExit(2)


def require_ready_write():
    lock = state.read_lock()
    manual = state.read_manual()
    return lock, manual


def cmd_status(args):
    print_json({
        "waitroom": C.waitroom_snapshot_sync(),
        "llama_swap_running": C.llama_swap_running_sync()[0],
        "registry": C.registry_models_sync()[0],
        "lock": state.read_lock(),
        "manual_control": state.read_manual(),
    })


def cmd_list(args):
    registry, err = C.registry_models_sync()
    lock = state.read_lock()
    loaded = current_loaded_model()
    rows = []
    for m in registry or []:
        rows.append({
            **m,
            "loaded": m.get("name") == loaded,
            "locked": bool(lock and state._normalize_model(lock.get("model")) == state._normalize_model(m.get("name"))),
        })
    print_json({"ok": err is None, "models": rows, "error": err})


def cmd_lock(args):
    m = resolve(args.model)
    state.write_lock(m["name"], unit=m.get("unit") or "", kind="media" if m.get("text") is False else "text", reason=args.reason, by="cli")
    state.write_manual(mode="manual", lock_model=m["name"], updated_by="cli:lock")
    print_json({"ok": True, "lock": state.read_lock()})


def cmd_unlock(args):
    state.clear_lock()
    state.write_manual(mode="manual", lock_model=None, updated_by="cli:unlock")
    print_json({"ok": True})


def cmd_resume_auto(args):
    state.resume_auto()
    print_json({"ok": True})


def cmd_load(args):
    m = resolve(args.model)
    lock, _ = require_ready_write()
    if state.lock_conflict(m["name"], lock=lock):
        fail(state.lock_error_payload(lock)[1])
    if args.dry_run:
        print_json({"ok": True, "dry_run": True, "model": m, "manual": state.read_manual()})
        return
    state.write_manual(mode="manual", lock_model=m["name"] if args.lock else None, updated_by="cli:load")
    if m.get("text") is False:
        key = C.read_llama_swap_key()
        url = f"{C.LLAMA_SWAP_URL}/upstream/{urllib.parse.quote(m['name'], safe='')}/system_stats"
        req = urllib.request.Request(url, method="GET", headers={"Authorization": f"Bearer {key}"} if key else {})
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                resp.read()
        except Exception as exc:
            fail({"code": "media_load_failed", "message": str(exc), "url": url})
    else:
        cmd = [str(C.LLPC_SWAP_TO), m["name"]]
        if args.force:
            cmd.append("--force")
        rc = subprocess.call(cmd)
        if rc != 0:
            fail({"code": "swap_to_failed", "rc": rc, "command": cmd})
    if args.lock:
        state.write_lock(m["name"], unit=m.get("unit") or "", kind="media" if m.get("text") is False else "text", reason="manual load and lock", by="cli")
    print_json({"ok": True, "model": m, "lock": state.read_lock()})


def cmd_unload(args):
    m = resolve(args.model)
    lock, _ = require_ready_write()
    if lock and not args.break_lock and state.lock_conflict(m["name"], lock=lock):
        fail(state.lock_error_payload(lock)[1])
    if lock and not args.break_lock:
        fail({"code": "lock_active", "message": "target model is locked; pass --break-lock"})
    snap = C.waitroom_snapshot_sync()
    inflight = snap.get("inflight") or 0
    if inflight and not args.force:
        fail({"code": "request_inflight", "message": f"{inflight} waitroom request(s) inflight; pass --force only in an emergency", "inflight": inflight})
    if args.dry_run:
        print_json({"ok": True, "dry_run": True, "model": m, "inflight": inflight})
        return
    state.write_manual(mode="manual", lock_model=None, updated_by="cli:unload")
    if lock and args.break_lock:
        state.clear_lock()
    key = C.read_llama_swap_key()
    url = f"{C.LLAMA_SWAP_URL}/api/models/unload/{urllib.parse.quote(m['name'], safe='')}"
    req = urllib.request.Request(url, method="POST", headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception as exc:
        fail({"code": "unload_failed", "message": str(exc), "url": url})
    for _ in range(30):
        running, _ = C.llama_swap_running_sync()
        names = {r.get("model") for r in (running or {}).get("running", [])}
        if m["name"] not in names:
            break
        time.sleep(1)
    print_json({"ok": True, "model": m, "unload_response": body})


def main():
    p = argparse.ArgumentParser(prog="model-control.py", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    l = sub.add_parser("lock"); l.add_argument("model"); l.add_argument("--reason", default="manual lock"); l.set_defaults(fn=cmd_lock)
    sub.add_parser("unlock").set_defaults(fn=cmd_unlock)
    sub.add_parser("resume-auto").set_defaults(fn=cmd_resume_auto)
    ld = sub.add_parser("load"); ld.add_argument("model"); ld.add_argument("--force", action="store_true", help="Only for emergency generation tear-down before swap-to"); ld.add_argument("--lock", action="store_true"); ld.add_argument("--dry-run", action="store_true"); ld.set_defaults(fn=cmd_load)
    ul = sub.add_parser("unload"); ul.add_argument("model"); ul.add_argument("--force", action="store_true", help="Ignore waitroom inflight guard; emergency only"); ul.add_argument("--break-lock", action="store_true"); ul.add_argument("--dry-run", action="store_true"); ul.set_defaults(fn=cmd_unload)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
