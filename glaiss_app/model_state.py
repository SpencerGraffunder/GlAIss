"""Shared manual model-control state for GlAIss and llpc's model-waitroom.

This module is intentionally dependency-free so it can be imported by both the
GlAIss dashboard and the llpc-side ``~/llama-swap/model_control.py`` install
target.

Production semantics:

- ``model-lock.json`` means only the named catalog model may hold the GPU slot.
- ``manual-control.json`` means operator control is active and automatic model
  backstops should stand down until the suppression deadline passes.
- ``model-ondemand.json`` lists models that are OFF the on-demand path: the
  waitroom never loads them for a chat request, so a request naming one simply
  waits while requests for enabled models are admitted ahead of it. A manual
  load (console/CLI → ``swap-to.sh``) still works. Absent file = everything is
  on-demand, which is the default state of a fresh box.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_HOME = Path("/home/graffunder")
DEFAULT_SWAP_DIR = Path(os.environ.get("LLPC_SWAP_DIR", DEFAULT_HOME / "llama-swap"))
DEFAULT_STATE_DIR = Path(os.environ.get("LLPC_STATE_DIR", DEFAULT_HOME / ".llama-swap"))

DEFAULT_LOCK_FILE = Path(
    os.environ.get("MODEL_LOCK_FILE", str(DEFAULT_STATE_DIR / "model-lock.json"))
)
DEFAULT_MANUAL_FILE = Path(
    os.environ.get("MANUAL_CONTROL_FILE", str(DEFAULT_STATE_DIR / "manual-control.json"))
)
DEFAULT_LASTSWAP_FILE = Path(
    os.environ.get("WAITROOM_LASTSWAP_FILE", str(DEFAULT_SWAP_DIR / "waitroom-last-swap"))
)
DEFAULT_INTENT_FILE = Path(
    os.environ.get("WAITROOM_INTENT_FILE", str(DEFAULT_SWAP_DIR / "waitroom-intent"))
)
DEFAULT_LASTMODEL_FILE = Path(
    os.environ.get("WAITROOM_LASTMODEL_FILE", str(DEFAULT_SWAP_DIR / "last-model"))
)
DEFAULT_SWAP_DESIRED_FILE = Path(
    os.environ.get("SWAP_DESIRED_FILE", str(DEFAULT_STATE_DIR / "swap-desired"))
)
DEFAULT_ONDEMAND_FILE = Path(
    os.environ.get("MODEL_ONDEMAND_FILE", str(DEFAULT_STATE_DIR / "model-ondemand.json"))
)

DEFAULT_MANUAL_SUPPRESS_S = float(os.environ.get("MANUAL_CONTROL_SUPPRESS_S", "900"))


def _normalize_model(value: Optional[str]) -> str:
    return (value or "").strip().casefold()


def _now() -> float:
    return time.time()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _delete_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def read_lock(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    return _read_json(path or DEFAULT_LOCK_FILE)


def read_manual(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    data = _read_json(path or DEFAULT_MANUAL_FILE)
    if not data:
        return None
    suppress_until = data.get("suppress_backstops_until")
    if suppress_until is not None:
        try:
            if float(suppress_until) <= _now():
                data["suppress_backstops_until"] = None
                data["suppress_expired"] = True
        except Exception:
            data["suppress_backstops_until"] = None
    return data


def lock_conflict(
    requested_model: Optional[str],
    locked_model: Optional[str] = None,
    lock: Optional[Dict[str, Any]] = None,
) -> bool:
    lock = lock if lock is not None else read_lock()
    if not lock:
        return False
    locked_model = locked_model or lock.get("model")
    if not requested_model or not locked_model:
        return False
    return _normalize_model(requested_model) != _normalize_model(locked_model)


def lock_error_payload(lock: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
    lock = lock if lock is not None else read_lock()
    locked_model = (lock or {}).get("model") or "a manually locked model"
    return (
        409,
        {
            "error": {
                "type": "model_locked",
                "code": "model_locked",
                "message": (
                    f"{locked_model} is manually locked as loaded. "
                    "Unlock it before loading another model."
                ),
                "locked_model": locked_model,
            }
        },
    )


def read_on_demand(path: Optional[Path] = None) -> Dict[str, Any]:
    """The on-demand opt-out list: models the waitroom must not load on its own.

    Missing file / unreadable file = nobody is opted out. Only ``disabled`` is
    authoritative; the rest is provenance for the dashboard.
    """
    data = _read_json(path or DEFAULT_ONDEMAND_FILE) or {}
    disabled = [str(x).strip() for x in (data.get("disabled") or []) if str(x).strip()]
    return {
        "version": 1,
        "disabled": disabled,
        "updated_at": data.get("updated_at"),
        "updated_by": data.get("updated_by"),
    }


def on_demand_blocked(
    model: Optional[str],
    disabled: Optional[Any] = None,
    path: Optional[Path] = None,
) -> bool:
    """True when `model` (name or alias spelling) is off the on-demand path."""
    if not model:
        return False
    if disabled is None:
        disabled = read_on_demand(path)["disabled"]
    target = _normalize_model(model)
    return any(_normalize_model(d) == target for d in (disabled or []))


def set_model_on_demand(
    model: str,
    enabled: bool,
    *,
    path: Optional[Path] = None,
    by: str = "glaiss",
) -> Dict[str, Any]:
    """Opt a model in to (or out of) automatic on-demand loading."""
    if not (model or "").strip():
        raise ValueError("model required")
    current = read_on_demand(path)["disabled"]
    disabled = [d for d in current if _normalize_model(d) != _normalize_model(model)]
    if not enabled:
        disabled.append(model.strip())
    payload = {
        "version": 1,
        "disabled": sorted(disabled, key=str.casefold),
        "updated_at": _now(),
        "updated_by": by,
    }
    _write_json_atomic(path or DEFAULT_ONDEMAND_FILE, payload)
    return payload


def is_backstop_suppressed(
    lock: Optional[Dict[str, Any]] = None,
    manual: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> bool:
    now = now if now is not None else _now()
    if lock is None:
        lock = read_lock()
    if manual is None:
        manual = read_manual()
    if lock:
        return True
    if not manual:
        return False
    if manual.get("mode") == "manual":
        suppress_until = manual.get("suppress_backstops_until")
        try:
            if suppress_until is None or float(suppress_until) > now:
                return True
        except Exception:
            return True
    return False


def control_snapshot() -> Dict[str, Any]:
    return {"lock": read_lock(), "manual_control": read_manual()}


def write_lock(
    model: str,
    *,
    unit: str = "",
    kind: str = "text",
    reason: str = "",
    path: Optional[Path] = None,
    by: str = "glaiss",
) -> Dict[str, Any]:
    payload = {
        "version": 1,
        "model": model,
        "unit": unit,
        "kind": kind,
        "locked_at": _now(),
        "locked_by": by,
        "reason": reason,
    }
    _write_json_atomic(path or DEFAULT_LOCK_FILE, payload)
    return payload


def clear_lock(path: Optional[Path] = None) -> None:
    _delete_if_exists(path or DEFAULT_LOCK_FILE)


def write_manual(
    *,
    mode: str = "manual",
    suppress_s: Optional[float] = DEFAULT_MANUAL_SUPPRESS_S,
    lock_model: Optional[str] = None,
    updated_by: str = "glaiss",
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    now = _now()
    suppress_until = now + suppress_s if suppress_s and suppress_s > 0 else None
    payload = {
        "version": 1,
        "mode": mode,
        "suppress_backstops_until": suppress_until,
        "lock_model": lock_model,
        "updated_at": now,
        "updated_by": updated_by,
    }
    _write_json_atomic(path or DEFAULT_MANUAL_FILE, payload)
    return payload


def clear_manual(path: Optional[Path] = None) -> None:
    _delete_if_exists(path or DEFAULT_MANUAL_FILE)


def resume_auto(
    *,
    lock_path: Optional[Path] = None,
    manual_path: Optional[Path] = None,
) -> Dict[str, Any]:
    clear_lock(lock_path)
    write_manual(
        mode="auto",
        suppress_s=0,
        lock_model=None,
        updated_by="glaiss:auto",
        path=manual_path,
    )
    return {"lock": None, "manual_control": read_manual(manual_path)}


def read_lastswap(path: Optional[Path] = None) -> float:
    try:
        return float((path or DEFAULT_LASTSWAP_FILE).read_text().strip())
    except Exception:
        return 0.0


def read_text(path: Path) -> Optional[str]:
    try:
        value = path.read_text().strip()
        return value or None
    except Exception:
        return None
