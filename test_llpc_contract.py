"""Contract tests between the llpc waitroom/watchdog and model_control.

The 2026-11-20 outage class: `waitroom.py` and `slot-watchdog.py` import
`model_control` and call helpers on it at module level (`read_lock`,
`lock_conflict`, ...). When those names are missing the import still succeeds,
so every `/state` poll 500s and manual-lock enforcement silently disappears.
These tests fail loudly instead.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent
LLPC = REPO / "llpc"

# Files that import model_control and call helpers on it as `mctl`.
CALLERS = ("waitroom.py", "slot-watchdog.py")
REQUIRED = {
    "read_lock",
    "read_manual",
    "lock_conflict",
    "lock_error_payload",
    "is_backstop_suppressed",
}


def _load_model_control():
    """Import llpc/model_control.py the way the installed layout does."""
    spec = importlib.util.spec_from_file_location("llpc_model_control", LLPC / "model_control.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_model_control_imports_without_collectors(monkeypatch):
    """The waitroom imports model_control inside a try/except: a raise here
    disables manual locking silently, so the import itself is part of the contract."""
    monkeypatch.setenv("GLAISS_ROOT", str(REPO))
    mod = _load_model_control()
    assert mod.state is not None


@pytest.mark.parametrize("attr", sorted(REQUIRED))
def test_model_control_exposes_helpers_used_by_waitroom(attr):
    mod = _load_model_control()
    assert hasattr(mod, attr), f"model_control module-level {attr}() missing — waitroom/watchdog call it"
    assert callable(getattr(mod, attr))


def test_every_mctl_attribute_reference_exists():
    """Static scan: every `mctl.<name>` in the callers must exist on the module."""
    mod = _load_model_control()
    seen = set()
    for name in CALLERS:
        text = (LLPC / name).read_text()
        for attr in re.findall(r"\bmctl\.([A-Za-z_][A-Za-z0-9_]*)", text):
            seen.add(attr)
    assert seen, "expected the callers to reference model_control as mctl"
    missing = sorted(a for a in seen if not hasattr(mod, a))
    assert not missing, f"callers reference missing model_control attributes: {missing}"


def test_lock_roundtrip_is_shared_with_glaiss_state(tmp_path):
    """A lock written by the CLI module is the same lock the dashboard reads."""
    from glaiss_app import model_state

    mod = _load_model_control()
    lock_file = tmp_path / "model-lock.json"
    mod.write_lock("Qwen 3.8 Flash Next Q4", unit="strata31", kind="text",
                   reason="contract test", by="test", path=lock_file)
    seen = mod.read_lock(lock_file)
    assert seen["model"] == "Qwen 3.8 Flash Next Q4"
    assert seen == model_state.read_lock(lock_file)
    assert mod.lock_conflict("Qwen 3.8 Flash Next Q4", lock=seen) is False
    assert mod.lock_conflict("Qwen 3.8 27B Q6", lock=seen) is True
