import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from glaiss_app import collectors, monitor, model_state as state


class ConsoleLike:
    """Stand-in for glaiss_app.console: an app plus the websocket fan-out."""

    def __init__(self):
        self.app = FastAPI()
        self.hub: object | None = None


def make_monitor(tmp_path: Path, monkeypatch):
    for attr in ("DEFAULT_LOCK_FILE", "DEFAULT_MANUAL_FILE"):
        monkeypatch.setattr(state, attr, tmp_path / attr.lower())
    c = ConsoleLike()
    monitor.install(c)
    m = c.app.state.glaiss_monitor
    m.registry_models = [
        {"name": "Text A", "aliases": ["a"], "unit": "unit-a", "text": True, "port": 8085},
        {"name": "Media B", "aliases": ["b"], "unit": "unit-b", "text": False, "port": 8188},
    ]
    return c, m


def test_console_routes_are_its_own_and_unknown_paths_404(tmp_path, monkeypatch):
    """The console answers /api/* and 404s everything else — it forwards nothing."""
    c, _ = make_monitor(tmp_path, monkeypatch)
    with TestClient(c.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True
        assert client.get("/v1/chat/completions").status_code == 404
        assert client.get("/some/backend/path").status_code == 404


def test_monitor_history_and_lock_state(tmp_path, monkeypatch):
    _, m = make_monitor(tmp_path, monkeypatch)

    async def fake_waitroom():
        return {"loaded_model": "Text A", "active": None, "inflight": 0, "waiting": []}

    async def fake_running():
        return {"running": [{"model": "Text A", "state": "ready"}]}, None

    async def fake_registry():
        return m.registry_models, None

    async def fake_engine(port, key=None):
        assert port == 8085, "token speed must be read from the loaded model's engine"
        return {"ok": True, "error": None, "decode_tps": 64.2, "prefill_tps": 102.9,
                "phase": "generating", "cache_hit_rate": 0.68,
                "lifetime_requests": 4100, "source": "engine/metrics"}

    monkeypatch.setattr(collectors, "registry_models", fake_registry)
    monkeypatch.setattr(collectors, "waitroom_snapshot", fake_waitroom)
    monkeypatch.setattr(collectors, "llama_swap_running", fake_running)
    monkeypatch.setattr(collectors, "engine_metrics", fake_engine)

    async def scenario():
        snap = await m.snapshot()
        m.history.add(snap)
        return snap

    snap = asyncio.run(scenario())
    assert snap["model"]["current_loaded_model"] == "Text A"
    assert snap["models"][0]["loaded"] is True
    assert snap["system"]["decode_tps"] == 64.2
    assert snap["system"]["prefill_tps"] == 102.9
    assert snap["system"]["token_speed_source"] == "engine/metrics"
    assert snap["system"]["engine_requests"] == 4100
    assert "traffic" not in snap, "no proxy traffic block in a console-only repo"
    assert m.history.samples[-1]["loaded_model"] == "Text A"
    assert m.history.samples[-1]["decode_tps"] == 64.2


def test_control_lock_and_inflight_guards(tmp_path, monkeypatch):
    _, m = make_monitor(tmp_path, monkeypatch)

    async def fake_waitroom():
        return {"loaded_model": "Text A", "active": "Media B", "inflight": 1, "waiting": []}

    monkeypatch.setattr(collectors, "waitroom_snapshot", fake_waitroom)

    async def scenario():
        unknown = await m.control.load("missing")
        lock = state.write_lock("Text A", unit="unit-a", reason="test")
        conflict = await m.control.load("Media B")
        unload_locked = await m.control.unload("Text A")
        unload_without_force = await m.control.unload("Media B", break_lock=True)
        return unknown, lock, conflict, unload_locked, unload_without_force

    unknown, lock, conflict, unload_locked, unload_without_force = asyncio.run(scenario())
    assert unknown["error"]["code"] == "unknown_model"
    assert lock["model"] == "Text A"
    assert conflict["status"] == 409
    assert conflict["error"]["code"] == "model_locked"
    assert unload_locked["error"]["code"] == "lock_active"
    assert unload_without_force["error"]["code"] == "request_inflight"


def test_event_endpoint_feeds_journal_and_hub(tmp_path, monkeypatch):
    class FakeHub:
        def __init__(self):
            self.messages = []
        async def emit(self, msg):
            self.messages.append(msg)

    c, m = make_monitor(tmp_path, monkeypatch)
    hub = FakeHub()
    c.hub = hub
    m.journal.add(json.dumps({
        "__REALTIME_TIMESTAMP": "0",
        "UNIT": "model-waitroom",
        "PRIORITY": "6",
        "COMM": "seed",
        "MESSAGE": "seed",
    }))
    with TestClient(c.app) as client:
        r = client.post("/internal/waitroom/events", json={"type": "arbiter_handoff", "model": "Text A"})
        assert r.status_code == 200
    assert hub.messages[-1]["event"]["model"] == "Text A"
    assert any("arbiter_handoff" in x["text"] for x in m.journal.tail())


def test_journal_units_are_json_serializable():
    """Regression: a set here made /api/state and /api/logs 500 for every caller."""
    from glaiss_app.journal import JournalRing

    ring = JournalRing()
    ring.add(json.dumps({
        "__REALTIME_TIMESTAMP": "1700000000000000",
        "UNIT": "strata31",
        "PRIORITY": "6",
        "COMM": "llama-server",
        "MESSAGE": "hello",
    }))
    entry = ring.records[-1]
    assert isinstance(entry["unit"], list)
    assert json.dumps(entry)  # must not raise
    assert "llama-server" in entry["unit"]


def test_state_endpoint_serializes_with_journal_data(tmp_path, monkeypatch):
    """The combined /api/state payload must render once journal records exist."""
    from fastapi.testclient import TestClient

    c, m = make_monitor(tmp_path, monkeypatch)
    m.journal.add(json.dumps({
        "__REALTIME_TIMESTAMP": "1700000000000000",
        "UNIT": "comfyui",
        "PRIORITY": "3",
        "COMM": "python3",
        "MESSAGE": "boom",
    }))
    with TestClient(c.app) as client:
        r = client.get("/api/state")
        assert r.status_code == 200, r.text[:400]
        assert "services" in r.json()
