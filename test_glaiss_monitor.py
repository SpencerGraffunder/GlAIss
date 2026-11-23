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
    for attr in ("DEFAULT_LOCK_FILE", "DEFAULT_MANUAL_FILE", "DEFAULT_ONDEMAND_FILE"):
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


def test_on_demand_switch_parks_a_model_without_touching_units(tmp_path, monkeypatch):
    """The checkbox in the Models tab is a routing switch: it writes the opt-out
    list the door reads, and never loads, unloads or locks anything."""
    c, m = make_monitor(tmp_path, monkeypatch)

    async def fake_waitroom():
        return {"loaded_model": "Text A", "active": None, "inflight": 0, "waiting": []}

    async def fake_registry():
        return m.registry_models, None

    monkeypatch.setattr(collectors, "registry_models", fake_registry)
    monkeypatch.setattr(collectors, "waitroom_snapshot", fake_waitroom)
    snap = asyncio.run(m.snapshot())
    assert [r["on_demand"] for r in snap["models"]] == [True, True]
    assert snap["model"]["on_demand_disabled"] == []

    with TestClient(c.app) as client:
        assert client.post("/api/control/on_demand", json={}).status_code == 400
        # alias spelling resolves to the catalog name
        off = client.post("/api/control/on_demand", json={"model": "a", "enabled": False})
        assert off.status_code == 200, off.text
        assert off.json()["model"] == "Text A"
        assert off.json()["on_demand_disabled"] == ["Text A"]

    snap = asyncio.run(m.snapshot())
    assert {r["name"]: r["on_demand"] for r in snap["models"]} == {
        "Text A": False, "Media B": True}
    assert snap["model"]["on_demand_disabled"] == ["Text A"]
    assert state.read_lock() is None, "parking a model must not create a lock"

    with TestClient(c.app) as client:
        on = client.post("/api/control/on_demand",
                         json={"model": "Text A", "enabled": True})
    assert on.json()["on_demand_disabled"] == []


def test_parse_engine_metrics_ds4_schema():
    """DS4 engines report prompt_total + reused instead of prompt_read.

    The console must derive the same prefill rate (tokens actually re-read / prompt_ms)
    and the same KV-reuse percentage, or the DeepSeek tiles sit blank.
    """
    payload = {
        "engine": {"model": "deepseek-v4.1-flash"},
        "live": {"state": "reading", "phase": "reading the prompt",
                 "prompt_tokens": 32340, "tok_s": None},
        "requests": [{
            "prompt_total": 31885, "reused": 25512, "prompt_read": None,
            "prompt_ms": 23527.8, "decode_ms": 21281.3, "decode_tok_s": 18.5,
            "hit_rate": None, "output_tokens": 394,
        }],
        "totals": {"requests": 9},
        "history": {"tok_s": [16.3, 18.6]},
    }
    out = collectors.parse_engine_metrics(payload)
    assert out["ok"] is True
    assert out["decode_tps"] == 18.5, "last finished request's decode rate"
    assert abs(out["prefill_tps"] - 270.8) < 1.0, "(31885-25512)/23.5278 s"
    assert abs(out["cache_hit_rate"] - 0.800) < 0.01, "reused/prompt_total"
    assert out["prompt_total"] == 31885 and out["reused_tokens"] == 25512
    assert out["lifetime_requests"] == 9


def test_parse_engine_metrics_strata_schema_and_live_priority():
    """Strata's prompt_read path still works, and a live tok_s beats a finished request."""
    payload = {
        "live": {"phase": "generating", "tok_s": 42.0, "prompt_tokens": 100},
        "requests": [{"prompt_read": 5000, "prompt_ms": 1000.0,
                      "decode_tok_s": 33.0, "hit_rate": 0.5}],
    }
    out = collectors.parse_engine_metrics(payload)
    assert out["prefill_tps"] == 5000.0
    assert out["decode_tps"] == 42.0, "live tok_s wins over the completed request"
    assert out["cache_hit_rate"] == 0.5


def test_history_last_non_null_keeps_the_last_prefill():
    """The tile keeps the most recent engine value instead of blanking on a gap."""
    h = monitor.History()
    h.add({"system": {"prefill_tps": 199.1, "decode_tps": 16.0},
           "model": {"current_loaded_model": "DeepSeek V4.1 Flash"}})
    h.add({"system": {"prefill_tps": None, "decode_tps": None},
           "model": {"current_loaded_model": "DeepSeek V4.1 Flash"}})
    assert h.last_non_null("prefill_tps", "DeepSeek V4.1 Flash") == 199.1
    assert h.last_non_null("decode_tps", "DeepSeek V4.1 Flash") == 16.0
    assert h.last_non_null("prefill_tps", "Qwen 3.8 Flash Next Q4") is None
