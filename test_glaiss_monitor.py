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
    for attr in ("DEFAULT_LOCK_FILE", "DEFAULT_MANUAL_FILE", "DEFAULT_ONDEMAND_FILE",
                 "DEFAULT_SCHEDULE_FILE", "DEFAULT_TOKEN_STATS_FILE"):
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


def test_reset_stats_route_forgets_a_models_averages(tmp_path, monkeypatch):
    """The Models-tab reset button: 400 without a model, True on the first
    reset, False when the model has no stored history to forget."""
    c, m = make_monitor(tmp_path, monkeypatch)
    state.fold_engine_totals("Text A", {"since": 1.0, "requests": 2, "prompt_tokens": 100,
                                        "reused": 0, "output_tokens": 40, "prompt_ms": 50.0,
                                        "decode_ms": 800.0})
    with TestClient(c.app) as client:
        assert client.post("/api/control/reset_stats", json={}).status_code == 400
        r = client.post("/api/control/reset_stats", json={"model": "Text A"})
        assert r.status_code == 200 and r.json()["reset"] is True
        assert state.read_token_stats().get("Text A") is None
        assert client.post("/api/control/reset_stats", json={"model": "Text A"}).json()["reset"] is False


def test_schedule_window_route_and_allowed_now(tmp_path, monkeypatch):
    """POST /api/control/schedule writes the daily window file the door reads,
    and the catalog marks a model outside its window as not allowed right now
    (without ever disabling its manual On demand checkbox)."""
    c, m = make_monitor(tmp_path, monkeypatch)

    async def fake_waitroom():
        return {"loaded_model": "Text A", "active": None, "inflight": 0, "waiting": []}

    async def fake_registry():
        return m.registry_models, None

    monkeypatch.setattr(collectors, "registry_models", fake_registry)
    monkeypatch.setattr(collectors, "waitroom_snapshot", fake_waitroom)

    with TestClient(c.app) as client:
        assert client.post("/api/control/schedule", json={}).status_code == 400
        assert client.post("/api/control/schedule",
                           json={"model": "a", "start": "25:00", "end": "06:00"}).status_code == 400
        r = client.post("/api/control/schedule",
                        json={"model": "a", "start": "20:00", "end": "06:00"})
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "Text A"
        assert r.json()["window"] == {"start": "20:00", "end": "06:00"}

    snap = asyncio.run(m.snapshot())
    row = {x["name"]: x for x in snap["models"]}["Text A"]
    assert row["schedule"] == {"start": "20:00", "end": "06:00"}
    # 20:00-06:00 is a tiny slice of the day; whatever the wall clock says,
    # allowed_now must agree with model_state, and the manual switch stays on.
    assert row["allowed_now"] == state.schedule_allows("Text A", {"Text A": {"start": "20:00", "end": "06:00"}})
    assert row["on_demand"] is True, "a window does not touch the manual switch"
    assert snap["model"]["schedule_windows"]["Text A"]["start"] == "20:00"

    # 00:00-23:59 covers every minute of the day, so allowed_now flips True.
    with TestClient(c.app) as client:
        client.post("/api/control/schedule",
                    json={"model": "Text A", "start": "00:00", "end": "23:59"})
    row = {x["name"]: x for x in asyncio.run(m.snapshot())["models"]}["Text A"]
    assert row["allowed_now"] is True

    with TestClient(c.app) as client:
        r = client.post("/api/control/schedule", json={"model": "Text A"})
    assert r.json()["window"] is None
    row = {x["name"]: x for x in asyncio.run(m.snapshot())["models"]}["Text A"]
    assert row["schedule"] is None and row["allowed_now"] is True


def test_model_rows_show_all_time_averages_that_survive_model_change(tmp_path, monkeypatch):
    """Spec 2026-11-25: every model row carries its ALL-TIME average speed,
    persisted by folding the engine's own lifetime counters; parked models
    keep showing theirs, and the fold adds only the delta between polls."""
    c, m = make_monitor(tmp_path, monkeypatch)
    stats_file = tmp_path / "model-token-stats.json"
    monkeypatch.setattr(state, "DEFAULT_TOKEN_STATS_FILE", stats_file)

    totals_a = {"since": 100.0, "requests": 10, "prompt_tokens": 15000, "reused": 5000,
                "output_tokens": 600, "prompt_ms": 4000.0, "decode_ms": 10000.0}

    async def fake_engine_metrics(port, key):
        return {"ok": True, "decode_tps": 66, "prefill_tps": 1500,
                "prompt_total": 5000, "reused_tokens": 2000, "prompt_read": 3000,
                "prompt_tokens": 6000, "totals": dict(totals_a),
                "source": "engine/metrics", "phase": "generating",
                "live_state": "reading", "live_prompt_read": 500,
                "live_prompt_total": 2000}

    async def fake_registry():
        return m.registry_models, None

    async def fake_waitroom():
        return {"loaded_model": "Text A", "active": "Text A", "inflight": 1, "waiting": []}

    monkeypatch.setattr(collectors, "registry_models", fake_registry)
    monkeypatch.setattr(collectors, "waitroom_snapshot", fake_waitroom)
    monkeypatch.setattr(collectors, "engine_metrics", fake_engine_metrics)

    snap = asyncio.run(m.snapshot())
    m.history.add(snap)
    rows = {x["name"]: x for x in snap["models"]}
    # All-time means from the engine's lifetime totals: decode 600 tok / 10 s
    # = 60 t/s; prefill counts only tokens actually READ: (15000-5000)/4 s.
    assert rows["Text A"]["avg_decode_tps"] == 60.0
    assert rows["Text A"]["avg_prefill_tps"] == 2500.0
    assert rows["Text A"]["stats_requests"] == 10
    # Never-loaded model has no history yet — the row exists, the cells sit blank.
    assert rows["Media B"]["avg_decode_tps"] is None
    assert stats_file.exists(), "the all-time sums are persisted"
    # Live-block fields ride through for the top-bar prompt-progress pill.
    assert snap["system"]["engine_live_state"] == "reading"
    assert snap["system"]["engine_live_prompt_read"] == 500
    assert snap["system"]["engine_live_prompt_total"] == 2000
    # Engine-measured prompt accounting, flattened for the queue boxes.
    assert snap["system"]["engine_prompt_total"] == 5000
    assert snap["system"]["engine_prompt_read"] == 3000
    assert snap["system"]["engine_reused_tokens"] == 2000

    # A second poll of the SAME engine run folds only the DELTA, not the totals.
    totals_a.update({"requests": 12, "prompt_tokens": 17000, "reused": 5000,
                     "output_tokens": 700, "prompt_ms": 4400.0, "decode_ms": 11000.0})
    snap = asyncio.run(m.snapshot())
    rows = {x["name"]: x for x in snap["models"]}
    assert abs(rows["Text A"]["avg_decode_tps"] - 700 / 11.0) < 0.01
    assert abs(rows["Text A"]["avg_prefill_tps"] - 12000 / 4.4) < 0.5

    # Swap to Media B (a new engine run with its own totals): Media B gets its
    # own averages, Text A's row keeps its all-time numbers from the file.
    async def fake_waitroom_b():
        return {"loaded_model": "Media B", "active": "Media B", "inflight": 1, "waiting": []}

    async def fake_engine_metrics_b(port, key):
        # strata-style schema: no explicit reused — the cached share must be
        # derived as total minus recomputed.
        return {"ok": True, "decode_tps": 9, "prefill_tps": 120,
                "prompt_total": 400, "prompt_read": 100, "reused_tokens": None,
                "totals": {"since": 200.0, "requests": 4, "prompt_tokens": 800, "reused": 0,
                           "output_tokens": 200, "prompt_ms": 500.0, "decode_ms": 20000.0},
                "source": "engine/metrics"}

    monkeypatch.setattr(collectors, "waitroom_snapshot", fake_waitroom_b)
    monkeypatch.setattr(collectors, "engine_metrics", fake_engine_metrics_b)
    snap = asyncio.run(m.snapshot())
    rows = {x["name"]: x for x in snap["models"]}
    assert rows["Media B"]["avg_decode_tps"] == 10.0 and rows["Media B"]["avg_prefill_tps"] == 1600.0
    assert abs(rows["Text A"]["avg_decode_tps"] - 700 / 11.0) < 0.01, \
        "a parked model keeps its own all-time averages"
    assert snap["system"]["decode_tps"] == 9, "the live fields follow the loaded model only"
    assert snap["system"]["engine_prompt_total"] == 400
    assert snap["system"]["engine_reused_tokens"] == 300, \
        "cached share derived from total - recomputed when the engine omits it"

    # Reset forgets a model; its row blanks while the sums restart from zero.
    assert state.reset_model_stats("Text A", path=stats_file) is True
    snap = asyncio.run(m.snapshot())
    rows = {x["name"]: x for x in snap["models"]}
    assert rows["Text A"]["avg_decode_tps"] is None and rows["Text A"]["avg_prefill_tps"] is None


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
    assert out["decode_tps"] == round(18.5), "last finished request's decode rate"
    assert abs(out["prefill_tps"] - 270.8) < 1.0, "(31885-25512)/23.5278 s"
    assert abs(out["cache_hit_rate"] - 0.800) < 0.01, "reused/prompt_total"
    assert out["prompt_total"] == 31885 and out["reused_tokens"] == 25512
    assert out["lifetime_requests"] == 9
    assert out["totals"] == {"requests": 9}, "lifetime counters pass through for the all-time fold"
    assert out["live_state"] == "reading", "the live block's state drives the top-bar pill"


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
