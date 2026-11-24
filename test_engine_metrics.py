"""Engine token-speed collector tests."""

from __future__ import annotations

from glaiss_app import collectors as C
from glaiss_app.monitor import History

ENGINE_FIXTURE = {
    "live": {"state": "decoding", "phase": "generating", "prompt_tokens": 51081,
             "generated": 120, "tok_s": 64.2},
    "requests": [
        {"time": 1, "prompt_total": 81709, "prompt_read": 200, "prompt_ms": 5000.0,
         "decode_ms": 4000.0, "decode_tok_s": 66.7, "hit_rate": 0.789},
        {"time": 2, "prompt_total": 171122, "prompt_read": 196, "prompt_ms": 5353.1,
         "decode_ms": 3623.5, "decode_tok_s": 67.1, "hit_rate": 0.773},
    ],
    "history": {"tok_s": [61.9, 0.0, 72.0]},
    "totals": {"requests": 250, "prompt_tokens": 25359010, "prompt_ms": 1856541.6},
}


def test_parse_engine_metrics_prefers_live_then_last_request():
    out = C.parse_engine_metrics(ENGINE_FIXTURE)
    assert out["ok"] is True
    # An actively generating engine reports tok_s; that live rate beats the
    # finished request's average (the fixture's live tok_s is 64.2). Speeds are
    # whole t/s — Spec: secondary stats never show decimals.
    assert out["decode_tps"] == 64
    # prefill must use prompt_read (recomputed tokens), not prompt_total (KV-reused)
    assert out["prefill_tps"] == round(196 / 5.3531)
    assert out["cache_hit_rate"] == 0.773
    assert out["phase"] == "generating"
    assert out["requests"] == 2
    assert out["lifetime_requests"] == 250

    # Idle engine: no live tok_s, so the last finished request's rate is used.
    idle = {**ENGINE_FIXTURE, "live": {"state": "idle"}}
    assert C.parse_engine_metrics(idle)["decode_tps"] == 67


def test_parse_engine_metrics_falls_back_to_history_tail():
    fixture = {"history": {"tok_s": [10.0, 0.0, 45.5, 0.0]}, "live": {}}
    out = C.parse_engine_metrics(fixture)
    assert out["decode_tps"] == round(45.5)
    assert out["prefill_tps"] is None


def test_parse_engine_metrics_handles_missing_payload():
    for bad in (None, "text", {}):
        out = C.parse_engine_metrics(bad)
        assert out["ok"] is False
        assert out["decode_tps"] is None


def test_history_records_engine_speed_only():
    """The console is not in the request path, so history has no proxy fallback."""
    hist = History(maxlen=10)
    hist.add({
        "time": 1.0,
        "system": {"decode_tps": 67.1, "prefill_tps": 36.6, "token_speed_source": "engine/metrics"},
    })
    hist.add({"time": 2.0, "system": {"decode_tps": None, "prefill_tps": None}})
    samples = hist.samples
    assert samples[0]["decode_tps"] == 67.1 and samples[0]["prefill_tps"] == 36.6
    assert samples[0]["token_speed_source"] == "engine/metrics"
    assert samples[1]["decode_tps"] is None and "traffic" not in samples[1]
