"""Tests for the non-door stats the console now renders."""
import time

from glaiss_app import collectors
from glaiss_app.monitor import Traffic


def test_gpu_cards_from_engine():
    payload = {
        "hardware": {
            "gpus": [
                {"index": 0, "name": "NVIDIA RTX 3090", "util": 99,
                 "mem_used": 24 * 1048576, "mem_total": 24 * 1048576,
                 "temp": 63, "power": 163.2},
            ],
            "gpu_pcie_rx_mb": 118.9,
            "gpu_pcie_tx_mb": 15.8,
        }
    }
    out = collectors.gpu_cards_from_engine(payload)
    assert out["gpu_cards"][0]["mem_used_mib"] == 24.0
    assert out["gpu_cards"][0]["power_w"] == 163.2
    assert out["pcie_rx_mb"] == 118.9
    assert out["pcie_tx_mb"] == 15.8


def test_nvidia_gpu_cards_shape():
    gpus = [{
        "index": 0, "name": "GPU", "power_draw_w": 12.0,
        "util_gpu_percent": 3.0, "memory_used_mib": 100.0,
        "memory_total_mib": 24000.0, "temperature_c": 30.0,
    }]
    cards = collectors.nvidia_gpu_cards(gpus)
    assert cards[0]["util_percent"] == 3.0
    assert cards[0]["mem_used_mib"] == 100.0
    assert cards[0]["power_w"] == 12.0


def test_parse_comfy_workload():
    stats = {
        "system": {"ram_free": 16 * 1024 * 1024 * 1024},
        "devices": [{"name": "cuda:0", "type": "cuda",
                     "vram_total": 24 * 1024 * 1024 * 1024,
                     "vram_free": 20 * 1024 * 1024 * 1024}],
    }
    queue = {
        "queue_running": [[1, "prompt-abc", {}, {}, {}]],
        "queue_pending": [[2, "p2", {}, {}, {}], [3, "p3", {}, {}, {}]],
    }
    out = collectors.parse_comfy_workload(stats, queue)
    assert out["ok"] is True
    assert out["running"] == 1
    assert out["pending"] == 2
    assert out["running_prompt_id"] == "prompt-abc"
    assert out["devices"][0]["vram_total_mib"] == 24576.0
    assert out["devices"][0]["vram_free_mib"] == 20480.0
    assert out["ram_free_mib"] == 16384.0


def test_traffic_summary_median_and_models():
    t = Traffic()
    now = time.time()
    t.add({"type": "queue_arrive", "model": "Text A", "req_id": 1, "ts": now - 10})
    t.add({"type": "queue_admit", "model": "Text A", "req_id": 1, "ts": now - 6})
    t.add({"type": "queue_admit", "model": "Text B", "req_id": 2, "ts": now - 4})
    t.add({"type": "request_cancel", "model": "Text B", "ts": now - 2})
    t.add({"type": "queue_timeout", "model": "Text B", "ts": now - 1})
    s = t.summary(3600)
    assert s["admissions"] == 2
    assert s["cancellations"] == 1
    assert s["timeouts"] == 1
    assert s["median_queue_wait_s"] == 4.0
    assert s["models"] == {"Text A": 1, "Text B": 1}
