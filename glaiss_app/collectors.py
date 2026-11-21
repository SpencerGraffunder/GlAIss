"""LLPC/llama-swap/system collectors for the GlAIss dashboard."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError
# `quote` is re-exported on purpose: monitor.py, slot-watchdog.py and waitroom.py
# call it as C.quote(), so it looks unused inside this module.
from urllib.parse import quote
import urllib.request

import httpx

from .model_state import (
    DEFAULT_INTENT_FILE,
    DEFAULT_LASTMODEL_FILE,
    DEFAULT_LASTSWAP_FILE,
    DEFAULT_MANUAL_FILE,
    DEFAULT_SWAP_DESIRED_FILE,
    read_lock,
    read_manual,
)

LLPC_HOME = Path(os.environ.get("LLPC_HOME", "/home/graffunder"))
LLPC_SWAP_DIR = Path(os.environ.get("LLPC_SWAP_DIR", str(LLPC_HOME / "llama-swap")))
LLPC_STATE_DIR = Path(os.environ.get("LLPC_STATE_DIR", str(LLPC_HOME / ".llama-swap")))
LLPC_MODELS_PY = Path(os.environ.get("LLPC_MODELS_PY", str(LLPC_SWAP_DIR / "llpc_models.py")))
LLPC_CONFIG_YAML = Path(os.environ.get("LLPC_CONFIG_YAML", str(LLPC_SWAP_DIR / "config.yaml")))
LLPC_SWAP_TO = Path(os.environ.get("LLPC_SWAP_TO", str(LLPC_SWAP_DIR / "swap-to.sh")))
WAITROOM_URL = os.environ.get("WAITROOM_URL", "http://127.0.0.1:8080").rstrip("/")
LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://127.0.0.1:8090").rstrip("/")
RAPL_SAMPLE_SCRIPT = Path(__file__).resolve().parent / "rapl_sample.py"
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

SERVICE_ROWS: List[Tuple[str, str, Optional[str]]] = [
    ("strata31", "strata31", "Strata 3.1 · Qwen Flash Next Q4"),
    ("llama-qwen", "llama-qwen", "llama.cpp Qwen 27B"),
    ("comfy-ui", "comfyui", "ComfyUI media engine"),
    ("llama-swap", "llama-swap", "llama-swap gatekeeper"),
    ("whisper-server", "whisper-server", "Whisper.cpp GPU STT"),
    ("whisper-shim", "whisper-shim", "OpenAI STT shim"),
    ("switcheroo-control", "switcheroo-control", "PRIME switch proxy"),
    ("model-waitroom", "model-waitroom", "llpc model waitroom"),
    ("strata-dsv41.service", "strata-dsv41", "Strata DeepSeek V4.1"),
    ("llama-server", None, "llama-server process under llama-qwen"),
]

MODEL_PORTS = {
    "strata31": 8085,
    "llama-qwen": 8086,
    "strata-dsv41": 8087,
    "comfyui": 8188,
}


async def run_cmd(cmd: List[str], timeout: float = 5.0, text: bool = True) -> Tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if text:
            return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")
        return proc.returncode or 0, stdout, stderr  # type: ignore[return-value]
    except asyncio.TimeoutError:
        try:
            proc.kill()  # type: ignore[name-defined]
        except Exception:
            pass
        return 124, "", f"timeout after {timeout}s"
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except Exception as exc:
        return 1, "", str(exc)


async def http_json(url: str, *, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0, method: str = "GET", data: Any = None):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, headers=headers or {}, json=data if method != "GET" else None)
            if resp.status_code >= 400:
                try:
                    body = resp.json()
                except Exception:
                    body = {"detail": resp.text[:500]}
                return {"ok": False, "status": resp.status_code, "error": body}
            try:
                return {"ok": True, "status": resp.status_code, "data": resp.json()}
            except Exception:
                return {"ok": True, "status": resp.status_code, "data": resp.text[:5000]}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": url}


def http_json_sync(url: str, *, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0, method: str = "GET", data: Any = None):
    try:
        body = json.dumps(data).encode() if (method != "GET" and data is not None) else None
        req = urllib.request.Request(url, method=method, headers=headers or {}, data=body)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return {"ok": True, "status": resp.status, "data": json.loads(raw)}
            except Exception:
                return {"ok": True, "status": resp.status, "data": raw[:5000]}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"detail": raw[:500]}
        return {"ok": False, "status": exc.code, "error": parsed}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": url}


ENGINE_METRICS_TIMEOUT = float(os.environ.get("GLAISS_ENGINE_METRICS_TIMEOUT", "4"))


def parse_engine_metrics(payload: Any) -> Dict[str, Any]:
    """Normalize an engine ``/metrics`` payload into console token-speed numbers.

    Strata reports per-request ``decode_tok_s`` plus ``prompt_read``/``prompt_ms``
    for the prefill phase. ``prompt_total`` includes KV-cache-reused tokens, so the
    recompute rate uses ``prompt_read`` — otherwise a warm prompt would read as an
    impossibly fast prefill.
    """
    out: Dict[str, Any] = {
        "ok": False, "error": "no metrics payload", "decode_tps": None, "prefill_tps": None,
        "phase": None, "prompt_tokens": None, "prompt_read": None, "generated": None,
        "cache_hit_rate": None, "requests": 0, "lifetime_requests": None, "source": None,
    }
    if not isinstance(payload, dict) or not payload:
        out["error"] = "no metrics payload"
        return out
    live_raw = payload.get("live")
    live: Dict[str, Any] = live_raw if isinstance(live_raw, dict) else {}
    reqs = [r for r in (payload.get("requests") or []) if isinstance(r, dict)]
    last: Dict[str, Any] = reqs[-1] if reqs else {}
    if not live and not reqs and not payload.get("history"):
        out["error"] = "engine returned a metrics payload with no data"
        return out

    decode = last.get("decode_tok_s")
    if not decode:
        tail = [v for v in ((payload.get("history") or {}).get("tok_s") or [])
                if isinstance(v, (int, float)) and v > 0]
        decode = tail[-1] if tail else (live.get("tok_s") or None)
    prefill = None
    if last.get("prompt_ms") and last.get("prompt_read"):
        prefill = float(last["prompt_read"]) / (float(last["prompt_ms"]) / 1000.0)
    if not prefill and live.get("prefill_tok_s"):
        prefill = live.get("prefill_tok_s")

    out.update({
        "ok": True, "error": None,
        "decode_tps": round(float(decode), 1) if decode else None,
        "prefill_tps": round(float(prefill), 1) if prefill else None,
        "phase": live.get("phase") or live.get("state"),
        "prompt_tokens": live.get("prompt_tokens"),
        "prompt_read": last.get("prompt_read"),
        "generated": live.get("generated"),
        "cache_hit_rate": last.get("hit_rate"),
        "requests": len(reqs),
        "lifetime_requests": (payload.get("totals") or {}).get("requests") if isinstance(payload.get("totals"), dict) else None,
        "source": "engine/metrics",
    })
    return out


async def engine_metrics(port: Optional[int], api_key: Optional[str] = None) -> Dict[str, Any]:
    """Token speed straight from the loaded engine.

    This is the ONLY token-speed source: the console never sits in the request
    path (the waitroom door at :8080 is the single proxy), so it cannot observe
    traffic itself.
    """
    if not port:
        return {"ok": False, "error": "no engine port for the loaded model", "decode_tps": None, "prefill_tps": None}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    resp = await http_json(f"http://127.0.0.1:{port}/metrics", headers=headers, timeout=ENGINE_METRICS_TIMEOUT)
    parsed = parse_engine_metrics(resp.get("data") if resp.get("ok") else None)
    if not resp.get("ok"):
        err = resp.get("error")
        parsed["error"] = err if isinstance(err, str) else json.dumps(err)[:200]
    return parsed


def read_llama_swap_key() -> Optional[str]:
    """Return first llama-swap api key from the existing llpc config."""
    try:
        in_keys = False
        for line in LLPC_CONFIG_YAML.read_text().splitlines():
            if line.strip().startswith("apiKeys:"):
                in_keys = True
                continue
            if in_keys and re.match(r"^\s{2,}", line):
                if not line.strip().startswith("-"):
                    break
                m = re.match(r"^\s*-\s*[\"']?([^\"'#]+)", line)
                if m:
                    return m.group(1).strip()
            elif in_keys and line and not line.startswith(" "):
                break
    except Exception:
        return None
    return None


def read_last_swap() -> Optional[float]:
    try:
        value = float(DEFAULT_LASTSWAP_FILE.read_text().strip())
        return value or None
    except Exception:
        return None


def read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip() or None
    except Exception:
        return None


def read_json_file(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


async def registry_models() -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    if not LLPC_MODELS_PY.exists():
        return None, f"registry missing: {LLPC_MODELS_PY}"
    rc, out, err = await run_cmd([sys.executable, str(LLPC_MODELS_PY)], timeout=5)
    if rc != 0:
        return None, err or out
    try:
        data = json.loads(out)
        return data.get("models", []), None
    except Exception as exc:
        return None, f"registry JSON parse failed: {exc}"


def registry_models_sync() -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    if not LLPC_MODELS_PY.exists():
        return None, f"registry missing: {LLPC_MODELS_PY}"
    try:
        proc = subprocess.run([sys.executable, str(LLPC_MODELS_PY)], capture_output=True, text=True, timeout=5)
    except Exception as exc:
        return None, str(exc)
    if proc.returncode != 0:
        return None, proc.stderr or proc.stdout
    try:
        data = json.loads(proc.stdout)
        return data.get("models", []), None
    except Exception as exc:
        return None, f"registry JSON parse failed: {exc}"


def resolve_registry_model(models: Iterable[Dict[str, Any]], value: str) -> Optional[Dict[str, Any]]:
    needle = (value or "").casefold()
    for m in models:
        if (m.get("name") or "").casefold() == needle:
            return m
        aliases = [str(x).casefold() for x in (m.get("aliases") or [])]
        if needle in aliases or (m.get("unit") or "").casefold() == needle:
            return m
    return None


async def systemctl_show(unit: str) -> Dict[str, str]:
    rc, out, _ = await run_cmd(
        [
            "systemctl",
            "show",
            unit,
            "-p", "Id",
            "-p", "LoadState",
            "-p", "ActiveState",
            "-p", "SubState",
            "-p", "UnitFileState",
            "-p", "MainPID",
            "-p", "ControlGroup",
            "-p", "ExecMainStartTimestamp",
            "-p", "ExecMainExitTimestamp",
            "-p", "ExecMainCode",
            "-p", "ExecMainStatus",
            "-p", "Result",
            "-p", "NRestarts",
        ],
        timeout=5,
    )
    if rc != 0:
        return {"Error": out or f"systemctl rc={rc}"}
    parsed: Dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            parsed[k] = v
    return parsed


def unit_cgroup_pids(control_group: Optional[str]) -> List[int]:
    if not control_group:
        return []
    p = Path("/sys/fs/cgroup") / control_group.lstrip("/") / "cgroup.procs"
    try:
        return [int(x) for x in p.read_text().split()]
    except Exception:
        return []


def find_process_pids_by_comm(comm: str, parent_unit: Optional[str] = None) -> List[int]:
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "comm").read_text().strip() == comm:
                pid = int(proc.name)
                if parent_unit:
                    try:
                        cgroup = (proc / "cgroup").read_text()
                        if parent_unit not in cgroup:
                            continue
                    except Exception:
                        continue
                pids.append(pid)
        except Exception:
            continue
    return pids


def proc_summary(pids: List[int]) -> Dict[str, Any]:
    rss_kib = 0
    cpu_jiffies = 0
    threads = 0
    names: List[str] = []
    for pid in pids:
        try:
            stat = (Path("/proc") / str(pid) / "stat").read_text()
            # comm may contain spaces and parentheses: fields after ')'.
            after = stat.split(") ", 1)[1].split()
            rss_pages = int(after[19]) if len(after) > 19 else 0
            utime = int(after[11]) if len(after) > 11 else 0
            stime = int(after[12]) if len(after) > 12 else 0
            rss_kib += rss_pages * PAGE_SIZE // 1024
            cpu_jiffies += utime + stime
            try:
                names.append((Path("/proc") / str(pid) / "comm").read_text().strip())
            except Exception:
                pass
            try:
                for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
                    if line.startswith("Threads:"):
                        threads += int(line.split()[1])
                        break
            except Exception:
                pass
        except Exception:
            continue
    return {
        "pid_count": len(pids),
        "rss_mib": round(rss_kib / 1024.0, 1),
        "cpu_jiffies": cpu_jiffies,
        "threads": threads,
        "comm": names[0] if names else None,
    }


async def nvidia_query() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    rc, out, err = await run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,power.draw,utilization.gpu,memory.used,memory.total,temperature.gpu,fan.speed",
            "--format=csv,noheader,nounits",
        ],
        timeout=3,
    )
    if rc != 0:
        return [], [], err or out
    gpus = []
    for row in csv.DictReader(io.StringIO(out), fieldnames=[
        "index", "name", "power_draw_w", "util_gpu_percent",
        "memory_used_mib", "memory_total_mib", "temperature_c", "fan_percent",
    ]):
        row = {k: v.strip() if v else v for k, v in row.items()}
        try:
            row["power_draw_w"] = float(row.get("power_draw_w", 0) or 0)
        except Exception:
            row["power_draw_w"] = None
        for key in ["util_gpu_percent", "memory_used_mib", "memory_total_mib", "temperature_c", "fan_percent"]:
            try:
                row[key] = float(row.get(key, 0) or 0)
            except Exception:
                row[key] = None
        gpus.append(row)
    rc2, out2, _ = await run_cmd(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        timeout=3,
    )
    apps = []
    if rc2 == 0:
        for row in csv.DictReader(io.StringIO(out2), fieldnames=["pid", "process_name", "used_memory_mib"]):
            try:
                apps.append({
                    "pid": int(row["pid"]),
                    "process_name": row.get("process_name"),
                    "used_memory_mib": float(row.get("used_memory_mib") or 0),
                })
            except Exception:
                continue
    return gpus, apps, err if rc != 0 else None


def parse_meminfo() -> Dict[str, float]:
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            mem[key] = int(rest.strip().split()[0]) * 1024
    except Exception:
        return {}
    total = mem.get("MemTotal", 0)
    available = mem.get("MemAvailable", mem.get("MemFree", 0))
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": max(0, total - available),
        "free_bytes": mem.get("MemFree", 0),
        "swap_total_bytes": mem.get("SwapTotal", 0),
        "swap_used_bytes": max(0, mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)),
    }


def proc_stat_cpu() -> Tuple[float, Dict[str, Any]]:
    ts = time.time()
    try:
        first = Path("/proc/stat").read_text().splitlines()[0]
        parts = first.split()[1:]
        vals = [int(x) for x in parts]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        return ts, {"idle": idle, "total": total}
    except Exception:
        return ts, {"idle": 0, "total": 0}


def loadavg() -> List[float]:
    try:
        return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except Exception:
        return []


async def rapl_energy() -> Tuple[Dict[str, int], Dict[str, int], Optional[str]]:
    cmd = ["sudo", "-n", sys.executable, str(RAPL_SAMPLE_SCRIPT)]
    rc, out, err = await run_cmd(cmd, timeout=3)
    if rc != 0:
        # Fallback direct read; works only where permissions allow it.
        result = {}
        maxes = {}
        import glob
        for path in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj") + glob.glob("/sys/class/powercap/intel-rapl:*:*/energy_uj"):
            try:
                result[path] = int(Path(path).read_text().strip())
                max_path = Path(path).with_name("max_energy_range_uj")
                maxes[path] = int(max_path.read_text().strip()) if max_path.exists() else 0
            except Exception:
                continue
        if result:
            return result, maxes, None
        return {}, {}, (err or out).strip() or "unavailable"
    try:
        data = json.loads(out)
        return (
            {d["path"]: int(d["energy_uj"]) for d in data if d.get("path")},
            {d["path"]: int(d.get("max_energy_range_uj") or 0) for d in data if d.get("path")},
            None,
        )
    except Exception as exc:
        return {}, {}, f"RAPL parse failed: {exc}"


def read_waitroom_fallback() -> Dict[str, Any]:
    desired = read_text_file(DEFAULT_SWAP_DESIRED_FILE)
    return {
        "loaded_model": read_text_file(DEFAULT_LASTMODEL_FILE),
        "intent": read_json_file(DEFAULT_INTENT_FILE),
        "last_model": read_text_file(DEFAULT_LASTMODEL_FILE),
        "desired_unit": desired,
        "comfy_claimed": desired == "comfyui",
        "last_swap_unix": read_last_swap(),
        "lock": read_lock(),
        "manual_control": read_manual(),
        "source": "files",
    }


@dataclass
class CollectorCache:
    cpu_prev: Tuple[float, Dict[str, Any]] | None = None
    rapl_prev: Tuple[float, Dict[str, int]] | None = None
    service_cpu_prev: Dict[str, Tuple[float, int]] = field(default_factory=dict)

    def cpu_percent(self) -> Optional[float]:
        ts, cur = proc_stat_cpu()
        prev = self.cpu_prev
        self.cpu_prev = (ts, cur)
        if not prev:
            return None
        old_ts, old = prev
        dt = ts - old_ts
        dtotal = cur.get("total", 0) - old.get("total", 0)
        didle = cur.get("idle", 0) - old.get("idle", 0)
        if dt <= 0 or dtotal <= 0:
            return None
        return round(max(0.0, min(100.0, (1.0 - didle / dtotal) * 100.0)), 1)

    def rapl_power_w(self) -> Tuple[Optional[float], Optional[str]]:
        energies, maxes, err = rapl_energy_nowait() if False else ({}, {}, None)  # type: ignore
        return None, err


async def rapl_power_w(cache: CollectorCache) -> Tuple[Optional[float], Optional[str], List[Dict[str, Any]]]:
    energies, maxes, err = await rapl_energy()
    ts = time.time()
    prev = cache.rapl_prev
    cache.rapl_prev = (ts, energies)
    if not energies:
        return None, err or "no RAPL domains", []
    if not prev:
        return None, err or "waiting for second RAPL sample", [{"path": k, "energy_uj": v} for k, v in energies.items()]
    old_ts, old_energies = prev
    dt = max(0.001, ts - old_ts)
    total_uj = 0.0
    domains = []
    for path, value in energies.items():
        old = old_energies.get(path)
        if old is None:
            continue
        delta = value - old
        max_range = maxes.get(path) or 0
        if delta < 0 and max_range:
            delta += max_range
        if delta < 0:
            continue
        total_uj += delta
        domains.append({"path": path, "watts": round(delta / 1_000_000 / dt, 2)})
    watts = total_uj / 1_000_000 / dt if total_uj else None
    return (round(watts, 2) if watts is not None else None), err, domains


def rapl_energy_nowait():
    raise RuntimeError("unused")


async def waitroom_snapshot() -> Dict[str, Any]:
    state = await http_json(f"{WAITROOM_URL}/state", timeout=3)
    if state.get("ok") and isinstance(state.get("data"), dict):
        data = dict(state["data"])
        data["source"] = "waitroom/state"
        return data
    queue = await http_json(f"{WAITROOM_URL}/queue", timeout=3)
    if queue.get("ok") and isinstance(queue["data"], dict):
        data = dict(read_waitroom_fallback())
        data.update(queue["data"])
        data["loaded_model"] = queue["data"].get("loaded") or data.get("loaded_model")
        data["source"] = "waitroom/queue+files"
        return data
    data = read_waitroom_fallback()
    data["error"] = state.get("error") or queue.get("error") or "waitroom unreachable"
    return data


def waitroom_snapshot_sync() -> Dict[str, Any]:
    state = http_json_sync(f"{WAITROOM_URL}/state", timeout=3)
    if state.get("ok") and isinstance(state.get("data"), dict):
        data = dict(state["data"])
        data["source"] = "waitroom/state"
        return data
    queue = http_json_sync(f"{WAITROOM_URL}/queue", timeout=3)
    if queue.get("ok") and isinstance(queue["data"], dict):
        data = dict(read_waitroom_fallback())
        data.update(queue["data"])
        data["loaded_model"] = queue["data"].get("loaded") or data.get("loaded_model")
        data["source"] = "waitroom/queue+files"
        return data
    data = read_waitroom_fallback()
    data["error"] = state.get("error") or queue.get("error") or "waitroom unreachable"
    return data


async def llama_swap_running() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    key = read_llama_swap_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    result = await http_json(f"{LLAMA_SWAP_URL}/running", headers=headers, timeout=3)
    if result.get("ok"):
        return result.get("data"), None
    return None, result.get("error") or str(result)


def llama_swap_running_sync() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    key = read_llama_swap_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    result = http_json_sync(f"{LLAMA_SWAP_URL}/running", headers=headers, timeout=3)
    if result.get("ok"):
        return result.get("data"), None
    return None, result.get("error") or str(result)


async def service_row(
    display: str,
    canonical: Optional[str],
    label: Optional[str],
    cache: CollectorCache,
    nvidia_apps: List[Dict[str, Any]],
    logs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.time()
    if canonical is None:
        pids = find_process_pids_by_comm("llama-server", parent_unit="llama-qwen")
        summary = proc_summary(pids)
        vram = sum(a.get("used_memory_mib", 0) for a in nvidia_apps if a.get("pid") in pids)
        return {
            "display": display,
            "unit": None,
            "display_label": label,
            "kind": "synthetic",
            "state": "process-active" if pids else "inactive",
            "main_pid": pids[0] if pids else 0,
            "pids": pids,
            "rss_mib": summary.get("rss_mib", 0),
            "cpu_percent": None,
            "threads": summary.get("threads", 0),
            "vram_mib": vram,
            "logs": logs or {},
            "updated_at": now,
        }
    show = await systemctl_show(canonical)
    pids = unit_cgroup_pids(show.get("ControlGroup"))
    main_pid = int(show.get("MainPID") or 0)
    if main_pid and main_pid not in pids:
        pids.insert(0, main_pid)
    summary = proc_summary(pids)
    prev = cache.service_cpu_prev.get(canonical)
    cache.service_cpu_prev[canonical] = (now, int(summary.get("cpu_jiffies") or 0))
    cpu_percent = None
    if prev:
        old_ts, old_cpu = prev
        dt = max(0.001, now - old_ts)
        delta = int(summary.get("cpu_jiffies") or 0) - old_cpu
        if delta >= 0:
            cpu_percent = round(max(0.0, (delta / CLK_TCK / dt) * 100.0), 1)
    vram = sum(a.get("used_memory_mib", 0) for a in nvidia_apps if a.get("pid") in pids)
    return {
        "display": display,
        "unit": canonical,
        "display_label": label,
        "kind": "systemd",
        "state": show.get("ActiveState", "unknown"),
        "sub_state": show.get("SubState"),
        "load_state": show.get("LoadState"),
        "unit_file_state": show.get("UnitFileState"),
        "main_pid": main_pid,
        "pids": pids[:20],
        "control_group": show.get("ControlGroup"),
        "exec_main_start": show.get("ExecMainStartTimestamp") or None,
        "exec_main_exit": show.get("ExecMainExitTimestamp") or None,
        "result": show.get("Result"),
        "restarts": int(show.get("NRestarts") or 0),
        "rss_mib": summary.get("rss_mib", 0),
        "cpu_percent": cpu_percent,
        "threads": summary.get("threads", 0),
        "vram_mib": vram,
        "port": MODEL_PORTS.get(canonical),
        "logs": logs or {},
        "updated_at": now,
    }

