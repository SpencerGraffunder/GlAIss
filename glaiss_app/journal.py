"""journalctl-backed live log ring for GlAIss."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any, Dict, List, Optional

DEFAULT_UNITS = [
    "strata31",
    "llama-qwen",
    "comfyui",
    "llama-swap",
    "whisper-server",
    "whisper-shim",
    "model-waitroom",
    "strata-dsv41",
]

DISPLAY_FOR_UNIT = {
    "comfyui": "comfy-ui",
    "strata-dsv41": "strata-dsv41.service",
}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _ts(record: Dict[str, Any]) -> float:
    try:
        return float(record.get("__REALTIME_TIMESTAMP")) / 1_000_000.0
    except Exception:
        return time.time()


def _priority(record: Dict[str, Any]) -> int:
    return _int(record.get("PRIORITY"), 6)


class JournalRing:
    def __init__(self, units: Optional[List[str]] = None, maxlen: int = 3000):
        self.units = units or DEFAULT_UNITS
        self.maxlen = maxlen
        self.records: deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self.task: Optional[asyncio.Task] = None
        self.started_at: Optional[float] = None
        self.status: Dict[str, Any] = {"running": False, "error": None, "last_ts": None}

    def add(self, raw: str) -> None:
        try:
            rec = json.loads(raw)
        except Exception:
            return
        if not isinstance(rec, dict):
            return
        unit = rec.get("UNIT") or rec.get("SYSTEMD_UNIT")
        comm = rec.get("COMM")
        text = rec.get("MESSAGE") or ""
        routed = set()
        if unit:
            routed.add(DISPLAY_FOR_UNIT.get(unit, unit))
        if comm == "llama-server":
            routed.add("llama-server")
        entry = {
            "ts": _ts(rec),
            # list, not set: records are serialized straight into HTTP JSON
            "unit": sorted(routed) if routed else [unit or "unknown"],
            "priority": _priority(rec),
            "severity": "error" if _priority(rec) <= 3 else "warn" if _priority(rec) <= 4 else "info",
            "comm": comm,
            "pid": rec.get("_PID"),
            "text": str(text)[:2000],
        }
        self.records.append(entry)
        self.status["last_ts"] = entry["ts"]

    async def _stream(self) -> None:
        cmd = ["journalctl", "-f", "-o", "json", "--no-pager"]
        for unit in self.units:
            cmd.extend(["-u", unit])
        while True:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self.status["running"] = True
                self.status["error"] = None
                self.started_at = time.time()
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    self.add(line.decode(errors="replace"))
            except Exception as exc:
                self.status["error"] = str(exc)
            self.status["running"] = False
            await asyncio.sleep(2)

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._stream())

    async def stop(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.status["running"] = False

    def tail(self, limit: int = 300, unit: Optional[str] = None, level: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for entry in reversed(self.records):
            if unit and unit not in entry.get("unit", []):
                continue
            if level and entry.get("severity") != level:
                continue
            out.append(entry)
            if len(out) >= limit:
                break
        return list(reversed(out))

    def counts(self, minutes: float = 5.0) -> Dict[str, Dict[str, int]]:
        cutoff = time.time() - minutes * 60
        counts: Dict[str, Dict[str, int]] = {}
        for entry in self.records:
            if entry.get("ts", 0) < cutoff:
                continue
            for unit in entry.get("unit", []):
                row = counts.setdefault(unit, {"error": 0, "warn": 0, "info": 0, "total": 0})
                sev = entry.get("severity", "info")
                row[sev] = row.get(sev, 0) + 1
                row["total"] += 1
        return counts

    def last_line(self, unit: Optional[str] = None) -> Optional[Dict[str, Any]]:
        for entry in reversed(self.records):
            if unit and unit not in entry.get("unit", []):
                continue
            return entry
        return None
