"""FastAPI monitor/control endpoints for GlAIss + llpc waitroom."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from . import collectors as C
from . import journal as J
from . import model_state as state

DEFAULT_POLL_S = float(os.environ.get("GLAISS_MONITOR_POLL_S", "2"))


@dataclass
class ControlJob:
    id: str
    action: str
    model: Optional[str] = None
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "detail": self.detail,
        }


class History:
    def __init__(self, maxlen: int = 1800):
        self.samples: deque[Dict[str, Any]] = deque(maxlen=maxlen)

    def add(self, snapshot: Dict[str, Any]) -> None:
        system = snapshot.get("system", {}) or {}
        model = snapshot.get("model", {}) or {}
        traffic = snapshot.get("traffic", {}) or {}
        self.samples.append({
            "ts": snapshot.get("time", time.time()),
            "cpu_power_w": system.get("cpu_power_w"),
            "gpu_power_w": system.get("gpu_power_w"),
            "system_power_w": system.get("system_power_w"),
            "cpu_util_percent": system.get("cpu_util_percent"),
            "ram_used_bytes": system.get("ram_used_bytes"),
            "ram_total_bytes": system.get("ram_total_bytes"),
            "vram_used_mib": system.get("vram_used_mib"),
            "gpu_util_percent": system.get("gpu_util_percent"),
            "loaded_model": model.get("current_loaded_model"),
            "queue_active_model": model.get("active_model"),
            "hold_remaining_s": model.get("hold_remaining_s"),
            "inflight": model.get("inflight"),
            "waiting": len(model.get("waiting") or []),
            "decode_tps": traffic.get("avg_decode_tps"),
            "prefill_tps": traffic.get("avg_pp_tps"),
        })

    def tail(self, seconds: float = 900) -> List[Dict[str, Any]]:
        cutoff = time.time() - seconds
        return [s for s in self.samples if s.get("ts", 0) >= cutoff]


class ControlManager:
    def __init__(self, monitor: "Monitor"):
        self.monitor = monitor
        self.jobs: Dict[str, ControlJob] = {}

    async def broadcast(self, msg: Dict[str, Any]) -> None:
        hub = getattr(self.monitor.proxy, "hub", None)
        if hub:
            try:
                await hub.emit(msg)
            except Exception:
                pass

    def active(self) -> List[ControlJob]:
        return [j for j in self.jobs.values() if j.status in {"queued", "running"}]

    def get_catalog_model(self, value: str) -> Optional[Dict[str, Any]]:
        return C.resolve_registry_model(self.monitor.registry_models or [], value)

    async def current_loaded_model(self) -> Optional[str]:
        snap = await C.waitroom_snapshot()
        loaded = snap.get("current_loaded_model") or snap.get("loaded_model") or snap.get("active")
        if loaded:
            return loaded
        running, _ = await C.llama_swap_running()
        for item in (running or {}).get("running", []):
            if item.get("state") == "ready":
                return item.get("model")
        return None

    def new_job(self, action: str, model: Optional[str], detail: Optional[Dict[str, Any]] = None) -> ControlJob:
        job = ControlJob(id=f"ctrl_{int(time.time())}_{uuid.uuid4().hex[:6]}", action=action, model=model, detail=detail or {})
        self.jobs[job.id] = job
        return job

    async def start(self, job: ControlJob, coro_factory) -> Dict[str, Any]:
        if self.active():
            return {"error": {"code": "control_busy", "message": "another manual model-control job is already running"}}

        async def _run() -> None:
            job.status = "running"
            job.started_at = time.time()
            await self.broadcast({"type": "control_job", "job": job.as_dict()})
            try:
                result = await coro_factory(job)
                if job.status not in {"failed", "cancelled"}:
                    job.status = "succeeded"
                    job.detail.update(result or {})
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.finished_at = time.time()
                await self.broadcast({"type": "control_job", "job": job.as_dict()})
                await self.monitor.refresh_now(force=True)

        asyncio.create_task(_run())
        return {"job": job.as_dict()}

    async def load(self, model: str, lock_after_load: bool = False, force_kill_generation: bool = False) -> Dict[str, Any]:
        model_entry = self.get_catalog_model(model)
        if self.monitor.registry_models and not model_entry:
            return {"error": {"code": "unknown_model", "message": f"model not found in llpc registry: {model}"}}
        catalog_name = (model_entry or {}).get("name") or model
        kind = "media" if (model_entry or {}).get("text") is False else "text"
        unit = (model_entry or {}).get("unit")
        if state.lock_conflict(catalog_name):
            status, payload = state.lock_error_payload()
            return {**payload, "status": status}
        state.write_manual(mode="manual", lock_model=catalog_name if lock_after_load else None, updated_by="glaiss:load")

        async def _do(job: ControlJob) -> Dict[str, Any]:
            if kind == "media":
                key = C.read_llama_swap_key()
                headers = {"Authorization": f"Bearer {key}"} if key else {}
                url = f"{C.LLAMA_SWAP_URL}/upstream/{C.quote(catalog_name, safe='')}/system_stats"
                result = await C.http_json(url, headers=headers, timeout=900)
                if not result.get("ok"):
                    raise RuntimeError(str(result))
                job.detail["media_health"] = result
            else:
                cmd = [str(C.LLPC_SWAP_TO), catalog_name]
                if force_kill_generation:
                    cmd.append("--force")
                rc, out, err = await C.run_cmd(cmd, timeout=2400)
                job.detail["swap_to_rc"] = rc
                job.detail["swap_to_tail"] = (out or "")[-1000:]
                if rc != 0:
                    job.exit_code = rc
                    raise RuntimeError((err or out or f"swap-to rc={rc}")[-1000:])
            if lock_after_load:
                state.write_lock(catalog_name, unit=unit or "", kind=kind, reason="manual load and lock", by="glaiss")
            return {"catalog_model": catalog_name, "unit": unit, "kind": kind}

        return await self.start(self.new_job("load", catalog_name, {"force_kill_generation": force_kill_generation}), _do)

    async def unload(self, model: str, force: bool = False, break_lock: bool = False) -> Dict[str, Any]:
        model_entry = self.get_catalog_model(model)
        if self.monitor.registry_models and not model_entry:
            return {"error": {"code": "unknown_model", "message": f"model not found in llpc registry: {model}"}}
        catalog_name = (model_entry or {}).get("name") or model
        lock = state.read_lock()
        if lock and not break_lock and state.lock_conflict(catalog_name, lock=lock):
            status, payload = state.lock_error_payload(lock)
            return {**payload, "status": status}
        if lock and not break_lock:
            return {
                "error": {
                    "code": "lock_active",
                    "message": "The target model is currently locked. Use break_lock=true to unlock and unload.",
                }
            }
        snap = await C.waitroom_snapshot()
        inflight = snap.get("inflight") or 0
        if inflight and not force:
            return {"error": {"code": "request_inflight", "message": f"{inflight} waitroom request(s) are inflight; use force=true only in an emergency", "inflight": inflight}}

        async def _do(job: ControlJob) -> Dict[str, Any]:
            state.write_manual(mode="manual", lock_model=None, updated_by="glaiss:unload")
            if lock and break_lock:
                state.clear_lock()
            key = C.read_llama_swap_key()
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            url = f"{C.LLAMA_SWAP_URL}/api/models/unload/{C.quote(catalog_name, safe='')}"
            result = await C.http_json(url, headers=headers, timeout=30, method="POST")
            if not result.get("ok"):
                raise RuntimeError(str(result))
            for _ in range(30):
                running, _ = await C.llama_swap_running()
                names = {r.get("model") for r in (running or {}).get("running", [])}
                if catalog_name not in names:
                    break
                await asyncio.sleep(1)
            return {"unload_api": result, "catalog_model": catalog_name}

        return await self.start(self.new_job("unload", catalog_name, {"force": force, "break_lock": break_lock}), _do)

    async def lock(self, model: str, load_if_needed: bool = True) -> Dict[str, Any]:
        model_entry = self.get_catalog_model(model)
        if self.monitor.registry_models and not model_entry:
            return {"error": {"code": "unknown_model", "message": f"model not found in llpc registry: {model}"}}
        catalog_name = (model_entry or {}).get("name") or model
        kind = "media" if (model_entry or {}).get("text") is False else "text"
        unit = (model_entry or {}).get("unit")
        current = await self.current_loaded_model()
        if current and state.lock_conflict(current, locked_model=catalog_name):
            if load_if_needed:
                return await self.load(catalog_name, lock_after_load=True)
            return {
                "error": {
                    "code": "other_model_loaded",
                    "message": f"{current} is loaded. Load {catalog_name} first or use load_if_needed=true.",
                }
            }
        state.write_lock(catalog_name, unit=unit or "", kind=kind, reason="manual lock", by="glaiss")
        state.write_manual(mode="manual", lock_model=catalog_name, updated_by="glaiss:lock")
        await self.monitor.refresh_now(force=True)
        return {"lock": state.read_lock()}

    async def unlock(self) -> Dict[str, Any]:
        state.clear_lock()
        state.write_manual(mode="manual", lock_model=None, updated_by="glaiss:unlock")
        await self.refresh_monitor()
        return {"ok": True}

    async def resume_auto(self) -> Dict[str, Any]:
        state.resume_auto()
        await self.refresh_monitor()
        return {"ok": True}

    async def refresh_monitor(self) -> None:
        await self.monitor.refresh_now(force=True)


class Monitor:
    def __init__(self, proxy_module):
        self.proxy = proxy_module
        self.cache = C.CollectorCache()
        self.journal = J.JournalRing()
        self.history = History(maxlen=int(os.environ.get("GLAISS_HISTORY_SAMPLES", "1800")))
        self.control = ControlManager(self)
        self.last_snapshot: Optional[Dict[str, Any]] = None
        self.registry_models: List[Dict[str, Any]] = []
        self.registry_error: Optional[str] = None
        self.last_refresh = 0.0
        self.last_refresh_error: Optional[str] = None
        self.running = False

    async def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        registry, registry_error = await C.registry_models()
        if registry is not None:
            self.registry_models = registry
        if registry_error:
            self.registry_error = registry_error
        waitroom = await C.waitroom_snapshot()
        running, running_error = await C.llama_swap_running()
        gpus, gpu_apps, gpu_error = await C.nvidia_query()
        rapl_w, rapl_error, rapl_domains = await C.rapl_power_w(self.cache)
        cpu = self.cache.cpu_percent()
        mem = C.parse_meminfo()
        load = C.loadavg()
        counts = self.journal.counts()

        services = []
        for display, canonical, label in C.SERVICE_ROWS:
            log_unit = "comfy-ui" if display == "comfy-ui" else display
            logs = {
                "counts": counts.get(log_unit, {}),
                "last": self.journal.last_line(log_unit),
            }
            services.append(await C.service_row(display, canonical, label, self.cache, gpu_apps, logs))

        gpu_power_values = [float(g["power_draw_w"]) for g in gpus if g.get("power_draw_w") is not None]
        vram_used_values = [float(g["memory_used_mib"]) for g in gpus if g.get("memory_used_mib") is not None]
        vram_total_values = [float(g["memory_total_mib"]) for g in gpus if g.get("memory_total_mib") is not None]
        gpu_power = round(sum(gpu_power_values), 2) if gpu_power_values else None
        vram_used = round(sum(vram_used_values), 2) if vram_used_values else None
        vram_total = round(sum(vram_total_values), 2) if vram_total_values else None
        gpu_utils = [g.get("util_gpu_percent") for g in gpus if g.get("util_gpu_percent") is not None]
        gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
        system_power = None
        if rapl_w is not None and gpu_power is not None:
            system_power = round(rapl_w + gpu_power, 2)
        elif rapl_w is not None:
            system_power = rapl_w
        elif gpu_power is not None:
            system_power = gpu_power

        lock = state.read_lock()
        manual = state.read_manual()
        current_loaded_model = waitroom.get("loaded_model") or waitroom.get("active")
        loaded_state = None
        loaded_unit = None
        for item in (running or {}).get("running", []) if isinstance(running, dict) else []:
            if current_loaded_model and item.get("model") == current_loaded_model:
                loaded_state = item.get("state")
                loaded_unit = (C.resolve_registry_model(self.registry_models, item.get("model") or "") or {}).get("unit")
                break

        models = []
        for m in self.registry_models:
            name = m.get("name")
            models.append({
                **m,
                "loaded": bool(current_loaded_model and current_loaded_model == name),
                "locked": bool(lock and state._normalize_model(lock.get("model")) == state._normalize_model(name)),
                "visible_in_picker": not bool(m.get("unlisted")),
            })

        snap = {
            "time": now,
            "collectors": {
                "waitroom": {"ok": "error" not in waitroom, "error": waitroom.get("error"), "source": waitroom.get("source")},
                "llama_swap": {"ok": running is not None, "error": running_error},
                "registry": {"ok": self.registry_error is None, "error": self.registry_error, "models": len(self.registry_models)},
                "gpu": {"ok": bool(gpus), "error": gpu_error},
                "cpu_power": {"ok": rapl_w is not None, "error": rapl_error},
                "journal": self.journal.status,
            },
            "model": {
                **waitroom,
                "current_loaded_model": current_loaded_model,
                "loaded_state": loaded_state,
                "loaded_unit": loaded_unit,
                "llama_swap_running": running,
                "lock": lock,
                "manual_control": manual,
                "model_control_state_files": {
                    "lock": str(state.DEFAULT_LOCK_FILE),
                    "manual": str(state.DEFAULT_MANUAL_FILE),
                    "last_swap": str(state.DEFAULT_LASTSWAP_FILE),
                },
            },
            "models": models,
            "system": {
                "cpu_power_w": rapl_w,
                "cpu_power_domains": rapl_domains,
                "gpu_power_w": round(gpu_power, 2) if gpu_power is not None else None,
                "system_power_w": system_power,
                "system_power_label": "CPU package + GPU board draw",
                "cpu_util_percent": cpu,
                "load_1": load[0] if len(load) > 0 else None,
                "load_5": load[1] if len(load) > 1 else None,
                "load_15": load[2] if len(load) > 2 else None,
                "ram_total_bytes": mem.get("total_bytes"),
                "ram_used_bytes": mem.get("used_bytes"),
                "ram_available_bytes": mem.get("available_bytes"),
                "swap_total_bytes": mem.get("swap_total_bytes"),
                "swap_used_bytes": mem.get("swap_used_bytes"),
                "vram_used_mib": vram_used,
                "vram_total_mib": vram_total,
                "gpu_util_percent": gpu_util,
                "gpus": gpus,
            },
            "services": services,
            "traffic": self.proxy.stats.snapshot() if hasattr(self.proxy, "stats") else {},
            "control": {"jobs": [j.as_dict() for j in sorted(self.control.jobs.values(), key=lambda x: x.created_at, reverse=True)[:30]]},
            "config": {
                "waitroom_url": C.WAITROOM_URL,
                "llama_swap_url": C.LLAMA_SWAP_URL,
                "registry": str(C.LLPC_MODELS_PY),
                "service_rows": [d for d, _, _ in C.SERVICE_ROWS],
            },
        }
        self.last_snapshot = snap
        self.last_refresh = now
        return snap

    async def refresh_now(self, force: bool = False) -> Dict[str, Any]:
        if not force and self.last_snapshot and time.time() - self.last_refresh < 1:
            return self.last_snapshot
        try:
            return await self.snapshot()
        except Exception as exc:
            self.last_refresh_error = str(exc)
            return self.last_snapshot or {"time": time.time(), "error": str(exc)}

    async def loop(self, interval: float = DEFAULT_POLL_S) -> None:
        self.running = True
        while self.running:
            try:
                snap = await self.snapshot()
                self.history.add(snap)
                hub = getattr(self.proxy, "hub", None)
                if hub and hub.clients:
                    lightweight = {
                        "type": "monitor",
                        "time": snap.get("time"),
                        "model": snap.get("model"),
                        "system": snap.get("system"),
                        "services": snap.get("services"),
                        "collectors": snap.get("collectors"),
                    }
                    await hub.broadcast(lightweight)
            except Exception as exc:
                self.last_refresh_error = str(exc)
            await asyncio.sleep(interval)

    def start(self) -> None:
        self.journal.start()
        self._task = asyncio.create_task(self.loop())

    def stop(self) -> None:
        self.running = False


def install(proxy_module) -> None:
    app = proxy_module.app
    monitor = Monitor(proxy_module)
    setattr(app.state, "glaiss_monitor", monitor)
    control = monitor.control

    router = APIRouter()

    @router.get("/api/health")
    async def api_health():
        return {
            "ok": True,
            "time": time.time(),
            "monitor": {
                "last_refresh": monitor.last_refresh,
                "last_error": monitor.last_refresh_error,
                "journal": monitor.journal.status,
                "history_samples": len(monitor.history.samples),
            },
        }

    @router.get("/api/state")
    async def api_state():
        return JSONResponse(await monitor.refresh_now(force=True))

    @router.get("/api/history")
    async def api_history(seconds: float = 900):
        return {"samples": monitor.history.tail(seconds)}

    @router.get("/api/model")
    async def api_model():
        snap = await monitor.refresh_now(force=True)
        return {"model": snap.get("model"), "models": snap.get("models")}

    @router.get("/api/services")
    async def api_services():
        snap = await monitor.refresh_now(force=True)
        return {"services": snap.get("services"), "collectors": snap.get("collectors")}

    @router.get("/api/system")
    async def api_system():
        snap = await monitor.refresh_now(force=True)
        return {"system": snap.get("system"), "traffic": snap.get("traffic")}

    @router.get("/api/logs")
    async def api_logs(limit: int = 300, unit: Optional[str] = None, level: Optional[str] = None):
        return {
            "logs": monitor.journal.tail(limit=limit, unit=unit, level=level),
            "counts": monitor.journal.counts(),
            "status": monitor.journal.status,
        }

    @router.websocket("/ws/logs")
    async def logs_ws(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_text(json.dumps({
                    "type": "logs",
                    "ts": time.time(),
                    "logs": monitor.journal.tail(limit=200),
                    "counts": monitor.journal.counts(),
                }))
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    @router.post("/internal/waitroom/events")
    async def waitroom_event(request: Request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        monitor.journal.add(json.dumps({
            "__REALTIME_TIMESTAMP": str(int(time.time() * 1_000_000)),
            "UNIT": "model-waitroom",
            "PRIORITY": "6",
            "COMM": "waitroom-event",
            "MESSAGE": f"event {payload.get('type', 'unknown')} model={payload.get('model', '')} id={payload.get('req_id', '')}",
        }))
        hub = getattr(proxy_module, "hub", None)
        if hub:
            await hub.emit({"type": "waitroom_event", "event": payload, "ts": time.time()})
        return {"ok": True}

    @router.get("/api/control/jobs")
    async def control_jobs():
        return {"jobs": [j.as_dict() for j in sorted(control.jobs.values(), key=lambda x: x.created_at, reverse=True)]}

    @router.post("/api/control/load")
    async def control_load(request: Request):
        data = await request.json()
        model = data.get("model")
        if not model:
            return JSONResponse({"error": {"message": "model is required"}}, status_code=400)
        result = await control.load(
            model,
            lock_after_load=bool(data.get("lock_after_load", data.get("lock", False))),
            force_kill_generation=bool(data.get("force_kill_generation", False)),
        )
        return JSONResponse(result, status_code=result.get("status") or (409 if "error" in result else 202))

    @router.post("/api/control/unload")
    async def control_unload(request: Request):
        data = await request.json()
        model = data.get("model")
        if not model:
            return JSONResponse({"error": {"message": "model is required"}}, status_code=400)
        result = await control.unload(
            model,
            force=bool(data.get("force", False)),
            break_lock=bool(data.get("break_lock", False)),
        )
        return JSONResponse(result, status_code=result.get("status") or (409 if "error" in result else 202))

    @router.post("/api/control/lock")
    async def control_lock(request: Request):
        data = await request.json()
        model = data.get("model")
        if not model:
            return JSONResponse({"error": {"message": "model is required"}}, status_code=400)
        result = await control.lock(model, load_if_needed=bool(data.get("load_if_needed", True)))
        return JSONResponse(result, status_code=result.get("status") or (409 if "error" in result else 202))

    @router.post("/api/control/unlock")
    async def control_unlock():
        result = await control.unlock()
        return JSONResponse(result)

    @router.post("/api/control/resume_auto")
    async def control_resume_auto():
        result = await control.resume_auto()
        return JSONResponse(result)

    catch_routes = [r for r in app.router.routes if getattr(r, "path", None) == "/{full_path:path}"]
    for route in catch_routes:
        app.router.routes.remove(route)

    before = list(app.router.routes)
    app.include_router(router)
    new_routes = list(app.router.routes)[len(before):]
    app.router.routes = new_routes + before
    for route in catch_routes:
        app.router.routes.append(route)

    enabled_tasks = os.environ.get("GLAISS_DISABLE_MONITOR_TASKS", "").lower() not in {"1", "true", "yes"}

    @app.on_event("startup")
    async def monitor_startup():
        if enabled_tasks:
            monitor.start()

    @app.on_event("shutdown")
    async def monitor_shutdown():
        monitor.stop()
        await monitor.journal.stop()
