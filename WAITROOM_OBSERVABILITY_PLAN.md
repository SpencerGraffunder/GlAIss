> **Historical record (pre-2026-11).** This plan was written while GlAIss still
> shipped its own transparent proxy in front of the waitroom. That proxy was
> removed on 2026-11-21: the repo now has exactly one proxy, `llpc/waitroom.py`
> on :8080, and GlAIss is a console only. Read this for rationale, not for the
> current shape of the code — see `README.md` for that.

# GlAIss + llpc Waitroom Observability Plan

Date: 2026-11-20  
Status: implementation-ready design only  
Owner for implementation: another agent/model  
Primary repo: `/mnt/ai-workspace/glaiss`  
Production target machine: llpc

## 1. Goal

Combine the running llpc model-waitroom system with GlAIss so Spec can see, in one dashboard, everything currently watched manually with:

```bash
journalctl -u strata31 \
  -u llama-qwen \
  -u comfy-ui \
  -u llama-swap \
  -u whisper-server \
  -u whisper-shim \
  -u switcheroo-control \
  -u model-waitroom \
  -u strata-dsv41.service \
  -u llama-server -f
```

The dashboard must visualize:

1. The waitroom queue and which model is loaded.
2. Hysteresis / swap-hold timing.
3. Generating tokens/sec and prefill tokens/sec.
4. Total estimated system power: CPU package power plus GPU power.
5. RAM, swap, CPU utilization, load average, GPU utilization, VRAM, GPU temperatures.
6. Per-service status and live log activity for each unit in the command above.
7. The old GlAIss-style request queue visualization: incoming prompts, response stream, live request chips, queued/running state, and prompt/response panes.
8. Manual controls to load, unload, and lock models as loaded.

## 2. Non-goals

This phase does not replace llama-swap, the waitroom, or any engine.

Do not:

- Replace `:8080` as the canonical llpc inference door.
- Bypass the waitroom and call engine ports directly for normal inference.
- Use `systemctl start/stop` directly against text/media engine units from dashboard controls.
- Use llama-swap `POST /api/models/pause`.
- Test manual load/unload/lock behavior on the live production slot without an explicit maintenance window and Spec's approval.

Allowed:

- GlAIss may run as an optional transparent proxy on a separate port (`:8899`) forwarding to waitroom `:8080`.
- GlAIss may read waitroom and llama-swap state APIs.
- Manual load must use the sanctioned llpc switch path (`~/llama-swap/swap-to.sh`) for text models.
- Manual unload may use llama-swap `POST /api/models/unload/<model>` only after inflight checks.
- Manual lock is a new explicit policy state implemented in waitroom/control state, not an ad hoc systemd kill.

## 3. Current state to preserve

### llpc serving regime

Current model path:

```text
clients
  -> model-waitroom :8080
  -> llama-swap :8090
  -> engine loopback port
```

Current catalog from `~/llama-swap/config.yaml` / `~/llama-swap/llpc_models.py`:

| Catalog model | Unit | Port | Kind | Notes |
|---|---:|---:|---|---|
| `Qwen 3.8 Flash Next Q4` | `strata31` | `:8085` | text | default/resident text model |
| `Qwen 3.8 27B Q6` | `llama-qwen` | `:8086` | text | llama.cpp `llama-server` process |
| `DeepSeek V4.1 Flash` | `strata-dsv41` | `:8087` | text | first boot is production-protected |
| `Comfy Studio` | `comfyui` | `:8188` | media | hidden from `/v1/models`, driven by OWUI pipe |

Important state files:

| File | Meaning |
|---|---|
| `~/llama-swap/waitroom-intent` | explicit switch intent, model plus stale label and timestamp |
| `~/llama-swap/last-model` | last observed served text model; no expiry |
| `~/llama-swap/default-model` | picker/boot fallback |
| `~/llama-swap/waitroom-last-swap` | epoch seconds of last swap trigger/observation; used by hysteresis |
| `~/.llama-swap/swap-desired` | unit name declared by `unit-keepalive.sh`; `comfyui` means the media cycle holds the slot claim |

Existing waitroom endpoints:

- `GET /queue`: active model, inflight, quantum left, handoffs, waiting list.
- `GET /v1/models`: full visible catalog.
- `GET /health`: forwards to current backend health path.
- `POST /v1/chat/completions`: queued inference path.

Existing llama-swap endpoints:

- `GET /running`: currently loaded member(s), state, proxy, ttl.
- `GET /metrics`: authenticated Prometheus metrics for CPU, RAM, network, load, GPU temp, fan, util, VRAM, and GPU power.
- `POST /api/models/unload/{model}`: unload endpoint exists in llama-swap 0.4.x.
- `GET /logs/stream`: engine log stream.

Existing GlAIss capabilities:

- `glaiss.py`: FastAPI transparent proxy, WebSocket hub, request/response tap, queue chips, PP estimate, decode t/s from streamed token deltas.
- `index.html`: prompt pane, response pane, top stats, settings panel, horizontal queue strip.

Existing gaps:

- GlAIss has no waitroom queue model state.
- GlAIss has no service table or journalctl equivalent.
- GlAIss has no system power collector.
- GlAIss has no manual model control.
- The user command includes `comfy-ui`, but the systemd unit is `comfyui`.
- The user command includes `llama-server`, but `llama-server` is not a systemd unit; it is the llama.cpp process used by `llama-qwen.service`.

## 4. Recommended target architecture

Make GlAIss the observability and control plane, while waitroom remains the production inference policy plane.

```text
Clients:
  optional clients -> GlAIss :8899 -> waitroom :8080 -> llama-swap :8090 -> engine
  existing clients -> waitroom :8080 -> llama-swap :8090 -> engine

Dashboard:
  browser -> GlAIss :8899 /, /ws, /api/state, /api/control/*

GlAIss backend collectors:
  - waitroom collector       : GET :8080/state, :8080/queue, :8080/v1/models
  - llama-swap collector     : GET :8090/running, :8090/metrics
  - registry collector       : python3 ~/llama-swap/llpc_models.py
  - systemd collector        : systemctl show, cgroup PID scan
  - journal collector        : journalctl -f -o json for watched units
  - hardware collector       : nvidia-smi, /proc, RAPL via sudo -n
  - engine log parser        : strata/llama.cpp/Comfy/Whisper log lines
  - proxy traffic tap        : existing GlAIss request tap when clients route through GlAIss
```

Recommended deployment:

- Source repo stays at `/mnt/ai-workspace/glaiss`.
- Runtime state should move out of the repo to `~/.local/state/glaiss`.
- Use a local llpc venv or runtime directory; do not rely on a venv checked into the CIFS share.
- Run as `graffunder`, not root.
- Listen on `0.0.0.0:8899` for LAN dashboard only; do not tunnel it unless Cloudflare Access or equivalent is added.

Suggested systemd unit:

```ini
[Unit]
Description=GlAIss waitroom observability dashboard
After=network-online.target model-waitroom.service llama-swap.service
Wants=model-waitroom.service

[Service]
User=graffunder
WorkingDirectory=/home/graffunder/.local/state/glaiss
Environment=PYTHONUNBUFFERED=1
Environment=GLAISS_STATE_DIR=/home/graffunder/.local/state/glaiss
Environment=GLAISS_REPO_DIR=/mnt/ai-workspace/glaiss
ExecStart=/home/graffunder/.venvs/glaiss/bin/python /mnt/ai-workspace/glaiss/glaiss.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## 5. Configuration spec

Extend `glaiss_config.json` with a new `waitroom` block. Existing `provider_*` keys remain for optional proxy mode.

```json
{
  "listen_port": 8899,
  "provider_scheme": "http",
  "provider_host": "127.0.0.1",
  "provider_port": 8080,
  "waitroom": {
    "enabled": true,
    "base_url": "http://127.0.0.1:8080",
    "llama_swap_url": "http://127.0.0.1:8090",
    "llama_swap_key_source": "llpc_config",
    "registry_command": "/home/graffunder/llama-swap/llpc_models.py",
    "state_poll_s": 1,
    "system_poll_s": 1,
    "service_poll_s": 5,
    "metrics_poll_s": 5,
    "history_samples": 1800,
    "proxy_mode": "optional",
    "admin_enabled": true,
    "admin_token_configured": false,
    "manual_control_file": "/home/graffunder/.llama-swap/manual-control.json",
    "model_lock_file": "/home/graffunder/.llama-swap/model-lock.json"
  }
}
```

Security rules:

- `GET /` and `GET /ws` may show read-only telemetry on the LAN.
- `POST /api/control/*` must be disabled unless an admin token is configured.
- If admin is enabled, control requests require:

```http
Authorization: Bearer <GLAISS_ADMIN_TOKEN>
```

- Do not store the llama-swap API key in GlAIss config. Read it from `~/llama-swap/config.yaml` at runtime via a helper.
- Admin token should live in `~/.local/state/glaiss/admin_token` mode `0600`, not in the repo.

## 6. Functional specification

### 6.1 Top bar

Required top-bar values:

| Label | Source | Notes |
|---|---|---|
| `loaded model` | `/api/state.loaded_model` | human catalog name |
| `loaded state` | llama-swap `/running` | `ready`, `loading`, `stopping`, `none` |
| `queue active` | waitroom `/state.active_model` | queue arbiter model |
| `inflight` | waitroom `/queue` | current forward count |
| `waiting` | waitroom `/queue` | number of waiting requests |
| `hysteresis` | waitroom `/state.hold_remaining_s` | countdown; label `idle` if `null` |
| `decode t/s` | traffic tap plus engine log parser | live and rolling average |
| `prefill t/s` | traffic tap, usage, engine log parser | live estimate and recent measured average |
| `CPU power` | RAPL delta | watts |
| `GPU power` | `nvidia-smi` / llama-swap metrics | sum of both GPUs |
| `System power` | `CPU power + GPU power` | label as estimated CPU package + GPU draw, not wall power |
| `RAM` | `/proc/meminfo` | used/total, percent |
| `VRAM` | GPU metrics | per-GPU total plus aggregate |
| `CPU util` | `/proc/stat` delta | aggregate percent |
| `GPU util` | GPU metrics | per-GPU percent |
| `uptime` | systemd collector | GlAIss and waitroom uptime |

Behavior:

- Hysteresis value should show as a countdown ring or bar.
- If CPU power is unavailable, show `CPU power: unavailable` rather than zero.
- If a collector is stale, mark the value yellow/red and include `collected_at`.

### 6.2 Waitroom queue visualization

The dashboard needs two queue visualizations.

#### Model-level waitroom queue

Visual lanes/chips for:

- current loaded model
- queue active model
- waiting models, arrival order
- queue quantum remaining
- handoffs
- last swap time
- hysteresis hold
- comfy claim state
- lock state

Example representation:

```text
LOADED: DeepSeek V4.1 Flash (ready)
ACTIVE: DeepSeek V4.1 Flash
INFLIGHT: 1
WAITING: Qwen 3.8 Flash Next Q4 · 378s
HYSTERESIS: 112s left
QUANTUM: 41s left
HANDOFFS: 22
LOCK: DeepSeek V4.1 Flash
```

Required colors:

- green: ready/active
- amber: waiting/loading/hysteresis active
- red: error/failed
- purple: locked
- blue: media engine
- gray: not loaded

#### Request-level queue from GlAIss

Preserve the existing GlAIss `#queue-strip` request chips.

For requests routed through GlAIss:

- `queued`
- `prefill`
- `decoding`
- `done`
- `cancelled`
- `error`

If waitroom request events are implemented, also show requests that reached `:8080` directly, even though prompt/response text may not be available for those requests.

### 6.3 Prompt and response panes

Preserve existing behavior:

- left pane: incoming prompts
- right pane: response stream
- request IDs
- prompt token estimate
- thinking/reasoning deltas when present
- real decode speed
- real prefill seconds
- output tokens

Add badges:

- `through GlAIss` for requests proxied by GlAIss
- `waitroom event` for events seen only from waitroom
- `direct :8080` for waitroom-only requests where prompt text is unavailable

Important truth: GlAIss can capture prompt/response text only when clients route through GlAIss. For clients still pointing directly at `:8080`, the system can still show queue and lifecycle events if waitroom emits events.

### 6.4 Service monitoring table

Show one row for each unit in Spec's journalctl command, using aliases where needed.

| Display row | Canonical source | Mapping |
|---|---|---|
| `strata31` | `strata31.service` | direct |
| `llama-qwen` | `llama-qwen.service` | direct |
| `comfy-ui` | `comfyui.service` | journal alias; show mapping `comfy-ui -> comfyui` |
| `llama-swap` | `llama-swap.service` | direct |
| `whisper-server` | `whisper-server.service` | direct |
| `whisper-shim` | `whisper-shim.service` | direct |
| `switcheroo-control` | `switcheroo-control.service` | direct |
| `model-waitroom` | `model-waitroom.service` | direct |
| `strata-dsv41.service` | `strata-dsv41.service` | direct |
| `llama-server` | synthetic row | not a systemd unit; map to llama.cpp child process under `llama-qwen.service` |

Per-service fields:

| Field | Source |
|---|---|
| state | `systemctl show -p ActiveState,SubState,LoadState` |
| unit file | `UnitFileState` |
| main PID | `MainPID` |
| control group | `ControlGroup` |
| uptime | `ExecMainStartTimestamp` |
| restart count | `NRestarts` where available |
| exit code/status | `ExecMainCode`, `ExecMainStatus`, `Result` |
| CPU percent | `/proc/<pid>/stat` delta across cgroup PIDs |
| RSS | `/proc/<pid>/smaps_rollup` or `/proc/<pid>/statm` fallback |
| swap | `/proc/<pid>/status` `VmSwap` where available |
| VRAM | `nvidia-smi --query-compute-apps=pid,process_name,used_memory` mapped to cgroup PIDs |
| GPU apps | per-process `nvidia-smi` mapping |
| error count | journal PRIORITY <= 3 in last 5 minutes |
| warning count | journal PRIORITY <= 4 in last 5 minutes |
| last log line | journal tail |
| port | registry or `ss` when active |
| model | reverse lookup from registry when unit is a model engine |

Synthetic `llama-server` row:

- State: active when a process named `llama-server` exists under `llama-qwen.service` cgroup; otherwise `inactive`.
- Label: `llama-server (process under llama-qwen)`.
- Logs: filter journal lines from `llama-qwen` with `COMM=llama-server`; if unavailable, show latest `llama-qwen` process logs.
- Important: do not fake a systemd state for `llama-server`.

Suggested extra rows, collapsible:

- `slot-watchdog` / `slot-watchdog.timer`
- `open-webui`
- `hermes-gateway`

These are not required by the original command, but they explain swap and UI behavior.

### 6.5 Journal stream drawer

Add a live log drawer equivalent to the Spec journalctl command.

Backend implementation:

- Spawn one `journalctl -f -o json` process with the watched units plus aliases:
  - `strata31`
  - `llama-qwen`
  - `comfyui`
  - `llama-swap`
  - `whisper-server`
  - `whisper-shim`
  - `switcheroo-control`
  - `model-waitroom`
  - `strata-dsv41`
- Parse JSON journal records.
- Bucket by `UNIT` first, then by `COMM` or `SYSLOG_IDENTIFIER`.
- Route `llama-server` to the synthetic row by `COMM=llama-server`.
- Maintain a bounded in-memory ring, default last 2000 lines.
- Push to WebSocket as `log` events.
- Frontend supports:
  - all logs
  - unit filter chips
  - level filter
  - pause
  - clear
  - download current visible buffer

### 6.6 System power spec

Target value: `CPU package power + GPU board power`.

GPU:

- Primary source: `nvidia-smi --query-gpu=index,name,power.draw,utilization.gpu,memory.used,memory.total,temperature.gpu,fan.speed --format=csv,noheader,nounits`.
- Secondary source: llama-swap `/metrics` Prometheus gauges if `nvidia-smi` fails.
- Aggregate:
  - `gpu_power_total_w = sum(gpu[].power_draw_w)`
  - `gpu_mem_used_total_mib = sum(gpu[].memory_used_mib)`

CPU package power:

- Primary source: RAPL energy counters sampled at 1 second:
  - package domain: `/sys/class/powercap/intel-rapl:0/energy_uj`
  - optional DRAM/PSYS domains if present under `/sys/class/powercap/intel-rapl:*`
- Counter computation:

```text
power_w = max(0, (energy_uj_now - energy_uj_prev) / 1_000_000 / elapsed_s)
```

- On llpc, `energy_uj` is root-readable; use `sudo -n cat` from the GlAIss service or a small helper.
- If RAPL is unavailable, do not substitute a fake value. Show `unavailable` and include the reason.

Total system power:

```text
system_power_estimate_w = cpu_package_power_w + gpu_power_total_w
```

Label it `estimated CPU package + GPU draw`, not wall-power.

Other system metrics:

- RAM from `/proc/meminfo`:
  - total
  - free
  - available
  - used = total - available
  - swap
- CPU utilization from `/proc/stat` delta:
  - aggregate and per-core available, aggregate shown prominently.
- Load average from `/proc/loadavg`.
- CPU frequency from `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` when available.

### 6.7 Throughput spec

Metrics needed:

- live decode t/s for the active request
- final decode t/s per request
- rolling average decode t/s per model
- live prefill estimate while waiting for first token
- final prefill t/s per request
- prefill t/s source tag

Sources, in priority order:

1. GlAIss streaming tap:
   - prefill seconds = request start to first streamed token
   - decode t/s = token count / generation time after first token
   - best quality because it matches what the client sees
2. Provider usage object:
   - if `usage.prompt_tokens` exists, use it for prefill denominator
3. Engine log parser:
   - Strata logs:
     - `[strata] reading the prompt: N of M tokens, T s so far`
     - `[strata] done: N tokens in T s (X tok/s)`
     - `[strata] thinking: ...`
     - `[strata] writing a tool call: ...`
   - llama.cpp logs:
     - `prompt eval time = ... ms / N tokens: ... ms per token, ... tokens per second`
     - `eval time = ... ms / N tokens: ... ms per token, ... tokens per second`
   - ComfyUI/Whisper:
     - no OpenAI token t/s; show request rate, job duration, queue state, and unit activity instead
4. GlAIss estimate:
   - `cl100k_base` prompt token estimate
   - rolling per-model PP calibration

UI display:

- `decode t/s` should show live while generating and final when done.
- `prefill t/s` should show:
  - `~1,240 t/s est` while waiting for first token
  - `1,380 t/s` once measured
  - source badge: `usage`, `engine-log`, or `estimate`

### 6.8 Manual load, unload, and lock semantics

Manual control is the highest-risk feature. It must be explicit, stateful, and protected by the same production constraints as the existing llpc model regime.

#### New control state files

Recommended new files:

```text
~/.llama-swap/model-lock.json
~/.llama-swap/manual-control.json
```

`model-lock.json` shape:

```json
{
  "version": 1,
  "model": "DeepSeek V4.1 Flash",
  "unit": "strata-dsv41",
  "kind": "text",
  "locked_at": 1795200000.0,
  "locked_by": "glaiss",
  "reason": "manual load and lock"
}
```

`manual-control.json` shape:

```json
{
  "version": 1,
  "mode": "manual",
  "suppress_backstops_until": 1795200900.0,
  "lock_model": "DeepSeek V4.1 Flash",
  "updated_at": 1795200000.0,
  "updated_by": "glaiss:lock"
}
```

Rules:

- Files must be written atomically: temp file plus `os.replace`.
- Waitroom, `slot-watchdog.py`, and any keep-warm path must read these files before taking automatic model action.
- `slot-watchdog.py` must stand down when:
  - a lock is active, or
  - `suppress_backstops_until` is in the future, or
  - control mode is manual and the user has not returned to auto.
- `current-model.sh` may be extended later to prefer `lock_model`, but the first implementation can enforce lock in waitroom/watchdog without changing `current-model.sh`.

#### Lock semantics

Lock means: only the locked catalog model may occupy the GPU slot.

If `model-lock.json` contains `A`:

- Requests for `A` are allowed through the normal queue.
- Requests for another model are rejected with:

```json
{
  "error": {
    "type": "model_locked",
    "code": "model_locked",
    "message": "DeepSeek V4.1 Flash is manually locked as loaded. Unlock before loading another model.",
    "locked_model": "DeepSeek V4.1 Flash"
  }
}
```

HTTP status:

- Use `409 Conflict` with `error.code = model_locked`.

Lock behavior while unloaded:

- Lock should normally be applied only after the target is loaded.
- Dashboard action `Load + lock` is permitted:
  1. start load job
  2. wait until `/running` reports target ready
  3. write lock file
- If lock exists but target is not loaded, waitroom should surface it as `lock_pending` in state and reject other models.

#### Manual load

For text models:

1. Validate catalog name through registry.
2. If another model is locked, reject with `409 model_locked`.
3. Set `manual-control.json.mode = "manual"` and suppress backstops for a short window, recommended 900 seconds during the job.
4. Run `~/llama-swap/swap-to.sh <model>` as a background job.
5. Do not pass `--force` unless the operator explicitly confirms `force_kill_generation=true`.
6. Track job state:
   - `queued`
   - `running`
   - `succeeded`
   - `failed`
7. Poll `/api/state` until `/running` reports target ready.

For media model `Comfy Studio`:

- Do not send a fake chat completion to Comfy.
- Use llama-swap `/upstream/Comfy Studio/system_stats` as a load/health trigger only if no conflicting text lock is active.
- Show media loads in the same job table but mark them `media`.

#### Manual unload

1. Validate catalog name.
2. Reject if waitroom `/queue.inflight > 0` unless the operator explicitly confirms `force=true`.
3. If another model is locked and target is not locked model, reject.
4. If target is locked, require explicit `break_lock=true` and remove lock as part of the action.
5. Call llama-swap:

```http
POST /api/models/unload/<URL-encoded catalog model name>
Authorization: Bearer <llama-swap key read from llpc config>
```

6. Do not call `POST /api/models/pause`.
7. Do not call `systemctl stop` directly.
8. Write `manual-control.json.mode = "manual"` with `suppress_backstops_until = now + 900` so `slot-watchdog` does not immediately re-warm the unloaded model.
9. Poll `/running` until target is absent.
10. Emit a control event with old state, requested action, result, and duration.

#### Unlock / return to auto

Actions:

- `Unlock`: remove `model-lock.json`, leave control mode manual until `suppress_backstops_until` expires or auto is selected.
- `Resume auto`: clear lock and mode, clear suppression.

### 6.9 Alerts

Optional but recommended:

- engine unit entered failed state
- waitroom became unreachable
- llama-swap became unreachable
- loaded model state stayed `loading` longer than configured timeout
- inflight request older than configurable threshold
- GPU temp over warning threshold
- GPU power spike over threshold
- RAM free under threshold
- lock active for longer than configured threshold

Alerts should be visible in dashboard only in phase 1; no external notification yet.

## 7. API contract for GlAIss

### Existing endpoints to preserve

- `GET /` serves the dashboard.
- `GET /ws` WebSocket stream.
- Existing proxy routes remain available for optional GlAIss proxy mode.

### New endpoints

#### `GET /api/health`

Returns collector health:

```json
{
  "ok": true,
  "time": 1795200000.0,
  "collectors": {
    "waitroom": {"ok": true, "age_s": 0.4},
    "llama_swap": {"ok": true, "age_s": 1.2},
    "system": {"ok": true, "age_s": 0.2},
    "services": {"ok": true, "age_s": 2.1},
    "journal": {"ok": true, "age_s": 0.1}
  }
}
```

#### `GET /api/state`

Full combined snapshot:

```json
{
  "time": 1795200000.0,
  "model": {
    "loaded_model": "DeepSeek V4.1 Flash",
    "loaded_unit": "strata-dsv41",
    "loaded_state": "ready",
    "active_model": "DeepSeek V4.1 Flash",
    "inflight": 1,
    "waiting": [{"model": "Qwen 3.8 Flash Next Q4", "waited_s": 378.8}],
    "handoffs": 22,
    "quantum_left_s": 41.0,
    "hold_remaining_s": 112.0,
    "swap_hold_s": 180.0,
    "last_swap_unix": 1795199888.0,
    "last_swap_age_s": 112.0,
    "intent": {
      "model": "Qwen 3.8 Flash Next Q4",
      "stale_label": null,
      "at": 1795188011.0,
      "ttl_s": 43200
    },
    "last_model": "DeepSeek V4.1 Flash",
    "desired_unit": "strata-dsv41",
    "comfy_claimed": false,
    "lock": null,
    "manual_control": null
  },
  "models": [
    {
      "name": "Qwen 3.8 Flash Next Q4",
      "unit": "strata31",
      "port": 8085,
      "kind": "text",
      "loaded": false,
      "locked": false,
      "visible_in_picker": true
    }
  ],
  "system": {
    "cpu_power_w": 95.2,
    "gpu_power_w": 245.7,
    "system_power_w": 340.9,
    "cpu_util_percent": 18.4,
    "load_1": 4.2,
    "ram_total_bytes": 134000000000,
    "ram_used_bytes": 90000000000,
    "ram_available_bytes": 44000000000,
    "swap_used_bytes": 1200000000,
    "gpus": []
  },
  "services": [],
  "traffic": {
    "requests_total": 12,
    "requests_inflight": 1,
    "tokens_generated": 1234,
    "avg_decode_tps": 21.2,
    "avg_prefill_tps": 1220.0,
    "recent_requests": []
  }
}
```

#### `GET /api/services`

Returns just the service table rows.

#### `GET /api/logs?unit=strata31&limit=500`

Returns recent journal lines from the in-memory ring.

#### `GET /api/history?metric=power&seconds=900`

Optional in phase 1. Returns time-series samples for charts.

#### Control endpoints

All control endpoints require admin token.

```text
POST /api/control/load
POST /api/control/unload
POST /api/control/lock
POST /api/control/unlock
POST /api/control/resume_auto
GET  /api/control/jobs
GET  /api/control/jobs/{job_id}
```

Examples:

```json
POST /api/control/load
{"model": "DeepSeek V4.1 Flash", "lock_after_load": true}
```

```json
POST /api/control/unload
{"model": "DeepSeek V4.1 Flash", "break_lock": false, "force": false}
```

```json
POST /api/control/lock
{"model": "DeepSeek V4.1 Flash", "load_if_needed": true}
```

Response:

```json
{
  "job_id": "ctrl_20261120T103000Z_a1b2",
  "status": "queued",
  "action": "lock",
  "model": "DeepSeek V4.1 Flash",
  "created_at": 1795200000.0
}
```

## 8. Waitroom-side additions required

Minimal waitroom additions make the design robust and preserve production constraints.

### 8.1 Add `GET /state`

Expose the state needed by GlAIss:

```json
{
  "loaded_model": "DeepSeek V4.1 Flash",
  "loaded_unit": "strata-dsv41",
  "loaded_raw": ["DeepSeek V4.1 Flash"],
  "active_model": "DeepSeek V4.1 Flash",
  "inflight": 1,
  "quantum_left_s": 41.0,
  "handoffs": 22,
  "waiting": [{"model": "Qwen 3.8 Flash Next Q4", "waited_s": 378.8}],
  "swap_hold_s": 180.0,
  "hold_remaining_s": 112.0,
  "last_swap_unix": 1795199888.0,
  "intent": null,
  "last_model": "DeepSeek V4.1 Flash",
  "desired_unit": "strata-dsv41",
  "comfy_claimed": false,
  "lock": null,
  "manual_control": null
}
```

Implementation notes:

- Use `SwapState.hold_remaining()` already present in `waitroom.py`.
- Use `Arbiter.snapshot()` already present in `waitroom.py`.
- Add wall-clock `last_swap_unix` from `SwapState.last_swap`.
- Add `desired_unit` from `~/.llama-swap/swap-desired`.
- Add `lock` and `manual_control` from the new state files.

### 8.2 Add lock enforcement in waitroom

In request decision path, before admitting a different model:

1. Read `~/.llama-swap/model-lock.json` with cache TTL <= 1 second.
2. Resolve request model to catalog model.
3. If lock is active and resolved model is not the locked model:
   - return `409 model_locked`
   - include locked model in body
4. If comfy claim is active, preserve existing comfy wait behavior. Lock enforcement should sit before or after comfy claim, but it must not let text requests evict a claimed media cycle.
5. If the request does not name a model and the resolved default conflicts with the lock, route to the locked model if possible; otherwise return `409 model_locked`.

### 8.3 Optional: emit waitroom request events

To visualize direct `:8080` clients without moving clients through GlAIss, add optional event emission.

Environment:

```text
WAITROOM_OBSERVE_URL=http://127.0.0.1:8899/internal/waitroom/events
```

Fire-and-forget POSTs:

```json
{
  "type": "request_queued",
  "req_id": "wr_ab12cd",
  "model": "Qwen 3.8 Flash Next Q4",
  "client": "192.168.4.164",
  "at": 1795200000.0
}
```

Event types:

- `request_queued`
- `request_admitted`
- `forward_started`
- `first_byte`
- `request_finished`
- `request_cancelled`
- `swap_requested`
- `swap_hold_parked`

Payload may include duration, status, target model, and whether the request was direct or proxied. It must not include prompt or response text.

### 8.4 Backstop suppression files

Update helpers to respect:

- `~/.llama-swap/model-lock.json`
- `~/.llama-swap/manual-control.json`

Required helper behavior:

- `slot-watchdog.py`: stand down when lock exists or manual suppression is active.
- Any future keep-warm timer: same.
- `current-model.sh`: may prefer locked model, but that is optional for phase 1 because waitroom rejection is enough.

## 9. Frontend implementation plan

Files:

- `index.html` remains single-file dashboard in phase 1.
- Move CSS into a `<style>` block as is.
- Add tabs to avoid crowding:
  - `Queue`
  - `Services`
  - `System`
  - `Models`
  - `Logs`
  - `Settings`

Keep the current prompt/response panes visible on `Queue`.

Recommended layout:

```text
+--------------------------------------------------------------+
| Top bar: loaded | queue | hysteresis | t/s | power | RAM     |
+---------------------------+----------------------------------+
| Model queue               | Request queue chips              |
+---------------------------+----------------------------------+
| Incoming prompts          | Response stream                  |
+---------------------------+----------------------------------+
| Service table             | Charts / engine log              |
+--------------------------------------------------------------+
```

The current GlAIss visual language should be retained: dark panels, request chips, prompt cards, response stream, compact top-bar stats.

Required controls:

- Pause auto-refresh.
- Filter logs by unit and level.
- Show raw JSON state for debugging.
- Model cards with `Load`, `Unload`, `Lock`, `Unlock`, `Resume auto`.
- Confirmation dialog for `Unload`, `Break lock`, and `Force`.
- Inline job progress panel for control actions.

## 10. Backend module plan

Keep `glaiss.py` working, but refactor into modules to avoid a 5,000-line single file.

Suggested repo layout:

```text
glaiss/
  glaiss.py
  glaiss/
    __init__.py
    app.py
    config.py
    hub.py
    proxy.py
    control.py
    collectors/
      __init__.py
      waitroom.py
      llamaswap.py
      registry.py
      services.py
      journal.py
      hardware.py
      engine_logs.py
  static/
    index.html
  tests/
    test_config.py
    test_waitroom_collector.py
    test_llamaswap_collector.py
    test_services.py
    test_journal.py
    test_hardware.py
    test_engine_logs.py
    test_control.py
    test_api.py
    test_ws.py
```

Migration rule:

- Phase 1 may add modules while `glaiss.py` remains the entry point.
- Preserve existing behavior before moving code.

## 11. Data collection details

### Waitroom collector

Poll every 1 second:

- `GET http://127.0.0.1:8080/state`
- fallback: `GET http://127.0.0.1:8080/queue`
- fallback: `GET http://127.0.0.1:8080/v1/models`

Handle fail-open: if waitroom is down, cache last state for up to 30 seconds and mark stale.

### llama-swap collector

Poll every 2-5 seconds:

- `GET /running` with auth
- `GET /metrics` with auth

Parse Prometheus lines into typed metrics. Use `nvidia-smi` as primary for GPU because it gives current values independent of llama-swap implementation.

### Registry collector

Run every 10 seconds:

```bash
/home/graffunder/llama-swap/llpc_models.py
```

Current command already prints JSON with all catalog models, including `unlisted`. Map fields to dashboard:

- `name`
- `aliases`
- `unit`
- `port`
- `text`
- `unlisted`
- `context`
- `siblings`

### System collector

Run every 1 second:

- `/proc/stat`
- `/proc/meminfo`
- `/proc/loadavg`
- `nvidia-smi`
- RAPL delta

Use async subprocess and small timeouts:

- `nvidia-smi`: 2 seconds
- `sudo -n cat`: 0.5 seconds per sample
- registry command: 5 seconds

Never block the event loop; run subprocess collectors in a background task and store the last successful result.

### Service collector

Run every 5 seconds:

```bash
systemctl show <unit> \
  -p Id -p LoadState -p ActiveState -p SubState -p UnitFileState \
  -p MainPID -p ControlGroup -p ExecMainStartTimestamp \
  -p ExecMainExitTimestamp -p ExecMainCode -p ExecMainStatus \
  -p Result -p NRestarts
```

Then map `ControlGroup` to PIDs and sum per-process CPU/RSS.

For `llama-server` synthetic row:

- scan `/proc/*/comm` for `llama-server`
- validate PID belongs to `llama-qwen` cgroup
- if found, state `process-active`; otherwise `inactive`

### Journal collector

Spawn one process:

```bash
journalctl -f -o json \
  -u strata31 \
  -u llama-qwen \
  -u comfyui \
  -u llama-swap \
  -u whisper-server \
  -u whisper-shim \
  -u switcheroo-control \
  -u model-waitroom \
  -u strata-dsv41
```

If journalctl supports it, use `--rotate-offset` or startup snapshot logic carefully; otherwise start streaming and mark history unavailable after a GlAIss restart.

## 12. Implementation plan for another model

### Phase 0: safe read-only foundation

Goal: add dashboard without touching running model state.

Tasks:

1. Add `GLAISS_STATE_DIR` and move config/runtime state out of repo if missing.
2. Add `/api/health`.
3. Add registry collector using `llpc_models.py`.
4. Add waitroom collector reading `/queue` and `/v1/models`.
5. Add system collector:
   - `/proc/stat`
   - `/proc/meminfo`
   - `nvidia-smi`
   - RAPL via `sudo -n`
6. Add llama-swap collector:
   - `/running`
   - `/metrics`
7. Extend WebSocket snapshot protocol.
8. Update top bar.
9. Add unit tests for parsing only; no control actions.

Acceptance:

- Dashboard opens on `:8899`.
- It shows loaded model, queue, inflight, waiting, CPU, RAM, GPU power, VRAM.
- No action endpoint changes model state.

### Phase 1: waitroom state endpoint

Goal: expose hysteresis and control state cleanly.

Tasks:

1. Patch `~/llama-swap/waitroom.py` to add `GET /state`.
2. Return queue snapshot plus:
   - `hold_remaining_s`
   - `swap_hold_s`
   - `last_swap_unix`
   - `intent`
   - `last_model`
   - `desired_unit`
   - `comfy_claimed`
   - placeholder `lock` and `manual_control`
3. Add waitroom tests:
   - `/state` shape
   - hysteresis remaining decays
   - last-swap survives restart if file present
   - comfy claim appears when `swap-desired` is `comfyui`

Acceptance:

- GlAIss displays hysteresis countdown.
- Waitroom restart keeps last-swap behavior.

### Phase 2: service table and logs

Goal: replace journalctl monitoring command.

Tasks:

1. Add `services.py`.
2. Add `journal.py`.
3. Add alias mapping:
   - `comfy-ui` -> `comfyui`
   - `llama-server` -> synthetic llama.cpp process row
4. Add service UI table.
5. Add logs drawer.
6. Add per-service ring buffers and counts.

Acceptance:

- Each row in Spec's service list appears.
- Live logs stream into UI.
- Service restarts update uptime/restart state.

### Phase 3: queue and request visualization

Goal: preserve old GlAIss visual queue while adding waitroom queue.

Tasks:

1. Keep existing request queue chips.
2. Add model-level waitroom lane.
3. Add optional waitroom event receiver:
   - `POST /internal/waitroom/events`
   - loopback-only or admin token required
4. Add waitroom event emitter to waitroom if approved.
5. Merge direct-waitroom events into request timeline.

Acceptance:

- Requests routed through GlAIss show prompt/response as before.
- Requests routed directly to `:8080` at least show queue lifecycle if waitroom events are enabled.

### Phase 4: throughput enrichment

Goal: accurate prefill/decode metrics.

Tasks:

1. Add engine log parser for Strata and llama.cpp lines.
2. Add source badges: proxy, usage, engine-log, estimate.
3. Store recent metrics by model.
4. Add charts:
   - decode t/s
   - prefill t/s
   - CPU/RAM/GPU power

Acceptance:

- A request with a prompt through GlAIss shows prefill estimate then final measured prefill.
- A streaming request shows live decode.
- Engine log parser tests cover fixture logs.

### Phase 5: manual controls

Goal: load, unload, lock, unlock, resume auto.

Tasks:

1. Implement admin auth.
2. Add `manual-control.json` and `model-lock.json` helper module.
3. Add control job runner.
4. Implement text load via `swap-to.sh`.
5. Implement media load via llama-swap `/upstream/.../system_stats`.
6. Implement unload via llama-swap `/api/models/unload/{model}`.
7. Add lock enforcement to waitroom.
8. Update slot watchdog to respect manual/lock files.
9. Add control UI cards and confirmations.
10. Add full mock integration tests.

Acceptance:

- `Load + lock` works in mock suite.
- Lock rejects other model loads with `model_locked`.
- Unload is blocked when inflight exists unless forced.
- Manual unload suppresses backstops so watchdog does not instantly re-warm.
- Resume auto restores normal behavior.

### Phase 6: optional client cutover

Goal: optional full prompt/response visibility.

Do this only as an explicit production decision.

Tasks:

1. Point one test client to GlAIss `:8899`.
2. Verify auth headers pass through.
3. Verify streaming works.
4. Verify queue events are not duplicated.
5. Decide whether:
   - keep clients on `:8080` and accept no prompt text for direct clients, or
   - move selected clients to `:8899`, or
   - build a future waitroom tap that redacts prompt text.

Recommended default: keep production clients on `:8080`; move only selected clients to GlAIss for debugging/observability.

## 13. Test plan

### 13.1 Unit tests

#### Config tests

- Existing GlAIss config loads without breaking.
- `GLAISS_STATE_DIR` is honored.
- Missing admin token disables control endpoints.
- Repo path state does not leak secrets into config.

#### Waitroom collector tests

Use fake aiohttp waitroom:

- `/state` parses correctly.
- fallback to `/queue` when `/state` absent.
- stale cache behavior.
- comfy claim parses.
- lock file parses.
- hysteresis countdown computes correctly.

#### Llama-swap collector tests

Use fake FastAPI:

- `/running` loaded/ready/loading/stopping.
- `/metrics` parses Prometheus metrics.
- auth failure marks collector unavailable.

#### Hardware collector tests

Fixture files:

- `/proc/stat` delta gives expected CPU percent.
- `/proc/meminfo` parses total/available.
- `nvidia-smi` CSV fixture parses two GPUs.
- RAPL delta fixture produces expected watts.
- RAPL permission-denied path returns unavailable, not zero.

#### Service collector tests

Fake `systemctl show` output:

- unit active/failed/inactive.
- MainPID parsing.
- missing unit handling.
- `comfy-ui -> comfyui` alias.
- `llama-server` synthetic row from fixture `/proc/<pid>/comm`.

#### Journal collector tests

Fixture JSON journal records:

- routes `UNIT=comfyui` to display row `comfy-ui`.
- routes `COMM=llama-server` to synthetic row.
- counts error/warning priorities.
- ring buffer caps memory.

#### Engine log parser tests

Fixture logs:

- Strata prompt reading lines.
- Strata done line.
- llama.cpp prompt eval and eval time.
- malformed lines do not crash parser.

#### Control tests

Use fake waitroom/llama-swap:

- lock file writes atomically.
- lock rejects other model with `409`.
- unlock clears file.
- load job runs fake `swap-to.sh` only when no conflicting lock.
- unload blocked when inflight > 0.
- unload calls fake `/api/models/unload/{model}`.
- unload breaks lock only when `break_lock=true`.
- manual unload writes `manual-control.json` and suppresses watchdog.
- resume auto clears state.

### 13.2 Integration tests

#### Mock full stack

Create ephemeral mock services:

- mock waitroom on random port
- mock llama-swap on random port
- mock engine stream on random port

Validate:

- GlAIss `/api/state` combines waitroom, llama-swap, registry, hardware fixtures.
- GlAIss proxy forwards chat request to mock waitroom.
- WebSocket emits stats and request events.
- Direct waitroom event receiver merges lifecycle events.

#### Waitroom contract tests

In llpc-side `test_waitroom.py`:

- `GET /state` schema.
- lock enforcement returns 409 with correct JSON code.
- comfy claim path still waits.
- normal model request path unchanged.
- hysteresis unchanged.
- cancel override unchanged.

#### Slot watchdog tests

In llpc-side tests:

- watchdog exits early when `model-lock.json` exists.
- watchdog exits early when `manual-control.json.suppress_backstops_until` is future.
- watchdog normal behavior when no manual/lock file.

### 13.3 UI tests

Use Playwright or equivalent.

Required cases:

- Dashboard loads offline fixture data and shows top bar.
- Hysteresis countdown decreases.
- Service table includes all rows from Spec's journalctl command.
- Logs filter by unit works.
- Model control buttons are hidden/disabled when admin token absent.
- Load confirmation shows target model.
- Unload requires confirmation when inflight exists.
- Lock badge appears on active model card and queue lane.
- Chart renders power and decode history.

### 13.4 Security tests

- `POST /api/control/*` returns 403 when no admin token.
- `POST /api/control/*` returns 401/403 with wrong token.
- `GET /api/state` does not expose API keys or environment values.
- Admin token is not written to repo config.
- Waitroom event receiver rejects external IPs or token missing if enabled.

### 13.5 Manual production verification checklist

Do this read-only in production first.

1. Start GlAIss service on a non-production branch/build.
2. Confirm `:8899` dashboard is up.
3. Confirm loaded model matches `journalctl` and `:8090/running`.
4. Confirm hysteresis countdown changes when a real swap is triggered by normal use.
5. Confirm CPU/GPU/RAM values are plausible.
6. Confirm service table matches `systemctl` state.
7. Confirm live logs contain the same recent lines as Spec's `journalctl -f` command.
8. Confirm no control endpoints are enabled yet.

Only in a maintenance window, with Spec explicitly available:

9. Enable admin token.
10. Load a non-disruptive target during idle/no live clients.
11. Lock it.
12. Verify another model request is rejected with `model_locked`.
13. Unlock and resume auto.
14. Verify normal named-model switching works.

Never use production as the first test of unload/lock behavior.

## 14. Rollout plan

1. Merge code and docs into GlAIss repo.
2. Install local venv on llpc.
3. Run read-only instance manually on `:8899`.
4. Install `glaiss.service` but keep control endpoints disabled.
5. Ship service/log/queue dashboard first.
6. Add waitroom `/state`.
7. Add slot-watchdog manual suppression support.
8. Add manual controls behind admin token.
9. Update llpc `~/README.md` section `Model layer` to document:
   - GlAIss dashboard port
   - admin control behavior
   - lock/manual-control files
   - production rule that controls are operator-initiated only
10. Update ai-workspace GlAIss README with operational commands.

## 15. Rollback plan

Read-only rollback:

- Stop `glaiss.service`.
- Unmap/stop listener `:8899`.
- Existing waitroom `:8080` remains untouched.

Waitroom `/state` rollback:

- Remove `/state` route.
- GlAIss falls back to `/queue`.

Lock/manual rollback:

- Delete `~/.llama-swap/model-lock.json`.
- Delete `~/.llama-swap/manual-control.json`.
- Remove waitroom lock check.
- Revert slot-watchdog manual-file checks.

Proxy client rollback:

- Move any client pointed at `:8899` back to `:8080`.

## 16. Recommended defaults

| Setting | Default |
|---|---:|
| Dashboard port | `8899` |
| Dashboard bind | `0.0.0.0` on LAN, no tunnel |
| Waitroom poll | `1s` |
| System poll | `1s` |
| Service poll | `5s` |
| Journal ring | `2000` lines |
| Metrics history | `1800` samples |
| Admin control | disabled until token present |
| Manual unload backstop suppression | `900s` |
| Lock behavior | reject other models with `409 model_locked` |
| ComfyUI display name | `comfy-ui -> comfyui.service` |
| llama-server display | synthetic process row, not fake unit |

## 17. Open decisions for the implementing agent

These should not block implementation.

1. Whether waitroom event emission is implemented in phase 3 or deferred.
   - Recommended: implement it, because it gives queue visibility for direct `:8080` clients without moving production clients behind GlAIss.
2. Whether `current-model.sh` should prefer lock model.
   - Recommended: not in phase 1; lock enforcement in waitroom/watchdog is enough.
3. Whether ComfyUI manual load should be exposed in the first control release.
   - Recommended: show it but mark it media-only and require explicit confirmation, because media jobs can interfere with text lock semantics.
4. Whether charts are client-side or server-side.
   - Recommended: server ring for reload history, client charting from WebSocket snapshots.

## 18. Definition of done

The feature is done when:

1. The GlAIss dashboard shows all Spec-required state without requiring `journalctl`.
2. Every service in the journalctl command has a visible row or explicitly labeled synthetic process row.
3. The old GlAIss queue visualizations still work for requests routed through GlAIss.
4. The waitroom queue is visualized with loaded model, waiting models, inflight, handoffs, and hysteresis.
5. Power monitoring shows CPU package power, GPU power, and total estimated system power.
6. Prefill and decode t/s are shown with source tags.
7. Admin controls allow load, unload, lock, unlock, and resume auto safely.
8. Lock prevents other model loads at the policy layer.
9. Tests pass for read-only collectors, waitroom `/state`, lock enforcement, and mock control jobs.
10. Production manual control verification is performed only in a maintenance window or mock suite.
