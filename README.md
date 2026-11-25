# GlAIss

GlAIss is the llpc inference operations console: a read-only window onto the
running stack plus manual model control. **It is not in the request path.** The
repo contains exactly one proxy — the llpc waitroom door (`llpc/waitroom.py`,
`:8080`) — and every client talks to that door directly.

The original GlAIss transparent proxy that used to sit in front of the door was
removed on 2026-11-21 (Spec's call): two proxies in one repo meant two routing
brains, two sets of token stats, and a console whose traffic panels were empty
because real clients never went through it. `test_single_proxy.py` keeps it gone.

The web UI has **no login, MFA, or authorization layer by design**. It is an
admin-only operations surface and is expected to be reachable only through the
trusted VPN network (in practice: Cloudflare Access on
`glaiss.nuclearquads.com`).

## What it shows

- llpc waitroom state: loaded model, what is generating right now, one queue box
  per in-flight request (each with a prompt-size stat: the engine's measured
  total / cached / to-process numbers while it runs, or a ~chars/4 estimate while
  it waits), ordered waiting models, swap hysteresis, Comfy claim, intent, manual
  lock, manual-control suppression, and which models are switched off for
  on-demand loading. Door internals (quantum timer, handoffs) stay in `/state`
  for the API but are off the page.
- **On-demand model switch (the Models tab checkbox).** Off means the door will
  not load that model by itself: a request naming it parks in the queue until
  you switch it back on or load it by hand, while requests behind it for an
  enabled model are admitted ahead of it. A model already generating finishes
  its stream; nothing is unloaded and no unit is touched. CLI equivalent:
  `model-control.py on-demand "<model>" --off`.
- **DeepSeek V4.1 Flash concurrency:** the door admits one in-flight DeepSeek
  request at a time, matching the engine's `concurrency.serving: 1`; a second
  request waits as its own queue entry. Other models keep the arbiter's batch
  behavior.
- Token speed measured **by the engine** (`GET /metrics` on the loaded model's
  port): decode t/s, prefill recompute rate, KV-cache reuse, engine phase. The
  console cannot measure traffic it does not carry, so there is no proxy-side
  estimate any more. Two engine schemas exist and both are parsed:
  Strata reports per-request `decode_tok_s` plus `prompt_read`/`prompt_ms`;
  the DS4 engine (DeepSeek V4.1) reports `prompt_total` + `reused` and a null
  `hit_rate`, so prefill is `(prompt_total - reused) / prompt_ms` and KV reuse is
  `reused / prompt_total`. A live `tok_s` from the engine's `live` block wins over
  the last finished request's average while it is generating. The Metrics chart
  plots decode and prefill on the same canvas: decode axis and label on the left,
  prefill axis and label on the right, each with its own scale. The tiles keep the
  most recent prefill/decode value (history fallback) when a metric fetch is
  momentarily empty instead of blanking.
- Per-service rows matching the standard llpc journal command:

  `journalctl -u strata31 -u llama-qwen -u comfy-ui -u llama-swap -u whisper-server -u whisper-shim -u model-waitroom -u strata-dsv41.service -u llama-server`

  (Only the AI services. `switcheroo-control`, the NVIDIA PRIME switch proxy,
  was dropped on 2026-11-21 — it is not a model service and never logged
  anything worth watching.)

- Service CPU, RSS, VRAM, PID, state, restart count, recent log count, last log,
  and a live journal console for the same units.
- CPU package power, total GPU power, combined system power, RAM, VRAM, CPU and
  GPU utilization, decode t/s, and prefill t/s history.
- **Per-GPU cards** from the engine's `hardware.gpus` block (utilization, memory,
  temperature, power) with `nvidia-smi` as fallback when an engine reports no
  hardware block; PCIe rx/tx are aggregate totals from the engine.
- **Comfy workload** from `GET /system_stats` and `GET /queue` on `:8188`, read
  only while Comfy owns the slot — running/pending counts, prompt id, device VRAM
  and free RAM; otherwise the panel says `comfy not loaded`.
- **STT activity** from the `whisper-server`/`whisper-shim` service rows (state,
  CPU, RSS, VRAM) because Whisper exposes no request/throughput endpoint.
- **Door traffic counters** (admissions, handoffs, cancellations, timeouts,
  median queue wait, per-model use) fed by the door's event POST; the console
  never sees client traffic itself.
- Registry-backed model catalog and manual model controls.

## Architecture

```text
clients ──────────────────────────────> llpc waitroom :8080 -> llama-swap :8090 -> engine
                                              │  (the only proxy in the repo)
                                              │ events (/internal/waitroom/events)
llpc state files, systemd, nvidia-smi, RAPL, Comfy `/system_stats` + `/queue`,
engine `/metrics`
                                              │
                        GlAIss console :8899 (observe + manual control only)
```

The console answers `/`, `/ws`, `/api/*` and `/internal/waitroom/events`, and
**404s everything else** — there is no catch-all, no provider host/port, and no
route that accepts inference traffic. `WAITROOM_URL` defaults to
`http://127.0.0.1:8080`; llama-swap at `LLAMA_SWAP_URL` is used read-only for
running state and for the manual unload call.

```text
glaiss.py                      runtime entrypoint
glaiss_app/console.py          static page, websocket fan-out (no forwarding)
glaiss_app/monitor.py          monitor/control router and model-control jobs
glaiss_app/collectors.py       waitroom, systemd, GPU, RAPL, RAM, engine-metrics collectors
glaiss_app/journal.py          journalctl follow ring
glaiss_app/model_state.py      shared manual-lock/manual-control file format
glaiss_app/static/index.html   operations console
llpc/waitroom.py               THE PROXY: llpc source of truth, with GlAIss event feed
llpc/model_control.py          CLI with the same lock/inflight guards
systemd/                       canonical unit files
scripts/install.sh             local deployment helper
```

## Manual model controls

Dashboard actions and CLI equivalents:

```bash
llpc/model_control.py list
llpc/model_control.py load 'DeepSeek V4.1 Flash'
llpc/model_control.py load --lock 'Qwen 3.8 Flash Next Q4'
llpc/model_control.py lock 'Qwen 3.8 Flash Next Q4'
llpc/model_control.py unload 'Qwen 3.8 27B Q6'
llpc/model_control.py unlock
llpc/model_control.py resume-auto
```

UI note (2026-11-22): the console no longer exposes AUTO / Resume auto / NO LOCK /
Unlock or Load + lock. Model availability is controlled by the per-model On demand
checkbox; the CLI lock/unlock/resume-auto routes remain for scripted or manual use.

Safety behavior:

- Text loads use `swap-to.sh`; direct unit start/stop is not used.
- A manual lock rejects requests and control actions for other models with
  HTTP 409 / `model_locked`.
- Unload refuses while waitroom reports a request in flight unless
  `force`/`--force` is explicitly used.
- Unload refuses to break a manual lock unless `break_lock`/`--break-lock` is
  explicitly used.
- Load/unload/lock are single-flight jobs. The UI disables controls while a job
  is queued or running and refreshes state after completion.
- The server only performs model transitions the CLI was always allowed to
  perform. First boot of a disabled engine is still a human decision.

## Deployment

From this repository:

```bash
./scripts/install.sh all
```

Use `scripts/install.sh web` for only GlAIss. Use `--no-restart` to stage
without touching the active waitroom. The installer syncs to `/opt/glaiss`,
creates `/var/lib/glaiss`, installs its virtualenv, installs the canonical units,
and restarts only `glaiss.service` and/or `model-waitroom.service`. It never
starts or stops model engines.

Development run:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
GLAISS_STATE_DIR=.venv/glaiss-state .venv/bin/python glaiss.py
```

Open **https://glaiss.nuclearquads.com/** — the canonical entry point. It is
served from llpc `:8899` through the UnCube `uncube` cloudflared tunnel
(ingress rule `glaiss.nuclearquads.com → http://192.168.4.164:8899`) and sits
behind the account's catch-all Cloudflare Access app (Google login): the UI
itself has no auth by design, and Zero Trust is the only gate. Direct LAN/Tailscale
access to `http://<llpc-address>:8899/` still works for local debugging.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `GLAISS_STATE_DIR` | repo directory | writable config directory |
| `GLAISS_LISTEN_PORT` | config/8899 | listen port override |
| `GLAISS_ENGINE_METRICS_TIMEOUT` | `4` | seconds for the engine `/metrics` fetch (token speed) |
| `WAITROOM_URL` | `http://127.0.0.1:8080` | monitor source (the door; not a forwarding target) |
| `LLAMA_SWAP_URL` | `http://127.0.0.1:8090` | running state and unload API |
| `LLPC_MODELS_PY` | `~/llama-swap/llpc_models.py` | model registry |
| `GLAISS_MONITOR_POLL_S` | `2` | collector interval |
| `GLAISS_DISABLE_MONITOR_TASKS` | off | disable background collectors for tests |

## API

The console's entire HTTP surface — anything else is a 404:

- `GET /` — console page
- `GET /ws` — websocket: `monitor`, `control_job`, `waitroom_event` pushes
- `GET /api/health`
- `GET /api/state` — the whole snapshot (model, models, system, services, engine, control, config, comfy, stt, door_traffic)
- `GET /api/history?seconds=900`
- `GET /api/logs?limit=300&unit=strata31&level=error`
- `POST /internal/waitroom/events` (waitroom fire-and-forget feed)
- `POST /api/control/load`
- `POST /api/control/unload`
- `POST /api/control/lock`
- `POST /api/control/unlock`
- `POST /api/control/resume_auto`
- `POST /api/control/on_demand` — `{"model": "...", "enabled": true|false}`;
  parks a model out of the on-demand path (no unit is loaded, unloaded or locked)
- `POST /api/control/schedule` — `{"model": "...", "start": "20:00", "end": "06:00"}`;
  sets a model's daily enabled window (overnight windows allowed, local time).
  Outside the window the model behaves like an off-demand model: a request
  naming it parks in the door queue and runs on its own when the window opens.
  `{"model": "..."}` with no start/end clears the window

`/api/state` is the single read endpoint the page uses; the old slice aliases
(`/api/model`, `/api/services`, `/api/system`, `/api/control/jobs`) and the
`/ws/logs` socket were removed on 2026-11-21 as dead — every one of them was a
subset of `/api/state`, and nothing called them. If you ever need inference
traffic, that is the door's job on `:8080`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests cover the single-proxy rule and the exact HTTP surface
(`test_single_proxy.py`), the console page's DOM/JS contract including "the page
calls no route that doesn't exist" (`test_ui_contract.py`), the ban on
unreferenced top-level names (`test_no_dead_code.py`), engine token-speed
parsing (`test_engine_metrics.py`), per-GPU card/nvidia-smi fallback, Comfy queue
parsing and door-traffic aggregation (`test_glaiss_stats.py`), monitor routes and
waitroom event fan-out, hysteresis/lock state, manual-control guards, the removed
UI controls and the dual-axis token chart, and the DeepSeek one-in-flight queue
rule. The waitroom's own 119-check suite runs standalone: `python3 llpc/test_waitroom.py`.
No test starts, loads, or stops a production model.
