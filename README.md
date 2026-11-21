# GlAIss

GlAIss is the llpc inference operations console. It combines the original
transparent LLM proxy window with the live `model-waitroom` queue and llama-swap
model control.

The web UI has **no login, MFA, or authorization layer by design**. It is an
admin-only operations surface and is expected to be reachable only through the
trusted VPN network.

## What it shows

- Original GlAIss request queue, FIFO position, prefill estimate, real decode
  t/s, streaming response text, and reasoning stream.
- llpc waitroom state: loaded model, active arbiter model, in-flight requests,
  ordered waiting models, swap hysteresis, quantum timer, handoffs, Comfy claim,
  intent, manual lock, and manual-control suppression.
- Per-service rows matching the standard llpc journal command:

  `journalctl -u strata31 -u llama-qwen -u comfy-ui -u llama-swap -u whisper-server -u whisper-shim -u switcheroo-control -u model-waitroom -u strata-dsv41.service -u llama-server`

- Service CPU, RSS, VRAM, PID, state, restart count, recent log count, last log,
  and a live journal console for the same units.
- CPU package power, total GPU power, combined system power, RAM, VRAM, CPU and
  GPU utilization, decode t/s, and prefill t/s history.
- Registry-backed model catalog and manual model controls.

## Architecture

```text
clients -> GlAIss :8899 -> llpc waitroom :8080 -> llama-swap :8090 -> engine
                         -> monitor/control router
                         -> journal + RAPL + nvidia-smi + llama-swap + registry
```

The proxy remains transparent for inference. Monitor routes are mounted before
the proxy catch-all. `WAITROOM_URL` defaults to `http://127.0.0.1:8080`;
monitoring llama-swap directly at `LLAMA_SWAP_URL` is read-only and used for
running state and manual unload operations.

```text
glaiss.py                      runtime entrypoint
glaiss_app/proxy.py            original FIFO proxy and websocket log
glaiss_app/monitor.py          monitor/control router and model-control jobs
glaiss_app/collectors.py       waitroom, systemd, GPU, RAPL, RAM collectors
glaiss_app/journal.py          journalctl follow ring
glaiss_app/model_state.py      shared manual-lock/manual-control file format
glaiss_app/static/index.html   operations console
llpc/waitroom.py               llpc source of truth, with GlAIss event feed
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
| `GLAISS_PROVIDER_HOST` | `127.0.0.1` | proxy destination |
| `GLAISS_PROVIDER_PORT` | config/8080 | proxy destination port |
| `WAITROOM_URL` | `http://127.0.0.1:8080` | monitor source |
| `LLAMA_SWAP_URL` | `http://127.0.0.1:8090` | running state and unload API |
| `LLPC_MODELS_PY` | `~/llama-swap/llpc_models.py` | model registry |
| `GLAISS_MONITOR_POLL_S` | `2` | collector interval |
| `GLAISS_DISABLE_MONITOR_TASKS` | off | disable background collectors for tests |

## API

- `GET /api/health`
- `GET /api/state`
- `GET /api/history?seconds=900`
- `GET /api/model`
- `GET /api/services`
- `GET /api/system`
- `GET /api/logs?limit=300&unit=strata31&level=error`
- `POST /internal/waitroom/events` (waitroom fire-and-forget feed)
- `POST /api/control/load`
- `POST /api/control/unload`
- `POST /api/control/lock`
- `POST /api/control/unlock`
- `POST /api/control/resume_auto`

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests cover FIFO serialization, cancellation, prompt diffing, history replay,
monitor route precedence, waitroom event fan-out, hysteresis/lock state, and
manual-control guards. No test starts, loads, or stops a production model.
