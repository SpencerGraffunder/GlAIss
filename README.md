# Glaiss

A window into what your local LLMs are doing. A transparent reverse proxy
you put in front of your OpenAI-compatible backend (vLLM, llama.cpp server,
LM Studio, etc). Requests/responses pass through unmodified — Glaiss just
taps the traffic to show live stats and a scrolling prompt/response log.

## Setup (Ubuntu 26, on the X299 box)

```bash
cd ~/glaiss   # or wherever you put this folder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 glaiss.py
```

First run creates `glaiss_config.json` next to the script with defaults:
- listens on port 8899
- forwards to http://127.0.0.1:8000 (change this in the dashboard settings
  panel, or edit the JSON directly)

Dashboard: `http://<box-ip>:8899/`

## Point your clients at Glaiss

Wherever you were pointing at vLLM/llama.cpp directly (e.g.
`http://127.0.0.1:8000/v1/chat/completions`), point at Glaiss instead:

```
http://<box-ip>:8899/v1/chat/completions
```

Glaiss forwards everything else through unchanged, so `/v1/models`, auth
headers, etc. all pass through as normal — it's a transparent proxy, not a
gateway that needs its own auth or model registry.

## Settings panel

Top-right gear icon. Lets you change:
- **provider host/port/scheme** — where Glaiss forwards to (your real
  backend). Takes effect immediately, no restart.
- **models** — free-text list, purely informational/display (shown in the
  top stats bar). Glaiss doesn't validate or route based on this — it just
  forwards whatever `model` field the client sent.

## Notes on the numbers

- **Prompt token counts and PP t/s are estimates.** Glaiss tokenizes with
  `cl100k_base` (a generic tokenizer) since it has no access to your actual
  model's tokenizer — the real count will differ somewhat depending on
  model. Good enough for a progress readout, not exact.
- **PP progress %** is a *projected* bar: Glaiss estimates prefill time from
  (estimated prompt tokens ÷ a rolling average PP-tokens/sec for that model)
  and ticks a bar toward 100% while waiting for the first streamed token. The
  instant the real first token arrives, it snaps to 100% and records the
  *actual* PP time, which folds back into the rolling average — so the
  estimate self-calibrates the more you use it. There's no way to get true
  mid-prefill progress from most backends since prefill is opaque until the
  first token emits.
- **Decode t/s** is real, computed from actual inter-token arrival times
  during streaming — not an estimate.
- Non-streaming requests (`stream: false`) still get logged, but decode t/s
  isn't meaningful for those (you get PP time + total tokens only).

## Multiple concurrent requests

Each request gets a short ID shown as a tag on both the prompt entry and the
matching response stream, and in a live chip in the top bar while it's in
flight. If you fire off several requests at once, use the ID tags to match
prompts to responses in the two panes.

## Files

- `glaiss.py` — the proxy + stats engine + websocket hub (single file)
- `static/index.html` — the dashboard (single file, no build step)
- `glaiss_config.json` — created on first run, holds provider address and
  the self-calibrating PP-speed table per model
