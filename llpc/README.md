# llpc side of GlAIss

This directory contains the llpc scripts that run outside the web process.

- `waitroom.py` — the :8080 single inference door. It now emits optional
  fire-and-forget events to GlAIss (`GLAISS_EVENTS_URL`) for arbiter handoff,
  request admission, hysteresis hold, Comfy slot claim, and lock rejection.
- `model_control.py` — CLI using the same shared state and guards as the web
  control API.
- `current-model.sh` and `slot-watchdog.py` — existing llpc helpers retained in
  the product repo so the installed runtime has one upstream source.

When installed by `scripts/install.sh`, the scripts are copied to
`~/llama-swap/`, while GlAIss itself is at `/opt/glaiss`. `PYTHONPATH=/opt/glaiss`
lets the installed `model_control.py` import `glaiss_app.model_state`, keeping
the web control, waitroom enforcement, and CLI on one lock format.

Event delivery is best-effort and has a one-second timeout. A GlAIss outage
cannot slow or block inference. Set
`WAITROOM_DISABLE_GLAISS_EVENTS=1` to turn it off.
