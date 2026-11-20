#!/bin/bash
# Which model should the box hold when nothing is loaded?
#   last-model file  >  fresh waitroom intent  >  ~/llama-swap/default-model
#   >  first catalog text model
# One answer for every backstop (keep-warm, slot-watchdog boot path, the plan
# installer) so they can never disagree about the model the box "wants".
# LAST-USED STICKINESS (Spec 2026-11-18): the waitroom writes the last-model
# file from the OBSERVED loaded model (no expiry) — an idle box comes back to
# what it was serving and never drifts to the default model. default-model
# survives only as the picker/first-boot default and the fallback when the
# box has never served anything since the file's history began.
set -u
DIR=/home/graffunder/llama-swap
INTENT="${WAITROOM_INTENT_FILE:-$DIR/waitroom-intent}"
TTL="${WAITROOM_INTENT_TTL_S:-43200}"
DEFL="${WAITROOM_DEFAULT_FILE:-$DIR/default-model}"
LAST="${WAITROOM_LASTMODEL_FILE:-$DIR/last-model}"
LOCK="${MODEL_LOCK_FILE:-/home/graffunder/.llama-swap/model-lock.json}"
FALLBACK=$(python3 "$DIR/llpc_models.py" first-text 2>/dev/null) || FALLBACK=""

python3 - "$LOCK" "$LAST" "$INTENT" "$TTL" "$DEFL" "${FALLBACK:-}" <<'PY'
import json, sys, time
lock_file, last_file, intent_file, ttl, dflt, fallback = sys.argv[1:7]
ttl = float(ttl)
try:
    m = json.load(open(lock_file)).get("model")
    if m:
        print(m.strip()); raise SystemExit
except SystemExit:
    raise
except Exception:
    pass
try:
    m = open(last_file).read().strip()
    if m:
        print(m); raise SystemExit
except SystemExit:
    raise
except Exception:
    pass
try:
    d = json.load(open(intent_file))
    if time.time() - float(d.get("at", 0)) <= ttl and d.get("model"):
        print(d["model"]); raise SystemExit
except Exception:
    pass
try:
    print(open(dflt).read().strip() or fallback)
except Exception:
    print(fallback)
PY
