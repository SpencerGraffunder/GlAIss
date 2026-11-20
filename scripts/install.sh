#!/usr/bin/env bash
# Install/deploy GlAIss and its llpc waitroom-side integration.
#
# Usage:
#   scripts/install.sh web       # install/restart the GlAIss :8899 web console
#   scripts/install.sh waitroom  # install/waitroom code and unit (restarts routing)
#   scripts/install.sh all       # both
#
# Safety: model units (strata31, llama-qwen, comfyui, strata-dsv41) are never
# touched. waitroom restarts only model-waitroom, which can pause a routing
# decision briefly; use --no-restart to stage without restarting.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEST="${DEST:-/opt/glaiss}"
STATE_DIR="${STATE_DIR:-/var/lib/glaiss}"
LLPC_HOME="${LLPC_HOME:-/home/graffunder}"
LLPC_SWAP="${LLPC_SWAP:-${LLPC_HOME}/llama-swap}"
WEB_UNIT="${WEB_UNIT:-/etc/systemd/system/glaiss.service}"
WAITROOM_UNIT="${WAITROOM_UNIT:-/etc/systemd/system/model-waitroom.service}"
DO_WEB=0
DO_WAITROOM=0
RESTART=1

usage() { sed -n '2,9p' "$0" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    web) DO_WEB=1 ;;
    waitroom) DO_WAITROOM=1 ;;
    all) DO_WEB=1; DO_WAITROOM=1 ;;
    --no-restart) RESTART=0 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
  shift
done
[[ $((DO_WEB + DO_WAITROOM)) -eq 0 ]] && usage

if [[ $EUID -ne 0 ]]; then
  exec sudo "$0" "$@"
fi

sync_repo() {
  install -d -o graffunder -g graffunder "$DEST"
  rsync -a --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.py[cod]' \
    --exclude 'glaiss_config.json' \
    --exclude 'Thaumcraft-6-Source-Code/' \
    "$REPO/" "$DEST/"
}

install_web() {
  echo "==> syncing GlAIss from $REPO to $DEST"
  sync_repo
  install -d -o graffunder -g graffunder "$STATE_DIR"
  if [[ ! -d "$DEST/.venv" ]]; then
    sudo -u graffunder python3 -m venv "$DEST/.venv"
  fi
  sudo -u graffunder "$DEST/.venv/bin/pip" install -q -U pip wheel
  sudo -u graffunder "$DEST/.venv/bin/pip" install -q -r "$REPO/requirements.txt"
  install -m 0644 "$REPO/systemd/glaiss.service" "$WEB_UNIT"
  systemctl daemon-reload
  systemctl enable glaiss.service
  if [[ "$RESTART" == "1" ]]; then
    systemctl restart glaiss.service
  fi
  systemctl --no-pager --full status glaiss.service || true
}

install_waitroom() {
  echo "==> installing llpc waitroom-side code into $LLPC_SWAP"
  backup="$LLPC_SWAP/.pre-glaiss-$(date +%Y%m%d-%H%M%S)"
  install -d -o graffunder -g graffunder "$backup"
  for f in waitroom.py current-model.sh slot-watchdog.py; do
    if [[ -e "$LLPC_SWAP/$f" ]]; then
      install -o graffunder -g graffunder "$LLPC_SWAP/$f" "$backup/$f"
    fi
    install -o graffunder -g graffunder -m 0755 "$REPO/llpc/$f" "$LLPC_SWAP/$f"
  done
  install -o graffunder -g graffunder -m 0644 "$REPO/llpc/model_control.py" "$LLPC_SWAP/model_control.py"
  install -m 0644 "$REPO/systemd/model-waitroom.service" "$WAITROOM_UNIT"
  systemctl daemon-reload
  systemctl enable model-waitroom.service
  if [[ "$RESTART" == "1" ]]; then
    systemctl restart model-waitroom.service
  fi
  systemctl --no-pager --full status model-waitroom.service || true
}

[[ $DO_WEB == 1 ]] && install_web
[[ $DO_WAITROOM == 1 ]] && install_waitroom
echo "GlAIss install complete."
