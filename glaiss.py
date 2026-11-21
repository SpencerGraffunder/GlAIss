#!/usr/bin/env python3
"""GlAIss console entrypoint (default :8899).

The console OBSERVES the llpc inference stack and lets an operator move the model
slot by hand. It does not proxy anything: the single proxy in this repo is the
waitroom door (``llpc/waitroom.py`` on :8080), which every client talks to
directly. See ``glaiss_app/console.py`` for what this module must never become.
"""

from __future__ import annotations

import uvicorn

from glaiss_app import __version__, console, monitor

monitor.install(console)

app = console.app
CONFIG = console.CONFIG
CONFIG_PATH = console.CONFIG_PATH
BASE_DIR = console.BASE_DIR
STATE_DIR = console.STATE_DIR
STATIC_DIR = console.STATIC_DIR
Hub = console.Hub
hub = console.hub


def main() -> None:
    from glaiss_app import collectors as C

    print(f"GlAIss console {__version__} listening on 0.0.0.0:{CONFIG['listen_port']}")
    print(f"Observing waitroom door at {C.WAITROOM_URL} (this process proxies nothing)")
    print("Model control: /api/control/* -> swap-to.sh | monitor: /api/state, /api/services, /api/logs")
    uvicorn.run(app, host="0.0.0.0", port=CONFIG["listen_port"])


if __name__ == "__main__":
    main()
