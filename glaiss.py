#!/usr/bin/env python3
"""GlAIss combined waitroom observability entrypoint.

The old proxy implementation now lives in ``glaiss_app/proxy.py``; monitor and
manual model-control endpoints are mounted before the catch-all proxy route.
"""

from __future__ import annotations

import uvicorn

from glaiss_app import monitor
from glaiss_app import proxy

monitor.install(proxy)

app = proxy.app
CONFIG = proxy.CONFIG
CONFIG_PATH = proxy.CONFIG_PATH
BASE_DIR = proxy.BASE_DIR
STATE_DIR = proxy.STATE_DIR
STATIC_DIR = proxy.STATIC_DIR

Hub = proxy.Hub
hub = proxy.hub
GenQueue = proxy.GenQueue
gen_queue = proxy.gen_queue
Stats = proxy.Stats
stats = proxy.stats
request_tasks = proxy.request_tasks
stats_pusher = proxy.stats_pusher
load_config = proxy.load_config
save_config = proxy.save_config


def main() -> None:
    print(f"GlAIss listening on 0.0.0.0:{CONFIG['listen_port']}")
    print(f"Proxying chat traffic to provider at {proxy.provider_base_url()}")
    print(f"LLPC monitor mounted on /api/state and /api/services")
    uvicorn.run(app, host="0.0.0.0", port=CONFIG["listen_port"])


if __name__ == "__main__":
    main()
