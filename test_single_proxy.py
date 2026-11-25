"""The repo must contain exactly ONE proxy: the llpc waitroom door.

Spec's rule (2026-11-21): GlAIss used to ship its own transparent request-queue
proxy in front of the waitroom. Two proxies in one repo meant two routing brains,
two sets of token stats, and a console that quietly showed nothing because real
clients used the door. The proxy half was deleted; these checks keep it deleted.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
APP = REPO / "glaiss_app"
ENTRYPOINT = REPO / "glaiss.py"


def _app_sources():
    return {p.name: p.read_text() for p in sorted(APP.glob("*.py"))}


def console_routes():
    """Every path the console declares, from its route decorators."""
    routes = set()
    for name in ("monitor.py", "console.py"):
        for _verb, path in re.findall(
            r'@(?:router|app)\.(get|post|websocket)\("([^"]+)"', (APP / name).read_text()
        ):
            routes.add(path)
    return routes


# The console's whole HTTP surface. Adding an entry means it is a real feature the
# page (or the door) uses; deleting dead code should shrink this set, never let it
# drift from README.md.
EXPECTED_ROUTES = {
    "/", "/ws",
    "/api/health", "/api/state", "/api/history", "/api/logs",
    "/internal/waitroom/events",
    "/api/control/load", "/api/control/unload", "/api/control/lock",
    "/api/control/unlock", "/api/control/resume_auto", "/api/control/on_demand",
    "/api/control/schedule", "/api/control/reset_stats",
}


def test_console_route_surface_is_exactly_the_documented_set():
    routes = console_routes()
    assert routes == EXPECTED_ROUTES, (
        f"unexpected: {sorted(routes - EXPECTED_ROUTES)}; "
        f"missing: {sorted(EXPECTED_ROUTES - routes)}"
    )


def test_readme_documents_every_route():
    """README's API bullets and the code's route table must be the same set."""
    readme = (REPO / "README.md").read_text()
    documented = set()
    for line in readme.splitlines():
        m = re.match(r"^\s*-\s*`(GET|POST|WEBSOCKET|WS|websocket)(\s+(/[^`\s]*))?`", line)
        if m:
            documented.add((m.group(3) or "/").split("?")[0])
    assert documented == console_routes(), (
        f"documented but not implemented: {sorted(documented - console_routes())}; "
        f"implemented but undocumented: {sorted(console_routes() - documented)}"
    )


def test_waitroom_is_the_only_server_side_request_handler():
    """Only llpc/waitroom.py may accept client inference traffic."""
    assert (REPO / "llpc" / "waitroom.py").exists()
    assert not (APP / "proxy.py").exists(), "the old GlAIss proxy must stay deleted"
    for name, src in _app_sources().items():
        assert "/{full_path:path}" not in src, f"{name}: passthrough route reintroduced"
        assert "/v1/chat/completions" not in src, f"{name}: console must not serve chat completions"


def test_console_has_no_forwarding_target():
    """No provider host/port/scheme means nothing to forward *to*."""
    src = (APP / "console.py").read_text() + ENTRYPOINT.read_text()
    for token in ("provider_host", "provider_port", "provider_scheme", "GLAISS_PROVIDER"):
        assert token not in src, f"{token}: the console must not hold a forwarding target"
    assert "httpx" not in src and "aiohttp" not in src, "console module must not call upstreams"


def test_entrypoint_mounts_console_not_proxy():
    src = ENTRYPOINT.read_text()
    assert not re.search(r"^\s*(from|import)\s+\S*proxy", src, re.M), \
        "entrypoint must not import a proxy module"
    assert "monitor.install(console)" in src


def test_monitor_reports_engine_as_lone_token_source():
    src = (APP / "monitor.py").read_text()
    assert '"traffic"' not in src, "proxy traffic snapshot is gone from the state payload"
    assert "engine_metrics(" in src, "token speed must come from the loaded engine"
