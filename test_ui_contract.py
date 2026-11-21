"""Guard the console page's JS against DOM drift.

The 2026-11-20 outage class: `$('…')` is `getElementById`, but a call site passed
a CSS selector (`'#monitor-dot'`), so the render function threw on its first line
every 2 s — the page kept streaming logs while every stat, tile and table stayed
empty, with no server-side signal at all. These checks fail that statically.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "glaiss_app" / "static" / "index.html"


def _html_and_js():
    html = STATIC.read_text()
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert scripts, "console page must ship an inline script"
    return html, "\n".join(scripts)


def test_all_js_element_ids_exist():
    html, js = _html_and_js()
    ids = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))
    used = set(re.findall(r"\$\('#?([A-Za-z0-9_-]+)'\)", js))
    used |= set(re.findall(r"setStatus\('#?([A-Za-z0-9_-]+)'", js))
    assert used, "expected the page to look elements up by id"
    missing = sorted(used - ids)
    assert not missing, f"JS looks up ids that the page does not define: {missing}"


def test_dollar_helper_tolerates_selector_prefix():
    """`$` must strip a leading '#', otherwise selector-style call sites return null."""
    _, js = _html_and_js()
    decl = re.search(r"const \$ = ([^\n]+)", js)
    assert decl, "expected a const $ helper"
    assert "getElementById" in decl.group(1)
    assert re.search(r"replace\(/\^#", decl.group(1)), "$ must normalize '#id' to 'id'"


def test_lookup_results_are_guarded_where_optional():
    _, js = _html_and_js()
    assert re.search(r"function setStatus\([^)]*\)\s*\{[^}]*if\(!d\)\s*return", js), \
        "setStatus must no-op on a missing element rather than throw"


def test_no_proxy_era_survivors():
    """The page must not reference the deleted proxy's routes or panes."""
    html, js = _html_and_js()
    for token in ("/api/cancel", "/api/config", "provider_host", "prompt-body",
                  "response-body", "history_start", "openResponses", "stat-proxy"):
        assert token not in html, f"{token}: proxy-era UI still referenced"


def test_door_is_presented_as_the_only_request_path():
    _, js = _html_and_js()
    assert "stat-door" in js, "header must show the waitroom door address"
    assert "cfg.waitroom_url" in js, "door address comes from the monitor's config block"
    assert "renderQueue(m)" in js, "queue strip must be fed by the door's arbiter (model block)"


def test_page_only_calls_routes_the_console_serves():
    """The inverse of the route-surface test: no call site pointing at dead code."""
    from test_single_proxy import console_routes

    _, js = _html_and_js()
    called = set(re.findall(r"(?:fetch|post)\('(/[^']*)'", js))
    called |= {"/ws"} if "WebSocket(`${p}://${location.host}/ws`)" in js else set()
    called = {p.split("?")[0] for p in called}
    unknown = sorted(p for p in called if p not in console_routes())
    assert not unknown, f"page calls routes that do not exist: {unknown}"
