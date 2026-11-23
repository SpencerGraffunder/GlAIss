"""Guard the console page's JS against DOM drift.

Two outage classes have already happened here, both invisible to the API:
`$('…')` is `getElementById` while a call site passed a CSS selector, so the
render loop threw every 2 s and the page stayed blank while every endpoint
returned 200; and dead status chrome (live/connected/door/arbiter tiles, a
placeholder queue box) told lies about state that was already shown elsewhere.
These checks pin the contract the page must keep.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "glaiss_app" / "static" / "index.html"

# Chrome that used to sit in the header and said nothing useful (Spec 2026-11-21).
BANNED_IDS = ("stat-live", "stat-monitor", "stat-door", "stat-active",
              "ws-dot", "monitor-dot", "job-status", "waitroom-lane")
BANNED_WORDS = ("arbiter", "/api/cancel", "/api/config", "provider_host",
                "prompt-body", "response-body", "openResponses", "stat-proxy")


def _html_and_js():
    html = STATIC.read_text()
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert scripts, "console page must ship an inline script"
    return html, "\n".join(scripts)


def test_all_js_element_ids_exist():
    html, js = _html_and_js()
    ids = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))
    used = set(re.findall(r"\$\('#?([A-Za-z0-9_-]+)'\)", js))
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


def test_dead_chrome_stays_dead():
    """No connection tiles, no door address, no 'arbiter' label, no fake queue box."""
    html, js = _html_and_js()
    for token in BANNED_IDS:
        assert token not in html, f"{token}: removed status chrome is back"
    for word in BANNED_WORDS:
        assert word not in html, f"{word}: removed/dead concept is back"
    assert "setStatus" not in js, "the dot/status helpers went away with the tiles"
    assert "nothing queued" not in html, "the queue must not show a placeholder request"


def test_brokenness_is_what_shows_the_indicator():
    """Quiet when healthy; one named reason when the channel or the door is down."""
    html, js = _html_and_js()
    assert 'id="offline"' in html and 'id="offline-text"' in html
    assert "#offline{" in html and "#offline.show{display:inline-flex}" in html, \
        "the offline pill must be hidden by default"
    assert "state.wsOpen" in js and "state.doorOk" in js
    for reason in ("console channel down", "waitroom door unreachable"):
        assert reason in js, f"offline indicator must name '{reason}'"


def test_redundant_manual_controls_are_removed():
    """Spec 2026-11-22: AUTO / Resume auto / NO LOCK / Unlock duplicated the
    per-model enable checkboxes and are gone. Load is a plain model action."""
    html, js = _html_and_js()
    for cid in ("chip-mode", "chip-lock", "chip-mode-text", "chip-lock-text",
                "resume-auto", "unlock-btn"):
        assert f'id="{cid}"' not in html, f"{cid} must not be in the page"
    for token in ("Resume auto", "Unlock", "AUTO", "MANUAL", "NO LOCK", "LOCKED ·"):
        assert token not in html and token not in js, f"removed control token reappeared: {token}"
    assert "/api/control/unlock" not in js and "/api/control/resume_auto" not in js
    assert "lock_after_load" not in js, "Load must not re-add the lock action"


def test_queue_shows_only_live_activity_for_five_seconds():
    html, js = _html_and_js()
    assert re.search(r"const DONE_SHOW_S = 5;", js), "finished requests age out after 5 s"
    assert "state.done = state.done.filter" in js
    assert "m.inflight > 0" in js, "a box only counts as generating with work in flight"
    assert 'id="queue-hint"' in html and "nothing generating or queued" in html, \
        "an empty queue reads as a hint, never as a request box"


def test_token_chart_shows_decode_and_prefill_on_two_axes():
    html, js = _html_and_js()
    assert "function drawTokenChart" in js
    assert "decode t/s" in js and "prefill t/s" in js, "both series must be labeled"
    assert "drawChart('tps-chart'" not in js, "token speed is no longer decode-only"
    assert "inflight_requests" in js, "the queue renders one box per in-flight request"


def test_queue_is_horizontal_and_stats_can_be_scrolled_past():
    """Mobile: the queue is a left/right strip and the page itself scrolls, so the
    stats block can never pin the top of the screen."""
    html, _ = _html_and_js()
    assert re.search(r"\.qboxes\{[^}]*flex-wrap:nowrap", html), "queue boxes must not stack"
    assert re.search(r"#queue-strip\{[^}]*overflow-x:auto", html)
    mob = re.search(r"@media\(max-width:700px\)\{(.*?)\}\s*</style>", html, re.S)
    assert mob, "expected a mobile block in the stylesheet"
    css = mob.group(1)
    assert re.search(r"body\{[^}]*overflow:auto", css), "the page must scroll on mobile"
    assert re.search(r"\.stats\{[^}]*overflow-x:auto", css), "stats must swipe away, not stack"
    assert re.search(r"#ops\{[^}]*flex:none", css), "the tab area must release the viewport"
    assert re.search(r"\.ops-body\{[^}]*overflow:visible", css)


def test_on_demand_switch_lives_in_the_model_catalog():
    html, js = _html_and_js()
    assert 'id="models-body"' in html and "On demand" in html
    assert "type=\"checkbox\"" in html and "setOnDemand(" in js
    assert "/api/control/on_demand" in js
    assert "on_demand_disabled" in js, "the parked list comes from the door state"
    assert 'id="ondemand-warn"' in html, "parking every model must be called out"


def test_model_table_does_not_repaint_under_a_click():
    """The 2 s poll must not rebuild the catalog when nothing changed: a rebuild
    replaces the checkbox a click is landing on, and showed a stale value for a
    cycle after toggling (found in the rendered DOM, 2026-11-21)."""
    _, js = _html_and_js()
    assert re.search(r"const sig=JSON\.stringify\(", js), "row signature required"
    assert re.search(r"if \(sig===state\.modelsSig\) return;", js), \
        "an unchanged model table must not be re-rendered"
    assert "state.modelsSig=null" in js, "a toggle must force the row to repaint"
    assert re.search(r"r\.name===name\) r\.on_demand=on", js), \
        "the toggle must update the row optimistically, not wait for the next poll"


def test_the_tab_bar_wraps_rather_than_hiding_its_actions():
    """"I… after the metrics tab" (Spec, 2026-11-21): the bar scrolled sideways,
    so anything past the last visible tab was unreachable. Content inside
    #ops-tabs must wrap into view instead."""
    html, _ = _html_and_js()
    tabs_css = re.search(r"#ops-tabs\{([^}]*)\}", html).group(1)
    assert "flex-wrap:wrap" in tabs_css
    assert "overflow" not in tabs_css, "no scroll container inside the tab bar"
    actions_css = re.search(r"#ops-actions\{([^}]*)\}", html).group(1)
    assert "flex-wrap:wrap" in actions_css


def test_page_only_calls_routes_the_console_serves():
    """The inverse of the route-surface test: no call site pointing at dead code."""
    from test_single_proxy import console_routes

    _, js = _html_and_js()
    called = set(re.findall(r"(?:fetch|post)\('(/[^']*)'", js))
    called |= {"/ws"} if "WebSocket(`${p}://${location.host}/ws`)" in js else set()
    called = {p.split("?")[0] for p in called}
    unknown = sorted(p for p in called if p not in console_routes())
    assert not unknown, f"page calls routes that do not exist: {unknown}"
