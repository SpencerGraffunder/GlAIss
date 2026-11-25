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


def test_prompt_sizes_render_on_queue_boxes():
    """Spec 2026-11-24: every queue box carries a prompt-size stat. A running
    request on the loaded model shows the ENGINE's measured numbers — total,
    cached, and the remainder actually processed — and everything else shows
    the door's ~4-chars/token estimate. Whole numbers, thousands separators."""
    html, js = _html_and_js()
    assert "function promptSpan" not in js and "const promptSpan" in js, \
        "the queue section builds one prompt stat span per box"
    assert "engine_prompt_total" in js and "engine_reused_tokens" in js \
        and "engine_prompt_read" in js, "the exact engine numbers drive the box"
    assert "cached" in js and "to process" in js, "Spec's cached-subtraction wording"
    assert "prompt_tokens_est" in js, "the door estimate feeds queued/running boxes"
    assert "toLocaleString('en-US')" in js, "token counts render whole with separators"
    assert "~" in js.split("const promptSpan")[1].split("};")[0], \
        "an estimate is labeled approximate, never dressed as exact"


def test_redundant_tiles_and_queue_jargon_are_gone():
    """Spec 2026-11-24: the hold tile, the engine-requests tile and the
    'llpc waitroom console' subtitle are gone; the queue boxes stopped speaking
    door internals (quantum/handoffs) and the loaded-model name is not echoed
    next to the loaded-model tile."""
    html, js = _html_and_js()
    for cid in ("stat-hold", "stat-reqs"):
        assert f'id="{cid}"' not in html, f"{cid} must not be in the page"
    assert "llpc waitroom console" not in html, "the brand line is just GLAISS"
    assert ">generating<" not in html, "the top bar tile is 'queue length' now"
    assert 'title="Requests the door is holding' in html and "queue length" in html
    busy_line = next((l for l in js.splitlines() if "$('stat-busy').textContent" in l), "")
    assert "m.active" not in busy_line, "the queue-length tile must not echo the model name"
    assert "quantum" not in js.split("function activityBoxes")[1].split("function renderQueue")[0], \
        "queue boxes are one box per request, not quantum counters"
    assert "handoffs" not in js.split("function activityBoxes")[1].split("function renderQueue")[0]
    assert "quantum left" not in js, "the Models tab must say it in plain words"


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


def test_queue_is_horizontal_and_logs_scroll_only_inside_the_box():
    """Mobile: the queue is a left/right strip and the PAGE never scrolls
    (Spec, 2026-11-24: the earlier whole-page-scroll mobile rule created a
    second scroll area that stranded log follow). Scrolling happens inside the
    ops pane, and on the Logs tab only inside #log-box."""
    html, _ = _html_and_js()
    assert re.search(r"\.qboxes\{[^}]*flex-wrap:nowrap", html), "queue boxes must not stack"
    assert re.search(r"#queue-strip\{[^}]*overflow-x:auto", html)
    assert re.search(r"body\{[^}]*overflow:hidden", html), "the page itself must not scroll"
    mob = re.search(r"@media\(max-width:700px\)\{(.*?)\}\s*</style>", html, re.S)
    assert mob, "expected a mobile block in the stylesheet"
    css = mob.group(1)
    assert "overflow:auto" not in css and "overflow:visible" not in css, \
        "mobile must not re-introduce a page-level scroll container"
    assert re.search(r"\.stats\{[^}]*overflow-x:auto", html), "stats must swipe away, not stack"
    assert re.search(r"#pane-logs\.active\{[^}]*display:flex", html), \
        "the logs pane must be a flex column so the box can fill the screen"
    assert re.search(r"#log-box\{[^}]*flex:1", html), "the log box fills the pane"
    assert re.search(r"#log-box\{[^}]*overflow:auto", html), "the log box owns the scroll"
    assert "box.scrollTop = box.scrollHeight" in html, "follow mode pins the newest line"


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
    assert re.search(r"if \(sig===state\.modelsSig\) \{ updateSpeedCells\(rows\); return; \}", js), \
        "an unchanged model table must not be re-built (speeds refresh in place)"
    assert "state.modelsSig=null" in js, "a toggle must force the row to repaint"
    assert re.search(r"r\.name===name\) r\.on_demand=on", js), \
        "the toggle must update the row optimistically, not wait for the next poll"


def test_model_rows_carry_their_own_speed_and_window():
    """Per-model prefill/decode live in the catalog row, and the schedule window
    is editable there (Spec, 2026-11-24: keep separate stats per model)."""
    html, js = _html_and_js()
    assert "modelSpeedCell" in js and "updateSpeedCells" in js
    assert "x.decode_tps" in js and "x.prefill_tps" in js
    assert "/api/control/schedule" in js and "editWindow" in js
    assert "Window" in html, "the catalog has a schedule column"


def test_honest_state_words_no_parked_and_loaded_together():
    """Unloading shows 'unloading', the incoming model shows 'loading', and the
    loaded model reports 'processing prompt'/'working' instead of 'idle'."""
    html, js = _html_and_js()
    for word in ("unloading", "loading", "processing prompt", "working"):
        assert word in js, f"modelRowStatus must be able to say '{word}'"
    assert "badge('idle'" not in js, "the catalog no longer labels a loaded model idle"


def test_secondary_stats_render_as_whole_numbers():
    """Spec, 2026-11-24: 'all second stats don't show decimals — round to the
    nearest one'. The shared fmt/bytes helpers must round to integers."""
    html, js = _html_and_js()
    assert re.search(r"const fmt = \(x,suf=''\) => x==null \? '—' : \(Number\.isFinite\(\+x\) \? Math\.round\(\+x\) : x\)\+suf;", js)
    assert "Math.round(n)+' '+u[i]" in js, "byte sizes round to whole units too"


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
