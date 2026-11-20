"""
Tests for the Glaiss FIFO generation queue.

Strategy: spin up a real mock upstream (a separate FastAPI app served by
uvicorn on a localhost port) that records the order it sees requests and the
max concurrency it observes. Then drive the glaiss app through httpx's
ASGITransport. Because glaiss reaches the upstream over real TCP, this
genuinely exercises the serialization guarantee (the whole point of the
queue: the backend can only hold one context at a time).

No pytest-asyncio needed — each test runs its own loop via asyncio.run().
"""

import asyncio
import json
import socket
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState

from glaiss_app import proxy as glaiss
from glaiss_app.proxy import GenQueue, Stats


# ---------------------------------------------------------------------------
# Mock upstream
# ---------------------------------------------------------------------------


class UpstreamState:
    def __init__(self, block_seconds=0.3):
        self.block_seconds = block_seconds
        self.order = []  # _rid values, in the order the upstream STARTS them
        self.max_concurrency = 0
        self.current = 0
        self.models_calls = 0


def make_upstream(state: UpstreamState) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        rid = body.get("_rid", "?")
        # record start order + track concurrency (single loop, no lock needed)
        state.current += 1
        state.max_concurrency = max(state.max_concurrency, state.current)
        state.order.append(rid)
        await asyncio.sleep(state.block_seconds)  # simulate VRAM-bound generation
        state.current -= 1

        if body.get("stream"):
            async def gen():
                yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                yield 'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return {
            "choices": [{"message": {"content": "Hello world"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    @app.get("/v1/models")
    async def models():
        state.models_calls += 1
        return {"data": [{"id": "mock-model"}]}

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def dead_port():
    """Return (port, socket) where the socket stays bound so nothing can
    listen there — guarantees connection-refused for the connect-error test."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1], s


async def start_upstream(app: FastAPI, port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            break
        except OSError:
            await asyncio.sleep(0.02)
    return server, task


async def stop_upstream(server: uvicorn.Server, task: asyncio.Task):
    server.should_exit = True
    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        task.cancel()


def reset_glaiss():
    """Give each test a clean queue, stats, hub (incl. log buffer), and prompt history."""
    glaiss.gen_queue = GenQueue()
    glaiss.stats = Stats()
    glaiss.hub.clients.clear()
    glaiss.hub.log.clear()
    glaiss.prompt_history.clear()


class FakeWS:
    """Records broadcast messages; satisfies hub.broadcast's interface."""

    def __init__(self):
        self.messages = []
        self.application_state = WebSocketState.CONNECTED

    async def send_text(self, payload: str):
        self.messages.append(json.loads(payload))


@asynccontextmanager
async def upstream_env(state: UpstreamState):
    port = free_port()
    app = make_upstream(state)
    server, task = await start_upstream(app, port)
    old = (
        glaiss.CONFIG.get("provider_host"),
        glaiss.CONFIG.get("provider_port"),
        glaiss.CONFIG.get("provider_scheme"),
    )
    glaiss.CONFIG.update(
        provider_host="127.0.0.1", provider_port=port, provider_scheme="http"
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=glaiss.app)
        ) as client:
            yield client
    finally:
        glaiss.CONFIG.update(
            provider_host=old[0], provider_port=old[1], provider_scheme=old[2]
        )
        await stop_upstream(server, task)


CHAT_URL = "http://glaiss/v1/chat/completions"
CHAT_BODY = {"model": "m", "stream": False, "messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# Unit: GenQueue FIFO
# ---------------------------------------------------------------------------


def test_genqueue_fifo_unit():
    async def main():
        q = GenQueue()
        order = []

        async def worker(i):
            await q.acquire()
            order.append(i)
            await asyncio.sleep(0.01)
            q.release()

        await asyncio.gather(*(worker(i) for i in range(5)))
        assert order == [0, 1, 2, 3, 4], f"FIFO violated: {order}"

    asyncio.run(main())


def test_genqueue_contention_created_outside_loop():
    """A GenQueue created OUTSIDE the running event loop (exactly like the module
    global ``gen_queue`` at import time) must still serialize FIFO when contended
    inside the loop. A plain asyncio.Lock created at import time is bound to the
    default loop and raises 'got Future attached to a different loop' here under
    contention — LoopLock (lazy binding to the running loop) avoids that."""
    q = GenQueue()  # created here, outside asyncio.run -> mirrors import time

    async def main():
        order = []

        async def worker(i):
            await q.acquire()
            order.append(i)
            await asyncio.sleep(0.01)
            q.release()

        await asyncio.gather(*(worker(i) for i in range(5)))
        assert order == [0, 1, 2, 3, 4], f"FIFO violated: {order}"

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Integration: serialization + FIFO over real TCP
# ---------------------------------------------------------------------------


def test_serialization_and_fifo():
    async def main():
        reset_glaiss()
        state = UpstreamState(block_seconds=0.3)
        async with upstream_env(state) as client:

            async def send(i):
                r = await client.post(
                    CHAT_URL, json={**CHAT_BODY, "_rid": f"r{i}"}
                )
                assert r.status_code == 200
                return r.json()

            # stagger sends so arrival order at glaiss is deterministic
            tasks = []
            for i in range(3):
                tasks.append(asyncio.create_task(send(i)))
                await asyncio.sleep(0.05)
            results = await asyncio.gather(*tasks)

            assert all(
                x["choices"][0]["message"]["content"] == "Hello world"
                for x in results
            )

        # the critical VRAM property: never more than one in flight
        assert state.max_concurrency == 1, (
            f"expected strict serialization, saw max_concurrency="
            f"{state.max_concurrency}"
        )
        # and they drained in arrival order
        assert state.order == ["r0", "r1", "r2"], f"FIFO order violated: {state.order}"

    asyncio.run(main())


def test_passthrough_bypasses_queue():
    async def main():
        reset_glaiss()
        state = UpstreamState()
        async with upstream_env(state) as client:
            r = await client.get("http://glaiss/v1/models")
            assert r.status_code == 200
            assert r.json() == {"data": [{"id": "mock-model"}]}

        # lightweight endpoints don't count as generation requests
        assert glaiss.stats.total_requests == 0
        assert state.models_calls == 1

    asyncio.run(main())


def test_streaming_releases_lock():
    async def main():
        reset_glaiss()
        state = UpstreamState(block_seconds=0.2)
        async with upstream_env(state) as client:
            # streaming request holds the slot until fully consumed
            r1 = await client.post(
                CHAT_URL, json={**CHAT_BODY, "stream": True, "_rid": "s0"}
            )
            assert r1.status_code == 200
            assert "Hello" in r1.text
            # this must proceed — proves the lock was released after the stream
            r2 = await client.post(
                CHAT_URL, json={**CHAT_BODY, "_rid": "s1"}
            )
            assert r2.status_code == 200

        assert state.max_concurrency == 1
        assert state.order == ["s0", "s1"]

    asyncio.run(main())


def test_connect_error_releases_lock():
    async def main():
        reset_glaiss()
        port, keep_open = dead_port()
        glaiss.CONFIG.update(
            provider_host="127.0.0.1", provider_port=port, provider_scheme="http"
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=glaiss.app)
            ) as client:
                r = await client.post(
                    CHAT_URL, json={**CHAT_BODY, "_rid": "err"}
                )
                assert r.status_code == 502
        finally:
            keep_open.close()

        # the failed request must not have left the slot held
        assert glaiss.gen_queue.waiting == 0
        await glaiss.gen_queue.acquire()  # should succeed immediately
        glaiss.gen_queue.release()

    asyncio.run(main())


def test_queued_count_in_snapshot():
    async def main():
        reset_glaiss()
        state = UpstreamState(block_seconds=0.5)
        async with upstream_env(state) as client:
            t1 = asyncio.create_task(
                client.post(CHAT_URL, json={**CHAT_BODY, "_rid": "q0"})
            )
            await asyncio.sleep(0.1)  # q0 acquires and is in flight
            t2 = asyncio.create_task(
                client.post(CHAT_URL, json={**CHAT_BODY, "_rid": "q1"})
            )
            await asyncio.sleep(0.1)  # q1 is now waiting in the queue
            snap = glaiss.stats.snapshot()
            assert snap["queued_count"] >= 1, (
                f"expected queued_count>=1, got {snap['queued_count']}"
            )
            await asyncio.gather(t1, t2)

    asyncio.run(main())


def test_queue_events_broadcast():
    async def main():
        reset_glaiss()
        state = UpstreamState(block_seconds=0.2)
        fake = FakeWS()
        glaiss.hub.clients.add(fake)
        try:
            async with upstream_env(state) as client:
                t1 = asyncio.create_task(
                    client.post(CHAT_URL, json={**CHAT_BODY, "_rid": "e0"})
                )
                await asyncio.sleep(0.05)
                t2 = asyncio.create_task(
                    client.post(CHAT_URL, json={**CHAT_BODY, "_rid": "e1"})
                )
                await asyncio.gather(t1, t2)
        finally:
            glaiss.hub.clients.discard(fake)

        queued = [m for m in fake.messages if m.get("type") == "queued"]
        started = [m for m in fake.messages if m.get("type") == "started"]
        assert len(queued) >= 2, f"expected >=2 queued events, got {len(queued)}"
        assert len(started) >= 2, f"expected >=2 started events, got {len(started)}"
        assert all("position" in m for m in queued)
        assert all("wait_seconds" in m for m in started)

    asyncio.run(main())


def test_cancel_queued_request():
    """Cancelling a queued request: it never reaches the upstream, the slot is
    freed for the next request, a 'cancelled' event is broadcast, the request
    drops out of the active set, and the waiter counter does not leak."""
    async def main():
        reset_glaiss()
        state = UpstreamState(block_seconds=0.5)
        fake = FakeWS()
        glaiss.hub.clients.add(fake)
        try:
            async with upstream_env(state) as client:
                # A acquires the slot and holds it (blocking at the upstream).
                ta = asyncio.create_task(
                    client.post(CHAT_URL, json={**CHAT_BODY, "_rid": "A"})
                )
                await asyncio.sleep(0.1)          # A is now in flight
                # B arrives and is queued behind A.
                tb = asyncio.create_task(
                    client.post(CHAT_URL, json={**CHAT_BODY, "_rid": "B"})
                )
                await asyncio.sleep(0.1)          # B is now waiting in the queue

                # B's server-side req_id is the one still in the 'queued' phase.
                queued = [
                    rid for rid, e in glaiss.stats.active_requests.items()
                    if e.get("phase") == "queued"
                ]
                assert len(queued) == 1, f"expected exactly 1 queued, got {queued}"
                b_id = queued[0]

                rc = await client.post(f"http://glaiss/api/cancel/{b_id}")
                assert rc.status_code == 200, f"cancel -> {rc.status_code}: {rc.text}"
                await asyncio.sleep(0.1)          # let the cancellation settle

                # B is gone from the active set ...
                assert b_id not in glaiss.stats.active_requests
                # ... and a 'cancelled' event was broadcast for it.
                cancelled = [m for m in fake.messages if m.get("type") == "cancelled"]
                assert any(m["id"] == b_id for m in cancelled), "no cancelled event for B"

                # Settle both: A completes normally; B was cancelled mid-wait.
                ra, rb = await asyncio.gather(ta, tb, return_exceptions=True)
                assert isinstance(ra, httpx.Response) and ra.status_code == 200

                # THE CONTRACT: B never reached the upstream.
                assert "A" in state.order
                assert "B" not in state.order, f"B leaked to upstream: {state.order}"

                # And the slot was freed: a fresh request C proceeds.
                tc = await client.post(CHAT_URL, json={**CHAT_BODY, "_rid": "C"})
                assert tc.status_code == 200
                assert state.order[-1] == "C"
                assert glaiss.gen_queue.waiting == 0, "waiter counter leaked"
        finally:
            glaiss.hub.clients.discard(fake)

    asyncio.run(main())


def test_cancel_unknown_request_404():
    """Cancelling a req_id that is not active returns 404 (nothing to abort)."""
    async def main():
        reset_glaiss()
        state = UpstreamState(block_seconds=0.1)
        async with upstream_env(state) as client:
            r = await client.post("http://glaiss/api/cancel/doesnotexist")
            assert r.status_code == 404
            assert glaiss.stats.active_requests == {}
            assert glaiss.gen_queue.waiting == 0

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Persistence: log buffer + history replay across reloads
# ---------------------------------------------------------------------------


def test_log_buffer_records_request_events():
    """Non-streaming request logs prompt/queued/started/response; stats/config
    are live-only and never enter the buffer."""
    async def main():
        reset_glaiss()
        state = UpstreamState(block_seconds=0.1)
        async with upstream_env(state) as client:
            await client.post(CHAT_URL, json={**CHAT_BODY, "_rid": "b0"})

        types = [m.get("type") for m in glaiss.hub.history()]
        assert "prompt" in types
        assert "queued" in types
        assert "started" in types
        assert "response" in types
        # live-state types must NOT be buffered
        assert "stats" not in types
        assert "config" not in types

    asyncio.run(main())


def test_streaming_logs_final_not_deltas():
    """A streaming response records ONE final entry (with full text), not one
    entry per token delta — otherwise the buffer would explode."""
    async def main():
        reset_glaiss()
        state = UpstreamState(block_seconds=0.1)
        async with upstream_env(state) as client:
            r = await client.post(
                CHAT_URL, json={**CHAT_BODY, "stream": True, "_rid": "s0"}
            )
            assert r.status_code == 200

        responses = [m for m in glaiss.hub.history() if m.get("type") == "response"]
        # exactly one logged response (the final one), not one per delta
        assert len(responses) == 1, f"expected 1 logged response, got {len(responses)}"
        assert responses[0]["done"] is True
        # and it carries the full accumulated text for replay
        assert responses[0]["text"] == "Hello world"

    asyncio.run(main())


def test_log_buffer_bounded():
    async def main():
        reset_glaiss()
        for i in range(glaiss.LOG_BUFFER_MAX + 50):
            await glaiss.hub.emit({"type": "prompt", "id": f"x{i}", "ts": 0})
        assert len(glaiss.hub.history()) == glaiss.LOG_BUFFER_MAX
        # oldest entries dropped off the front
        ids = [m["id"] for m in glaiss.hub.history()]
        assert ids[0] != "x0"

    asyncio.run(main())


class FakeWSClient:
    """Stands in for a dashboard WebSocket: records sent messages and
    immediately 'disconnects' on the first receive so the endpoint returns."""

    def __init__(self):
        self.messages = []
        self.application_state = WebSocketState.CONNECTED

    async def accept(self):
        pass

    async def send_text(self, payload: str):
        self.messages.append(json.loads(payload))

    async def receive_text(self):
        raise WebSocketDisconnect()


def test_history_replayed_on_connect():
    """A reconnecting client receives history_start + the full buffered log,
    so a page reload restores the log (and stats already arrive via snapshot)."""
    async def main():
        reset_glaiss()
        state = UpstreamState(block_seconds=0.1)
        async with upstream_env(state) as client:
            await client.post(CHAT_URL, json={**CHAT_BODY, "_rid": "h0"})
            await client.post(
                CHAT_URL, json={**CHAT_BODY, "stream": True, "_rid": "h1"}
            )

        fake = FakeWSClient()
        await glaiss.websocket_endpoint(fake)

        types = [m.get("type") for m in fake.messages]
        # header: snapshot + config, then the replay marker
        assert types[0] == "stats"
        assert types[1] == "config"
        assert types[2] == "history_start"

        replayed = fake.messages[3:]
        replay_types = [m.get("type") for m in replayed]
        # both requests' events are present in the replay
        assert replay_types.count("prompt") == 2
        assert replay_types.count("response") == 2
        # every logged response is a final (done) entry carrying its full text,
        # so a replay can reconstruct each response without the live deltas
        done_responses = [
            m for m in replayed if m.get("type") == "response" and m.get("done")
        ]
        assert len(done_responses) == 2
        assert all(m["text"] == "Hello world" for m in done_responses)

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Prompt dedup: diff against the most recent similar request
# ---------------------------------------------------------------------------


def test_prompt_history_best_match_unit():
    from glaiss_app.proxy import PromptHistory

    base = "[system] You are helpful.\n[user] What is 2+2?"
    extended = base + "\n[assistant] 4\n[user] And 3+3?"

    h = PromptHistory()
    assert h.best_match(base) is None  # nothing in history yet

    h.add(base)
    m = h.best_match(extended)
    assert m is not None
    matched, prefix_len = m
    assert matched == base
    assert prefix_len == len(base)
    assert extended[prefix_len:] == "\n[assistant] 4\n[user] And 3+3?"


def test_prompt_history_no_match_when_unrelated():
    from glaiss_app.proxy import PromptHistory

    h = PromptHistory()
    h.add("[system] A completely different topic about cooking pasta.")
    m = h.best_match("[system] Tell me a joke about programming.")
    assert m is None  # no long shared prefix -> treated as a fresh conversation


def test_prompt_diff_emitted_on_second_request():
    async def main():
        reset_glaiss()
        state = UpstreamState(block_seconds=0.1)
        fake = FakeWS()
        glaiss.hub.clients.add(fake)
        try:
            async with upstream_env(state) as client:
                # first request: fresh conversation -> not a diff
                await client.post(
                    CHAT_URL,
                    json={
                        **CHAT_BODY,
                        "messages": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "What is 2+2?"},
                        ],
                    },
                )
                # second request: same context + one new turn -> should be a diff
                await client.post(
                    CHAT_URL,
                    json={
                        **CHAT_BODY,
                        "messages": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "4"},
                            {"role": "user", "content": "And 3+3?"},
                        ],
                    },
                )
        finally:
            glaiss.hub.clients.discard(fake)

        prompts = [m for m in fake.messages if m.get("type") == "prompt"]
        assert len(prompts) == 2
        first, second = prompts
        assert first["is_diff"] is False
        assert second["is_diff"] is True
        assert second["repeated_len"] > 0
        # the added tail is the new turn (server strips the leading separator)
        assert second["added_text"].startswith("[assistant] 4")
        # full text is still present for expand/replay
        assert second["text"].endswith("And 3+3?")

    asyncio.run(main())
