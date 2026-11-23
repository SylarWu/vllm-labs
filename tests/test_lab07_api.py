import asyncio
import json

import anyio
import httpx

from tinyserve.api import ChatEvent, ChatRequest, EngineChatBackend, Message, create_app
from tinyserve.requests import TokenEvent


class DemoBackend:
    model = "toy"

    async def stream(self, request):
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "Hi"
        assert request.max_tokens == 4
        yield ChatEvent("你")
        yield ChatEvent("好", "stop")


def payload(**overrides):
    return {
        "model": "toy",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4,
    } | overrides


def test_chat_completion_and_sse_share_the_same_text_and_finish_reason():
    async def run():
        app = create_app(DemoBackend())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            plain = await client.post("/v1/chat/completions", json=payload())
            assert plain.status_code == 200
            data = plain.json()
            assert (data["object"], data["model"]) == ("chat.completion", "toy")
            assert data["choices"][0]["message"] == {"role": "assistant", "content": "你好"}
            assert data["choices"][0]["finish_reason"] == "stop"
            streamed = await client.post("/v1/chat/completions", json=payload(stream=True))
            assert streamed.status_code == 200
            assert streamed.headers["content-type"].startswith("text/event-stream")
            chunks = [line.removeprefix("data:").strip() for line in streamed.text.splitlines() if line.startswith("data:")]
            assert chunks[-1] == "[DONE]" and chunks.count("[DONE]") == 1
            frames = [json.loads(chunk) for chunk in chunks[:-1]]
            assert all(frame["object"] == "chat.completion.chunk" for frame in frames)
            choices = [frame["choices"][0] for frame in frames]
            assert "".join(choice["delta"].get("content") or "" for choice in choices) == "你好"
            assert choices[-1]["finish_reason"] == "stop"
    asyncio.run(run())


def test_http_validation_does_not_start_the_backend():
    class UnexpectedBackend:
        model = "toy"

        def stream(self, request):
            raise AssertionError(f"backend called for invalid input {request}")

    async def run():
        app = create_app(UnexpectedBackend(), max_output_tokens=8)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/v1/chat/completions", json=payload(model="missing"))).status_code == 404
            for invalid in (
                payload(max_tokens=0),
                payload(max_tokens=9),
                payload(messages=[]),
                payload(messages=[{"role": "bad", "content": "Hi"}]),
                payload(temperature=0.4),
            ):
                response = await client.post("/v1/chat/completions", json=invalid)
                assert response.status_code == 422, response.text
    asyncio.run(run())


def test_inflight_limit_recovers_when_request_finishes():
    class SlowBackend:
        model = "toy"

        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request):
            self.started.set()
            await self.release.wait()
            yield ChatEvent("ok", "length")

    async def run():
        backend = SlowBackend()
        app = create_app(backend, max_inflight=1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/v1/chat/completions", json=payload()))
            try:
                await asyncio.wait_for(backend.started.wait(), 2)
                blocked = await asyncio.wait_for(client.post("/v1/chat/completions", json=payload()), 2)
                assert blocked.status_code == 429
            finally:
                backend.release.set()
                done = await asyncio.wait_for(first, 2)
            assert done.status_code == 200
            assert (await client.post("/v1/chat/completions", json=payload())).status_code == 200
    asyncio.run(run())


def test_asgi_disconnect_closes_stream_and_restores_admission():
    async def run():
        class DropBackend:
            model = "toy"
            closed = False

            async def stream(self, request):
                try:
                    yield ChatEvent("hello")
                    await anyio.sleep_forever()
                finally:
                    with anyio.CancelScope(shield=True):
                        await anyio.sleep(0)
                        self.closed = True

        backend = DropBackend()
        app = create_app(backend, max_inflight=1)
        first_body = asyncio.Event()
        sent_request = False
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "scheme": "http", "method": "POST",
            "path": "/v1/chat/completions", "root_path": "", "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        }

        async def receive():
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {"type": "http.request", "body": json.dumps(payload(stream=True)).encode()}
            await first_body.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and b"hello" in message.get("body", b""):
                first_body.set()

        await asyncio.wait_for(app(scope, receive, send), 2)
        assert first_body.is_set() and backend.closed

        async def finish(request):
            yield ChatEvent("ok", "stop")

        backend.stream = finish
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/v1/chat/completions", json=payload())).status_code == 200
    asyncio.run(run())


class ToyTokenizer:
    def render(self, messages):
        return (ord(messages[-1].content[0]),)

    def decode(self, token_ids):
        return "".join(chr(token) for token in token_ids)


class ToyEngine:
    def __init__(self):
        self.active = {}
        self.cancelled = []
        self.peak_active = 0

    def submit(self, request):
        assert request.request_id not in self.active
        self.active[request.request_id] = [request.prompt_ids[0], request.max_new_tokens]
        self.peak_active = max(self.peak_active, len(self.active))

    def step(self):
        events = []
        for request_id, state in list(self.active.items()):
            state[1] -= 1
            reason = "length" if state[1] == 0 else None
            events.append(TokenEvent(request_id, state[0], reason))
            if reason:
                del self.active[request_id]
        return events

    def cancel(self, request_id):
        self.cancelled.append(request_id)
        return self.active.pop(request_id, None) is not None


def test_engine_backend_routes_concurrent_rows_and_cancels_early_close():
    async def run():
        engine = ToyEngine()
        backend = EngineChatBackend(engine, ToyTokenizer(), model="toy")

        async def ask(text):
            request = ChatRequest((Message("user", text),), 3)
            events = [event async for event in backend.stream(request)]
            assert events[-1].finish_reason == "length"
            return "".join(event.delta for event in events)

        assert await asyncio.gather(ask("A"), ask("B")) == ["AAA", "BBB"]
        assert engine.peak_active == 2
        stream = backend.stream(ChatRequest((Message("user", "C"),), 4))
        assert (await anext(stream)).delta == "C"
        await stream.aclose()
        assert len(engine.cancelled) == 1 and engine.active == {}
    asyncio.run(run())


def test_engine_backend_holds_incomplete_utf8_until_the_stable_prefix_grows():
    class SplitUtf8Tokenizer:
        pieces = {1: b"\xe4", 2: b"\xbd", 3: b"\xa0", 4: b"A"}

        def render(self, messages):
            return (0,)

        def decode(self, token_ids):
            return b"".join(self.pieces[token] for token in token_ids).decode(
                "utf-8", errors="replace"
            )

    class SplitUtf8Engine:
        def __init__(self):
            self.active = {}

        def submit(self, request):
            self.active[request.request_id] = 0

        def step(self):
            events = []
            for request_id, index in list(self.active.items()):
                token_id = index + 1
                self.active[request_id] = index + 1
                reason = "length" if token_id == 4 else None
                events.append(TokenEvent(request_id, token_id, reason))
                if reason:
                    del self.active[request_id]
            return events

        def cancel(self, request_id):
            return self.active.pop(request_id, None) is not None

    async def run():
        backend = EngineChatBackend(
            SplitUtf8Engine(), SplitUtf8Tokenizer(), model="toy"
        )
        request = ChatRequest((Message("user", "ignored"),), 4)
        events = [event async for event in backend.stream(request)]
        assert [event.delta for event in events] == ["", "", "你", "A"]
        assert "".join(event.delta for event in events) == "你A"
        assert events[-1].finish_reason == "length"

    asyncio.run(run())


def test_engine_backend_delivers_terminal_event_routed_by_a_peer_stream():
    class PeerEngine:
        def __init__(self):
            self.ids = {}
            self.steps = 0

        def submit(self, request):
            self.ids[request.prompt_ids[0]] = request.request_id

        def step(self):
            self.steps += 1
            if self.steps == 1:
                return [TokenEvent(self.ids[ord("B")], ord("B"))]
            return [
                TokenEvent(self.ids[ord("A")], ord("A")),
                TokenEvent(self.ids[ord("B")], ord("B"), "length"),
            ]

        def cancel(self, request_id):
            return False

    async def run():
        backend = EngineChatBackend(PeerEngine(), ToyTokenizer(), model="toy")
        stream_b = backend.stream(ChatRequest((Message("user", "B"),), 2))
        assert (await anext(stream_b)).delta == "B"

        stream_a = backend.stream(ChatRequest((Message("user", "A"),), 2))
        assert (await anext(stream_a)).delta == "A"

        final_b = await anext(stream_b)
        assert final_b.delta == "B" and final_b.finish_reason == "length"
        await stream_a.aclose()

    asyncio.run(run())


def test_engine_backend_task_cancellation_cleans_up_and_propagates():
    class IdleEngine:
        def __init__(self):
            self.active = {}
            self.cancelled = []

        def submit(self, request):
            self.active[request.request_id] = request

        def step(self):
            return []

        def cancel(self, request_id):
            self.cancelled.append(request_id)
            return self.active.pop(request_id, None) is not None

    async def run():
        engine = IdleEngine()
        backend = EngineChatBackend(engine, ToyTokenizer(), model="toy")
        stream = backend.stream(ChatRequest((Message("user", "A"),), 2))
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)

        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("stream swallowed task cancellation")

        assert len(engine.cancelled) == 1 and engine.active == {}
        assert backend.running_state == {}

    asyncio.run(run())


def test_engine_backend_backlogged_events_keep_their_own_deltas():
    class BacklogEngine:
        def __init__(self):
            self.ids = {}
            self.steps = 0

        def submit(self, request):
            self.ids[request.prompt_ids[0]] = request.request_id

        def step(self):
            self.steps += 1
            a_id = self.ids.get(ord("A"))
            b_id = self.ids[ord("B")]
            if self.steps == 1:
                return [TokenEvent(b_id, ord("B"))]
            if self.steps == 2:
                return [
                    TokenEvent(a_id, ord("A")),
                    TokenEvent(b_id, ord("B")),
                ]
            return [
                TokenEvent(a_id, ord("A")),
                TokenEvent(b_id, ord("B"), "length"),
            ]

        def cancel(self, request_id):
            return False

    async def run():
        backend = EngineChatBackend(BacklogEngine(), ToyTokenizer(), model="toy")
        stream_b = backend.stream(ChatRequest((Message("user", "B"),), 3))
        assert (await anext(stream_b)).delta == "B"

        stream_a = backend.stream(ChatRequest((Message("user", "A"),), 2))
        assert (await anext(stream_a)).delta == "A"
        assert (await anext(stream_a)).delta == "A"

        second_b = await anext(stream_b)
        final_b = await anext(stream_b)
        assert second_b.delta == "B" and second_b.finish_reason is None
        assert final_b.delta == "B" and final_b.finish_reason == "length"
        await stream_a.aclose()
        await stream_b.aclose()

    asyncio.run(run())
