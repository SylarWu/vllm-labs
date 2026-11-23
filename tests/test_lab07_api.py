import asyncio
import json
from contextlib import aclosing

import anyio
import httpx
import pytest
from starlette.requests import ClientDisconnect

from tests.toy import make_model
from tinyserve.api import ChatEvent, ChatRequest, EngineChatBackend, Message, create_app
from tinyserve.engine import Engine
from tinyserve.requests import TokenEvent


class AcceptingBackend:
    model = "toy"

    def is_applicable(self, request: ChatRequest) -> tuple[bool, str]:
        return True, ""


class UnexpectedBackend:
    model = "toy"

    def is_applicable(self, request: ChatRequest) -> tuple[bool, str]:
        raise AssertionError(f"backend checked invalid input {request}")

    def stream(self, request):
        raise AssertionError(f"backend opened a stream for invalid input {request}")


class DemoBackend(AcceptingBackend):
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
            assert isinstance(data["created"], int)
            assert data["choices"][0]["message"] == {"role": "assistant", "content": "你好"}
            assert data["choices"][0]["finish_reason"] == "stop"
            streamed = await client.post("/v1/chat/completions", json=payload(stream=True))
            assert streamed.status_code == 200
            assert streamed.headers["content-type"].startswith("text/event-stream")
            chunks = [line.removeprefix("data:").strip() for line in streamed.text.splitlines() if
                      line.startswith("data:")]
            assert chunks[-1] == "[DONE]" and chunks.count("[DONE]") == 1
            frames = [json.loads(chunk) for chunk in chunks[:-1]]
            assert all(frame["object"] == "chat.completion.chunk" for frame in frames)
            assert all(frame["model"] == "toy" for frame in frames)
            assert len({frame["id"] for frame in frames}) == 1
            assert len({frame["created"] for frame in frames}) == 1
            assert isinstance(frames[0]["created"], int)
            choices = [frame["choices"][0] for frame in frames]
            assert "".join(choice["delta"].get("content") or "" for choice in choices) == "你好"
            assert choices[-1]["finish_reason"] == "stop"

    asyncio.run(run())


def test_http_validation_does_not_start_the_backend():
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


def test_http_validation_rejects_missing_and_wrong_shaped_fields():
    def without(key):
        data = payload()
        del data[key]
        return data

    async def run():
        app = create_app(UnexpectedBackend())
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 缺少字段/字段类型错误是 422；未知模型字符串仍由上一测试验证为 404。
            invalid = (
                without("model"),
                without("messages"),
                without("max_tokens"),
                payload(model=None),
                payload(messages=None),
                payload(messages="user"),
                payload(messages=[None]),
                payload(messages=[{}]),
                payload(max_tokens=True),
                payload(stream=None),
                payload(temperature=None),
                payload(temperature=False),
            )
            for data in invalid:
                response = await client.post("/v1/chat/completions", json=data)
                assert response.status_code == 422, (
                    data, response.status_code, response.text
                )

    asyncio.run(run())


@pytest.mark.parametrize("rejected_count", [1, 2])
def test_inflight_limit_recovers_when_request_finishes(rejected_count):
    class SlowBackend(AcceptingBackend):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request):
            if request.messages[0].content == "hold":
                self.started.set()
                await self.release.wait()
            yield ChatEvent("ok", "length")

    async def run():
        backend = SlowBackend()
        app = create_app(backend, max_inflight=1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            held_request = payload(messages=[{"role": "user", "content": "hold"}])
            first = asyncio.create_task(client.post("/v1/chat/completions", json=held_request))
            try:
                await asyncio.wait_for(backend.started.wait(), 2)
                for _ in range(rejected_count):
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
        class DropBackend(AcceptingBackend):
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

        async def send(message):
            if message["type"] == "http.response.body" and b"hello" in message.get("body", b""):
                first_body.set()

        await call_sse_asgi(app, send, first_body)
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
        engine = BacklogEngine()
        backend = EngineChatBackend(engine, ToyTokenizer(), model="toy")
        stream_b = backend.stream(ChatRequest((Message("user", "B"),), 3))
        assert (await anext(stream_b)).delta == "B"

        stream_a = backend.stream(ChatRequest((Message("user", "A"),), 2))
        assert (await anext(stream_a)).delta == "A"
        assert (await anext(stream_a)).delta == "A"

        steps_before_drain = engine.steps
        second_b = await anext(stream_b)
        final_b = await anext(stream_b)
        assert second_b.delta == "B" and second_b.finish_reason is None
        assert final_b.delta == "B" and final_b.finish_reason == "length"
        assert engine.steps == steps_before_drain
        await stream_a.aclose()
        await stream_b.aclose()

    asyncio.run(run())


async def call_sse_asgi(app, send, disconnect, *, spec_version="2.3", data=None):
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1", "scheme": "http", "method": "POST",
        "path": "/v1/chat/completions", "root_path": "", "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    sent_request = False

    async def receive():
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {
                "type": "http.request",
                "body": json.dumps(payload(stream=True) if data is None else data).encode(),
            }
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async with asyncio.timeout(2):
        await app(scope, receive, send)


@pytest.mark.parametrize("wait_at", ["response-start", "before-first-event"])
def test_sse_holds_admission_before_first_event_and_releases_on_disconnect(wait_at):
    async def run():
        reached_wait = asyncio.Event()
        disconnect = asyncio.Event()

        class SlowBackend(AcceptingBackend):
            async def stream(self, request):
                if request.messages[0].content == "hold" and wait_at == "before-first-event":
                    reached_wait.set()
                    await anyio.sleep_forever()
                yield ChatEvent("ok", "stop")

        app = create_app(SlowBackend(), max_inflight=1)

        async def send(message):
            if message["type"] == "http.response.start" and wait_at == "response-start":
                reached_wait.set()
                await anyio.sleep_forever()

        data = payload(stream=True, messages=[{"role": "user", "content": "hold"}])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(call_sse_asgi(app, send, disconnect, data=data))
            try:
                await asyncio.wait_for(reached_wait.wait(), 2)
                blocked = await asyncio.wait_for(client.post("/v1/chat/completions", json=payload()), 2)
            finally:
                disconnect.set()
                await first
            recovered = await client.post("/v1/chat/completions", json=payload())
        assert blocked.status_code == 429
        assert recovered.status_code == 200

    asyncio.run(run())


class PromptLengthTokenizer:
    def render(self, messages):
        return (1,) * len(messages[0].content)

    def decode(self, token_ids):
        return "x" * len(token_ids)


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_sse_disconnect_during_send_closes_engine_request_before_response_returns(spec_version):
    async def run():
        engine = Engine(make_model(), num_blocks=6, block_size=2, max_active=1)
        backend = EngineChatBackend(engine, PromptLengthTokenizer(), model="toy")
        app = create_app(backend, max_inflight=1)
        disconnect = asyncio.Event()
        attempted_chunk = False

        async def send(message):
            nonlocal attempted_chunk
            if not message.get("body", b"").startswith(b"data: "):
                return
            attempted_chunk = True
            assert engine.has_work and engine.free_blocks < 6
            if spec_version == "2.4":
                raise BrokenPipeError("simulated closed client")
            disconnect.set()
            await anyio.sleep_forever()

        try:
            await call_sse_asgi(app, send, disconnect, spec_version=spec_version)
        except (ClientDisconnect, BrokenPipeError):
            pass

        # Record the state before the next await can run an async-generator finalizer.
        assert attempted_chunk
        assert not engine.has_work
        assert engine.free_blocks == 6
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/chat/completions", json=payload())
        assert response.status_code == 200

    asyncio.run(run())


@pytest.mark.parametrize("fail_at", ["open", "close"])
def test_sse_backend_failure_restores_admission(fail_at):
    class FailingBackend(AcceptingBackend):
        def __init__(self):
            self.failure_injected = False

        def stream(self, request):
            if request.messages[0].content == "fail" and fail_at == "open":
                self.failure_injected = True
                raise RuntimeError("stream creation failed")
            return self.events(request)

        async def events(self, request):
            try:
                yield ChatEvent("ok", "stop")
            finally:
                if request.messages[0].content == "fail":
                    self.failure_injected = True
                    raise RuntimeError("stream cleanup failed")

    async def run():
        backend = FailingBackend()
        app = create_app(backend, max_inflight=1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            bad_request = payload(stream=True, messages=[{"role": "user", "content": "fail"}])
            await client.post("/v1/chat/completions", json=bad_request)
            recovered = await client.post("/v1/chat/completions", json=payload())
        assert backend.failure_injected
        assert recovered.status_code == 200

    asyncio.run(run())


def test_engine_backend_stream_survives_prefill_tick_without_output():
    async def run():
        engine = Engine(make_model(), num_blocks=6, block_size=2, max_active=1, max_prefill_chunk=2)
        backend = EngineChatBackend(engine, PromptLengthTokenizer(), model="toy")
        request = ChatRequest((Message("user", "ABC"),), 2)
        async with asyncio.timeout(2), aclosing(backend.stream(request)) as stream:
            events = [event async for event in stream]
        assert [event.finish_reason for event in events] == [None, "length"]
        assert engine.free_blocks == 6

    asyncio.run(run())


def test_engine_backend_stream_waits_after_tick_that_only_advances_its_peer():
    async def run():
        engine = Engine(make_model(), num_blocks=6, block_size=2, max_active=1)
        backend = EngineChatBackend(engine, PromptLengthTokenizer(), model="toy")
        first = ChatRequest((Message("user", "A"),), 2)
        second = ChatRequest((Message("user", "B"),), 1)
        async with asyncio.timeout(2), aclosing(backend.stream(first)) as first_stream:
            assert (await anext(first_stream)).finish_reason is None
            # The second stream first drives the active request's terminal tick.
            async with aclosing(backend.stream(second)) as second_stream:
                assert (await anext(second_stream)).finish_reason == "length"
            assert (await anext(first_stream)).finish_reason == "length"
        assert engine.free_blocks == 6

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_http_rejects_prompt_that_can_never_fit_the_engine(stream):
    async def run():
        engine = Engine(make_model(), num_blocks=2, block_size=2, max_active=1)
        backend = EngineChatBackend(engine, PromptLengthTokenizer(), model="toy")
        app = create_app(backend, max_inflight=1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            too_large = payload(max_tokens=1, stream=stream, messages=[{"role": "user", "content": "12345"}])
            rejected = await client.post("/v1/chat/completions", json=too_large)
            assert not engine.has_work and engine.free_blocks == 2
            fits = payload(max_tokens=1, stream=stream, messages=[{"role": "user", "content": "1234"}])
            recovered = await client.post("/v1/chat/completions", json=fits)
        assert rejected.status_code == 422
        assert recovered.status_code == 200

    asyncio.run(run())


def test_engine_backend_delivers_two_first_events_from_one_tick():
    class CountingEngine(ToyEngine):
        def __init__(self):
            super().__init__()
            self.ticks = 0

        def step(self):
            self.ticks += 1
            return super().step()

    async def run():
        engine = CountingEngine()
        backend = EngineChatBackend(engine, ToyTokenizer(), model="toy")
        first = ChatRequest((Message("user", "A"),), 3)
        second = ChatRequest((Message("user", "B"),), 3)
        async with aclosing(backend.stream(first)) as first_stream:
            async with aclosing(backend.stream(second)) as second_stream:
                events = await asyncio.wait_for(
                    asyncio.gather(anext(first_stream), anext(second_stream)), 2
                )
                assert [event.delta for event in events] == ["A", "B"]
                assert all(event.finish_reason is None for event in events)
                assert engine.peak_active == 2
                assert engine.ticks == 1

    asyncio.run(run())


def test_sse_close_failure_after_disconnect_releases_admission():
    class CloseError(RuntimeError):
        pass

    class FailingCloseBackend(AcceptingBackend):
        close_attempted = False

        async def stream(self, request):
            if request.messages[0].content != "fail":
                yield ChatEvent("ok", "stop")
                return
            try:
                yield ChatEvent("first")
                await anyio.sleep_forever()
            finally:
                self.close_attempted = True
                raise CloseError("backend cleanup failed during explicit close")

    async def run():
        backend = FailingCloseBackend()
        app = create_app(backend, max_inflight=1)
        disconnect = asyncio.Event()

        async def send(message):
            if b"first" in message.get("body", b""):
                disconnect.set()
                await anyio.sleep_forever()

        bad_request = payload(stream=True, messages=[{"role": "user", "content": "fail"}])
        try:
            await call_sse_asgi(app, send, disconnect, data=bad_request)
        except CloseError:
            pass
        assert backend.close_attempted

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            recovered = await client.post("/v1/chat/completions", json=payload())
        assert recovered.status_code == 200

    asyncio.run(run())
