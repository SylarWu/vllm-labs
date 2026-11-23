from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal, Protocol, AsyncIterable

import fastapi.responses
from fastapi import FastAPI, Body, HTTPException

from tinyserve.engine import Engine
from tinyserve.requests import FinishReason, Request, TokenEvent


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[Message, ...]
    max_tokens: int


@dataclass(frozen=True)
class ChatEvent:
    delta: str
    finish_reason: FinishReason | None = None


class ChatBackend(Protocol):
    model: str

    def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent, None]:
        ...


class Tokenizer(Protocol):
    def render(self, messages: Sequence[Message]) -> Sequence[int]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...


class EngineChatBackend:
    def __init__(
            self,
            engine: Engine,
            tokenizer: Tokenizer,
            *,
            model: str,
            eos_token_id: int | None = None,
    ) -> None:
        """Lab 07. Drive one engine cooperatively from all request streams."""
        self.model = model

        self.engine = engine

        self.tokenizer = tokenizer

        self.eos_token_id = eos_token_id

        self.event_queue: dict[str, asyncio.Queue[ChatEvent]] = dict()
        self.running_state: dict[str, dict] = dict()

    def _init_running_state(self, request_id: str) -> tuple[asyncio.Queue[ChatEvent], dict]:
        if request_id not in self.running_state:
            self.running_state[request_id] = {
                "token_ids": [],
                "queued_text": "",
                "finished": False,
            }
            self.event_queue[request_id] = asyncio.Queue()

        return self.event_queue[request_id], self.running_state[request_id]

    def _destroy_running_state(self, request_id: str) -> tuple[asyncio.Queue[ChatEvent] | None, dict | None]:
        return self.event_queue.pop(request_id, None), self.running_state.pop(request_id, None)

    def _prepare_token_event(self, request_id: str, event: TokenEvent):
        if request_id not in self.running_state:
            return
        queue, state = self._init_running_state(request_id)

        state["token_ids"].append(event.token_id)
        if not state["finished"] and event.finish_reason:
            state["finished"] = True

        def get_current_stable_text(token_ids: list[int]) -> str:
            text = self.tokenizer.decode(token_ids)
            if len(text) >= 1:
                i = len(text) - 1
                while i >= 0 and text[i] == '�':
                    i = i - 1
                if i >= 0:
                    return text[:i + 1]
            return ""

        stable_text = get_current_stable_text(state["token_ids"])

        delta = ""
        if stable_text == "":
            delta = ""
        else:
            delta = stable_text[len(state["queued_text"]):]
            state["queued_text"] = stable_text

        queue.put_nowait(ChatEvent(
            delta=delta,
            finish_reason=event.finish_reason,
        ))

    def _build_request(self, c_req: ChatRequest) -> Request:
        return Request(
            request_id=str(uuid.uuid4()),
            prompt_ids=(*self.tokenizer.render(c_req.messages),),
            max_new_tokens=c_req.max_tokens,
            eos_token_id=self.eos_token_id
        )

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent, None]:

        req = self._build_request(request)

        try:
            queue, state = self._init_running_state(req.request_id)

            self.engine.submit(req)

            while True:
                if not queue.empty():

                    event = queue.get_nowait()

                    yield event

                    if event.finish_reason:
                        return
                else:
                    # wait instant multiple request to batch
                    await asyncio.sleep(0)

                    token_events = self.engine.step()

                    for e in token_events:
                        self._prepare_token_event(e.request_id, e)
        finally:
            _, state = self._destroy_running_state(req.request_id)
            if state and not state["finished"]:
                self.engine.cancel(req.request_id)


def create_app(
        backend: ChatBackend, *, max_inflight: int = 8, max_output_tokens: int = 128
) -> FastAPI:
    """Lab 07. Minimal OpenAI-style chat endpoint with SSE and admission limits."""
    app = FastAPI()

    inflight = 0

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict = Body()) -> fastapi.responses.Response:
        # model check
        model = payload.get("model", None)
        if not (model is not None and isinstance(model, str)):
            raise HTTPException(422, f"Model {model} is required!")
        if not (model == backend.model):
            raise HTTPException(404, f"Model {model} not found!")
        # messages check
        messages_raw = payload.get("messages", None)
        if not (messages_raw is not None and isinstance(messages_raw, list) and len(messages_raw) > 0):
            raise HTTPException(422, f"Empty messages!")

        messages: list[Message] = []
        for message in messages_raw:
            if not (message is not None and isinstance(message, dict)):
                raise HTTPException(422, f"Invalid message!: Message {message}")
            role = message.get("role", None)
            content = message.get("content", None)
            if not (role is not None and isinstance(role, str) and role in ("system", "user", "assistant")):
                raise HTTPException(422, f"Invalid message!: Role {role}")
            if not (content is not None and isinstance(content, str) and len(content) > 0):
                raise HTTPException(422, f"Invalid message!: Content {content}")

            messages.append(Message(role, content))

        # max_output_tokens check
        max_tokens = payload.get("max_tokens", None)
        if not (max_tokens is not None and isinstance(max_tokens, int) and not isinstance(max_tokens, bool)):
            raise HTTPException(422, f"Invalid max_tokens!: MaxTokens {max_tokens}")
        if not 1 <= max_tokens <= max_output_tokens:
            raise HTTPException(422, f"Invalid max_tokens!: MaxTokens {max_tokens}")

        # temperature check
        temperature_v = payload.get("temperature", 0)
        if not (temperature_v is not None and not isinstance(temperature_v, bool)
                and (isinstance(temperature_v, int) or isinstance(temperature_v, float))
                and temperature_v == 0):
            raise HTTPException(422, f"Invalid temperature!: Temperature {temperature_v}")

        stream = payload.get("stream", False)
        if not (stream is not None and isinstance(stream, bool)):
            raise HTTPException(422, f"Invalid stream!: Stream {stream}")

        # inflight check
        nonlocal inflight
        if inflight >= max_inflight:
            raise HTTPException(429, "Too Many Requests!")

        inflight = inflight + 1

        req_id = "chatcmpl-" + str(uuid.uuid4().hex)
        req = ChatRequest(
            messages=(*messages,), max_tokens=max_tokens
        )
        req_created = int(time.time())

        async def normal_chat_completion(id: str, request: ChatRequest, created: int) -> fastapi.responses.JSONResponse:
            content = ""
            finish_reason = None

            async with aclosing(backend.stream(request)) as event_stream:
                async for event in event_stream:
                    content = content + event.delta
                    if event.finish_reason:
                        finish_reason = event.finish_reason

            return fastapi.responses.JSONResponse({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": backend.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": finish_reason
                    }
                ]
            }, 200)

        def stream_chat_completion(id: str, request: ChatRequest, created: int) -> fastapi.responses.StreamingResponse:
            async def body() -> AsyncIterable[str]:
                try:
                    async with aclosing(backend.stream(request)) as event_stream:
                        async for event in event_stream:
                            yield "data: " + json.dumps({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "model": backend.model,
                                "created": created,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": event.delta
                                        },
                                        "finish_reason": event.finish_reason,
                                    }
                                ]
                            }) + "\n\n"
                            if event.finish_reason:
                                yield "data: [DONE]\n\n"
                                return
                finally:
                    nonlocal inflight
                    inflight = inflight - 1

            return fastapi.responses.StreamingResponse(body(), status_code=200, media_type="text/event-stream")

        if stream:
            return stream_chat_completion(req_id, req, req_created)
        else:
            try:
                return await normal_chat_completion(req_id, req, req_created)
            finally:
                inflight = inflight - 1

    return app
