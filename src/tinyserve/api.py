from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from tinyserve.requests import FinishReason, Request, TokenEvent

if TYPE_CHECKING:
    from fastapi import FastAPI

    from tinyserve.engine import Engine


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

        self.event_queue: dict[str, asyncio.Queue[TokenEvent]] = dict()
        self.running_state: dict[str, dict] = dict()

    def _init_running_state(self, request_id: str) -> (asyncio.Queue[ChatEvent], dict):
        if request_id not in self.running_state:
            self.running_state[request_id] = {
                "token_ids": [],
                "stable_text": "",
                "emitted_text": "",
                "finished": False,
            }
            self.event_queue[request_id] = asyncio.Queue()

        return self.event_queue[request_id], self.running_state[request_id]

    def _destroy_running_state(self, request_id: str) -> (asyncio.Queue[dict] | None, dict | None):
        return self.event_queue.pop(request_id, None), self.running_state.pop(request_id, None)

    def _prepare_token_event(self, request_id: str, event: TokenEvent):
        if request_id not in self.running_state:
            return
        queue, state = self._init_running_state(request_id)

        state["token_ids"].append(event.token_id)
        state["finished"] = event.finish_reason is not None

        text = self.tokenizer.decode(state["token_ids"])
        if len(text) >= 1:
            i = len(text) - 1
            while i >= 0 and text[i] == '�':
                i = i - 1
            if i >= 0:
                state["stable_text"] = text[:i + 1]

        queue.put_nowait(event)

    def _get_ready_delta(self, request_id: str) -> str:
        queue, state = self._init_running_state(request_id)

        stable_text = state["stable_text"]
        if stable_text == "":
            return ""
        else:
            emitted_text = state["emitted_text"]
            state["emitted_text"] = stable_text
            return stable_text[len(emitted_text):]

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

                    yield ChatEvent(
                        delta=self._get_ready_delta(req.request_id),
                        finish_reason=event.finish_reason,
                    )

                    if event.finish_reason:
                        return
                else:
                    # wait instant multiple request to batch
                    await asyncio.sleep(0)

                    token_events = self.engine.step()

                    for e in token_events:
                        self._prepare_token_event(e.request_id, e)
        finally:
            queue, state = self._destroy_running_state(req.request_id)
            if not state["finished"]:
                self.engine.cancel(req.request_id)


def create_app(
        backend: ChatBackend, *, max_inflight: int = 8, max_output_tokens: int = 128
) -> FastAPI:
    """Lab 07. Minimal OpenAI-style chat endpoint with SSE and admission limits."""
    raise NotImplementedError("Lab 07: FastAPI app and streaming lifecycle")
