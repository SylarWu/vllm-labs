from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from tinyserve.requests import FinishReason

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
        raise NotImplementedError("Lab 07: engine adapter and stream state")

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent, None]:
        raise NotImplementedError("Lab 07: submit, route token events, and cancel on close")
        yield ChatEvent("")  # Marks this stub as an async generator.


def create_app(
    backend: ChatBackend, *, max_inflight: int = 8, max_output_tokens: int = 128
) -> FastAPI:
    """Lab 07. Minimal OpenAI-style chat endpoint with SSE and admission limits."""
    raise NotImplementedError("Lab 07: FastAPI app and streaming lifecycle")
