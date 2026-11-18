from dataclasses import dataclass
from typing import Literal

FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int | None = None


@dataclass(frozen=True)
class Step:
    request_id: str
    phase: Literal["prefill", "decode"]
    start_pos: int
    input_ids: tuple[int, ...]
    emit_token: bool


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token_id: int
    finish_reason: FinishReason | None = None
