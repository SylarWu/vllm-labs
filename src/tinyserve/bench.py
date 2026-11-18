from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter, sleep as wait
from typing import TYPE_CHECKING

from tinyserve.requests import Request

if TYPE_CHECKING:
    from tinyserve.engine import Engine


@dataclass(frozen=True)
class Arrival:
    at_s: float
    request: Request


@dataclass(frozen=True)
class RequestTiming:
    request_id: str
    submitted_s: float
    emitted_s: tuple[float, ...]
    completed_s: float


@dataclass(frozen=True)
class BenchStats:
    requests: int
    output_tokens: int
    output_tokens_per_s: float
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    itl_p50_ms: float | None
    itl_p95_ms: float | None


def summarize(samples: Sequence[RequestTiming]) -> BenchStats:
    """Lab 08. Use nearest-rank percentiles and pool all intra-request gaps."""
    raise NotImplementedError("Lab 08: summarize output throughput, TTFT and ITL")


def run_trace(
    engine: Engine,
    trace: Sequence[Arrival],
    *,
    clock: Callable[[], float] = perf_counter,
    sleep: Callable[[float], None] = wait,
) -> list[RequestTiming]:
    """Lab 08. Replay a sorted trace and record observed token event times."""
    raise NotImplementedError("Lab 08: replay arrivals against a real engine")
