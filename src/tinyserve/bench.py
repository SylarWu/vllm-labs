from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter, sleep as wait
from typing import TYPE_CHECKING

from tinyserve.requests import Request, TokenEvent

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


def calc_quantile(sorted_samples: list[int | float], q: float) -> int | float:
    return sorted_samples[math.ceil(len(sorted_samples) * q) - 1]


def summarize(samples: Sequence[RequestTiming]) -> BenchStats:
    """Lab 08. Use nearest-rank percentiles and pool all intra-request gaps."""
    total_request = len(samples)
    if total_request == 0:
        return BenchStats(0, 0, 0.0, None, None, None, None)

    total_token_count = 0
    early_t, latest_t = samples[0].submitted_s, samples[0].completed_s

    ttft_of_req = []
    itl_list = []

    for sample in samples:
        early_t = min(early_t, sample.submitted_s)
        latest_t = max(latest_t, sample.completed_s)
        total_token_count = total_token_count + len(sample.emitted_s)

        if len(sample.emitted_s) > 0:
            req_emitted_s = sample.emitted_s
            ttft_of_req.append(req_emitted_s[0] - sample.submitted_s)
            if len(req_emitted_s) > 1:
                for i in range(1, len(req_emitted_s)):
                    itl_list.append(req_emitted_s[i] - req_emitted_s[i - 1])

    output_tokens_per_s = total_token_count / (latest_t - early_t) if latest_t > early_t else 0.0

    ttft_of_req = sorted(ttft_of_req)
    itl_list = sorted(itl_list)

    ttft_p50_ms = calc_quantile(ttft_of_req, 0.5) * 1000 if len(ttft_of_req) > 0 else None
    ttft_p95_ms = calc_quantile(ttft_of_req, 0.95) * 1000 if len(ttft_of_req) > 0 else None
    itl_p50_ms = calc_quantile(itl_list, 0.5) * 1000 if len(itl_list) > 0 else None
    itl_p95_ms = calc_quantile(itl_list, 0.95) * 1000 if len(itl_list) > 0 else None

    return BenchStats(
        total_request, total_token_count, output_tokens_per_s,
        ttft_p50_ms, ttft_p95_ms, itl_p50_ms, itl_p95_ms,
    )


def run_trace(
        engine: Engine,
        trace: Sequence[Arrival],
        *,
        clock: Callable[[], float] = perf_counter,
        sleep: Callable[[float], None] = wait,
) -> list[RequestTiming]:
    """Lab 08. Replay a sorted trace and record observed token event times."""

    if len(trace) == 0:
        return []

    def on_submit(request_id: str, submit_s: float, states: dict[str, dict]):
        if request_id not in states:
            states[request_id] = {
                "submitted_s": submit_s,
                "emitted_s": list(),
                "completed_s": None,
            }

    def on_step(token_events: list[TokenEvent], step_s: float, states: dict[str, dict]):
        for event in token_events:
            state = states[event.request_id]
            if event.token_id is not None:
                state["emitted_s"].append(step_s)
            if event.finish_reason:
                state["completed_s"] = step_s

    replay_start = clock()
    replay_states = dict()

    idx = 0
    while idx < len(trace) or engine.has_work:
        if idx >= len(trace):
            # all request has been submitted, just engine step is ok
            events = engine.step()
            on_step(events, clock(), replay_states)
            continue
        # exist un-submit request, and engine may be still working
        # submit waiting batch
        i = idx
        current = clock()
        wait_for_submit = []
        while i < len(trace) and replay_start + trace[i].at_s <= current:
            wait_for_submit.append(trace[i])
            i = i + 1
        for r in wait_for_submit:
            engine.submit(r.request)
            on_submit(r.request.request_id, clock(), replay_states)
        idx = i

        if engine.has_work:
            events = engine.step()
            on_step(events, clock(), replay_states)
        else:
            # this means engine has no work, and no waiting request has been submitted
            # just sleep
            if idx >= len(trace):
                break
            current = clock()
            if replay_start + trace[idx].at_s > current:
                sleep(replay_start + trace[idx].at_s - current)

    timings = []
    for req_id, state in replay_states.items():
        if state["completed_s"] is None:
            raise ValueError(f"Request: {req_id} has not completed yet")
        timings.append(RequestTiming(
            request_id=req_id,
            submitted_s=state["submitted_s"],
            emitted_s=(*state["emitted_s"],),
            completed_s=state["completed_s"],
        ))
    return timings
