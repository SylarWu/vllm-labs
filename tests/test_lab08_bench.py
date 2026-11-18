import pytest

from tinyserve.bench import Arrival, BenchStats, RequestTiming, run_trace, summarize
from tinyserve.requests import Request, TokenEvent


def test_summary_counts_output_tokens_and_uses_request_and_token_populations():
    stats = summarize(
        [
            RequestTiming("a", 0.00, (0.02, 0.05, 0.08), 0.08),
            RequestTiming("b", 0.01, (0.07,), 0.07),
        ]
    )
    assert (stats.requests, stats.output_tokens) == (2, 4)
    assert stats.output_tokens_per_s == pytest.approx(50.0)
    assert (stats.ttft_p50_ms, stats.ttft_p95_ms) == pytest.approx((20, 60))
    assert (stats.itl_p50_ms, stats.itl_p95_ms) == pytest.approx((30, 30))


def test_summary_handles_empty_and_single_token_runs():
    assert summarize([]) == BenchStats(0, 0, 0.0, None, None, None, None)
    stats = summarize([RequestTiming("only", 1.0, (1.2,), 1.2)])
    assert stats.output_tokens_per_s == pytest.approx(5.0)
    assert stats.itl_p50_ms is None and stats.itl_p95_ms is None


def test_replay_observes_arrivals_and_sleeps_during_idle_time():
    class FakeClock:
        now = 10.0
        waits = []

        def read(self):
            return self.now

        def sleep(self, duration):
            assert duration > 0
            self.waits.append(duration)
            self.now += duration

    clock = FakeClock()

    class Engine:
        def __init__(self):
            self.active = {}

        @property
        def has_work(self):
            return bool(self.active)

        def submit(self, request):
            self.active[request.request_id] = request.max_new_tokens

        def step(self):
            clock.now += 0.01
            events = []
            for req, remaining in list(self.active.items()):
                reason = "length" if remaining == 1 else None
                events.append(TokenEvent(req, 2, reason))
                if reason:
                    del self.active[req]
                else:
                    self.active[req] = remaining - 1
            return events

    timings = run_trace(
        Engine(),
        [Arrival(0.0, Request("a", (1,), 3)), Arrival(0.015, Request("b", (1,), 1)),
         Arrival(0.10, Request("c", (1,), 1))],
        clock=clock.read,
        sleep=clock.sleep,
    )
    result = {timing.request_id: timing for timing in timings}
    assert result["a"].submitted_s == pytest.approx(10.00)
    assert result["a"].emitted_s == pytest.approx((10.01, 10.02, 10.03))
    assert result["b"].submitted_s == pytest.approx(10.02)
    assert result["b"].completed_s == pytest.approx(10.03)
    assert result["c"].emitted_s == pytest.approx((10.11,))
    assert clock.waits == pytest.approx([0.07])
