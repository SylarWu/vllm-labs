import pytest

from tinyserve.requests import Request, Step, TokenEvent
from tinyserve.scheduler import Scheduler


def test_short_request_joins_and_leaves_during_long_decode():
    scheduler = Scheduler(max_active=2)
    scheduler.submit(Request("A", (1, 2, 3), 4))
    first, = scheduler.plan()
    assert first == Step("A", "prefill", 0, (1, 2, 3), True)
    assert scheduler.complete(first, 11) == TokenEvent("A", 11)
    scheduler.submit(Request("B", (4,), 1))
    scheduler.submit(Request("C", (5, 6), 1))
    second = {step.request_id: step for step in scheduler.plan()}
    assert second == {
        "A": Step("A", "decode", 3, (11,), True),
        "B": Step("B", "prefill", 0, (4,), True),
    }
    assert scheduler.waiting_ids == ("C",)
    assert scheduler.complete(second["A"], 12) == TokenEvent("A", 12)
    assert scheduler.complete(second["B"], 13) == TokenEvent("B", 13, "length")
    third = {step.request_id: step for step in scheduler.plan()}
    assert third["A"] == Step("A", "decode", 4, (12,), True)
    assert third["C"] == Step("C", "prefill", 0, (5, 6), True)
    scheduler.complete(third["A"], 14)
    scheduler.complete(third["C"], 15)
    last, = scheduler.plan()
    assert scheduler.complete(last, 16) == TokenEvent("A", 16, "length")
    assert scheduler.active_ids == scheduler.waiting_ids == ()
    assert scheduler.plan() == []


def test_eos_takes_priority_at_output_limit_and_retires_once():
    retired = []
    scheduler = Scheduler(max_active=1, on_retire=retired.append)
    scheduler.submit(Request("id", (1, 2), 1, eos_token_id=9))
    step, = scheduler.plan()
    assert scheduler.complete(step, 9) == TokenEvent("id", 9, "stop")
    assert scheduler.cancel("id") is False
    assert retired == ["id"]


def test_cancel_active_and_queued_requests():
    scheduler = Scheduler(max_active=1)
    scheduler.submit(Request("active", (1, 2), 3))
    scheduler.submit(Request("queued", (3,), 3))
    active, = scheduler.plan()
    scheduler.complete(active, 8)
    assert scheduler.cancel("queued") is True
    assert scheduler.cancel("active") is True
    assert scheduler.cancel("missing") is False
    assert scheduler.plan() == []


def test_reject_bad_inputs_and_duplicate_live_ids():
    scheduler = Scheduler(max_active=1)
    with pytest.raises(ValueError):
        scheduler.submit(Request("empty", (), 1))
    with pytest.raises(ValueError):
        scheduler.submit(Request("zero", (1,), 0))
    scheduler.submit(Request("same", (1,), 2))
    with pytest.raises(ValueError):
        scheduler.submit(Request("same", (2,), 3))
    assert scheduler.waiting_ids == ("same",)


def test_admission_gate_preserves_fifo_and_retries_waiting():
    can_fit = set()
    scheduler = Scheduler(max_active=2, try_admit=lambda request: request.request_id in can_fit)
    scheduler.submit(Request("head", (1,), 1))
    scheduler.submit(Request("tail", (2,), 1))
    can_fit.add("tail")
    assert scheduler.plan() == []
    assert scheduler.waiting_ids == ("head", "tail")
    can_fit.add("head")
    assert [step.request_id for step in scheduler.plan()] == ["head", "tail"]
