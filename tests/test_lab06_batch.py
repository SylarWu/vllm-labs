from collections import defaultdict
from unittest.mock import patch

import pytest
import torch

from tests.toy import ids, make_cache, make_model
from tinyserve.engine import Engine
from tinyserve.generation import greedy_generate
from tinyserve.model import apply_rope, generate_attn_position_allowed
from tinyserve.requests import Request, Step, TokenEvent
from tinyserve.scheduler import Scheduler


def test_rope_can_use_different_positions_per_row():
    rng = torch.Generator().manual_seed(232)
    x = torch.randn(2, 3, 1, 8, generator=rng)
    positions = torch.tensor([[3], [7]])
    expected = torch.cat(
        [apply_rope(x[i : i + 1], positions[i], 10_000.0) for i in range(2)]
    )
    torch.testing.assert_close(apply_rope(x, positions, 10_000.0), expected)


def test_scheduler_prioritizes_decode_and_does_not_emit_before_last_prompt_chunk():
    scheduler = Scheduler(max_active=2, max_batch_tokens=4, max_prefill_chunk=3)
    scheduler.submit(Request("chat", (1, 2), 4))
    prefill, = scheduler.plan()
    scheduler.complete(prefill, 20)
    scheduler.submit(Request("long", (3, 4, 5, 6, 7, 8, 9), 1))
    for offset, expected_ids in [(0, (3, 4, 5)), (3, (6, 7, 8)), (6, (9,))]:
        decode, chunk = scheduler.plan()
        assert decode == Step("chat", "decode", 2 + offset // 3, (20 + offset // 3,), True)
        assert chunk == Step("long", "prefill", offset, expected_ids, offset == 6)
        assert len(decode.input_ids) + len(chunk.input_ids) <= 4
        scheduler.complete(decode, 21 + offset // 3)
        event = scheduler.complete(chunk, 31 if offset == 6 else None)
        expected = TokenEvent("long", 31, "length") if offset == 6 else None
        assert event == expected


def test_scheduler_rejects_budget_too_small_for_all_live_rows():
    with pytest.raises(ValueError):
        Scheduler(max_active=4, max_batch_tokens=3, max_prefill_chunk=2)


@torch.inference_mode()
def test_real_batch_logits_match_different_length_cached_and_full_rows():
    model = make_model()
    prompts = ((12, 4), (5, 2, 23, 9, 6))
    caches = [make_cache(), make_cache()]
    for prompt, cache in zip(prompts, caches):
        model(ids(prompt), cache)
    batch_ids = torch.tensor([[8], [17]])
    actual = model.decode_batch(batch_ids, caches, [len(prompt) for prompt in prompts])
    expected = torch.cat(
        [model(ids(prompt + (token,)))[:, -1:] for prompt, token in zip(prompts, (8, 17))]
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for prompt, token, cache in zip(prompts, (8, 17), caches):
        actual_next = model(ids((10,)), cache, start_pos=len(prompt) + 1)
        expected_next = model(ids(prompt + (token, 10)))[:, -1:]
        torch.testing.assert_close(actual_next, expected_next, atol=2e-5, rtol=2e-5)


def test_mixed_batch_mask_masks_future_and_uses_safe_zero_for_decode_padding():
    allowed = generate_attn_position_allowed(
        query_starts=[1, 0],
        actual_lengths=[1, 3],
        q_length=3,
        kv_length=3,
        on_device=torch.device("cpu"),
    )
    expected = torch.tensor(
        [
            [
                [True, True, False],
                [True, False, False],
                [True, False, False],
            ],
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
        ]
    )
    torch.testing.assert_close(allowed, expected)
    assert allowed.any(dim=-1).all()


@torch.inference_mode()
def test_forward_batch_mixed_lengths_match_full_recompute_and_preserve_next_decode():
    model = make_model()
    prompts = ((12, 4), (5, 2, 23), (7,))
    chunks = ((8,), (9, 6, 17), (11, 12))
    positions = [len(prompt) for prompt in prompts]
    actual_lengths = [len(chunk) for chunk in chunks]
    caches = [make_cache() for _ in prompts]
    for prompt, cache in zip(prompts, caches):
        model(ids(prompt), cache)

    padded_ids = torch.tensor([[8, 0, 0], [9, 6, 17], [11, 12, 0]])
    actual = model.forward_batch(
        padded_ids,
        caches,
        positions=positions,
        actual_lengths=actual_lengths,
    )

    assert torch.isfinite(actual).all()
    for b, (prompt, chunk, position, length, cache) in enumerate(
        zip(prompts, chunks, positions, actual_lengths, caches)
    ):
        expected = model(ids(prompt + chunk))[:, -length:]
        torch.testing.assert_close(actual[b : b + 1, :length], expected, atol=2e-5, rtol=2e-5)
        assert cache.write_length == [position + length] * model.config.num_hidden_layers

    for prompt, chunk, position, length, cache, next_token in zip(
        prompts, chunks, positions, actual_lengths, caches, (18, 19, 20)
    ):
        actual_next = model(ids((next_token,)), cache, start_pos=position + length)
        expected_next = model(ids(prompt + chunk + (next_token,)))[:, -1:]
        torch.testing.assert_close(actual_next, expected_next, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    ("positions", "actual_lengths"),
    [
        ([0, 0], [0, 2]),
        ([0, 0], [-1, 2]),
        ([0, 0], [1, 1]),
        ([-1, 0], [1, 2]),
        ([0], [1, 2]),
    ],
)
def test_forward_batch_rejects_invalid_metadata_before_writing_cache(positions, actual_lengths):
    model = make_model()
    caches = [make_cache(), make_cache()]
    with pytest.raises(ValueError):
        model.forward_batch(
            torch.tensor([[8, 0], [9, 10]]),
            caches,
            positions=positions,
            actual_lengths=actual_lengths,
        )
    assert all(cache.write_length == [0, 0] for cache in caches)


def test_decode_batch_rejects_multiple_tokens_per_row_before_writing_cache():
    model = make_model()
    caches = [make_cache(), make_cache()]
    with pytest.raises(ValueError):
        model.decode_batch(torch.tensor([[8, 0], [9, 10]]), caches, [0, 0])
    assert all(cache.write_length == [0, 0] for cache in caches)


def test_engine_calls_one_batched_decode_for_two_rows():
    model = make_model()
    engine = Engine(model, num_blocks=12, block_size=2, max_active=2)
    engine.submit(Request("a", (8, 7), 2))
    engine.submit(Request("b", (6, 5, 4, 3), 2))
    assert len(engine.step()) == 2
    with patch.object(model, "decode_batch", wraps=model.decode_batch) as batched:
        with patch.object(model, "forward", wraps=model.forward) as serial:
            events = engine.step()
    assert len(events) == 2
    batched.assert_called_once()
    serial.assert_not_called()
    assert all(type(event.token_id) is int for event in events)
    assert not engine.has_work


def test_engine_calls_one_forward_batch_for_mixed_decode_and_prefill():
    model = make_model()
    engine = Engine(
        model, num_blocks=16, block_size=2, max_active=2, max_batch_tokens=4, max_prefill_chunk=2
    )
    engine.submit(Request("chat", (1, 2), 3))
    engine.submit(Request("long", (3, 4, 5, 6, 7), 1))
    first_events = engine.step()
    assert [event.request_id for event in first_events] == ["chat"]

    with patch.object(model, "forward_batch", wraps=model.forward_batch) as batched:
        with patch.object(model, "forward", wraps=model.forward) as serial:
            with patch.object(model, "decode_batch", wraps=model.decode_batch) as decode:
                events = engine.step()

    batched.assert_called_once()
    serial.assert_not_called()
    decode.assert_not_called()
    assert batched.call_args.kwargs["input_ids"].shape == (2, 2)
    assert batched.call_args.kwargs["positions"] == [2, 2]
    assert batched.call_args.kwargs["actual_lengths"] == [1, 2]
    assert [event.request_id for event in events] == ["chat"]
    assert all(type(event.token_id) is int for event in events)


def test_chunked_engine_matches_full_recompute_and_cancelled_prefill_releases_pages():
    model = make_model()
    engine = Engine(
        model, num_blocks=16, block_size=2, max_active=2, max_batch_tokens=4, max_prefill_chunk=2
    )
    engine.submit(Request("cancel", (1, 2, 3, 4, 5), 3))
    assert engine.step() == []
    assert engine.cancel("cancel") and engine.free_blocks == 16
    requests = [Request("a", (6, 7, 8, 9, 11, 12), 3), Request("b", (10, 11, 12), 3)]
    outputs = defaultdict(list)
    for request in requests:
        engine.submit(request)
    for _ in range(16):
        if not engine.has_work:
            break
        for event in engine.step():
            outputs[event.request_id].append(event.token_id)
    for request in requests:
        assert outputs[request.request_id] == greedy_generate(
            model, request.prompt_ids, request.max_new_tokens
        )
    assert not engine.has_work and engine.free_blocks == 16
