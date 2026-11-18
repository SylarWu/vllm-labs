from collections import defaultdict

import pytest
import torch

from tests.toy import make_model
from tinyserve.engine import Engine
from tinyserve.generation import greedy_generate
from tinyserve.paging import BlockPool, BlockTable, PagedStorage
from tinyserve.requests import Request


def test_paged_store_gathers_fragmented_blocks_and_isolates_requests_and_layers():
    pool = BlockPool(4)
    storage = PagedStorage(num_layers=2, num_blocks=4, block_size=2, kv_heads=2, head_dim=3)
    taken = sorted(pool.allocate(4))
    pool.free(taken[::2])
    first = BlockTable(pool, 2)
    first.reserve(3)
    pool.free(taken[1::2])
    second = BlockTable(pool, 2)
    second.reserve(2)
    a, b = storage.view(first), storage.view(second)
    data = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) + 1
    a.write(0, 0, data[:2], -data[:2])
    a.write(0, 2, data[2:], -data[2:])
    a.write(1, 0, data[:1] * 10, data[:1] * 20)
    b.write(0, 0, data[:2] + 100, data[:2] + 200)
    torch.testing.assert_close(a.read(0, 3)[0], data)
    torch.testing.assert_close(a.read(0, 3)[1], -data)
    torch.testing.assert_close(a.read(1, 1)[0], data[:1] * 10)
    torch.testing.assert_close(b.read(0, 2)[1], data[:2] + 200)
    with pytest.raises((ValueError, IndexError)):
        a.read(1, 2)
    with pytest.raises((ValueError, IndexError)):
        a.write(0, 3, data[:1], data[:1])


def test_serving_matches_serial_token_ids_and_releases_all_pages():
    model = make_model()
    engine = Engine(model, num_blocks=15, block_size=2, max_active=3)
    requests = [Request("a", (7, 15, 4, 6), 5), Request("b", (1, 33, 6), 3)]
    expected = {
        request.request_id: greedy_generate(model, request.prompt_ids, request.max_new_tokens)
        for request in requests
    }
    for request in requests:
        engine.submit(request)
    generated = defaultdict(list)
    finished = {}
    for _ in range(15):
        if not engine.has_work:
            break
        for event in engine.step():
            generated[event.request_id].append(event.token_id)
            if event.finish_reason:
                finished[event.request_id] = event.finish_reason
    assert dict(generated) == expected
    assert finished == {"a": "length", "b": "length"}
    assert engine.free_blocks == 15 and not engine.has_work
    assert engine.step() == []


def test_small_pool_waits_then_makes_progress_and_cancel_frees_capacity():
    engine = Engine(make_model(), num_blocks=2, block_size=2, max_active=2)
    with pytest.raises(ValueError):
        engine.submit(Request("too-large", (1, 2, 3, 4, 5), 1))
    engine.submit(Request("a", (1, 2, 3), 2))
    engine.submit(Request("b", (4, 5, 6), 1))
    engine.submit(Request("queued-cancel", (7,), 1))
    assert engine.cancel("queued-cancel")
    assert engine.free_blocks == 2
    assert [event.request_id for event in engine.step()] == ["a"]
    assert engine.free_blocks == 0
    assert engine.cancel("a")
    assert engine.free_blocks == 2
    event, = engine.step()
    assert (event.request_id, event.finish_reason) == ("b", "length")
    assert engine.free_blocks == 2 and not engine.has_work
