import random

import pytest

from tinyserve.paging import BlockPool, BlockTable, CapacityError


def test_allocation_is_atomic_under_pressure_and_invalid_free():
    pool = BlockPool(3)
    owned = pool.allocate(2)
    assert len(set(owned)) == 2
    with pytest.raises(CapacityError):
        pool.allocate(2)
    assert pool.free_count == 1
    with pytest.raises(ValueError):
        pool.free((owned[0], owned[0]))
    with pytest.raises(ValueError):
        pool.free((owned[0], 3))
    assert pool.free_count == 1
    pool.free(owned)
    assert pool.free_count == 3
    with pytest.raises(ValueError):
        pool.free(owned)


def test_reserve_translates_noncontiguous_pages():
    pool = BlockPool(4)
    start = sorted(pool.allocate(4))
    pool.free((start[1], start[3]))
    table = BlockTable(pool, block_size=4)
    table.reserve(4)
    first = table.block_ids
    table.reserve(7)
    assert set(table.block_ids) == {start[1], start[3]}
    assert table.block_ids[:1] == first
    one, two = table.block_ids
    assert table.slots(2, 5) == (one*4+2, one*4+3, two*4, two*4+1, two*4+2)
    with pytest.raises(IndexError):
        table.slots(7, 1)
    with pytest.raises(CapacityError):
        table.reserve(9)
    assert (table.block_ids, table.reserved_tokens, pool.free_count) == ((one, two), 7, 0)
    table.release()
    table.release()
    assert table.block_ids == () and table.reserved_tokens == 0
    pool.free((start[0], start[2]))
    assert pool.free_count == 4


def test_random_reserve_release_conserves_unique_blocks():
    rng = random.Random(4080)
    pool = BlockPool(23)
    tables = [BlockTable(pool, block_size=4) for _ in range(8)]
    for _ in range(150):
        table = rng.choice(tables)
        before = (table.reserved_tokens, table.block_ids, pool.free_count)
        if rng.randrange(3) == 0:
            table.release()
        else:
            target = rng.randrange(1, 32)
            try:
                table.reserve(target)
            except CapacityError:
                assert before == (table.reserved_tokens, table.block_ids, pool.free_count)
            else:
                assert table.reserved_tokens == max(target, before[0])
                assert table.block_ids[: len(before[1])] == before[1]
        allocated = [block for owner in tables for block in owner.block_ids]
        assert len(set(allocated)) == len(allocated)
        assert len(allocated) + pool.free_count == pool.capacity
    for table in tables:
        table.release()
    assert pool.free_count == pool.capacity
