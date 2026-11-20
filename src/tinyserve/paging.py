from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


class CapacityError(RuntimeError):
    """The block pool cannot satisfy a reservation."""


class BlockPool:
    def __init__(self, capacity: int) -> None:
        """Lab 03. Physical block IDs are in [0, capacity)."""
        if capacity <= 0:
            raise CapacityError(f"Block pool cannot satisfy a {capacity} capacity reservation")
        self.capacity = capacity

        self.free_ids = [i for i in range(capacity)]
        self.allocated = [False for _ in range(capacity)]

    @property
    def free_count(self) -> int:
        return len(self.free_ids)

    def allocate(self, count: int) -> list[int]:
        """All-or-nothing. Raise CapacityError if available space is too small."""
        if count < 0 or count > self.free_count:
            raise CapacityError(f"Current BlockPool free block counts is {self.free_count}, can't allocate {count}")
        if count == 0:
            return []

        ret = [self.free_ids.pop() for _ in range(count)]

        for i in ret:
            self.allocated[i] = True

        return ret

    def free(self, block_ids: Iterable[int]) -> None:
        """All-or-nothing. Raise ValueError for duplicates or unowned blocks."""
        block_id_set = set()
        for block_id in block_ids:
            if block_id in block_id_set:
                raise ValueError(f"Duplicate Block ID: {block_id}")
            if not 0 <= block_id < self.capacity:
                raise ValueError(f"Block ID {block_id} is out of range")
            if not self.allocated[block_id]:
                raise ValueError(f"Block ID {block_id} is not owned")

            block_id_set.add(block_id)

        for block_id in block_id_set:
            self.allocated[block_id] = False
            self.free_ids.append(block_id)


class BlockTable:
    def __init__(self, pool: BlockPool, block_size: int) -> None:
        self.pool = pool
        self.block_size = block_size

        # logical allocated block id -> physical poll allocated block id
        self.allocated = []
        self.slot_counts = []

    @property
    def block_ids(self) -> tuple[int, ...]:
        return (*self.allocated,)

    @property
    def reserved_tokens(self) -> int:
        if len(self.allocated) == 0:
            return 0
        return (len(self.allocated) - 1) * self.block_size + self.slot_counts[-1]

    def reserve(self, total_tokens: int) -> None:
        """Grow to cover total_tokens. Existing mappings are stable."""
        if total_tokens <= 0:
            raise ValueError("reserve total tokens cannot less or equal zero")
        if total_tokens <= self.reserved_tokens:
            return

        total_block_counts = (total_tokens - 1) // self.block_size + 1
        already_allocated_block_count = len(self.allocated)

        if total_block_counts > already_allocated_block_count:
            new_allocated_block_ids = self.pool.allocate(total_block_counts - already_allocated_block_count)

            self.allocated.extend(new_allocated_block_ids)
            if len(self.slot_counts) > 0:
                self.slot_counts[-1] = self.block_size
            self.slot_counts.extend(
                [self.block_size for _ in range(total_block_counts - already_allocated_block_count)]
            )

        self.slot_counts[-1] = total_tokens - (self.block_size * (len(self.allocated) - 1))

    def slots(self, start: int, count: int) -> tuple[int, ...]:
        """Resolve logical positions to flat physical slots; out of range raises IndexError."""
        if start < 0 or start + count > self.reserved_tokens:
            raise IndexError(f"Slot {start} is out of range")
        if count < 0:
            raise ValueError(f"Count {count} is negative")

        ret = []

        for idx in range(start, start + count):
            block_id = self.allocated[idx // self.block_size]
            offset = idx % self.block_size
            ret.append(block_id * self.block_size + offset)

        return (*ret,)

    def release(self) -> None:
        """Return all owned blocks. Repeated release is safe."""
        if self.reserved_tokens == 0:
            return
        self.pool.free(self.allocated)
        self.allocated = []
        self.slot_counts = []


class PagedStorage:
    def __init__(
            self,
            *,
            num_layers: int,
            num_blocks: int,
            block_size: int,
            kv_heads: int,
            head_dim: int,
            device: str | torch.device = "cpu",
            dtype: torch.dtype | None = None,
    ) -> None:
        """Lab 05. Allocate two physical tensors [L, blocks, block_size, Hkv, D]."""
        raise NotImplementedError("Lab 05: allocate the shared K/V tensors")

    def view(self, table: BlockTable) -> PagedKV:
        """Return a request view sharing physical storage; default dtype is FP32."""
        raise NotImplementedError("Lab 05: bind a block table to the tensor pool")


class PagedKV:
    def __init__(self, storage: PagedStorage, table: BlockTable) -> None:
        raise NotImplementedError("Lab 05: per-request state over shared storage")

    def write(self, layer: int, start_pos: int, key: Tensor, value: Tensor) -> None:
        raise NotImplementedError("Lab 05: scatter K/V into physical slots")

    def read(self, layer: int, length: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("Lab 05: gather logically contiguous K/V")
