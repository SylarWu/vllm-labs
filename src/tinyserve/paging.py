from __future__ import annotations

from collections.abc import Iterable

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
        if num_layers < 1:
            raise ValueError(f"PagedStorage: Number of layers: {num_layers} is invalid")
        self.num_layers = num_layers
        if num_blocks < 1:
            raise ValueError(f"PagedStorage: Number of blocks: {num_blocks} is invalid")
        self.num_blocks = num_blocks
        if block_size < 1:
            raise ValueError(f"PagedStorage: Block size: {block_size} is invalid")
        self.block_size = block_size
        if kv_heads < 1:
            raise ValueError(f"PagedStorage: KV heads: {kv_heads} is invalid")
        self.kv_heads = kv_heads
        if head_dim < 1:
            raise ValueError(f"PagedStorage: Head dim: {head_dim} is invalid")
        self.head_dim = head_dim

        self.device = device
        self.dtype = dtype

        self.k_cache = torch.empty(
            (num_layers, num_blocks * block_size, kv_heads, head_dim),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        self.v_cache = torch.empty(
            (num_layers, num_blocks * block_size, kv_heads, head_dim),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )

    def view(self, table: BlockTable) -> PagedKV:
        """Return a request view sharing physical storage; default dtype is FP32."""
        return PagedKV(storage=self, table=table)


class PagedKV:
    def __init__(self, storage: PagedStorage, table: BlockTable) -> None:
        self.storage = storage
        self.table = table

        self.written_length = [0 for _ in range(storage.num_layers)]

    def _check_layer(self, layer: int) -> None:
        if layer < 0 or layer >= self.storage.num_layers:
            raise ValueError(f"PagedKV: Layer {layer} is invalid")

    def _write(self, layer: int, start_pos: int, key: Tensor, value: Tensor) -> None:
        token_length, num_heads, head_dim = key.shape

        slot_indexes = self.table.slots(start_pos, token_length)

        for idx, slot_idx in enumerate(slot_indexes):
            self.storage.k_cache[layer][slot_idx].copy_(key[idx])
            self.storage.v_cache[layer][slot_idx].copy_(value[idx])
        self.written_length[layer] = self.written_length[layer] + token_length

    def write(self, layer: int, start_pos: int, key: Tensor, value: Tensor) -> None:
        self._check_layer(layer)
        if start_pos < 0 or start_pos > self.table.reserved_tokens or start_pos != self.written_length[layer]:
            raise ValueError(f"PagedKV: Start position {start_pos} is invalid")
        if key is None or value is None:
            raise ValueError(f"PagedKV: Key or Value is empty")
        if len(key.shape) != 3 or len(value.shape) != 3:
            raise ValueError(
                f"PagedKV: Key or Value Shape is invalid. Key Shape: {key.shape}, Value Shape: {value.shape}"
            )
        token_length, num_heads, head_dim = key.shape
        if token_length != value.shape[0] or num_heads != value.shape[1] or head_dim != value.shape[2]:
            raise ValueError(
                f"PagedKV: Key Value Token Length is not aligned. Key Shape: {key.shape}, Value Shape: {value.shape}"
            )
        if token_length <= 0:
            raise ValueError(
                f"PagedKV: Not allow empty write. Key Shape: {key.shape}, Value Shape: {value.shape}"
            )
        if num_heads != self.storage.kv_heads:
            raise ValueError(
                f"PagedKV: Key-Value number of heads is mismatch. KV Shape: {key.shape}, Cache number of heads: {self.storage.kv_heads}"
            )
        if head_dim != self.storage.head_dim:
            raise ValueError(
                f"PagedKV: Key-Value head dim is mismatch. KV Shape: {key.shape}, Cache head dim: {self.storage.head_dim}"
            )

        self._write(layer, start_pos, key, value)

    def _read(self, layer: int, length: int) -> tuple[Tensor, Tensor]:
        slot_indexes = self.table.slots(0, length)

        k = torch.empty(
            (length, self.storage.kv_heads, self.storage.head_dim),
            device=self.storage.device,
            dtype=self.storage.dtype,
        )
        v = torch.empty(
            (length, self.storage.kv_heads, self.storage.head_dim),
            device=self.storage.device,
            dtype=self.storage.dtype,
        )
        for idx, slot_idx in enumerate(slot_indexes):
            k[idx].copy_(self.storage.k_cache[layer][slot_idx])
            v[idx].copy_(self.storage.v_cache[layer][slot_idx])
        return k, v

    def read(self, layer: int, length: int) -> tuple[Tensor, Tensor]:
        self._check_layer(layer)
        if length < 1 or length > self.written_length[layer]:
            raise ValueError(f"PagedKV: Length is invalid")

        return self._read(layer, length)
