from typing import Protocol

import torch
from torch import Tensor


class KVStore(Protocol):
    def write(self, layer: int, start_pos: int, key: Tensor, value: Tensor) -> None:
        """Append K/V [T, Hkv, D] at the current layer length."""
        ...

    def read(self, layer: int, length: int) -> tuple[Tensor, Tensor]:
        """Return K/V for positions [0, length), with shape [length, Hkv, D]."""
        ...


class ContiguousKV:
    def __init__(
        self,
        *,
        num_layers: int,
        capacity: int,
        kv_heads: int,
        head_dim: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Lab 02. Keep separate lengths for each layer."""
        raise NotImplementedError("Lab 02: allocate a bounded contiguous cache")

    def write(self, layer: int, start_pos: int, key: Tensor, value: Tensor) -> None:
        raise NotImplementedError("Lab 02: append contiguous K/V")

    def read(self, layer: int, length: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("Lab 02: read contiguous K/V")
