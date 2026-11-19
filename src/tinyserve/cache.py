from typing import Protocol

import torch
from torch import Tensor


class KVStore(Protocol):
    def write(self, layer: int, start_pos: int, key: Tensor, value: Tensor) -> None:
        """Append K/V [T, Hkv, D] at the current layer length."""

    def read(self, layer: int, length: int) -> tuple[Tensor, Tensor]:
        """Return K/V for positions [0, length), with shape [length, Hkv, D]."""


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
        if num_layers == 0:
            raise ValueError("not allowed empty layer")
        self.num_layers = num_layers
        if capacity == 0:
            raise ValueError("not allowed empty capacity")
        self.capacity = capacity
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.k_cache = torch.empty(num_layers, capacity, kv_heads, head_dim,
                                   device=device, dtype=dtype, requires_grad=False)
        self.v_cache = torch.empty(num_layers, capacity, kv_heads, head_dim,
                                   device=device, dtype=dtype, requires_grad=False)
        self.write_length = [0] * num_layers

    def _write(self, layer: int, start_pos: int, key: Tensor, value: Tensor) -> None:
        end_pos = start_pos + key.shape[0]

        self.k_cache[layer][start_pos: end_pos].copy_(key)
        self.v_cache[layer][start_pos: end_pos].copy_(value)
        self.write_length[layer] = end_pos

    def write(self, layer: int, start_pos: int, key: Tensor, value: Tensor) -> None:

        self._check_layer(layer)
        self._check_kv_shape_alignment(key.shape, value.shape)
        self._check_pos_length(start_pos, key.shape[0])

        if start_pos != self.write_length[layer]:
            raise IndexError(f"write must append from {self.write_length[layer]}, your start pos is {start_pos}")

        self._write(layer, start_pos, key, value)

    def _read(self, layer: int, length: int) -> tuple[Tensor, Tensor]:
        return self.k_cache[layer][:length], self.v_cache[layer][:length]

    def read(self, layer: int, length: int) -> tuple[Tensor, Tensor]:

        self._check_layer(layer)
        self._check_pos_length(0, length)

        if length == 0:
            raise ValueError("not allowed empty read")
        if not 0 <= length <= self.write_length[layer]:
            raise ValueError(f"read length {length} outside write cache [0, {self.write_length[layer]}]")

        return self._read(layer, length)

    def _check_layer(self, layer: int):
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"invalid layer index: {layer}, num_layers: {self.num_layers}")

    def _check_pos_length(self, start_pos: int, length: int):
        if not 0 <= start_pos < self.capacity:
            raise IndexError(f"invalid start position: {start_pos}, capacity: {self.capacity}")
        if not 0 <= start_pos + length <= self.capacity:
            raise IndexError(f"invalid end position: {start_pos + length}, capacity: {self.capacity}")

    def _check_kv_shape_alignment(self, k_shape: torch.Size, v_shape: torch.Size):
        if not len(k_shape) == 3:
            raise ValueError(f"invalid key tensor shape: {k_shape}")
        if not len(v_shape) == 3:
            raise ValueError(f"invalid value tensor shape: {v_shape}")

        for i in range(len(k_shape)):
            if k_shape[i] != v_shape[i]:
                raise ValueError(f"key value shape should be align key shape:{k_shape}, value shape:{v_shape}")

        if k_shape[0] == 0:
            raise ValueError(f"invalid key-value tensor shape: {k_shape}")
        if k_shape[1] != self.kv_heads:
            raise ValueError(
                f"key-value number of heads mismatch, Tensor heads:{k_shape[1]}, Config Heads:{self.kv_heads}"
            )
        if k_shape[2] != self.head_dim:
            raise ValueError(
                f"key-value head dim mismatch, Tensor dim:{k_shape[2]}, Config Heads:{self.head_dim}"
            )
