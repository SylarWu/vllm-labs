from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tinyserve.cache import KVStore


@dataclass(frozen=True)
class QwenConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool = True


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Lab 01. Normalize the last axis, accumulating in FP32 for half types."""
    original_dtype = x.dtype
    x_fp32 = x.float()

    rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = x_fp32 * torch.rsqrt(rms + eps)

    return (normalized * weight).to(dtype=original_dtype)


def apply_rope(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    """Lab 01/06. x is [B, H, T, D]. Positions are [T] or, in Lab 06, [B, T]."""
    original_dtype = x.dtype
    d = x.shape[-1]
    batch_size = x.shape[0]

    freq_indices = torch.arange(d // 2, device=x.device, dtype=torch.float32)
    inv_freq = theta ** (-2.0 * freq_indices / d)

    positions_fp32 = positions.to(device=x.device, dtype=torch.float32)
    if positions_fp32.ndim == 1:
        # [T] -> [B, T]
        positions_fp32 = positions_fp32.unsqueeze(0).expand(batch_size, -1)
    # [B, T, 1] * [d // 2] -> [B, T, d // 2]
    angles = positions_fp32.unsqueeze(-1) * inv_freq

    cos = angles.unsqueeze(1).cos()
    sin = angles.unsqueeze(1).sin()

    x_fp32 = x.float()

    x1 = x_fp32[..., :(d // 2)]
    x2 = x_fp32[..., (d // 2):]

    rotated_1 = x1 * cos - x2 * sin
    rotated_2 = x2 * cos + x1 * sin

    return torch.cat((rotated_1, rotated_2), dim=-1).to(dtype=original_dtype)


def causal_gqa(q: Tensor, k: Tensor, v: Tensor, *, query_start: int = 0) -> Tensor:
    """Lab 01/02. x is [B, H, T, D]. K/V begin at position zero."""
    raise NotImplementedError("Lab 01: causal grouped-query attention")


class TinyQwen3(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config

    @classmethod
    def from_state_dict(cls, config: QwenConfig, weights: Mapping[str, Tensor]) -> TinyQwen3:
        """Lab 01. Load tensors with HF Qwen3 names. A tied LM head may be absent."""
        raise NotImplementedError("Lab 01: Qwen3 layers and weight mapping")

    def forward(
            self, input_ids: Tensor, cache: KVStore | None = None, *, start_pos: int = 0
    ) -> Tensor:
        """Lab 01/02. [B, T] IDs -> [B, T, V] logits. With cache, B=1."""
        raise NotImplementedError("Lab 01: forward; Lab 02: cached forward")

    def decode_batch(
            self, input_ids: Tensor, caches: Sequence[KVStore], positions: Sequence[int]
    ) -> Tensor:
        """Lab 06. [B, 1] IDs -> [B, 1, V]. Each sequence has its own cache."""
        raise NotImplementedError("Lab 06: a real batched decode forward")
