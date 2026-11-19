from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
    original_dtype = q.dtype
    on_device = q.device
    group_num = q.shape[1] // k.shape[1]

    d = q.shape[-1]
    assert d == k.shape[-1]
    assert d == v.shape[-1]

    t_q = q.shape[-2]
    t_kv = k.shape[-2]
    assert t_kv == v.shape[-2]

    # [B, H_q, T_q, D]
    q_fp32 = q.float()
    # [B, H_kv, T_kv, D] -> [B, H_q, T_kv, D]
    k_fp32 = k.float().repeat_interleave(group_num, dim=1)
    # [B, H_kv, T_kv, D] -> [B, H_q, T_kv, D]
    v_fp32 = v.float().repeat_interleave(group_num, dim=1)

    # [B, H_q, T_q, T_kv]
    scaled_dot = torch.matmul(q_fp32, k_fp32.transpose(-1, -2)) / math.sqrt(d)

    query_positions = query_start + torch.arange(t_q, device=on_device)
    kv_positions = torch.arange(t_kv, device=on_device)
    # [T_q, T_kv]
    allowed = query_positions.unsqueeze(1) >= kv_positions.unsqueeze(0)

    scaled_dot = torch.masked_fill(scaled_dot, ~allowed, float('-inf'))

    out = torch.matmul(torch.softmax(scaled_dot, dim=-1), v_fp32)

    # [B, H_q, T_q, D]
    return out.to(dtype=original_dtype)


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


class TinyQwen3Attention(nn.Module):
    def __init__(self, attn_idx: int, hidden_size: int, num_q_heads: int, num_kv_heads: int, head_dim: int,
                 rms_norm_eps: float, rope_theta: float):
        super().__init__()

        self.attn_idx = attn_idx

        self.hidden_size = hidden_size

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        self.w_q = nn.Linear(in_features=hidden_size, out_features=num_q_heads * head_dim, bias=False)
        self.w_k = nn.Linear(in_features=hidden_size, out_features=num_kv_heads * head_dim, bias=False)
        self.w_v = nn.Linear(in_features=hidden_size, out_features=num_kv_heads * head_dim, bias=False)
        self.w_out = nn.Linear(in_features=num_q_heads * head_dim, out_features=hidden_size, bias=False)

        self.q_rms_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_rms_norm = RMSNorm(head_dim, rms_norm_eps)

    def forward(self, input: Tensor, kv_cache: KVStore | None = None, query_start: int = 0) -> Tensor:
        batch_size, token_length, hidden_size = input.shape

        q = self.w_q(input).reshape(batch_size, token_length, self.num_q_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.w_k(input).reshape(batch_size, token_length, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.w_v(input).reshape(batch_size, token_length, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)

        positions = torch.arange(query_start, query_start + token_length, device=input.device)

        q = apply_rope(q, positions, self.rope_theta)
        k = apply_rope(k, positions, self.rope_theta)

        if kv_cache is not None:
            kv_cache.write(
                self.attn_idx, query_start,
                k.squeeze(0).permute(1, 0, 2), v.squeeze(0).permute(1, 0, 2),
            )
            cached_k, cached_v = kv_cache.read(self.attn_idx, query_start + token_length)
            k = cached_k.permute(1, 0, 2).unsqueeze(0)
            v = cached_v.permute(1, 0, 2).unsqueeze(0)

            assert k.shape[-2] == query_start + q.shape[-2]
            assert v.shape[-2] == query_start + q.shape[-2]

        # [B, H_q, T_q, D] -> [B, T_q, H_q, D] -> [B, T_q, H_q * D]
        attn = causal_gqa(q, k, v, query_start=query_start).permute(0, 2, 1, 3).reshape(batch_size, token_length, -1)
        # [B, T_q, hidden_size]
        return self.w_out(attn)


class TinyQwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)

        out = self.w_down(F.silu(gate) * up)
        return out


class TinyQwen3Decoder(nn.Module):
    def __init__(self, layer_idx: int, config: QwenConfig):
        super().__init__()

        self.layer_idx = layer_idx
        self.pre_attn_layer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = TinyQwen3Attention(
            attn_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
        )
        self.pre_mlp_layer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = TinyQwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(self, input: Tensor, kv_cache: KVStore | None = None, query_start: int = 0) -> Tensor:
        # [batch_size, token_length, hidden_size]
        x = self.pre_attn_layer_norm(input)
        x = self.attention(x, kv_cache, query_start) + input

        middle_x = x

        x = self.pre_mlp_layer_norm(x)
        out = self.mlp(x) + middle_x

        return out


class TinyQwen3(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config

        self.vocab = nn.Parameter(torch.randn(config.vocab_size, config.hidden_size))

        self.decoders = nn.ModuleList(
            TinyQwen3Decoder(i, config) for i in range(config.num_hidden_layers)
        )

        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.vocab

    @classmethod
    def from_state_dict(cls, config: QwenConfig, weights: Mapping[str, Tensor]) -> TinyQwen3:
        """Lab 01. Load tensors with HF Qwen3 names. A tied LM head may be absent."""
        model = cls(config)

        def copy_weight(target: nn.Parameter, name: str) -> None:
            if name not in weights:
                raise KeyError(f"missing weight: {name}")

            source = weights[name]
            if target.shape != source.shape:
                raise ValueError(
                    f"shape mismatch for {name}: "
                    f"expected {tuple(target.shape)}, "
                    f"got {tuple(source.shape)}"
                )

            target.copy_(source)

        with torch.no_grad():
            copy_weight(
                model.vocab,
                "model.embed_tokens.weight",
            )
            copy_weight(
                model.final_norm.weight,
                "model.norm.weight",
            )

            for index, decoder in enumerate(model.decoders):
                prefix = f"model.layers.{index}."

                copy_weight(
                    decoder.pre_attn_layer_norm.weight,
                    prefix + "input_layernorm.weight",
                )
                copy_weight(
                    decoder.pre_mlp_layer_norm.weight,
                    prefix + "post_attention_layernorm.weight",
                )

                attention = decoder.attention
                copy_weight(
                    attention.w_q.weight,
                    prefix + "self_attn.q_proj.weight",
                )
                copy_weight(
                    attention.w_k.weight,
                    prefix + "self_attn.k_proj.weight",
                )
                copy_weight(
                    attention.w_v.weight,
                    prefix + "self_attn.v_proj.weight",
                )
                copy_weight(
                    attention.w_out.weight,
                    prefix + "self_attn.o_proj.weight",
                )
                copy_weight(
                    attention.q_rms_norm.weight,
                    prefix + "self_attn.q_norm.weight",
                )
                copy_weight(
                    attention.k_rms_norm.weight,
                    prefix + "self_attn.k_norm.weight",
                )

                mlp = decoder.mlp
                copy_weight(
                    mlp.w_gate.weight,
                    prefix + "mlp.gate_proj.weight",
                )
                copy_weight(
                    mlp.w_up.weight,
                    prefix + "mlp.up_proj.weight",
                )
                copy_weight(
                    mlp.w_down.weight,
                    prefix + "mlp.down_proj.weight",
                )

            if not config.tie_word_embeddings:
                copy_weight(
                    model.lm_head.weight,
                    "lm_head.weight",
                )

        return model

    def _select_token(self, input_ids: Tensor) -> Tensor:
        return self.vocab[input_ids]

    def forward(
            self, input_ids: Tensor, cache: KVStore | None = None, *, start_pos: int = 0
    ) -> Tensor:
        """Lab 01/02. [B, T] IDs -> [B, T, V] logits. With cache, B=1."""
        x = self._select_token(input_ids)

        for decoder in self.decoders:
            x = decoder(x, cache, start_pos)

        x = self.final_norm(x)

        return self.lm_head(x)

    def decode_batch(
            self, input_ids: Tensor, caches: Sequence[KVStore], positions: Sequence[int]
    ) -> Tensor:
        """Lab 06. [B, 1] IDs -> [B, 1, V]. Each sequence has its own cache."""
        raise NotImplementedError("Lab 06: a real batched decode forward")
