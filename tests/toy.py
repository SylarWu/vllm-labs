from collections.abc import Iterable
from math import prod

import torch
from torch import Tensor

from tinyserve.cache import ContiguousKV
from tinyserve.model import QwenConfig, TinyQwen3

CONFIG = QwenConfig(
    vocab_size=43,
    hidden_size=24,
    intermediate_size=48,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    rms_norm_eps=1e-6,
    rope_theta=10_000.0,
)


def fixture_weights() -> dict[str, Tensor]:
    shapes: list[tuple[str, tuple[int, ...]]] = [
        ("model.embed_tokens.weight", (43, 24)),
        ("model.norm.weight", (24,)),
    ]
    for layer in range(CONFIG.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        shapes.extend(
            (prefix + key, shape)
            for key, shape in [
                ("input_layernorm.weight", (24,)),
                ("post_attention_layernorm.weight", (24,)),
                ("self_attn.q_proj.weight", (32, 24)),
                ("self_attn.k_proj.weight", (16, 24)),
                ("self_attn.v_proj.weight", (16, 24)),
                ("self_attn.o_proj.weight", (24, 32)),
                ("self_attn.q_norm.weight", (8,)),
                ("self_attn.k_norm.weight", (8,)),
                ("mlp.gate_proj.weight", (48, 24)),
                ("mlp.up_proj.weight", (48, 24)),
                ("mlp.down_proj.weight", (24, 48)),
            ]
        )
    weights = {}
    for seed, (name, shape) in enumerate(shapes, start=1):
        idx = torch.arange(prod(shape), dtype=torch.int64)
        code = (9_973 * idx * idx + 37 * idx * seed + 71 * seed * seed) % 8_191
        values = (code.to(torch.float32) - 4_095) / 32_768
        weights[name] = (values + (1 if len(shape) == 1 else 0)).reshape(shape)
    # As in a tied safetensors snapshot, lm_head.weight is deliberately absent.
    return weights


def make_model() -> TinyQwen3:
    return TinyQwen3.from_state_dict(CONFIG, fixture_weights()).eval()


def make_cache(capacity: int = 32) -> ContiguousKV:
    return ContiguousKV(
        num_layers=CONFIG.num_hidden_layers,
        capacity=capacity,
        kv_heads=CONFIG.num_key_value_heads,
        head_dim=CONFIG.head_dim,
    )


def ids(values: Iterable[int]) -> Tensor:
    return torch.tensor([list(values)], dtype=torch.long)
