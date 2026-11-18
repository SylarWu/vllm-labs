"""Provided plumbing for local unquantized Qwen3 snapshots."""

import json
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import torch

from tinyserve.api import Message
from tinyserve.cache import ContiguousKV
from tinyserve.model import QwenConfig, TinyQwen3


def load_model(
    directory: str | Path, *, device: str, dtype: torch.dtype
) -> TinyQwen3:
    from safetensors.torch import load_file

    directory = Path(directory)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if (
        raw.get("model_type") != "qwen3"
        or raw.get("attention_bias")
        or raw.get("use_sliding_window")
        or raw.get("quantization_config")
    ):
        raise ValueError("expected an unquantized Qwen3 snapshot with dense, bias-free attention")
    config = QwenConfig(**{field.name: raw[field.name] for field in fields(QwenConfig) if field.name in raw})
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = ["model.safetensors"]
    state_dict = {}
    for name in shard_names:
        state_dict.update(load_file(str(directory / name), device="cpu"))
    return TinyQwen3.from_state_dict(config, state_dict).to(device=device, dtype=dtype).eval()


def make_contiguous_cache(model: TinyQwen3, capacity: int) -> ContiguousKV:
    weights = next(model.parameters())
    config = model.config
    return ContiguousKV(
        num_layers=config.num_hidden_layers,
        capacity=capacity,
        kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        device=weights.device,
        dtype=weights.dtype,
    )


class HfTokenizer:
    def __init__(self, directory: str | Path) -> None:
        from transformers import AutoTokenizer

        self.inner = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.eos_token_id = self.inner.eos_token_id

    def render(self, messages: Sequence[Message]) -> list[int]:
        return self.inner.apply_chat_template(
            [{"role": msg.role, "content": msg.content} for msg in messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.inner.decode(list(token_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)
