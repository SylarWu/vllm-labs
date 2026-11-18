import os

import pytest
import torch
from torch import nn

from tests.toy import CONFIG, ids, make_cache, make_model
from tinyserve.generation import greedy_generate
from tinyserve.model import causal_gqa


def test_contiguous_cache_is_layer_local_and_validates_before_writing():
    store = make_cache(capacity=3)
    key = torch.arange(2 * CONFIG.num_key_value_heads * CONFIG.head_dim, dtype=torch.float32)
    key = key.reshape(2, CONFIG.num_key_value_heads, CONFIG.head_dim)
    value = -key - 1
    store.write(0, 0, key, value)
    store.write(1, 0, value[:1], key[:1])
    with pytest.raises((IndexError, ValueError)):
        store.read(1, 2)
    with pytest.raises((IndexError, ValueError)):
        store.write(0, 2, key, value)
    with pytest.raises((IndexError, ValueError)):
        store.write(0, 0, key[:1], value[:1])
    store.write(0, 2, key[:1] + 100, value[:1] - 100)
    actual_key, actual_value = store.read(0, 3)
    torch.testing.assert_close(actual_key, torch.cat((key, key[:1] + 100)))
    torch.testing.assert_close(actual_value, torch.cat((value, value[:1] - 100)))
    torch.testing.assert_close(store.read(1, 1)[0], value[:1])


def test_gqa_uses_absolute_query_positions_for_chunk():
    random = torch.Generator().manual_seed(39)
    q = torch.randn(1, 4, 5, 8, generator=random)
    k = torch.randn(1, 2, 5, 8, generator=random)
    v = torch.randn(1, 2, 5, 8, generator=random)
    expected = causal_gqa(q, k, v)[:, :, 3:]
    actual = causal_gqa(q[:, :, 3:], k, v, query_start=3)
    torch.testing.assert_close(actual, expected)


@torch.inference_mode()
@pytest.mark.parametrize("parts", [(4, 1, 1), (1, 2, 3)])
def test_cached_logits_match_full_recomputation(parts):
    model = make_model()
    prompt = ids((10, 8, 1, 6, 42, 4))
    expected = model(prompt)
    cache = make_cache(capacity=prompt.shape[1])
    chunks = []
    cursor = 0
    for count in parts:
        chunks.append(model(prompt[:, cursor : cursor + count], cache, start_pos=cursor))
        cursor += count
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, atol=2e-5, rtol=2e-5)


def test_greedy_output_matches_with_and_without_cache():
    model = make_model()
    prompt = (7, 12, 6, 32)
    expected = greedy_generate(model, prompt, 7)
    actual = greedy_generate(model, prompt, 7, cache=make_cache())
    assert len(expected) == 7
    assert actual == expected


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0))
        self.calls = []

    def forward(self, input_ids, cache=None, *, start_pos=0):
        self.calls.append((input_ids.tolist(), start_pos))
        logits = torch.zeros((*input_ids.shape, 10))
        logits[..., 3] = 1.0
        return logits


def test_greedy_stops_on_eos_and_zero_budget_is_a_noop():
    model = FixedLogits()
    assert greedy_generate(model, (5, 6), 0, eos_token_id=3) == []
    assert model.calls == []
    assert greedy_generate(model, (5, 6), 8, eos_token_id=3) == [3]
    assert model.calls == [([[5, 6]], 0)]


@pytest.mark.real_model
@torch.inference_mode()
def test_real_qwen3_matches_hf_fp32():
    location = os.environ.get("QWEN_MODEL_DIR")
    if not location:
        pytest.skip("set QWEN_MODEL_DIR to a local Qwen3-0.6B snapshot")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from tinyserve.hf_io import load_model, make_contiguous_cache

    device = os.environ.get("QWEN_DEVICE", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; QWEN_DEVICE=cpu also works")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(location, device=device, dtype=torch.float32)
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        location, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()
    prompt = (198, 785, 506, 2853, 30)
    generated = greedy_generate(model, prompt, 4, cache=make_contiguous_cache(model, 9))
    assert len(generated) == 4
    tokens = torch.tensor([prompt + tuple(generated)], device=device)
    expected = reference(tokens, use_cache=False).logits
    actual = model(tokens)
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=1e-3)
    start = len(prompt) - 1
    assert expected[0, start : start + len(generated)].argmax(dim=-1).tolist() == generated
