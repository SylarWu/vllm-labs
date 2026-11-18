import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from tests.toy import make_model
from tinyserve.model import apply_rope, causal_gqa, rms_norm


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_handles_large_half_values(dtype):
    x = torch.tensor([[512, -256, 128, 64], [0, 0, 0, 0]], dtype=dtype)
    weight = torch.tensor([0.5, 1.5, 2.0, 1.0], dtype=dtype)
    expected = F.rms_norm(x.float(), (4,), weight.float(), 1e-6).to(dtype)
    result = rms_norm(x, weight, 1e-6)
    assert result.dtype == dtype
    torch.testing.assert_close(result, expected)


def test_rope_rotates_split_halves():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]]])
    s0, c0 = math.sin(1.0), math.cos(1.0)
    s1, c1 = math.sin(0.1), math.cos(0.1)
    expected = x.clone()
    expected[0, 0, 1] = torch.tensor([c0 - 3*s0, 2*c1 - 4*s1, 3*c0 + s0, 4*c1 + 2*s1])
    torch.testing.assert_close(apply_rope(x, torch.tensor([0, 1]), 100.0), expected)


def test_gqa_matches_sdpa_and_ignores_future_tokens():
    random = torch.Generator().manual_seed(37)
    q = torch.randn(2, 4, 5, 8, generator=random)
    k = torch.randn(2, 2, 5, 8, generator=random)
    v = torch.randn(2, 2, 5, 8, generator=random)
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1), is_causal=True
    )
    torch.testing.assert_close(causal_gqa(q, k, v), expected, atol=1e-6, rtol=1e-5)
    v[:, :, 3:] += 100
    torch.testing.assert_close(causal_gqa(q, k, v)[:, :, :3], expected[:, :, :3])


@torch.inference_mode()
def test_qwen3_matches_frozen_fp32_logits():
    fixture = json.loads((Path(__file__).parent / "data" / "qwen3_gold.json").read_text())
    model = make_model()
    actual = model(torch.tensor(fixture["input_ids"], dtype=torch.long))
    torch.testing.assert_close(actual, torch.tensor(fixture["logits"]), atol=2e-5, rtol=2e-5)
