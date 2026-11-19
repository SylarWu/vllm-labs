from collections.abc import Sequence

import torch

from tinyserve.cache import KVStore
from tinyserve.model import TinyQwen3


def _argmax_next_token(next_logits: torch.Tensor) -> int:
    return next_logits.argmax(dim=-1).item()


@torch.inference_mode()
def greedy_generate(
        model: TinyQwen3,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        *,
        eos_token_id: int | None = None,
        cache: KVStore | None = None,
) -> list[int]:
    """Lab 02. Return only generated IDs, including EOS when emitted."""
    if max_new_tokens <= 0:
        return []

    device = next(model.parameters()).device

    start_pos = 0
    input_ids: torch.Tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = []

    # prefill
    next_logits = model(input_ids=input_ids, cache=cache, start_pos=start_pos)[:, -1, :]
    generated_ids.append(_argmax_next_token(next_logits))

    start_pos = start_pos + input_ids.shape[-1] if cache is not None else 0

    while len(generated_ids) < max_new_tokens and not (eos_token_id is not None and generated_ids[-1] == eos_token_id):
        if cache is None:
            input_ids = torch.cat(
                (input_ids, torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device),), dim=-1
            )
        else:
            input_ids = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)

        next_logits = model(input_ids=input_ids, cache=cache, start_pos=start_pos)[:, -1, :]

        generated_ids.append(_argmax_next_token(next_logits))
        start_pos = start_pos + input_ids.shape[-1] if cache is not None else 0

    return generated_ids
