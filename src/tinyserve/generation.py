from collections.abc import Sequence

from tinyserve.cache import KVStore
from tinyserve.model import TinyQwen3


def greedy_generate(
    model: TinyQwen3,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    eos_token_id: int | None = None,
    cache: KVStore | None = None,
) -> list[int]:
    """Lab 02. Return only generated IDs, including EOS when emitted."""
    raise NotImplementedError("Lab 02: prefill, decode, and stop conditions")
