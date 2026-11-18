from __future__ import annotations

from tinyserve.model import TinyQwen3
from tinyserve.requests import Request, TokenEvent


class Engine:
    def __init__(
        self,
        model: TinyQwen3,
        *,
        num_blocks: int,
        block_size: int,
        max_active: int,
        max_batch_tokens: int | None = None,
        max_prefill_chunk: int | None = None,
    ) -> None:
        """Lab 05. Own one model, scheduler, page pool and paged tensor store."""
        raise NotImplementedError("Lab 05: connect scheduling and model execution")

    @property
    def free_blocks(self) -> int:
        raise NotImplementedError("Lab 05: expose current page availability")

    @property
    def has_work(self) -> bool:
        raise NotImplementedError("Lab 05: active or waiting work")

    def submit(self, request: Request) -> None:
        """Reject requests that cannot ever fit. Reserve on admission."""
        raise NotImplementedError("Lab 05: validate and enqueue")

    def step(self) -> list[TokenEvent]:
        """Run the next iteration. Lab 06 batches all decode rows in one forward."""
        raise NotImplementedError("Lab 05: execute planned steps")

    def cancel(self, request_id: str) -> bool:
        """Cancel at a tick boundary and release any reservation."""
        raise NotImplementedError("Lab 05: cancel active or waiting work")
