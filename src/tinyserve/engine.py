from __future__ import annotations

import torch
from torch import Tensor

from tinyserve.model import TinyQwen3
from tinyserve.paging import PagedKV, PagedStorage, BlockTable, BlockPool
from tinyserve.requests import Request, TokenEvent
from tinyserve.scheduler import Scheduler


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
        # model check
        if model is None:
            raise ValueError("Engine: Model must provided")
        self.model = model

        # cache parameter check
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("Engine: num_blocks and block_size must be > 0")
        self.num_blocks = num_blocks
        self.block_size = block_size

        # scheduler parameter check
        if max_active < 1:
            raise ValueError("Engine: max_active must be > 0")

        self.pool = BlockPool(self.num_blocks)
        self.storage = PagedStorage(
            num_layers=self.model.config.num_hidden_layers,
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            kv_heads=self.model.config.num_key_value_heads,
            head_dim=self.model.config.head_dim,
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        self.kv_cache: dict[str, PagedKV] = dict()

        self.scheduler = Scheduler(
            max_active,
            max_batch_tokens=max_batch_tokens,
            max_prefill_chunk=max_prefill_chunk,
            try_admit=self.on_try_admit,
            on_retire=self.on_retire,
        )

    @property
    def free_blocks(self) -> int:
        return self.pool.free_count

    @property
    def has_work(self) -> bool:
        return len(self.scheduler.active_ids) + len(self.scheduler.waiting_ids) > 0

    def on_try_admit(self, request: Request) -> bool:
        reserved_tokens = len(request.prompt_ids) + request.max_new_tokens - 1
        required_blocks = (reserved_tokens - 1) // self.block_size + 1

        if required_blocks > self.free_blocks:
            return False

        self._prepare_kv_cache(request)
        return True

    def on_retire(self, request_id: str) -> None:
        if request_id not in self.kv_cache.keys():
            return

        cache = self.kv_cache.pop(request_id)
        cache.table.release()

    def submit(self, request: Request) -> None:
        """Reject requests that cannot ever fit. Reserve on admission."""
        if request is None:
            raise ValueError("Engine: request must provided")
        if len(request.request_id) <= 0:
            raise ValueError("Engine: request must provide request_id")
        if len(request.prompt_ids) <= 0:
            raise ValueError("Engine: request prompt is empty")
        if request.max_new_tokens <= 0:
            raise ValueError("Engine: request requires token generation at least one")

        reserved_tokens = len(request.prompt_ids) + request.max_new_tokens - 1
        required_blocks = (reserved_tokens - 1) // self.block_size + 1
        if required_blocks > self.num_blocks:
            raise ValueError("Engine: request ask too much cache")

        self.scheduler.submit(request)

    def _prepare_kv_cache(self, request: Request):
        request_id = request.request_id

        if request_id not in self.kv_cache.keys():
            self.kv_cache[request_id] = self.storage.view(
                BlockTable(self.pool, self.block_size)
            )

        self.kv_cache[request_id].table.reserve(len(request.prompt_ids) + request.max_new_tokens - 1)

    def _prepare_input_ids(self, input_ids: tuple[int, ...]) -> Tensor:
        return torch.tensor([input_ids], dtype=torch.long, device=self.storage.device)

    def _sample_next_token(self, next_logits: Tensor) -> int:
        return torch.argmax(next_logits, dim=-1).item()

    @torch.inference_mode()
    def step(self) -> list[TokenEvent]:
        """Run the next iteration. Lab 06 batches all decode rows in one forward."""
        steps = self.scheduler.plan()

        events: list[TokenEvent] = []
        for step in steps:
            next_logits = self.model(
                input_ids=self._prepare_input_ids(step.input_ids),
                cache=self.kv_cache[step.request_id],
                start_pos=step.start_pos,
            )[:, -1, :]
            next_token = self._sample_next_token(next_logits)

            token_event = self.scheduler.complete(step=step, sampled_token=next_token)
            events.append(token_event)

        return events

    def cancel(self, request_id: str) -> bool:
        """Cancel at a tick boundary and release any reservation."""
        return self.scheduler.cancel(request_id)
