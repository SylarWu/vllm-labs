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

    def _prepare_kv_cache(self, request: Request) -> None:
        request_id = request.request_id

        if request_id not in self.kv_cache.keys():
            self.kv_cache[request_id] = self.storage.view(
                BlockTable(self.pool, self.block_size)
            )

        self.kv_cache[request_id].table.reserve(len(request.prompt_ids) + request.max_new_tokens - 1)

    def _sample_next_token(self, next_logits: Tensor) -> Tensor:
        return torch.argmax(next_logits, dim=-1)

    @torch.inference_mode()
    def step(self) -> list[TokenEvent]:
        """Run the next iteration. Lab 06 batches all decode rows in one forward."""
        steps = self.scheduler.plan()

        if len(steps) == 0:
            return []

        batch_size = len(steps)
        input_ids: Tensor
        caches = [self.kv_cache[step.request_id] for step in steps]
        positions = [step.start_pos for step in steps]
        events: list[TokenEvent] = []
        if steps[-1].phase == "decode":
            # pure decode
            token_length = 1

            input_ids = torch.zeros((batch_size, token_length), dtype=torch.long, device=self.storage.device)
            for idx, step in enumerate(steps):
                input_ids[idx][:] = step.input_ids[0]

            # [B, 1, V] -> [B, V]
            logits = self.model.decode_batch(input_ids=input_ids, caches=caches, positions=positions)[:, -1, :]
            # [B]
            next_token = self._sample_next_token(logits)

            for idx, step in enumerate(steps):
                event = self.scheduler.complete(step=step, sampled_token=next_token[idx].item())
                if event is not None:
                    events.append(event)
        else:
            # decode prefill mixture
            # [B, T, V]
            token_length = max([len(step.input_ids) for step in steps])
            actual_lengths = [len(step.input_ids) for step in steps]

            input_ids = torch.zeros((batch_size, token_length), dtype=torch.long, device=self.storage.device)
            for idx, step in enumerate(steps):
                input_ids[idx, :actual_lengths[idx]] = torch.tensor(
                    step.input_ids, dtype=torch.long, device=self.storage.device
                )

            # [B, T, V] -> [B, V]
            logits = self.model.forward_batch(
                input_ids=input_ids, caches=caches,
                positions=positions, actual_lengths=actual_lengths,
            )

            for idx, step in enumerate(steps):
                event = None
                next_token = None
                if step.emit_token:
                    logit = logits[idx, actual_lengths[idx] - 1, :]
                    next_token = self._sample_next_token(logit)
                event = self.scheduler.complete(
                    step=step,
                    sampled_token=next_token.item() if next_token is not None else next_token
                )
                if event is not None:
                    events.append(event)
        return events

    def cancel(self, request_id: str) -> bool:
        """Cancel at a tick boundary and release any reservation."""
        return self.scheduler.cancel(request_id)
