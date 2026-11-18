from collections.abc import Callable

from tinyserve.requests import Request, Step, TokenEvent


class Scheduler:
    def __init__(
        self,
        max_active: int,
        *,
        max_batch_tokens: int | None = None,
        max_prefill_chunk: int | None = None,
        try_admit: Callable[[Request], bool] | None = None,
        on_retire: Callable[[str], None] | None = None,
    ) -> None:
        """Lab 04: iteration-level FIFO. Lab 06: decode first and a token budget."""
        raise NotImplementedError("Lab 04: scheduler state")

    @property
    def waiting_ids(self) -> tuple[str, ...]:
        raise NotImplementedError("Lab 04: waiting requests in FIFO order")

    @property
    def active_ids(self) -> tuple[str, ...]:
        raise NotImplementedError("Lab 04: active requests in admission order")

    def submit(self, request: Request) -> None:
        """Reject an empty prompt, nonpositive budget, or duplicate live ID."""
        raise NotImplementedError("Lab 04: submit a request")

    def plan(self) -> list[Step]:
        """Admit at the tick boundary and plan at most one Step per active request."""
        raise NotImplementedError("Lab 04: plan the next inference iteration")

    def complete(self, step: Step, sampled_token: int | None) -> TokenEvent | None:
        """Commit the step. Retire at EOS or output limit."""
        raise NotImplementedError("Lab 04: advance a request after a model step")

    def cancel(self, request_id: str) -> bool:
        """Cancel a waiting or active request between iterations."""
        raise NotImplementedError("Lab 04: cancel and retire a request")
