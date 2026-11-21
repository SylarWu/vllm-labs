from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, List

from tinyserve.requests import Request, Step, TokenEvent


@dataclass
class ActiveStatus:
    request_id: str
    processed_tokens: int
    last_token: int | None
    generated_tokens: int

    def __init__(self, request_id: str,
                 processed_tokens: int = 0,
                 last_token: int | None = None,
                 generated_tokens: int = 0):
        self.request_id = request_id
        self.processed_tokens = processed_tokens
        self.last_token = last_token
        self.generated_tokens = generated_tokens


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
        if max_active <= 0:
            raise ValueError("max_active must be greater than zero")
        self.max_active = max_active

        self.max_batch_tokens = max_batch_tokens
        self.max_prefill_chunk = max_prefill_chunk
        self.try_admit = try_admit
        self.on_retire = on_retire

        self.waiting: List[str] = []

        self.requests: Dict[str, Request] = dict()
        self.running: Dict[str, ActiveStatus] = dict()

    @property
    def waiting_ids(self) -> tuple[str, ...]:
        return (*self.waiting,)

    @property
    def active_ids(self) -> tuple[str, ...]:
        return (*self.running.keys(),)

    def _submit(self, request: Request) -> None:

        request_id = request.request_id

        self.requests[request_id] = request
        self.waiting.append(request_id)

    def submit(self, request: Request) -> None:
        """Reject an empty prompt, nonpositive budget, or duplicate live ID."""
        if request is None:
            raise ValueError("Request cannot be None")
        if request.request_id is None or request.request_id == "":
            raise ValueError("Request id cannot be empty")
        if request.prompt_ids is None or len(request.prompt_ids) == 0:
            raise ValueError("Request prompts cannot be empty")
        if request.max_new_tokens <= 0:
            raise ValueError("Request requires token generation at least one")
        if request.request_id in self.requests.keys():
            raise ValueError(f"Request:{request.request_id} is already in-process")

        self._submit(request)

    def _applicable(self) -> bool:
        if len(self.waiting) == 0:
            return False
        if len(self.running) >= self.max_active:
            return False
        next_waiting_request = self.requests[self.waiting[0]]
        if self.try_admit is None:
            return True

        if self.try_admit(next_waiting_request):
            return True
        else:
            return False

    def _generate_step(self, request: Request, status: ActiveStatus) -> Step:
        req_id = request.request_id
        if status.last_token is None:
            return Step(
                request_id=req_id,
                phase="prefill",
                start_pos=0,
                input_ids=request.prompt_ids,
                emit_token=True,
            )
        else:
            return Step(
                request_id=req_id,
                phase="decode",
                start_pos=status.processed_tokens,
                input_ids=(status.last_token,),
                emit_token=True,
            )

    def plan(self) -> list[Step]:
        """Admit at the tick boundary and plan at most one Step per active request."""
        while self._applicable():
            next_waiting_req_id = self.waiting.pop(0)
            self.running[next_waiting_req_id] = ActiveStatus(next_waiting_req_id)

        return [self._generate_step(
            self.requests[req_id], self.running[req_id]
        ) for req_id in self.running.keys()]

    def _retire(self, req_id: str) -> None:
        if req_id not in self.requests.keys():
            return

        self.requests.pop(req_id)
        if req_id in self.waiting:
            self.waiting.remove(req_id)
        if req_id in self.running:
            self.running.pop(req_id)

        if self.on_retire is not None:
            self.on_retire(req_id)

    def complete(self, step: Step, sampled_token: int | None) -> TokenEvent | None:
        """Commit the step. Retire at EOS or output limit."""
        req_id = step.request_id
        request = self.requests[req_id]
        if sampled_token is None:
            raise ValueError("Sampled token cannot be None")
        # update status
        status = self.running[req_id]
        status.processed_tokens = status.processed_tokens + len(step.input_ids)
        status.generated_tokens = status.generated_tokens + (1 if sampled_token is not None else 0)
        status.last_token = sampled_token
        # judge finish condition

        if sampled_token == request.eos_token_id:
            self._retire(req_id)
            return TokenEvent(request_id=req_id, token_id=sampled_token, finish_reason="stop")
        if status.generated_tokens >= request.max_new_tokens:
            self._retire(req_id)
            return TokenEvent(request_id=req_id, token_id=sampled_token, finish_reason="length")
        return TokenEvent(request_id=req_id, token_id=sampled_token, finish_reason=None)

    def cancel(self, request_id: str) -> bool:
        """Cancel a waiting or active request between iterations."""
        if request_id not in self.requests.keys():
            return False
        self._retire(request_id)
        return True
