from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, List, Literal

from tinyserve.requests import Request, Step, TokenEvent


@dataclass
class StepResource:
    token_budget: int

    def take(self, require_tokens: int) -> bool:
        if self.token_budget - require_tokens >= 0:
            self.token_budget = self.token_budget - require_tokens
            return True
        else:
            return False

    def remain(self) -> int:
        return self.token_budget


@dataclass
class ActiveStatus:
    request_id: str

    phase: Literal["prefill", "decode"] = "prefill"

    processed_tokens: int = 0
    generated_tokens: int = 0

    last_token: int | None = None


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
            raise ValueError(f"max_active must be greater than zero. max_active: {max_active}")
        self.max_active = max_active

        if max_batch_tokens is not None and max_batch_tokens < self.max_active:
            raise ValueError(
                f"max_batch_tokens must be greater or equal to max_active."
                + f"max_batch_tokens: {max_batch_tokens}, max_active: {self.max_active}"
            )
        self.max_batch_tokens = max_batch_tokens
        if max_prefill_chunk is not None and max_prefill_chunk < 1:
            raise ValueError(f"max_prefill_chunk must be greater than 0.")
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

    def _pre_generate_step(self, request: Request,
                           status: ActiveStatus,
                           resource: StepResource | None = None) -> Step | None:
        start_pos: int = status.processed_tokens
        input_ids: tuple[int, ...]
        emit_token: bool
        if status.phase == "decode":

            if resource is not None and resource.remain() < 1:
                return None

            input_ids = (status.last_token,)
            emit_token = True
        else:
            allowed_chunk = len(request.prompt_ids) - start_pos
            if resource is not None:
                allowed_chunk = min(allowed_chunk, resource.remain())
            if self.max_prefill_chunk is not None:
                allowed_chunk = min(allowed_chunk, self.max_prefill_chunk)

            if allowed_chunk == 0:
                return None

            input_ids = request.prompt_ids[start_pos:start_pos + allowed_chunk]
            emit_token = start_pos + allowed_chunk >= len(request.prompt_ids)

        return Step(
            request_id=request.request_id,
            phase=status.phase,
            start_pos=start_pos,
            input_ids=input_ids,
            emit_token=emit_token,
        )

    def plan(self) -> list[Step]:
        """Admit at the tick boundary and plan at most one Step per active request."""
        # apply waiting to active
        while self._applicable():
            next_waiting_req_id = self.waiting.pop(0)
            self.running[next_waiting_req_id] = ActiveStatus(request_id=next_waiting_req_id)

        resource = None
        if self.max_batch_tokens is not None:
            resource = StepResource(self.max_batch_tokens)

        decode_req_ids = [req_id for req_id, status in self.running.items() if status.phase == "decode"]
        prefill_req_ids = [req_id for req_id, status in self.running.items() if status.phase == "prefill"]

        def filter_req_with_budget(req_ids: list[str]) -> list[Step]:
            steps = []
            for req_id in req_ids:
                request = self.requests[req_id]
                status = self.running[req_id]
                step = self._pre_generate_step(request, status, resource)

                if step is None:
                    continue

                if resource is not None:
                    resource.take(len(step.input_ids))
                steps.append(step)

            return steps

        return [*filter_req_with_budget(decode_req_ids), *filter_req_with_budget(prefill_req_ids)]

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
        if step.emit_token and sampled_token is None:
            raise ValueError("Scheduler: Sampled token is required but got None")
        if not step.emit_token and sampled_token is not None:
            raise ValueError("Scheduler: Sampled token is not required but got one")
        # update status
        status = self.running[req_id]
        status.processed_tokens = status.processed_tokens + len(step.input_ids)
        if step.emit_token:
            if status.phase == "prefill":
                status.phase = "decode"
            status.generated_tokens = status.generated_tokens + 1
            status.last_token = sampled_token

        # judge finish condition
        if not step.emit_token:
            return None

        if request.eos_token_id is not None and sampled_token is not None and sampled_token == request.eos_token_id:
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
