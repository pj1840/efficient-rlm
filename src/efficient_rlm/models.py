from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class Subtask:
    id: int | str
    text: str
    depth: int
    index: int
    parent_id: str | None = None
    prompt: str | None = None
    estimated_difficulty: float | None = None
    dependencies: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class RLMResult:
    answer: str
    mode: str
    calls: int
    successful_calls: int
    failed_calls: int
    retries: int
    tasks: int
    max_depth_reached: int
    wall_time_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    decomposition_count: int = 0
    aggregation_count: int = 0
    budget_termination_reason: str | None = None
    trace_path: str | None = None
    intermediate_results: list[str] = field(default_factory=list)


class Metrics:
    def __init__(self) -> None:
        self.started_at = perf_counter()
        self.calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retries = 0
        self.tasks = 0
        self.max_depth_reached = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.has_token_metadata = False
        self.decomposition_count = 0
        self.aggregation_count = 0
        self.budget_termination_reason: str | None = None
        self._lock = Lock()

    def record_task(self, depth: int) -> None:
        with self._lock:
            self.tasks += 1
            self.max_depth_reached = max(self.max_depth_reached, depth)

    def record_call(self) -> None:
        with self._lock:
            self.calls += 1

    def reserve_call(self, max_calls: int) -> int:
        with self._lock:
            if self.calls >= max_calls:
                raise RuntimeError(f"max_calls={max_calls} exceeded")
            self.calls += 1
            return self.calls

    def reserve_request(
        self,
        max_calls: int,
        estimated_prompt_tokens: int,
        max_prompt_tokens: int | None = None,
        max_total_tokens: int | None = None,
    ) -> int:
        with self._lock:
            if self.calls >= max_calls:
                raise RuntimeError(f"max_calls={max_calls} exceeded")
            if max_prompt_tokens is not None and self.prompt_tokens + estimated_prompt_tokens > max_prompt_tokens:
                raise RuntimeError(f"max_prompt_tokens={max_prompt_tokens}")
            if max_total_tokens is not None and self.total_tokens + estimated_prompt_tokens > max_total_tokens:
                raise RuntimeError(f"max_total_tokens={max_total_tokens}")
            self.calls += 1
            self.prompt_tokens += estimated_prompt_tokens
            self.total_tokens += estimated_prompt_tokens
            self.has_token_metadata = True
            return self.calls

    def record_response(
        self,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        reserved_prompt_tokens: int | None = None,
    ) -> None:
        with self._lock:
            self.successful_calls += 1
            if prompt_tokens is not None:
                if reserved_prompt_tokens is None:
                    self.prompt_tokens += prompt_tokens
                    self.total_tokens += prompt_tokens
                else:
                    delta = prompt_tokens - reserved_prompt_tokens
                    self.prompt_tokens += delta
                    self.total_tokens += delta
                self.has_token_metadata = True
            if completion_tokens is not None:
                self.completion_tokens += completion_tokens
                self.total_tokens += completion_tokens
                self.has_token_metadata = True
            if total_tokens is not None and prompt_tokens is None and completion_tokens is None:
                self.total_tokens += total_tokens
                self.has_token_metadata = True

    def record_failure(self) -> None:
        with self._lock:
            self.failed_calls += 1

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_decomposition(self) -> None:
        with self._lock:
            self.decomposition_count += 1

    def record_aggregation(self) -> None:
        with self._lock:
            self.aggregation_count += 1

    def mark_budget_stop(self, reason: str) -> None:
        with self._lock:
            if self.budget_termination_reason is None:
                self.budget_termination_reason = reason

    def elapsed(self) -> float:
        return perf_counter() - self.started_at

    def token_summary(self) -> tuple[int | None, int | None, int | None]:
        if not self.has_token_metadata:
            return None, None, None
        return self.prompt_tokens, self.completion_tokens, self.total_tokens


@dataclass
class TraceNode:
    node_id: str
    parent_id: str | None
    depth: int
    task_preview: str
    status: str = "pending"
    start_time: float | None = None
    end_time: float | None = None
    latency_seconds: float | None = None
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    decomposed: bool = False
    child_node_ids: list[str] = field(default_factory=list)
    aggregation_status: str | None = None
    error: str | None = None
    stopping_reason: str | None = None
    output_preview: str | None = None


@dataclass
class Trace:
    run_id: str
    started_at: float
    config: dict[str, Any]
    nodes: dict[str, TraceNode] = field(default_factory=dict)
    final_answer_preview: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
