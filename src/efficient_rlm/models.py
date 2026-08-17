from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter


@dataclass(frozen=True)
class Subtask:
    id: int
    text: str
    depth: int
    index: int

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class RLMResult:
    answer: str
    mode: str
    calls: int
    tasks: int
    max_depth_reached: int
    wall_time_seconds: float
    intermediate_results: list[str] = field(default_factory=list)


class Metrics:
    def __init__(self) -> None:
        self.started_at = perf_counter()
        self.calls = 0
        self.tasks = 0
        self.max_depth_reached = 0
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

    def elapsed(self) -> float:
        return perf_counter() - self.started_at
