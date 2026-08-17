from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str | None = None
    latency_seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    provider: str | None = None
    retries: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_response(self, prompt: str) -> LLMResponse:
        started = perf_counter()
        text = self.generate(prompt)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split())
        return LLMResponse(
            text=text,
            latency_seconds=perf_counter() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop",
        )
