from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

ExecutionMode = Literal["sequential", "threaded", "parallel", "async"]
ProviderName = Literal["mock", "ollama", "openai_compatible"]
DecomposerName = Literal["fixed", "semantic"]
PolicyName = Literal["deterministic", "adaptive"]


@dataclass(frozen=True)
class RLMConfig:
    provider: ProviderName = "mock"
    model: str = "mock-rlm"
    endpoint: str | None = None
    api_key_env: str = "RLM_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 800
    timeout_seconds: float = 60.0
    max_retries: int = 1
    execution_mode: ExecutionMode = "threaded"
    workers: int = 4
    decomposer: DecomposerName = "fixed"
    policy: PolicyName = "deterministic"
    chunk_size_words: int = 80
    min_chunk_words: int = 20
    max_depth: int = 4
    max_children: int = 16
    max_calls: int = 128
    max_wall_time_seconds: float | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None
    complexity_threshold_words: int = 120
    enable_curriculum: bool = False
    results_dir: str = "results"
    debug: bool = False

    def validate(self) -> None:
        if self.execution_mode == "parallel":
            object.__setattr__(self, "execution_mode", "threaded")
        if self.execution_mode not in {"sequential", "threaded", "async"}:
            raise ValueError("execution_mode must be sequential, threaded, or async")
        if self.provider not in {"mock", "ollama", "openai_compatible"}:
            raise ValueError("provider must be mock, ollama, or openai_compatible")
        if self.decomposer not in {"fixed", "semantic"}:
            raise ValueError("decomposer must be fixed or semantic")
        if self.policy not in {"deterministic", "adaptive"}:
            raise ValueError("policy must be deterministic or adaptive")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.chunk_size_words < 1:
            raise ValueError("chunk_size_words must be >= 1")
        if self.min_chunk_words < 1:
            raise ValueError("min_chunk_words must be >= 1")
        if self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if self.max_children < 1:
            raise ValueError("max_children must be >= 1")
        if self.max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_wall_time_seconds is not None and self.max_wall_time_seconds <= 0:
            raise ValueError("max_wall_time_seconds must be > 0 when set")
        for field_name in ("max_prompt_tokens", "max_completion_tokens", "max_total_tokens"):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be >= 1 when set")


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def load_config(path: str | Path | None = None, **overrides: Any) -> RLMConfig:
    data: dict[str, Any] = {}
    if path:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"Invalid config line: {line}")
            key, raw_value = stripped.split(":", 1)
            data[key.strip()] = _parse_scalar(raw_value)

    data.update({key: value for key, value in overrides.items() if value is not None})
    config = replace(RLMConfig(), **data)
    config.validate()
    return config
