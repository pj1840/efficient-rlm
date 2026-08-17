from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from efficient_rlm.config import RLMConfig
from efficient_rlm.models import Metrics

Decision = Literal["ANSWER_DIRECTLY", "DECOMPOSE", "STOP"]


@dataclass(frozen=True)
class PolicyDecision:
    action: Decision
    reason: str


class RecursivePolicy:
    def decide(self, text: str, depth: int, metrics: Metrics, config: RLMConfig) -> PolicyDecision:
        raise NotImplementedError


class DeterministicPolicy(RecursivePolicy):
    def decide(self, text: str, depth: int, metrics: Metrics, config: RLMConfig) -> PolicyDecision:
        if metrics.calls >= config.max_calls:
            return PolicyDecision("STOP", f"max_calls={config.max_calls}")
        if config.max_wall_time_seconds is not None and metrics.elapsed() >= config.max_wall_time_seconds:
            return PolicyDecision("STOP", f"max_wall_time_seconds={config.max_wall_time_seconds}")
        words = len(text.split())
        if depth >= config.max_depth:
            return PolicyDecision("ANSWER_DIRECTLY", f"max_depth={config.max_depth}")
        if words <= config.min_chunk_words:
            return PolicyDecision("ANSWER_DIRECTLY", f"min_chunk_words={config.min_chunk_words}")
        if words <= config.chunk_size_words:
            return PolicyDecision("ANSWER_DIRECTLY", f"chunk_size_words={config.chunk_size_words}")
        return PolicyDecision("DECOMPOSE", f"words={words} exceeds chunk_size_words={config.chunk_size_words}")


class AdaptivePolicy(DeterministicPolicy):
    def decide(self, text: str, depth: int, metrics: Metrics, config: RLMConfig) -> PolicyDecision:
        hard = super().decide(text, depth, metrics, config)
        if hard.action != "DECOMPOSE":
            return hard
        remaining_calls = config.max_calls - metrics.calls
        if remaining_calls <= 2:
            return PolicyDecision("ANSWER_DIRECTLY", "low remaining call budget")
        words = len(text.split())
        if words < config.complexity_threshold_words and depth > 0:
            return PolicyDecision("ANSWER_DIRECTLY", "below adaptive complexity threshold")
        return hard


def build_policy(config: RLMConfig) -> RecursivePolicy:
    if config.policy == "adaptive":
        return AdaptivePolicy()
    return DeterministicPolicy()
