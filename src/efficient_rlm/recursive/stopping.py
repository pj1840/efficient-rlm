from __future__ import annotations

from efficient_rlm.config import RLMConfig


def should_stop(text: str, depth: int, calls: int, config: RLMConfig) -> bool:
    words = len(text.split())
    return (
        depth >= config.max_depth
        or words <= config.min_chunk_words
        or words <= config.chunk_size_words
        or calls >= config.max_calls
    )

