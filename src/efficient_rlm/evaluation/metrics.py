from __future__ import annotations


def speedup(sequential_seconds: float, parallel_seconds: float) -> float | None:
    if sequential_seconds <= 0 or parallel_seconds <= 0:
        return None
    return sequential_seconds / parallel_seconds

