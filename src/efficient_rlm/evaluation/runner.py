from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from efficient_rlm.config import RLMConfig
from efficient_rlm.evaluation.metrics import speedup
from efficient_rlm.llm.http import build_llm_client
from efficient_rlm.recursive.pipeline import RecursivePipeline


def run_comparison(text: str, config: RLMConfig, output_path: str | None = None) -> dict:
    sequential_config = RLMConfig(**{**config.__dict__, "execution_mode": "sequential"})
    parallel_config = RLMConfig(**{**config.__dict__, "execution_mode": "parallel"})

    sequential = RecursivePipeline(build_llm_client(sequential_config), sequential_config).run(text)
    parallel = RecursivePipeline(build_llm_client(parallel_config), parallel_config).run(text)
    result = {
        "sequential": asdict(sequential),
        "parallel": asdict(parallel),
        "speedup": speedup(sequential.wall_time_seconds, parallel.wall_time_seconds),
        "note": "Mock results measure orchestration overhead only unless a real provider is configured.",
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

