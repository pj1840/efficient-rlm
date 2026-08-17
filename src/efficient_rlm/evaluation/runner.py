from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from efficient_rlm.config import RLMConfig
from efficient_rlm.evaluation.evaluators import evaluate_answer
from efficient_rlm.evaluation.metrics import speedup
from efficient_rlm.evaluation.suite import load_suite
from efficient_rlm.llm.http import build_llm_client
from efficient_rlm.recursive.aggregation import build_chunk_summary_prompt
from efficient_rlm.recursive.pipeline import RecursivePipeline

RESULT_SCHEMA_VERSION = "2.0"


def run_direct(text: str, config: RLMConfig) -> dict:
    client = build_llm_client(config)
    response = client.generate_response(build_chunk_summary_prompt(text, index=0, total=1))
    return {
        "answer": response.text,
        "mode": "direct",
        "calls": 1,
        "successful_calls": 1,
        "failed_calls": 0,
        "retries": response.retries,
        "tasks": 1,
        "max_depth_reached": 0,
        "wall_time_seconds": response.latency_seconds,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
        "decomposition_count": 0,
        "aggregation_count": 0,
        "budget_termination_reason": None,
    }


def _config_for_mode(config: RLMConfig, mode: str) -> RLMConfig:
    values = dict(config.__dict__)
    if mode == "direct":
        return RLMConfig(**values)
    if mode in {"sequential", "threaded", "async"}:
        values["execution_mode"] = mode
        return RLMConfig(**values)
    if mode == "adaptive":
        values["execution_mode"] = "threaded"
        values["policy"] = "adaptive"
        return RLMConfig(**values)
    if mode == "adaptive_scheduled":
        values["execution_mode"] = "threaded"
        values["policy"] = "adaptive"
        values["enable_curriculum"] = True
        return RLMConfig(**values)
    raise ValueError(f"unknown benchmark mode: {mode}")


def run_mode(text: str, config: RLMConfig, mode: str) -> dict:
    mode_config = _config_for_mode(config, mode)
    if mode == "direct":
        return run_direct(text, mode_config)
    pipeline = RecursivePipeline(build_llm_client(mode_config), mode_config)
    result = (
        pipeline.run_curriculum_summary(text)
        if mode_config.enable_curriculum
        else pipeline.run(text)
    )
    data = asdict(result)
    data["mode"] = mode
    return data


def _benchmark_config(config: RLMConfig) -> dict[str, Any]:
    return {
        "provider": config.provider,
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "decomposer": config.decomposer,
        "policy": config.policy,
        "chunk_size_words": config.chunk_size_words,
        "min_chunk_words": config.min_chunk_words,
        "max_depth": config.max_depth,
        "max_children": config.max_children,
        "max_calls": config.max_calls,
        "max_wall_time_seconds": config.max_wall_time_seconds,
        "max_prompt_tokens": config.max_prompt_tokens,
        "max_completion_tokens": config.max_completion_tokens,
        "max_total_tokens": config.max_total_tokens,
        "complexity_threshold_words": config.complexity_threshold_words,
        "workers": config.workers,
    }


def _mode_config(config: RLMConfig, mode: str) -> dict[str, Any]:
    mode_config = _config_for_mode(config, mode)
    return {
        "provider": mode_config.provider,
        "model": mode_config.model,
        "decomposer": mode_config.decomposer,
        "policy": mode_config.policy,
        "execution_mode": mode_config.execution_mode,
        "enable_curriculum": mode_config.enable_curriculum,
        "workers": mode_config.workers,
    }


def _warmup(config: RLMConfig) -> dict[str, Any]:
    prompt = "Warmup request for local benchmark timing. Reply with one short sentence."
    response = build_llm_client(config).generate_response(prompt)
    return {
        "performed": True,
        "provider": config.provider,
        "model": response.model or config.model,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
        "included_in_aggregates": False,
    }


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "stdev": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_records(records: list[dict]) -> dict[str, Any]:
    by_mode: dict[str, list[dict]] = {}
    by_task_mode: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        mode = str(record["mode"])
        by_mode.setdefault(mode, []).append(record)
        by_task_mode.setdefault((str(record["task_id"]), mode), []).append(record)

    modes: dict[str, dict[str, Any]] = {}
    for mode, mode_records in sorted(by_mode.items()):
        metrics = [record.get("metrics", {}) for record in mode_records if not record.get("error")]
        scores = [float(record["evaluation"]["score"]) for record in mode_records if not record.get("error")]
        modes[mode] = {
            "runs": len(mode_records),
            "failures": sum(1 for record in mode_records if record.get("error")),
            "latency_seconds": _stats([float(item["wall_time_seconds"]) for item in metrics]),
            "fact_coverage": {
                "mean": mean(scores) if scores else None,
                "stdev": stdev(scores) if len(scores) > 1 else 0.0 if scores else None,
            },
            "total_tokens": _stats(
                [float(item["total_tokens"]) for item in metrics if item.get("total_tokens") is not None]
            ),
            "calls": {"mean": mean([float(item["calls"]) for item in metrics]) if metrics else None},
        }

    sequential_mean = modes.get("sequential", {}).get("latency_seconds", {}).get("mean")
    if sequential_mean:
        for mode, data in modes.items():
            mode_mean = data["latency_seconds"]["mean"]
            data["speedup_vs_sequential"] = speedup(sequential_mean, mode_mean) if mode_mean else None

    tasks = []
    direct_scores = {
        task_id: mean([float(record["evaluation"]["score"]) for record in task_records if not record.get("error")])
        for (task_id, mode), task_records in by_task_mode.items()
        if mode == "direct"
    }
    for (task_id, mode), task_records in sorted(by_task_mode.items()):
        successful = [record for record in task_records if not record.get("error")]
        metrics = [record["metrics"] for record in successful]
        scores = [float(record["evaluation"]["score"]) for record in successful]
        tokens = [float(item["total_tokens"]) for item in metrics if item.get("total_tokens") is not None]
        latency_values = [float(item["wall_time_seconds"]) for item in metrics]
        token_mean = mean(tokens) if tokens else None
        score_mean = mean(scores) if scores else None
        direct_score = direct_scores.get(task_id)
        tasks.append(
            {
                "task_id": task_id,
                "mode": mode,
                "runs": len(task_records),
                "failures": len(task_records) - len(successful),
                "input_word_count": task_records[0].get("input_word_count"),
                "latency_mean_seconds": mean(latency_values) if latency_values else None,
                "calls_mean": mean([float(item["calls"]) for item in metrics]) if metrics else None,
                "total_tokens_mean": token_mean,
                "fact_coverage_mean": score_mean,
                "fact_coverage_gain_vs_direct": (
                    score_mean - direct_score if score_mean is not None and direct_score is not None else None
                ),
                "fact_coverage_per_1000_tokens": (
                    score_mean / (token_mean / 1000) if score_mean is not None and token_mean else None
                ),
            }
        )

    return {"by_mode": modes, "by_task_mode": tasks}


def run_benchmark_suite(
    suite_path: str,
    config: RLMConfig,
    modes: list[str],
    repetitions: int = 1,
    warmup: bool | None = None,
) -> dict:
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    tasks = load_suite(suite_path)
    records = []
    warmup = config.provider == "ollama" if warmup is None else warmup
    warmup_result = _warmup(config) if warmup else {"performed": False, "included_in_aggregates": False}
    for task in tasks:
        for mode in modes:
            for repetition in range(1, repetitions + 1):
                record = {
                    "task_id": task["task_id"],
                    "category": task.get("category"),
                    "description": task.get("description"),
                    "input_word_count": len(str(task["input"]).split()),
                    "mode": mode,
                    "repetition": repetition,
                    "config": _mode_config(config, mode),
                }
                try:
                    run = run_mode(str(task["input"]), config, mode)
                    evaluation = evaluate_answer(task, str(run["answer"]))
                    record.update(
                        {
                            "metrics": {key: value for key, value in run.items() if key != "answer"},
                            "answer": run["answer"],
                            "evaluation": asdict(evaluation),
                        }
                    )
                except Exception as exc:
                    record.update(
                        {
                            "metrics": {
                                "failed_calls": 1,
                                "retries": 0,
                                "budget_termination_reason": None,
                            },
                            "answer": "",
                            "evaluation": {"method": "none", "score": 0.0, "details": {"reason": "run_failed"}},
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    )
                records.append(record)
    aggregates = aggregate_records(records)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite_path,
        "modes": modes,
        "repetitions": repetitions,
        "benchmark_config": _benchmark_config(config),
        "warmup": warmup_result,
        "records": records,
        "aggregates": aggregates,
        "note": (
            "Mock-provider results measure orchestration and deterministic reference metrics only."
            if config.provider == "mock"
            else (
                "Real-provider benchmark results are local repeated-run measurements; "
                "interpret latency and quality descriptively."
            )
        ),
    }


def run_comparison(
    text: str,
    config: RLMConfig,
    output_path: str | None = None,
    suite_path: str | None = None,
    modes: list[str] | None = None,
    repetitions: int = 1,
    warmup: bool | None = None,
) -> dict:
    modes = modes or ["direct", "sequential", "threaded", "async"]
    if suite_path:
        result = run_benchmark_suite(suite_path, config, modes, repetitions=repetitions, warmup=warmup)
    else:
        records = []
        for mode in modes:
            records.append({"mode": mode, "metrics": run_mode(text, config, mode)})
        sequential = next((record["metrics"] for record in records if record["mode"] == "sequential"), None)
        threaded = next((record["metrics"] for record in records if record["mode"] == "threaded"), None)
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "records": records,
            "speedup_threaded_vs_sequential": (
                speedup(sequential["wall_time_seconds"], threaded["wall_time_seconds"])
                if sequential and threaded
                else None
            ),
            "note": "Mock results measure orchestration overhead only unless a real provider is configured.",
        }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
