from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from efficient_rlm.config import load_config
from efficient_rlm.evaluation.runner import run_comparison
from efficient_rlm.llm.http import build_llm_client
from efficient_rlm.recursive.pipeline import RecursivePipeline
from efficient_rlm.tracing import render_trace, trace_to_mermaid


def _read_text(args: argparse.Namespace) -> str:
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    return (
        "Artificial intelligence is used in healthcare, finance, education, robotics, "
        "manufacturing, and scientific research. It can improve diagnostics, automate "
        "routine work, personalize learning, and discover patterns in large datasets. "
        "These systems also raise privacy, fairness, accountability, and oversight concerns."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="efficient-rlm",
        description="Recursive summarization with sequential or parallel LLM subcalls.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a simple YAML-style config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run recursive summarization")
    run.add_argument("--text", help="Input text. If omitted, a built-in demo text is used.")
    run.add_argument("--input-file", help="Path to a UTF-8 text file to summarize.")
    run.add_argument(
        "--mode",
        choices=["sequential", "threaded", "parallel", "async"],
        help="Execution strategy for independent subtasks. 'parallel' is an alias for 'threaded'.",
    )
    run.add_argument("--provider", choices=["mock", "ollama", "openai_compatible"], help="LLM backend.")
    run.add_argument("--model", help="Provider model name.")
    run.add_argument("--endpoint", help="Provider HTTP endpoint. Ollama defaults to localhost if omitted.")
    run.add_argument("--workers", type=int, help="Maximum thread workers for parallel mode.")
    run.add_argument("--timeout-seconds", type=float, help="Per-request provider timeout in seconds.")
    run.add_argument("--max-tokens", type=int, help="Provider maximum generated tokens per call.")
    run.add_argument("--temperature", type=float, help="Provider generation temperature.")
    run.add_argument("--decomposer", choices=["fixed", "semantic"], help="Subproblem decomposition strategy.")
    run.add_argument("--policy", choices=["deterministic", "adaptive"], help="Recursive decision policy.")
    run.add_argument("--chunk-size-words", type=int, help="Target words per decomposed chunk.")
    run.add_argument("--max-depth", type=int, help="Maximum recursive decomposition depth.")
    run.add_argument("--max-calls", type=int, help="Maximum total LLM calls before stopping.")
    run.add_argument("--max-wall-time-seconds", type=float, help="Maximum wall-clock seconds for one recursive run.")
    run.add_argument("--max-prompt-tokens", type=int, help="Maximum reserved or reported prompt tokens.")
    run.add_argument("--max-completion-tokens", type=int, help="Maximum provider-reported completion tokens.")
    run.add_argument("--max-total-tokens", type=int, help="Maximum total provider-reported or estimated tokens.")
    run.add_argument("--trace", help="Optional path to save an execution trace JSON file.")
    run.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable inference-time curriculum-aware scheduling and coarse-to-detailed summarization.",
    )
    run.add_argument("--json", action="store_true", help="Print full result metadata as JSON.")

    bench = subparsers.add_parser("benchmark", help="Compare sequential and parallel modes")
    bench.add_argument("--text", help="Input text. If omitted, a built-in demo text is used.")
    bench.add_argument("--input-file", help="Path to a UTF-8 text file to summarize.")
    bench.add_argument("--provider", choices=["mock", "ollama", "openai_compatible"], help="LLM backend.")
    bench.add_argument("--model", help="Provider model name.")
    bench.add_argument("--endpoint", help="Provider HTTP endpoint.")
    bench.add_argument("--workers", type=int, help="Maximum thread workers for parallel mode.")
    bench.add_argument("--timeout-seconds", type=float, help="Per-request provider timeout in seconds.")
    bench.add_argument("--max-tokens", type=int, help="Provider maximum generated tokens per call.")
    bench.add_argument("--temperature", type=float, help="Provider generation temperature.")
    bench.add_argument("--decomposer", choices=["fixed", "semantic"], help="Subproblem decomposition strategy.")
    bench.add_argument("--policy", choices=["deterministic", "adaptive"], help="Recursive decision policy.")
    bench.add_argument("--chunk-size-words", type=int, help="Target words per decomposed chunk.")
    bench.add_argument("--max-depth", type=int, help="Maximum recursive decomposition depth.")
    bench.add_argument("--max-calls", type=int, help="Maximum total LLM calls before stopping.")
    bench.add_argument("--max-wall-time-seconds", type=float, help="Maximum wall-clock seconds for each mode run.")
    bench.add_argument("--max-prompt-tokens", type=int, help="Maximum reserved or reported prompt tokens.")
    bench.add_argument("--max-completion-tokens", type=int, help="Maximum provider-reported completion tokens.")
    bench.add_argument("--max-total-tokens", type=int, help="Maximum total provider-reported or estimated tokens.")
    bench.add_argument("--suite", default="benchmarks/core/tasks.json", help="Benchmark suite JSON path.")
    bench.add_argument("--modes", default="direct,sequential,threaded,async", help="Comma-separated modes to run.")
    bench.add_argument("--repetitions", type=int, default=1, help="Raw repetitions per task/mode combination.")
    bench.add_argument(
        "--warmup",
        action="store_true",
        default=None,
        help="Run one untimed provider warmup request before timed benchmark records.",
    )
    bench.add_argument(
        "--no-warmup",
        action="store_false",
        dest="warmup",
        help="Disable benchmark warmup. By default, Ollama benchmarks warm up once and mock benchmarks do not.",
    )
    bench.add_argument("--output", default="results/mock_comparison.json", help="Path for benchmark JSON output.")

    trace = subparsers.add_parser("trace", help="Render a saved execution trace")
    trace.add_argument("trace_file", help="Path to trace JSON")
    trace.add_argument("--mermaid", action="store_true", help="Print Mermaid graph syntax instead of a text tree.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(
        args.config,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        endpoint=getattr(args, "endpoint", None),
        temperature=getattr(args, "temperature", None),
        max_tokens=getattr(args, "max_tokens", None),
        timeout_seconds=getattr(args, "timeout_seconds", None),
        execution_mode=getattr(args, "mode", None),
        workers=getattr(args, "workers", None),
        decomposer=getattr(args, "decomposer", None),
        policy=getattr(args, "policy", None),
        chunk_size_words=getattr(args, "chunk_size_words", None),
        max_depth=getattr(args, "max_depth", None),
        max_calls=getattr(args, "max_calls", None),
        max_wall_time_seconds=getattr(args, "max_wall_time_seconds", None),
        max_prompt_tokens=getattr(args, "max_prompt_tokens", None),
        max_completion_tokens=getattr(args, "max_completion_tokens", None),
        max_total_tokens=getattr(args, "max_total_tokens", None),
        enable_curriculum=True if getattr(args, "curriculum", False) else None,
    )
    if args.command == "trace":
        output = trace_to_mermaid(args.trace_file) if args.mermaid else render_trace(args.trace_file)
        print(output)
        return

    text = _read_text(args)

    if args.command == "benchmark":
        result = run_comparison(
            text,
            config,
            output_path=args.output,
            suite_path=getattr(args, "suite", None),
            modes=[mode.strip() for mode in getattr(args, "modes", "").split(",") if mode.strip()],
            repetitions=getattr(args, "repetitions", 1),
            warmup=getattr(args, "warmup", None),
        )
        print(json.dumps(result, indent=2))
        return

    pipeline = RecursivePipeline(build_llm_client(config), config)
    result = (
        pipeline.run_curriculum_summary(text, trace_path=getattr(args, "trace", None))
        if config.enable_curriculum
        else pipeline.run(text, trace_path=getattr(args, "trace", None))
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(result.answer)
        print(
            f"\nmode={result.mode} calls={result.calls} tasks={result.tasks} "
            f"max_depth={result.max_depth_reached} seconds={result.wall_time_seconds:.4f}"
        )
