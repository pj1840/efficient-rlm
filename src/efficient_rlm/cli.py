from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from efficient_rlm.config import load_config
from efficient_rlm.evaluation.runner import run_comparison
from efficient_rlm.llm.http import build_llm_client
from efficient_rlm.recursive.pipeline import RecursivePipeline


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
    run.add_argument("--mode", choices=["sequential", "parallel"], help="Execution strategy for independent subtasks.")
    run.add_argument("--provider", choices=["mock", "ollama", "openai_compatible"], help="LLM backend.")
    run.add_argument("--model", help="Provider model name.")
    run.add_argument("--endpoint", help="Provider HTTP endpoint. Ollama defaults to localhost if omitted.")
    run.add_argument("--workers", type=int, help="Maximum thread workers for parallel mode.")
    run.add_argument("--chunk-size-words", type=int, help="Target words per decomposed chunk.")
    run.add_argument("--max-depth", type=int, help="Maximum recursive decomposition depth.")
    run.add_argument("--max-calls", type=int, help="Maximum total LLM calls before stopping.")
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
    bench.add_argument("--chunk-size-words", type=int, help="Target words per decomposed chunk.")
    bench.add_argument("--max-depth", type=int, help="Maximum recursive decomposition depth.")
    bench.add_argument("--max-calls", type=int, help="Maximum total LLM calls before stopping.")
    bench.add_argument("--output", default="results/mock_comparison.json", help="Path for benchmark JSON output.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(
        args.config,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        endpoint=getattr(args, "endpoint", None),
        execution_mode=getattr(args, "mode", None),
        workers=getattr(args, "workers", None),
        chunk_size_words=getattr(args, "chunk_size_words", None),
        max_depth=getattr(args, "max_depth", None),
        max_calls=getattr(args, "max_calls", None),
        enable_curriculum=True if getattr(args, "curriculum", False) else None,
    )
    text = _read_text(args)

    if args.command == "benchmark":
        result = run_comparison(text, config, output_path=args.output)
        print(json.dumps(result, indent=2))
        return

    pipeline = RecursivePipeline(build_llm_client(config), config)
    result = pipeline.run_curriculum_summary(text) if config.enable_curriculum else pipeline.run(text)
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(result.answer)
        print(
            f"\nmode={result.mode} calls={result.calls} tasks={result.tasks} "
            f"max_depth={result.max_depth_reached} seconds={result.wall_time_seconds:.4f}"
        )
