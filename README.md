# Efficient Recursive Language Models via Parallel Execution and Curriculum-Aware Scheduling

Efficient RLM is a Python framework for recursively decomposing long-context summarization tasks into bounded subproblems, executing independent LLM subcalls sequentially or concurrently, and recursively aggregating intermediate outputs into a final response.

The repository is designed as an engineering-focused implementation of recursive LLM orchestration. It runs out of the box with a deterministic mock provider, so installation, tests, demos, and local benchmarks do not require paid API calls.

## Overview

The canonical implementation lives in `src/efficient_rlm`. It currently supports recursive summarization with:

- word-based chunk decomposition
- sequential and thread-based parallel execution modes
- recursive pairwise aggregation
- explicit recursion and call-budget controls
- optional inference-time curriculum-aware scheduling
- mock, Ollama, and OpenAI-compatible HTTP providers

## Motivation

Large-context LLM tasks are often easier to orchestrate when the input is split into smaller independent units. Recursive decomposition lets the system process bounded chunks, summarize or transform each one, and merge partial outputs into a final answer.

Parallel execution matters when subcalls are independent and I/O-bound. Remote LLM inference and local HTTP model servers often spend most wall-clock time waiting on model responses, so concurrent subcalls can reduce latency in real-provider settings. This project does not claim Python threads accelerate CPU-bound model inference.

## Architecture

Actual implemented flow:

```text
Input text
  |
  v
RecursivePipeline
  |
  +--> stopping / budget checks
  |
  v
Word chunk decomposition
  |
  v
Curriculum-aware ordering (optional)
  |
  +------------------------------+
  |                              |
Sequential executor      Parallel executor
                                 |
                         ThreadPoolExecutor
  |                              |
  +--------------+---------------+
                 |
                 v
          Partial summaries
                 |
                 v
       Recursive pairwise merge
                 |
                 v
       Final response + metrics
```

The archived prototypes explored a REPL-driven recursive model scaffold. The public implementation uses an explicit pipeline because it is easier to test, safer to run locally, and clearer for reproducible benchmarking.

## Key Features

- Recursive summarization pipeline with bounded word chunks.
- Sequential mode for deterministic baseline runs.
- Parallel mode implemented with `concurrent.futures.ThreadPoolExecutor`.
- Deterministic result ordering even when parallel workers finish out of order.
- Recursion controls: `max_depth`, `min_chunk_words`, `max_children`.
- Budget controls: `max_calls`, `workers`, `timeout_seconds`.
- Provider abstraction behind a small `generate(prompt)` interface.
- Mock provider for tests and demos without network access.
- Optional coarse-to-detailed recursive inference.
- JSON benchmark output for sequential vs parallel comparisons.

## Parallel Execution

Parallel mode uses Python threads through `ThreadPoolExecutor`. This is appropriate for I/O-bound LLM calls, where the process is often waiting for a local server or remote API response.

The executor preserves original chunk order:

1. each chunk receives a stable index,
2. workers may complete in any order,
3. results are written back by index,
4. aggregation receives the same order in sequential and parallel modes.

The current mock provider responds immediately, so local mock benchmarks usually show parallel mode as slower due to thread-management overhead. That is expected and is not evidence against real-provider parallelism.

## Curriculum-Aware Scheduling

This project implements curriculum-aware scheduling, not curriculum learning.

The optional scheduler orders inference-time subtasks from shorter chunks to longer chunks, then restores original order before aggregation. The CLI also supports a coarse-to-detailed recursive summarization pass: first a coarse pass, then a guided detailed pass, then final refinement.

No model parameters are trained. There is no fine-tuning, learned scheduler, or claim of curriculum-training improvement.

## Recursion and Budget Safety

The recursive pipeline has explicit controls:

- `max_depth`: maximum decomposition depth
- `chunk_size_words`: target chunk size
- `min_chunk_words`: terminal threshold for small inputs
- `max_children`: maximum fan-out per decomposition step
- `max_calls`: maximum total LLM calls
- `timeout_seconds`: request and executor timeout setting

Malformed or unexpectedly long inputs cannot create unbounded decomposition because fan-out, depth, and call count are all bounded.

## Provider Abstraction

Available providers:

- `mock`: deterministic local provider, no network or credentials required
- `ollama`: optional local Ollama HTTP generation endpoint
- `openai_compatible`: optional chat-completions-style HTTP endpoint

API keys are read from environment variables only. The default key variable is `RLM_API_KEY`. Importing `efficient_rlm` does not require credentials.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development, the standard-library `unittest` suite is sufficient. No runtime dependency is required for the mock provider.

## Quick Start

```bash
python -m efficient_rlm run \
  --input-file examples/sample_context.txt \
  --mode parallel \
  --workers 4
```

Run with curriculum-aware scheduling and coarse-to-detailed inference:

```bash
python -m efficient_rlm run \
  --input-file examples/sample_context.txt \
  --mode parallel \
  --workers 4 \
  --curriculum
```

## CLI Usage

Show help:

```bash
python -m efficient_rlm --help
python -m efficient_rlm run --help
python -m efficient_rlm benchmark --help
```

Sequential run:

```bash
python -m efficient_rlm run \
  --input-file examples/sample_context.txt \
  --mode sequential
```

Parallel run:

```bash
python -m efficient_rlm run \
  --input-file examples/sample_context.txt \
  --mode parallel \
  --workers 4
```

Ollama run:

```bash
python -m efficient_rlm run \
  --provider ollama \
  --model qwen2.5-coder:7b \
  --endpoint http://localhost:11434/api/generate \
  --input-file examples/sample_context.txt
```

OpenAI-compatible endpoint:

```bash
export RLM_API_KEY=replace-with-real-key
python -m efficient_rlm run \
  --provider openai_compatible \
  --endpoint https://api.example.com/v1/chat/completions \
  --model provider-model-name \
  --input-file examples/sample_context.txt
```

## Benchmarking

```bash
python -m efficient_rlm benchmark \
  --input-file examples/sample_context.txt \
  --output results/mock_comparison.json
```

The benchmark records:

- final answer
- execution mode
- model-call count
- task count
- maximum depth reached
- wall-clock runtime
- computed speedup

## Project Structure

```text
configs/                 Default runtime configuration
examples/                Sample input text
results/                 Small reproducible benchmark artifacts
src/efficient_rlm/        Canonical Python package
  llm/                    Provider interface and HTTP/mock clients
  recursive/              Decomposition, execution, aggregation, stopping, pipeline
  scheduling/             Curriculum-aware inference scheduler
  evaluation/             Benchmark runner and metrics
tests/                   Standard-library unittest suite
```

Local historical prototypes are kept under `archive/` but excluded from Git.

## Testing

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

The tests use the mock provider only. They do not call paid APIs or require network access.

## Current Results

`results/mock_comparison.json` is the only checked-in benchmark artifact.

Latest checked-in mock run on `examples/sample_context.txt`:

| Mode | Calls | Tasks | Max depth | Runtime |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 5 | 6 | 2 | ~0.0001416 s |
| Parallel | 5 | 6 | 2 | ~0.0005710 s |

Mock speedup: ~0.2480x.

This does not show real LLM latency improvement. It measures local orchestration overhead with an instant mock backend. Because the mock provider returns immediately, thread overhead makes parallel mode slower. The benchmark infrastructure is intended for future real-provider latency comparisons.

## Limitations

- The canonical pipeline currently implements summarization only.
- Decomposition is word-count based, not semantic.
- Parallelism is thread-based and intended for I/O-bound subcalls.
- There is no genuine curriculum learning, fine-tuning, or learned scheduling.
- There is no answer-quality evaluator.
- Provider token accounting is not implemented.
- The implementation is not a production sandbox for arbitrary model-generated code.

## Future Work

- Add semantic decomposition strategies.
- Add quality evaluation datasets and scoring.
- Add rate-limit-aware async execution.
- Add provider usage/token accounting when APIs expose it.
- Add richer task types beyond summarization.
- Revisit a sandboxed REPL mode only with strict execution controls.
