# Efficient RLM

**Adaptive and Parallel Recursive Inference for Language Models**

Efficient RLM is a Python framework for studying when recursive decomposition improves language-model output quality enough to justify its additional inference cost. It decomposes synthesis tasks into bounded subproblems, executes independent subcalls sequentially, with `ThreadPoolExecutor`, or with bounded asyncio concurrency, and recursively aggregates partial outputs with traceable metrics.

## What I Built

```text
Input task
  |
  v
RecursivePipeline
  |
  +--> RecursivePolicy: answer directly, decompose, or stop
  +--> Budget checks: calls, depth, wall time, prompt/completion/total tokens
  |
  v
Decomposer: fixed chunks or semantic JSON decomposition
  |
  v
Optional inference-time curriculum-aware ordering
  |
  +----------------+----------------+----------------+
  |                |                |                |
Sequential      Threaded          Async
executor        executor          executor
                 |                semaphore + timeout
                 |                retries + cancellation
  +--------------+----------------+----------------+
                 |
                 v
          Partial responses
                 |
                 v
       Recursive pairwise merge
                 |
                 v
   Final answer + metrics + trace JSON/HTML
```

Implemented systems components:

- recursive summarization/synthesis with deterministic stopping limits
- fixed and semantic decomposition
- dependency-aware subtask scheduling
- sequential, threaded, and bounded asyncio execution
- provider abstraction for mock, Ollama, and OpenAI-compatible endpoints
- call, latency, recursion-depth, retry, token, and budget accounting
- repeated benchmark runner with raw records and descriptive aggregates
- execution traces rendered as text, Mermaid, and self-contained HTML
- deterministic reference evaluation with required-fact coverage

## Real-Model Evaluation

The checked-in real benchmark uses the only local model available during validation: `qwen2.5-coder:7b` through Ollama. This is a code-specialized model used because it was already installed locally; these results are not universal behavior for general-purpose instruction models.

Benchmark artifact:

`results/benchmarks/ollama_qwen2_5_coder_7b_repeated.json`

Configuration summary:

- 4 synthetic benchmark tasks
- 5 repetitions per task/mode
- 100 timed raw records
- one untimed Ollama warmup request, excluded from aggregates
- modes: `direct`, `sequential`, `threaded`, `async`, `adaptive`
- workers: `2`
- chunk size: `90` words
- max depth: `2`
- max generated tokens per call: `160`

Aggregate repeated results:

| Mode | Mean latency ± sd (s) | Median latency (s) | Mean calls | Mean tokens ± sd | Mean fact coverage ± sd | Speedup vs sequential |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 4.48 ± 2.27 | 3.47 | 1.0 | 276.05 ± 55.17 | 0.442 ± 0.338 | 3.147x |
| sequential | 14.09 ± 5.10 | 15.24 | 3.5 | 692.55 ± 273.82 | 0.542 ± 0.324 | 1.000x |
| threaded | 16.31 ± 8.03 | 15.62 | 3.5 | 701.05 ± 302.51 | 0.567 ± 0.329 | 0.863x |
| async | 17.48 ± 9.01 | 15.11 | 3.5 | 697.75 ± 293.23 | 0.517 ± 0.305 | 0.806x |
| adaptive | 17.93 ± 8.62 | 16.95 | 3.5 | 710.10 ± 290.89 | 0.560 ± 0.329 | 0.785x |

Primary finding: on this local Qwen2.5-Coder 7B benchmark, recursive execution sometimes improved fact coverage, especially on the largest distributed synthesis task, but it required substantially more model calls, tokens, and wall-clock time. Threaded and async execution did not outperform sequential execution because the local Ollama runtime remained the dominant bottleneck.

The strongest positive result was `long_context_summary`: direct fact coverage averaged `0.25`, while recursive modes reached `0.675` to `0.775`. The strongest negative result was `multi_section_synthesis`: direct already reached `1.0` fact coverage, so recursive modes added calls, tokens, and latency without a quality benefit.

## Execution Trace

Polished real trace demo:

- `results/traces/ollama_recursive_demo.json`
- `results/traces/ollama_recursive_demo.html`

The trace captures the recursive tree, node status, latency, model metadata, token counts, stopping reasons, child ordering, aggregation state, and output previews. Open the HTML file directly in a browser.

Render traces yourself:

```bash
python -m efficient_rlm trace results/traces/ollama_recursive_demo.json
python -m efficient_rlm trace results/traces/ollama_recursive_demo.json --mermaid
python scripts/render_trace_html.py \
  results/traces/ollama_recursive_demo.json \
  --output results/traces/ollama_recursive_demo.html
```

## Execution Modes

- `direct`: one provider call; benchmark baseline.
- `sequential`: recursive decomposition with one subtask at a time.
- `threaded`: recursive decomposition using `ThreadPoolExecutor`.
- `async`: bounded `asyncio` concurrency with semaphore, timeout, retries, backoff, cancellation cleanup, and `asyncio.to_thread` around synchronous provider calls.
- `adaptive`: recursive execution using heuristic adaptive policy decisions.
- `adaptive_scheduled`: adaptive execution plus inference-time curriculum-aware scheduling and coarse-to-detailed summarization.

Threads and asyncio are useful for I/O-bound provider calls. This project does not claim Python threads accelerate CPU-bound model inference.

## Decomposition and Scheduling

The decomposer interface supports:

- `fixed`: deterministic word chunks; reliable fallback and default.
- `semantic`: asks the configured provider for JSON subproblems with IDs, prompts, context slices, estimated difficulty, and dependency IDs.

Malformed semantic output falls back to fixed chunking by default. Dependencies are enforced: missing IDs, self-dependencies, duplicate IDs, and cycles are rejected, and valid dependent subtasks wait for prerequisites.

Curriculum-aware scheduling here means inference-time ordering, usually shorter/easier chunks first. It is not curriculum learning, model training, fine-tuning, or learned scheduling.

## Budgets and Safety

Hard controls include:

- `max_depth`
- `max_calls`
- `max_wall_time_seconds`
- `max_prompt_tokens`
- `max_completion_tokens`
- `max_total_tokens`
- `chunk_size_words`
- `min_chunk_words`
- `max_children`
- per-request timeout and retries

Call count and estimated prompt tokens are reserved before a request starts, which prevents concurrent workers from all passing a stale budget check. Completion tokens and provider-reported actual totals are only known after responses return, so in-flight requests can create bounded overshoot for those post-hoc token limits.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Development checks:

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
python -m efficient_rlm run --input-file examples/sample_context.txt

efficient-rlm run \
  --input-file examples/sample_context.txt \
  --mode async \
  --workers 4 \
  --policy adaptive \
  --trace results/traces/example.json
```

## CLI Usage

```bash
efficient-rlm --help
efficient-rlm run --help
efficient-rlm benchmark --help
```

Common flags:

- `--provider mock|ollama|openai_compatible`
- `--model <model-name>`
- `--endpoint <provider-url>`
- `--mode sequential|threaded|parallel|async`
- `--workers <n>`
- `--decomposer fixed|semantic`
- `--policy deterministic|adaptive`
- `--chunk-size-words <n>`
- `--max-depth <n>`
- `--max-calls <n>`
- `--max-tokens <n>`
- `--timeout-seconds <seconds>`
- `--repetitions <n>`
- `--warmup` / `--no-warmup`
- `--curriculum`

## Provider Support

Mock provider, no network:

```bash
python -m efficient_rlm run --input-file examples/sample_context.txt --provider mock
```

Ollama:

```bash
python -m efficient_rlm run \
  --provider ollama \
  --model <local-ollama-model> \
  --endpoint http://localhost:11434/api/generate \
  --input-file examples/sample_context.txt
```

OpenAI-compatible HTTP endpoint:

```bash
export RLM_API_KEY=replace-with-real-key
python -m efficient_rlm run \
  --provider openai_compatible \
  --endpoint https://api.example.com/v1/chat/completions \
  --model provider-model-name \
  --input-file examples/sample_context.txt
```

API keys are read from environment variables only. Importing the package does not require credentials.

## Reproduce Benchmarks

Mock benchmark:

```bash
python -m efficient_rlm benchmark \
  --suite benchmarks/core/tasks.json \
  --modes direct,sequential,threaded,async,adaptive,adaptive_scheduled \
  --repetitions 1 \
  --output results/benchmarks/mock_core.json
```

Repeated Ollama benchmark:

```bash
python -m efficient_rlm benchmark \
  --provider ollama \
  --model qwen2.5-coder:7b \
  --suite benchmarks/core/tasks.json \
  --modes direct,sequential,threaded,async,adaptive \
  --repetitions 5 \
  --workers 2 \
  --chunk-size-words 90 \
  --max-depth 2 \
  --max-calls 16 \
  --max-tokens 160 \
  --timeout-seconds 120 \
  --output results/benchmarks/ollama_qwen2_5_coder_7b_repeated.json
```

Generate charts:

```bash
python scripts/plot_benchmarks.py \
  results/benchmarks/ollama_qwen2_5_coder_7b_repeated.json \
  --out-dir results/figures/ollama_qwen2_5_coder_7b_repeated
```

Optional future general-purpose model run, after you install a suitable Ollama model yourself:

```bash
ollama pull qwen2.5:7b-instruct
python -m efficient_rlm benchmark \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --suite benchmarks/core/tasks.json \
  --modes direct,sequential,threaded,async,adaptive \
  --repetitions 5 \
  --workers 2 \
  --chunk-size-words 90 \
  --max-depth 2 \
  --max-calls 16 \
  --max-tokens 160 \
  --timeout-seconds 120 \
  --output results/benchmarks/ollama_qwen2_5_7b_instruct_repeated.json
```

Do not compare those future results to the checked-in Qwen2.5-Coder run unless the model, hardware, runtime, and benchmark configuration are documented.

## Project Structure

```text
benchmarks/core/tasks.json                 Synthetic benchmark suite
configs/default.yaml                       Default runtime configuration
docs/architecture.md                       Technical design document
docs/benchmark_methodology.md              Experiment methodology
examples/sample_context.txt                Demo input
scripts/plot_benchmarks.py                 Benchmark chart generation
scripts/render_trace_html.py               Static trace HTML renderer
src/efficient_rlm/config.py                Runtime configuration
src/efficient_rlm/llm/                     Provider interface and clients
src/efficient_rlm/recursive/decomposers.py Fixed and semantic decomposition
src/efficient_rlm/recursive/policy.py      Deterministic/adaptive policies
src/efficient_rlm/recursive/executor.py    Sequential/threaded/async execution
src/efficient_rlm/recursive/pipeline.py    Recursive controller
src/efficient_rlm/tracing.py               Trace JSON and text/Mermaid renderers
src/efficient_rlm/evaluation/              Benchmarks and evaluation
tests/                                     Unit and regression tests
```

## Testing and CI

```bash
ruff check .
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Normal tests use deterministic fake providers. They do not require network access, paid APIs, or Ollama. GitHub Actions CI runs the mock/fake-provider checks only.

## Limitations

- The main implemented task type is summarization/synthesis.
- The benchmark suite is synthetic and small.
- The real benchmark uses a code-specialized local model because it was already installed.
- Required-fact coverage is deterministic and auditable, but it is not a full measure of answer faithfulness or usefulness.
- LLM-as-judge support is optional and should not be treated as objective truth.
- Completion-token and actual-total-token budgets can only be checked after in-flight requests return.
- The repeated Ollama benchmark did not show parallel speedup.
- No model training, fine-tuning, learned scheduler, or genuine curriculum learning is implemented.
