# Benchmark Methodology

## Purpose

The benchmark suite evaluates the systems tradeoff behind recursive inference:

> When does recursive decomposition improve output quality enough to justify additional inference cost?

The benchmark measures latency, model calls, token usage, recursion depth, retries/failures, and deterministic fact coverage. It does not claim universal speedups or statistical significance.

## Environment

Repeated real-model benchmark environment:

- OS: macOS 26.2
- CPU: Apple M4
- Memory: 16 GB
- Ollama client: 0.32.8
- Model: `qwen2.5-coder:7b`

No username, hostname, or absolute local filesystem path is stored in public benchmark artifacts.

`qwen2.5-coder:7b` was used because it was the only local Ollama model already installed during validation. It is code-specialized and is not necessarily representative of general-purpose instruction models.

## Task Suite

The checked-in suite is `benchmarks/core/tasks.json`. It is synthetic and redistributable.

Current tasks:

- `long_context_summary`: 198 words, multi-section summarization with facts distributed across domains.
- `multi_section_synthesis`: 95 words, engineering status synthesis.
- `synthetic_multidoc_qa`: 100 words, synthetic multi-document question answering.
- `hierarchical_aggregation`: 98 words, aggregation of distributed project observations.

Each task includes:

- task ID
- category and description
- input text
- reference answer
- deterministic evaluation method
- required facts and/or keywords

No external datasets are downloaded.

## Baselines

The repeated Ollama benchmark compares:

- `direct`: one provider call with no recursive decomposition.
- `sequential`: recursive decomposition with sequential subtask execution.
- `threaded`: recursive decomposition using `ThreadPoolExecutor`.
- `async`: recursive decomposition using bounded asyncio concurrency.
- `adaptive`: recursive execution using the adaptive policy.

The mock benchmark may additionally include `adaptive_scheduled`, which enables inference-time curriculum-aware scheduling and coarse-to-detailed summarization.

## Real Benchmark Configuration

Repeated Ollama command:

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

Generation configuration:

- temperature: `0.2`
- max generated tokens per call: `160`
- request timeout: `120` seconds
- max retries: `1`

Recursion configuration:

- decomposer: `fixed`
- base policy: `deterministic`
- adaptive mode overrides policy to `adaptive`
- chunk size: `90` words
- min chunk size: `15` words
- max recursion depth: `2`
- max model calls per run: `16`
- workers/concurrency: `2`

## Warmup Policy

For Ollama benchmarks, the runner performs one small untimed warmup request before timed records:

```text
Warmup request for local benchmark timing. Reply with one short sentence.
```

Warmup latency is not recorded and is excluded from aggregates. The result JSON records that warmup was performed and stores returned token metadata when available. The model is not repeatedly unloaded or reloaded between modes.

Mock benchmarks do not warm up by default.

## Repetitions and Raw Records

The repeated real benchmark uses 5 repetitions per task/mode combination:

- 4 tasks
- 5 modes
- 5 repetitions
- 100 timed raw records

Every raw repetition is saved in `records`. Each record includes:

- task ID, category, description, and input word count
- mode
- repetition number
- provider/model/mode configuration
- answer
- wall-clock latency
- calls, retries, failures
- recursion depth and task count
- prompt, completion, and total tokens when available
- deterministic evaluation result
- explicit error payload if a run fails

## Timing

Timing uses `time.perf_counter()` inside provider calls and the recursive pipeline. Reported run latency includes orchestration overhead, recursive decomposition, provider calls, aggregation, retries, and trace-free benchmark bookkeeping.

Single-machine local timings are sensitive to machine load and Ollama runtime behavior. The benchmark reports descriptive statistics rather than significance tests.

## Evaluation

Reference-based deterministic metrics:

- `required_fact_coverage`: fraction of required facts present in the answer.
- `keyword_recall`: fraction of expected keywords present in the answer.
- `exact_match`: strict normalized string match.

The repeated real benchmark uses required-fact coverage. It is auditable and deterministic, but it is not a full measure of coherence, faithfulness, or usefulness.

Optional LLM-as-judge evaluation is implemented separately and must be reported as subjective model-judge output, not objective truth.

## Aggregation

For each mode, the runner computes:

- latency mean, median, standard deviation, min, and max
- fact-coverage mean and standard deviation
- total-token mean, median, standard deviation, min, and max
- mean model calls
- speedup relative to sequential mean latency

For each task/mode pair, the runner computes:

- input word count
- mean latency
- mean calls
- mean total tokens
- mean fact coverage
- fact coverage gain vs direct
- fact coverage per 1,000 tokens

The project does not claim statistical significance. Descriptive statistics are the appropriate evidence level for this small synthetic benchmark.

## Results Interpretation

The repeated `qwen2.5-coder:7b` run showed:

- direct mode was fastest and cheapest in calls/tokens
- recursive modes improved fact coverage on `long_context_summary`
- recursive modes provided no benefit on `synthetic_multidoc_qa`
- recursive modes were mostly wasteful on `multi_section_synthesis`, where direct already reached full fact coverage
- threaded and async modes did not beat sequential latency on the local Ollama runtime

This supports the project framing: recursive inference is a compute/quality tradeoff, not a universal speedup technique.

## Reproducibility

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

Charts:

```bash
python scripts/plot_benchmarks.py \
  results/benchmarks/ollama_qwen2_5_coder_7b_repeated.json \
  --out-dir results/figures/ollama_qwen2_5_coder_7b_repeated
```

Trace demo:

```bash
python -m efficient_rlm trace results/traces/ollama_recursive_demo.json
python -m efficient_rlm trace results/traces/ollama_recursive_demo.json --mermaid
python scripts/render_trace_html.py \
  results/traces/ollama_recursive_demo.json \
  --output results/traces/ollama_recursive_demo.html
```

## Limitations

- The suite is small and synthetic.
- The real benchmark uses a code-specialized local model.
- Hardware/runtime state can affect local latency.
- Required-fact coverage does not measure all dimensions of answer quality.
- The benchmark is descriptive and does not include significance testing.
- Real API providers may behave differently from local Ollama.
