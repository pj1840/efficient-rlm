# Results

Public result artifacts are intentionally small and reproducible.

## Benchmarks

`benchmarks/mock_core.json` is generated with the deterministic mock provider. It exercises the orchestration paths without network access and does not represent real LLM latency.

`benchmarks/ollama_qwen2_5_coder_7b_repeated.json` is the final repeated real-model benchmark:

- provider: Ollama
- model: `qwen2.5-coder:7b`
- repetitions: 5 per task/mode
- timed records: 100
- warmup: one untimed request, excluded from aggregates

The repeated Ollama run shows a compute/quality tradeoff: recursive modes improved fact coverage on the largest distributed synthesis task, but direct mode remained substantially faster and cheaper in model calls/tokens. Threaded and async modes did not beat sequential execution on this local runtime.

## Trace

`traces/ollama_recursive_demo.json` and `traces/ollama_recursive_demo.html` are the polished real local-model execution trace.

## Figures

`figures/ollama_qwen2_5_coder_7b_repeated/` contains charts generated from the repeated benchmark JSON:

- `latency_by_mode.svg`
- `fact_coverage_by_mode.svg`
- `tokens_by_mode.svg`
- `quality_vs_latency.svg`
- `per_task_fact_coverage.svg`

Older one-off Phase 3 artifacts were moved to ignored local archive storage under `archive/results/phase3/`.
