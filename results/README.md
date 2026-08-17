# Results

`mock_comparison.json` is a reproducible local benchmark generated with the deterministic mock provider.

It measures orchestration behavior only: decomposition, sequential execution, thread-pool execution, recursive aggregation, and metric recording. It does not measure real LLM latency or answer quality. Because the mock backend returns immediately, thread-management overhead can make parallel mode slower than sequential mode.

