import json
import tempfile
import time
import unittest
from pathlib import Path

from efficient_rlm.config import RLMConfig
from efficient_rlm.evaluation.evaluators import keyword_recall, required_fact_coverage
from efficient_rlm.evaluation.runner import run_benchmark_suite
from efficient_rlm.llm.base import LLMClient, LLMResponse
from efficient_rlm.llm.mock import MockLLMClient
from efficient_rlm.models import Metrics, Subtask
from efficient_rlm.recursive.decomposers import FixedChunkDecomposer, SemanticDecomposer
from efficient_rlm.recursive.executor import ExecutionError, execute_tasks
from efficient_rlm.recursive.pipeline import BudgetExceeded, RecursivePipeline
from efficient_rlm.recursive.policy import AdaptivePolicy, DeterministicPolicy
from efficient_rlm.tracing import render_trace
from scripts.render_trace_html import render_file


class BadJsonLLM(LLMClient):
    def generate(self, prompt: str) -> str:
        return "not json"


class FlakyTask:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, task: Subtask) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary")
        return task.text


class FixedTokenLLM(LLMClient):
    def __init__(self, completion_tokens: int = 10) -> None:
        self.completion_tokens = completion_tokens

    def generate(self, prompt: str) -> str:
        return "ok"

    def generate_response(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text="ok",
            provider="fake",
            model="fixed-token",
            prompt_tokens=len(prompt.split()),
            completion_tokens=self.completion_tokens,
            total_tokens=len(prompt.split()) + self.completion_tokens,
        )


class Phase2FeatureTests(unittest.TestCase):
    def test_semantic_decomposition_parses_structured_mock_output(self):
        config = RLMConfig(decomposer="semantic", max_children=3)
        decomposer = SemanticDecomposer(MockLLMClient(), config)
        tasks = decomposer.decompose("summarize", "First section. Second section.", 0, "root")
        self.assertGreaterEqual(len(tasks), 1)
        self.assertEqual(tasks[0].parent_id, "root")
        self.assertIsNotNone(tasks[0].estimated_difficulty)

    def test_malformed_semantic_decomposition_falls_back(self):
        config = RLMConfig(decomposer="semantic", chunk_size_words=2)
        fallback = FixedChunkDecomposer(config)
        decomposer = SemanticDecomposer(BadJsonLLM(), config, fallback=fallback)
        tasks = decomposer.decompose("summarize", "one two three four", 0, "root")
        self.assertEqual(len(tasks), 2)

    def test_policy_decisions(self):
        config = RLMConfig(chunk_size_words=5, complexity_threshold_words=20)
        metrics = Metrics()
        self.assertEqual(DeterministicPolicy().decide("one two", 0, metrics, config).action, "ANSWER_DIRECTLY")
        self.assertEqual(DeterministicPolicy().decide(" ".join(["x"] * 30), 0, metrics, config).action, "DECOMPOSE")
        self.assertEqual(AdaptivePolicy().decide(" ".join(["x"] * 10), 1, metrics, config).action, "ANSWER_DIRECTLY")

    def test_async_executor_preserves_order_and_retries(self):
        config = RLMConfig(execution_mode="async", workers=2, max_retries=1)
        tasks = [Subtask(id=i, text=str(i), depth=1, index=i) for i in range(3)]
        flaky = FlakyTask()
        results = execute_tasks(tasks, flaky, config)
        self.assertEqual(results, ["0", "1", "2"])

    def test_async_executor_runs_with_bounded_concurrency(self):
        config = RLMConfig(execution_mode="async", workers=2, max_retries=0, timeout_seconds=1)
        tasks = [Subtask(id=i, text=str(i), depth=1, index=i) for i in range(4)]

        def delayed(task: Subtask) -> str:
            time.sleep(0.1)
            return str(task.id)

        started = time.perf_counter()
        results = execute_tasks(tasks, delayed, config)
        elapsed = time.perf_counter() - started
        self.assertEqual(results, ["0", "1", "2", "3"])
        self.assertGreaterEqual(elapsed, 0.18)
        self.assertLess(elapsed, 0.35)

    def test_async_executor_reports_timeouts_and_cleans_up_pending_tasks(self):
        config = RLMConfig(execution_mode="async", workers=2, max_retries=0, timeout_seconds=0.05)
        tasks = [Subtask(id=i, text=str(i), depth=1, index=i) for i in range(4)]

        def too_slow(task: Subtask) -> str:
            time.sleep(0.2)
            return str(task.id)

        started = time.perf_counter()
        with self.assertRaisesRegex(ExecutionError, "failed"):
            execute_tasks(tasks, too_slow, config)
        self.assertLess(time.perf_counter() - started, 0.3)

    def test_dependency_execution_waits_for_prerequisites(self):
        config = RLMConfig(execution_mode="threaded", workers=3)
        tasks = [
            Subtask(id="a", text="A", depth=1, index=0),
            Subtask(id="b", text="B", depth=1, index=1, dependencies=["a"]),
            Subtask(id="c", text="C", depth=1, index=2, dependencies=["b"]),
        ]
        order = []

        def record(task: Subtask) -> str:
            order.append(str(task.id))
            return task.text

        self.assertEqual(execute_tasks(tasks, record, config), ["A", "B", "C"])
        self.assertEqual(order, ["a", "b", "c"])

    def test_dependency_validation_rejects_missing_and_cycles(self):
        config = RLMConfig()
        missing = [Subtask(id="a", text="A", depth=1, index=0, dependencies=["z"])]
        with self.assertRaisesRegex(ExecutionError, "missing"):
            execute_tasks(missing, lambda task: task.text, config)

        cyclic = [
            Subtask(id="a", text="A", depth=1, index=0, dependencies=["b"]),
            Subtask(id="b", text="B", depth=1, index=1, dependencies=["a"]),
        ]
        with self.assertRaisesRegex(ExecutionError, "Cycle"):
            execute_tasks(cyclic, lambda task: task.text, config)

    def test_token_budget_enforced(self):
        config = RLMConfig(max_total_tokens=3)
        with self.assertRaises(BudgetExceeded):
            RecursivePipeline(MockLLMClient(), config).run("one two three four five six")

    def test_concurrent_prompt_budget_is_reserved_before_requests(self):
        config = RLMConfig(execution_mode="threaded", workers=4, chunk_size_words=5, max_prompt_tokens=60)
        with self.assertRaisesRegex(ExecutionError, "max_prompt_tokens"):
            RecursivePipeline(FixedTokenLLM(completion_tokens=1), config).run(" ".join(["word"] * 80))

    def test_wall_clock_budget_enforced(self):
        config = RLMConfig(max_wall_time_seconds=0.000001)
        with self.assertRaises(BudgetExceeded):
            RecursivePipeline(MockLLMClient(), config).run("short input")

    def test_trace_generation_and_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            result = RecursivePipeline(MockLLMClient(), RLMConfig()).run("one two three", trace_path=str(trace_path))
            self.assertEqual(result.trace_path, str(trace_path))
            data = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertIn("root", data["nodes"])
            self.assertIn("Trace", render_trace(trace_path))

    def test_trace_html_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            html_path = Path(tmp) / "trace.html"
            RecursivePipeline(MockLLMClient(), RLMConfig()).run("one two three", trace_path=str(trace_path))
            render_file(trace_path, html_path)
            rendered = html_path.read_text(encoding="utf-8")
            self.assertIn("Efficient RLM Trace", rendered)
            self.assertIn("root", rendered)

    def test_benchmark_runner_serializes_records(self):
        result = run_benchmark_suite("benchmarks/core/tasks.json", RLMConfig(), ["direct"])
        self.assertEqual(len(result["records"]), 4)
        self.assertIn("evaluation", result["records"][0])
        self.assertEqual(result["schema_version"], "2.0")
        self.assertEqual(result["repetitions"], 1)
        self.assertIn("aggregates", result)

    def test_benchmark_runner_keeps_raw_repetitions(self):
        result = run_benchmark_suite("benchmarks/core/tasks.json", RLMConfig(), ["direct"], repetitions=2)
        self.assertEqual(len(result["records"]), 8)
        self.assertEqual([record["repetition"] for record in result["records"][:2]], [1, 2])
        self.assertIn("latency_seconds", result["aggregates"]["by_mode"]["direct"])

    def test_keyword_recall(self):
        result = keyword_recall("alpha beta", ["alpha", "gamma"])
        self.assertEqual(result.score, 0.5)

    def test_required_fact_coverage(self):
        result = required_fact_coverage(
            "Alpha uses no network access. Gamma needs RLM_API_KEY.",
            ["Alpha", "Gamma", "Beta"],
        )
        self.assertEqual(result.score, 2 / 3)
        self.assertEqual(result.details["missing"], ["Beta"])

    def test_provider_metadata_normalization_mock(self):
        response = MockLLMClient().generate_response("summarize this")
        self.assertEqual(response.provider, "mock")
        self.assertIsNotNone(response.total_tokens)


if __name__ == "__main__":
    unittest.main()
