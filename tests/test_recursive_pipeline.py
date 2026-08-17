import unittest

from efficient_rlm.config import RLMConfig
from efficient_rlm.llm.base import LLMClient
from efficient_rlm.llm.mock import MockLLMClient
from efficient_rlm.recursive.decomposition import chunk_text, decompose_text
from efficient_rlm.recursive.pipeline import BudgetExceeded, RecursivePipeline


class ExplodingLLM(LLMClient):
    def generate(self, prompt: str) -> str:
        raise RuntimeError("boom")


def sample_text(words: int = 120) -> str:
    return " ".join(f"word{i}" for i in range(words))


class RecursivePipelineTests(unittest.TestCase):
    def test_decomposition_returns_valid_subproblems(self):
        chunks = chunk_text(sample_text(10), 4)
        self.assertEqual([len(chunk.split()) for chunk in chunks], [4, 4, 2])
        tasks = decompose_text(sample_text(10), depth=0, chunk_size_words=4, max_children=2)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].depth, 1)
        self.assertEqual(tasks[0].index, 0)

    def test_terminal_small_text_stops_without_decomposition(self):
        config = RLMConfig(execution_mode="sequential", chunk_size_words=50, min_chunk_words=10)
        result = RecursivePipeline(MockLLMClient(), config).run("short input")
        self.assertEqual(result.calls, 1)
        self.assertEqual(result.tasks, 1)
        self.assertEqual(result.max_depth_reached, 0)

    def test_max_call_budget_is_enforced(self):
        config = RLMConfig(execution_mode="sequential", chunk_size_words=5, max_calls=1)
        pipeline = RecursivePipeline(MockLLMClient(), config)
        with self.assertRaises(BudgetExceeded):
            pipeline.run(sample_text(30))

    def test_parallel_max_call_budget_is_enforced(self):
        config = RLMConfig(execution_mode="parallel", workers=4, chunk_size_words=5, max_calls=2)
        pipeline = RecursivePipeline(MockLLMClient(), config)
        with self.assertRaises(RuntimeError):
            pipeline.run(sample_text(60))

    def test_empty_input_behaves_sensibly(self):
        config = RLMConfig(execution_mode="parallel")
        result = RecursivePipeline(MockLLMClient(), config).run("")
        self.assertEqual(result.answer, "")
        self.assertEqual(result.calls, 1)
        self.assertEqual(result.tasks, 1)

    def test_sequential_execution_works(self):
        config = RLMConfig(execution_mode="sequential", chunk_size_words=20, max_depth=3)
        result = RecursivePipeline(MockLLMClient(), config).run(sample_text())
        self.assertTrue(result.answer)
        self.assertEqual(result.mode, "sequential")
        self.assertGreater(result.calls, 1)
        self.assertGreater(result.tasks, 1)

    def test_parallel_execution_works(self):
        config = RLMConfig(execution_mode="parallel", workers=4, chunk_size_words=20, max_depth=3)
        result = RecursivePipeline(MockLLMClient(), config).run(sample_text())
        self.assertTrue(result.answer)
        self.assertEqual(result.mode, "parallel")
        self.assertGreater(result.calls, 1)
        self.assertGreater(result.tasks, 1)

    def test_parallel_exceptions_are_reported(self):
        config = RLMConfig(execution_mode="parallel", workers=2, chunk_size_words=10)
        pipeline = RecursivePipeline(ExplodingLLM(), config)
        with self.assertRaises(RuntimeError):
            pipeline.run(sample_text(40))

    def test_parallel_aggregation_order_is_deterministic(self):
        text = sample_text(90)
        seq = RecursivePipeline(
            MockLLMClient(),
            RLMConfig(execution_mode="sequential", chunk_size_words=15),
        ).run(text)
        par = RecursivePipeline(
            MockLLMClient(),
            RLMConfig(execution_mode="parallel", workers=4, chunk_size_words=15),
        ).run(text)
        self.assertEqual(par.intermediate_results, seq.intermediate_results)
        self.assertEqual(par.answer, seq.answer)

    def test_repeated_parallel_runs_preserve_order(self):
        text = sample_text(120)
        config = RLMConfig(execution_mode="parallel", workers=4, chunk_size_words=12)
        answers = [
            RecursivePipeline(MockLLMClient(), config).run(text).intermediate_results
            for _ in range(5)
        ]
        self.assertTrue(all(item == answers[0] for item in answers))

    def test_curriculum_mode_runs_without_claiming_training(self):
        config = RLMConfig(execution_mode="parallel", workers=2, chunk_size_words=20, enable_curriculum=True)
        result = RecursivePipeline(MockLLMClient(), config).run_curriculum_summary(sample_text(80))
        self.assertTrue(result.answer)
        self.assertGreaterEqual(result.calls, 3)


if __name__ == "__main__":
    unittest.main()
