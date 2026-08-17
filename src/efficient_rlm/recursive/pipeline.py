from __future__ import annotations

from efficient_rlm.config import RLMConfig
from efficient_rlm.llm.base import LLMClient
from efficient_rlm.models import Metrics, RLMResult, Subtask
from efficient_rlm.recursive.aggregation import (
    build_chunk_summary_prompt,
    build_final_refine_prompt,
    build_pair_merge_prompt,
)
from efficient_rlm.recursive.decomposition import decompose_text
from efficient_rlm.recursive.executor import execute_tasks
from efficient_rlm.recursive.stopping import should_stop
from efficient_rlm.scheduling.curriculum import CurriculumScheduler


class BudgetExceeded(RuntimeError):
    pass


class RecursivePipeline:
    def __init__(self, llm: LLMClient, config: RLMConfig) -> None:
        config.validate()
        self.llm = llm
        self.config = config
        self.scheduler = CurriculumScheduler() if config.enable_curriculum else None

    def run(self, text: str, task: str = "summarize") -> RLMResult:
        if task != "summarize":
            raise ValueError("Only summarization is currently implemented")
        metrics = Metrics()
        answer, intermediates = self._summarize_recursive(text=text, depth=0, metrics=metrics)
        return RLMResult(
            answer=answer,
            mode=self.config.execution_mode,
            calls=metrics.calls,
            tasks=metrics.tasks,
            max_depth_reached=metrics.max_depth_reached,
            wall_time_seconds=metrics.elapsed(),
            intermediate_results=intermediates,
        )

    def run_curriculum_summary(self, text: str) -> RLMResult:
        metrics = Metrics()
        coarse = self._call_llm(
            build_chunk_summary_prompt(text, index=0, total=1),
            metrics=metrics,
        )
        detailed, intermediates = self._summarize_recursive(
            text=text,
            depth=0,
            metrics=metrics,
            guidance=f"Preserve these coarse themes: {coarse}",
        )
        final = self._call_llm(build_final_refine_prompt(coarse, detailed), metrics=metrics)
        return RLMResult(
            answer=final,
            mode=self.config.execution_mode,
            calls=metrics.calls,
            tasks=metrics.tasks,
            max_depth_reached=metrics.max_depth_reached,
            wall_time_seconds=metrics.elapsed(),
            intermediate_results=[coarse, *intermediates, detailed],
        )

    def _summarize_recursive(
        self,
        text: str,
        depth: int,
        metrics: Metrics,
        guidance: str | None = None,
    ) -> tuple[str, list[str]]:
        metrics.record_task(depth)
        if should_stop(text, depth, metrics.calls, self.config):
            prompt = build_chunk_summary_prompt(text, index=0, total=1, guidance=guidance)
            return self._call_llm(prompt, metrics), []

        children = decompose_text(
            text=text,
            depth=depth,
            chunk_size_words=self.config.chunk_size_words,
            max_children=self.config.max_children,
        )
        if len(children) <= 1:
            prompt = build_chunk_summary_prompt(text, index=0, total=1, guidance=guidance)
            return self._call_llm(prompt, metrics), []

        scheduled = self.scheduler.order(children) if self.scheduler else children

        def run_child(task: Subtask) -> str:
            metrics.record_task(task.depth)
            prompt = build_chunk_summary_prompt(
                task.text,
                index=task.index,
                total=len(children),
                guidance=guidance,
            )
            return self._call_llm(prompt, metrics)

        summaries_by_original_order = execute_tasks(scheduled, run_child, self.config)
        final = self._reduce_summaries(summaries_by_original_order, depth + 1, metrics)
        return final, summaries_by_original_order

    def _reduce_summaries(self, summaries: list[str], depth: int, metrics: Metrics) -> str:
        current = [summary for summary in summaries if summary.strip()]
        while len(current) > 1:
            pairs = [(current[i], current[i + 1]) for i in range(0, len(current) - 1, 2)]
            carry = current[-1] if len(current) % 2 else None
            tasks = [
                Subtask(id=index, text=build_pair_merge_prompt(a, b), depth=depth, index=index)
                for index, (a, b) in enumerate(pairs)
            ]

            def merge(task: Subtask) -> str:
                metrics.record_task(task.depth)
                return self._call_llm(task.text, metrics)

            merged = execute_tasks(tasks, merge, self.config)
            current = merged + ([carry] if carry is not None else [])
            depth += 1
            if depth > self.config.max_depth + 8:
                raise RuntimeError("Aggregation exceeded safety depth")
        return current[0] if current else ""

    def _call_llm(self, prompt: str, metrics: Metrics) -> str:
        try:
            metrics.reserve_call(self.config.max_calls)
        except RuntimeError as exc:
            raise BudgetExceeded(str(exc)) from exc
        response = self.llm.generate(prompt)
        return response.strip()
