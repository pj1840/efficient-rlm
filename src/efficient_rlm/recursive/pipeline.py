from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from efficient_rlm.config import RLMConfig
from efficient_rlm.llm.base import LLMClient
from efficient_rlm.models import Metrics, RLMResult, Subtask
from efficient_rlm.recursive.aggregation import (
    build_chunk_summary_prompt,
    build_final_refine_prompt,
    build_pair_merge_prompt,
)
from efficient_rlm.recursive.decomposers import build_decomposer
from efficient_rlm.recursive.executor import execute_tasks
from efficient_rlm.recursive.policy import build_policy
from efficient_rlm.scheduling.curriculum import CurriculumScheduler
from efficient_rlm.tracing import TraceRecorder


class BudgetExceeded(RuntimeError):
    pass


class RecursivePipeline:
    def __init__(self, llm: LLMClient, config: RLMConfig) -> None:
        config.validate()
        self.llm = llm
        self.config = config
        self.scheduler = CurriculumScheduler() if config.enable_curriculum else None
        self.decomposer = build_decomposer(config, llm)
        self.policy = build_policy(config)

    def run(self, text: str, task: str = "summarize", trace_path: str | None = None) -> RLMResult:
        if task != "summarize":
            raise ValueError("Only summarization is currently implemented")
        metrics = Metrics()
        trace = TraceRecorder(asdict(self.config)) if trace_path else None
        answer, intermediates = self._summarize_recursive(
            text=text,
            depth=0,
            metrics=metrics,
            node_id="root",
            parent_id=None,
            trace=trace,
        )
        result = self._build_result(answer, intermediates, metrics)
        if trace is not None:
            trace.finalize(answer, asdict(result))
            result.trace_path = trace.save(trace_path or Path(self.config.results_dir) / "trace.json")
        return result

    def run_curriculum_summary(self, text: str, trace_path: str | None = None) -> RLMResult:
        metrics = Metrics()
        trace = TraceRecorder(asdict(self.config)) if trace_path else None
        if trace is not None:
            trace.add_node("root", None, 0, "coarse-to-detailed summarization")
        coarse = self._call_llm(
            build_chunk_summary_prompt(text, index=0, total=1),
            metrics=metrics,
            node_id="root",
            trace=trace,
        )
        detailed, intermediates = self._summarize_recursive(
            text=text,
            depth=0,
            metrics=metrics,
            guidance=f"Preserve these coarse themes: {coarse}",
            node_id="detail",
            parent_id="root" if trace is not None else None,
            trace=trace,
        )
        final = self._call_llm(
            build_final_refine_prompt(coarse, detailed),
            metrics=metrics,
            node_id="root",
            trace=trace,
            aggregation_status="final_refine",
        )
        result = self._build_result(final, [coarse, *intermediates, detailed], metrics)
        if trace is not None:
            trace.finalize(final, asdict(result))
            result.trace_path = trace.save(trace_path or Path(self.config.results_dir) / "trace.json")
        return result

    def _build_result(self, answer: str, intermediates: list[str], metrics: Metrics) -> RLMResult:
        prompt_tokens, completion_tokens, total_tokens = metrics.token_summary()
        return RLMResult(
            answer=answer,
            mode=self.config.execution_mode,
            calls=metrics.calls,
            successful_calls=metrics.successful_calls,
            failed_calls=metrics.failed_calls,
            retries=metrics.retries,
            tasks=metrics.tasks,
            max_depth_reached=metrics.max_depth_reached,
            wall_time_seconds=metrics.elapsed(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            decomposition_count=metrics.decomposition_count,
            aggregation_count=metrics.aggregation_count,
            budget_termination_reason=metrics.budget_termination_reason,
            intermediate_results=intermediates,
        )

    def _summarize_recursive(
        self,
        text: str,
        depth: int,
        metrics: Metrics,
        guidance: str | None = None,
        node_id: str = "root",
        parent_id: str | None = None,
        trace: TraceRecorder | None = None,
    ) -> tuple[str, list[str]]:
        metrics.record_task(depth)
        if trace is not None and node_id not in trace.trace.nodes:
            trace.add_node(node_id, parent_id, depth, text)

        decision = self.policy.decide(text, depth, metrics, self.config)
        if decision.action in {"ANSWER_DIRECTLY", "STOP"}:
            if decision.action == "STOP":
                metrics.mark_budget_stop(decision.reason)
            prompt = build_chunk_summary_prompt(text, index=0, total=1, guidance=guidance)
            answer = self._call_llm(prompt, metrics, node_id=node_id, trace=trace, stopping_reason=decision.reason)
            return answer, []

        metrics.record_decomposition()
        if trace is not None:
            trace.mark_decomposed(node_id)
        children = self.decomposer.decompose("summarize", text, depth, parent_id=node_id, metrics=metrics)
        self._check_token_budgets(metrics)
        if len(children) <= 1:
            prompt = build_chunk_summary_prompt(text, index=0, total=1, guidance=guidance)
            answer = self._call_llm(prompt, metrics, node_id=node_id, trace=trace, stopping_reason="single_child")
            return answer, []

        scheduled = self.scheduler.order(children) if self.scheduler else children

        def run_child(task: Subtask) -> str:
            child_id = f"{node_id}.{task.index}"
            answer, _ = self._summarize_recursive(
                text=task.text,
                depth=task.depth,
                metrics=metrics,
                guidance=guidance,
                node_id=child_id,
                parent_id=node_id,
                trace=trace,
            )
            return answer

        summaries_by_original_order = execute_tasks(scheduled, run_child, self.config)
        final = self._reduce_summaries(summaries_by_original_order, depth + 1, metrics)
        if trace is not None:
            trace.finish_node(node_id, status="aggregated", output=final, aggregation_status="pairwise_merge")
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
                metrics.record_aggregation()
                return self._call_llm(task.text, metrics, aggregation_status="pairwise_merge")

            merged = execute_tasks(tasks, merge, self.config)
            current = merged + ([carry] if carry is not None else [])
            depth += 1
            if depth > self.config.max_depth + 8:
                raise RuntimeError("Aggregation exceeded safety depth")
        return current[0] if current else ""

    def _call_llm(
        self,
        prompt: str,
        metrics: Metrics,
        node_id: str | None = None,
        trace: TraceRecorder | None = None,
        stopping_reason: str | None = None,
        aggregation_status: str | None = None,
    ) -> str:
        if self.config.max_wall_time_seconds is not None and metrics.elapsed() >= self.config.max_wall_time_seconds:
            reason = f"max_wall_time_seconds={self.config.max_wall_time_seconds}"
            metrics.mark_budget_stop(reason)
            raise BudgetExceeded(reason)
        estimated_prompt_tokens = len(prompt.split())
        try:
            metrics.reserve_request(
                max_calls=self.config.max_calls,
                estimated_prompt_tokens=estimated_prompt_tokens,
                max_prompt_tokens=self.config.max_prompt_tokens,
                max_total_tokens=self.config.max_total_tokens,
            )
        except RuntimeError as exc:
            metrics.mark_budget_stop(str(exc))
            raise BudgetExceeded(str(exc)) from exc
        try:
            response = self.llm.generate_response(prompt)
            metrics.record_response(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                reserved_prompt_tokens=estimated_prompt_tokens,
            )
            for _ in range(response.retries):
                metrics.record_retry()
            self._check_token_budgets(metrics)
            if trace is not None and node_id is not None:
                trace.finish_node(
                    node_id,
                    status="completed",
                    output=response.text,
                    provider=response.provider,
                    model=response.model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    stopping_reason=stopping_reason,
                    aggregation_status=aggregation_status,
                )
            return response.text.strip()
        except Exception as exc:
            metrics.record_failure()
            if trace is not None and node_id is not None:
                trace.finish_node(node_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

    def _check_token_budgets(self, metrics: Metrics) -> None:
        checks = (
            ("max_prompt_tokens", metrics.prompt_tokens),
            ("max_completion_tokens", metrics.completion_tokens),
            ("max_total_tokens", metrics.total_tokens),
        )
        for field_name, used in checks:
            limit = getattr(self.config, field_name)
            if limit is not None and used > limit:
                reason = f"{field_name}={limit}"
                metrics.mark_budget_stop(reason)
                raise BudgetExceeded(reason)
