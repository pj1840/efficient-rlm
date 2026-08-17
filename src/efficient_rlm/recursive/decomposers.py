from __future__ import annotations

import json
from abc import ABC, abstractmethod

from efficient_rlm.config import RLMConfig
from efficient_rlm.llm.base import LLMClient
from efficient_rlm.models import Metrics, Subtask
from efficient_rlm.recursive.decomposition import decompose_text


class DecompositionError(RuntimeError):
    pass


class Decomposer(ABC):
    @abstractmethod
    def decompose(
        self,
        task: str,
        context: str,
        depth: int,
        parent_id: str | None,
        metrics: Metrics | None = None,
    ) -> list[Subtask]:
        raise NotImplementedError


class FixedChunkDecomposer(Decomposer):
    def __init__(self, config: RLMConfig) -> None:
        self.config = config

    def decompose(
        self,
        task: str,
        context: str,
        depth: int,
        parent_id: str | None,
        metrics: Metrics | None = None,
    ) -> list[Subtask]:
        return decompose_text(
            text=context,
            depth=depth,
            chunk_size_words=self.config.chunk_size_words,
            max_children=self.config.max_children,
        )


class SemanticDecomposer(Decomposer):
    def __init__(self, llm: LLMClient, config: RLMConfig, fallback: Decomposer | None = None) -> None:
        self.llm = llm
        self.config = config
        self.fallback = fallback

    def decompose(
        self,
        task: str,
        context: str,
        depth: int,
        parent_id: str | None,
        metrics: Metrics | None = None,
    ) -> list[Subtask]:
        prompt = build_semantic_decomposition_prompt(task, context, self.config.max_children)
        estimated_prompt_tokens = len(prompt.split())
        if metrics is not None:
            metrics.reserve_request(
                max_calls=self.config.max_calls,
                estimated_prompt_tokens=estimated_prompt_tokens,
                max_prompt_tokens=self.config.max_prompt_tokens,
                max_total_tokens=self.config.max_total_tokens,
            )
        response = self.llm.generate_response(prompt)
        if metrics is not None:
            metrics.record_response(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                reserved_prompt_tokens=estimated_prompt_tokens,
            )
            for _ in range(response.retries):
                metrics.record_retry()
        raw = response.text
        try:
            data = json.loads(raw)
            items = data["subproblems"]
            if not isinstance(items, list):
                raise DecompositionError("subproblems must be a list")
            tasks = []
            for index, item in enumerate(items[: self.config.max_children]):
                if not isinstance(item, dict):
                    raise DecompositionError("subproblem entries must be objects")
                sub_context = str(item.get("context") or "").strip()
                sub_prompt = str(item.get("prompt") or task).strip()
                if not sub_context:
                    raise DecompositionError("semantic subproblem missing context")
                dependencies = item.get("dependencies") or []
                if not isinstance(dependencies, list):
                    dependencies = []
                difficulty = item.get("estimated_difficulty")
                tasks.append(
                    Subtask(
                        id=str(item.get("id") or f"s{index + 1}"),
                        text=sub_context,
                        depth=depth + 1,
                        index=index,
                        parent_id=parent_id,
                        prompt=sub_prompt,
                        estimated_difficulty=float(difficulty) if isinstance(difficulty, (int, float)) else None,
                        dependencies=[str(dep) for dep in dependencies],
                    )
                )
            if not tasks:
                raise DecompositionError("semantic decomposition returned no subproblems")
            return tasks
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, DecompositionError):
            if self.fallback is not None:
                return self.fallback.decompose(task, context, depth, parent_id, metrics=metrics)
            raise


def build_semantic_decomposition_prompt(task: str, context: str, max_children: int) -> str:
    return f"""
You are a semantic decomposition planner.

Split the context into at most {max_children} logically independent subproblems for the task.
Return only valid JSON with this shape:
{{"subproblems":[{{"id":"s1","prompt":"...","context":"...","estimated_difficulty":0.5,"dependencies":[]}}]}}

Task:
{task}

Context:
{context}
""".strip()


def build_decomposer(config: RLMConfig, llm: LLMClient) -> Decomposer:
    fixed = FixedChunkDecomposer(config)
    if config.decomposer == "semantic":
        return SemanticDecomposer(llm, config, fallback=fixed)
    return fixed
