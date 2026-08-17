from __future__ import annotations

import re
from time import perf_counter

from efficient_rlm.llm.base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    """Deterministic fake backend for tests, demos, and benchmarks."""

    def generate(self, prompt: str) -> str:
        return self.generate_response(prompt).text

    def generate_response(self, prompt: str) -> LLMResponse:
        started = perf_counter()
        lower = prompt.lower()
        if "merge the two summaries" in lower:
            parts = re.findall(r"Summary [AB]:\n(.*?)(?=\n\nSummary [AB]:|\Z)", prompt, re.S)
            text = " ".join(part.strip() for part in parts if part.strip())
        elif "final polished summary" in lower:
            detailed = prompt.split("Detailed summary:", 1)[-1].strip()
            text = self._compact(detailed, limit=90)
        elif "semantic decomposition planner" in lower:
            text = self._semantic_json(prompt)
        elif "summarize chunk" in lower or "summarize the following text" in lower:
            source = prompt.split("Text:", 1)[-1] if "Text:" in prompt else prompt
            text = self._compact(source, limit=45)
        else:
            text = self._compact(prompt, limit=60)

        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split())
        return LLMResponse(
            text=text,
            model="mock-rlm",
            latency_seconds=perf_counter() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop",
            provider="mock",
        )

    def _compact(self, text: str, limit: int) -> str:
        words = re.findall(r"\S+", text)
        if not words:
            return ""
        compact = " ".join(words[:limit])
        if len(words) > limit:
            compact += "..."
        return compact

    def _semantic_json(self, prompt: str) -> str:
        text = prompt.split("Context:", 1)[-1] if "Context:" in prompt else prompt
        sections = [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()]
        if not sections:
            sections = [text.strip()]
        items = []
        for index, section in enumerate(sections[:4], start=1):
            safe = section.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
            difficulty = min(1.0, max(0.1, len(section.split()) / 80))
            items.append(
                f'{{"id":"s{index}","prompt":"Summarize semantic section {index}",'
                f'"context":"{safe}","estimated_difficulty":{difficulty:.2f},"dependencies":[]}}'
            )
        return '{"subproblems":[' + ",".join(items) + "]}"
