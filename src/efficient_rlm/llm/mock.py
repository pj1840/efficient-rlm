from __future__ import annotations

import re

from efficient_rlm.llm.base import LLMClient


class MockLLMClient(LLMClient):
    """Deterministic fake backend for tests, demos, and benchmarks."""

    def generate(self, prompt: str) -> str:
        lower = prompt.lower()
        if "merge the two summaries" in lower:
            parts = re.findall(r"Summary [AB]:\n(.*?)(?=\n\nSummary [AB]:|\Z)", prompt, re.S)
            return " ".join(part.strip() for part in parts if part.strip())
        if "final polished summary" in lower:
            detailed = prompt.split("Detailed summary:", 1)[-1].strip()
            return self._compact(detailed, limit=90)
        if "summarize chunk" in lower or "summarize the following text" in lower:
            text = prompt.split("Text:", 1)[-1] if "Text:" in prompt else prompt
            return self._compact(text, limit=45)
        return self._compact(prompt, limit=60)

    def _compact(self, text: str, limit: int) -> str:
        words = re.findall(r"\S+", text)
        if not words:
            return ""
        compact = " ".join(words[:limit])
        if len(words) > limit:
            compact += "..."
        return compact

