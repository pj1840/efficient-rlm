from __future__ import annotations

from efficient_rlm.models import Subtask


def chunk_text(text: str, chunk_size_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_size_words]) for i in range(0, len(words), chunk_size_words)]


def decompose_text(text: str, depth: int, chunk_size_words: int, max_children: int) -> list[Subtask]:
    chunks = chunk_text(text, chunk_size_words)[:max_children]
    return [Subtask(id=index, text=chunk, depth=depth + 1, index=index) for index, chunk in enumerate(chunks)]

