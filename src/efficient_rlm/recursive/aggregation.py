from __future__ import annotations


def build_chunk_summary_prompt(chunk: str, index: int, total: int, guidance: str | None = None) -> str:
    guidance_block = f"\nGuidance:\n{guidance}\n" if guidance else ""
    return f"""
Summarize chunk {index + 1} of {total} in 1-2 concise sentences.

Rules:
- Return only the summary.
- Preserve important details.
{guidance_block}
Text:
{chunk}
""".strip()


def build_pair_merge_prompt(summary_a: str, summary_b: str) -> str:
    return f"""
Merge the two summaries below into one shorter, coherent summary.

Rules:
- Return only the merged summary.
- Preserve important meaning from both inputs.

Summary A:
{summary_a}

Summary B:
{summary_b}
""".strip()


def build_final_refine_prompt(coarse_summary: str, detailed_summary: str) -> str:
    return f"""
Create one final polished summary from the coarse and detailed summaries.
Return only the final polished summary.

Coarse summary:
{coarse_summary}

Detailed summary:
{detailed_summary}
""".strip()

