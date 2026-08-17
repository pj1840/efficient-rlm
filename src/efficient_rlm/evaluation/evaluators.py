from __future__ import annotations

from dataclasses import dataclass

from efficient_rlm.llm.base import LLMClient


@dataclass(frozen=True)
class EvaluationResult:
    method: str
    score: float
    details: dict


def keyword_recall(answer: str, keywords: list[str]) -> EvaluationResult:
    normalized = answer.lower()
    found = [keyword for keyword in keywords if keyword.lower() in normalized]
    score = len(found) / len(keywords) if keywords else 0.0
    return EvaluationResult(
        method="keyword_recall",
        score=score,
        details={"found": found, "total_keywords": len(keywords)},
    )


def required_fact_coverage(answer: str, facts: list[str]) -> EvaluationResult:
    normalized = " ".join(answer.lower().split())
    normalized_facts = [" ".join(fact.lower().split()) for fact in facts]
    found = [fact for fact, normalized_fact in zip(facts, normalized_facts) if normalized_fact in normalized]
    score = len(found) / len(facts) if facts else 0.0
    return EvaluationResult(
        method="required_fact_coverage",
        score=score,
        details={"found": found, "missing": [fact for fact in facts if fact not in found], "total_facts": len(facts)},
    )


def exact_match(answer: str, reference: str) -> EvaluationResult:
    score = 1.0 if answer.strip().lower() == reference.strip().lower() else 0.0
    return EvaluationResult(method="exact_match", score=score, details={})


def llm_as_judge(answer: str, reference: str, judge: LLMClient, model_name: str | None = None) -> EvaluationResult:
    prompt = f"""
Judge whether the candidate answer captures the reference answer.
Return only a number from 0 to 1.

Reference:
{reference}

Candidate:
{answer}
""".strip()
    response = judge.generate_response(prompt)
    try:
        score = max(0.0, min(1.0, float(response.text.strip().split()[0])))
    except (ValueError, IndexError):
        score = 0.0
    return EvaluationResult(
        method="llm_as_judge",
        score=score,
        details={
            "judge_model": response.model or model_name,
            "judge_provider": response.provider,
            "judge_prompt": prompt,
            "raw_judge_response": response.text,
        },
    )


def evaluate_answer(task: dict, answer: str) -> EvaluationResult:
    method = task.get("evaluation", "keyword_recall")
    if method == "exact_match":
        return exact_match(answer, str(task.get("reference", "")))
    if method == "keyword_recall":
        return keyword_recall(answer, [str(item) for item in task.get("keywords", [])])
    if method == "required_fact_coverage":
        facts = task.get("required_facts", task.get("keywords", []))
        return required_fact_coverage(answer, [str(item) for item in facts])
    return EvaluationResult(method="none", score=0.0, details={"reason": f"unsupported evaluation={method}"})
