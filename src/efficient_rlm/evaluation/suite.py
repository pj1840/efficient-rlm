from __future__ import annotations

import json
from pathlib import Path


def load_suite(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("benchmark suite must be a list of tasks")
    required = {"task_id", "input", "evaluation"}
    for item in data:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"invalid benchmark task: {item}")
    return data
