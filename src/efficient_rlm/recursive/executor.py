from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from efficient_rlm.config import RLMConfig
from efficient_rlm.models import Subtask


class ExecutionError(RuntimeError):
    pass


def execute_tasks(
    tasks: list[Subtask],
    fn: Callable[[Subtask], str],
    config: RLMConfig,
) -> list[str]:
    if config.execution_mode == "sequential" or len(tasks) <= 1:
        return [fn(task) for task in tasks]

    results: list[str | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=min(config.workers, len(tasks))) as pool:
        future_to_task = {pool.submit(fn, task): task for task in tasks}
        for future in as_completed(future_to_task, timeout=config.timeout_seconds * max(1, len(tasks))):
            task = future_to_task[future]
            try:
                results[task.index] = future.result(timeout=config.timeout_seconds)
            except Exception as exc:
                raise ExecutionError(f"Task {task.index} failed: {exc}") from exc

    missing = [index for index, value in enumerate(results) if value is None]
    if missing:
        raise ExecutionError(f"Missing results for task indexes: {missing}")
    return [str(value) for value in results]

