from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from efficient_rlm.config import RLMConfig
from efficient_rlm.models import Subtask


class ExecutionError(RuntimeError):
    pass


def execute_tasks(
    tasks: list[Subtask],
    fn: Callable[[Subtask], str],
    config: RLMConfig,
) -> list[str]:
    if any(task.dependencies for task in tasks):
        return execute_tasks_with_dependencies(tasks, fn, config)
    return _execute_independent_tasks(tasks, fn, config)


def _execute_independent_tasks(
    tasks: list[Subtask],
    fn: Callable[[Subtask], str],
    config: RLMConfig,
) -> list[str]:
    if config.execution_mode == "sequential" or len(tasks) <= 1:
        results = {task.index: fn(task) for task in tasks}
        return [results[task.index] for task in sorted(tasks, key=lambda item: item.index)]
    if config.execution_mode == "async":
        return asyncio.run(execute_tasks_async(tasks, fn, config))

    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=min(config.workers, len(tasks))) as pool:
        future_to_task = {pool.submit(fn, task): task for task in tasks}
        for future in as_completed(future_to_task, timeout=config.timeout_seconds * max(1, len(tasks))):
            task = future_to_task[future]
            try:
                results[task.index] = future.result(timeout=config.timeout_seconds)
            except Exception as exc:
                raise ExecutionError(f"Task {task.index} failed: {exc}") from exc

    expected = [task.index for task in tasks]
    missing = [index for index in expected if index not in results]
    if missing:
        raise ExecutionError(f"Missing results for task indexes: {missing}")
    return [results[task.index] for task in sorted(tasks, key=lambda item: item.index)]


def validate_dependencies(tasks: list[Subtask]) -> None:
    ids = [str(task.id) for task in tasks]
    if len(ids) != len(set(ids)):
        raise ExecutionError("Duplicate task IDs in dependency graph")
    id_set = set(ids)
    for task in tasks:
        task_id = str(task.id)
        deps = {str(dep) for dep in task.dependencies}
        if task_id in deps:
            raise ExecutionError(f"Task {task_id} depends on itself")
        missing = sorted(deps - id_set)
        if missing:
            raise ExecutionError(f"Task {task_id} has missing dependencies: {missing}")


def execute_tasks_with_dependencies(
    tasks: list[Subtask],
    fn: Callable[[Subtask], str],
    config: RLMConfig,
) -> list[str]:
    validate_dependencies(tasks)
    remaining = {str(task.id): task for task in tasks}
    completed: set[str] = set()
    results: dict[int, str] = {}

    while remaining:
        ready = [
            task
            for task_id, task in sorted(remaining.items(), key=lambda item: item[1].index)
            if set(map(str, task.dependencies)).issubset(completed)
        ]
        if not ready:
            raise ExecutionError("Cycle detected in task dependency graph")
        wave_results = _execute_independent_tasks(ready, fn, config)
        for task, result in zip(sorted(ready, key=lambda item: item.index), wave_results):
            results[task.index] = result
            completed.add(str(task.id))
            remaining.pop(str(task.id))

    return [results[task.index] for task in sorted(tasks, key=lambda item: item.index)]


async def execute_tasks_async(
    tasks: list[Subtask],
    fn: Callable[[Subtask], str],
    config: RLMConfig,
) -> list[str]:
    if not tasks:
        return []

    semaphore = asyncio.Semaphore(min(config.workers, len(tasks)))
    results: dict[int, str] = {}

    async def run_one(task: Subtask) -> None:
        last_error: Exception | None = None
        for attempt in range(config.max_retries + 1):
            try:
                async with semaphore:
                    value = await asyncio.wait_for(
                        asyncio.to_thread(fn, task),
                        timeout=config.timeout_seconds,
                    )
                results[task.index] = value
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < config.max_retries:
                    await asyncio.sleep(0.1 * (2**attempt))
        raise ExecutionError(f"Task {task.index} failed: {last_error}")

    pending = [asyncio.create_task(run_one(task)) for task in tasks]
    try:
        await asyncio.gather(*pending)
    except Exception:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise

    expected = [task.index for task in tasks]
    missing = [index for index in expected if index not in results]
    if missing:
        raise ExecutionError(f"Missing results for task indexes: {missing}")
    return [results[task.index] for task in sorted(tasks, key=lambda item: item.index)]
