from __future__ import annotations

from efficient_rlm.models import Subtask


class CurriculumScheduler:
    """Orders subtasks from shorter/easier chunks to longer chunks.

    This is an inference-time curriculum scheduler, not model training.
    Results are restored to original chunk order before aggregation.
    """

    def order(self, tasks: list[Subtask]) -> list[Subtask]:
        return sorted(tasks, key=lambda task: (task.word_count, task.index))

