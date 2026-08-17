import unittest

from efficient_rlm.evaluation.metrics import speedup
from efficient_rlm.models import Subtask
from efficient_rlm.scheduling.curriculum import CurriculumScheduler


class SchedulerAndMetricsTests(unittest.TestCase):
    def test_curriculum_orders_shorter_tasks_first(self):
        tasks = [
            Subtask(id=0, text="one two three", depth=1, index=0),
            Subtask(id=1, text="one", depth=1, index=1),
            Subtask(id=2, text="one two", depth=1, index=2),
        ]
        ordered = CurriculumScheduler().order(tasks)
        self.assertEqual([task.index for task in ordered], [1, 2, 0])

    def test_speedup_metric(self):
        self.assertEqual(speedup(10.0, 2.0), 5.0)
        self.assertIsNone(speedup(0.0, 2.0))


if __name__ == "__main__":
    unittest.main()

