import tempfile
import unittest
from pathlib import Path

from efficient_rlm.config import load_config


class ConfigTests(unittest.TestCase):
    def test_config_loads_defaults(self):
        config = load_config("configs/default.yaml")
        self.assertEqual(config.provider, "mock")
        self.assertEqual(config.execution_mode, "threaded")
        self.assertEqual(config.workers, 4)

    def test_parallel_alias_maps_to_threaded(self):
        config = load_config(None, execution_mode="parallel")
        self.assertEqual(config.execution_mode, "threaded")

    def test_config_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("workers: 2\nexecution_mode: sequential\n", encoding="utf-8")
            config = load_config(path, workers=3)
        self.assertEqual(config.execution_mode, "sequential")
        self.assertEqual(config.workers, 3)


if __name__ == "__main__":
    unittest.main()
