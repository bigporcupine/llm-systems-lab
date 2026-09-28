import json
import tempfile
import unittest
from pathlib import Path

from llm_systems_lab.experiment002 import load_config, run_backend_matrix
from llm_systems_lab.workloads import deterministic_prompt


class Experiment002Tests(unittest.TestCase):
    def test_workload_is_deterministic_and_seeded(self):
        first = deterministic_prompt(20, seed=7)
        self.assertEqual(first, deterministic_prompt(20, seed=7))
        self.assertNotEqual(first, deterministic_prompt(20, seed=8))

    def test_workload_rejects_nonpositive_size(self):
        with self.assertRaises(ValueError):
            deterministic_prompt(0)

    def test_config_reports_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"model": "test"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_revision"):
                load_config(path)

    def test_formal_runner_rejects_unknown_backend_version_before_execution(self):
        with self.assertRaisesRegex(ValueError, "backend_version"):
            run_backend_matrix("vllm", "http://unused", Path("unused.json"), Path("unused"), "unknown")


if __name__ == "__main__":
    unittest.main()
