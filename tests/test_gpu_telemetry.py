import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_systems_lab.gpu_telemetry import GpuTelemetrySampler, collect_nvidia_sample


class GpuTelemetryTests(unittest.TestCase):
    @patch("llm_systems_lab.gpu_telemetry.subprocess.run")
    def test_collects_all_visible_gpus_and_preserves_unsupported_values(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="0, 82, 12000, 23000, 145.5, 61\n1, 40, 8000, 23000, N/A, 55\n",
            stderr="",
        )
        sample = collect_nvidia_sample()
        self.assertTrue(sample["available"])
        self.assertEqual(len(sample["gpus"]), 2)
        self.assertEqual(sample["gpus"][0]["utilization_gpu"], 82.0)
        self.assertEqual(sample["gpus"][0]["power_draw"], 145.5)
        self.assertIsNone(sample["gpus"][1]["power_draw"])

    def test_sampler_writes_raw_series_and_unavailable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gpu-telemetry.json"
            sampler = GpuTelemetrySampler(
                output, interval_seconds=0.001,
                collector=lambda: {"available": False, "error": "no GPU", "gpus": []},
            ).start()
            time.sleep(0.01)
            artifact = sampler.stop()
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(artifact["available"])
            self.assertEqual(persisted["sample_count"], 1)
            self.assertEqual(persisted["samples"][0]["error"], "no GPU")


if __name__ == "__main__":
    unittest.main()
