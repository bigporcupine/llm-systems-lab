import json
import os
import tempfile
import unittest
from pathlib import Path

from llm_systems_lab.experiment002 import run_backend_matrix
from llm_systems_lab.mock_server import MockServerConfig, start_in_thread


@unittest.skipUnless(
    os.environ.get("LLMS_LAB_SOCKET_TESTS") == "1",
    "set LLMS_LAB_SOCKET_TESTS=1 to allow localhost binding",
)
class Experiment002IntegrationTests(unittest.TestCase):
    def test_matrix_runs_against_openai_compatible_stream(self):
        server, thread = start_in_thread(
            MockServerConfig(prefill_ms=1, token_delay_ms=1, output_tokens=3)
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = {
                    "model": "controlled-mock",
                    "model_revision": "test-revision",
                    "dtype": "float16",
                    "input_size_hints": [8],
                    "output_tokens": 3,
                    "concurrency": [2],
                    "warmup_requests": 1,
                    "measured_requests": 2,
                    "repetitions": 1,
                    "ttft_slo_ms": 100,
                    "e2e_slo_ms": 100,
                    "seed": 7,
                }
                config_path = root / "config.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                summary = run_backend_matrix(
                    "mock",
                    f"http://127.0.0.1:{server.server_port}/v1",
                    config_path,
                    root / "results",
                    "test-version",
                )
                self.assertEqual(summary["backend_version"], "test-version")
                self.assertEqual(len(summary["cells"]), 1)
                self.assertTrue((root / "results" / "summary.json").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
