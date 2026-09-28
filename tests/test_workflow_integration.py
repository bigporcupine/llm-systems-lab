import json
import os
import tempfile
import unittest
from pathlib import Path

from llm_systems_lab.cache_experiment import run_prefix_cache_experiment
from llm_systems_lab.evaluation import evaluate_endpoint
from llm_systems_lab.mock_server import MockServerConfig, start_in_thread
from llm_systems_lab.speculative_experiment import run_speculative_comparison


@unittest.skipUnless(
    os.environ.get("LLMS_LAB_SOCKET_TESTS") == "1",
    "set LLMS_LAB_SOCKET_TESTS=1 to allow localhost binding",
)
class WorkflowIntegrationTests(unittest.TestCase):
    def setUp(self):
        config = MockServerConfig(prefill_ms=1, token_delay_ms=1, output_tokens=2)
        self.first, self.first_thread = start_in_thread(config)
        self.second, self.second_thread = start_in_thread(config)
        self.first_url = f"http://127.0.0.1:{self.first.server_port}/v1"
        self.second_url = f"http://127.0.0.1:{self.second.server_port}/v1"

    def tearDown(self):
        for server, thread in (
            (self.first, self.first_thread), (self.second, self.second_thread)
        ):
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_quality_workflow_writes_predictions_and_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "evaluation.jsonl"
            dataset.write_text(
                json.dumps({"id": "one", "prompt": "answer", "references": ["xx"]}) + "\n",
                encoding="utf-8",
            )
            artifact = evaluate_endpoint(
                self.first_url, "mock", dataset, root / "out", max_tokens=2
            )
            self.assertEqual(artifact["scores"]["exact_match"], 1.0)
            self.assertEqual(artifact["predictions"]["one"], "xx")
            self.assertEqual(len(artifact["traces"]), 1)
            self.assertTrue((root / "out" / "evaluation.json").is_file())

    def test_prefix_cache_workflow_writes_raw_break_even_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            artifact = run_prefix_cache_experiment(
                self.first_url, self.second_url, "mock", [8], [1, 2],
                output_tokens=2, concurrency=2, repetitions=1,
                timeout_seconds=10, output_dir=output,
            )
            self.assertEqual(len(artifact["rows"]), 2)
            self.assertEqual(len(artifact["raw_runs"]), 4)
            self.assertTrue(all(run["traces"] for run in artifact["raw_runs"]))
            self.assertTrue((output / "prefix-cache.json").is_file())

    def test_speculative_workflow_writes_each_raw_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            artifact = run_speculative_comparison(
                self.first_url, self.second_url, "mock", ["general"], [8],
                output_tokens=2, concurrency=2, measured_requests=2,
                repetitions=1, timeout_seconds=10, output_dir=output,
            )
            self.assertEqual(len(artifact["cells"]), 2)
            self.assertEqual(len(list(output.glob("target-only_*.json"))), 1)
            self.assertEqual(len(list(output.glob("speculative_*.json"))), 1)
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
