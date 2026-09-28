import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from llm_systems_lab.canary import assign_lane, should_rollback
from llm_systems_lab.capacity import capacity_plan, cost_per_million_tokens
from llm_systems_lab.models import RequestTrace
from llm_systems_lab.online import OnlineRun
from llm_systems_lab.production_experiment import run_gateway_load


class ProductionTests(unittest.TestCase):
    def test_assignment_is_sticky(self):
        self.assertEqual(assign_lane("request-7", 0.1), assign_lane("request-7", 0.1))

    def test_rollback_guardrail(self):
        stable = {"requests": 1000, "error_rate": 0.01, "p95_latency_ms": 100, "quality": 0.9}
        canary = {"requests": 100, "error_rate": 0.05, "p95_latency_ms": 110, "quality": 0.9}
        policy = {"minimum_requests": 100, "max_error_rate": 0.02, "max_p95_latency_regression": 0.2, "max_quality_regression": 0.03}
        self.assertTrue(should_rollback(stable, canary, policy)["rollback"])

    def test_capacity_and_cost(self):
        self.assertEqual(capacity_plan(10, 5, 0.5)["replicas"], 4)
        self.assertAlmostEqual(cost_per_million_tokens(1.0, 1000), 1 / 3.6)

    @patch("llm_systems_lab.production_experiment.capture_environment", return_value={"git_commit": "abc"})
    @patch("llm_systems_lab.production_experiment.fetch_prometheus_text", side_effect=["requests 0\n", "requests 2\n"])
    @patch("llm_systems_lab.production_experiment.run_online_benchmark")
    def test_load_uses_slo_goodput_and_records_price_provenance(self, benchmark, _fetch, _environment):
        benchmark.return_value = OnlineRun(traces=[
            RequestTrace("ok", 100, 10, 100, 20, [20, 100]),
            RequestTrace("slow", 100, 10, 900, 500, [500, 900]),
        ], wall_time_ms=1000)
        with tempfile.TemporaryDirectory() as directory:
            artifact = run_gateway_load(
                "http://gateway/v1", "http://gateway/metrics", "model", "d" * 40,
                "engine-1", "gateway-1", 2, 2, 100, 10, 10, 1.0, 1.0,
                1.0, Path(directory), "https://provider.example/pricing", "2026-09-27",
                100, 500,
            )
        self.assertEqual(artifact["metrics"]["goodput_rps"], 1.0)
        self.assertEqual(artifact["capacity"]["replicas"], 1)
        self.assertEqual(artifact["pricing"]["as_of_date"], "2026-09-27")
        self.assertAlmostEqual(artifact["cost_per_million_total_tokens_usd"], 1.0 / (220 * 3600) * 1_000_000)


if __name__ == "__main__":
    unittest.main()
