import unittest

from llm_systems_lab.canary import assign_lane, should_rollback
from llm_systems_lab.capacity import capacity_plan, cost_per_million_tokens


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


if __name__ == "__main__":
    unittest.main()
