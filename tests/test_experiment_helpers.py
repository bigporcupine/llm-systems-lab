import unittest

from llm_systems_lab.gpu_memory import theoretical_kv_bytes
from llm_systems_lab.prometheus import metric_delta, parse_prometheus
from llm_systems_lab.rollout import evaluate_rollout
from llm_systems_lab.workloads import shared_prefix_prompts, speculative_prompts


class ExperimentHelperTests(unittest.TestCase):
    def test_shared_prefix_prompts_have_unique_suffixes(self):
        prompts = shared_prefix_prompts(20, 3, seed=7)
        prefixes = [prompt.split("\nUnique request", 1)[0] for prompt in prompts]
        self.assertEqual(len(set(prefixes)), 1)
        self.assertEqual(len(set(prompts)), 3)

    def test_theoretical_kv_cache_accounts_for_keys_and_values(self):
        self.assertEqual(theoretical_kv_bytes(2, 4, 8, 2, 16, 3), 2 * 2 * 4 * 8 * 2 * 16 * 3)

    def test_speculative_workloads_are_distinct_and_deterministic(self):
        general = speculative_prompts("general", 8, 2, seed=7)
        self.assertEqual(general, speculative_prompts("general", 8, 2, seed=7))
        self.assertNotEqual(general, speculative_prompts("high-agreement", 8, 2, seed=7))
        with self.assertRaises(ValueError):
            speculative_prompts("unknown", 8, 1)

    def test_prometheus_parser_sums_labeled_series_and_deltas(self):
        before = parse_prometheus('tokens_total{kind="accepted"} 2\ntokens_total{kind="draft"} 5\n')
        after = parse_prometheus('tokens_total{kind="accepted"} 7\ntokens_total{kind="draft"} 9\n')
        self.assertEqual(before["tokens_total"], 7)
        self.assertEqual(metric_delta(before, after)["tokens_total"], 9)

    def test_rollout_promotes_and_rolls_back(self):
        stable = {"requests": 1000, "error_rate": 0.01, "p95_latency_ms": 100, "quality": 0.9}
        policy = {"minimum_requests": 100, "max_error_rate": 0.02, "max_p95_latency_regression": 0.2, "max_quality_regression": 0.03}
        good = {"requests": 100, "error_rate": 0.01, "p95_latency_ms": 105, "quality": 0.9}
        self.assertEqual(evaluate_rollout(stable, [good] * 5, policy)["status"], "promoted")
        bad = {"requests": 100, "error_rate": 0.1, "p95_latency_ms": 105, "quality": 0.9}
        self.assertEqual(evaluate_rollout(stable, [good, bad], policy)["status"], "rolled-back")


if __name__ == "__main__":
    unittest.main()
