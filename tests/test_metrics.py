import unittest

from llm_systems_lab.metrics import (
    inter_token_latencies,
    percentile,
    summarize,
    time_per_output_token,
)
from llm_systems_lab.models import RequestTrace


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.traces = [
            RequestTrace("a", 10, 3, 100.0, 20.0, [20.0, 50.0, 80.0]),
            RequestTrace("b", 10, 2, 200.0, 40.0, [40.0, 100.0]),
            RequestTrace("c", 10, 0, 50.0, None, [], False, "timeout"),
        ]

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([10.0, 20.0], 0.5), 15.0)
        self.assertIsNone(percentile([], 0.5))

    def test_inter_token_latency_excludes_time_to_first_token(self):
        self.assertEqual(inter_token_latencies(self.traces[0]), [30.0, 30.0])
        self.assertEqual(time_per_output_token(self.traces[0]), 30.0)

    def test_summary_uses_wall_clock_for_throughput(self):
        metrics = summarize(
            self.traces,
            wall_time_ms=1000.0,
            ttft_slo_ms=30.0,
            total_latency_slo_ms=150.0,
        )
        self.assertEqual(metrics["requests_succeeded"], 2.0)
        self.assertEqual(metrics["success_rate"], 0.667)
        self.assertEqual(metrics["request_throughput_rps"], 2.0)
        self.assertEqual(metrics["output_throughput_tokens_s"], 5.0)
        self.assertEqual(metrics["goodput_rps"], 1.0)
        self.assertEqual(metrics["ttft_p50_ms"], 30.0)


if __name__ == "__main__":
    unittest.main()
