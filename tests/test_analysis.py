import unittest

from llm_systems_lab.analysis import break_even, quality_gate, saturation_knee


class AnalysisTests(unittest.TestCase):
    def test_saturation_knee_respects_latency_budget(self):
        cells = [
            {"concurrency": 1, "aggregate": {"e2e_p95_ms": {"mean": 100}, "output_throughput_tokens_s": {"mean": 10}}},
            {"concurrency": 4, "aggregate": {"e2e_p95_ms": {"mean": 180}, "output_throughput_tokens_s": {"mean": 35}}},
            {"concurrency": 8, "aggregate": {"e2e_p95_ms": {"mean": 400}, "output_throughput_tokens_s": {"mean": 50}}},
        ]
        self.assertEqual(saturation_knee(cells, 200)["concurrency"], 4)

    def test_break_even_and_quality_gate(self):
        rows = [{"n": 1, "off": 10, "on": 12}, {"n": 2, "off": 20, "on": 18}]
        self.assertEqual(break_even(rows, "off", "on")["n"], 2)
        self.assertTrue(quality_gate(0.9, 0.88, 0.03)["passed"])


if __name__ == "__main__":
    unittest.main()
