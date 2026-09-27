import unittest

from llm_systems_lab.quality import exact_match, score_predictions, token_f1


class QualityTests(unittest.TestCase):
    def test_normalized_exact_match(self):
        self.assertTrue(exact_match(" Paris. ", ["paris"]))

    def test_token_f1(self):
        self.assertAlmostEqual(token_f1("red blue", ["red green"]), 0.5)

    def test_dataset_scoring(self):
        data = [{"id": "x", "prompt": "p", "references": ["answer"]}]
        self.assertEqual(score_predictions(data, {"x": "Answer!"})["exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
