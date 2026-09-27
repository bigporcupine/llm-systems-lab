import unittest

from llm_systems_lab.online import _completion_url


class OnlineClientTests(unittest.TestCase):
    def test_completion_url_accepts_base_or_complete_endpoint(self):
        self.assertEqual(
            _completion_url("http://localhost:8000/v1"),
            "http://localhost:8000/v1/chat/completions",
        )
        self.assertEqual(
            _completion_url("http://localhost:8000/v1/chat/completions"),
            "http://localhost:8000/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
