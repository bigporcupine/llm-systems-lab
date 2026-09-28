import unittest

from unittest.mock import patch

from llm_systems_lab.online import _completion_url, stream_request


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __iter__(self):
        return iter([
            b'data: {"choices":[{"delta":{"content":"hello "}}]}\n',
            b'data: {"choices":[{"delta":{"content":"world"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n',
            b'data: [DONE]\n',
        ])


class OnlineClientTests(unittest.TestCase):
    def test_completion_url_accepts_base_or_complete_endpoint(self):
        self.assertEqual(
            _completion_url("http://localhost:8000/v1"),
            "http://localhost:8000/v1/chat/completions",
        )

    @patch("llm_systems_lab.online.request.urlopen", return_value=_Response())
    def test_stream_request_retains_completion_text(self, _urlopen):
        trace = stream_request("http://test/v1", "model", "r1", 3, 2, 1)
        self.assertTrue(trace.succeeded)
        self.assertEqual(trace.completion_text, "hello world")
        self.assertEqual(trace.output_tokens, 2)
        self.assertEqual(
            _completion_url("http://localhost:8000/v1/chat/completions"),
            "http://localhost:8000/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
