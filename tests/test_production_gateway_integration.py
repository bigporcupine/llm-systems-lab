import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None

from llm_systems_lab.mock_server import MockServerConfig, start_in_thread


@unittest.skipUnless(
    os.environ.get("LLMS_LAB_SOCKET_TESTS") == "1" and TestClient is not None,
    "set LLMS_LAB_SOCKET_TESTS=1 and install the production extra",
)
class ProductionGatewayIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.upstream, self.thread = start_in_thread(
            MockServerConfig(prefill_ms=200, token_delay_ms=1, output_tokens=2)
        )
        upstream_url = f"http://127.0.0.1:{self.upstream.server_port}/v1"
        from llm_systems_lab.production_gateway import create_app

        self.client = TestClient(create_app({
            "stable_base_url": upstream_url,
            "canary_base_url": upstream_url,
            "canary_fraction": 0.0,
            "max_in_flight": 1,
            "queue_timeout_ms": 20,
            "request_timeout_seconds": 5,
        }))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.upstream.shutdown()
        self.upstream.server_close()
        self.thread.join(timeout=2)

    @staticmethod
    def _payload():
        return {
            "model": "mock", "messages": [{"role": "user", "content": "test"}],
            "stream": True, "max_tokens": 2, "temperature": 0,
        }

    def test_streaming_proxy_metrics_and_overload_rejection(self):
        started = threading.Event()

        def first_request():
            started.set()
            return self.client.post("/v1/chat/completions", json=self._payload(), headers={"x-request-id": "stable-1"})

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(first_request)
            started.wait(timeout=1)
            time.sleep(0.03)
            second = executor.submit(
                self.client.post, "/v1/chat/completions", json=self._payload(),
                headers={"x-request-id": "stable-2"},
            )
            responses = [first.result(timeout=5), second.result(timeout=5)]

        self.assertEqual(sorted(response.status_code for response in responses), [200, 429])
        overloaded = next(response for response in responses if response.status_code == 429)
        self.assertEqual(overloaded.json()["error"]["type"], "overloaded")
        success = next(response for response in responses if response.status_code == 200)
        self.assertIn("data:", success.text)
        metrics = self.client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("llm_gateway_rejections_total 1", metrics.text)
        self.assertIn('llm_gateway_requests_total{lane="stable",status="200"} 1', metrics.text)


if __name__ == "__main__":
    unittest.main()
