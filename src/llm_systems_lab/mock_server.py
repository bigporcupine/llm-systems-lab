"""A controlled OpenAI-compatible streaming server for measurement tests."""

import argparse
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple


@dataclass(frozen=True)
class MockServerConfig:
    prefill_ms: float = 20.0
    token_delay_ms: float = 5.0
    output_tokens: int = 8
    cold_start_ms: float = 0.0
    tail_every: int = 0
    tail_delay_ms: float = 0.0


class _RequestCounter:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            value = self._value
            self._value += 1
            return value


def create_server(
    host: str = "127.0.0.1",
    port: int = 0,
    config: Optional[MockServerConfig] = None,
) -> ThreadingHTTPServer:
    """Create, but do not start, a deterministic threaded HTTP server."""
    settings = config or MockServerConfig()
    counter = _RequestCounter()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path.rstrip("/") not in {
                "/v1/chat/completions",
                "/chat/completions",
            }:
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_error(400, "invalid JSON")
                return

            request_index = counter.next()
            output_tokens = int(payload.get("max_tokens", settings.output_tokens))
            input_tokens = int(payload.get("mock_input_tokens", 1))
            request_id = str(payload.get("mock_request_id", request_index))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            initial_delay = settings.prefill_ms
            if request_index == 0:
                initial_delay += settings.cold_start_ms
            if settings.tail_every > 0 and request_index % settings.tail_every == 0:
                initial_delay += settings.tail_delay_ms
            time.sleep(initial_delay / 1000.0)

            for token_index in range(output_tokens):
                if token_index > 0:
                    time.sleep(settings.token_delay_ms / 1000.0)
                event = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": "x"}}],
                    "mock_token_index": token_index,
                }
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                self.wfile.flush()

            usage = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return ThreadingHTTPServer((host, port), Handler)


def start_in_thread(
    config: MockServerConfig,
) -> Tuple[ThreadingHTTPServer, threading.Thread]:
    server = create_server(config=config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def serve_forever(config: MockServerConfig, host: str, port: int) -> None:
    server = create_server(host, port, config)
    print(f"Mock server listening on http://{host}:{server.server_port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefill-ms", type=float, default=20.0)
    parser.add_argument("--token-delay-ms", type=float, default=5.0)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--cold-start-ms", type=float, default=0.0)
    parser.add_argument("--tail-every", type=int, default=0)
    parser.add_argument("--tail-delay-ms", type=float, default=0.0)
    args = parser.parse_args()
    serve_forever(
        MockServerConfig(
            prefill_ms=args.prefill_ms,
            token_delay_ms=args.token_delay_ms,
            output_tokens=args.output_tokens,
            cold_start_ms=args.cold_start_ms,
            tail_every=args.tail_every,
            tail_delay_ms=args.tail_delay_ms,
        ),
        args.host,
        args.port,
    )


if __name__ == "__main__":
    main()

