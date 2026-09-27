"""Concurrent client for OpenAI-compatible streaming chat completions."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List
from urllib import error, request

from .models import RequestTrace


@dataclass(frozen=True)
class OnlineRun:
    traces: List[RequestTrace]
    wall_time_ms: float


def _completion_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def stream_request(
    base_url: str,
    model: str,
    request_id: str,
    input_tokens: int,
    output_tokens: int,
    timeout_seconds: float,
    controlled_mock: bool = False,
    prompt: str = "",
    force_output_length: bool = False,
) -> RequestTrace:
    """Measure one streamed request using a monotonic high-resolution clock."""
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt or ("x " * input_tokens)}
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": output_tokens,
        "temperature": 0.0,
    }
    if force_output_length:
        # vLLM supports min_tokens as an OpenAI-compatible extension. The
        # reference Transformers server always emits exactly max_tokens.
        payload["min_tokens"] = output_tokens
    if controlled_mock:
        payload.update(
            {"mock_request_id": request_id, "mock_input_tokens": input_tokens}
        )
    http_request = request.Request(
        _completion_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    timestamps: List[float] = []
    reported_input_tokens = 0
    reported_output_tokens = 0
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                usage = event.get("usage") or {}
                if usage.get("prompt_tokens") is not None:
                    reported_input_tokens = int(usage["prompt_tokens"])
                if usage.get("completion_tokens") is not None:
                    reported_output_tokens = int(usage["completion_tokens"])
                choices = event.get("choices") or []
                content = (choices[0].get("delta") or {}).get("content") if choices else None
                if content:
                    timestamps.append((time.perf_counter() - started) * 1000.0)
        total_ms = (time.perf_counter() - started) * 1000.0
        return RequestTrace(
            request_id=request_id,
            input_tokens=reported_input_tokens,
            output_tokens=reported_output_tokens or len(timestamps),
            total_latency_ms=round(total_ms, 3),
            first_token_latency_ms=round(timestamps[0], 3) if timestamps else None,
            token_timestamps_ms=[round(value, 3) for value in timestamps],
            succeeded=bool(timestamps),
            error=None if timestamps else "stream completed without content",
        )
    except (error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        total_ms = (time.perf_counter() - started) * 1000.0
        return RequestTrace(
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=0,
            total_latency_ms=round(total_ms, 3),
            first_token_latency_ms=None,
            succeeded=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_online_benchmark(
    base_url: str,
    model: str,
    requests: int,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    timeout_seconds: float = 30.0,
    controlled_mock: bool = False,
    prompt: str = "",
    force_output_length: bool = False,
) -> OnlineRun:
    if requests <= 0 or concurrency <= 0:
        raise ValueError("requests and concurrency must be positive")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                stream_request,
                base_url,
                model,
                f"request-{index:05d}",
                input_tokens,
                output_tokens,
                timeout_seconds,
                controlled_mock,
                prompt,
                force_output_length,
            )
            for index in range(requests)
        ]
        traces = [future.result() for future in as_completed(futures)]
    wall_time_ms = (time.perf_counter() - started) * 1000.0
    traces.sort(key=lambda trace: trace.request_id)
    return OnlineRun(traces=traces, wall_time_ms=round(wall_time_ms, 3))
