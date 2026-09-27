"""Experiment 001: validate inference measurement under controlled timing."""

import json
import statistics
from pathlib import Path
from typing import Dict, List

from .environment import capture_environment
from .metrics import summarize
from .mock_server import MockServerConfig, start_in_thread
from .models import BenchmarkMetadata, BenchmarkResult
from .online import run_online_benchmark
from .reporting import write_json


def _wrong_serial_throughput(result: BenchmarkResult) -> float:
    total_seconds = sum(trace.total_latency_ms for trace in result.traces) / 1000.0
    return round(len(result.traces) / total_seconds, 3)


def _table(results: List[BenchmarkResult]) -> str:
    rows = [
        "| Concurrency | TTFT P50 | TTFT P95 | E2E P95 | Throughput | Naive throughput | Goodput |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metric = result.metrics
        rows.append(
            "| {concurrency} | {ttft50} ms | {ttft95} ms | {e2e95} ms | "
            "{throughput} req/s | {wrong} req/s | {goodput} req/s |".format(
                concurrency=result.metadata.concurrency,
                ttft50=metric["ttft_p50_ms"],
                ttft95=metric["ttft_p95_ms"],
                e2e95=metric["e2e_p95_ms"],
                throughput=metric["request_throughput_rps"],
                wrong=_wrong_serial_throughput(result),
                goodput=metric["goodput_rps"],
            )
        )
    return "\n".join(rows)


def _report(
    results: List[BenchmarkResult],
    cold_ttft_ms: float,
    warm_ttft_p50_ms: float,
) -> str:
    high = results[-1]
    high_naive = _wrong_serial_throughput(high)
    high_actual = high.metrics["request_throughput_rps"] or 0.0
    multiplier = high_actual / high_naive if high_naive else 0.0
    ttfts = [trace.first_token_latency_ms for trace in high.traces if trace.first_token_latency_ms]
    mean_ttft = statistics.mean(ttfts)
    return f"""# Experiment 001 Results: Measuring LLM Inference Correctly

## Result status

Completed against a controlled OpenAI-compatible streaming server. These values validate the benchmark method; they are not model or hardware performance claims.

## Controlled configuration

- Prefill delay: 20 ms
- Decode delay: 5 ms per token after the first token
- Output length: 8 exact mock tokens
- Tail injection: 100 ms on every tenth server request
- Measured requests per concurrency: 30
- Warm-up requests before each measured run: 2
- TTFT SLO: 80 ms
- End-to-end latency SLO: 150 ms

## Concurrency results

{_table(results)}

“Naive throughput” is intentionally calculated as `requests / sum(request latency)`. It is incorrect for concurrent execution because overlapping requests are counted as if they ran serially.

At concurrency {high.metadata.concurrency}, wall-clock throughput was {high_actual} req/s, while the naive calculation produced {high_naive} req/s—a {multiplier:.2f}× difference. This confirms that concurrent throughput must use the elapsed time of the entire benchmark run.

## Tail behavior

At concurrency {high.metadata.concurrency}, mean TTFT was {mean_ttft:.3f} ms, P50 was {high.metrics['ttft_p50_ms']} ms, and P95 was {high.metrics['ttft_p95_ms']} ms. The injected slow requests have limited effect on the median but are visible in the tail percentile. Reporting only an average or median would hide this behavior.

## Warm-up effect

The isolated cold request had a TTFT of {cold_ttft_ms:.3f} ms. After warm-up, steady-state TTFT P50 was {warm_ttft_p50_ms:.3f} ms. Cold-start behavior should be reported separately rather than silently mixed into steady-state measurements.

## Throughput versus goodput

Throughput counts all successful requests. Goodput counts only requests satisfying both SLOs. Because the server injects tail delays, the two values diverge even with a 100% transport success rate. The configuration with the highest raw throughput is therefore not automatically the most useful configuration.

## Conclusions

1. TTFT, inter-token latency, and end-to-end latency measure different phases and must remain separate.
2. Concurrent throughput must be derived from benchmark wall-clock time.
3. Tail percentiles expose behavior hidden by averages and medians.
4. Warm-up policy materially changes reported latency and must be explicit.
5. SLO-aware goodput is a stronger capacity metric than unconstrained throughput.

## Limitations

- The server uses controlled sleeps and does not execute a model.
- One SSE content event equals exactly one token only because the mock protocol guarantees it. Real servers may emit zero, one, or multiple tokens per event.
- Localhost removes real network variability and does not model distributed serving.
- Thirty requests are enough to demonstrate the instrumentation, not to estimate a production P99 reliably.
- Python scheduling and timer resolution introduce small deviations from configured delays.

## Reproduce

```bash
llms-lab experiment-001
```
"""


def run_experiment(output_dir: Path) -> List[BenchmarkResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = capture_environment()
    results: List[BenchmarkResult] = []
    config = MockServerConfig(
        prefill_ms=20.0,
        token_delay_ms=5.0,
        output_tokens=8,
        tail_every=10,
        tail_delay_ms=100.0,
    )

    for concurrency in (1, 4, 8):
        server, thread = start_in_thread(config)
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        try:
            run_online_benchmark(
                base_url, "controlled-mock", 2, 1, 128, 8, controlled_mock=True
            )
            run = run_online_benchmark(
                base_url,
                "controlled-mock",
                30,
                concurrency,
                128,
                8,
                controlled_mock=True,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        metrics = summarize(
            run.traces,
            wall_time_ms=run.wall_time_ms,
            ttft_slo_ms=80.0,
            total_latency_slo_ms=150.0,
        )
        metadata = BenchmarkMetadata(
            experiment_id="exp001-measurement-basics",
            backend="controlled-openai-compatible-mock",
            model="controlled-mock",
            hardware="localhost",
            precision="not-applicable",
            concurrency=concurrency,
            source="controlled-measurement",
            notes="One SSE content event is exactly one token in this mock protocol.",
            environment=environment,
            workload={
                "requests": 30,
                "warmup_requests": 2,
                "input_tokens": 128,
                "output_tokens": 8,
                "ttft_slo_ms": 80.0,
                "total_latency_slo_ms": 150.0,
            },
        )
        result = BenchmarkResult.create(metadata, metrics, run.traces)
        results.append(result)
        write_json(result, output_dir / f"concurrency-{concurrency}.json")

    cold_config = MockServerConfig(
        prefill_ms=20.0, token_delay_ms=5.0, output_tokens=8, cold_start_ms=150.0
    )
    server, thread = start_in_thread(cold_config)
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        cold = run_online_benchmark(
            base_url, "controlled-mock", 1, 1, 128, 8, controlled_mock=True
        )
        warm = run_online_benchmark(
            base_url, "controlled-mock", 10, 1, 128, 8, controlled_mock=True
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    cold_ttft = cold.traces[0].first_token_latency_ms or 0.0
    warm_metrics = summarize(warm.traces, wall_time_ms=warm.wall_time_ms)
    report = _report(results, cold_ttft, warm_metrics["ttft_p50_ms"] or 0.0)
    (output_dir.parent / "RESULTS.md").write_text(report, encoding="utf-8")
    summary: Dict[str, object] = {
        "experiment": "exp001-measurement-basics",
        "results": [result.to_dict()["metrics"] for result in results],
        "cold_ttft_ms": cold_ttft,
        "warm_ttft_p50_ms": warm_metrics["ttft_p50_ms"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return results
