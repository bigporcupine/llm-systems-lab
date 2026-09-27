"""JSON persistence and a small human-readable Markdown report."""

import json
from pathlib import Path
from typing import Any, Dict

from .models import BenchmarkResult


def write_json(result: BenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")


def markdown_report(result: BenchmarkResult) -> str:
    metadata = result.metadata
    metrics = result.metrics

    def display(key: str, suffix: str = "") -> str:
        value = metrics.get(key)
        return "n/a" if value is None else f"{value}{suffix}"

    return f"""# Benchmark Report: {metadata.experiment_id}

> Source: **{metadata.source}**. Synthetic results validate the tooling only and must not be cited as hardware performance.

## Experimental setup

| Field | Value |
|---|---|
| Backend | {metadata.backend} |
| Model | {metadata.model} |
| Hardware | {metadata.hardware} |
| Precision | {metadata.precision} |
| Concurrency | {metadata.concurrency} |

## Results

| Metric | Value |
|---|---:|
| TTFT P50 | {display('ttft_p50_ms', ' ms')} |
| TTFT P95 | {display('ttft_p95_ms', ' ms')} |
| End-to-end latency P95 | {display('e2e_p95_ms', ' ms')} |
| Inter-token latency P50 | {display('itl_p50_ms', ' ms')} |
| Request throughput | {display('request_throughput_rps', ' req/s')} |
| Output throughput | {display('output_throughput_tokens_s', ' tok/s')} |
| SLO goodput | {display('goodput_rps', ' req/s')} |
| Success rate | {display('success_rate')} |

## Interpretation

Add the hypothesis, observed bottleneck, confounders, quality impact, and production recommendation here. Do not generalize beyond the recorded workload and environment.

## Limitations

- Token timestamps must come from a streaming API to represent inter-token latency.
- Throughput requires wall-clock benchmark duration, not the sum of request latencies.
- A performance result is incomplete until model quality is evaluated under the same configuration.
"""


def write_markdown(result: BenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(result), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

