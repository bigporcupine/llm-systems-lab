"""Load, observability, capacity, cost, and canary analysis for Experiment 008."""

import json
from pathlib import Path
from typing import Any, Dict

from .capacity import capacity_plan, cost_per_million_tokens
from .environment import capture_environment
from .metrics import summarize
from .online import run_online_benchmark
from .prometheus import fetch_prometheus_text, parse_prometheus, metric_delta
from .rollout import evaluate_rollout


def run_gateway_load(
    base_url: str,
    metrics_url: str,
    model: str,
    requests: int,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    timeout_seconds: float,
    hourly_cost_usd: float,
    peak_rps: float,
    target_utilization: float,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_text = fetch_prometheus_text(metrics_url)
    run = run_online_benchmark(
        base_url, model, requests, concurrency, input_tokens,
        output_tokens, timeout_seconds,
    )
    after_text = fetch_prometheus_text(metrics_url)
    metrics = summarize(run.traces, run.wall_time_ms)
    goodput = float(metrics["request_throughput_rps"] or 0)
    output_throughput = float(metrics["output_throughput_tokens_s"] or 0)
    artifact = {
        "schema_version": "1.0", "experiment": "exp008-production-load",
        "environment": capture_environment(), "workload": {
            "requests": requests, "concurrency": concurrency,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        },
        "metrics": metrics,
        "gateway_metric_delta": metric_delta(parse_prometheus(before_text), parse_prometheus(after_text)),
        "gateway_metrics_before_text": before_text,
        "gateway_metrics_after_text": after_text,
        "capacity": capacity_plan(peak_rps, goodput, target_utilization) if goodput > 0 else None,
        "cost_per_million_output_tokens_usd": cost_per_million_tokens(hourly_cost_usd, output_throughput) if output_throughput > 0 else None,
        "traces": [trace.__dict__ for trace in run.traces],
    }
    (output_dir / "load.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def analyze_canary(stable_path: Path, canary_path: Path, policy: Dict[str, float], output: Path) -> Dict[str, Any]:
    stable = json.loads(stable_path.read_text(encoding="utf-8"))
    windows = json.loads(canary_path.read_text(encoding="utf-8"))
    result = evaluate_rollout(stable, windows, policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
