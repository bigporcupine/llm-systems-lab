"""Compare target-only and speculative endpoints with engine metric deltas."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .environment import capture_environment
from .experiment002 import _aggregate
from .metrics import summarize
from .models import BenchmarkMetadata, BenchmarkResult
from .online import run_prompt_benchmark
from .prometheus import fetch_prometheus, metric_delta
from .reporting import write_json
from .workloads import deterministic_prompt, speculative_prompts


def run_speculative_comparison(
    baseline_url: str,
    speculative_url: str,
    model: str,
    workloads: List[str],
    input_size_hints: List[int],
    output_tokens: int,
    concurrency: int,
    measured_requests: int,
    repetitions: int,
    timeout_seconds: float,
    output_dir: Path,
    baseline_metrics_url: Optional[str] = None,
    speculative_metrics_url: Optional[str] = None,
    seed: int = 7,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    raw_metric_deltas = {}
    for variant, url, metrics_url in (
        ("target-only", baseline_url, baseline_metrics_url),
        ("speculative", speculative_url, speculative_metrics_url),
    ):
        run_prompt_benchmark(
            url, model, [deterministic_prompt(64, seed + 999)], 1,
            min(output_tokens, 8), timeout_seconds,
            request_id_prefix=f"{variant}-warmup", force_output_length=True,
        )
        before = fetch_prometheus(metrics_url) if metrics_url else {}
        for workload in workloads:
            for input_hint in input_size_hints:
                results = []
                for repetition in range(repetitions):
                    prompts = speculative_prompts(
                        workload, input_hint, measured_requests, seed + repetition
                    )
                    run = run_prompt_benchmark(
                        url, model, prompts, concurrency, output_tokens,
                        timeout_seconds, request_id_prefix=f"{variant}-{workload}-run{repetition + 1}",
                        force_output_length=True,
                    )
                    metrics = summarize(run.traces, run.wall_time_ms)
                    result = BenchmarkResult.create(
                        BenchmarkMetadata(
                            experiment_id="exp006-speculative-decoding", backend=variant,
                            model=model, hardware="captured-in-environment", precision="configured-by-server",
                            concurrency=concurrency, source="measured", environment=capture_environment(),
                            workload={"workload": workload, "input_size_hint": input_hint,
                                      "output_tokens": output_tokens,
                                      "measured_requests": measured_requests,
                                      "repetition": repetition},
                        ), metrics, run.traces,
                    )
                    results.append(result)
                    write_json(
                        result,
                        output_dir / f"{variant}_{workload}_input-{input_hint}_run-{repetition + 1}.json",
                    )
                cells.append({"variant": variant, "workload": workload,
                              "input_size_hint": input_hint,
                              "aggregate": _aggregate(results)})
        after = fetch_prometheus(metrics_url) if metrics_url else {}
        raw_metric_deltas[variant] = metric_delta(before, after) if metrics_url else {}
    artifact = {
        "schema_version": "1.0", "experiment": "exp006-speculative-decoding",
        "model": model, "environment": capture_environment(), "cells": cells,
        "engine_metric_deltas": raw_metric_deltas,
    }
    (output_dir / "summary.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact
