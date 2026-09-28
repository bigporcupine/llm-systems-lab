"""Compare target-only and speculative endpoints with engine metric deltas."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .environment import capture_environment
from .config_resolution import require_immutable_revision
from .experiment002 import _aggregate
from .metrics import summarize
from .models import BenchmarkMetadata, BenchmarkResult
from .online import run_prompt_benchmark
from .prometheus import fetch_prometheus_text, metric_delta, parse_prometheus
from .reporting import write_json
from .workloads import deterministic_prompt, speculative_prompts


def run_speculative_comparison(
    baseline_url: str,
    speculative_url: str,
    model: str,
    target_revision: str,
    baseline_backend_version: str,
    speculative_backend_version: str,
    speculative_config: str,
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
    draft_model: str = "",
    draft_revision: str = "",
) -> Dict[str, Any]:
    if not draft_model.strip() or not draft_revision.strip():
        raise ValueError("draft_model and immutable draft_revision are required")
    require_immutable_revision(target_revision, "target_revision")
    require_immutable_revision(draft_revision, "draft_revision")
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    raw_metric_deltas = {}
    raw_metric_snapshots = {}
    for variant, url, metrics_url in (
        ("target-only", baseline_url, baseline_metrics_url),
        ("speculative", speculative_url, speculative_metrics_url),
    ):
        run_prompt_benchmark(
            url, model, [deterministic_prompt(64, seed + 999)], 1,
            min(output_tokens, 8), timeout_seconds,
            request_id_prefix=f"{variant}-warmup", force_output_length=True,
        )
        before_text = fetch_prometheus_text(metrics_url) if metrics_url else None
        before = parse_prometheus(before_text) if before_text is not None else {}
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
                                      "model_revision": target_revision,
                                      "backend_version": baseline_backend_version if variant == "target-only" else speculative_backend_version,
                                      "speculative_config": speculative_config if variant == "speculative" else None,
                                      "draft_model": draft_model if variant == "speculative" else None,
                                      "draft_revision": draft_revision if variant == "speculative" else None,
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
        after_text = fetch_prometheus_text(metrics_url) if metrics_url else None
        after = parse_prometheus(after_text) if after_text is not None else {}
        raw_metric_deltas[variant] = metric_delta(before, after) if metrics_url else {}
        raw_metric_snapshots[variant] = {
            "url": metrics_url, "before_text": before_text, "after_text": after_text,
        }
    artifact = {
        "schema_version": "1.0", "experiment": "exp006-speculative-decoding",
        "source": "derived-from-measured", "model": model,
        "target_revision": target_revision,
        "draft_model": draft_model, "draft_revision": draft_revision,
        "baseline_backend_version": baseline_backend_version,
        "speculative_backend_version": speculative_backend_version,
        "speculative_config": speculative_config,
        "environment": capture_environment(), "cells": cells,
        "engine_metric_deltas": raw_metric_deltas,
        "engine_metric_snapshots": raw_metric_snapshots,
    }
    (output_dir / "summary.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact
