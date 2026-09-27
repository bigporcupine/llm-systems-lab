"""Reusable measured-run engine for endpoint-backed experiment matrices."""

import json
from pathlib import Path
from typing import Any, Dict, List

from .environment import capture_environment
from .experiment002 import _aggregate
from .metrics import summarize
from .models import BenchmarkMetadata, BenchmarkResult
from .online import run_online_benchmark
from .reporting import write_json
from .workloads import deterministic_prompt


def run_endpoint_matrix(
    experiment_id: str,
    variant: str,
    base_url: str,
    model: str,
    model_revision: str,
    precision: str,
    backend_version: str,
    cells: List[Dict[str, Any]],
    output_dir: Path,
    *,
    warmup_requests: int,
    measured_requests: int,
    repetitions: int,
    output_tokens: int,
    timeout_seconds: float,
    ttft_slo_ms: float,
    e2e_slo_ms: float,
    seed: int,
) -> Dict[str, Any]:
    """Measure arbitrary input/concurrency cells using one protocol and schema."""
    if repetitions <= 0 or measured_requests <= 0:
        raise ValueError("repetitions and measured_requests must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = capture_environment()
    completed = []
    for cell in cells:
        input_hint = int(cell["input_size_hint"])
        concurrency = int(cell["concurrency"])
        prompt = cell.get("prompt") or deterministic_prompt(input_hint, seed)
        run_online_benchmark(
            base_url, model, warmup_requests, min(concurrency, warmup_requests),
            input_hint, output_tokens, timeout_seconds=timeout_seconds,
            prompt=prompt, force_output_length=True,
        )
        results = []
        for repetition in range(repetitions):
            measured = run_online_benchmark(
                base_url, model, measured_requests, concurrency, input_hint,
                output_tokens, timeout_seconds=timeout_seconds, prompt=prompt,
                force_output_length=True,
            )
            metrics = summarize(
                measured.traces, measured.wall_time_ms, ttft_slo_ms, e2e_slo_ms
            )
            result = BenchmarkResult.create(
                BenchmarkMetadata(
                    experiment_id=experiment_id,
                    backend=variant,
                    model=model,
                    hardware=(environment.get("nvidia_gpu") or {}).get("name", "no-nvidia-gpu-detected"),
                    precision=precision,
                    concurrency=concurrency,
                    source="measured",
                    notes="Endpoint-backed measured run; startup is excluded.",
                    environment=environment,
                    workload={**cell, "model_revision": model_revision,
                              "backend_version": backend_version,
                              "repetition": repetition,
                              "output_tokens_requested": output_tokens},
                ), metrics, measured.traces,
            )
            results.append(result)
            write_json(result, output_dir / f"input-{input_hint}_c-{concurrency}_run-{repetition + 1}.json")
        completed.append({**cell, "aggregate": _aggregate(results)})
    summary = {
        "schema_version": "1.0", "experiment": experiment_id,
        "variant": variant, "model": model, "model_revision": model_revision,
        "precision": precision, "backend_version": backend_version,
        "environment": environment, "cells": completed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
