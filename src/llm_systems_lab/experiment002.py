"""Run Experiment 002 against one already-running serving backend."""

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from .environment import capture_environment
from .config_resolution import require_immutable_revision
from .gpu_telemetry import GpuTelemetrySampler
from .metrics import summarize
from .models import BenchmarkMetadata, BenchmarkResult
from .online import run_online_benchmark
from .reporting import write_json
from .statistics import bootstrap_mean_ci
from .workloads import deterministic_prompt


def load_config(path: Path) -> Dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model",
        "model_revision",
        "dtype",
        "input_size_hints",
        "output_tokens",
        "concurrency",
        "warmup_requests",
        "measured_requests",
        "repetitions",
        "ttft_slo_ms",
        "e2e_slo_ms",
        "seed",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing configuration fields: {', '.join(missing)}")
    return config


def _aggregate(runs: List[BenchmarkResult]) -> Dict[str, Any]:
    metric_names = (
        "ttft_p50_ms",
        "ttft_p95_ms",
        "e2e_p95_ms",
        "itl_p50_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "request_throughput_rps",
        "output_throughput_tokens_s",
        "goodput_rps",
    )
    aggregate: Dict[str, Any] = {}
    for name in metric_names:
        values = [float(run.metrics[name]) for run in runs if run.metrics[name] is not None]
        mean, lower, upper = bootstrap_mean_ci(values, samples=2000, seed=7)
        aggregate[name] = {
            "mean": round(mean, 3),
            "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "ci95_lower": round(lower, 3),
            "ci95_upper": round(upper, 3),
            "runs": len(values),
        }
    return aggregate


def run_backend_matrix(
    backend: str,
    base_url: str,
    config_path: Path,
    output_dir: Path,
    backend_version: str,
) -> Dict[str, Any]:
    if not backend_version.strip() or backend_version.strip().lower() == "unknown":
        raise ValueError("backend_version must identify the measured serving engine")
    config = load_config(config_path)
    require_immutable_revision(config["model_revision"])
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = capture_environment()
    cells: List[Dict[str, Any]] = []
    telemetry = GpuTelemetrySampler(
        output_dir / "gpu-telemetry.json",
        interval_seconds=float(config.get("gpu_telemetry_interval_seconds", 1.0)),
    ).start()

    for input_hint in config["input_size_hints"]:
        prompt = deterministic_prompt(input_hint, config["seed"])
        for concurrency in config["concurrency"]:
            run_online_benchmark(
                base_url,
                config["model"],
                config["warmup_requests"],
                min(concurrency, config["warmup_requests"]),
                input_hint,
                config["output_tokens"],
                prompt=prompt,
                force_output_length=True,
            )
            repetitions: List[BenchmarkResult] = []
            for repetition in range(config["repetitions"]):
                measured = run_online_benchmark(
                    base_url,
                    config["model"],
                    config["measured_requests"],
                    concurrency,
                    input_hint,
                    config["output_tokens"],
                    timeout_seconds=config.get("timeout_seconds", 120.0),
                    prompt=prompt,
                    force_output_length=True,
                )
                metrics = summarize(
                    measured.traces,
                    wall_time_ms=measured.wall_time_ms,
                    ttft_slo_ms=config["ttft_slo_ms"],
                    total_latency_slo_ms=config["e2e_slo_ms"],
                )
                result = BenchmarkResult.create(
                    BenchmarkMetadata(
                        experiment_id="exp002-serving-engines",
                        backend=backend,
                        model=config["model"],
                        hardware=(environment.get("nvidia_gpu") or {}).get(
                            "name", "no-nvidia-gpu-detected"
                        ),
                        precision=config["dtype"],
                        concurrency=concurrency,
                        source="measured",
                        notes="Actual token counts come from server-reported usage.",
                        environment=environment,
                        workload={
                            "input_size_hint": input_hint,
                            "output_tokens_requested": config["output_tokens"],
                            "warmup_requests": config["warmup_requests"],
                            "measured_requests": config["measured_requests"],
                            "repetition": repetition,
                            "model_revision": config["model_revision"],
                            "backend_version": backend_version,
                        },
                    ),
                    metrics,
                    measured.traces,
                )
                repetitions.append(result)
                filename = (
                    f"input-{input_hint}_concurrency-{concurrency}_run-{repetition + 1}.json"
                )
                write_json(result, output_dir / filename)
            cells.append(
                {
                    "input_size_hint": input_hint,
                    "concurrency": concurrency,
                    "aggregate": _aggregate(repetitions),
                }
            )

    telemetry_artifact = telemetry.stop()
    summary = {
        "schema_version": "1.0",
        "experiment": "exp002-serving-engines",
        "backend": backend,
        "backend_version": backend_version,
        "config": config,
        "environment": environment,
        "gpu_telemetry": {
            "path": "gpu-telemetry.json",
            "available": telemetry_artifact["available"],
            "sample_count": telemetry_artifact["sample_count"],
        },
        "cells": cells,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
