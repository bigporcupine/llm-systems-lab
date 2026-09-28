"""Measured prefix-cache break-even experiment against two live policies."""

import json
from pathlib import Path
from typing import Any, Dict, List

from .analysis import break_even
from .config_resolution import require_immutable_revision
from .environment import capture_environment
from .metrics import summarize
from .online import run_prompt_benchmark
from .prometheus import fetch_prometheus_text, metric_delta, parse_prometheus
from .workloads import deterministic_prompt, shared_prefix_prompts


def run_prefix_cache_experiment(
    disabled_url: str,
    enabled_url: str,
    model: str,
    model_revision: str,
    backend_version: str,
    prefix_lengths: List[int],
    reuse_counts: List[int],
    output_tokens: int,
    concurrency: int,
    repetitions: int,
    timeout_seconds: float,
    output_dir: Path,
    seed: int = 7,
    disabled_metrics_url: str = None,
    enabled_metrics_url: str = None,
) -> Dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    require_immutable_revision(model_revision)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    raw = []
    metrics_urls = {"disabled": disabled_metrics_url, "enabled": enabled_metrics_url}
    # Warm kernels and HTTP paths with an unrelated prompt. The measured
    # shared prefixes remain cold at the start of every cell.
    for policy, url in (("disabled", disabled_url), ("enabled", enabled_url)):
        run_prompt_benchmark(
            url, model, [deterministic_prompt(64, seed + 999)], 1,
            min(output_tokens, 8), timeout_seconds,
            request_id_prefix=f"{policy}-warmup", force_output_length=True,
        )
    for prefix_length in prefix_lengths:
        for reuse_count in reuse_counts:
            latency = {"disabled": [], "enabled": []}
            for policy, url in (("disabled", disabled_url), ("enabled", enabled_url)):
                for repetition in range(repetitions):
                    metrics_url = metrics_urls[policy]
                    before_text = fetch_prometheus_text(metrics_url, timeout_seconds) if metrics_url else None
                    cell_seed = seed + prefix_length * 10_000 + reuse_count * 100 + repetition
                    prompts = shared_prefix_prompts(prefix_length, reuse_count, cell_seed)
                    run = run_prompt_benchmark(
                        url, model, prompts, min(concurrency, reuse_count),
                        output_tokens, timeout_seconds,
                        request_id_prefix=f"{policy}-p{prefix_length}-r{reuse_count}-run{repetition + 1}",
                        force_output_length=True,
                    )
                    metrics = summarize(run.traces, run.wall_time_ms)
                    after_text = fetch_prometheus_text(metrics_url, timeout_seconds) if metrics_url else None
                    engine_delta = (
                        metric_delta(parse_prometheus(before_text), parse_prometheus(after_text))
                        if before_text is not None and after_text is not None else None
                    )
                    latency[policy].append(float(metrics["e2e_p95_ms"]))
                    raw.append({
                        "prefix_length": prefix_length, "reuse_count": reuse_count,
                        "policy": policy, "repetition": repetition,
                        "wall_time_ms": run.wall_time_ms, "metrics": metrics,
                        "engine_metrics_url": metrics_url,
                        "engine_metrics_before_text": before_text,
                        "engine_metrics_after_text": after_text,
                        "engine_metric_delta": engine_delta,
                        "traces": [trace.__dict__ for trace in run.traces],
                    })
            rows.append({
                "prefix_length": prefix_length, "reuse_count": reuse_count,
                "disabled_p95_ms": sum(latency["disabled"]) / repetitions,
                "enabled_p95_ms": sum(latency["enabled"]) / repetitions,
            })
    break_evens = {}
    for prefix_length in prefix_lengths:
        group = [row for row in rows if row["prefix_length"] == prefix_length]
        break_evens[str(prefix_length)] = break_even(group, "disabled_p95_ms", "enabled_p95_ms")
    artifact = {
        "schema_version": "1.0", "experiment": "exp005-prefix-cache",
        "source": "measured", "model": model,
        "model_revision": model_revision, "backend_version": backend_version,
        "environment": capture_environment(),
        "metrics_endpoints": metrics_urls,
        "rows": rows, "break_even_by_prefix_length": break_evens,
        "raw_runs": raw,
    }
    (output_dir / "prefix-cache.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact
