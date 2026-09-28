"""Measure GPU-memory growth across prompt length and concurrency."""

import json
from pathlib import Path
from typing import Any, Dict, List

from .environment import capture_environment
from .config_resolution import require_immutable_revision
from .gpu_memory import MemorySampler, nvidia_used_memory_mb, theoretical_kv_bytes
from .online import run_prompt_benchmark
from .workloads import deterministic_prompt


def run_kv_memory_experiment(
    base_url: str,
    model: str,
    model_revision: str,
    backend_version: str,
    sequence_lengths: List[int],
    concurrencies: List[int],
    output_tokens: int,
    timeout_seconds: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
    output_dir: Path,
    seed: int = 7,
) -> Dict[str, Any]:
    require_immutable_revision(model_revision)
    output_dir.mkdir(parents=True, exist_ok=True)
    idle_mb = nvidia_used_memory_mb()
    cells = []
    for length in sequence_lengths:
        for concurrency in concurrencies:
            prompts = [f"{deterministic_prompt(length, seed)}\nSequence {index}." for index in range(concurrency)]
            cell_start_mb = nvidia_used_memory_mb()
            sampler = MemorySampler()
            sampler.start()
            try:
                run = run_prompt_benchmark(
                    base_url, model, prompts, concurrency, output_tokens,
                    timeout_seconds, request_id_prefix=f"kv-l{length}-c{concurrency}",
                    force_output_length=True,
                )
            finally:
                sampler.stop()
            peak_mb = max(sampler.samples_mb) if sampler.samples_mb else None
            actual_cached_tokens = max(
                (trace.input_tokens + trace.output_tokens for trace in run.traces),
                default=length + output_tokens,
            )
            if actual_cached_tokens <= 0:
                actual_cached_tokens = length + output_tokens
            theoretical = theoretical_kv_bytes(
                layers, kv_heads, head_dim, bytes_per_element,
                actual_cached_tokens, concurrency,
            )
            cells.append({
                "sequence_length_hint": length, "concurrency": concurrency,
                "idle_memory_mb": idle_mb, "cell_start_memory_mb": cell_start_mb,
                "actual_cached_tokens_per_sequence": actual_cached_tokens,
                "peak_memory_mb": peak_mb,
                "observed_growth_mb": None if peak_mb is None else peak_mb - idle_mb,
                "cell_incremental_growth_mb": None if peak_mb is None else peak_mb - cell_start_mb,
                "theoretical_kv_bytes": theoretical,
                "memory_samples_mb": sampler.samples_mb,
                "wall_time_ms": run.wall_time_ms,
                "traces": [trace.__dict__ for trace in run.traces],
            })
    artifact = {
        "schema_version": "1.0", "experiment": "exp005-kv-memory",
        "source": "measured", "model": model,
        "model_revision": model_revision, "backend_version": backend_version,
        "environment": capture_environment(),
        "model_dimensions": {"layers": layers, "kv_heads": kv_heads,
                             "head_dim": head_dim, "bytes_per_element": bytes_per_element},
        "cells": cells,
    }
    (output_dir / "kv-memory.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact
