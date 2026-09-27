"""Metric definitions for latency, throughput, and SLO-aware goodput."""

import math
from typing import Dict, Iterable, List, Optional, Sequence

from .models import RequestTrace


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    """Return a linearly interpolated percentile, or None for empty input."""
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def inter_token_latencies(trace: RequestTrace) -> List[float]:
    """Return gaps between streamed token timestamps, excluding TTFT."""
    timestamps = trace.token_timestamps_ms
    return [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]


def _rounded(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 3)


def summarize(
    traces: Iterable[RequestTrace],
    wall_time_ms: Optional[float] = None,
    ttft_slo_ms: Optional[float] = None,
    total_latency_slo_ms: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Aggregate raw request traces into explicitly named system metrics."""
    records = list(traces)
    successful = [trace for trace in records if trace.succeeded]
    ttfts = [
        trace.first_token_latency_ms
        for trace in successful
        if trace.first_token_latency_ms is not None
    ]
    totals = [trace.total_latency_ms for trace in successful]
    itls = [latency for trace in successful for latency in inter_token_latencies(trace)]

    if wall_time_ms is None:
        # The only safe default is serial execution. Concurrent benchmark
        # drivers must pass their observed wall-clock duration explicitly.
        wall_time_ms = sum(totals)
    wall_time_seconds = wall_time_ms / 1000.0
    output_tokens = sum(trace.output_tokens for trace in successful)

    def meets_slo(trace: RequestTrace) -> bool:
        ttft_ok = ttft_slo_ms is None or (
            trace.first_token_latency_ms is not None
            and trace.first_token_latency_ms <= ttft_slo_ms
        )
        latency_ok = (
            total_latency_slo_ms is None
            or trace.total_latency_ms <= total_latency_slo_ms
        )
        return ttft_ok and latency_ok

    good_requests = sum(meets_slo(trace) for trace in successful)
    divisor = wall_time_seconds if wall_time_seconds > 0 else None

    return {
        "benchmark_wall_time_ms": _rounded(wall_time_ms),
        "requests_total": float(len(records)),
        "requests_succeeded": float(len(successful)),
        "success_rate": _rounded(len(successful) / len(records) if records else 0.0),
        "ttft_p50_ms": _rounded(percentile(ttfts, 0.50)),
        "ttft_p95_ms": _rounded(percentile(ttfts, 0.95)),
        "ttft_p99_ms": _rounded(percentile(ttfts, 0.99)),
        "e2e_p50_ms": _rounded(percentile(totals, 0.50)),
        "e2e_p95_ms": _rounded(percentile(totals, 0.95)),
        "e2e_p99_ms": _rounded(percentile(totals, 0.99)),
        "itl_p50_ms": _rounded(percentile(itls, 0.50)),
        "itl_p95_ms": _rounded(percentile(itls, 0.95)),
        "request_throughput_rps": _rounded(len(successful) / divisor if divisor else None),
        "output_throughput_tokens_s": _rounded(output_tokens / divisor if divisor else None),
        "goodput_rps": _rounded(good_requests / divisor if divisor else None),
    }
