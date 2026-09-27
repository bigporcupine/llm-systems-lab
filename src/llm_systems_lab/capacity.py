"""Capacity and unit-cost calculations with explicit assumptions."""

import math
from typing import Dict


def capacity_plan(peak_rps: float, goodput_per_replica_rps: float, target_utilization: float) -> Dict[str, float]:
    if peak_rps <= 0 or goodput_per_replica_rps <= 0:
        raise ValueError("rates must be positive")
    if not 0 < target_utilization <= 1:
        raise ValueError("target_utilization must be in (0, 1]")
    usable = goodput_per_replica_rps * target_utilization
    return {"usable_rps_per_replica": usable, "replicas": math.ceil(peak_rps / usable)}


def cost_per_million_tokens(hourly_cost: float, tokens_per_second: float) -> float:
    if hourly_cost < 0 or tokens_per_second <= 0:
        raise ValueError("hourly_cost must be non-negative and throughput positive")
    return hourly_cost / (tokens_per_second * 3600) * 1_000_000
