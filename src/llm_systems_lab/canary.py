"""Deterministic canary assignment and rollback evaluation."""

import hashlib
from typing import Any, Dict


def assign_lane(request_id: str, canary_fraction: float) -> str:
    if not 0 <= canary_fraction <= 1:
        raise ValueError("canary_fraction must be in [0, 1]")
    bucket = int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "big") / 2**64
    return "canary" if bucket < canary_fraction else "stable"


def should_rollback(stable: Dict[str, float], canary: Dict[str, float], policy: Dict[str, float]) -> Dict[str, Any]:
    if canary["requests"] < policy["minimum_requests"]:
        return {"rollback": False, "reason": "insufficient-samples"}
    reasons = []
    if canary["error_rate"] > policy["max_error_rate"]:
        reasons.append("error-rate")
    if stable["p95_latency_ms"] > 0 and canary["p95_latency_ms"] / stable["p95_latency_ms"] - 1 > policy["max_p95_latency_regression"]:
        reasons.append("latency-regression")
    if stable["quality"] - canary["quality"] > policy["max_quality_regression"]:
        reasons.append("quality-regression")
    return {"rollback": bool(reasons), "reason": ",".join(reasons) if reasons else "guardrails-pass"}
