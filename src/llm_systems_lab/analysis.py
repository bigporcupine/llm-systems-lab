"""Cross-run analysis used by optimization experiments."""

from typing import Any, Dict, Iterable, List, Optional


def saturation_knee(cells: Iterable[Dict[str, Any]], latency_budget_ms: float) -> Optional[Dict[str, Any]]:
    """Highest-throughput cell whose mean p95 latency remains inside budget."""
    eligible: List[Dict[str, Any]] = []
    for cell in cells:
        aggregate = cell["aggregate"]
        latency = aggregate["e2e_p95_ms"]["mean"]
        if latency <= latency_budget_ms:
            eligible.append(cell)
    if not eligible:
        return None
    return max(eligible, key=lambda cell: cell["aggregate"]["output_throughput_tokens_s"]["mean"])


def break_even(rows: Iterable[Dict[str, float]], baseline_key: str, candidate_key: str) -> Optional[Dict[str, float]]:
    """First ordered row where candidate latency is no worse than baseline."""
    for row in rows:
        if row[candidate_key] <= row[baseline_key]:
            return row
    return None


def quality_gate(baseline: float, candidate: float, max_absolute_drop: float) -> Dict[str, Any]:
    drop = baseline - candidate
    return {"baseline": baseline, "candidate": candidate, "absolute_drop": drop,
            "max_absolute_drop": max_absolute_drop, "passed": drop <= max_absolute_drop}
