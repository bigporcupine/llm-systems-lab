"""Canary promotion state machine used by the production experiment."""

from typing import Any, Dict, Iterable, List

from .canary import should_rollback


def evaluate_rollout(
    stable: Dict[str, float],
    canary_windows: Iterable[Dict[str, float]],
    policy: Dict[str, float],
    stages: Iterable[float] = (0.01, 0.05, 0.25, 0.5, 1.0),
) -> Dict[str, Any]:
    windows = list(canary_windows)
    fractions = list(stages)
    if not windows:
        raise ValueError("at least one canary window is required")
    events: List[Dict[str, Any]] = []
    stage_index = 0
    for window_index, window in enumerate(windows):
        decision = should_rollback(stable, window, policy)
        event = {"window": window_index, "fraction": fractions[stage_index], **decision}
        events.append(event)
        if decision["rollback"]:
            return {"status": "rolled-back", "final_fraction": 0.0, "events": events}
        if decision["reason"] == "guardrails-pass" and stage_index < len(fractions) - 1:
            stage_index += 1
    status = "promoted" if stage_index == len(fractions) - 1 else "observing"
    return {"status": status, "final_fraction": fractions[stage_index], "events": events}
