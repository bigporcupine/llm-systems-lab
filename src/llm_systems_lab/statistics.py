"""Small dependency-free statistical helpers used by experiment reports."""

import math
import random
import statistics
from typing import Iterable, List, Tuple


def bootstrap_mean_ci(
    values: Iterable[float], confidence: float = 0.95, samples: int = 2000, seed: int = 7
) -> Tuple[float, float, float]:
    """Return (mean, lower, upper) using a deterministic percentile bootstrap."""
    data = [float(value) for value in values]
    if not data:
        raise ValueError("values must not be empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if len(data) == 1:
        return data[0], data[0], data[0]
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(samples):
        means.append(statistics.mean(rng.choice(data) for _ in data))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    lower = means[max(0, math.floor(tail * samples))]
    upper = means[min(samples - 1, math.ceil((1.0 - tail) * samples) - 1)]
    return statistics.mean(data), lower, upper


def relative_change(candidate: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return (candidate - baseline) / baseline
