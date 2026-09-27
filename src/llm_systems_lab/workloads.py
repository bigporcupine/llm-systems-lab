"""Deterministic workloads shared by every serving backend."""

import random
from typing import List


_WORDS = (
    "system model request token cache latency throughput scheduler memory "
    "attention tensor serving benchmark inference prompt decode prefill "
    "quality workload concurrency measurement production experiment"
).split()


def deterministic_prompt(size_hint: int, seed: int = 7) -> str:
    """Build stable natural-language-like input; the server reports actual tokens."""
    if size_hint <= 0:
        raise ValueError("size_hint must be positive")
    rng = random.Random(seed)
    words: List[str] = [
        "Analyze the following sequence and summarize its systems implications:"
    ]
    words.extend(rng.choice(_WORDS) for _ in range(size_hint))
    return " ".join(words)

