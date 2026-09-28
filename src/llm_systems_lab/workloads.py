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


def shared_prefix_prompts(prefix_size: int, count: int, seed: int = 7) -> List[str]:
    """Create byte-identical prefixes with deterministic request-unique suffixes."""
    if prefix_size < 0 or count <= 0:
        raise ValueError("prefix_size must be non-negative and count must be positive")
    prefix = deterministic_prompt(prefix_size, seed) if prefix_size else ""
    return [f"{prefix}\nUnique request {index}: explain one implication." for index in range(count)]


def speculative_prompts(workload: str, size_hint: int, count: int, seed: int = 7) -> List[str]:
    """Build deterministic agreement-oriented workloads for speculative decoding."""
    if size_hint <= 0 or count <= 0:
        raise ValueError("size_hint and count must be positive")
    if workload == "high-agreement":
        body = "Repeat the next predictable sequence: " + "one two three four " * max(1, size_hint // 4)
    elif workload == "general":
        body = deterministic_prompt(size_hint, seed)
    elif workload == "adversarial-low-agreement":
        rng = random.Random(seed)
        symbols = [f"{rng.getrandbits(32):08x}" for _ in range(size_hint)]
        body = "Continue and analyze these unrelated identifiers: " + " ".join(symbols)
    else:
        raise ValueError(f"unknown speculative workload: {workload}")
    return [f"{body}\nRequest {index}." for index in range(count)]
