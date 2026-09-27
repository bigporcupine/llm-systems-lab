"""Deterministic traces used to validate the measurement pipeline.

Synthetic traces are never presented as model or hardware benchmark results.
"""

import random
from typing import List

from .models import RequestTrace


def generate_traces(requests: int, seed: int = 7) -> List[RequestTrace]:
    rng = random.Random(seed)
    traces = []
    for index in range(requests):
        input_tokens = rng.randint(128, 768)
        output_tokens = rng.randint(24, 96)
        ttft = 90.0 + input_tokens * 0.18 + rng.uniform(-8.0, 8.0)
        token_gap = 21.0 + rng.uniform(-2.5, 2.5)
        timestamps = [ttft + token_gap * token for token in range(output_tokens)]
        total = timestamps[-1] + rng.uniform(2.0, 8.0)
        traces.append(
            RequestTrace(
                request_id=f"synthetic-{index:04d}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_latency_ms=round(total, 3),
                first_token_latency_ms=round(ttft, 3),
                token_timestamps_ms=[round(value, 3) for value in timestamps],
            )
        )
    return traces

