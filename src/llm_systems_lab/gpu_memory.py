"""Bounded NVIDIA memory sampling for KV-cache experiments."""

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List


def nvidia_used_memory_mb(device: int = 0) -> float:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"--id={device}"],
        text=True, timeout=5,
    ).strip()
    return float(output.splitlines()[0].strip())


@dataclass
class MemorySampler:
    interval_seconds: float = 0.05
    device: int = 0
    samples_mb: List[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread = field(init=False)

    def start(self) -> None:
        self.samples_mb = []
        self._stop.clear()

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    self.samples_mb.append(nvidia_used_memory_mb(self.device))
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4))


def theoretical_kv_bytes(
    layers: int, kv_heads: int, head_dim: int, bytes_per_element: int,
    cached_tokens: int, sequences: int = 1,
) -> int:
    values = (layers, kv_heads, head_dim, bytes_per_element, cached_tokens, sequences)
    if any(value <= 0 for value in values):
        raise ValueError("KV-cache dimensions must be positive")
    return 2 * layers * kv_heads * head_dim * bytes_per_element * cached_tokens * sequences
