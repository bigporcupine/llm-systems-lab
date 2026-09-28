"""Raw NVIDIA GPU utilization, memory, temperature, and power sampling."""

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


QUERY_FIELDS = (
    "index", "utilization.gpu", "memory.used", "memory.total",
    "power.draw", "temperature.gpu",
)


def collect_nvidia_sample() -> Dict[str, Any]:
    """Collect one sample from every visible NVIDIA GPU."""
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": type(exc).__name__, "gpus": []}
    if completed.returncode != 0:
        return {
            "available": False,
            "error": completed.stderr.strip() or f"nvidia-smi exited {completed.returncode}",
            "gpus": [],
        }
    gpus = []
    for line in completed.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(QUERY_FIELDS):
            continue
        parsed: Dict[str, Any] = {}
        for field, value in zip(QUERY_FIELDS, values):
            key = field.replace(".", "_")
            if value.lower() in {"n/a", "[not supported]", "not supported"}:
                parsed[key] = None
            else:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
        gpus.append(parsed)
    return {"available": bool(gpus), "error": None, "gpus": gpus}


class GpuTelemetrySampler:
    """Sample GPU state in a daemon thread and persist the unaggregated series."""

    def __init__(
        self,
        output: Path,
        interval_seconds: float = 1.0,
        collector: Callable[[], Dict[str, Any]] = collect_nvidia_sample,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.output = output
        self.interval_seconds = interval_seconds
        self.collector = collector
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self.collector()
            self.samples.append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **sample,
            })
            if not sample.get("available"):
                break
            self._stop.wait(self.interval_seconds)

    def start(self) -> "GpuTelemetrySampler":
        if self._thread is not None:
            raise RuntimeError("telemetry sampler is already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(6.0, self.interval_seconds + 5.0))
        artifact = {
            "schema_version": "1.0",
            "source": "system-telemetry",
            "collector": "nvidia-smi",
            "query_fields": list(QUERY_FIELDS),
            "interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "available": any(sample.get("available") for sample in self.samples),
            "samples": self.samples,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        return artifact

    def __enter__(self) -> "GpuTelemetrySampler":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
