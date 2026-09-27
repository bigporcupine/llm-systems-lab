"""Typed, serializable records used by every experiment."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RequestTrace:
    """Timing data for one completed inference request.

    All timestamps are durations relative to request start. Keeping raw request
    traces makes it possible to recompute aggregates instead of trusting a
    chart or a single average.
    """

    request_id: str
    input_tokens: int
    output_tokens: int
    total_latency_ms: float
    first_token_latency_ms: Optional[float]
    token_timestamps_ms: List[float] = field(default_factory=list)
    succeeded: bool = True
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RequestTrace":
        return cls(**value)


@dataclass(frozen=True)
class BenchmarkMetadata:
    experiment_id: str
    backend: str
    model: str
    hardware: str
    precision: str
    concurrency: int
    source: str
    notes: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)
    workload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: str
    created_at: str
    metadata: BenchmarkMetadata
    metrics: Dict[str, Optional[float]]
    traces: List[RequestTrace]

    @classmethod
    def create(
        cls,
        metadata: BenchmarkMetadata,
        metrics: Dict[str, Optional[float]],
        traces: List[RequestTrace],
    ) -> "BenchmarkResult":
        return cls(
            schema_version="1.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
            metrics=metrics,
            traces=traces,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
