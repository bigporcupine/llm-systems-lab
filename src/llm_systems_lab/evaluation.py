"""Generate and score deterministic predictions through a live endpoint."""

import json
from pathlib import Path
from typing import Any, Dict

from .environment import capture_environment
from .online import stream_request
from .quality import load_jsonl, score_predictions


def evaluate_endpoint(
    base_url: str,
    model: str,
    dataset_path: Path,
    output_dir: Path,
    max_tokens: int = 128,
    timeout_seconds: float = 120.0,
) -> Dict[str, Any]:
    dataset = load_jsonl(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: Dict[str, str] = {}
    traces = []
    for row in dataset:
        trace = stream_request(
            base_url, model, str(row["id"]), 0, max_tokens,
            timeout_seconds, prompt=row["prompt"], force_output_length=False,
        )
        traces.append(trace)
        if not trace.succeeded:
            raise RuntimeError(f"evaluation request {row['id']} failed: {trace.error}")
        predictions[str(row["id"])] = trace.completion_text or ""
    scores = score_predictions(dataset, predictions)
    artifact = {
        "schema_version": "1.0", "model": model,
        "dataset": str(dataset_path), "environment": capture_environment(),
        "scores": scores, "predictions": predictions,
        "traces": [trace.__dict__ for trace in traces],
    }
    (output_dir / "evaluation.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    return artifact
