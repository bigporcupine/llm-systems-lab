"""Formal-run evidence checks for raw experiment artifact directories."""

import json
from pathlib import Path
from typing import Any, Dict, List

from .config_resolution import unresolved_revisions
from .manifest import sha256_file


TRACE_FIELDS = {
    "request_id", "input_tokens", "output_tokens", "total_latency_ms",
    "first_token_latency_ms", "token_timestamps_ms", "succeeded",
}


def _audit_trace(trace: Dict[str, Any], location: str) -> List[str]:
    errors = []
    missing = sorted(TRACE_FIELDS.difference(trace))
    if missing:
        errors.append(f"{location}: missing trace fields: {', '.join(missing)}")
    if trace.get("succeeded") and trace.get("first_token_latency_ms") is None:
        errors.append(f"{location}: successful trace has no TTFT")
    if trace.get("total_latency_ms", 0) < 0:
        errors.append(f"{location}: negative total latency")
    return errors


def _audit_benchmark(value: Dict[str, Any], location: str) -> List[str]:
    errors = []
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        return [f"{location}: benchmark artifact has no metadata object"]
    if metadata.get("source") != "measured":
        errors.append(f"{location}: formal artifact source must be measured")
    environment = metadata.get("environment") or {}
    if not environment.get("git_commit"):
        errors.append(f"{location}: missing Git commit")
    if not environment.get("nvidia_gpu") and metadata.get("hardware") != "cpu":
        errors.append(f"{location}: missing accelerator metadata")
    workload = metadata.get("workload") or {}
    if not workload.get("model_revision"):
        errors.append(f"{location}: missing immutable model revision")
    if not workload.get("backend_version"):
        errors.append(f"{location}: missing backend version")
    traces = value.get("traces")
    if not isinstance(traces, list) or not traces:
        errors.append(f"{location}: no raw request traces")
    else:
        for index, trace in enumerate(traces):
            errors.extend(_audit_trace(trace, f"{location}.traces[{index}]"))
    return errors


def _trace_lists(value: Any) -> List[List[Dict[str, Any]]]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "traces" and isinstance(child, list):
                found.append(child)
            else:
                found.extend(_trace_lists(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_trace_lists(child))
    return found


def _audit_specialized(value: Dict[str, Any], location: str) -> List[str]:
    errors = []
    environment = value.get("environment") or {}
    if not environment.get("git_commit"):
        errors.append(f"{location}: missing Git commit")
    if not environment.get("nvidia_gpu") and value.get("hardware") != "cpu":
        errors.append(f"{location}: missing accelerator metadata")
    if not (value.get("model_revision") or value.get("target_revision")):
        errors.append(f"{location}: missing immutable model revision")
    if not (
        value.get("backend_version")
        or value.get("baseline_backend_version")
        or value.get("gateway_version")
    ):
        errors.append(f"{location}: missing backend version")
    trace_lists = _trace_lists(value)
    if not trace_lists or not any(trace_lists):
        errors.append(f"{location}: no raw request traces")
    for list_index, traces in enumerate(trace_lists):
        for trace_index, trace in enumerate(traces):
            errors.extend(_audit_trace(trace, f"{location}.trace_sets[{list_index}][{trace_index}]"))
    return errors


def _audit_training(value: Dict[str, Any], location: str) -> List[str]:
    errors = []
    environment = value.get("environment") or {}
    if not environment.get("git_commit"):
        errors.append(f"{location}: training artifact missing Git commit")
    if not environment.get("nvidia_gpu"):
        errors.append(f"{location}: training artifact missing accelerator metadata")
    if not value.get("model_revision"):
        errors.append(f"{location}: training artifact missing immutable model revision")
    if not value.get("training_dataset_sha256"):
        errors.append(f"{location}: training artifact missing dataset hash")
    if not value.get("selected_training_ids"):
        errors.append(f"{location}: training artifact missing selected example IDs")
    if not value.get("train_metrics"):
        errors.append(f"{location}: training artifact missing raw trainer metrics")
    return errors


def audit_artifacts(root: Path, output: Path = None) -> Dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    json_files = sorted(path for path in root.rglob("*.json") if path != output)
    errors: List[str] = []
    files = []
    benchmark_count = 0
    trace_count = 0
    training_count = 0
    for path in json_files:
        relative = str(path.relative_to(root))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(f"{relative}: invalid JSON: {exc}")
            continue
        placeholders = unresolved_revisions(value)
        errors.extend(f"{relative}: unresolved revision at {item}" for item in placeholders)
        if isinstance(value, dict) and value.get("metadata") is not None and value.get("traces") is not None:
            benchmark_count += 1
            trace_count += len(value.get("traces") or [])
            errors.extend(_audit_benchmark(value, relative))
        elif isinstance(value, dict) and value.get("source") == "measured":
            benchmark_count += 1
            lists = _trace_lists(value)
            trace_count += sum(len(traces) for traces in lists)
            errors.extend(_audit_specialized(value, relative))
        elif isinstance(value, dict) and value.get("source") == "training":
            training_count += 1
            errors.extend(_audit_training(value, relative))
        files.append({"path": relative, "sha256": sha256_file(path)})
    if not json_files:
        errors.append("artifact directory contains no JSON files")
    if benchmark_count == 0 and training_count == 0:
        errors.append("artifact directory contains no measured or training artifacts")
    manifest_paths = [path for path in json_files if path.name == "manifest.json"]
    if not manifest_paths:
        errors.append("artifact directory contains no manifest.json")
    else:
        manifest_value = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
        if not manifest_value.get("resolved_config_sha256"):
            errors.append("manifest.json does not identify a resolved configuration")
    result = {
        "schema_version": "1.0", "artifact_root": str(root),
        "passed": not errors, "benchmark_artifacts": benchmark_count,
        "training_artifacts": training_count, "request_traces": trace_count,
        "files": files, "errors": errors,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
