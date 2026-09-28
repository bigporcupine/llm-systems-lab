"""Build a versioned Markdown report only from successfully audited artifacts."""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .manifest import sha256_file


VERSION_PATTERN = re.compile(r"^v?\d+\.\d+(?:\.\d+)?(?:-[0-9A-Za-z.-]+)?$")
DISPLAY_METRICS = (
    "ttft_p50_ms", "ttft_p95_ms", "e2e_p95_ms", "itl_p95_ms",
    "request_throughput_rps", "output_throughput_tokens_s", "goodput_rps",
    "success_rate",
)


def _load_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _escape(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _link(report: Path, target: Path) -> str:
    relative = os.path.relpath(target, report.parent).replace(os.sep, "/")
    return f"[{_escape(target.name)}]({relative})"


def _verify_audit(artifact_dir: Path, audit_path: Path, audit: Dict[str, Any], output: Path) -> None:
    if audit.get("passed") is not True:
        raise ValueError("artifact audit did not pass")
    recorded_root = Path(str(audit.get("artifact_root", ""))).expanduser()
    if not recorded_root.is_absolute():
        recorded_root = Path.cwd() / recorded_root
    recorded_root = recorded_root.resolve()
    if recorded_root != artifact_dir.resolve():
        raise ValueError("audit artifact_root does not match --artifact-dir")

    expected = {}
    for entry in audit.get("files", []):
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError("audit contains an invalid file entry")
        expected[str(entry["path"])] = str(entry["sha256"])
    if not expected:
        raise ValueError("audit contains no hashed artifact files")

    for relative, digest in expected.items():
        path = (artifact_dir / relative).resolve()
        try:
            path.relative_to(artifact_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"audit path escapes artifact directory: {relative}") from exc
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"artifact changed after audit: {relative}")

    ignored = {audit_path.resolve(), output.resolve()}
    current = {
        str(path.relative_to(artifact_dir))
        for path in artifact_dir.rglob("*.json")
        if path.resolve() not in ignored
    }
    if current != set(expected):
        changed = sorted(current.symmetric_difference(expected))
        raise ValueError("artifact JSON set changed after audit: " + ", ".join(changed))


def _identity_rows(records: Iterable[Tuple[Path, Dict[str, Any]]]) -> List[Tuple[str, str]]:
    commits, models, backends, hardware, precisions = set(), set(), set(), set(), set()
    for _, value in records:
        metadata = value.get("metadata") or value
        environment = metadata.get("environment") or value.get("environment") or {}
        workload = metadata.get("workload") or value.get("workload") or {}
        if environment.get("git_commit"):
            commits.add(str(environment["git_commit"]))
        model = metadata.get("model") or value.get("model") or value.get("base_model")
        revision = workload.get("model_revision") or value.get("model_revision") or value.get("target_revision")
        if model or revision:
            models.add(f"{model or 'unspecified'} @ {revision or 'unspecified'}")
        draft_model = workload.get("draft_model") or value.get("draft_model")
        draft_revision = workload.get("draft_revision") or value.get("draft_revision")
        if draft_model or draft_revision:
            models.add(f"draft: {draft_model or 'unspecified'} @ {draft_revision or 'unspecified'}")
        backend = metadata.get("backend") or value.get("backend") or "unspecified"
        version = workload.get("backend_version") or value.get("backend_version") or value.get("baseline_backend_version")
        backends.add(f"{backend} {version or 'unspecified'}")
        accelerator = environment.get("nvidia_gpu") or metadata.get("hardware") or value.get("hardware")
        if accelerator:
            hardware.add(json.dumps(accelerator, sort_keys=True) if isinstance(accelerator, dict) else str(accelerator))
        precision = metadata.get("precision") or value.get("precision")
        if precision:
            precisions.add(str(precision))
    return [
        ("Repository commit", ", ".join(sorted(commits)) or "unspecified"),
        ("Model and immutable revision", "; ".join(sorted(models)) or "unspecified"),
        ("Backend and version", "; ".join(sorted(backends)) or "unspecified"),
        ("Hardware", "; ".join(sorted(hardware)) or "unspecified"),
        ("Precision / quantization", "; ".join(sorted(precisions)) or "unspecified"),
    ]


def build_report(
    artifact_dir: Path,
    audit_path: Path,
    title: str,
    version: str,
    claim: str,
    limitations_path: Path,
    output: Path,
    decision: str = "Pending review against the stated quality and SLO guardrails.",
) -> Path:
    """Validate an audit and render its measured evidence without inventing values."""
    artifact_dir = artifact_dir.resolve()
    audit_path = audit_path.resolve()
    output = output.resolve()
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("version must look like v1.0, 1.0.0, or 1.0.0-rc1")
    if not claim.strip():
        raise ValueError("claim must not be empty")
    if not limitations_path.is_file() or not limitations_path.read_text(encoding="utf-8").strip():
        raise ValueError("limitations file must exist and must not be empty")

    audit = _load_object(audit_path)
    _verify_audit(artifact_dir, audit_path, audit, output)
    records = []
    for entry in audit["files"]:
        path = artifact_dir / entry["path"]
        value = _load_object(path)
        if ((isinstance(value.get("metadata"), dict) and isinstance(value.get("traces"), list))
                or value.get("source") in {"measured", "training", "derived-from-measured"}):
            records.append((path, value))
    if not records:
        raise ValueError("audit contains no reportable measured or training artifacts")

    lines = [f"# {title} {version}", "", "## Claim", "", claim.strip(), "",
             "## Reproduction identity", "", "| Field | Value |", "|---|---|"]
    for field, value in _identity_rows(records):
        lines.append(f"| {_escape(field)} | {_escape(value)} |")
    lines.extend([
        f"| Report version | {_escape(version)} |",
        f"| Artifact audit | {_link(output, audit_path)} (`sha256:{sha256_file(audit_path)}`) |",
        "", "## Audited results", "",
        "Every row below is generated from an artifact covered by the passing audit.", "",
        "| Artifact | Experiment / variant | " + " | ".join(DISPLAY_METRICS) + " |",
        "|---|---|" + "---|" * len(DISPLAY_METRICS),
    ])
    benchmark_rows = 0
    failures = []
    specialized = []
    for path, value in records:
        metadata = value.get("metadata")
        metrics = value.get("metrics")
        if isinstance(metadata, dict) and isinstance(metrics, dict):
            label = metadata.get("experiment_id", "benchmark")
            workload = metadata.get("workload") or {}
            variant = workload.get("variant") or metadata.get("backend") or ""
            cells = [_link(output, path), _escape(f"{label} / {variant}")]
            cells.extend(_escape(metrics.get(name)) for name in DISPLAY_METRICS)
            lines.append("| " + " | ".join(cells) + " |")
            benchmark_rows += 1
            failed = sum(1 for trace in value.get("traces", []) if not trace.get("succeeded"))
            if failed:
                failures.append(f"- {_link(output, path)}: {failed} failed request trace(s).")
        else:
            specialized.append((path, value))
    if benchmark_rows == 0:
        lines.append("| — | No standalone BenchmarkResult artifacts | " + " | ".join("—" for _ in DISPLAY_METRICS) + " |")

    if specialized:
        lines.extend(["", "### Specialized artifacts", ""])
        for path, value in specialized:
            kind = value.get("experiment") or value.get("source") or "specialized result"
            details = []
            for key in (
                "cost_per_million_total_tokens_usd",
                "cost_per_million_output_tokens_usd",
                "pricing", "capacity", "break_even_by_prefix_length",
                "train_metrics", "evaluation_scores",
            ):
                if key in value:
                    details.append(f"{key}={json.dumps(value[key], sort_keys=True)}")
            suffix = "; ".join(details) if details else "See audited raw artifact."
            lines.append(f"- {_link(output, path)} — **{_escape(kind)}**: {_escape(suffix)}")

    lines.extend(["", "## Failed runs and exclusions", ""])
    lines.extend(failures or ["No failed request traces were present in the audited artifacts. No exclusions were applied by the report builder."])
    lines.extend([
        "", "## Evidence audit", "",
        f"Audit passed with **{audit.get('benchmark_artifacts', 0)}** benchmark artifact(s), "
        f"**{audit.get('training_artifacts', 0)}** training artifact(s), and "
        f"**{audit.get('request_traces', 0)}** request trace(s).",
        "", "## Limitations", "", limitations_path.read_text(encoding="utf-8").strip(),
        "", "## Decision", "", decision.strip(), "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
