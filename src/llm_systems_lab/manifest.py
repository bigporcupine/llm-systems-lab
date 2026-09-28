"""Create immutable pre-run manifests for experiment artifacts."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config_resolution import unresolved_revisions
from .environment import capture_environment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(
    experiment_dir: Path, output: Path, resolved_config: Path = None
) -> Dict[str, Any]:
    if not experiment_dir.is_dir():
        raise ValueError(f"experiment directory does not exist: {experiment_dir}")
    files = {
        str(path.relative_to(experiment_dir)): sha256_file(path)
        for path in sorted(experiment_dir.rglob("*")) if path.is_file()
    }
    if "README.md" not in files:
        raise ValueError("experiment directory must contain README.md")
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_dir": str(experiment_dir),
        "files_sha256": files,
        "environment": capture_environment(),
    }
    if resolved_config is not None:
        config_value = json.loads(resolved_config.read_text(encoding="utf-8"))
        unresolved = unresolved_revisions(config_value)
        if unresolved:
            raise ValueError(
                "resolved config still contains placeholders: " + ", ".join(unresolved)
            )
        manifest["resolved_config"] = str(resolved_config)
        manifest["resolved_config_sha256"] = sha256_file(resolved_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
