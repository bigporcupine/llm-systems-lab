"""Resolve moving Hugging Face model names to immutable repository commits."""

import json
from pathlib import Path
from typing import Any, Callable, Dict


def resolve_config(
    config_path: Path,
    output_path: Path,
    resolver: Callable[[str], str] = None,
) -> Dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if resolver is None:
        try:
            from huggingface_hub import model_info
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub to resolve model revisions") from exc
        resolver = lambda model: model_info(model).sha

    pairs = (
        ("model", "model_revision"),
        ("base_model", "model_revision"),
        ("target_model", "target_revision"),
    )
    resolved_any = False
    for model_key, revision_key in pairs:
        model = config.get(model_key)
        if not model:
            continue
        revision = config.get(revision_key)
        if not revision or revision == "resolve-before-run":
            config[revision_key] = resolver(str(model))
            resolved_any = True

    if config.get("draft_models"):
        existing = config.get("draft_revisions", {})
        config["draft_revisions"] = {
            model: existing.get(model) or resolver(str(model))
            for model in config["draft_models"]
        }
        resolved_any = True

    if not resolved_any:
        # Still write a normalized copy; this is useful when the input was
        # already pinned and makes the command idempotent.
        config = dict(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def unresolved_revisions(value: Any, path: str = "") -> list:
    """Return JSON paths containing explicit moving-revision placeholders."""
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            found.extend(unresolved_revisions(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(unresolved_revisions(child, f"{path}[{index}]"))
    elif value == "resolve-before-run":
        found.append(path)
    return found
