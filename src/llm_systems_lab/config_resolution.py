"""Resolve moving Hugging Face model names to immutable repository commits."""

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict


COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def is_immutable_revision(value: Any) -> bool:
    """Return whether a Hugging Face revision is a full immutable commit SHA."""
    return isinstance(value, str) and COMMIT_SHA.fullmatch(value) is not None


def require_immutable_revision(value: Any, label: str = "model_revision") -> str:
    if not is_immutable_revision(value):
        raise ValueError(f"{label} must be a full 40-character commit SHA")
    return str(value)


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
        if not is_immutable_revision(revision):
            config[revision_key] = resolver(str(model))
            resolved_any = True

    if config.get("draft_models"):
        existing = config.get("draft_revisions", {})
        config["draft_revisions"] = {
            model: (existing.get(model) if is_immutable_revision(existing.get(model))
                    else resolver(str(model)))
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


def unresolved_revisions(value: Any, path: str = "", revision_context: bool = False) -> list:
    """Return JSON paths containing explicit moving-revision placeholders."""
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key == "draft_revisions" and isinstance(child, dict):
                for model, revision in child.items():
                    if not is_immutable_revision(revision):
                        found.append(f"{child_path}.{model}")
            else:
                is_revision = key in {"model_revision", "target_revision", "draft_revision"}
                found.extend(unresolved_revisions(child, child_path, is_revision))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(unresolved_revisions(child, f"{path}[{index}]", revision_context))
    elif value == "resolve-before-run" or (revision_context and not is_immutable_revision(value)):
        found.append(path)
    return found
