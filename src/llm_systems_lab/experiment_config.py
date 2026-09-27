"""Validation and expansion for declarative experiment matrices."""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_matrix(path: Path, required: Iterable[str]) -> Dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(set(required).difference(config))
    if missing:
        raise ValueError(f"missing configuration fields: {', '.join(missing)}")
    return config


def expand_axes(axes: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not axes:
        raise ValueError("axes must not be empty")
    names = list(axes)
    for name, values in axes.items():
        if not values:
            raise ValueError(f"axis {name!r} must not be empty")
    return [
        dict(zip(names, combination))
        for combination in itertools.product(*(axes[name] for name in names))
    ]
