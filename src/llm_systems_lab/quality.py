"""Deterministic task-quality metrics for model and quantization comparisons."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def normalize_answer(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    return " ".join(value.split())


def exact_match(prediction: str, references: Iterable[str]) -> bool:
    normalized = normalize_answer(prediction)
    return any(normalized == normalize_answer(reference) for reference in references)


def token_f1(prediction: str, references: Iterable[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    best = 0.0
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        remaining = list(reference_tokens)
        common = 0
        for token in prediction_tokens:
            if token in remaining:
                common += 1
                remaining.remove(token)
        if not prediction_tokens or not reference_tokens:
            score = float(prediction_tokens == reference_tokens)
        else:
            precision = common / len(prediction_tokens)
            recall = common / len(reference_tokens)
            score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not {"id", "prompt", "references"}.issubset(row):
            raise ValueError(f"invalid evaluation row at line {line_number}")
        rows.append(row)
    if not rows:
        raise ValueError("evaluation set must not be empty")
    return rows


def score_predictions(dataset: List[Dict[str, Any]], predictions: Dict[str, str]) -> Dict[str, float]:
    missing = [row["id"] for row in dataset if row["id"] not in predictions]
    if missing:
        raise ValueError(f"missing predictions for: {', '.join(missing)}")
    em = [exact_match(predictions[row["id"]], row["references"]) for row in dataset]
    f1 = [token_f1(predictions[row["id"]], row["references"]) for row in dataset]
    return {"examples": len(dataset), "exact_match": sum(em) / len(em), "token_f1": sum(f1) / len(f1)}
