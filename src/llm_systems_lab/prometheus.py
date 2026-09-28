"""Minimal Prometheus text snapshot utilities for engine experiments."""

import re
from typing import Dict
from urllib import request


_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)$")


def parse_prometheus(text: str) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for line in text.splitlines():
        match = _SAMPLE.match(line.strip())
        if not match:
            continue
        name, labels, value = match.groups()
        numeric = float(value)
        totals[name] = totals.get(name, 0.0) + numeric
        if labels:
            totals[f"{name}{labels}"] = numeric
    return totals


def fetch_prometheus(url: str, timeout_seconds: float = 10.0) -> Dict[str, float]:
    return parse_prometheus(fetch_prometheus_text(url, timeout_seconds))


def fetch_prometheus_text(url: str, timeout_seconds: float = 10.0) -> str:
    with request.urlopen(url, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")


def metric_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    return {name: value - before.get(name, 0.0) for name, value in after.items()}
