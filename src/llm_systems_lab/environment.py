"""Capture enough runtime context to make benchmark results auditable."""

import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any, Dict, Optional


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _nvidia_gpu() -> Optional[Dict[str, str]]:
    fields = "name,memory.total,driver_version"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not output:
        return None
    name, memory_mb, driver = [part.strip() for part in output.splitlines()[0].split(",")]
    return {"name": name, "memory_total_mb": memory_mb, "driver": driver}


def capture_environment() -> Dict[str, Any]:
    """Return stable environment facts without collecting user identifiers."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: version
            for name in ("torch", "transformers", "vllm")
            if (version := _package_version(name)) is not None
        },
        "git_commit": _git_commit(),
        "executable": os.path.basename(sys.executable),
        "nvidia_gpu": _nvidia_gpu(),
    }
