#!/usr/bin/env python3
"""Plan or execute the one-factor-at-a-time Experiment 007 ablation matrix."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    baseline = {"rank": 8, "learning_rate": 0.0001, "dataset_fraction": 1.0}
    cells = set()
    for method in config["methods"]:
        for seed in config["seeds"]:
            for rank in config["ranks"]:
                cells.add((method, rank, baseline["learning_rate"], baseline["dataset_fraction"], seed))
            for learning_rate in config["learning_rates"]:
                cells.add((method, baseline["rank"], learning_rate, baseline["dataset_fraction"], seed))
            for fraction in config["dataset_fractions"]:
                cells.add((method, baseline["rank"], baseline["learning_rate"], fraction, seed))
    commands = []
    for method, rank, learning_rate, fraction, seed in sorted(cells):
        name = f"{method}_r{rank}_lr{learning_rate:g}_data{fraction:g}_seed{seed}"
        output = args.output_root / name
        command = [
            sys.executable, "scripts/train_adapter.py", "--config", str(args.config),
            "--method", method, "--rank", str(rank), "--learning-rate", str(learning_rate),
            "--dataset-fraction", str(fraction), "--seed", str(seed),
            "--output-dir", str(output),
        ]
        commands.append({"name": name, "command": command})
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "ablation-plan.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    for cell in commands:
        print(" ".join(cell["command"]))
        if args.execute:
            subprocess.run(cell["command"], check=True)


if __name__ == "__main__":
    main()
