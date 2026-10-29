"""Reproducible full-row walk-forward trainer for the frozen v3 context set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


STABLE_TARGETS = (
    "opportunity_7d",
    "drawdown_risk_7d",
    "vol_expansion_risk_7d",
    "profitable_after_fee_72h",
    "future_min_return_7d",
    "future_vol_7d",
)


def completed_fold(directory: Path, targets: tuple[str, ...], parameters: dict) -> bool:
    metrics = directory / "metrics.json"
    if not metrics.exists():
        return False
    report = json.loads(metrics.read_text())
    if set(report.get("targets", {})) != set(targets):
        return False
    saved = report.get("parameters", {})
    return all(saved.get(key) == value for key, value in parameters.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("train_ready/supervised_dataset_causal_v3.parquet"))
    parser.add_argument("--models-root", type=Path, default=Path("models/supervised_causal_v3_stable6"))
    parser.add_argument("--first-validation-year", type=int, default=2018)
    parser.add_argument("--last-validation-year", type=int, default=2026)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--targets", default=",".join(STABLE_TARGETS))
    args = parser.parse_args()
    targets = tuple(args.targets.split(","))
    args.models_root.mkdir(parents=True, exist_ok=True)
    for validation_year in range(args.first_validation_year, args.last_validation_year + 1):
        train_end = validation_year - 1
        directory = args.models_root / f"fold_train_through_{train_end}_val_{validation_year}"
        identity = {
            "train_end": train_end,
            "validation_year": validation_year,
            "depth": args.depth,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "targets": args.targets,
            "spec_version": 3,
            "partial_targets": True,
        }
        if completed_fold(directory, targets, identity):
            print(f"[SKIP COMPLETE] {directory}")
            continue
        command = [
            sys.executable, "-m", "supervised.train_v2",
            "--spec-version", "3", "--partial-targets",
            "--data", str(args.data), "--output-dir", str(directory),
            "--train-end", str(train_end), "--validation-year", str(validation_year),
            "--depth", str(args.depth), "--iterations", str(args.iterations),
            "--learning-rate", str(args.learning_rate), "--seed", str(args.seed),
            "--targets", args.targets, "--threads", str(args.threads),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
