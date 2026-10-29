"""Reproducible walk-forward training for causal supervised context models."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from supervised.spec_v2 import CLS_TARGETS as V2_CLS_TARGETS
from supervised.spec_v2 import MAX_HORIZON_HOURS
from supervised.spec_v2 import REG_TARGETS as V2_REG_TARGETS
from supervised.spec_v3 import CLS_TARGETS as V3_CLS_TARGETS
from supervised.spec_v3 import REG_TARGETS as V3_REG_TARGETS


IDENTIFIERS = {"timestamp", "coin", "year", "open", "high", "low", "close", "volume"}


def feature_columns(
    frame: pd.DataFrame,
    cls_targets=V2_CLS_TARGETS,
    reg_targets=V2_REG_TARGETS,
) -> list[str]:
    excluded = IDENTIFIERS | set(cls_targets) | set(reg_targets)
    return [
        column for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def balanced_diagnostic_sample(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    """Cap a split while retaining every available coin/year group."""
    if maximum is None or len(frame) <= maximum:
        return frame
    groups = list(frame.groupby(["coin", "year"], sort=True).indices.values())
    per_group = max(1, maximum // len(groups))
    chosen = []
    for indices in groups:
        indices = np.asarray(indices)
        if len(indices) <= per_group:
            chosen.extend(indices.tolist())
        else:
            offsets = np.linspace(0, len(indices) - 1, per_group).round().astype(int)
            chosen.extend(indices[offsets].tolist())
    return frame.iloc[sorted(chosen)].reset_index(drop=True)


def split(frame: pd.DataFrame, train_end: int, validation_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_start = pd.Timestamp(f"{validation_year}-01-01", tz="UTC")
    embargo = validation_start - pd.Timedelta(hours=MAX_HORIZON_HOURS)
    train = frame.loc[(frame["year"] <= train_end) & (frame["timestamp"] < embargo)].copy()
    validation = frame.loc[frame["year"] == validation_year].copy()
    if train.empty or validation.empty:
        raise RuntimeError(f"Empty split train_end={train_end} validation={validation_year}")
    if train["timestamp"].max() >= validation_start - pd.Timedelta(hours=MAX_HORIZON_HOURS):
        raise RuntimeError("Embargo invariant failed")
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("train_ready/supervised_dataset_causal_v2.parquet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-end", type=int, required=True)
    parser.add_argument("--validation-year", type=int, required=True)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--targets", help="Comma-separated subset of v2 targets")
    parser.add_argument(
        "--partial-targets", action="store_true",
        help="Train each target on all rows where that exact horizon is valid",
    )
    parser.add_argument("--spec-version", type=int, choices=(2, 3), default=2)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["year"] = frame["timestamp"].dt.year.astype(np.int16)
    cls_targets, reg_targets = (
        (V3_CLS_TARGETS, V3_REG_TARGETS)
        if args.spec_version == 3 else (V2_CLS_TARGETS, V2_REG_TARGETS)
    )
    features = feature_columns(frame, cls_targets, reg_targets)
    requested = set(args.targets.split(",")) if args.targets else set(cls_targets + reg_targets)
    unknown = requested - set(cls_targets + reg_targets)
    if unknown:
        raise ValueError(f"Unknown targets: {sorted(unknown)}")
    train, validation = split(frame, args.train_end, args.validation_year)
    if not args.partial_targets:
        train = balanced_diagnostic_sample(train, args.max_train_rows)
        validation = balanced_diagnostic_sample(validation, args.max_validation_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "feature_list.json").write_text(json.dumps(features, indent=2) + "\n")

    common = dict(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=10.0,
        random_seed=args.seed,
        thread_count=args.threads,
        task_type="CPU",
        verbose=max(20, args.iterations // 10),
        allow_writing_files=False,
    )
    report = {
        "spec_version": args.spec_version,
        "train_end": args.train_end,
        "validation_year": args.validation_year,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_coins": int(train["coin"].nunique()),
        "validation_coins": int(validation["coin"].nunique()),
        "feature_count": len(features),
        "parameters": vars(args) | {"data": str(args.data), "output_dir": str(args.output_dir)},
        "targets": {},
    }
    for target in cls_targets + reg_targets:
        if target not in requested:
            continue
        target_train = train.dropna(subset=[target]).copy()
        target_validation = validation.dropna(subset=[target]).copy()
        if target_train.empty or target_validation.empty:
            raise RuntimeError(f"Empty valid target split for {target}")
        if args.partial_targets:
            target_train = balanced_diagnostic_sample(target_train, args.max_train_rows)
            target_validation = balanced_diagnostic_sample(target_validation, args.max_validation_rows)
        x_train, x_validation = target_train[features], target_validation[features]
        internal_start = target_train["timestamp"].max() - pd.Timedelta(days=90)
        internal_cutoff = internal_start - pd.Timedelta(hours=MAX_HORIZON_HOURS)
        tune_train = target_train.loc[target_train["timestamp"] < internal_cutoff]
        tune_validation = target_train.loc[target_train["timestamp"] >= internal_start]
        has_internal_tuning = not tune_train.empty and not tune_validation.empty
        started = time.perf_counter()
        if target in cls_targets:
            if has_internal_tuning:
                tuning_model = CatBoostClassifier(
                    loss_function="Logloss", eval_metric="AUC", od_type="Iter",
                    od_wait=max(50, args.iterations // 10), **common,
                )
                tuning_model.fit(
                    Pool(tune_train[features], tune_train[target].astype(int)),
                    eval_set=Pool(tune_validation[features], tune_validation[target].astype(int)),
                    use_best_model=True,
                )
                best_iterations = max(1, int(tuning_model.get_best_iteration()) + 1)
            else:
                best_iterations = min(50, args.iterations)
            model = CatBoostClassifier(
                loss_function="Logloss", eval_metric="AUC",
                **(common | {"iterations": best_iterations, "verbose": False}),
            )
            model.fit(Pool(x_train, target_train[target].astype(int)))
            prediction = model.predict_proba(x_validation)[:, 1]
            truth = target_validation[target].astype(int).to_numpy()
            metrics = {
                "roc_auc": float(roc_auc_score(truth, prediction)),
                "pr_auc": float(average_precision_score(truth, prediction)),
                "logloss": float(log_loss(truth, prediction)),
                "brier": float(brier_score_loss(truth, prediction)),
                "base_rate": float(truth.mean()),
            }
            prefix = "cls"
        else:
            if has_internal_tuning:
                tuning_model = CatBoostRegressor(
                    loss_function="RMSE", eval_metric="RMSE", od_type="Iter",
                    od_wait=max(50, args.iterations // 10), **common,
                )
                tuning_model.fit(
                    Pool(tune_train[features], tune_train[target]),
                    eval_set=Pool(tune_validation[features], tune_validation[target]),
                    use_best_model=True,
                )
                best_iterations = max(1, int(tuning_model.get_best_iteration()) + 1)
            else:
                best_iterations = min(50, args.iterations)
            model = CatBoostRegressor(
                loss_function="RMSE", eval_metric="RMSE",
                **(common | {"iterations": best_iterations, "verbose": False}),
            )
            model.fit(Pool(x_train, target_train[target]))
            prediction = model.predict(x_validation)
            truth = target_validation[target].to_numpy(dtype=np.float64)
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
                "mae": float(mean_absolute_error(truth, prediction)),
                "r2": float(r2_score(truth, prediction)),
                "spearman": float(pd.Series(truth).corr(pd.Series(prediction), method="spearman")),
            }
            prefix = "reg"
        model.save_model(str(args.output_dir / f"{prefix}_{target}.cbm"))
        metrics["train_rows"] = len(target_train)
        metrics["validation_rows"] = len(target_validation)
        metrics["selected_iterations"] = best_iterations
        metrics["seconds"] = time.perf_counter() - started
        report["targets"][target] = metrics
        (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"[METRICS] {target}: {metrics}")
    print(f"[OK] {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
