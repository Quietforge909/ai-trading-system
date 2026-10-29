"""Generate strict out-of-fold supervised context on causal features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from features.continuity import segment_ids
from supervised.spec_v3 import CLS_TARGETS, REG_TARGETS


def load_models(directory: Path, targets: list[str]):
    models = {}
    for target in targets:
        if target in CLS_TARGETS:
            model = CatBoostClassifier()
            model.load_model(str(directory / f"cls_{target}.cbm"))
        elif target in REG_TARGETS:
            model = CatBoostRegressor()
            model.load_model(str(directory / f"reg_{target}.cbm"))
        else:
            raise ValueError(target)
        models[target] = model
    return models


def gap_safe_market_regime(frame: pd.DataFrame) -> pd.Series:
    """Past-only cross-sectional breadth without bridging candle gaps."""
    positive = pd.Series(np.nan, index=frame.index, dtype=np.float32)
    for _, indices in frame.groupby("coin", sort=False).groups.items():
        coin = frame.loc[indices].sort_values("timestamp")
        segments = segment_ids(coin["timestamp"])
        for _, segment in coin.groupby(segments.to_numpy(), sort=True):
            trailing = segment["close"].pct_change(24)
            positive.loc[segment.index] = trailing.gt(0).where(trailing.notna()).astype(np.float32)
    breadth = positive.groupby(frame["timestamp"]).mean()
    return frame["timestamp"].map(breadth).fillna(0.5).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("train_ready/features/final_causal_v1"))
    parser.add_argument("--models-root", type=Path, default=Path("models/supervised_causal_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("train_ready/features/final_causal_v2_supervised6"))
    parser.add_argument(
        "--targets",
        default="opportunity_7d,drawdown_risk_7d,vol_expansion_risk_7d,future_max_return_7d,future_min_return_7d,future_vol_7d",
    )
    parser.add_argument("--first-validation-year", type=int, default=2018)
    parser.add_argument("--last-validation-year", type=int, default=2026)
    parser.add_argument("--spec-version", type=int, default=2)
    args = parser.parse_args()
    targets = args.targets.split(",")
    if args.output_dir.exists() and (
        (args.output_dir / "manifest.json").exists()
        or any(args.output_dir.glob("*.parquet"))
    ):
        raise RuntimeError(
            f"Refusing to overwrite existing OOF artifact family: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.feature_dir.glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No feature files in {args.feature_dir}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["coin"] = path.stem
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = combined["timestamp"].dt.year.astype(np.int16)
    combined["market_regime_strength"] = gap_safe_market_regime(combined)

    covered = pd.Series(False, index=combined.index)
    feature_list = None
    fold_manifest = []
    for validation_year in range(args.first_validation_year, args.last_validation_year + 1):
        train_end = validation_year - 1
        directory = args.models_root / f"fold_train_through_{train_end}_val_{validation_year}"
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(f"Missing strict fold: {directory}")
        features = json.loads((directory / "feature_list.json").read_text())
        if feature_list is None:
            feature_list = features
        elif feature_list != features:
            raise RuntimeError(f"Feature-list mismatch in {directory}")
        models = load_models(directory, targets)
        mask = combined["year"].eq(validation_year)
        if not mask.any():
            continue
        x = combined.loc[mask, feature_list]
        for target, model in models.items():
            if target in CLS_TARGETS:
                prediction = model.predict_proba(x)[:, 1]
            else:
                prediction = model.predict(x)
            combined.loc[mask, f"sup_{target}"] = prediction.astype(np.float32)
        covered |= mask
        metrics = json.loads(metrics_path.read_text())
        fold_manifest.append({
            "train_end": train_end,
            "validation_year": validation_year,
            "rows": int(mask.sum()),
            "train_rows": metrics["train_rows"],
            "validation_coins": metrics["validation_coins"],
        })
        print(f"[FOLD] train<={train_end} -> {validation_year}: {int(mask.sum()):,} predictions")

    combined = combined.loc[covered].copy()
    for target in targets:
        column = f"sup_{target}"
        if combined[column].isna().any():
            raise RuntimeError(f"Missing OOF prediction in {column}")
        combined[f"{column}_pct"] = (
            combined.groupby("timestamp")[column].rank(pct=True).astype(np.float32)
        )
    for coin, frame in combined.groupby("coin", sort=True):
        frame.drop(columns=["coin"], inplace=True)
        frame.to_parquet(args.output_dir / f"{coin}.parquet", index=False)
    manifest = {
        "spec_version": args.spec_version,
        "targets": targets,
        "feature_count": len(feature_list or []),
        "output_rows": len(combined),
        "output_coins": int(combined["coin"].nunique()),
        "folds": fold_manifest,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[OK] {args.output_dir}: rows={len(combined):,} coins={combined['coin'].nunique()}")


if __name__ == "__main__":
    main()
