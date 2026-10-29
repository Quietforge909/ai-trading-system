"""Build gap-safe, per-target-valid supervised data from causal-v3 features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features.continuity import contiguous_segments, prepare_hourly_frame
from supervised.dataset_v2 import build_targets
from supervised.spec_v3 import CLS_TARGETS, REG_TARGETS


SEVEN_DAY_TARGETS = {
    "opportunity_7d", "drawdown_risk_7d", "vol_expansion_risk_7d",
    "future_max_return_7d", "future_min_return_7d", "future_vol_7d",
    "future_terminal_return_7d", "trend_efficiency_7d",
    "profitable_after_fee_168h", "future_net_return_168h",
}


def after_fee_return(entry: pd.Series, exit_: pd.Series, fee_rate: float = 0.005) -> pd.Series:
    budget = 1000.0
    estimated_notional = budget / (1.0 + fee_rate)
    entry_fee = round(estimated_notional * fee_rate, 1)
    notional = budget - entry_fee
    size = notional / entry.astype(np.float64)
    exit_notional = size * exit_.astype(np.float64)
    exit_fee = np.round(exit_notional * fee_rate, 1)
    return (exit_notional - exit_fee) / budget - 1.0


def build_gap_safe_targets(frame: pd.DataFrame, coin: str) -> pd.DataFrame:
    pieces = []
    for segment in contiguous_segments(frame):
        built = build_targets(segment, coin)
        for hours in (24, 72, 168):
            future = built["close"].shift(-hours)
            valid = future.notna()
            net = after_fee_return(built["close"], future)
            built[f"future_net_return_{hours}h"] = net.clip(-0.95, 2.0).where(valid).astype(np.float32)
            built[f"profitable_after_fee_{hours}h"] = net.gt(0.0).where(valid).astype(np.float32)
        complete_7d = built["close"].shift(-(7 * 24)).notna()
        complete_14d = built["close"].shift(-(14 * 24)).notna()
        complete_30d = built["close"].shift(-(30 * 24)).notna()
        for target in SEVEN_DAY_TARGETS:
            built.loc[~complete_7d, target] = np.nan
        for hours in (24, 72):
            complete = built["close"].shift(-hours).notna()
            built.loc[~complete, f"future_net_return_{hours}h"] = np.nan
            built.loc[~complete, f"profitable_after_fee_{hours}h"] = np.nan
        built.loc[~complete_14d, "opportunity_14d"] = np.nan
        built.loc[~complete_30d, "opportunity_30d"] = np.nan
        pieces.append(built)
    return pd.concat(pieces, ignore_index=True)


def add_gap_safe_regime(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, coin_frame in frame.groupby("coin", sort=False):
        for segment in contiguous_segments(coin_frame):
            segment = segment.copy()
            segment["coin"] = coin_frame["coin"].iloc[0]
            trailing = segment["close"].pct_change(24)
            segment["current_positive_24h"] = trailing.gt(0).where(trailing.notna())
            pieces.append(segment)
    rebuilt = pd.concat(pieces, ignore_index=True)
    regime = rebuilt.groupby("timestamp")["current_positive_24h"].mean()
    rebuilt["market_regime_strength"] = rebuilt["timestamp"].map(regime).fillna(0.5).astype(np.float32)
    return rebuilt.drop(columns=["current_positive_24h"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("train_ready/features/final_causal_v3"))
    parser.add_argument("--output", type=Path, default=Path("train_ready/supervised_dataset_causal_v3.parquet"))
    args = parser.parse_args()
    paths = sorted(args.feature_dir.glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No causal-v3 feature files in {args.feature_dir}")
    frames = []
    per_coin = {}
    for number, path in enumerate(paths, start=1):
        built = build_gap_safe_targets(prepare_hourly_frame(pd.read_parquet(path)), path.stem)
        frames.append(built)
        per_coin[path.stem] = {
            "rows": len(built),
            "valid_targets": {target: int(built[target].notna().sum()) for target in CLS_TARGETS + REG_TARGETS},
        }
        print(f"[{number:03d}/{len(paths):03d}] {path.stem}: {len(built):,} rows")
    dataset = add_gap_safe_regime(pd.concat(frames, ignore_index=True))
    dataset["year"] = pd.to_datetime(dataset["timestamp"], utc=True).dt.year.astype(np.int16)
    feature_columns = [column for column in dataset.columns if column not in set(CLS_TARGETS + REG_TARGETS)]
    if dataset[feature_columns].select_dtypes(include=[np.number]).isna().any().any():
        raise RuntimeError("NaN found in causal feature columns")
    for column in dataset.select_dtypes(include=["float64"]).columns:
        dataset[column] = dataset[column].astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)
    manifest = {
        "spec_version": 3,
        "feature_dir": str(args.feature_dir), "output": str(args.output),
        "rows": len(dataset), "coins": int(dataset["coin"].nunique()),
        "target_valid_rows": {
            target: int(dataset[target].notna().sum()) for target in CLS_TARGETS + REG_TARGETS
        },
        "per_coin": per_coin,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[OK] {args.output}: rows={len(dataset):,} coins={manifest['coins']}")


if __name__ == "__main__":
    main()
