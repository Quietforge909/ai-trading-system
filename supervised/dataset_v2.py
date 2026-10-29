"""Build bounded supervised targets from causal feature artifacts.

Targets use future values only as labels.  The last 30 days of each coin are
unlabelled and excluded; no early history is discarded merely to make the
learning problem easier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

from supervised.spec_v2 import CLS_TARGETS, MAX_HORIZON_HOURS, REG_TARGETS


def _future_rolling(series: pd.Series, hours: int, operation: str) -> pd.Series:
    indexer = FixedForwardWindowIndexer(window_size=hours)
    roller = series.rolling(indexer, min_periods=hours)
    result = getattr(roller, operation)()
    # The forward window at t includes t; the window at t+1 is exactly t+1:t+H.
    return result.shift(-1)


def build_targets(frame: pd.DataFrame, coin: str) -> pd.DataFrame:
    frame = frame.sort_values("timestamp").reset_index(drop=True).copy()
    close = frame["close"].astype(np.float64)
    hourly_return = close.pct_change()

    max_7d = _future_rolling(close, 7 * 24, "max")
    max_14d = _future_rolling(close, 14 * 24, "max")
    max_30d = _future_rolling(close, 30 * 24, "max")
    min_7d = _future_rolling(close, 7 * 24, "min")
    future_vol = _future_rolling(hourly_return, 7 * 24, "std")
    path_variation = _future_rolling(hourly_return.abs(), 7 * 24, "sum")
    terminal_return = close.shift(-(7 * 24)) / close - 1.0

    max_return_7d = max_7d / close - 1.0
    min_return_7d = min_7d / close - 1.0
    trend_efficiency = terminal_return / (path_variation + 1e-6)
    trailing_vol = hourly_return.rolling(24, min_periods=24).std()

    # Thresholds exceed the configured 1% round-trip fee by a useful margin.
    frame["opportunity_7d"] = (max_return_7d >= 0.05).astype(np.float32)
    frame["opportunity_14d"] = (max_14d / close - 1.0 >= 0.10).astype(np.float32)
    frame["opportunity_30d"] = (max_30d / close - 1.0 >= 0.15).astype(np.float32)
    frame["drawdown_risk_7d"] = (min_return_7d <= -0.06).astype(np.float32)
    frame["vol_expansion_risk_7d"] = (future_vol >= trailing_vol * 1.5).astype(np.float32)

    frame["future_max_return_7d"] = max_return_7d.clip(0.0, 2.0).astype(np.float32)
    frame["future_min_return_7d"] = min_return_7d.clip(-0.95, 0.0).astype(np.float32)
    frame["future_vol_7d"] = future_vol.clip(0.0, 0.25).astype(np.float32)
    frame["future_terminal_return_7d"] = terminal_return.clip(-0.95, 2.0).astype(np.float32)
    frame["trend_efficiency_7d"] = trend_efficiency.clip(-1.0, 1.0).astype(np.float32)
    frame["coin"] = coin
    return frame


def market_regime(frame: pd.DataFrame) -> pd.DataFrame:
    current_positive = frame.groupby("coin", sort=False)["close"].pct_change(24).gt(0)
    regime = current_positive.groupby(frame["timestamp"]).mean().rename("market_regime_strength")
    return frame.merge(regime, left_on="timestamp", right_index=True, how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("train_ready/features/final_causal_v1"))
    parser.add_argument("--output", type=Path, default=Path("train_ready/supervised_dataset_causal_v2.parquet"))
    args = parser.parse_args()
    files = sorted(args.feature_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No causal feature files in {args.feature_dir}")
    frames = []
    for index, path in enumerate(files, start=1):
        built = build_targets(pd.read_parquet(path), path.stem)
        frames.append(built)
        print(f"[{index:03d}/{len(files):03d}] {path.stem}: {len(built):,} source rows")
    dataset = pd.concat(frames, ignore_index=True)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
    dataset = market_regime(dataset)
    # Only tail rows lacking a complete maximum horizon are unusable.  Class
    # labels alone cannot identify them because comparisons with NaN are False.
    valid = dataset["future_max_return_7d"].notna()
    for target in CLS_TARGETS + REG_TARGETS:
        valid &= dataset[target].notna()
    # opportunity_30d needs the longest horizon explicitly.
    per_coin_last = dataset.groupby("coin")["timestamp"].transform("max")
    valid &= dataset["timestamp"] <= per_coin_last - pd.Timedelta(hours=MAX_HORIZON_HOURS)
    dataset = dataset.loc[valid].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    for column in dataset.select_dtypes(include=["float64"]).columns:
        dataset[column] = dataset[column].astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)
    print(f"[OK] {args.output}: rows={len(dataset):,} coins={dataset['coin'].nunique()} cols={len(dataset.columns)}")


if __name__ == "__main__":
    main()
