"""Build leakage-safe local + global features without touching legacy artifacts.

The legacy final builder standardized each complete coin history and used
backward filling.  Both operations let later observations influence earlier
rows.  This version uses only forward information and writes to a separate
artifact family so existing checkpoints remain reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = {"open", "high", "low", "close", "volume"}


def normalize_to_hour(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="s", utc=True, errors="raise").dt.floor("h")
    return pd.to_datetime(values, utc=True, errors="raise").dt.floor("h")


def causal_standardize(
    values: pd.Series,
    min_history: int = 168,
    clip: float = 10.0,
) -> pd.Series:
    """Standardize x[t] using moments from x[:t] only.

    Prefix invariance is the key contract: appending future rows cannot change
    any already-produced value.  Early rows without enough history are neutral
    zeros rather than backward-filled future information.
    """
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    past = clean.shift(1)
    mean = past.expanding(min_periods=min_history).mean()
    std = past.expanding(min_periods=min_history).std(ddof=0)
    out = (clean - mean) / std.where(std > 1e-8)
    return out.clip(-clip, clip).fillna(0.0).astype(np.float32)


def build_coin_frame(
    local: pd.DataFrame,
    global_indexed: pd.DataFrame,
    min_history: int = 168,
    clip: float = 10.0,
) -> pd.DataFrame:
    local = local.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    local["timestamp"] = normalize_to_hour(local["timestamp"])
    local = local.loc[local["timestamp"] >= global_indexed.index.min()].reset_index(drop=True)
    if local.empty:
        return local

    # Local engineered features should already be finite.  A neutral zero is
    # safer than carrying a stale observation across a missing local candle.
    numeric_columns = local.select_dtypes(include=[np.number]).columns
    local[numeric_columns] = (
        local[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    aligned_global = global_indexed.reindex(local["timestamp"])
    # Forward fill is causal.  Leading or still-missing context is neutral.
    aligned_global = aligned_global.ffill().fillna(0.0).reset_index(drop=True)
    merged = pd.concat([local.reset_index(drop=True), aligned_global], axis=1)

    for column in merged.columns:
        if column == "timestamp" or column in RAW_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(merged[column]):
            merged[column] = causal_standardize(merged[column], min_history=min_history, clip=clip)
    for column in RAW_COLUMNS.intersection(merged.columns):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).astype(np.float32)
    if merged.isna().any().any():
        raise RuntimeError(f"NaN columns after causal build: {merged.columns[merged.isna().any()].tolist()}")
    numeric = merged.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite value after causal build")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, default=Path("train_ready/features/local"))
    parser.add_argument("--global-path", type=Path, default=Path("train_ready/features/global/global_features.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("train_ready/features/final_causal_v1"))
    parser.add_argument("--min-history", type=int, default=168)
    parser.add_argument("--clip", type=float, default=10.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_frame = pd.read_parquet(args.global_path).sort_values("ts_dt")
    global_frame["ts_dt"] = normalize_to_hour(global_frame["ts_dt"])
    global_frame = global_frame.drop_duplicates("ts_dt", keep="last").set_index("ts_dt")
    files = sorted(args.local_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No local features in {args.local_dir}")
    for index, path in enumerate(files, start=1):
        frame = build_coin_frame(
            pd.read_parquet(path), global_frame, min_history=args.min_history, clip=args.clip
        )
        if frame.empty:
            print(f"[SKIP] {path.stem}: no global overlap")
            continue
        output = args.output_dir / path.name
        frame.to_parquet(output, index=False)
        print(f"[{index:03d}/{len(files):03d}] {path.stem}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
