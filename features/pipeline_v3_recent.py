"""Build causal v3-compatible features for short recent windows.

This is for data sources like Kraken REST OHLC where only the latest ~720 hourly
bars are available. It keeps the same feature schema as pipeline_v3, but removes
the 2,160-hour warm-up drop so the recent window remains testable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features.causal_final_feature import build_coin_frame
from features.continuity import continuity_report, contiguous_segments, prepare_hourly_frame
from features.global_feature_builder import build_features as build_global_features
from features.main_feature_builder import MAX_GAP_HOURS_CLIP, build_local_features


CONTEXT_COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gap_safe_local_recent(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_hourly_frame(frame)
    pieces = []
    for segment in contiguous_segments(prepared):
        feature = build_local_features(segment)
        pieces.append(
            pd.concat(
                [
                    segment[["timestamp", "open", "high", "low", "close", "volume"]],
                    feature,
                ],
                axis=1,
            )
        )
    result = pd.concat(pieces, ignore_index=True)
    delta = result["timestamp"].diff().dt.total_seconds().div(3600.0).fillna(1.0)
    result["gap_affected"] = delta.gt(1.0).astype(np.float32)
    result["time_since_last_observation_normalized"] = (
        delta.clip(lower=1.0, upper=MAX_GAP_HOURS_CLIP) / MAX_GAP_HOURS_CLIP
    ).astype(np.float32)
    return result.reset_index(drop=True)


def gap_safe_context_recent(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for segment in contiguous_segments(frame):
        segment = segment.copy()
        segment["ts_dt"] = segment["timestamp"]
        pieces.append(build_global_features(segment, segment))
    return pd.concat(pieces, ignore_index=True).sort_values("ts_dt")


def build_global_recent(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    contexts = {coin: gap_safe_context_recent(raw[coin]).set_index("ts_dt") for coin in CONTEXT_COINS}
    master = contexts["BTCUSDT"].index
    pieces = []
    for coin in CONTEXT_COINS:
        aligned = contexts[coin].reindex(master).fillna(0.0)
        pieces.append(aligned.add_prefix(coin.lower() + "_"))
    return pd.concat(pieces, axis=1).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--global-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-history", type=int, default=24)
    parser.add_argument("--clip", type=float, default=10.0)
    args = parser.parse_args()

    paths = sorted(args.raw_dir.glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No stored OHLCV files in {args.raw_dir}")
    raw = {path.stem: prepare_hourly_frame(pd.read_parquet(path)) for path in paths}
    missing_context = set(CONTEXT_COINS) - set(raw)
    if missing_context:
        raise RuntimeError(f"Missing stored context coins: {sorted(missing_context)}")
    for directory in (args.local_dir, args.global_path.parent, args.output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    global_frame = build_global_recent(raw)
    global_frame.to_parquet(args.global_path, index=False)
    global_indexed = global_frame.sort_values("ts_dt").drop_duplicates("ts_dt", keep="last").set_index("ts_dt")

    coin_reports = {}
    schema = None
    skipped = {}
    for number, path in enumerate(paths, start=1):
        local = gap_safe_local_recent(raw[path.stem])
        if local.empty:
            skipped[path.stem] = "empty local frame"
            continue
        local_path = args.local_dir / path.name
        local.to_parquet(local_path, index=False)
        final = build_coin_frame(local, global_indexed, min_history=args.min_history, clip=args.clip)
        if final.empty:
            skipped[path.stem] = "no global overlap"
            continue
        final_path = args.output_dir / path.name
        final.to_parquet(final_path, index=False)
        current_schema = list(final.columns)
        if schema is None:
            schema = current_schema
        elif schema != current_schema:
            raise RuntimeError(f"Feature schema mismatch for {path.stem}")
        coin_reports[path.stem] = continuity_report(raw[path.stem]) | {
            "local_rows": len(local),
            "final_rows": len(final),
            "source_sha256": sha256(path),
            "output_sha256": sha256(final_path),
        }
        print(f"[{number:03d}/{len(paths):03d}] {path.stem}: {len(final):,} rows")

    manifest = {
        "spec_version": 3,
        "recent_only": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "causal_contract": "past-only normalization; rolling features reset at hourly gaps; no candle synthesis; no long-history warm-up drop",
        "raw_dir": str(args.raw_dir),
        "local_dir": str(args.local_dir),
        "global_path": str(args.global_path),
        "output_dir": str(args.output_dir),
        "coins": len(coin_reports),
        "rows": sum(v["final_rows"] for v in coin_reports.values()),
        "feature_columns": schema,
        "coin_reports": coin_reports,
        "skipped": skipped,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {args.output_dir}: coins={manifest['coins']} rows={manifest['rows']:,} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
