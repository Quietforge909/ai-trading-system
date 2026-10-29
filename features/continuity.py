"""Timestamp-continuity helpers shared by causal-v3 builders."""

from __future__ import annotations

import numpy as np
import pandas as pd


ONE_HOUR = pd.Timedelta(hours=1)


def normalized_timestamps(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="s", utc=True, errors="raise").dt.floor("h")
    return pd.to_datetime(values, utc=True, errors="raise").dt.floor("h")


def prepare_hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort/deduplicate observed candles without fabricating missing rows."""
    result = frame.copy()
    result["timestamp"] = normalized_timestamps(result["timestamp"])
    return (
        result.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def segment_ids(timestamps: pd.Series) -> pd.Series:
    timestamps = normalized_timestamps(timestamps)
    boundary = timestamps.diff().ne(ONE_HOUR)
    if len(boundary):
        boundary.iloc[0] = True
    return boundary.cumsum().astype(np.int32) - 1


def contiguous_segments(frame: pd.DataFrame):
    prepared = prepare_hourly_frame(frame)
    ids = segment_ids(prepared["timestamp"])
    for _, segment in prepared.groupby(ids, sort=True):
        yield segment.reset_index(drop=True)


def exact_future_locations(timestamps: pd.Series, hours: int) -> np.ndarray:
    """Return the row at exactly t+hours, or -1 when the horizon has a gap."""
    values = normalized_timestamps(timestamps)
    index = pd.Index(values)
    return index.get_indexer(values + pd.Timedelta(hours=hours))


def continuity_report(frame: pd.DataFrame) -> dict:
    prepared = prepare_hourly_frame(frame)
    delta = prepared["timestamp"].diff().dt.total_seconds().div(3600.0)
    lengths = [len(segment) for segment in contiguous_segments(prepared)]
    return {
        "rows": int(len(prepared)),
        "start": prepared["timestamp"].min().isoformat() if len(prepared) else None,
        "end": prepared["timestamp"].max().isoformat() if len(prepared) else None,
        "gaps": int(delta.gt(1.0).sum()),
        "largest_gap_hours": float(delta.max()) if len(prepared) > 1 else 0.0,
        "segments": len(lengths),
        "shortest_segment_rows": min(lengths) if lengths else 0,
        "longest_segment_rows": max(lengths) if lengths else 0,
    }
