"""Causal per-coin history-coverage context for RL observations."""

from __future__ import annotations

import numpy as np
import pandas as pd


COIN_HISTORY_CONTEXT_COLUMNS = [
    "coin_history_age_90d",
    "coin_history_log_age",
    "coin_history_has_90d",
]


def add_coin_history_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with causal age/coverage columns.

    The features depend only on the row's position in that coin's stored
    chronological history.  They do not use returns, future availability,
    cross-sectional ranks, or any information after the current timestamp.
    """
    if frame.empty:
        result = frame.copy()
        for column in COIN_HISTORY_CONTEXT_COLUMNS:
            result[column] = np.asarray([], dtype=np.float32)
        return result
    result = frame.sort_values("timestamp").drop_duplicates(
        "timestamp", keep="last"
    ).reset_index(drop=True)
    age_hours = np.arange(len(result), dtype=np.float32)
    result["coin_history_age_90d"] = np.clip(age_hours / (24.0 * 90.0), 0.0, 1.0)
    result["coin_history_log_age"] = (
        np.log1p(age_hours) / np.log1p(24.0 * 365.0 * 3.0)
    ).clip(0.0, 1.0).astype(np.float32)
    result["coin_history_has_90d"] = (age_hours >= 24.0 * 90.0).astype(np.float32)
    return result
