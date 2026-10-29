import os
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
# ======================================================
# CONFIG
# ======================================================
# New pipeline: after sanity checker copies GOOD coins here
SRC_DIR = "train_ready/good_ohlcv"
OUT_DIR = "train_ready/features/local"

SECONDS_PER_HOUR = 3600.0
MAX_GAP_HOURS_CLIP = 12.0

# Warm-up removal (90 days)
MAX_WINDOW_HOURS = 2160  # 90 days warm-up at 1h candles

# ======================================================
# UTILS
# ======================================================
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def safe_div(a, b):
    # element-wise safe divide
    return np.where(b != 0, a / b, 0.0)


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return pd.Series(safe_div(series - mean, std), index=series.index)


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window).mean()


def parse_timestamp_series(ts: pd.Series) -> pd.Series:
    """
    Accepts:
      - pandas datetime64 (tz-aware or tz-naive)
      - unix seconds (int/float)
      - ISO datetime strings

    Returns:
      - pd.Series of UTC timestamps
    """
    # Already datetime (including tz-aware)
    if is_datetime64_any_dtype(ts):
        return pd.to_datetime(ts, utc=True)

    # Unix seconds
    if np.issubdtype(ts.dtype, np.number):
        return pd.to_datetime(ts, unit="s", utc=True)

    # Strings or mixed → try parse
    return pd.to_datetime(ts, utc=True, errors="coerce")

# ======================================================
# FEATURE BUILDERS
# ======================================================
def build_local_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic column checks (fail fast)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing columns: {missing_cols}")

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    feats = {}

    # ======================================================
    # GAP / OBSERVATION QUALITY (NEW LOGIC)
    # ======================================================
    ts_dt = parse_timestamp_series(df["timestamp"])
    if ts_dt.isna().any():
        bad_idx = np.where(ts_dt.isna().to_numpy())[0][:5]
        raise RuntimeError(f"Unparseable timestamps at rows {bad_idx.tolist()}")

    dt_hours = ts_dt.diff().dt.total_seconds() / SECONDS_PER_HOUR

    # In normal 1h candles, dt_hours ~= 1.0; gaps are dt_hours > 1.0
    # We clip time-since-observation to [1, 12] hours for stability.
    dt_hours = dt_hours.fillna(1.0)
    dt_hours_clipped = dt_hours.clip(lower=1.0, upper=MAX_GAP_HOURS_CLIP)

    feats["gap_affected"] = (dt_hours > 1.0).astype(float)
    feats["gap_affected"] = feats["gap_affected"].fillna(0.0)

    feats["time_since_last_observation_normalized"] = (dt_hours_clipped / MAX_GAP_HOURS_CLIP).astype(float)

    # ======================================================
    # RETURNS
    # ======================================================
    feats["return_1h"] = close.pct_change(1)
    feats["return_4h"] = close.pct_change(4)
    feats["return_24h"] = close.pct_change(24)
    feats["return_7d"] = close.pct_change(24 * 7)
    feats["return_14d"] = close.pct_change(24 * 14)
    feats["return_30d"] = close.pct_change(24 * 30)

    # ======================================================
    # ATR & VOLATILITY
    # ======================================================
    feats["atr_1h"] = atr(df, 1)
    feats["atr_24h"] = atr(df, 24)
    feats["atr_7d"] = atr(df, 24 * 7)
    feats["atr_14d"] = atr(df, 24 * 14)
    feats["atr_30d"] = atr(df, 24 * 30)
    feats["atr_90d"] = atr(df, 24 * 90)

    ret_1h = feats["return_1h"]
    feats["realized_vol_24h"] = ret_1h.rolling(24).std()
    feats["realized_vol_7d"] = ret_1h.rolling(24 * 7).std()
    feats["realized_vol_30d"] = ret_1h.rolling(24 * 30).std()
    feats["vol_regime_score"] = safe_div(
        feats["realized_vol_7d"], feats["realized_vol_30d"]
    )

    # ======================================================
    # RETURN NORMALIZED BY ATR (FIXED)
    # ======================================================
    # Convert ATR to % of price first
    atr_24h_pct = safe_div(feats["atr_24h"], close)
    atr_7d_pct = safe_div(feats["atr_7d"], close)
    atr_1h_pct = safe_div(feats["atr_1h"], close)

    # Prevent division explosion
    atr_24h_pct = np.clip(atr_24h_pct, 1e-4, None)
    atr_7d_pct = np.clip(atr_7d_pct, 1e-4, None)

    feats["return_24h_atr"] = safe_div(feats["return_24h"], atr_24h_pct)
    feats["return_7d_atr"] = safe_div(feats["return_7d"], atr_7d_pct)

    # ======================================================
    # PRICE POSITION
    # ======================================================
    for d in [7, 14, 30, 60, 90]:
        low_d = close.rolling(24 * d).min()
        high_d = close.rolling(24 * d).max()
        feats[f"price_position_{d}d"] = safe_div(
            close - low_d, high_d - low_d
        )

    feats["high_low_spread_14d"] = safe_div(
        df["high"].rolling(24 * 14).max() -
        df["low"].rolling(24 * 14).min(),
        close
    )

    feats["range_pressure_30d"] = safe_div(
        close - close.rolling(24 * 30).mean(),
        feats["atr_30d"]
    )

    # ======================================================
    # TREND STRUCTURE
    # ======================================================
    for h in [4, 24, 168, 720]:
        sma = close.rolling(h).mean()
        ema = close.ewm(span=h, adjust=False).mean()
        feats[f"sma_ratio_{h}h"] = safe_div(close, sma)
        feats[f"ema_ratio_{h}h"] = safe_div(close, ema)

    # ======================================================
    # VOLUME & LIQUIDITY (FIXED)
    # ======================================================
    feats["volume_normalized"] = safe_div(
        volume, volume.rolling(24 * 14).mean()
    )
    feats["volume_return_1h"] = volume.pct_change(1)
    feats["volume_return_24h"] = volume.pct_change(24)
    feats["volume_zscore_14d"] = zscore(volume, 24 * 14)

    usd_volume = close * volume
    feats["usd_volume"] = usd_volume
    feats["usd_volume_zscore_14d"] = zscore(usd_volume, 24 * 14)
    feats["usd_volume_zscore_30d"] = zscore(usd_volume, 24 * 30)

    feats["liquidity_ratio"] = np.log1p(
        safe_div(usd_volume, np.clip(feats["atr_1h"], 1e-6, None))
    )

    feats["impact_score"] = safe_div(feats["atr_1h"], close)

    # log compress huge values
    feats["dollar_volatility"] = np.log1p(
        feats["atr_1h"] * usd_volume
    )
    
    feats["dollar_volatility_zscore_14d"] = zscore(
        feats["dollar_volatility"], 24 * 14
    )

    feats["log_price"] = np.log10(close.replace(0, np.nan)).fillna(0.0)

    # ======================================================
    # CANDLE MICROSTRUCTURE
    # ======================================================
    body = (df["close"] - df["open"]).abs()
    wick = (df["high"] - df["low"]).replace(0, np.nan)

    feats["candle_strength"] = safe_div(body, wick)
    feats["micro_noise_ratio"] = safe_div(body, feats["atr_1h"])

    # ======================================================
    # TIME ENCODING
    # ======================================================
    feats["hour_of_day_normalized"] = ts_dt.dt.hour / 23.0
    feats["day_of_week_normalized"] = ts_dt.dt.dayofweek / 6.0
    feats["weekend_flag"] = (ts_dt.dt.dayofweek >= 5).astype(float)
    feats["month_of_year_normalized"] = (ts_dt.dt.month - 1) / 11.0

    out = pd.DataFrame(feats)

    # Final safety
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return out


# ======================================================
# MAIN
# ======================================================
def main():
    ensure_dirs()

    if not os.path.isdir(SRC_DIR):
        raise RuntimeError(
            f"SRC_DIR not found: {SRC_DIR}\n"
            f"Expected parquet files in: {SRC_DIR}"
        )

    coins = sorted([f.replace(".parquet", "") for f in os.listdir(SRC_DIR) if f.endswith(".parquet")])
    print(f"[START] Building local features for {len(coins)} coins")
    print(f"  SRC_DIR: {SRC_DIR}")
    print(f"  OUT_DIR: {OUT_DIR}\n")

    for pid in coins:
        path = os.path.join(SRC_DIR, f"{pid}.parquet")
        df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)

        # Build features
        feats = build_local_features(df)

        # Remove warm-up
        if len(feats) <= MAX_WINDOW_HOURS:
            print(f"[SKIP] {pid:<12} not enough data (rows={len(feats)})")
            continue

        feats = feats.iloc[MAX_WINDOW_HOURS:].reset_index(drop=True)
        df = df.iloc[MAX_WINDOW_HOURS:].reset_index(drop=True)

        # Assemble output (keep the 6 raw columns first)
        out = pd.concat(
            [
                df[["timestamp", "open", "high", "low", "close", "volume"]],
                feats,
            ],
            axis=1,
        )

        # Final safety checks
        if out.isna().any().any():
            bad_cols = out.columns[out.isna().any()].tolist()
            raise RuntimeError(f"NaN detected in {pid}. Bad columns: {bad_cols}")

        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        out.to_parquet(os.path.join(OUT_DIR, f"{pid}.parquet"), index=False)
        print(f"[OK] {pid:<12} | rows={len(out):6d} | features={out.shape[1]-6}")

    print("\n[COMPLETE] Local feature build finished")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

