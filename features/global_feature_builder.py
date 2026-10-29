import os
import numpy as np
import pandas as pd

# ======================================================
# CONFIG
# ======================================================
SRC_DIR = "train_ready/good_ohlcv"
OUT_DIR = "train_ready/features/global"
OUT_PATH = os.path.join(OUT_DIR, "global_features.parquet")

WARMUP_HOURS = 2160
BTC_PROXY = "BTCUSDT"

GLOBAL_COINS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    #"BNBUSDT",
    #"ADAUSDT",
    #"XRPUSDT",
    #"DOTUSDT",
    #"AVAXUSDT",
    #"LINKUSDT",
    #"LTCUSDT",
]

# ======================================================
# UTILS
# ======================================================
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def safe_div(a, b):
    b = np.where(np.abs(b) > 1e-12, b, np.nan)
    return np.nan_to_num(a / b, nan=0.0, posinf=0.0, neginf=0.0)


def parse_ts(ts):
    return pd.to_datetime(ts, utc=True)


def load_coin(symbol):
    path = os.path.join(SRC_DIR, f"{symbol}.parquet")
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df["ts_dt"] = parse_ts(df["timestamp"])
    return df


# ======================================================
# FEATURE ENGINEERING (NO FILL HERE)
# ======================================================
def build_features(df, btc_df):
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    # ✅ NEW: fast signals
    ret_1h = close.pct_change(1)
    vol_1h = ret_1h.abs()   # better than std for 1h

    ret_24h = close.pct_change(24)
    ret_7d = close.pct_change(24 * 7)
    ret_30d = close.pct_change(24 * 30)

    vol_24h = ret_24h.rolling(24).std()
    vol_7d = ret_24h.rolling(24 * 7).std()

    usd_volume = close * volume
    vol_z = (usd_volume - usd_volume.rolling(24 * 7).mean()) / (
        usd_volume.rolling(24 * 7).std() + 1e-8
    )

    ret_1h_abs = close.pct_change(1).abs().clip(lower=1e-4)
    liquidity = np.log1p(safe_div(usd_volume, ret_1h_abs))

    #btc_close = btc_df[["ts_dt", "close"]].rename(columns={"close": "btc_close"})
    #merged = pd.merge(df[["ts_dt"]], btc_close, on="ts_dt", how="left")

    #rel = safe_div(close.to_numpy(), merged["btc_close"].to_numpy())

    feats = pd.DataFrame({
        "ts_dt": df["ts_dt"],
        # ✅ NEW (add these first)
        "return_1h": ret_1h,
        "realized_vol_1h": vol_1h,
        "return_24h": ret_24h,
        "return_7d": ret_7d,
        #"return_30d": ret_30d,
        "realized_vol_24h": vol_24h,
        "realized_vol_7d": vol_7d,
        "volume_zscore_7d": vol_z,
        "liquidity_ratio": liquidity,
        #"relative_to_btc": rel,
    })

    return feats


# ======================================================
# MAIN
# ======================================================
def main():
    ensure_dirs()

    print("[START] Clean Global Feature Builder")

    # -------------------------
    # LOAD BTC (MASTER CLOCK)
    # -------------------------
    btc_df = load_coin(BTC_PROXY)
    btc_df = btc_df.sort_values("ts_dt").reset_index(drop=True)

    master_ts = btc_df["ts_dt"]

    # -------------------------
    # BUILD BTC FEATURES (NO MASK)
    # -------------------------
    btc_feats = build_features(btc_df, btc_df)
    #btc_feats["relative_to_btc"] = 1.0

    btc_feats = btc_feats.set_index("ts_dt")
    btc_feats = btc_feats.reindex(master_ts)

    # 🔥 ONLY fill forward AFTER reindex
    btc_feats = btc_feats.ffill()

    # final safety
    btc_feats = btc_feats.fillna(0.0)

    frames = []
    frames.append(btc_feats.add_prefix("btcusdt_"))

    # -------------------------
    # OTHER COINS
    # -------------------------
    for coin in GLOBAL_COINS:
        if coin == BTC_PROXY:
            continue

        print(f"[INFO] {coin}")

        df = load_coin(coin)

        feats = build_features(df, btc_df)
        feats = feats.set_index("ts_dt")

        valid_mask = feats.notna().any(axis=1)

        feats = feats.reindex(master_ts)

        # 🔥 ALIGN MASK TO NEW INDEX
        valid_mask = valid_mask.reindex(master_ts, fill_value=False)

        # 🔥 NOW SAFE
        feats.loc[valid_mask] = feats.loc[valid_mask].ffill()
        feats.loc[~valid_mask] = 0.0

        feats = feats.fillna(0.0)

        frames.append(feats.add_prefix(coin.lower() + "_"))

    # -------------------------
    # MERGE ALL
    # -------------------------
    global_df = pd.concat(frames, axis=1)

    # drop warmup
    global_df = global_df.iloc[WARMUP_HOURS:].copy()

    global_df = global_df.reset_index()

    # final safety
    global_df = global_df.fillna(0.0)

    if not np.isfinite(global_df.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("Non-finite values detected")

    global_df.to_parquet(OUT_PATH, index=False)

    print(f"[OK] Saved → {OUT_PATH}")
    print(f"Rows: {len(global_df)} | Cols: {len(global_df.columns)}")


if __name__ == "__main__":
    main()
