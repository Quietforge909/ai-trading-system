#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

# ==========================================================
# CONFIG
# ==========================================================
FEATURE_DIR = Path("train_ready/features/final")
OUT_DIR = Path("train_ready/features/final_with_supervised")
from datetime import datetime

current_year = datetime.utcnow().year
model_year = current_year - 1

MODEL_DIR = Path(f"models/supervised/live_train_through_{model_year}")
FEATURE_LIST_PATH = Path("models/supervised/feature_list.json")

CLS_TARGETS = ["hit_7d", "hit_14d", "hit_30d"]
REG_TARGETS = ["future_vol_7d"]

# ==========================================================
# LOAD MODELS
# ==========================================================

def load_models():
    models = {}

    for tgt in CLS_TARGETS:
        path = MODEL_DIR / f"cls_{tgt}.cbm"
        if not path.exists():
            raise RuntimeError(f"Missing model: {path}")
        m = CatBoostClassifier()
        m.load_model(str(path))
        models[tgt] = m

    for tgt in REG_TARGETS:
        path = MODEL_DIR / f"reg_{tgt}.cbm"
        if not path.exists():
            raise RuntimeError(f"Missing model: {path}")
        m = CatBoostRegressor()
        m.load_model(str(path))
        models[tgt] = m

    return models


# ==========================================================
# MAIN
# ==========================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load feature list from training
    if not FEATURE_LIST_PATH.exists():
        raise RuntimeError("Missing feature_list.json")

    with open(FEATURE_LIST_PATH, "r") as f:
        feature_cols = json.load(f)

    print(f"[INFO] Using {len(feature_cols)} features")

    models = load_models()
    print("[INFO] Models loaded")

    files = sorted(FEATURE_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError("No feature files found")

    print(f"[INFO] Found {len(files)} coins")

    # ==========================================================
    # 1️⃣ Load & stack ALL coins first
    # ==========================================================
    stacked = []

    for f in files:
        coin = f.stem
        df = pd.read_parquet(f)

        df["coin"] = coin
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        stacked.append(df)

    df_all = pd.concat(stacked, ignore_index=True)

    # ==========================================================
    # 2️⃣ Rebuild training-only features
    # ==========================================================

    # Add year (training used this)
    df_all["year"] = df_all["timestamp"].dt.year.astype(np.int32)

    # Rebuild market_regime_strength exactly like training
    # 24h return per coin
    df_all["ret_24h_tmp"] = (
        df_all.groupby("coin")["close"]
        .pct_change(24)
    )

    regime = (
        df_all.groupby("timestamp")["ret_24h_tmp"]
        .apply(lambda x: np.mean(x > 0))
        .rename("market_regime_strength")
        .reset_index()
    )

    df_all = df_all.merge(regime, on="timestamp", how="left")

    df_all.drop(columns=["ret_24h_tmp"], inplace=True)

    # ==========================================================
    # 3️⃣ Ensure feature parity
    # ==========================================================

    missing = [c for c in feature_cols if c not in df_all.columns]
    if missing:
        raise RuntimeError(f"Missing required features: {missing}")

    X = df_all[feature_cols]

    # ==========================================================
    # 4️⃣ Model predictions
    # ==========================================================

    print("[INFO] Running model predictions")

    for tgt in CLS_TARGETS:
        prob = models[tgt].predict_proba(X)[:, 1]
        df_all[f"sup_{tgt}_prob"] = prob.astype(np.float32)

    for tgt in REG_TARGETS:
        pred = models[tgt].predict(X)
        df_all[f"sup_{tgt}"] = pred.astype(np.float32)

    # ==========================================================
    # 5️⃣ Cross-sectional percentile per timestamp
    # ==========================================================

    print("[INFO] Computing cross-sectional percentiles")

    for tgt in ["hit_7d", "hit_14d", "hit_30d"]:
        prob_col = f"sup_{tgt}_prob"
        pct_col = f"sup_{tgt}_pct"

        df_all[pct_col] = (
            df_all
            .groupby("timestamp")[prob_col]
            .rank(pct=True)
            .astype(np.float32)
        )

    # ==========================================================
    # 6️⃣ Save per-coin merged files
    # ==========================================================

    print("[INFO] Saving merged feature files")

    for coin, sub in df_all.groupby("coin"):
        out_path = OUT_DIR / f"{coin}.parquet"

        sub = sub.drop(columns=["coin"])

        sub.to_parquet(out_path, index=False)

    print(f"[OK] Final dataset with supervised signals saved to:")
    print(f"     {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

