from __future__ import annotations

import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from fetchers.fetch_kraken_full_history import discover_markets

@dataclass(frozen=True)
class PipelineOutput:
    raw_dir: Path
    manifest: Path
    feature_dir: Path
    scored_dir: Path
    as_of: datetime


class PipelineCommandError(RuntimeError):
    pass


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_INTERVAL = "1h"
BINANCE_LIMIT = 1000


def run_command(args: list[str]) -> None:
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode != 0:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        detail = stderr or stdout or "No stdout/stderr captured."
        raise PipelineCommandError(
            "Pipeline command failed:\n"
            f"command: {' '.join(args)}\n"
            f"exit_code: {result.returncode}\n"
            f"output:\n{detail[-6000:]}"
        )


def merge_upside7d_scores(base_dir: Path, upside_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    added_columns = ["sup_future_max_return_7d", "sup_future_max_return_7d_pct"]
    merged_count = 0
    for base_path in sorted(base_dir.glob("*.parquet")):
        if base_path.name == "global.parquet":
            continue
        upside_path = upside_dir / base_path.name
        base = pd.read_parquet(base_path)
        if upside_path.exists():
            upside = pd.read_parquet(upside_path)
            missing = [column for column in ["timestamp", *added_columns] if column not in upside.columns]
            if missing:
                raise RuntimeError(f"{upside_path} missing columns: {missing}")
            base = base.merge(
                upside[["timestamp", *added_columns]],
                on="timestamp",
                how="left",
            )
            base["sup_future_max_return_7d"] = base["sup_future_max_return_7d"].fillna(0.0)
            base["sup_future_max_return_7d_pct"] = base["sup_future_max_return_7d_pct"].fillna(0.5)
        else:
            base["sup_future_max_return_7d"] = 0.0
            base["sup_future_max_return_7d_pct"] = 0.5
        base.to_parquet(output_dir / base_path.name, index=False)
        merged_count += 1

    manifest = {
        "source_feature_dir": str(base_dir),
        "upside_feature_dir": str(upside_dir),
        "output_dir": str(output_dir),
        "coins": merged_count,
        "added_columns": added_columns,
        "missing_upside_files_filled_neutral": sorted(
            path.name for path in base_dir.glob("*.parquet")
            if path.name != "global.parquet" and not (upside_dir / path.name).exists()
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _fetch_binance_symbol(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": BINANCE_INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": BINANCE_LIMIT,
    }
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()
            rows = response.json()
            if not rows:
                return pd.DataFrame()
            frame = pd.DataFrame(
                rows,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_base_volume",
                    "taker_quote_volume",
                    "ignore",
                ],
            )
            frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
            frame = frame[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]].rename(
                columns={"quote_volume": "usd_volume"}
            )
            for column in ["open", "high", "low", "close", "volume", "usd_volume"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            return frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"{symbol} Binance fetch failed after retries: {last_error}")


def fetch_live_binance_window(raw_dir: Path, start: datetime, end: datetime, coinlist_path: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    bases = json.loads(coinlist_path.read_text())
    symbols = [f"{base}USDT" for base in bases]
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    results: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_fetch_binance_symbol, symbol, start_ms, end_ms): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                frame = future.result()
                if frame.empty:
                    failures[symbol] = "empty"
                    continue
                output_path = raw_dir / f"{symbol}.parquet"
                frame.to_parquet(output_path, index=False)
                results.append({
                    "symbol": symbol,
                    "rows": int(len(frame)),
                    "start": frame["timestamp"].min().isoformat(),
                    "end": frame["timestamp"].max().isoformat(),
                    "path": str(output_path),
                })
            except Exception as exc:
                failures[symbol] = str(exc)
    manifest = {
        "source": "binance",
        "interval": "1h",
        "quote": "USDT",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "market_count": len(results),
        "markets": sorted(results, key=lambda row: row["symbol"]),
        "failures": failures,
    }
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not {"BTCUSDT", "ETHUSDT", "SOLUSDT"}.issubset({row["symbol"] for row in results}):
        raise RuntimeError("Live Binance fetch missing one or more context coins: BTCUSDT, ETHUSDT, SOLUSDT")
    return manifest_path


def build_kraken_execution_manifest(work_root: Path) -> Path:
    manifest_dir = work_root / "kraken_execution"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    markets = discover_markets()
    manifest = {
        "source": "kraken",
        "interval": "execution_only",
        "quote_policy": "prefer USD, fallback USDT; output symbols normalized as BASEUSDT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "market_count": len(markets),
        "markets": [
            {
                "symbol": market.symbol,
                "pair_id": market.pair_id,
                "wsname": market.wsname,
                "base": market.base,
                "quote": market.quote,
            }
            for market in markets
        ],
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def fetch_live_kraken_window(raw_dir: Path, start: datetime, end: datetime) -> Path:
    run_command([
        "python3", "-m", "fetchers.fetch_kraken_full_history",
        "--output-dir", str(raw_dir),
        "--start", start.strftime("%Y-%m-%d %H:%M:%S"),
        "--end", end.strftime("%Y-%m-%d %H:%M:%S"),
        "--max-workers", "3",
        "--sleep-seconds", "0.35",
    ])
    return raw_dir / "manifest.json"


def build_live_kraken_dataset(work_root: Path, lookback_days: int = 31) -> PipelineOutput:
    now = datetime.now(timezone.utc)
    end = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    start = end - timedelta(days=lookback_days)
    stamp = f"{end.strftime('%Y%m%d_%H%M%S')}_run_{now.strftime('%Y%m%d_%H%M%S_%f')}"

    raw_dir = work_root / "kraken_ohlcv_1h_live_all"
    local_dir = work_root / f"kraken_features_local_{stamp}"
    global_path = work_root / f"kraken_features_global_{stamp}" / "global.parquet"
    feature_dir = work_root / f"kraken_features_final_{stamp}"
    scored_dir = work_root / f"kraken_features_scored_{stamp}"
    upside_dir = work_root / f"kraken_features_upside7d_{stamp}"
    scored_plus_upside_dir = work_root / f"kraken_features_scored_plus_upside7d_{stamp}"

    manifest_path = fetch_live_kraken_window(raw_dir, start, end)
    run_command([
        "python3", "-m", "features.pipeline_v3_recent",
        "--raw-dir", str(raw_dir),
        "--local-dir", str(local_dir),
        "--global-path", str(global_path),
        "--output-dir", str(feature_dir),
        "--min-history", "24",
        "--clip", "10.0",
    ])
    run_command([
        "python3", "-m", "supervised.inference_v2",
        "--feature-dir", str(feature_dir),
        "--models-root", "models/supervised_causal_v3_stable6",
        "--output-dir", str(scored_dir),
        "--targets",
        "opportunity_7d,drawdown_risk_7d,vol_expansion_risk_7d,"
        "profitable_after_fee_72h,future_min_return_7d,future_vol_7d",
        "--first-validation-year", "2026",
        "--last-validation-year", "2026",
        "--spec-version", "3",
    ])
    run_command([
        "python3", "-m", "supervised.inference_v2",
        "--feature-dir", str(feature_dir),
        "--models-root", "models/supervised_causal_v3_upside7d_only",
        "--output-dir", str(upside_dir),
        "--targets", "future_max_return_7d",
        "--first-validation-year", "2026",
        "--last-validation-year", "2026",
        "--spec-version", "3",
    ])
    merge_upside7d_scores(scored_dir, upside_dir, scored_plus_upside_dir)
    return PipelineOutput(
        raw_dir=raw_dir,
        manifest=manifest_path,
        feature_dir=feature_dir,
        scored_dir=scored_plus_upside_dir,
        as_of=end,
    )


def build_live_binance_dataset(work_root: Path, lookback_days: int = 31) -> PipelineOutput:
    now = datetime.now(timezone.utc)
    end = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start = end - timedelta(days=lookback_days)
    stamp = f"{end.strftime('%Y%m%d_%H%M%S')}_run_{now.strftime('%Y%m%d_%H%M%S_%f')}"

    raw_dir = work_root / f"binance_ohlcv_1h_{stamp}"
    local_dir = work_root / f"features_local_{stamp}"
    global_path = work_root / f"features_global_{stamp}" / "global.parquet"
    feature_dir = work_root / f"features_final_{stamp}"
    scored_dir = work_root / f"features_scored_{stamp}"
    upside_dir = work_root / f"features_upside7d_{stamp}"
    scored_plus_upside_dir = work_root / f"features_scored_plus_upside7d_{stamp}"

    manifest_path = fetch_live_binance_window(raw_dir, start, end, Path("data/universe/main_coinlist.json"))
    run_command([
        "python3", "-m", "features.pipeline_v3_recent",
        "--raw-dir", str(raw_dir),
        "--local-dir", str(local_dir),
        "--global-path", str(global_path),
        "--output-dir", str(feature_dir),
        "--min-history", "24",
        "--clip", "10.0",
    ])
    run_command([
        "python3", "-m", "supervised.inference_v2",
        "--feature-dir", str(feature_dir),
        "--models-root", "models/supervised_causal_v3_stable6",
        "--output-dir", str(scored_dir),
        "--targets",
        "opportunity_7d,drawdown_risk_7d,vol_expansion_risk_7d,"
        "profitable_after_fee_72h,future_min_return_7d,future_vol_7d",
        "--first-validation-year", "2026",
        "--last-validation-year", "2026",
        "--spec-version", "3",
    ])
    run_command([
        "python3", "-m", "supervised.inference_v2",
        "--feature-dir", str(feature_dir),
        "--models-root", "models/supervised_causal_v3_upside7d_only",
        "--output-dir", str(upside_dir),
        "--targets", "future_max_return_7d",
        "--first-validation-year", "2026",
        "--last-validation-year", "2026",
        "--spec-version", "3",
    ])
    merge_upside7d_scores(scored_dir, upside_dir, scored_plus_upside_dir)
    return PipelineOutput(
        raw_dir=raw_dir,
        manifest=manifest_path,
        feature_dir=feature_dir,
        scored_dir=scored_plus_upside_dir,
        as_of=end,
    )
