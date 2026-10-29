#!/usr/bin/env python3
"""Fetch all available Kraken spot USD/USDT OHLCV history.

Kraken returns at most 720 OHLC rows per request, so this fetcher pages with the
public `last` cursor until it reaches the requested end time. Output filenames
use the project-compatible `BASEUSDT.parquet` convention even when the source
pair is quoted in USD. The original Kraken pair and quote are stored in the
manifest.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_BASE = "https://api.kraken.com/0/public"

ASSET_MAP = {
    "XBT": "BTC",
    "XXBT": "BTC",
    "XETH": "ETH",
    "XXRP": "XRP",
    "XLTC": "LTC",
    "XADA": "ADA",
    "XDOT": "DOT",
    "XXLM": "XLM",
    "XTRX": "TRX",
    "XETC": "ETC",
    "XXMR": "XMR",
    "XZEC": "ZEC",
    "ZUSD": "USD",
    "USDT": "USDT",
}

FIAT_BASES = {
    "AUD",
    "CAD",
    "CHF",
    "EUR",
    "GBP",
    "JPY",
    "USD",
}


@dataclass(frozen=True)
class KrakenMarket:
    symbol: str
    pair_id: str
    wsname: str
    base: str
    quote: str
    priority: int


def normalize_asset(value: str) -> str:
    value = (value or "").replace(".d", "").upper()
    return ASSET_MAP.get(value, value)


def request_json(session: requests.Session, endpoint: str, params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    url = f"{API_BASE}/{endpoint}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            errors = payload.get("error") or []
            if errors:
                raise RuntimeError(f"Kraken API error: {errors}")
            return payload["result"]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(10.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"request failed after {retries} retries: {last_error}")


def discover_markets() -> list[KrakenMarket]:
    with requests.Session() as session:
        result = request_json(session, "AssetPairs", {})

    by_symbol: dict[str, KrakenMarket] = {}
    for pair_id, info in result.items():
        status = str(info.get("status", "online")).lower()
        if status != "online":
            continue

        wsname = info.get("wsname") or info.get("altname") or pair_id
        if "/" in wsname:
            raw_base, raw_quote = wsname.split("/", 1)
        else:
            raw_base = info.get("base", "")
            raw_quote = info.get("quote", "")

        base = normalize_asset(raw_base)
        quote = normalize_asset(raw_quote)
        if quote not in {"USD", "USDT"}:
            continue
        if not base or base in FIAT_BASES:
            continue
        if any(ch in base for ch in {"/", ":", " "}) or len(base) > 16:
            continue

        symbol = f"{base}USDT"
        priority = 0 if quote == "USD" else 1
        market = KrakenMarket(
            symbol=symbol,
            pair_id=pair_id,
            wsname=wsname,
            base=base,
            quote=quote,
            priority=priority,
        )
        existing = by_symbol.get(symbol)
        if existing is None or market.priority < existing.priority:
            by_symbol[symbol] = market

    return sorted(by_symbol.values(), key=lambda market: market.symbol)


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df


def fetch_market(
    market: KrakenMarket,
    output_dir: Path,
    start_ts: int,
    end_ts: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    output_path = output_dir / f"{market.symbol}.parquet"
    existing = load_existing(output_path)
    since = start_ts
    if not existing.empty:
        max_ts = int(existing["timestamp"].max().timestamp())
        since = max(max_ts + 3600, start_ts)

    rows: list[list[Any]] = []
    requests_made = 0
    last_cursor = since

    with requests.Session() as session:
        while since <= end_ts:
            result = request_json(
                session,
                "OHLC",
                {"pair": market.pair_id, "interval": 60, "since": since},
            )
            batch = result.get(market.pair_id)
            if batch is None:
                batch = next((value for key, value in result.items() if key != "last"), [])
            requests_made += 1

            for row in batch:
                row_ts = int(row[0])
                if start_ts <= row_ts <= end_ts:
                    rows.append(row)

            next_cursor = int(result.get("last", since))
            if not batch or next_cursor <= last_cursor:
                break
            if int(batch[-1][0]) >= end_ts:
                break
            last_cursor = next_cursor
            since = next_cursor
            time.sleep(sleep_seconds)

    new_df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    if not new_df.empty:
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"].astype("int64"), unit="s", utc=True).dt.tz_localize(None)
        for column in ["open", "high", "low", "close", "volume"]:
            new_df[column] = pd.to_numeric(new_df[column], errors="coerce")
        new_df["usd_volume"] = new_df["close"] * new_df["volume"]
        new_df = new_df[["timestamp", "open", "high", "low", "close", "volume", "usd_volume"]]

    combined = pd.concat([existing, new_df], ignore_index=True)
    if not combined.empty:
        combined = combined.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        combined.to_parquet(output_path, index=False)

    return {
        **asdict(market),
        "rows": int(len(combined)),
        "new_rows": int(len(new_df)),
        "requests": int(requests_made),
        "start": str(combined["timestamp"].min()) if not combined.empty else None,
        "end": str(combined["timestamp"].max()) if not combined.empty else None,
        "path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/kraken/ohlcv_1h_all_20260716"))
    parser.add_argument("--start", default="2017-01-01 00:00:00")
    parser.add_argument("--end", default=None, help="UTC timestamp. Defaults to current UTC minus 2 hours.")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N market limit for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_ts = int(pd.Timestamp(args.start, tz="UTC").timestamp())
    if args.end:
        end_ts = int(pd.Timestamp(args.end, tz="UTC").timestamp())
    else:
        end_ts = int((pd.Timestamp.utcnow().floor("h") - pd.Timedelta(hours=2)).timestamp())

    markets = discover_markets()
    if args.limit > 0:
        markets = markets[: args.limit]
    if not any(market.symbol == "BTCUSDT" for market in markets):
        raise RuntimeError("BTCUSDT-compatible Kraken market was not discovered")
    if not any(market.symbol == "ETHUSDT" for market in markets):
        raise RuntimeError("ETHUSDT-compatible Kraken market was not discovered")
    if not any(market.symbol == "SOLUSDT" for market in markets):
        raise RuntimeError("SOLUSDT-compatible Kraken market was not discovered")

    print(f"Discovered {len(markets)} Kraken USD/USDT spot markets")
    print(f"Fetching {pd.to_datetime(start_ts, unit='s', utc=True)} -> {pd.to_datetime(end_ts, unit='s', utc=True)}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(fetch_market, market, args.output_dir, start_ts, end_ts, args.sleep_seconds)
            for market in markets
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{index:04d}/{len(markets):04d}] {result['symbol']} "
                f"rows={result['rows']} new={result['new_rows']} requests={result['requests']} "
                f"source={result['wsname']}"
            )

    manifest = {
        "source": "kraken",
        "interval": "1h",
        "quote_policy": "prefer USD, fallback USDT; output symbols normalized as BASEUSDT",
        "start": args.start,
        "end": str(pd.to_datetime(end_ts, unit="s", utc=True)),
        "market_count": len(results),
        "markets": sorted(results, key=lambda row: row["symbol"]),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
