#!/usr/bin/env python3
"""Single-window full-universe Kraken replay for recent 720-row fetches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rl.evaluation.evaluate_kraken_realistic_v280 import (
    build_corrected_rules_by_symbol,
    load_frames,
    load_pair_rules,
    run_ai_episode,
    summarize_episode_pnls,
)
from rl.evaluation.evaluate_random_v7 import load_checkpoint


def aligned_recent_window(
    frames: dict[str, pd.DataFrame],
    rows: int,
    min_visible_coins: int,
) -> dict[str, pd.DataFrame]:
    reference = "BTCUSDT" if "BTCUSDT" in frames else max(frames, key=lambda coin: len(frames[coin]))
    timeline = pd.Index(frames[reference]["timestamp"].tail(rows))
    if len(timeline) < rows:
        raise RuntimeError(f"Reference {reference} has only {len(timeline)} rows")
    if not pd.Series(timeline).diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise RuntimeError(f"Reference {reference} latest window is not hourly-contiguous")

    aligned = {}
    for coin, frame in sorted(frames.items()):
        indexed = frame.set_index("timestamp", drop=False)
        loc = indexed.reindex(timeline)
        if loc["timestamp"].isna().any():
            continue
        aligned[coin] = loc.reset_index(drop=True)
    if len(aligned) < min_visible_coins:
        raise RuntimeError(f"visible={len(aligned)} < {min_visible_coins}")
    return aligned


def summarize_full_window(record: dict[str, Any]) -> dict[str, Any]:
    trades = record["trades"]
    pnl = np.asarray([trade["pnl"] for trade in trades], dtype=np.float64)
    returns = np.asarray([trade["return"] for trade in trades], dtype=np.float64)
    gross_profit = float(pnl[pnl > 0].sum()) if pnl.size else 0.0
    gross_loss = float(-pnl[pnl < 0].sum()) if pnl.size else 0.0
    return {
        "start": record["start"],
        "end": record["end"],
        "coins": record["coins"],
        "rows": int(record.get("rows", 0)),
        "pnl": record["pnl"],
        "return": record["return"],
        "max_drawdown": record["max_drawdown"],
        "trades": len(trades),
        "manual_sells": record["manual_sells"],
        "forced_sells": record["forced_sells"],
        "forced_sell_ratio": record["forced_sells"] / max(1, len(trades)),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "win_rate": float((pnl > 0).mean()) if pnl.size else 0.0,
        "mean_trade_return": float(returns.mean()) if returns.size else 0.0,
        "median_trade_return": float(np.median(returns)) if returns.size else 0.0,
        "mean_exposure": record["mean_exposure"],
        "max_exposure": record["max_exposure"],
        "actions": record["actions"],
        "rejected": record["rejected"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=720)
    parser.add_argument("--min-visible-coins", type=int, default=600)
    parser.add_argument("--side-cost-bps", type=float, default=15.0)
    parser.add_argument(
        "--buy-prob-threshold",
        type=float,
        default=0.0,
        help="Evaluation-only gate: selected BUY actions below this probability become no-op.",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy-seed", type=int, default=123)
    args = parser.parse_args()

    if args.rows <= 1:
        parser.error("--rows must be greater than 1")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.buy_prob_threshold < 0.0 or args.buy_prob_threshold > 1.0:
        parser.error("--buy-prob-threshold must be between 0 and 1")
    np.random.seed(args.policy_seed)
    torch.manual_seed(args.policy_seed)

    source_rules = load_pair_rules(args.manifest)
    corrected_rules = build_corrected_rules_by_symbol(source_rules)
    frames = load_frames(args.data_dir, source_rules)
    aligned = aligned_recent_window(frames, args.rows, args.min_visible_coins)
    corrected_rules = {coin: corrected_rules[coin] for coin in aligned}

    device = torch.device("cpu")
    model, config, payload = load_checkpoint(args.checkpoint, device)
    fee_rate = float(config["fee_rate"])
    max_slots = int(config.get("max_slots", 1))
    record = run_ai_episode(
        model,
        config,
        aligned,
        corrected_rules,
        device,
        fee_rate,
        max_slots,
        args.side_cost_bps,
        0,
        not args.stochastic,
        args.temperature,
        args.buy_prob_threshold,
    )
    record["rows"] = len(next(iter(aligned.values())))

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_episode": int(payload["episode"]),
        "data_dir": str(args.data_dir),
        "manifest": str(args.manifest),
        "policy": {
            "stochastic": bool(args.stochastic),
            "temperature": float(args.temperature),
            "seed": int(args.policy_seed),
        },
        "realism": {
            "side_cost_bps": float(args.side_cost_bps),
            "fee_rate": fee_rate,
            "kraken_min_size_precision_costmin": True,
            "execution": "observe candle t, fill at next candle open; terminal liquidation at final close",
            "buy_prob_threshold": float(args.buy_prob_threshold),
        },
        "requested_rows": int(args.rows),
        "used_rows": int(record["rows"]),
        "available_full_window_coins": len(aligned),
        "summary": summarize_full_window(record),
        "concentration": summarize_episode_pnls([record]),
        "trades": record["trades"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "summary": result["summary"],
        "top_coin": result["concentration"]["top_coin"],
        "top_coin_pnl_share": result["concentration"]["top_coin_pnl_share"],
        "unique_traded_coins": result["concentration"]["unique_traded_coins"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
