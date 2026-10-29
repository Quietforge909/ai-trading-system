"""Random-window v7 evaluation for Phase 2 style checks.

Phase 1 promotion remains fixed full-universe/yearly evaluation through
``evaluate_phase1_v7`` and ``yearly_phase1_matrix``.  This module exists for
the later Phase 2 requirement: random aligned multi-coin selections across the
stored 2021-2026 history.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rl.env.coin_history_context import add_coin_history_context
from rl.env.multi_env import MultiCoinTradingEnv
from rl.evaluation.evaluate_v7 import (
    configured_allocation_sizing_mode,
    configured_buy_allocations,
    configured_position_dynamics_mode,
    load_checkpoint,
)


def load_frames(
    data_dir: Path,
    start: str,
    end: str,
    coin_history_context: bool,
) -> dict[str, pd.DataFrame]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    frames = {}
    for path in sorted(data_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if coin_history_context:
            frame = add_coin_history_context(frame)
        frame = frame.loc[(frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)]
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        if not frame.empty:
            frames[path.stem] = frame
    if not frames:
        raise RuntimeError("No evaluation data")
    return frames


def continuous_offsets(frame: pd.DataFrame, episode_hours: int) -> list[int]:
    offsets = []
    for offset in range(0, len(frame) - episode_hours + 1):
        window = frame.iloc[offset : offset + episode_hours]
        if window["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
            offsets.append(offset)
    return offsets


def sample_aligned_phase2_window(
    frames: dict[str, pd.DataFrame],
    indexes: dict[str, pd.Index],
    episode_hours: int,
    min_visible_coins: int,
    sample_coins: int | None,
    rng: np.random.Generator,
    max_attempts: int,
) -> dict[str, pd.DataFrame]:
    reference = max(frames, key=lambda coin: len(frames[coin]))
    max_offset = len(frames[reference]) - episode_hours
    if max_offset < 0:
        raise RuntimeError("Reference coin is shorter than the requested episode")
    for _ in range(max_attempts):
        offset = int(rng.integers(0, max_offset + 1))
        window = frames[reference].iloc[offset : offset + episode_hours]
        if not window["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
            continue
        aligned = phase2_window(frames, indexes, reference, offset, episode_hours)
        if len(aligned) >= min_visible_coins:
            if sample_coins is not None and len(aligned) > sample_coins:
                selected = rng.choice(sorted(aligned), size=sample_coins, replace=False)
                aligned = {coin: aligned[coin] for coin in selected}
            return aligned
    raise RuntimeError(
        "Could not sample a complete aligned Phase-2 window after "
        f"{max_attempts} attempts"
    )


def phase2_window(
    frames: dict[str, pd.DataFrame],
    indexes: dict[str, pd.Index],
    reference: str,
    offset: int,
    episode_hours: int,
) -> dict[str, pd.DataFrame]:
    timeline = pd.Index(frames[reference].iloc[offset : offset + episode_hours]["timestamp"])
    aligned = {}
    for coin, frame in frames.items():
        locations = indexes[coin].get_indexer(timeline)
        if not (locations < 0).any():
            aligned[coin] = frame.iloc[locations].reset_index(drop=True)
    return aligned


def run_episode(
    model,
    config: dict,
    payload: dict,
    frames: dict[str, pd.DataFrame],
    device: torch.device,
    fee_rate: float,
    stochastic: bool,
    action_temperature: float,
    max_slots: int,
    buy_prob_threshold: float,
) -> tuple[dict, float]:
    configured_targets = config.get("supervised_targets", "")
    supervised_targets = () if configured_targets == "none" else tuple(configured_targets.split(","))
    configured_raw_targets = config.get("supervised_raw_targets", "none")
    raw_targets = () if configured_raw_targets == "none" else tuple(configured_raw_targets.split(","))
    configured_neutral = config.get("neutral_supervised_targets", "none")
    neutral_targets = () if configured_neutral == "none" else tuple(configured_neutral.split(","))
    env = MultiCoinTradingEnv(
        frames,
        max_slots=min(max_slots, len(frames)),
        fee_rate=fee_rate,
        supervised_targets=supervised_targets,
        supervised_raw_targets=raw_targets,
        neutral_supervised_targets=neutral_targets,
        expose_episode_boundary=bool(config.get("expose_episode_boundary", True)),
        position_dynamics_mode=configured_position_dynamics_mode(config),
        buy_allocations=configured_buy_allocations(config),
        allocation_sizing_mode=configured_allocation_sizing_mode(config),
        cross_sectional_supervised_context=bool(config.get("cross_sectional_supervised_context", False)),
        coin_history_context=bool(config.get("coin_history_context", False)),
    )
    obs = env.reset()
    recurrent_hidden = model.initial_state(len(frames), device)
    actions = Counter()
    equities = []
    inference_seconds = 0.0
    done = False
    while not done:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output, recurrent_hidden = model.forward_step(obs.to(device), recurrent_hidden)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        obs, _, done, info = env.step(
            output,
            deterministic=not stochastic,
            action_temperature=action_temperature,
            buy_prob_threshold=buy_prob_threshold,
        )
        actions[info["action"]] += 1
        equities.append(info["equity"])
    equity = np.asarray(equities, dtype=np.float64)
    peak = np.maximum.accumulate(equity)
    first = next(iter(frames.values()))
    return {
        "start": first["timestamp"].iloc[0].isoformat(),
        "end": first["timestamp"].iloc[-1].isoformat(),
        "coins": len(frames),
        "return": float(equity[-1] / env.starting_cash - 1.0),
        "max_drawdown": float(((equity - peak) / np.maximum(peak, 1e-12)).min()),
        "trade_pnls": list(map(float, env.trade_pnls)),
        "trade_returns": list(map(float, env.trade_returns)),
        "trade_allocations": list(map(float, env.trade_allocations)),
        "trade_forced": list(map(bool, env.trade_forced)),
        "trade_details": env.trade_details,
        "manual_sells": env.manual_sell_count,
        "forced_sells": env.forced_sell_count,
        "actions": dict(actions),
        "checkpoint_episode": payload["episode"],
    }, inference_seconds


def aggregate(records: list[dict], inference_seconds: float, episode_hours: int) -> dict:
    returns = np.asarray([record["return"] for record in records], dtype=np.float64)
    pnl = np.asarray([
        pnl for record in records for pnl in record["trade_pnls"]
    ], dtype=np.float64)
    trade_returns = np.asarray([
        value for record in records for value in record["trade_returns"]
    ], dtype=np.float64)
    gross_profit = float(pnl[pnl > 0].sum()) if pnl.size else 0.0
    gross_loss = float(-pnl[pnl < 0].sum()) if pnl.size else 0.0
    manual = sum(int(record["manual_sells"]) for record in records)
    forced = sum(int(record["forced_sells"]) for record in records)
    actions = Counter()
    for record in records:
        actions.update(record["actions"])
    coin_steps = sum(record["coins"] * episode_hours for record in records)
    return {
        "episodes": len(records),
        "mean_visible_coins": float(np.mean([record["coins"] for record in records])),
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "profitable_episode_rate": float((returns > 0).mean()),
        "worst_max_drawdown": float(min(record["max_drawdown"] for record in records)),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "net_pnl_dollars": float(pnl.sum()) if pnl.size else 0.0,
        "gross_profit_dollars": gross_profit,
        "gross_loss_dollars": gross_loss,
        "expectancy_dollars": float(pnl.mean()) if pnl.size else 0.0,
        "win_rate": float((pnl > 0).mean()) if pnl.size else 0.0,
        "mean_trade_return_after_fees": float(trade_returns.mean()) if trade_returns.size else 0.0,
        "median_trade_return_after_fees": float(np.median(trade_returns)) if trade_returns.size else 0.0,
        "trades": int(pnl.size),
        "manual_sells": manual,
        "forced_sells": forced,
        "forced_sell_ratio": forced / max(1, forced + manual),
        "actions": dict(actions),
        "inference_seconds": inference_seconds,
        "inference_us_per_coin_step": inference_seconds * 1e6 / max(1, coin_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("train_ready/features/final_causal_v3_supervised_stable6_oof_2018_2026"),
    )
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--episode-days", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--min-visible-coins", type=int, default=1)
    parser.add_argument(
        "--sample-coins",
        type=int,
        help=(
            "Randomly select this many aligned visible coins per episode after "
            "sampling a valid window. Defaults to all visible coins."
        ),
    )
    parser.add_argument("--max-sample-attempts", type=int, default=5000)
    parser.add_argument("--max-slots", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--action-temperature", type=float, default=1.0)
    parser.add_argument("--buy-prob-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fee-rate", type=float)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.episode_days <= 0:
        parser.error("--episode-days must be positive")
    if args.action_temperature <= 0.0:
        parser.error("--action-temperature must be positive")
    if args.buy_prob_threshold < 0.0 or args.buy_prob_threshold > 1.0:
        parser.error("--buy-prob-threshold must be between 0 and 1")
    if args.max_sample_attempts <= 0:
        parser.error("--max-sample-attempts must be positive")
    if args.sample_coins is not None and args.sample_coins <= 0:
        parser.error("--sample-coins must be positive")
    if args.sample_coins is not None and args.sample_coins < args.min_visible_coins:
        parser.error("--sample-coins must be at least --min-visible-coins")
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model, config, payload = load_checkpoint(args.checkpoint, device)
    frames = load_frames(
        args.data_dir,
        args.start,
        args.end,
        coin_history_context=bool(config.get("coin_history_context", False)),
    )
    fee_rate = float(config["fee_rate"]) if args.fee_rate is None else float(args.fee_rate)
    episode_hours = args.episode_days * 24
    max_slots = args.max_slots if args.max_slots is not None else int(config.get("max_slots", 1))
    records = []
    inference_seconds = 0.0
    indexes = {coin: pd.Index(frame["timestamp"]) for coin, frame in frames.items()}
    for _ in range(args.episodes):
        aligned = sample_aligned_phase2_window(
            frames, indexes, episode_hours, max(1, args.min_visible_coins),
            args.sample_coins, rng, args.max_sample_attempts,
        )
        record, seconds = run_episode(
            model, config, payload, aligned, device, fee_rate,
            args.stochastic, args.action_temperature, max_slots,
            args.buy_prob_threshold,
        )
        records.append(record)
        inference_seconds += seconds
    result = {
        "phase": 2,
        "checkpoint": str(args.checkpoint),
        "checkpoint_episode": payload["episode"],
        "training_coverage": payload.get("training_coverage"),
        "period": {
            "start": args.start,
            "end": args.end,
            "sampling_scope": "random_phase2_aligned_windows",
            "episodes": args.episodes,
            "episode_days": args.episode_days,
            "min_visible_coins": args.min_visible_coins,
            "sample_coins": args.sample_coins,
        },
        "policy": {
            "stochastic": args.stochastic,
            "seed": args.seed,
            "action_temperature": args.action_temperature,
            "buy_prob_threshold": args.buy_prob_threshold,
        },
        "fee_rate": fee_rate,
        "aggregate": aggregate(records, inference_seconds, episode_hours),
        "per_episode": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["aggregate"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
