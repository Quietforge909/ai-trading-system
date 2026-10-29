"""Independent single-coin, full-universe Phase-1 evaluation for v7."""

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
    configured_buy_allocations, configured_position_dynamics_mode,
    load_checkpoint,
)


def resolve_window_scope(
    windows_per_coin: int | None, all_windows: bool,
) -> tuple[int | None, str]:
    if all_windows and windows_per_coin is not None:
        raise ValueError("all-windows and windows-per-coin are mutually exclusive")
    if windows_per_coin is not None and windows_per_coin < 1:
        raise ValueError("windows-per-coin must be positive")
    if all_windows:
        return None, "all"
    if windows_per_coin is None:
        return 1, "midpoint"
    return windows_per_coin, "midpoint" if windows_per_coin == 1 else "spaced"


def build_windows(
    data_dir: Path, start: str, end: str, days: int,
    windows_per_coin: int | None = None,
    coin_history_context: bool = False,
):
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    hours, windows = days * 24, []
    for path in sorted(data_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if coin_history_context:
            frame = add_coin_history_context(frame)
        frame = frame.loc[(frame["timestamp"] >= lo) & (frame["timestamp"] <= hi)]
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        offsets = [
            offset for offset in range(0, len(frame) - hours + 1, hours)
            if frame.iloc[offset : offset + hours]["timestamp"].diff().dropna()
            .eq(pd.Timedelta(hours=1)).all()
        ]
        if windows_per_coin is not None and len(offsets) > windows_per_coin:
            if windows_per_coin == 1:
                picked = np.asarray([len(offsets) // 2])
            else:
                picked = np.linspace(0, len(offsets) - 1, windows_per_coin).round().astype(int)
            offsets = [offsets[index] for index in sorted(set(picked.tolist()))]
        for offset in offsets:
            window = frame.iloc[offset : offset + hours].reset_index(drop=True)
            windows.append((path.stem, window))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("train_ready/features/final_causal_v2_supervised6"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--episode-days", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--action-temperature", type=float, default=1.0,
        help="Stochastic sampling temperature; deterministic evaluation ignores this",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--fee-rate", type=float,
        help="Optional round-trip robustness override per side; defaults to the checkpoint configuration.",
    )
    parser.add_argument("--windows-per-coin", type=int)
    parser.add_argument(
        "--all-windows", action="store_true",
        help="Explicitly evaluate every contiguous window (potentially expensive)",
    )
    args = parser.parse_args()
    try:
        windows_per_coin, sampling_scope = resolve_window_scope(
            args.windows_per_coin, args.all_windows
        )
    except ValueError as error:
        parser.error(str(error))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model, config, payload = load_checkpoint(args.checkpoint, device)
    fee_rate = float(config["fee_rate"]) if args.fee_rate is None else float(args.fee_rate)
    if fee_rate < 0:
        parser.error("fee-rate must be non-negative")
    if args.action_temperature <= 0.0:
        parser.error("action-temperature must be positive")
    targets = () if config.get("supervised_targets") == "none" else tuple(config["supervised_targets"].split(","))
    raw_targets = (
        () if config.get("supervised_raw_targets", "none") == "none"
        else tuple(config["supervised_raw_targets"].split(","))
    )
    neutral_targets = (
        () if config.get("neutral_supervised_targets", "none") == "none"
        else tuple(config["neutral_supervised_targets"].split(","))
    )
    windows = build_windows(
        args.data_dir, args.start, args.end, args.episode_days,
        windows_per_coin=windows_per_coin,
        coin_history_context=bool(config.get("coin_history_context", False)),
    )
    if not windows:
        raise RuntimeError("No complete Phase-1 windows")
    envs, observations, records = [], [], []
    for coin, frame in windows:
        env = MultiCoinTradingEnv(
            {coin: frame}, 1, fee_rate=fee_rate,
            supervised_targets=targets,
            supervised_raw_targets=raw_targets,
            neutral_supervised_targets=neutral_targets,
            expose_episode_boundary=bool(config.get("expose_episode_boundary", True)),
            position_dynamics_mode=configured_position_dynamics_mode(config),
            buy_allocations=configured_buy_allocations(config),
            cross_sectional_supervised_context=bool(config.get("cross_sectional_supervised_context", False)),
            coin_history_context=bool(config.get("coin_history_context", False)),
        )
        envs.append(env)
        observations.append(env.reset())
        records.append({
            "coin": coin, "start": frame["timestamp"].iloc[0].isoformat(),
            "buy_hold_return_before_fees": float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0),
            "return": 0.0, "max_drawdown": 0.0, "trade_pnls": [],
            "trade_returns": [], "trade_allocations": [], "trade_forced": [],
            "manual_sells": 0, "forced_sells": 0, "actions": Counter(), "equity": [],
            "buy_allocations": Counter(), "manual_holding_steps": [],
        })
    inference_seconds = 0.0
    recurrent_hidden = model.initial_state(len(envs), device)
    for _ in range(args.episode_days * 24):
        batch = torch.cat(observations, dim=0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output, recurrent_hidden = model.forward_step(batch, recurrent_hidden)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        next_observations = []
        for index, env in enumerate(envs):
            per_env = {
                "buy_logits": output["buy_logits"][index].unsqueeze(0),
                "sell_logits": output["sell_logits"][index].unsqueeze(0),
                "value": output["value"][index].reshape(1),
            }
            if "mode_logits" in output:
                per_env["mode_logits"] = output["mode_logits"][index].unsqueeze(0)
            next_obs, _, done, info = env.step(
                per_env, deterministic=not args.stochastic,
                action_temperature=args.action_temperature,
            )
            record = records[index]
            record["actions"][info["action"]] += 1
            if info["action"] == "buy":
                record["buy_allocations"][f'{info["allocation_fraction"]:.2f}'] += 1
            elif info["action"] == "sell" and info["holding_steps"] is not None:
                record["manual_holding_steps"].append(info["holding_steps"])
            record["equity"].append(info["equity"])
            if not done:
                next_observations.append(next_obs)
        observations = next_observations
    all_pnls, all_trade_returns, aggregate_actions = [], [], Counter()
    for record, env in zip(records, envs):
        equity = np.asarray(record.pop("equity"), dtype=np.float64)
        peak = np.maximum.accumulate(equity)
        record["return"] = float(equity[-1] / env.starting_cash - 1.0)
        record["max_drawdown"] = float(((equity - peak) / np.maximum(peak, 1e-12)).min())
        record["trade_pnls"] = list(map(float, env.trade_pnls))
        record["trade_returns"] = list(map(float, env.trade_returns))
        record["trade_allocations"] = list(map(float, env.trade_allocations))
        record["trade_forced"] = list(map(bool, env.trade_forced))
        if not (
            len(record["trade_pnls"]) == len(record["trade_returns"])
            == len(record["trade_allocations"]) == len(record["trade_forced"])
        ):
            raise RuntimeError("Trade diagnostic arrays are not aligned")
        record["manual_sells"] = env.manual_sell_count
        record["forced_sells"] = env.forced_sell_count
        record["actions"] = dict(record["actions"])
        record["buy_allocations"] = dict(record["buy_allocations"])
        aggregate_actions.update(record["actions"])
        all_pnls.extend(record["trade_pnls"])
        all_trade_returns.extend(record["trade_returns"])
    returns = np.asarray([record["return"] for record in records])
    baselines = np.asarray([record["buy_hold_return_before_fees"] for record in records])
    pnl = np.asarray(all_pnls)
    trade_returns = np.asarray(all_trade_returns)
    gross_profit = float(pnl[pnl > 0].sum()) if pnl.size else 0.0
    gross_loss = float(-pnl[pnl < 0].sum()) if pnl.size else 0.0
    net_pnl = float(pnl.sum()) if pnl.size else 0.0
    forced = sum(record["forced_sells"] for record in records)
    manual = sum(record["manual_sells"] for record in records)
    aggregate = {
        "windows": len(records), "coins": len({record["coin"] for record in records}),
        "mean_return": float(returns.mean()), "median_return": float(np.median(returns)),
        "mean_buy_hold_before_fees": float(baselines.mean()),
        "mean_excess_vs_buy_hold_before_fees": float((returns - baselines).mean()),
        "profitable_window_rate": float((returns > 0).mean()),
        "worst_max_drawdown": min(record["max_drawdown"] for record in records),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "net_pnl_dollars": net_pnl,
        "gross_profit_dollars": gross_profit,
        "gross_loss_dollars": gross_loss,
        "expectancy_dollars": float(pnl.mean()) if pnl.size else 0.0,
        "win_rate": float((pnl > 0).mean()) if pnl.size else 0.0,
        "mean_trade_return_after_fees": float(trade_returns.mean()) if trade_returns.size else 0.0,
        "median_trade_return_after_fees": float(np.median(trade_returns)) if trade_returns.size else 0.0,
        "trades_above_6pct_rate": float((trade_returns >= 0.06).mean()) if trade_returns.size else 0.0,
        "trades_above_12pct_rate": float((trade_returns >= 0.12).mean()) if trade_returns.size else 0.0,
        "trades": int(pnl.size), "manual_sells": manual, "forced_sells": forced,
        "forced_sell_ratio": forced / max(1, forced + manual),
        "actions": dict(aggregate_actions),
        "buy_allocations": dict(sum(
            (Counter(record["buy_allocations"]) for record in records), Counter()
        )),
        "mean_manual_holding_hours": float(np.mean([
            hours for record in records for hours in record["manual_holding_steps"]
        ])) if any(record["manual_holding_steps"] for record in records) else None,
        "inference_us_per_coin_step": inference_seconds * 1e6 / (len(records) * args.episode_days * 24),
    }
    result = {
        "phase": 1, "checkpoint": str(args.checkpoint), "checkpoint_episode": payload["episode"],
        "training_coverage": payload.get("training_coverage"),
        "period": {
            "start": args.start, "end": args.end,
            "windows_per_coin": windows_per_coin,
            "sampling_scope": sampling_scope,
        },
        "policy": {
            "stochastic": args.stochastic,
            "seed": args.seed,
            "action_temperature": args.action_temperature,
        },
        "fee_rate": fee_rate,
        "aggregate": aggregate, "per_window": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
