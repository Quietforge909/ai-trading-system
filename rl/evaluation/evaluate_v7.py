"""Chronological full-available-universe evaluation for v7 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rl.agent.model import PPOTradingModel
from rl.agent.joint_policy import joint_action_logits
from rl.env.coin_history_context import add_coin_history_context
from rl.env.multi_env import MultiCoinTradingEnv


SUPPORTED_CHECKPOINT_SPECS = frozenset((7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22))


def configured_buy_allocations(config: dict) -> tuple[float, ...]:
    encoded = config.get("buy_allocations", "0.25,0.50,0.75,1.00")
    if isinstance(encoded, str):
        return tuple(float(value) for value in encoded.split(","))
    return tuple(float(value) for value in encoded)


def configured_allocation_sizing_mode(config: dict) -> str:
    return str(config.get("allocation_sizing_mode", "action_fraction"))


def configured_position_dynamics_mode(config: dict) -> str:
    """Resolve the brief pre-versioning fee-context experiment compatibly."""
    mode = config.get("position_dynamics_mode", "disabled")
    columns = config.get("observation_columns", [])
    if mode == "active" and "realizable_net_return" in columns:
        return "active_fee"
    if mode == "neutral" and "realizable_net_return" in columns:
        return "neutral_fee"
    return mode


def evaluation_windows(
    data_dir: Path, start: str, end: str, episode_days: int,
    coin_history_context: bool = False,
):
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    frames, indexes = {}, {}
    for path in sorted(data_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if coin_history_context:
            frame = add_coin_history_context(frame)
        frame = frame.loc[(frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)]
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        if not frame.empty:
            frames[path.stem] = frame
            indexes[path.stem] = pd.Index(frame["timestamp"])
    if not frames:
        raise RuntimeError("No evaluation data")
    reference = max(frames, key=lambda coin: len(frames[coin]))
    hours = episode_days * 24
    windows = []
    for offset in range(0, len(frames[reference]) - hours + 1, hours):
        base = frames[reference].iloc[offset : offset + hours].reset_index(drop=True)
        timeline = pd.Index(base["timestamp"])
        if not pd.Series(timeline).diff().dropna().eq(pd.Timedelta(hours=1)).all():
            continue
        aligned = {}
        for coin in sorted(frames):
            locations = indexes[coin].get_indexer(timeline)
            if not (locations < 0).any():
                aligned[coin] = frames[coin].iloc[locations].reset_index(drop=True)
        windows.append(aligned)
    return windows


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("spec_version") not in SUPPORTED_CHECKPOINT_SPECS:
        raise RuntimeError(f"Not a supported robust checkpoint: {path}")
    config = payload["config"]
    model = PPOTradingModel(
        int(config["input_dim"]), int(config["width"]), int(config["width"]), int(config["width"]),
        num_res_blocks=int(config["blocks"]), dropout=float(config["dropout"]),
        sell_head_width=int(config.get("sell_head_width", 0)),
        temporal_width=int(config.get("temporal_width", 0)),
        actor_mode=config.get("actor_mode", "shared"),
        state_value_heads=bool(config.get("state_value_heads", False)),
        held_feature_index=config.get("held_feature_index"),
        action_value_heads=bool(config.get("action_value_heads", False)),
        flat_action_value_heads=bool(config.get("flat_action_value_heads", False)),
        action_value_detach_encoder=bool(config.get("sell_q_detach_encoder", False)),
        sell_raw_residual=bool(config.get("sell_raw_residual", False)),
        sell_raw_residual_width=int(config.get("sell_raw_residual_width", 0)),
        buy_raw_residual=bool(config.get("buy_raw_residual", False)),
        buy_raw_residual_width=int(config.get("buy_raw_residual_width", 0)),
        score_residual=bool(config.get("score_residual", False)),
        score_residual_width=int(config.get("score_residual_width", 0)),
        score_feature_indices=tuple(config.get("score_feature_indices", ())),
        score_residual_neutral_base=bool(config.get("score_residual_neutral_base", False)),
        universe_context_width=int(config.get("universe_context_width", 0)),
        mode_head=bool(config.get("mode_head", False)),
        boundary_sell_residual=bool(config.get("boundary_sell_residual", False)),
        boundary_sell_feature_index=config.get("boundary_sell_feature_index"),
        gated_buy_residual=bool(config.get("gated_buy_residual", False)),
        gated_buy_feature_index=config.get("gated_buy_feature_index"),
        gated_buy_feature_min=float(config.get("gated_buy_feature_min", float("-inf"))),
        gated_buy_feature_max=float(config.get("gated_buy_feature_max", float("inf"))),
        gated_buy_feature2_index=config.get("gated_buy_feature2_index"),
        gated_buy_feature2_min=float(config.get("gated_buy_feature2_min", float("-inf"))),
        gated_buy_feature2_max=float(config.get("gated_buy_feature2_max", float("inf"))),
        gated_buy2_residual=bool(config.get("gated_buy2_residual", False)),
        gated_buy2_feature_index=config.get("gated_buy2_feature_index"),
        gated_buy2_feature_min=float(config.get("gated_buy2_feature_min", float("-inf"))),
        gated_buy2_feature_max=float(config.get("gated_buy2_feature_max", float("inf"))),
        gated_buy2_feature2_index=config.get("gated_buy2_feature2_index"),
        gated_buy2_feature2_min=float(config.get("gated_buy2_feature2_min", float("-inf"))),
        gated_buy2_feature2_max=float(config.get("gated_buy2_feature2_max", float("inf"))),
        gated_buy3_residual=bool(config.get("gated_buy3_residual", False)),
        gated_buy3_feature_index=config.get("gated_buy3_feature_index"),
        gated_buy3_feature_min=float(config.get("gated_buy3_feature_min", float("-inf"))),
        gated_buy3_feature_max=float(config.get("gated_buy3_feature_max", float("inf"))),
        gated_buy3_feature2_index=config.get("gated_buy3_feature2_index"),
        gated_buy3_feature2_min=float(config.get("gated_buy3_feature2_min", float("-inf"))),
        gated_buy3_feature2_max=float(config.get("gated_buy3_feature2_max", float("inf"))),
        gated_sell_residual=bool(config.get("gated_sell_residual", False)),
        gated_sell_feature_index=config.get("gated_sell_feature_index"),
        gated_sell_feature_min=float(config.get("gated_sell_feature_min", float("-inf"))),
        gated_sell_feature_max=float(config.get("gated_sell_feature_max", float("inf"))),
        gated_sell_feature2_index=config.get("gated_sell_feature2_index"),
        gated_sell_feature2_min=float(config.get("gated_sell_feature2_min", float("-inf"))),
        gated_sell_feature2_max=float(config.get("gated_sell_feature2_max", float("inf"))),
        sell_target_aux_width=int(config.get("sell_target_aux_width", 0)),
        sell_target_policy_residual=bool(config.get("sell_target_policy_residual", False)),
        sell_target_policy_residual_width=int(config.get("sell_target_policy_residual_width", 0)),
        sell_target_policy_residual_source=str(
            config.get("sell_target_policy_residual_source", "score")
        ),
        buy_excluded_feature_indices=tuple(
            config.get("buy_excluded_feature_indices", ())
        ),
        sell_excluded_feature_indices=tuple(
            config.get("sell_excluded_feature_indices", ())
        ),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("train_ready/features/final_causal_v2_supervised6"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--episode-days", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fee-rate", type=float, help="Optional round-trip-cost stress via per-side fee")
    parser.add_argument("--stochastic", action="store_true", help="Evaluate the learned categorical policy by sampling")
    parser.add_argument(
        "--action-temperature", type=float, default=1.0,
        help="Stochastic sampling temperature; deterministic evaluation ignores this",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.action_temperature <= 0.0:
        parser.error("action-temperature must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model, config, payload = load_checkpoint(args.checkpoint, device)
    windows = evaluation_windows(
        args.data_dir, args.start, args.end, args.episode_days,
        coin_history_context=bool(config.get("coin_history_context", False)),
    )
    if not windows:
        raise RuntimeError("No complete evaluation windows")
    reports, all_pnls, all_trade_returns = [], [], []
    aggregate_actions = Counter()
    coin_presence, coin_actions = Counter(), Counter()
    inference_seconds = 0.0
    mode_probabilities = {"noop": [], "buy": [], "sell": []}
    for frames in windows:
        for coin in frames:
            coin_presence[coin] += 1
        configured_targets = config.get("supervised_targets", "")
        supervised_targets = () if configured_targets == "none" else tuple(configured_targets.split(","))
        configured_raw_targets = config.get("supervised_raw_targets", "none")
        raw_targets = () if configured_raw_targets == "none" else tuple(configured_raw_targets.split(","))
        configured_neutral = config.get("neutral_supervised_targets", "none")
        neutral_targets = () if configured_neutral == "none" else tuple(configured_neutral.split(","))
        env = MultiCoinTradingEnv(
            frames, max_slots=min(int(config["max_slots"]), len(frames)),
            fee_rate=float(args.fee_rate if args.fee_rate is not None else config["fee_rate"]),
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
        equities, actions = [], Counter()
        done = False
        while not done:
            batch = obs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output, recurrent_hidden = model.forward_step(batch, recurrent_hidden)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started
            joint_logits, candidates = joint_action_logits(
                output["buy_logits"], output["sell_logits"], env.held_mask().to(device),
                len(env.portfolio.positions) < env.portfolio.max_slots and env.portfolio.cash > 0.0,
                mode_logits=output.get("mode_logits"),
            )
            joint_probs = torch.softmax(joint_logits, dim=0)
            for kind in mode_probabilities:
                indices = [i for i, candidate in enumerate(candidates) if candidate.kind == kind]
                mode_probabilities[kind].append(
                    float(joint_probs[torch.tensor(indices, device=device)].sum()) if indices else 0.0
                )
            obs, _, done, info = env.step(
                output, deterministic=not args.stochastic,
                action_temperature=args.action_temperature,
            )
            actions[info["action"]] += 1
            aggregate_actions[info["action"]] += 1
            if info["coin"] is not None:
                coin_actions[f"{info['coin']}:{info['action']}"] += 1
            equities.append(info["equity"])
        equity = np.asarray(equities, dtype=np.float64)
        peak = np.maximum.accumulate(equity)
        pnl = np.asarray(env.trade_pnls, dtype=np.float64)
        trade_returns = np.asarray(env.trade_returns, dtype=np.float64)
        all_pnls.extend(pnl.tolist())
        all_trade_returns.extend(trade_returns.tolist())
        first = next(iter(frames.values()))
        reports.append({
            "start": first["timestamp"].iloc[0].isoformat(),
            "end": first["timestamp"].iloc[-1].isoformat(),
            "coins": len(frames),
            "return": float(equity[-1] / env.starting_cash - 1.0),
            "max_drawdown": float(((equity - peak) / np.maximum(peak, 1e-12)).min()),
            "trades": int(pnl.size),
            "mean_trade_return_after_fees": float(trade_returns.mean()) if trade_returns.size else 0.0,
            "trades_above_6pct_rate": float((trade_returns >= 0.06).mean()) if trade_returns.size else 0.0,
            "trades_above_12pct_rate": float((trade_returns >= 0.12).mean()) if trade_returns.size else 0.0,
            "manual_sells": env.manual_sell_count,
            "forced_sells": env.forced_sell_count,
            "actions": dict(actions),
        })
    returns = np.asarray([report["return"] for report in reports])
    pnl = np.asarray(all_pnls)
    trade_returns = np.asarray(all_trade_returns)
    gross_profit = float(pnl[pnl > 0].sum()) if pnl.size else 0.0
    gross_loss = float(-pnl[pnl < 0].sum()) if pnl.size else 0.0
    forced_sells = sum(report["forced_sells"] for report in reports)
    manual_sells = sum(report["manual_sells"] for report in reports)
    total_coin_steps = sum(report["coins"] * args.episode_days * 24 for report in reports)
    result = {
        "phase": int(config.get("phase", 1)),
        "checkpoint": str(args.checkpoint),
        "checkpoint_episode": payload["episode"],
        "config": config,
        "period": {
            "start": args.start, "end": args.end,
            "fee_rate": float(args.fee_rate if args.fee_rate is not None else config["fee_rate"]),
            "stochastic": args.stochastic,
            "seed": args.seed,
            "action_temperature": args.action_temperature,
        },
        "aggregate": {
            "windows": len(reports),
            "unique_coins_evaluated": len(coin_presence),
            "mean_return": float(returns.mean()),
            "median_return": float(np.median(returns)),
            "profitable_window_rate": float((returns > 0).mean()),
            "worst_max_drawdown": min(report["max_drawdown"] for report in reports),
            "profit_factor": gross_profit / gross_loss if gross_loss else None,
            "net_pnl_dollars": float(pnl.sum()) if pnl.size else 0.0,
            "gross_profit_dollars": gross_profit,
            "gross_loss_dollars": gross_loss,
            "expectancy_dollars": float(pnl.mean()) if pnl.size else 0.0,
            "win_rate": float((pnl > 0).mean()) if pnl.size else 0.0,
            "mean_trade_return_after_fees": float(trade_returns.mean()) if trade_returns.size else 0.0,
            "median_trade_return_after_fees": float(np.median(trade_returns)) if trade_returns.size else 0.0,
            "trades_above_6pct_rate": float((trade_returns >= 0.06).mean()) if trade_returns.size else 0.0,
            "trades_above_12pct_rate": float((trade_returns >= 0.12).mean()) if trade_returns.size else 0.0,
            "trades": int(pnl.size),
            "manual_sells": manual_sells,
            "forced_sells": forced_sells,
            "forced_sell_ratio": forced_sells / max(1, forced_sells + manual_sells),
            "actions": dict(aggregate_actions),
            "inference_seconds": inference_seconds,
            "inference_us_per_coin_step": inference_seconds * 1e6 / max(1, total_coin_steps),
            "mode_probabilities": {
                kind: {"mean": float(np.mean(values)), "max": float(np.max(values))}
                for kind, values in mode_probabilities.items()
            },
        },
        "coin_window_presence": dict(coin_presence),
        "coin_actions": dict(coin_actions),
        "per_window": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
