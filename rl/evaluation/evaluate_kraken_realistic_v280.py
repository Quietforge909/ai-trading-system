#!/usr/bin/env python3
"""Realistic Kraken replay audit for the frozen v280 checkpoint.

This preserves the original evaluation JSON as the control and changes only
evaluation accounting:
- unique control episode starts only
- decisions observe candle t and fill at candle t+1 open
- Kraken USD pair metadata, minimum size, lot precision, and cost minimum
- explicit spread/slippage assumptions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch

from rl.agent.joint_policy import select_joint_action
from rl.env.multi_env import MultiCoinTradingEnv
from rl.env.multi_portfolio import Position
from rl.evaluation.evaluate_random_v7 import (
    configured_allocation_sizing_mode,
    configured_buy_allocations,
    configured_position_dynamics_mode,
    load_checkpoint,
)


LEGACY_ASSET_MAP = {
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
}


@dataclass(frozen=True)
class PairRule:
    source_symbol: str
    corrected_symbol: str
    pair_id: str
    wsname: str
    base: str
    quote: str
    ordermin: float
    costmin: float
    pair_decimals: int
    lot_decimals: int


class RealisticPortfolio:
    def __init__(self, starting_cash: float, fee_rate: float, max_slots: int, rules: dict[str, PairRule]):
        self.starting_cash = float(starting_cash)
        self.fee_rate = float(fee_rate)
        self.fee_round_decimals = 1
        self.max_slots = int(max_slots)
        self.rules = rules
        self.reset()

    def reset(self) -> None:
        self.cash = self.starting_cash
        self.positions: dict[str, Position] = {}

    def can_buy(self, coin: str) -> bool:
        return coin not in self.positions and len(self.positions) < self.max_slots and self.cash > 0.0

    def buy(self, coin: str, price: float, step: int, allocation: float) -> tuple[bool, str | None]:
        if not self.can_buy(coin) or price <= 0.0:
            return False, "cannot_buy"
        rule = self.rules[coin]
        budget = self.cash * min(1.0, max(0.0, float(allocation)))
        if budget < rule.costmin:
            return False, "below_costmin_budget"
        notional_before_fee = budget / (1.0 + self.fee_rate)
        size = math.floor((notional_before_fee / price) * (10 ** rule.lot_decimals)) / (10 ** rule.lot_decimals)
        notional = size * price
        fee = notional * self.fee_rate
        total_cost = notional + fee
        if size < rule.ordermin:
            return False, "below_ordermin"
        if notional < rule.costmin:
            return False, "below_costmin_notional"
        if total_cost > self.cash + 1e-9:
            return False, "insufficient_cash_after_precision"
        self.cash -= total_cost
        self.positions[coin] = Position(
            size=size,
            entry_price=price,
            entry_step=step,
            cost_basis=total_cost,
            allocation_fraction=float(allocation),
            mfe_step=step,
            mae_step=step,
        )
        return True, None

    def sell(self, coin: str, price: float) -> float:
        position = self.positions.get(coin)
        if position is None:
            return 0.0
        notional = position.size * price
        fee = notional * self.fee_rate
        proceeds = notional - fee
        pnl = proceeds - position.cost_basis
        self.cash += proceeds
        del self.positions[coin]
        return float(pnl)

    def update(self, prices: dict[str, float], step: int | None = None) -> None:
        for coin, position in self.positions.items():
            current_return = prices[coin] / position.entry_price - 1.0
            if current_return > position.mfe:
                position.mfe = current_return
                if step is not None:
                    position.mfe_step = int(step)
            if current_return < position.mae:
                position.mae = current_return
                if step is not None:
                    position.mae_step = int(step)

    def equity(self, prices: dict[str, float], sell_cost_mult: float = 1.0) -> float:
        value = self.cash
        for coin, position in self.positions.items():
            notional = position.size * prices[coin] * sell_cost_mult
            value += notional - notional * self.fee_rate
        return float(value)

    def liquidate(self, prices: dict[str, float]) -> list[tuple[str, float]]:
        return [(coin, self.sell(coin, prices[coin])) for coin in list(self.positions)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_asset(asset: str) -> str:
    asset = (asset or "").replace(".d", "").upper()
    return LEGACY_ASSET_MAP.get(asset, asset)


def load_pair_rules(manifest_path: Path) -> dict[str, PairRule]:
    manifest = json.loads(manifest_path.read_text())
    api = requests.get("https://api.kraken.com/0/public/AssetPairs", timeout=30).json()
    if api.get("error"):
        raise RuntimeError(api["error"])
    asset_pairs = api["result"]
    rules = {}
    used_corrected = Counter()
    for market in manifest["markets"]:
        pair_id = market["pair_id"]
        info = asset_pairs[pair_id]
        wsname = info.get("wsname") or market["wsname"]
        raw_base, raw_quote = wsname.split("/", 1)
        base = normalize_asset(raw_base)
        quote = normalize_asset(raw_quote)
        corrected = f"{base}USDT"
        used_corrected[corrected] += 1
        rules[market["symbol"]] = PairRule(
            source_symbol=market["symbol"],
            corrected_symbol=corrected,
            pair_id=pair_id,
            wsname=wsname,
            base=base,
            quote=quote,
            ordermin=float(info.get("ordermin") or 0.0),
            costmin=float(info.get("costmin") or 0.0),
            pair_decimals=int(info.get("pair_decimals") or 8),
            lot_decimals=int(info.get("lot_decimals") or 8),
        )
    duplicates = [symbol for symbol, count in used_corrected.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Corrected symbol duplicates: {duplicates[:10]}")
    return rules


def load_frames(data_dir: Path, rules: dict[str, PairRule]) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(data_dir.glob("*.parquet")):
        if path.stem not in rules:
            continue
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        frames[rules[path.stem].corrected_symbol] = frame
    return dict(sorted(frames.items()))


def build_corrected_rules_by_symbol(source_rules: dict[str, PairRule]) -> dict[str, PairRule]:
    return {rule.corrected_symbol: rule for rule in source_rules.values()}


def episode_window(frames: dict[str, pd.DataFrame], start: str, episode_hours: int, min_visible: int) -> dict[str, pd.DataFrame]:
    start_ts = pd.Timestamp(start)
    timeline = pd.date_range(start_ts, periods=episode_hours, freq="h")
    aligned = {}
    for coin, frame in frames.items():
        indexed = frame.set_index("timestamp", drop=False)
        if timeline[0] < indexed.index.min() or timeline[-1] > indexed.index.max():
            continue
        loc = indexed.reindex(timeline)
        if loc["timestamp"].isna().any():
            continue
        aligned[coin] = loc.reset_index(drop=True)
    if len(aligned) < min_visible:
        raise RuntimeError(f"{start}: visible={len(aligned)} < {min_visible}")
    return dict(sorted(aligned.items()))


def adjusted_buy(open_price: float, side_bps: float) -> float:
    return float(open_price) * (1.0 + side_bps / 10000.0)


def adjusted_sell(open_price: float, side_bps: float) -> float:
    return float(open_price) * (1.0 - side_bps / 10000.0)


def feature_snapshot(env: MultiCoinTradingEnv, obs: torch.Tensor, coin_index: int) -> dict[str, float]:
    names = list(getattr(env, "feature_columns", []))
    row = obs.detach().cpu()[coin_index]
    return {
        name: float(row[index])
        for index, name in enumerate(names)
        if index < row.numel()
    }


def make_env(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    rules: dict[str, PairRule],
    fee_rate: float,
    max_slots: int,
) -> MultiCoinTradingEnv:
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
    env.portfolio = RealisticPortfolio(env.starting_cash, fee_rate, min(max_slots, len(frames)), rules)
    return env


def run_ai_episode(
    model,
    config: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    rules: dict[str, PairRule],
    device: torch.device,
    fee_rate: float,
    max_slots: int,
    side_cost_bps: float,
    episode_index: int,
    deterministic: bool,
    temperature: float,
    buy_prob_threshold: float = 0.0,
) -> dict[str, Any]:
    env = make_env(frames, config, rules, fee_rate, max_slots)
    obs = env.reset()
    recurrent_hidden = model.initial_state(len(frames), device)
    trades = []
    pending: dict[str, dict[str, Any]] = {}
    actions = Counter()
    rejected = Counter()
    equities = []
    exposures = []
    start_ts = frames[next(iter(frames))]["timestamp"].iloc[0]
    end_ts = frames[next(iter(frames))]["timestamp"].iloc[-1]
    while env.current_step < env.total_steps - 1:
        mark_prices = {coin: float(frame["close"].iloc[env.current_step]) for coin, frame in frames.items()}
        eq = env.portfolio.equity(mark_prices, sell_cost_mult=1.0 - side_cost_bps / 10000.0)
        exposures.append(max(0.0, eq - env.portfolio.cash) / max(eq, 1e-12))
        with torch.inference_mode():
            output, recurrent_hidden = model.forward_step(obs.to(device), recurrent_hidden)
        held_mask = env.held_mask()
        allow_buy = len(env.portfolio.positions) < env.portfolio.max_slots and env.portfolio.cash > 0.0
        candidate, _, _, confidence = select_joint_action(
            output["buy_logits"],
            output["sell_logits"],
            held_mask,
            allow_buy=allow_buy,
            deterministic=deterministic,
            mode_logits=output.get("mode_logits"),
            temperature=temperature,
        )
        action = "noop"
        exec_step = env.current_step + 1
        if candidate.kind == "buy":
            if float(confidence) < buy_prob_threshold:
                rejected["buy_below_prob_threshold"] += 1
                actions[action] += 1
                env.current_step = exec_step
                next_marks = {coin: float(frame["close"].iloc[env.current_step]) for coin, frame in frames.items()}
                env.portfolio.update(next_marks, step=env.current_step)
                equities.append(env.portfolio.equity(next_marks, sell_cost_mult=1.0 - side_cost_bps / 10000.0))
                obs = env.observation()
                continue
            coin = env.coins[candidate.coin_index]
            if configured_allocation_sizing_mode(config) == "equal_empty_slot":
                empty_slots = env.portfolio.max_slots - len(env.portfolio.positions)
                allocation = 0.0 if empty_slots <= 0 else min(1.0, 1.0 / float(empty_slots))
            else:
                allocation = env.buy_allocations[candidate.allocation_index]
            price = adjusted_buy(frames[coin]["open"].iloc[exec_step], side_cost_bps)
            ok, reason = env.portfolio.buy(coin, price, exec_step, allocation)
            if ok:
                action = "buy"
                pending[coin] = {
                    "entry_step": exec_step,
                    "entry_ts": frames[coin]["timestamp"].iloc[exec_step].isoformat(),
                    "entry_decision_step": env.current_step,
                    "entry_decision_ts": frames[coin]["timestamp"].iloc[env.current_step].isoformat(),
                    "allocation": allocation,
                    "entry_price": price,
                    "buy_confidence": float(confidence),
                    "entry_features": feature_snapshot(env, obs, candidate.coin_index),
                }
            else:
                rejected[f"buy_{reason}"] += 1
        elif candidate.kind == "sell":
            coin = env.coins[candidate.coin_index]
            if coin in env.portfolio.positions:
                position = env.portfolio.positions[coin]
                price = adjusted_sell(frames[coin]["open"].iloc[exec_step], side_cost_bps)
                cost_basis = position.cost_basis
                pnl = env.portfolio.sell(coin, price)
                entry = pending.pop(coin, {})
                trades.append({
                    "episode": episode_index,
                    "coin": coin,
                    "entry_step": entry.get("entry_step"),
                    "exit_step": exec_step,
                    "entry_ts": entry.get("entry_ts"),
                    "entry_decision_step": entry.get("entry_decision_step"),
                    "entry_decision_ts": entry.get("entry_decision_ts"),
                    "exit_ts": frames[coin]["timestamp"].iloc[exec_step].isoformat(),
                    "entry_price": entry.get("entry_price"),
                    "exit_price": price,
                    "allocation": entry.get("allocation", position.allocation_fraction),
                    "buy_confidence": entry.get("buy_confidence"),
                    "entry_features": entry.get("entry_features", {}),
                    "pnl": pnl,
                    "return": pnl / max(cost_basis, 1e-12),
                    "forced": False,
                })
                action = "sell"
        actions[action] += 1
        env.current_step = exec_step
        next_marks = {coin: float(frame["close"].iloc[env.current_step]) for coin, frame in frames.items()}
        env.portfolio.update(next_marks, step=env.current_step)
        equities.append(env.portfolio.equity(next_marks, sell_cost_mult=1.0 - side_cost_bps / 10000.0))
        obs = env.observation()
    final_step = env.total_steps - 1
    forced = []
    for coin in list(env.portfolio.positions):
        position = env.portfolio.positions[coin]
        price = adjusted_sell(frames[coin]["close"].iloc[final_step], side_cost_bps)
        cost_basis = position.cost_basis
        pnl = env.portfolio.sell(coin, price)
        entry = pending.pop(coin, {})
        row = {
            "episode": episode_index,
            "coin": coin,
            "entry_step": entry.get("entry_step"),
            "exit_step": final_step,
            "entry_ts": entry.get("entry_ts"),
            "entry_decision_step": entry.get("entry_decision_step"),
            "entry_decision_ts": entry.get("entry_decision_ts"),
            "exit_ts": frames[coin]["timestamp"].iloc[final_step].isoformat(),
            "entry_price": entry.get("entry_price"),
            "exit_price": price,
            "allocation": entry.get("allocation", position.allocation_fraction),
            "buy_confidence": entry.get("buy_confidence"),
            "entry_features": entry.get("entry_features", {}),
            "pnl": pnl,
            "return": pnl / max(cost_basis, 1e-12),
            "forced": True,
        }
        forced.append(row)
        trades.append(row)
    final_prices = {coin: float(frame["close"].iloc[final_step]) for coin, frame in frames.items()}
    final_equity = env.portfolio.equity(final_prices, sell_cost_mult=1.0 - side_cost_bps / 10000.0)
    equities.append(final_equity)
    equity = np.asarray(equities, dtype=np.float64)
    peak = np.maximum.accumulate(equity)
    return {
        "episode": episode_index,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "coins": len(frames),
        "pnl": float(final_equity - env.starting_cash),
        "return": float(final_equity / env.starting_cash - 1.0),
        "max_drawdown": float(((equity - peak) / np.maximum(peak, 1e-12)).min()) if len(equity) else 0.0,
        "trades": trades,
        "actions": dict(actions),
        "rejected": dict(rejected),
        "forced_sells": len(forced),
        "manual_sells": sum(1 for trade in trades if not trade["forced"]),
        "mean_exposure": float(np.mean(exposures)) if exposures else 0.0,
        "max_exposure": float(np.max(exposures)) if exposures else 0.0,
    }


def round_trip_pnl(entry: float, exit: float, budget: float, fee_rate: float, rule: PairRule) -> float | None:
    if budget < rule.costmin:
        return None
    notional_before_fee = budget / (1.0 + fee_rate)
    size = math.floor((notional_before_fee / entry) * (10 ** rule.lot_decimals)) / (10 ** rule.lot_decimals)
    notional = size * entry
    if size < rule.ordermin or notional < rule.costmin:
        return None
    fee_in = notional * fee_rate
    exit_notional = size * exit
    fee_out = exit_notional * fee_rate
    return float(exit_notional - fee_out - notional - fee_in)


def summarize_episode_pnls(records: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [record["pnl"] for record in records]
    trades = [trade for record in records for trade in record["trades"]]
    by_coin = defaultdict(float)
    by_entry_event = defaultdict(float)
    entry_event_counts = Counter()
    for trade in trades:
        by_coin[trade["coin"]] += trade["pnl"]
        key = (
            trade["coin"],
            trade.get("entry_ts"),
            trade.get("allocation"),
        )
        by_entry_event[key] += trade["pnl"]
        entry_event_counts[key] += 1
    total_pnl = float(sum(pnls))
    top_coin, top_coin_pnl = (None, 0.0)
    if by_coin:
        top_coin, top_coin_pnl = max(by_coin.items(), key=lambda item: item[1])
    top_event, top_event_pnl = (None, 0.0)
    if by_entry_event:
        top_event, top_event_pnl = max(by_entry_event.items(), key=lambda item: item[1])
    top_event_count = int(entry_event_counts[top_event]) if top_event is not None else 0
    dedup_event_mean_pnls = {
        key: pnl / max(1, entry_event_counts[key])
        for key, pnl in by_entry_event.items()
    }
    dedup_event_mean_total = float(sum(dedup_event_mean_pnls.values()))
    top_event_mean_pnl = float(dedup_event_mean_pnls[top_event]) if top_event is not None else 0.0
    return {
        "episodes": len(records),
        "best_episode_pnl": float(max(pnls)) if pnls else 0.0,
        "worst_episode_pnl": float(min(pnls)) if pnls else 0.0,
        "mean_episode_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "median_episode_pnl": float(np.median(pnls)) if pnls else 0.0,
        "profitable_episode_rate": float(np.mean([pnl > 0 for pnl in pnls])) if pnls else 0.0,
        "total_pnl_control_style_sum": total_pnl,
        "trades": len(trades),
        "manual_sells": int(sum(1 for trade in trades if not trade["forced"])),
        "forced_sells": int(sum(1 for trade in trades if trade["forced"])),
        "forced_sell_ratio": float(sum(1 for trade in trades if trade["forced"]) / max(1, len(trades))),
        "mean_exposure": float(np.mean([record["mean_exposure"] for record in records])) if records else 0.0,
        "max_exposure": float(max(record["max_exposure"] for record in records)) if records else 0.0,
        "by_coin": dict(sorted(by_coin.items(), key=lambda item: item[1], reverse=True)),
        "unique_traded_coins": len(by_coin),
        "top_coin": top_coin,
        "top_coin_pnl": float(top_coin_pnl),
        "top_coin_pnl_share": float(top_coin_pnl / total_pnl) if abs(total_pnl) > 1e-12 else 0.0,
        "pnl_without_top_coin": float(total_pnl - top_coin_pnl),
        "unique_entry_events": len(by_entry_event),
        "top_entry_event": {
            "coin": top_event[0],
            "entry_ts": top_event[1],
            "allocation": top_event[2],
        } if top_event is not None else None,
        "top_entry_event_repeat_count": top_event_count,
        "top_entry_event_pnl": float(top_event_pnl),
        "top_entry_event_pnl_share": float(top_event_pnl / total_pnl) if abs(total_pnl) > 1e-12 else 0.0,
        "pnl_without_top_entry_event": float(total_pnl - top_event_pnl),
        "dedup_entry_event_mean_pnl_sum": dedup_event_mean_total,
        "top_entry_event_mean_pnl": top_event_mean_pnl,
        "top_entry_event_mean_pnl_share_of_dedup": (
            float(top_event_mean_pnl / dedup_event_mean_total)
            if abs(dedup_event_mean_total) > 1e-12 else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/phase1_credit_diag/w64_v280_v276_sellboundary60_fullera_seed42/checkpoint_episode_000000.pt"))
    parser.add_argument("--control", type=Path, default=Path("experiments/results/kraken_all_recent_v280_det_21d_200ep_seed42_20260716.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("train_ready/features/kraken_all_recent_final_causal_v3_supervised_stable6_oof_2026_20260716"))
    parser.add_argument("--manifest", type=Path, default=Path("data/kraken/ohlcv_1h_all_20260716/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/kraken_all_recent_v280_realistic_unique_21d_seed42_20260716.json"))
    parser.add_argument("--side-cost-bps", type=float, default=15.0, help="Half spread plus slippage per side, in bps.")
    parser.add_argument("--min-visible-coins", type=int, default=600)
    parser.add_argument("--exclude-algo", action="store_true")
    parser.add_argument("--exclude-symbols", type=str, default="", help="Comma-separated corrected symbols, e.g. OGNUSDT,ALGOUSDT.")
    parser.add_argument("--random-reps", type=int, default=100)
    parser.add_argument("--max-unique-starts", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample model actions instead of greedy argmax.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Action sampling temperature when --stochastic is set.")
    parser.add_argument("--policy-seed", type=int, default=123, help="Seed for stochastic action sampling.")
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    random.seed(args.policy_seed)
    np.random.seed(args.policy_seed)
    torch.manual_seed(args.policy_seed)

    control = json.loads(args.control.read_text())
    unique_starts = []
    seen = set()
    for record in control["per_episode"]:
        if record["start"] not in seen:
            seen.add(record["start"])
            unique_starts.append(record["start"])
    if args.max_unique_starts > 0:
        unique_starts = unique_starts[: args.max_unique_starts]
    episode_hours = int(control["period"]["episode_days"]) * 24

    source_rules = load_pair_rules(args.manifest)
    corrected_rules = build_corrected_rules_by_symbol(source_rules)
    frames = load_frames(args.data_dir, source_rules)
    if args.exclude_algo:
        frames.pop("ALGOUSDT", None)
        corrected_rules.pop("ALGOUSDT", None)
    excluded_symbols = [symbol.strip().upper() for symbol in args.exclude_symbols.split(",") if symbol.strip()]
    for symbol in excluded_symbols:
        frames.pop(symbol, None)
        corrected_rules.pop(symbol, None)
    model, config, payload = load_checkpoint(args.checkpoint, torch.device("cpu"))
    device = torch.device("cpu")
    fee_rate = float(config["fee_rate"])
    max_slots = int(config.get("max_slots", 1))

    episode_cache = {
        start: episode_window(frames, start, episode_hours, args.min_visible_coins)
        for start in unique_starts
    }

    records = []
    for index, start in enumerate(unique_starts):
        aligned = episode_cache[start]
        records.append(run_ai_episode(
            model, config, aligned, corrected_rules, device, fee_rate, max_slots,
            args.side_cost_bps, index, not args.stochastic, args.temperature,
        ))
        if (index + 1) % 20 == 0:
            print(f"episodes {index + 1}/{len(unique_starts)}", flush=True)

    trades = [trade for record in records for trade in record["trades"]]
    train_symbols = {path.stem for path in Path("train_ready/features/final_causal_v3_supervised_stable6_oof_2018_2026").glob("*.parquet")}
    no_algo_pnl_by_removal = sum(trade["pnl"] for trade in trades if trade["coin"] != "ALGOUSDT")

    rng = random.Random(123)
    random_totals = []
    all_symbols = sorted(frames)
    for _ in range(args.random_reps):
        total = 0.0
        for record in records:
            aligned = episode_cache[record["start"]]
            for trade in record["trades"]:
                budget = 1000.0 * float(trade["allocation"] or 0.05)
                candidates = list(aligned)
                rng.shuffle(candidates)
                for symbol in candidates:
                    rule = corrected_rules[symbol]
                    entry_step = int(trade["entry_step"])
                    exit_step = int(trade["exit_step"])
                    entry = adjusted_buy(aligned[symbol]["open"].iloc[entry_step], args.side_cost_bps)
                    exit_col = "close" if trade["forced"] else "open"
                    exit_raw = aligned[symbol][exit_col].iloc[exit_step]
                    exit_price = adjusted_sell(exit_raw, args.side_cost_bps)
                    pnl = round_trip_pnl(entry, exit_price, budget, fee_rate, rule)
                    if pnl is not None:
                        total += pnl
                        break
        random_totals.append(total)

    def baseline_hold(symbol: str, allocation: float = 1.0) -> dict[str, float]:
        pnls = []
        for record in records:
            aligned = episode_cache[record["start"]]
            if symbol not in aligned:
                continue
            rule = corrected_rules[symbol]
            entry = adjusted_buy(aligned[symbol]["open"].iloc[1], args.side_cost_bps)
            exit_price = adjusted_sell(aligned[symbol]["close"].iloc[-1], args.side_cost_bps)
            pnl = round_trip_pnl(entry, exit_price, 1000.0 * allocation, fee_rate, rule)
            if pnl is not None:
                pnls.append(pnl)
        return {
            "episodes": len(pnls),
            "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
            "median_pnl": float(np.median(pnls)) if pnls else 0.0,
            "total_control_style_sum": float(sum(pnls)),
            "profitable_rate": float(np.mean([pnl > 0 for pnl in pnls])) if pnls else 0.0,
        }

    algo_timing_pnls = []
    fixed_algo_rep_pnls = []
    for record in records:
        aligned = episode_cache[record["start"]]
        if "ALGOUSDT" in aligned:
            rule = corrected_rules["ALGOUSDT"]
            for trade in record["trades"]:
                budget = 1000.0 * float(trade["allocation"] or 0.05)
                entry = adjusted_buy(aligned["ALGOUSDT"]["open"].iloc[int(trade["entry_step"])], args.side_cost_bps)
                exit_col = "close" if trade["forced"] else "open"
                exit_price = adjusted_sell(aligned["ALGOUSDT"][exit_col].iloc[int(trade["exit_step"])], args.side_cost_bps)
                pnl = round_trip_pnl(entry, exit_price, budget, fee_rate, rule)
                if pnl is not None:
                    algo_timing_pnls.append(pnl)
        # Fixed simple ALGO+REP policy: 2.5% ALGO and 2.5% REP from first executable open to final close.
        fixed = 0.0
        for symbol in ("ALGOUSDT", "REPUSDT"):
            if symbol in aligned:
                rule = corrected_rules[symbol]
                entry = adjusted_buy(aligned[symbol]["open"].iloc[1], args.side_cost_bps)
                exit_price = adjusted_sell(aligned[symbol]["close"].iloc[-1], args.side_cost_bps)
                pnl = round_trip_pnl(entry, exit_price, 25.0, fee_rate, rule)
                fixed += 0.0 if pnl is None else pnl
        fixed_algo_rep_pnls.append(fixed)

    algo_trades = [trade for trade in trades if trade["coin"] == "ALGOUSDT"]
    result = {
        "control_result": str(args.control),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data_dir": str(args.data_dir),
        "unique_control_starts": len(unique_starts),
        "episode_days": control["period"]["episode_days"],
        "side_cost_bps_per_side": args.side_cost_bps,
        "fee_rate": fee_rate,
        "policy_mode": "stochastic" if args.stochastic else "deterministic",
        "temperature": args.temperature,
        "policy_seed": args.policy_seed,
        "exclude_algo": args.exclude_algo,
        "excluded_symbols": excluded_symbols,
        "corrected_symbol_mappings": {
            rule.source_symbol: rule.corrected_symbol
            for rule in source_rules.values()
            if rule.source_symbol != rule.corrected_symbol
        },
        "summary": summarize_episode_pnls(records),
        "pnl_without_algo_by_removing_algo_trades": float(no_algo_pnl_by_removal),
        "pnl_from_symbols_seen_in_training": float(sum(trade["pnl"] for trade in trades if trade["coin"] in train_symbols)),
        "pnl_from_symbols_not_seen_in_training": float(sum(trade["pnl"] for trade in trades if trade["coin"] not in train_symbols)),
        "algo_entry_exit_timing": [
            {
                "episode": trade["episode"],
                "start": records[trade["episode"]]["start"],
                "entry_ts": trade["entry_ts"],
                "exit_ts": trade["exit_ts"],
                "entry_step": trade["entry_step"],
                "exit_step": trade["exit_step"],
                "forced": trade["forced"],
                "pnl": trade["pnl"],
            }
            for trade in algo_trades
        ],
        "baselines": {
            "algo_buy_and_hold_full_balance": baseline_hold("ALGOUSDT") if not args.exclude_algo else None,
            "algo_using_ai_exact_timing_and_exposure": {
                "trades": len(algo_timing_pnls),
                "mean_episode_scaled_trade_pnl": float(sum(algo_timing_pnls) / max(1, len(records))),
                "total_control_style_sum": float(sum(algo_timing_pnls)),
            } if not args.exclude_algo else None,
            "fixed_algo_rep_2p5pct_each_hold": {
                "mean_pnl": float(np.mean(fixed_algo_rep_pnls)) if fixed_algo_rep_pnls else 0.0,
                "median_pnl": float(np.median(fixed_algo_rep_pnls)) if fixed_algo_rep_pnls else 0.0,
                "total_control_style_sum": float(sum(fixed_algo_rep_pnls)),
                "profitable_rate": float(np.mean([pnl > 0 for pnl in fixed_algo_rep_pnls])) if fixed_algo_rep_pnls else 0.0,
            } if not args.exclude_algo else None,
            "random_coins_identical_timing_exposure": {
                "reps": len(random_totals),
                "mean_total": float(np.mean(random_totals)),
                "median_total": float(np.median(random_totals)),
                "best_total": float(max(random_totals)) if random_totals else 0.0,
                "worst_total": float(min(random_totals)) if random_totals else 0.0,
                "profitable_rate": float(np.mean([value > 0 for value in random_totals])) if random_totals else 0.0,
            },
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "summary": result["summary"],
        "pnl_without_algo_by_removing_algo_trades": result["pnl_without_algo_by_removing_algo_trades"],
        "baselines": result["baselines"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
