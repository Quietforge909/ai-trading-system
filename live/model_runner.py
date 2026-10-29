from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from rl.agent.joint_policy import select_joint_action
from rl.env.multi_portfolio import Position
from rl.evaluation.evaluate_kraken_full_universe_recent import aligned_recent_window
from rl.evaluation.evaluate_kraken_realistic_v280 import (
    PairRule,
    load_frames,
    make_env,
)
from rl.evaluation.evaluate_random_v7 import (
    configured_allocation_sizing_mode,
    load_checkpoint,
)


@dataclass(frozen=True)
class ModelDecision:
    action: str
    symbol: str | None
    confidence: float
    allocation_fraction: float
    reason: str
    visible_coins: int
    timestamp: str


def _position_from_holding(
    holding: dict[str, Any],
    frame: pd.DataFrame,
    current_step: int,
) -> Position:
    entry_ts = pd.Timestamp(holding.get("entry_ts"))
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize("UTC")
    else:
        entry_ts = entry_ts.tz_convert("UTC")
    matches = frame.index[frame["timestamp"] >= entry_ts]
    entry_step = int(matches[0]) if len(matches) else 0
    entry_step = min(entry_step, current_step)
    entry_price = float(holding["entry_price"])
    return Position(
        size=float(holding["size"]),
        entry_price=entry_price,
        entry_step=entry_step,
        cost_basis=float(holding.get("cost_basis", float(holding["size"]) * entry_price)),
        allocation_fraction=float(holding.get("allocation_fraction", 1.0)),
        mfe=float(holding.get("mfe", 0.0)),
        mae=float(holding.get("mae", 0.0)),
        mfe_step=entry_step,
        mae_step=entry_step,
    )


def _dummy_ai_rules(data_dir: Path) -> dict[str, PairRule]:
    rules = {}
    for path in sorted(data_dir.glob("*.parquet")):
        symbol = path.stem
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        rules[symbol] = PairRule(
            source_symbol=symbol,
            corrected_symbol=symbol,
            pair_id=symbol,
            wsname=f"{base}/USDT",
            base=base,
            quote="USDT",
            ordermin=0.0,
            costmin=0.0,
            pair_decimals=8,
            lot_decimals=8,
        )
    return rules


def decide(
    checkpoint: Path,
    data_dir: Path,
    state_holdings: dict[str, dict[str, Any]],
    rows: int,
    min_visible_coins: int,
    max_slots: int,
    buy_prob_threshold: float,
    device_name: str = "cpu",
    allowed_symbols: set[str] | None = None,
    excluded_symbols: set[str] | None = None,
) -> tuple[ModelDecision, dict[str, Any]]:
    corrected_rules = _dummy_ai_rules(data_dir)
    if allowed_symbols is not None:
        corrected_rules = {
            symbol: rule
            for symbol, rule in corrected_rules.items()
            if symbol in allowed_symbols
        }
    if excluded_symbols:
        corrected_rules = {
            symbol: rule
            for symbol, rule in corrected_rules.items()
            if symbol not in excluded_symbols or symbol in state_holdings
        }
    frames = load_frames(data_dir, corrected_rules)
    aligned = aligned_recent_window(frames, rows, min_visible_coins)
    corrected_rules = {coin: corrected_rules[coin] for coin in aligned}
    device = torch.device(device_name)
    model, config, payload = load_checkpoint(checkpoint, device)
    fee_rate = float(config["fee_rate"])
    env = make_env(aligned, config, corrected_rules, fee_rate, max_slots)
    env.reset()
    env.current_step = env.total_steps - 1
    current_step = env.current_step

    for symbol, holding in state_holdings.items():
        if symbol in aligned:
            env.portfolio.positions[symbol] = _position_from_holding(
                holding, aligned[symbol], current_step
            )
    mark_prices = {
        coin: float(frame["close"].iloc[current_step])
        for coin, frame in aligned.items()
    }
    env.portfolio.update(mark_prices, step=current_step)
    obs = env.observation()
    recurrent_hidden = model.initial_state(len(aligned), device)
    with torch.inference_mode():
        output, _ = model.forward_step(obs.to(device), recurrent_hidden)
    allow_buy = len(env.portfolio.positions) < env.portfolio.max_slots and env.portfolio.cash > 0.0
    candidate, _, _, confidence = select_joint_action(
        output["buy_logits"],
        output["sell_logits"],
        env.held_mask(),
        allow_buy=allow_buy,
        deterministic=True,
        mode_logits=output.get("mode_logits"),
        temperature=1.0,
    )
    ts = next(iter(aligned.values()))["timestamp"].iloc[current_step].isoformat()
    confidence = float(confidence)
    if candidate.kind == "buy":
        if confidence < buy_prob_threshold:
            return ModelDecision(
                action="noop",
                symbol=None,
                confidence=confidence,
                allocation_fraction=0.0,
                reason="buy_below_prob_threshold",
                visible_coins=len(aligned),
                timestamp=ts,
            ), {"checkpoint_episode": int(payload["episode"])}
        symbol = env.coins[candidate.coin_index]
        if configured_allocation_sizing_mode(config) == "equal_empty_slot":
            empty_slots = max(1, env.portfolio.max_slots - len(env.portfolio.positions))
            allocation = min(1.0, 1.0 / float(empty_slots))
        else:
            allocation = float(env.buy_allocations[candidate.allocation_index])
        return ModelDecision(
            action="buy",
            symbol=symbol,
            confidence=confidence,
            allocation_fraction=allocation,
            reason="model_buy",
            visible_coins=len(aligned),
            timestamp=ts,
        ), {"checkpoint_episode": int(payload["episode"])}
    if candidate.kind == "sell":
        symbol = env.coins[candidate.coin_index]
        if symbol not in state_holdings:
            return ModelDecision(
                action="noop",
                symbol=symbol,
                confidence=confidence,
                allocation_fraction=0.0,
                reason="sell_signal_for_unmanaged_or_empty_position",
                visible_coins=len(aligned),
                timestamp=ts,
            ), {"checkpoint_episode": int(payload["episode"])}
        return ModelDecision(
            action="sell",
            symbol=symbol,
            confidence=confidence,
            allocation_fraction=0.0,
            reason="model_sell",
            visible_coins=len(aligned),
            timestamp=ts,
        ), {"checkpoint_episode": int(payload["episode"])}
    return ModelDecision(
        action="noop",
        symbol=None,
        confidence=confidence if math.isfinite(confidence) else 0.0,
        allocation_fraction=0.0,
        reason="model_noop",
        visible_coins=len(aligned),
        timestamp=ts,
    ), {"checkpoint_episode": int(payload["episode"])}
