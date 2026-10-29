from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from live.account import ticker_bid_ask
from live.config import LiveConfig
from live.kraken_client import KrakenClient, floor_decimal
from rl.evaluation.evaluate_kraken_realistic_v280 import PairRule


@dataclass(frozen=True)
class ExecutionResult:
    action: str
    symbol: str | None
    submitted: bool
    dry_run: bool
    txid: str | None
    price: float | None
    volume: float | None
    notional: float | None
    reason: str
    raw: dict[str, Any]


def spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid <= 0 else (ask - bid) / mid * 10000.0


def current_taker_fee(
    client: KrakenClient,
    pair_id: str,
    fallback: float,
) -> float:
    try:
        result = client.trade_volume([pair_id])
        fees = result.get("fees", {}).get(pair_id) or result.get("fees", {}).get(pair_id.upper())
        if fees:
            return float(fees[0]["fee"]) / 100.0
    except Exception:
        return fallback
    return fallback


def _wait_for_order(client: KrakenClient, txid: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last: dict[str, Any] = {}
    while time.time() < deadline:
        result = client.query_orders([txid])
        last = result.get(txid, {})
        if last.get("status") in {"closed", "canceled", "expired"}:
            return last
        time.sleep(3)
    return last


def execute_sell(
    client: KrakenClient,
    config: LiveConfig,
    rule: PairRule,
    holding: dict[str, Any],
    ticker_payload: dict[str, Any] | None = None,
) -> ExecutionResult:
    latest_ticker = client.ticker([rule.pair_id])
    ticker_payload = (
        latest_ticker.get(rule.pair_id)
        or latest_ticker.get(rule.source_symbol)
        or ticker_payload
    )
    if not ticker_payload:
        return ExecutionResult("sell", rule.corrected_symbol, False, config.dry_run, None, None, None, None, "missing_ticker", {})
    bid, ask = ticker_bid_ask(ticker_payload)
    volume = float(holding["size"])
    price = bid
    volume_str = floor_decimal(volume, rule.lot_decimals)
    price_str = floor_decimal(price, rule.pair_decimals)
    volume = float(volume_str)
    price = float(price_str)
    notional = volume * price
    if notional < rule.costmin:
        return ExecutionResult("sell", rule.corrected_symbol, False, config.dry_run, None, price, volume, notional, "below_costmin", {})
    if config.dry_run:
        return ExecutionResult("sell", rule.corrected_symbol, False, True, None, price, volume, notional, "dry_run_sell", {})
    try:
        raw = client.add_order(
            pair=rule.pair_id,
            side="sell",
            ordertype="market",
            volume=volume_str,
            client_order_id=str(uuid.uuid4()),
        )
    except RuntimeError as exc:
        if "Insufficient funds" in str(exc):
            return ExecutionResult("sell", rule.corrected_symbol, False, False, None, price, volume, notional, "insufficient_position", {"error": str(exc)})
        if "Invalid permissions" in str(exc) or "trading restricted" in str(exc):
            return ExecutionResult("sell", rule.corrected_symbol, False, False, None, price, volume, notional, "trading_restricted", {"error": str(exc)})
        raise
    txid = raw.get("txid", [None])[0] if isinstance(raw.get("txid"), list) else None
    order = _wait_for_order(client, txid, config.order_timeout_seconds) if txid else {}
    return ExecutionResult("sell", rule.corrected_symbol, True, False, txid, price, volume, notional, "submitted_sell", {"add": raw, "order": order})


def execute_buy(
    client: KrakenClient,
    config: LiveConfig,
    rule: PairRule,
    ticker_payload: dict[str, Any],
    available_cash: float,
    allocation_fraction: float,
) -> ExecutionResult:
    bid, ask = ticker_bid_ask(ticker_payload)
    current_spread = spread_bps(bid, ask)
    if current_spread > config.max_spread_bps:
        return ExecutionResult("buy", rule.corrected_symbol, False, config.dry_run, None, None, None, None, "spread_too_wide", {})
    fee_rate = current_taker_fee(client, rule.pair_id, config.taker_fee_fallback)
    cash = max(0.0, available_cash - config.min_cash_reserve)
    cash *= 0.99
    budget = cash * min(1.0, max(0.0, allocation_fraction))
    min_required = max(config.min_order_usd, rule.costmin)
    min_budget_with_fee = min_required * (1.0 + fee_rate) * 1.005
    if budget < min_required:
        if budget >= min_required * 0.90 and cash >= min_budget_with_fee:
            budget = min_budget_with_fee
        else:
            return ExecutionResult("buy", rule.corrected_symbol, False, config.dry_run, None, None, None, budget, "below_min_order_budget", {})
    budget = min(budget, cash)
    price = ask * (1.0 + config.marketable_limit_bps / 10000.0)
    notional_before_fee = budget / (1.0 + fee_rate)
    volume = notional_before_fee / price
    volume_str = floor_decimal(volume, rule.lot_decimals)
    price_str = floor_decimal(price, rule.pair_decimals)
    volume = float(volume_str)
    price = float(price_str)
    notional = volume * price
    if volume < rule.ordermin:
        return ExecutionResult("buy", rule.corrected_symbol, False, config.dry_run, None, price, volume, notional, "below_ordermin", {})
    if notional < rule.costmin:
        return ExecutionResult("buy", rule.corrected_symbol, False, config.dry_run, None, price, volume, notional, "below_costmin", {})
    if config.dry_run:
        return ExecutionResult("buy", rule.corrected_symbol, False, True, None, price, volume, notional, "dry_run_buy", {})
    try:
        raw = client.add_order(
            pair=rule.pair_id,
            side="buy",
            ordertype="limit",
            volume=volume_str,
            price=price_str,
            client_order_id=str(uuid.uuid4()),
        )
    except RuntimeError as exc:
        if "Insufficient funds" in str(exc):
            return ExecutionResult("buy", rule.corrected_symbol, False, False, None, price, volume, notional, "insufficient_funds", {"error": str(exc)})
        if "Invalid permissions" in str(exc) or "trading restricted" in str(exc):
            return ExecutionResult("buy", rule.corrected_symbol, False, False, None, price, volume, notional, "trading_restricted", {"error": str(exc)})
        raise
    txid = raw.get("txid", [None])[0] if isinstance(raw.get("txid"), list) else None
    order = _wait_for_order(client, txid, config.order_timeout_seconds) if txid else {}
    return ExecutionResult("buy", rule.corrected_symbol, True, False, txid, price, volume, notional, "submitted_buy", {"add": raw, "order": order})
