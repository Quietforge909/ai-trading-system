from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from live.kraken_client import normalize_asset


@dataclass
class HoldingView:
    symbol: str
    base_asset: str
    size: float
    last_price: float
    value: float
    entry_price: float | None
    pnl: float | None
    pnl_pct: float | None
    managed: bool


@dataclass
class AccountSnapshot:
    cash: float
    total_equity: float
    holdings: list[HoldingView]
    unmanaged_assets: dict[str, float]
    open_orders: dict[str, Any]


def ticker_last(ticker_payload: dict[str, Any]) -> float:
    return float(ticker_payload["c"][0])


def ticker_bid_ask(ticker_payload: dict[str, Any]) -> tuple[float, float]:
    return float(ticker_payload["b"][0]), float(ticker_payload["a"][0])


def summarize_account(
    balances: dict[str, float],
    tickers_by_symbol: dict[str, Any],
    state_holdings: dict[str, dict[str, Any]],
    quote_asset: str,
    open_orders: dict[str, Any],
) -> AccountSnapshot:
    quote_asset = normalize_asset(quote_asset)
    cash = float(balances.get(quote_asset, 0.0))
    holdings: list[HoldingView] = []
    seen_assets = {quote_asset}
    total = cash

    for symbol, holding in sorted(state_holdings.items()):
        base_asset = normalize_asset(str(holding["base_asset"]))
        seen_assets.add(base_asset)
        size = float(balances.get(base_asset, holding.get("size", 0.0)))
        ticker = tickers_by_symbol.get(symbol)
        last = ticker_last(ticker) if ticker else float(holding.get("entry_price", 0.0))
        value = size * last
        entry = float(holding["entry_price"])
        cost_basis = float(holding.get("cost_basis", size * entry))
        pnl = value - cost_basis
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0.0
        total += value
        holdings.append(HoldingView(
            symbol=symbol,
            base_asset=base_asset,
            size=size,
            last_price=last,
            value=value,
            entry_price=entry,
            pnl=pnl,
            pnl_pct=pnl_pct,
            managed=True,
        ))

    unmanaged: dict[str, float] = {}
    for asset, amount in balances.items():
        asset = normalize_asset(asset)
        if asset in seen_assets or amount <= 0:
            continue
        unmanaged[asset] = amount

    return AccountSnapshot(
        cash=cash,
        total_equity=total,
        holdings=holdings,
        unmanaged_assets=unmanaged,
        open_orders=open_orders,
    )

