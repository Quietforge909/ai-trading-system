from __future__ import annotations

from typing import Any

import requests

from live.account import AccountSnapshot
from live.executor import ExecutionResult
from live.model_runner import ModelDecision


def money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.2f}%"


def build_report(
    account: AccountSnapshot,
    decision: ModelDecision,
    execution_results: list[ExecutionResult],
    dry_run: bool,
    checkpoint_label: str,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    status = "DRY RUN" if dry_run else "LIVE"
    holdings = "\n".join(
        f"`{h.symbol}` size `{h.size:.8f}` value {money(h.value)} "
        f"PnL {money(h.pnl)} ({pct(h.pnl_pct)})"
        for h in account.holdings
    ) or "No managed holdings"
    actions = "\n".join(
        f"{r.action.upper()} `{r.symbol or '-'}`: {r.reason}, "
        f"price={r.price}, volume={r.volume}, notional={r.notional}, txid={r.txid}"
        for r in execution_results
    ) or "No order submitted"
    error_text = "\n".join(errors or []) or "None"
    unmanaged = ", ".join(f"{k}:{v:.8f}" for k, v in sorted(account.unmanaged_assets.items())) or "None"

    description = (
        f"Mode: **{status}**\n"
        f"Model: `{checkpoint_label}`\n"
        f"Decision candle: `{decision.timestamp}`\n"
        f"Decision: **{decision.action.upper()}** `{decision.symbol or '-'}` "
        f"conf `{decision.confidence:.6f}` reason `{decision.reason}`\n"
        f"Visible coins: `{decision.visible_coins}`\n\n"
        f"Cash: **{money(account.cash)}**\n"
        f"Estimated equity: **{money(account.total_equity)}**\n"
        f"Open orders: `{len(account.open_orders)}`\n\n"
        f"Holdings:\n{holdings}\n\n"
        f"Orders:\n{actions}\n\n"
        f"Unmanaged assets: {unmanaged}\n"
        f"Errors: {error_text}"
    )
    return {
        "username": "Kraken AI Trader",
        "embeds": [{
            "title": "Hourly live cycle report",
            "description": description[:3900],
            "color": 0x2f80ed if not errors else 0xd64545,
        }],
    }


def build_status_report(title: str, description: str, ok: bool = True) -> dict[str, Any]:
    return {
        "username": "Kraken AI Trader",
        "embeds": [{
            "title": title,
            "description": description[:3900],
            "color": 0x27ae60 if ok else 0xd64545,
        }],
    }


def send_report(webhook_url: str, payload: dict[str, Any]) -> None:
    if not webhook_url:
        return
    response = requests.post(webhook_url, json=payload, timeout=20)
    response.raise_for_status()
