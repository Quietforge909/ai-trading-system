from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BotState:
    holdings: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_cycle_ts: str | None = None
    last_action: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "BotState":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(
            holdings=dict(raw.get("holdings") or {}),
            last_cycle_ts=raw.get("last_cycle_ts"),
            last_action=dict(raw.get("last_action") or {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "holdings": self.holdings,
            "last_cycle_ts": self.last_cycle_ts,
            "last_action": self.last_action,
        }, indent=2, sort_keys=True) + "\n")

    def used_slots(self) -> int:
        return len(self.holdings)

    def register_buy(
        self,
        symbol: str,
        base_asset: str,
        size: float,
        entry_price: float,
        cost_basis: float,
        allocation_fraction: float,
        timestamp: str,
        txid: str | None,
    ) -> None:
        self.holdings[symbol] = {
            "symbol": symbol,
            "base_asset": base_asset,
            "size": float(size),
            "entry_price": float(entry_price),
            "cost_basis": float(cost_basis),
            "allocation_fraction": float(allocation_fraction),
            "entry_ts": timestamp,
            "txid": txid,
            "mfe": 0.0,
            "mae": 0.0,
        }

    def register_sell(self, symbol: str) -> dict[str, Any] | None:
        return self.holdings.pop(symbol, None)
