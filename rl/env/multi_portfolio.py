"""Fee-aware multi-position portfolio used by the v7 joint policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    size: float
    entry_price: float
    entry_step: int
    cost_basis: float
    allocation_fraction: float
    mfe: float = 0.0
    mae: float = 0.0
    mfe_step: int = 0
    mae_step: int = 0


class MultiCoinPortfolio:
    def __init__(self, starting_cash: float, fee_rate: float, fee_round_decimals: int, max_slots: int):
        self.starting_cash = float(starting_cash)
        self.fee_rate = float(fee_rate)
        self.fee_round_decimals = int(fee_round_decimals)
        self.max_slots = int(max_slots)
        self.reset()

    def reset(self) -> None:
        self.cash = self.starting_cash
        self.positions: dict[str, Position] = {}

    def can_buy(self, coin: str) -> bool:
        return coin not in self.positions and len(self.positions) < self.max_slots and self.cash > 0.0

    def can_sell(self, coin: str) -> bool:
        return coin in self.positions

    def buy(self, coin: str, price: float, step: int, allocation: float) -> bool:
        if not self.can_buy(coin) or price <= 0.0:
            return False
        allocation = min(1.0, max(0.0, float(allocation)))
        budget = self.cash * allocation
        if budget <= 0.0:
            return False
        estimated_notional = budget / (1.0 + self.fee_rate)
        fee = round(estimated_notional * self.fee_rate, self.fee_round_decimals)
        notional = max(0.0, budget - fee)
        if notional <= 0.0:
            return False
        self.cash -= notional + fee
        self.positions[coin] = Position(
            size=notional / price,
            entry_price=price,
            entry_step=step,
            cost_basis=notional + fee,
            allocation_fraction=allocation,
            mfe_step=step,
            mae_step=step,
        )
        return True

    def sell(self, coin: str, price: float) -> float:
        position = self.positions.get(coin)
        if position is None:
            return 0.0
        notional = position.size * price
        fee = round(notional * self.fee_rate, self.fee_round_decimals)
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

    def equity(self, prices: dict[str, float]) -> float:
        # Report realizable liquidation equity, including the exit fee already
        # owed by every open position.  This makes the full round-trip cost
        # visible at entry instead of deferring half the fee signal to the end
        # of a long GAE trace; terminal liquidation then causes no accounting
        # discontinuity beyond price movement and fee rounding changes.
        liquidation_value = 0.0
        for coin, position in self.positions.items():
            notional = position.size * prices[coin]
            fee = round(notional * self.fee_rate, self.fee_round_decimals)
            liquidation_value += notional - fee
        return float(self.cash + liquidation_value)

    def liquidate(self, prices: dict[str, float]) -> list[tuple[str, float]]:
        return [(coin, self.sell(coin, prices[coin])) for coin in list(self.positions)]
