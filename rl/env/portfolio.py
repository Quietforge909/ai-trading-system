# rl/env/portfolio.py

from dataclasses import dataclass


@dataclass
class PortfolioState:
    cash: float
    position_size: float
    entry_price: float
    entry_step: int
    has_position: bool
    mfe: float  # max favorable excursion
    mae: float  # max adverse excursion
    entry_cost_basis: float  # remaining cost basis, including entry fee
    allocation_fraction: float


class Portfolio:
    """
    Phase 1 Portfolio:
    - Single position
    - Full balance all-in
    - Fixed taker fee
    """

    def __init__(
        self,
        starting_cash: float = 1000.0,
        fee_rate: float = 0.005,
        fee_round_decimals: int = 1,
    ):
        self.starting_cash = starting_cash
        self.fee_rate = fee_rate
        self.fee_round_decimals = fee_round_decimals

        self.reset()

    # ---------------------------------------------------------

    def reset(self):
        self.state = PortfolioState(
            cash=self.starting_cash,
            position_size=0.0,
            entry_price=0.0,
            entry_step=0,
            has_position=False,
            mfe=0.0,
            mae=0.0,
            entry_cost_basis=0.0,
            allocation_fraction=0.0,
        )

    # ---------------------------------------------------------

    def can_buy(self) -> bool:
        return (not self.state.has_position) and (self.state.cash > 0)

    def can_sell(self) -> bool:
        return self.state.has_position

    # ---------------------------------------------------------

    def execute_buy(self, price: float, step: int, allocation: float = 1.0):
        """
        Full balance buy.
        """
        if not self.can_buy():
            return

        cash = self.state.cash
        allocation = max(0.0, min(1.0, float(allocation)))
        budget = cash * allocation
        if budget <= 0.0:
            return

        # Compute the fee first, then size the position so total cash spent is
        # exactly the selected budget even when fees are rounded.
        estimated_notional = budget / (1.0 + self.fee_rate)
        fee = round(estimated_notional * self.fee_rate, self.fee_round_decimals)
        notional = max(0.0, budget - fee)
        size = notional / price
        total_cost = notional + fee

        # Update state
        self.state.cash = cash - total_cost
        self.state.position_size = size
        self.state.entry_price = price
        self.state.entry_step = step
        self.state.has_position = True
        self.state.mfe = 0.0
        self.state.mae = 0.0
        self.state.entry_cost_basis = total_cost
        self.state.allocation_fraction = allocation

    # ---------------------------------------------------------

    def execute_sell(self, price: float):
        """
        Close entire position.
        Returns realized PnL.
        """
        if not self.can_sell():
            return 0.0

        size = self.state.position_size
        entry_cost_basis = self.state.entry_cost_basis

        notional = price * size
        fee = round(notional * self.fee_rate, self.fee_round_decimals)

        proceeds = notional - fee

        # Realized PnL
        realized_pnl = proceeds - entry_cost_basis

        # Update cash
        self.state.cash += proceeds

        # Reset position
        self.state.position_size = 0.0
        self.state.entry_price = 0.0
        self.state.entry_step = 0
        self.state.has_position = False
        self.state.mfe = 0.0
        self.state.mae = 0.0
        self.state.entry_cost_basis = 0.0
        self.state.allocation_fraction = 0.0
        

        return realized_pnl

    # ---------------------------------------------------------

    def force_close(self, price: float):
        """
        Force close at episode end.
        """
        return self.execute_sell(price)

    # ---------------------------------------------------------

    def update_unrealized_metrics(self, price: float):
        """
        Track MFE and MAE.
        """
        if not self.state.has_position:
            return

        entry_price = self.state.entry_price
        current_return = (price - entry_price) / entry_price

        self.state.mfe = max(self.state.mfe, current_return)
        self.state.mae = min(self.state.mae, current_return)

    # ---------------------------------------------------------

    def current_equity(self, price: float) -> float:
        """
        Cash + current market value of position.
        """
        equity = self.state.cash
        if self.state.has_position:
            equity += self.state.position_size * price
        return equity
