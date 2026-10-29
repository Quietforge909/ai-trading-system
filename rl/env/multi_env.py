"""Multi-coin, multi-slot environment with one joint action per hour."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from rl.agent.joint_policy import select_joint_action
from rl.config import BUY_ALLOCATIONS
from rl.env.coin_history_context import COIN_HISTORY_CONTEXT_COLUMNS
from rl.env.multi_portfolio import MultiCoinPortfolio
from rl.env.sub_features import SubFeatureConfig, build_sub_features


POSITION_DYNAMICS_COLUMNS = [
    "drawdown_from_mfe",
    "rebound_from_mae",
    "hours_since_mfe_normalized",
    "hours_since_mae_normalized",
    "unrealized_return_velocity_1h",
    "unrealized_return_velocity_4h",
    "unrealized_return_velocity_24h",
]
FEE_POSITION_DYNAMICS_COLUMNS = [
    "realizable_net_return",
    "fee_drag_from_raw_return",
]
POSITION_MEMORY_COLUMNS = [
    "memory_realizable_net_return",
    "memory_pnl_change_1h",
    "memory_mfe",
    "memory_mae",
    "memory_fade_from_mfe",
]
BUY_MEMORY_WINDOWS = (
    ("4h", 4),
    ("24h", 24),
    ("7d", 24 * 7),
    ("14d", 24 * 14),
    ("30d", 24 * 30),
)
BUY_MEMORY_COLUMNS = [
    name
    for label, _ in BUY_MEMORY_WINDOWS
    for name in (
        f"buy_memory_rebound_from_low_{label}",
        f"buy_memory_drawdown_from_high_{label}",
    )
]


def position_dynamics_columns(mode: str) -> list[str]:
    if mode == "disabled":
        return []
    if mode in {"neutral", "active"}:
        return POSITION_DYNAMICS_COLUMNS
    if mode in {"neutral_fee", "active_fee"}:
        return POSITION_DYNAMICS_COLUMNS + FEE_POSITION_DYNAMICS_COLUMNS
    if mode in {"neutral_memory", "active_memory"}:
        return POSITION_DYNAMICS_COLUMNS + FEE_POSITION_DYNAMICS_COLUMNS + POSITION_MEMORY_COLUMNS
    if mode in {"neutral_memory_buy", "active_memory_buy"}:
        return (
            POSITION_DYNAMICS_COLUMNS
            + FEE_POSITION_DYNAMICS_COLUMNS
            + POSITION_MEMORY_COLUMNS
            + BUY_MEMORY_COLUMNS
        )
    raise ValueError(f"unknown position dynamics mode: {mode}")


class MultiCoinTradingEnv:
    def __init__(
        self,
        coin_frames: dict[str, pd.DataFrame],
        max_slots: int,
        starting_cash: float = 1000.0,
        fee_rate: float = 0.005,
        fee_round_decimals: int = 1,
        supervised_targets: tuple[str, ...] | None = None,
        supervised_raw_targets: tuple[str, ...] = (),
        neutral_supervised_targets: tuple[str, ...] = (),
        expose_episode_boundary: bool = False,
        position_dynamics_mode: str = "disabled",
        drawdown_penalty: float = 0.0,
        forced_sell_penalty: float = 0.0,
        turnover_penalty: float = 0.0,
        sell_realization_reward_coef: float = 0.0,
        sell_aux_horizon_hours: int = 0,
        sell_aux_margin: float = 0.0,
        sell_aux_min_return: float | None = None,
        sell_aux_min_hold_hours: int = 0,
        sell_aux_early_stop_return: float = -0.03,
        sell_clear_aux_horizon_hours: int = 0,
        sell_clear_aux_hold_opportunity_min: float = 0.06,
        sell_clear_aux_avoid_loss_min: float = 0.03,
        sell_clear_aux_stop_return: float = -0.03,
        sell_clear_aux_exclude_conflict: bool = False,
        sell_margin_align_horizon_hours: int = 0,
        sell_margin_align_target_mode: str = "continuous",
        sell_margin_align_clear_hold_opportunity_min: float = 0.06,
        sell_margin_align_clear_avoid_loss_min: float = 0.03,
        sell_margin_align_clear_stop_return: float = -0.03,
        sell_margin_align_hold_weight: float = 1.0,
        sell_margin_align_scale: float = 0.06,
        sell_margin_align_exclude_conflict: bool = False,
        buy_aux_horizon_hours: int = 0,
        buy_aux_min_return: float = 0.0,
        buy_allocations: tuple[float, ...] = BUY_ALLOCATIONS,
        allocation_sizing_mode: str = "action_fraction",
        cross_sectional_supervised_context: bool = False,
        coin_history_context: bool = False,
        execution_price_mode: str = "current_close",
        execution_cost_bps: float = 0.0,
    ):
        if not coin_frames:
            raise ValueError("coin_frames is empty")
        self.coins = list(coin_frames)
        self.frames = {coin: frame.reset_index(drop=True) for coin, frame in coin_frames.items()}
        lengths = {len(frame) for frame in self.frames.values()}
        if len(lengths) != 1:
            raise ValueError("All coin frames must have equal length")
        timestamps = [pd.to_datetime(frame["timestamp"], utc=True).to_numpy() for frame in self.frames.values()]
        if any(not np.array_equal(timestamps[0], other) for other in timestamps[1:]):
            raise ValueError("All coin frames must share an identical timeline")
        self.total_steps = lengths.pop()
        self.starting_cash = float(starting_cash)
        if execution_price_mode not in {"current_close", "next_open"}:
            raise ValueError("execution_price_mode must be current_close or next_open")
        self.execution_price_mode = execution_price_mode
        self.execution_cost_bps = float(execution_cost_bps)
        if self.execution_cost_bps < 0.0:
            raise ValueError("execution_cost_bps must be nonnegative")
        self.drawdown_penalty = float(drawdown_penalty)
        self.forced_sell_penalty = float(forced_sell_penalty)
        self.turnover_penalty = float(turnover_penalty)
        self.sell_realization_reward_coef = float(sell_realization_reward_coef)
        self.sell_aux_horizon_hours = int(sell_aux_horizon_hours)
        self.sell_aux_margin = float(sell_aux_margin)
        self.sell_aux_min_return = (
            None if sell_aux_min_return is None else float(sell_aux_min_return)
        )
        self.sell_aux_min_hold_hours = int(sell_aux_min_hold_hours)
        self.sell_aux_early_stop_return = float(sell_aux_early_stop_return)
        self.sell_clear_aux_horizon_hours = int(sell_clear_aux_horizon_hours)
        self.sell_clear_aux_hold_opportunity_min = float(sell_clear_aux_hold_opportunity_min)
        self.sell_clear_aux_avoid_loss_min = float(sell_clear_aux_avoid_loss_min)
        self.sell_clear_aux_stop_return = float(sell_clear_aux_stop_return)
        self.sell_clear_aux_exclude_conflict = bool(sell_clear_aux_exclude_conflict)
        self.sell_margin_align_horizon_hours = int(sell_margin_align_horizon_hours)
        self.sell_margin_align_target_mode = str(sell_margin_align_target_mode)
        self.sell_margin_align_clear_hold_opportunity_min = float(
            sell_margin_align_clear_hold_opportunity_min
        )
        self.sell_margin_align_clear_avoid_loss_min = float(
            sell_margin_align_clear_avoid_loss_min
        )
        self.sell_margin_align_clear_stop_return = float(
            sell_margin_align_clear_stop_return
        )
        self.sell_margin_align_hold_weight = float(sell_margin_align_hold_weight)
        self.sell_margin_align_scale = float(sell_margin_align_scale)
        self.sell_margin_align_exclude_conflict = bool(sell_margin_align_exclude_conflict)
        self.buy_aux_horizon_hours = int(buy_aux_horizon_hours)
        self.buy_aux_min_return = float(buy_aux_min_return)
        self.buy_allocations = tuple(float(value) for value in buy_allocations)
        if (
            len(self.buy_allocations) != len(BUY_ALLOCATIONS)
            or any(value <= 0.0 or value > 1.0 for value in self.buy_allocations)
            or tuple(sorted(self.buy_allocations)) != self.buy_allocations
        ):
            raise ValueError("buy_allocations must contain four increasing values in (0,1]")
        if allocation_sizing_mode not in {"action_fraction", "equal_empty_slot"}:
            raise ValueError("allocation_sizing_mode must be action_fraction or equal_empty_slot")
        self.allocation_sizing_mode = allocation_sizing_mode
        self.expose_episode_boundary = bool(expose_episode_boundary)
        if position_dynamics_mode not in {
            "disabled", "neutral", "active", "neutral_fee", "active_fee",
            "neutral_memory", "active_memory", "neutral_memory_buy", "active_memory_buy",
        }:
            raise ValueError("invalid position_dynamics_mode")
        self.position_dynamics_mode = position_dynamics_mode
        if self.drawdown_penalty < 0.0:
            raise ValueError("drawdown_penalty must be nonnegative")
        if self.forced_sell_penalty < 0.0:
            raise ValueError("forced_sell_penalty must be nonnegative")
        if self.turnover_penalty < 0.0:
            raise ValueError("turnover_penalty must be nonnegative")
        if self.sell_realization_reward_coef < 0.0:
            raise ValueError("sell_realization_reward_coef must be nonnegative")
        if self.sell_aux_horizon_hours < 0:
            raise ValueError("sell_aux_horizon_hours must be nonnegative")
        if self.sell_aux_margin < 0.0:
            raise ValueError("sell_aux_margin must be nonnegative")
        if self.sell_aux_min_hold_hours < 0:
            raise ValueError("sell_aux_min_hold_hours must be nonnegative")
        if self.sell_clear_aux_horizon_hours < 0:
            raise ValueError("sell_clear_aux_horizon_hours must be nonnegative")
        if self.sell_clear_aux_hold_opportunity_min < 0.0:
            raise ValueError("sell_clear_aux_hold_opportunity_min must be nonnegative")
        if self.sell_clear_aux_avoid_loss_min < 0.0:
            raise ValueError("sell_clear_aux_avoid_loss_min must be nonnegative")
        if self.sell_margin_align_horizon_hours < 0:
            raise ValueError("sell_margin_align_horizon_hours must be nonnegative")
        if self.sell_margin_align_target_mode not in {"continuous", "clear_direction"}:
            raise ValueError("invalid sell_margin_align_target_mode")
        if self.sell_margin_align_clear_hold_opportunity_min < 0.0:
            raise ValueError("sell_margin_align_clear_hold_opportunity_min must be nonnegative")
        if self.sell_margin_align_clear_avoid_loss_min < 0.0:
            raise ValueError("sell_margin_align_clear_avoid_loss_min must be nonnegative")
        if self.sell_margin_align_hold_weight < 0.0:
            raise ValueError("sell_margin_align_hold_weight must be nonnegative")
        if self.sell_margin_align_scale <= 0.0:
            raise ValueError("sell_margin_align_scale must be positive")
        if self.buy_aux_horizon_hours < 0:
            raise ValueError("buy_aux_horizon_hours must be nonnegative")
        self.portfolio = MultiCoinPortfolio(
            starting_cash, fee_rate, fee_round_decimals, max_slots=max_slots
        )
        self.sub_cfg = SubFeatureConfig(episode_len_steps=self.total_steps)
        self.coin_history_context = bool(coin_history_context)
        self.feature_columns = self._feature_columns(
            next(iter(self.frames.values())), supervised_targets, supervised_raw_targets,
            coin_history_context=self.coin_history_context,
        )
        self.cross_sectional_supervised_context = bool(cross_sectional_supervised_context)
        self.cross_context_columns = []
        self.cross_score_indices: tuple[int, ...] = ()
        if self.cross_sectional_supervised_context:
            targets = () if supervised_targets is None else supervised_targets
            columns = [f"sup_{target}_pct" for target in targets]
            unavailable = [column for column in columns if column not in self.feature_columns]
            if unavailable:
                raise ValueError("Cross-sectional context is missing scores: " + ", ".join(unavailable))
            if not columns:
                raise ValueError("Cross-sectional context requires supervised percentile columns")
            self.cross_score_indices = tuple(self.feature_columns.index(column) for column in columns)
            self.cross_context_columns = [
                f"cross_{column}_relative_rank" for column in columns
            ]
        self.static_arrays = {
            coin: frame[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
            for coin, frame in self.frames.items()
        }
        for target in neutral_supervised_targets:
            column = f"sup_{target}_pct"
            if column not in self.feature_columns:
                raise ValueError(f"Cannot neutralize unavailable supervised score: {column}")
            column_index = self.feature_columns.index(column)
            for array in self.static_arrays.values():
                # Cross-sectional percentile context is neutral at the median.
                array[:, column_index] = 0.5
        self.price_arrays = {
            coin: frame["close"].to_numpy(dtype=np.float64, copy=True)
            for coin, frame in self.frames.items()
        }
        self.open_arrays = {
            coin: frame["open"].to_numpy(dtype=np.float64, copy=True)
            for coin, frame in self.frames.items()
        }
        self.current_step = 0
        self.steps_since_last_sell = 0

    @staticmethod
    def _feature_columns(
        frame: pd.DataFrame, supervised_targets: tuple[str, ...] | None = None,
        supervised_raw_targets: tuple[str, ...] = (),
        coin_history_context: bool = False,
    ) -> list[str]:
        excluded = {"timestamp", "open", "high", "low", "close", "volume", "year"}
        columns = []
        allowed = None if supervised_targets is None else {
            f"sup_{target}_pct" for target in supervised_targets
        }
        allowed_raw = {f"sup_{target}" for target in supervised_raw_targets}
        for column in frame.columns:
            if column in excluded:
                continue
            if column in COIN_HISTORY_CONTEXT_COLUMNS and not coin_history_context:
                continue
            if column.startswith("sup_"):
                allowed_percentile = column.endswith("_pct") and (
                    allowed is None or column in allowed
                )
                if not (allowed_percentile or column in allowed_raw):
                    continue
            columns.append(column)
        return columns

    @property
    def input_dim(self) -> int:
        # Existing 12 per-coin state fields plus global cash availability.
        dynamics = len(position_dynamics_columns(self.position_dynamics_mode))
        return len(self.feature_columns) + len(self.cross_context_columns) + 13 + dynamics

    def _cross_sectional_context(self) -> dict[str, np.ndarray]:
        """Causal same-timestamp relative score ranks for visible assets only."""
        if not self.cross_sectional_supervised_context:
            return {}
        values = np.stack([
            self.static_arrays[coin][self.current_step, self.cross_score_indices]
            for coin in self.coins
        ]).astype(np.float32, copy=False)
        ranks = np.zeros_like(values, dtype=np.float32)
        if len(self.coins) > 1:
            for column in range(values.shape[1]):
                order = np.argsort(values[:, column], kind="stable")
                start = 0
                while start < len(order):
                    stop = start + 1
                    while stop < len(order) and values[order[stop], column] == values[order[start], column]:
                        stop += 1
                    rank = 2.0 * ((start + stop - 1) / 2.0) / (len(order) - 1) - 1.0
                    ranks[order[start:stop], column] = rank
                    start = stop
        return {coin: ranks[index] for index, coin in enumerate(self.coins)}

    def _position_dynamics(self, coin: str, position, current_return_raw: float) -> np.ndarray:
        if self.position_dynamics_mode == "disabled":
            return np.empty(0, dtype=np.float32)
        if self.position_dynamics_mode in {
            "neutral", "neutral_fee", "neutral_memory", "neutral_memory_buy",
        }:
            return np.zeros(
                len(position_dynamics_columns(self.position_dynamics_mode)),
                dtype=np.float32,
            )
        if position is None:
            if self.position_dynamics_mode == "active_memory_buy":
                sell_memory_width = (
                    len(POSITION_DYNAMICS_COLUMNS)
                    + len(FEE_POSITION_DYNAMICS_COLUMNS)
                    + len(POSITION_MEMORY_COLUMNS)
                )
                return np.concatenate([
                    np.zeros(sell_memory_width, dtype=np.float32),
                    self._buy_memory(coin),
                ])
            return np.zeros(
                len(position_dynamics_columns(self.position_dynamics_mode)),
                dtype=np.float32,
            )
        effective_mfe = max(position.mfe, current_return_raw)
        effective_mae = min(position.mae, current_return_raw)
        mfe_step = self.current_step if current_return_raw > position.mfe else position.mfe_step
        mae_step = self.current_step if current_return_raw < position.mae else position.mae_step
        values = [
            float(np.tanh((current_return_raw - effective_mfe) / 0.25)),
            float(np.tanh((current_return_raw - effective_mae) / 0.25)),
            min(1.0, max(0, self.current_step - mfe_step) / 720.0),
            min(1.0, max(0, self.current_step - mae_step) / 720.0),
        ]
        prices = self.price_arrays[coin]
        age = self.current_step - position.entry_step
        for horizon in (1, 4, 24):
            if age < horizon:
                values.append(0.0)
            else:
                past_return = prices[self.current_step - horizon] / position.entry_price - 1.0
                values.append(float(np.tanh((current_return_raw - past_return) / 0.10)))
        if self.position_dynamics_mode in {"active_fee", "active_memory", "active_memory_buy"}:
            notional = position.size * prices[self.current_step]
            exit_fee = round(
                notional * self.portfolio.fee_rate,
                self.portfolio.fee_round_decimals,
            )
            realizable_return = (
                (notional - exit_fee) / max(position.cost_basis, 1e-12) - 1.0
            )
            fee_drag = current_return_raw - realizable_return
            values.extend([
                float(np.tanh(realizable_return / 0.25)),
                float(np.tanh(fee_drag / 0.02)),
            ])
            if self.position_dynamics_mode in {"active_memory", "active_memory_buy"}:
                if age < 1:
                    pnl_change_1h = 0.0
                else:
                    previous_return = prices[self.current_step - 1] / position.entry_price - 1.0
                    pnl_change_1h = current_return_raw - previous_return
                values.extend([
                    float(np.tanh(realizable_return / 0.25)),
                    float(np.tanh(pnl_change_1h / 0.10)),
                    float(np.tanh(effective_mfe / 0.25)),
                    float(np.tanh(effective_mae / 0.25)),
                    float(np.tanh((current_return_raw - effective_mfe) / 0.25)),
                ])
        if self.position_dynamics_mode == "active_memory_buy":
            values.extend(self._buy_memory(coin).tolist())
        return np.asarray(values, dtype=np.float32)

    def _buy_memory(self, coin: str) -> np.ndarray:
        prices = self.price_arrays[coin]
        current_price = float(prices[self.current_step])
        values: list[float] = []
        for _, window in BUY_MEMORY_WINDOWS:
            start = max(0, self.current_step - int(window) + 1)
            history = prices[start : self.current_step + 1]
            if len(history) == 0 or current_price <= 0.0:
                values.extend([0.0, 0.0])
                continue
            low = float(np.min(history))
            high = float(np.max(history))
            rebound = current_price / max(low, 1e-12) - 1.0
            drawdown = current_price / max(high, 1e-12) - 1.0
            values.extend([
                float(np.tanh(rebound / 0.25)),
                float(np.tanh(drawdown / 0.25)),
            ])
        return np.asarray(values, dtype=np.float32)

    def reset(
        self,
        initial_position_allocation: float | None = None,
        initial_position_coin: str | None = None,
    ) -> torch.Tensor:
        """Reset flat or into a valid pre-existing Phase-1 holding state.

        The optional holding is curriculum state initialization only: it does
        not execute an AI trade, add a normal sell rule, or affect evaluation
        unless explicitly requested by a trainer.  Its already-paid entry cost
        is excluded from the new episode reward baseline.
        """
        self.portfolio.reset()
        self.current_step = 0
        self.steps_since_last_sell = 0
        self.previous_equity = self.starting_cash
        self.peak_equity = self.starting_cash
        self.previous_drawdown = 0.0
        self.manual_sell_count = 0
        self.forced_sell_count = 0
        self.trade_pnls: list[float] = []
        self.trade_returns: list[float] = []
        # Kept one-for-one with trade_pnls/trade_returns for diagnostics only.
        # It does not enter an observation, reward, legality decision, or
        # action selection, so evaluation can audit learned sizing separately
        # from forced terminal bookkeeping.
        self.trade_allocations: list[float] = []
        self.trade_forced: list[bool] = []
        self.trade_details: list[dict[str, Any]] = []
        self.initial_position_opened = False
        if initial_position_allocation is not None:
            allocation = float(initial_position_allocation)
            if not 0.0 < allocation <= 1.0:
                raise ValueError("Initial holding curriculum requires allocation in (0, 1]")
            coin = self.coins[0] if initial_position_coin is None else initial_position_coin
            if coin not in self.coins:
                raise ValueError("Initial holding coin must be visible in the episode")
            if not self.portfolio.buy(coin, self._prices()[coin], self.current_step, allocation):
                raise RuntimeError("Could not initialize valid curriculum holding")
            # The state represents an already-open position: neither its entry
            # fee nor its initial mark should be assigned to the first action.
            self.previous_equity = self.portfolio.equity(self._prices())
            self.peak_equity = self.previous_equity
            self.initial_position_opened = True
        return self.observation()

    def _prices(self) -> dict[str, float]:
        return {coin: float(self.price_arrays[coin][self.current_step]) for coin in self.coins}

    def _buy_allocation(self, allocation_index: int) -> float:
        if self.allocation_sizing_mode == "equal_empty_slot":
            empty_slots = self.portfolio.max_slots - len(self.portfolio.positions)
            if empty_slots <= 0:
                return 0.0
            return min(1.0, 1.0 / float(empty_slots))
        return self.buy_allocations[allocation_index]

    def _execution_prices(self, side: str) -> dict[str, float]:
        if self.execution_price_mode == "next_open":
            step = min(self.current_step + 1, self.total_steps - 1)
            raw = {coin: float(self.open_arrays[coin][step]) for coin in self.coins}
        else:
            raw = self._prices()
        if self.execution_cost_bps <= 0.0:
            return raw
        multiplier = (
            1.0 + self.execution_cost_bps / 10000.0
            if side == "buy" else 1.0 - self.execution_cost_bps / 10000.0
        )
        return {coin: price * multiplier for coin, price in raw.items()}

    def _timestamp_iso(self, coin: str, step: int) -> str:
        safe_step = min(max(int(step), 0), len(self.frames[coin]) - 1)
        return pd.Timestamp(self.frames[coin]["timestamp"].iloc[safe_step]).isoformat()

    def _position_net_return(self, position, price: float) -> float:
        notional = position.size * float(price)
        exit_fee = round(
            notional * self.portfolio.fee_rate,
            self.portfolio.fee_round_decimals,
        )
        return float((notional - exit_fee) / max(position.cost_basis, 1e-12) - 1.0)

    def _round_trip_return(self, entry_price: float, exit_price: float) -> float:
        """After-fee return for a hypothetical fixed-budget round trip.

        This is used only to build training/evaluation diagnostics.  It never
        executes a trade, filters the universe, or changes action legality.
        """
        if entry_price <= 0.0 or exit_price <= 0.0:
            return float("-inf")
        budget = self.starting_cash
        estimated_notional = budget / (1.0 + self.portfolio.fee_rate)
        entry_fee = round(
            estimated_notional * self.portfolio.fee_rate,
            self.portfolio.fee_round_decimals,
        )
        notional = max(0.0, budget - entry_fee)
        if notional <= 0.0:
            return float("-inf")
        size = notional / float(entry_price)
        exit_notional = size * float(exit_price)
        exit_fee = round(
            exit_notional * self.portfolio.fee_rate,
            self.portfolio.fee_round_decimals,
        )
        return float((exit_notional - exit_fee) / budget - 1.0)

    def buy_auxiliary_target(self) -> tuple[int | None, bool, float | None]:
        """Training-only flat-state buy/no-buy label.

        Returns the best future-opportunity coin index when a coin clears the
        configured after-fee return floor, or ``-1`` for NOOP when no visible
        coin clears it.  Future prices are used only as train-period labels;
        the environment never uses this method to execute, block, or rank
        actions at inference/evaluation time.
        """
        horizon = self.buy_aux_horizon_hours
        if horizon <= 0 or self.portfolio.positions:
            return None, False, None
        if self.current_step + horizon >= self.total_steps:
            return None, False, None
        best_index = -1
        best_return = float("-inf")
        for index, coin in enumerate(self.coins):
            prices = self.price_arrays[coin]
            entry_price = float(prices[self.current_step])
            future_best = float(
                np.max(prices[self.current_step + 1 : self.current_step + horizon + 1])
            )
            candidate_return = self._round_trip_return(entry_price, future_best)
            if candidate_return > best_return:
                best_return = candidate_return
                best_index = index
        if best_index < 0 or not np.isfinite(best_return):
            return -1, True, None
        if best_return >= self.buy_aux_min_return:
            return int(best_index), True, float(best_return)
        return -1, True, float(best_return)

    def sell_auxiliary_target(self) -> tuple[float | None, bool]:
        """Training-only causal exit label for the current held state.

        The label is available only when the configured future horizon fits
        inside the current training window. It is never used by the environment
        to execute a sell, alter action legality, filter coins, or evaluate a
        checkpoint.  A positive SELL label means exiting now is within
        ``sell_aux_margin`` of the best realizable after-fee return available
        over the next horizon; otherwise HOLD remains the target.
        """
        horizon = self.sell_aux_horizon_hours
        if horizon <= 0 or not self.portfolio.positions:
            return None, False
        if self.current_step + horizon >= self.total_steps:
            return None, False
        # Phase-1 diagnostics are one-slot.  For future multi-slot use this
        # remains well-defined by labeling the first held position only, while
        # PPO applies it to the held coin selected by the held mask.
        coin, position = next(iter(self.portfolio.positions.items()))
        prices = self.price_arrays[coin]
        current_return = self._position_net_return(position, prices[self.current_step])
        holding_steps = self.current_step - position.entry_step
        if (
            self.sell_aux_min_hold_hours > 0
            and holding_steps < self.sell_aux_min_hold_hours
        ):
            return float(current_return <= self.sell_aux_early_stop_return), True
        future = prices[self.current_step + 1 : self.current_step + horizon + 1]
        if len(future) == 0:
            return None, False
        future_best = max(self._position_net_return(position, price) for price in future)
        near_horizon_best = future_best <= current_return + self.sell_aux_margin
        meets_return_floor = (
            True if self.sell_aux_min_return is None
            else current_return >= self.sell_aux_min_return
        )
        label_sell = float(near_horizon_best and meets_return_floor)
        return label_sell, True

    def sell_clear_auxiliary_target(self) -> tuple[float | None, bool]:
        """Training-only clear-state sell/hold label for discriminative exits.

        Ambiguous held states are masked out.  A HOLD label is emitted only
        when the future training-window path still has substantial best-hold
        opportunity.  A SELL label is emitted only when there is no such
        opportunity and selling now avoids a meaningful near-future loss, or
        the current after-fee return is already below the configured stop
        threshold.  The label never executes trades, changes action legality,
        filters coins, or appears during evaluation.
        """
        horizon = self.sell_clear_aux_horizon_hours
        if horizon <= 0 or not self.portfolio.positions:
            return None, False
        if self.current_step + horizon >= self.total_steps:
            return None, False
        coin, position = next(iter(self.portfolio.positions.items()))
        prices = self.price_arrays[coin]
        current_return = self._position_net_return(position, prices[self.current_step])
        future = prices[self.current_step + 1 : self.current_step + horizon + 1]
        if len(future) == 0:
            return None, False
        future_returns = [
            self._position_net_return(position, price) for price in future
        ]
        best_hold_opportunity = max(future_returns) - current_return
        avoided_loss = current_return - min(future_returns)
        avoid_or_stop = (
            current_return <= self.sell_clear_aux_stop_return
            or avoided_loss >= self.sell_clear_aux_avoid_loss_min
        )
        if (
            self.sell_clear_aux_exclude_conflict
            and best_hold_opportunity >= self.sell_clear_aux_hold_opportunity_min
            and avoid_or_stop
        ):
            return None, False
        if best_hold_opportunity >= self.sell_clear_aux_hold_opportunity_min:
            return 0.0, True
        if avoid_or_stop:
            return 1.0, True
        return None, False

    def sell_margin_alignment_details(self) -> dict[str, float | bool | None]:
        """Continuous training-only target for SELL-HOLD margin alignment.

        Positive values mean selling now avoids more future downside than the
        best available hold opportunity; negative values mean the opposite.
        The target is diagnostic/training context only.  It never executes a
        trade, changes action legality, filters coins, ranks the universe, or
        appears during evaluation.
        """
        horizon = self.sell_margin_align_horizon_hours
        if horizon <= 0 or not self.portfolio.positions:
            return {"target": None, "mask": False}
        if self.current_step + horizon >= self.total_steps:
            return {"target": None, "mask": False}
        coin, position = next(iter(self.portfolio.positions.items()))
        prices = self.price_arrays[coin]
        current_return = self._position_net_return(position, prices[self.current_step])
        future = prices[self.current_step + 1 : self.current_step + horizon + 1]
        if len(future) == 0:
            return {"target": None, "mask": False}
        future_returns = [
            self._position_net_return(position, price) for price in future
        ]
        best_hold_opportunity = max(future_returns) - current_return
        avoided_loss = current_return - min(future_returns)
        if self.sell_margin_align_target_mode == "clear_direction":
            high_opportunity = (
                best_hold_opportunity >= self.sell_margin_align_clear_hold_opportunity_min
            )
            avoid_or_stop = (
                current_return <= self.sell_margin_align_clear_stop_return
                or avoided_loss >= self.sell_margin_align_clear_avoid_loss_min
            )
            if (
                self.sell_margin_align_exclude_conflict
                and high_opportunity
                and avoid_or_stop
            ):
                return {
                    "target": None,
                    "mask": False,
                    "best_hold_opportunity": float(best_hold_opportunity),
                    "avoided_loss": float(avoided_loss),
                    "current_return": float(current_return),
                }
            if high_opportunity:
                raw = -best_hold_opportunity
            elif avoid_or_stop:
                raw = max(avoided_loss, -current_return, 0.0)
            else:
                return {
                    "target": None,
                    "mask": False,
                    "best_hold_opportunity": float(best_hold_opportunity),
                    "avoided_loss": float(avoided_loss),
                    "current_return": float(current_return),
                }
        else:
            raw = avoided_loss - self.sell_margin_align_hold_weight * best_hold_opportunity
        return {
            "target": float(np.tanh(raw / self.sell_margin_align_scale)),
            "mask": True,
            "best_hold_opportunity": float(best_hold_opportunity),
            "avoided_loss": float(avoided_loss),
            "current_return": float(current_return),
        }

    def sell_margin_alignment_target(self) -> tuple[float | None, bool]:
        details = self.sell_margin_alignment_details()
        return details["target"], bool(details["mask"])

    def held_mask(self) -> torch.Tensor:
        return torch.tensor([coin in self.portfolio.positions for coin in self.coins], dtype=torch.bool)

    def has_future_after_buy_entry(self) -> bool:
        if self.execution_price_mode == "next_open":
            return self.current_step + 1 < self.total_steps - 1
        return self.current_step < self.total_steps - 1

    def observation(self) -> torch.Tensor:
        rows = []
        steps_left = self.total_steps - self.current_step
        prices = self._prices()
        cross_context = self._cross_sectional_context()
        cash_fraction = self.portfolio.cash / max(self.portfolio.equity(prices), 1e-12)
        buy_has_future = self.has_future_after_buy_entry()
        for index, coin in enumerate(self.coins):
            position = self.portfolio.positions.get(coin)
            has_position = position is not None
            price = prices[coin]
            current_return_raw = price / position.entry_price - 1.0 if position else 0.0
            # Stable, monotonic state encoding: 25% maps to tanh(1), while
            # extreme crypto moves cannot dominate normalized market features.
            current_return = float(np.tanh(current_return_raw / 0.25))
            sub, _ = build_sub_features(
                has_position=has_position,
                can_buy=self.portfolio.can_buy(coin) and buy_has_future,
                can_sell=has_position,
                entry_step=position.entry_step if position else 0,
                current_step=self.current_step,
                steps_since_last_sell=self.steps_since_last_sell,
                return_since_entry=current_return,
                mfe=float(np.tanh(position.mfe / 0.25)) if position else 0.0,
                mae=float(np.tanh(position.mae / 0.25)) if position else 0.0,
                scan_progress=index / max(1, len(self.coins) - 1),
                auto_sell_flag=(
                    float(steps_left <= 6) if self.expose_episode_boundary else 0.0
                ),
                time_remaining=(
                    steps_left / max(1, self.total_steps)
                    if self.expose_episode_boundary else 1.0
                ),
                position_allocation_fraction=position.allocation_fraction if position else 0.0,
                cfg=self.sub_cfg,
            )
            static = self.static_arrays[coin][self.current_step]
            rows.append(np.concatenate([
                static,
                cross_context.get(coin, np.empty(0, dtype=np.float32)),
                np.asarray(sub, dtype=np.float32),
                self._position_dynamics(coin, position, current_return_raw),
                np.asarray([cash_fraction], dtype=np.float32),
            ]))
        return torch.from_numpy(np.stack(rows))

    def step(
        self, model_output: dict[str, torch.Tensor], deterministic: bool = False,
        action_temperature: float = 1.0,
        buy_prob_threshold: float = 0.0,
    ):
        prices = self._prices()
        sell_aux_label, sell_aux_mask = self.sell_auxiliary_target()
        sell_clear_aux_label, sell_clear_aux_mask = self.sell_clear_auxiliary_target()
        sell_margin_align_details = self.sell_margin_alignment_details()
        sell_margin_align_target = sell_margin_align_details["target"]
        sell_margin_align_mask = bool(sell_margin_align_details["mask"])
        buy_aux_target, buy_aux_mask, buy_aux_best_return = self.buy_auxiliary_target()
        held_position_age = 0
        held_position_return = 0.0
        if self.portfolio.positions:
            held_coin, held_position = next(iter(self.portfolio.positions.items()))
            held_position_age = int(self.current_step - held_position.entry_step)
            held_position_return = self._position_net_return(
                held_position, prices[held_coin]
            )
        allow_buy = (
            len(self.portfolio.positions) < self.portfolio.max_slots
            and self.portfolio.cash > 0.0
            and self.has_future_after_buy_entry()
        )
        candidate, action_index, log_prob, confidence = select_joint_action(
            model_output["buy_logits"], model_output["sell_logits"], self.held_mask(),
            allow_buy=allow_buy, deterministic=deterministic,
            mode_logits=model_output.get("mode_logits"),
            temperature=action_temperature,
        )
        action = "noop"
        coin = None
        trade_closed = False
        trade_pnl = 0.0
        trade_return = None
        allocation_fraction = None
        holding_steps = None
        if candidate.kind == "buy":
            if float(confidence) < float(buy_prob_threshold):
                coin = None
            else:
                coin = self.coins[candidate.coin_index]
                allocation = self._buy_allocation(candidate.allocation_index)
                execution_prices = self._execution_prices("buy")
                execution_step = (
                    min(self.current_step + 1, self.total_steps - 1)
                    if self.execution_price_mode == "next_open" else self.current_step
                )
                if self.portfolio.buy(coin, execution_prices[coin], execution_step, allocation):
                    action = "buy"
                    allocation_fraction = float(allocation)
        elif candidate.kind == "sell":
            coin = self.coins[candidate.coin_index]
            position = self.portfolio.positions[coin]
            allocation_fraction = float(position.allocation_fraction)
            holding_steps = int(self.current_step - position.entry_step)
            cost_basis = float(position.cost_basis)
            execution_prices = self._execution_prices("sell")
            execution_step = (
                min(self.current_step + 1, self.total_steps - 1)
                if self.execution_price_mode == "next_open" else self.current_step
            )
            entry_step = int(position.entry_step)
            entry_price = float(position.entry_price)
            exit_price = float(execution_prices[coin])
            mfe = float(position.mfe)
            mae = float(position.mae)
            mfe_step = int(position.mfe_step)
            mae_step = int(position.mae_step)
            trade_pnl = self.portfolio.sell(coin, execution_prices[coin])
            self.trade_pnls.append(trade_pnl)
            trade_return = trade_pnl / max(cost_basis, 1e-12)
            self.trade_returns.append(float(trade_return))
            self.trade_allocations.append(allocation_fraction)
            self.trade_forced.append(False)
            self.trade_details.append({
                "coin": coin,
                "forced": False,
                "entry_step": entry_step,
                "entry_time": self._timestamp_iso(coin, entry_step),
                "entry_price": entry_price,
                "exit_step": execution_step,
                "exit_time": self._timestamp_iso(coin, execution_step),
                "exit_price": exit_price,
                "holding_steps": holding_steps,
                "pnl": float(trade_pnl),
                "return_after_fees": float(trade_return),
                "allocation_fraction": allocation_fraction,
                "cost_basis": cost_basis,
                "mfe": mfe,
                "mae": mae,
                "mfe_step": mfe_step,
                "mfe_time": self._timestamp_iso(coin, mfe_step),
                "mae_step": mae_step,
                "mae_time": self._timestamp_iso(coin, mae_step),
            })
            self.manual_sell_count += 1
            self.steps_since_last_sell = 0
            trade_closed = True
            action = "sell"
        if action != "sell":
            self.steps_since_last_sell += 1

        if self.execution_price_mode == "next_open":
            next_step = min(self.current_step + 1, self.total_steps - 1)
            self.current_step = next_step
            mark_prices = self._prices()
            is_time_limit = self.current_step >= self.total_steps - 1
        else:
            mark_prices = prices
            is_time_limit = self.current_step + 1 >= self.total_steps
        self.portfolio.update(mark_prices, step=self.current_step)
        # A fixed data window is a time-limit truncation for training, not an
        # economic reason to sell.  Preserve the post-action state and
        # mark-to-market equity before mandatory bookkeeping liquidation so a
        # trainer can bootstrap its critic without crediting a hidden terminal
        # sale to the preceding hold/noop action.  Normal environment behavior
        # and evaluator accounting still liquidate safely below.
        pre_forced_equity = self.portfolio.equity(mark_prices)
        time_limit_observation = self.observation().clone() if is_time_limit else None
        if self.execution_price_mode != "next_open":
            self.current_step += 1
        done = is_time_limit
        terminal = []
        if done and self.portfolio.positions:
            terminal_positions = {
                coin: position
                for coin, position in self.portfolio.positions.items()
            }
            terminal_cost_basis = {
                coin: float(position.cost_basis)
                for coin, position in self.portfolio.positions.items()
            }
            terminal_allocation = {
                coin: float(position.allocation_fraction)
                for coin, position in self.portfolio.positions.items()
            }
            terminal_prices = mark_prices
            if self.execution_cost_bps > 0.0:
                terminal_prices = {
                    coin: price * (1.0 - self.execution_cost_bps / 10000.0)
                    for coin, price in terminal_prices.items()
                }
            terminal = self.portfolio.liquidate(terminal_prices)
            self.forced_sell_count += len(terminal)
            self.trade_pnls.extend(pnl for _, pnl in terminal)
            terminal_returns = [
                float(pnl / max(terminal_cost_basis[coin], 1e-12))
                for coin, pnl in terminal
            ]
            self.trade_returns.extend(terminal_returns)
            self.trade_allocations.extend(
                terminal_allocation[coin] for coin, _ in terminal
            )
            self.trade_forced.extend(True for _ in terminal)
            for (coin, pnl), trade_return in zip(terminal, terminal_returns):
                position = terminal_positions[coin]
                entry_step = int(position.entry_step)
                exit_step = int(self.current_step)
                mfe_step = int(position.mfe_step)
                mae_step = int(position.mae_step)
                self.trade_details.append({
                    "coin": coin,
                    "forced": True,
                    "entry_step": entry_step,
                    "entry_time": self._timestamp_iso(coin, entry_step),
                    "entry_price": float(position.entry_price),
                    "exit_step": exit_step,
                    "exit_time": self._timestamp_iso(coin, exit_step),
                    "exit_price": float(terminal_prices[coin]),
                    "holding_steps": int(exit_step - entry_step),
                    "pnl": float(pnl),
                    "return_after_fees": float(trade_return),
                    "allocation_fraction": float(terminal_allocation[coin]),
                    "cost_basis": float(terminal_cost_basis[coin]),
                    "mfe": float(position.mfe),
                    "mae": float(position.mae),
                    "mfe_step": mfe_step,
                    "mfe_time": self._timestamp_iso(coin, mfe_step),
                    "mae_step": mae_step,
                    "mae_time": self._timestamp_iso(coin, mae_step),
                })
        equity = self.portfolio.equity(mark_prices)
        raw_equity_reward = (equity - self.previous_equity) / self.starting_cash
        time_limit_raw_equity_reward = (
            pre_forced_equity - self.previous_equity
        ) / self.starting_cash
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (equity - self.peak_equity) / max(self.peak_equity, 1e-12)
        drawdown_increase = max(0.0, self.previous_drawdown - drawdown)
        terminal_forced_penalty = self.forced_sell_penalty * len(terminal)
        trade_turnover_penalty = self.turnover_penalty * float(action in {"buy", "sell"})
        sell_realization_reward = (
            self.sell_realization_reward_coef * float(trade_return)
            if action == "sell" and trade_return is not None else 0.0
        )
        reward = (
            raw_equity_reward - self.drawdown_penalty * drawdown_increase
            - terminal_forced_penalty - trade_turnover_penalty
            + sell_realization_reward
        )
        self.previous_equity = equity
        self.previous_drawdown = drawdown
        info: dict[str, Any] = {
            "action": action,
            "coin": coin,
            "joint_action_index": action_index,
            "logp": log_prob.detach(),
            "confidence": confidence,
            "equity": equity,
            "raw_equity_reward": raw_equity_reward,
            "drawdown": drawdown,
            "drawdown_penalty_applied": self.drawdown_penalty * drawdown_increase,
            "forced_sell_penalty_applied": terminal_forced_penalty,
            "turnover_penalty_applied": trade_turnover_penalty,
            "sell_realization_reward_applied": sell_realization_reward,
            "trade_closed": trade_closed,
            "trade_pnl_dollars": trade_pnl,
            "trade_return_after_fees": trade_return,
            "allocation_fraction": allocation_fraction,
            "holding_steps": holding_steps,
            "held_position_age": held_position_age,
            "held_position_return_after_fees": float(held_position_return),
            "manual_sell_count": self.manual_sell_count,
            "forced_sell_count": self.forced_sell_count,
            "terminal_liquidations": terminal,
            "time_limit_truncated": done,
            "time_limit_observation": time_limit_observation if done else None,
            "time_limit_raw_equity_reward": float(time_limit_raw_equity_reward),
            "sell_aux_label": sell_aux_label,
            "sell_aux_mask": sell_aux_mask,
            "sell_clear_aux_label": sell_clear_aux_label,
            "sell_clear_aux_mask": sell_clear_aux_mask,
            "sell_margin_align_target": sell_margin_align_target,
            "sell_margin_align_mask": sell_margin_align_mask,
            "sell_margin_align_best_hold_opportunity": sell_margin_align_details.get(
                "best_hold_opportunity", 0.0,
            ),
            "sell_margin_align_avoided_loss": sell_margin_align_details.get(
                "avoided_loss", 0.0,
            ),
            "sell_margin_align_current_return": sell_margin_align_details.get(
                "current_return", 0.0,
            ),
            "buy_aux_target": buy_aux_target,
            "buy_aux_mask": buy_aux_mask,
            "buy_aux_best_return": buy_aux_best_return,
        }
        return None if done else self.observation(), float(reward), done, info
