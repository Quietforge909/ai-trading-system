"""Leakage-safe supervised context specification, version 2."""

CLS_TARGETS = (
    "opportunity_7d",
    "opportunity_14d",
    "opportunity_30d",
    "drawdown_risk_7d",
    "vol_expansion_risk_7d",
)

REG_TARGETS = (
    "future_max_return_7d",
    "future_min_return_7d",
    "future_vol_7d",
    "future_terminal_return_7d",
    "trend_efficiency_7d",
)

MAX_HORIZON_HOURS = 30 * 24

