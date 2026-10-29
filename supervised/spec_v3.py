"""Gap-safe supervised context specification with fee-aware targets."""

from supervised.spec_v2 import CLS_TARGETS as V2_CLS_TARGETS
from supervised.spec_v2 import REG_TARGETS as V2_REG_TARGETS


CLS_TARGETS = V2_CLS_TARGETS + (
    "profitable_after_fee_24h",
    "profitable_after_fee_72h",
    "profitable_after_fee_168h",
    "quality_upside_7d",
    "bad_rebound_risk_7d",
)

REG_TARGETS = V2_REG_TARGETS + (
    "future_net_return_24h",
    "future_net_return_72h",
    "future_net_return_168h",
    "safe_future_max_return_7d",
)

MAX_HORIZON_HOURS = 30 * 24
