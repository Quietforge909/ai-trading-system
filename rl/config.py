STARTING_CASH = 1000.0
FEE_RATE = 0.005
FEE_ROUND_DECIMALS = 1

TRAIN_START = "2020-01-01"
TRAIN_END   = "2025-12-30"
TEST_START  = "2026-01-01"

EPISODE_DAYS = 30
HOURS_PER_STEP = 1

# 88 bounded/static features plus 12 portfolio features.  Raw supervised
# predictions and the calendar year are intentionally excluded in the
# environment; the corresponding cross-sectional percentile features are kept.
INPUT_DIM = 100  # frozen for profit-v3 checkpoints

# One conditional policy head is active at each step.  Allocation is part of
# the buy action itself so PPO can learn it through the sampled action log-prob.
BUY_ALLOCATIONS = (0.25, 0.50, 0.75, 1.00)
BUY_ACTION_DIM = len(BUY_ALLOCATIONS) + 1  # final action is SKIP
SELL_ACTION_DIM = 2                        # SELL, HOLD
