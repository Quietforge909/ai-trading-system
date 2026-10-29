# rl/env/sub_features.py

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SubFeatureConfig:
    """
    Phase-agnostic config for sub-features.
    In Phase 1, slot_limit=1; later phases can extend.
    """
    episode_len_steps: int = 24 * 30  # 30 days * 24 hours
    # Normalization caps to keep values bounded and stable
    max_hold_steps: int = 24 * 30
    max_steps_since_sell: int = 24 * 30


# Fixed order: must never change once PPO starts training
SUB_FEATURE_COLUMNS: List[str] = [
    "in_held",
    "allow_buy",
    "allow_sell",
    "position_allocation_fraction",
    "position_age_normalized",
    "steps_since_last_sell_normalized",
    "return_since_entry",
    "max_unrealized_return",
    "min_unrealized_return",
    "scan_progress_normalized",
    "auto_sell_flag",
    "time_remaining",
]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def build_sub_features(
    *,
    has_position: bool,
    can_buy: bool,
    can_sell: bool,
    entry_step: int,
    current_step: int,
    steps_since_last_sell: int,
    return_since_entry: float,
    mfe: float,
    mae: float,
    scan_progress: float,
    auto_sell_flag: float,
    time_remaining: float,
    position_allocation_fraction: float,
    cfg: SubFeatureConfig,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Returns:
      - vector (list[float]) in fixed SUB_FEATURE_COLUMNS order
      - debug dict {name: value}
    """

    # --- legality flags
    in_held = 1.0 if has_position else 0.0
    allow_buy = 1.0 if can_buy else 0.0
    allow_sell = 1.0 if can_sell else 0.0
    position_allocation_fraction = _clamp01(float(position_allocation_fraction))

    # --- normalized time features
    if has_position:
        pos_age = max(0, current_step - entry_step)
    else:
        pos_age = 0

    position_age_normalized = _clamp01(pos_age / float(max(1, cfg.max_hold_steps)))
    steps_since_last_sell_normalized = _clamp01(
        steps_since_last_sell / float(max(1, cfg.max_steps_since_sell))
    )

    # --- return metrics (already bounded-ish, but still stable)
    # return_since_entry can be negative/positive; keep as-is.
    # mfe and mae are returns, too.
    max_unrealized_return = float(mfe)
    min_unrealized_return = float(mae)

    # --- scan progress (Phase 1 has only one coin, but keep field for future)
    scan_progress_normalized = _clamp01(float(scan_progress))

    out = {
        "in_held": in_held,
        "allow_buy": allow_buy,
        "allow_sell": allow_sell,
        "position_allocation_fraction": position_allocation_fraction,
        "position_age_normalized": position_age_normalized,
        "steps_since_last_sell_normalized": steps_since_last_sell_normalized,
        "return_since_entry": float(return_since_entry),
        "auto_sell_flag": float(auto_sell_flag),
        "time_remaining": float(time_remaining),
        "max_unrealized_return": max_unrealized_return,
        "min_unrealized_return": min_unrealized_return,
        "scan_progress_normalized": scan_progress_normalized,
    }

    vec = [out[name] for name in SUB_FEATURE_COLUMNS]
    return vec, out
