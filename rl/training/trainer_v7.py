"""Reproducible trainer for the v7 joint multi-coin PPO policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from rl.agent.joint_ppo import JointEpisodeBuffer, JointPPOAgent, JointPPOConfig
from rl.agent.model import PPOTradingModel
from rl.config import FEE_RATE, FEE_ROUND_DECIMALS, STARTING_CASH
from rl.env.multi_env import MultiCoinTradingEnv
from rl.env.multi_env import position_dynamics_columns
from rl.env.multi_selector import MixedHorizonCoinSelector, MultiCoinSelector, YearBalancedCoinSelector
from rl.training.legacy_resume_compat import (
    APPROVED_NEUTRAL_RESUME_CURRENT_SOURCE_SHA256,
    LEGACY_NEUTRAL_RESUME_SOURCE_SHA256,
)
from rl.training.v41_full_universe_resume_compat import (
    V41_FULL_UNIVERSE_CURRENT_NEUTRAL_SOURCE_SHA256,
    V41_FULL_UNIVERSE_SOURCE_SHA256,
    is_v41_full_universe_neutral_config,
)


SPEC_VERSION = 7
TEMPORAL_SPEC_VERSION = 8
FACTORIZED_SPEC_VERSION = 9
ACTION_VALUE_SPEC_VERSION = 11
SELL_RAW_RESIDUAL_SPEC_VERSION = 12
ACTION_VALUE_TD_SPEC_VERSION = 13
BUY_RAW_RESIDUAL_SPEC_VERSION = 14
BUY_ALLOCATION_Q_SPEC_VERSION = 15
SUPERVISED_SCORE_RESIDUAL_SPEC_VERSION = 16
CROSS_SECTIONAL_SUPERVISED_CONTEXT_SPEC_VERSION = 17
UNIVERSE_CONTEXT_SPEC_VERSION = 18
MODE_HEAD_SPEC_VERSION = 19
SELL_AUX_SPEC_VERSION = 20
SELL_TARGET_AUX_SPEC_VERSION = 21
SELL_TARGET_POLICY_RESIDUAL_SPEC_VERSION = 22

# Invocation-only options may change when a compatible run is resumed.  Every
# other training option is part of the experiment identity and must match the
# checkpoint; silently changing one would make the resulting artifact neither
# a continuation nor a controlled new run.
RESUME_MUTABLE_CONFIG_KEYS = {
    "device", "episodes", "resume", "run_dir", "prerequisite_gate",
    "allow_same_phase_parent",
    # Weight-only initialization is consumed before a run starts.  A resumed
    # checkpoint is its own initialization source, so retaining the original
    # provenance must not make a faithful continuation fail validation.
    "init_checkpoint",
}

# Fields introduced after the first v7 checkpoints.  Missing values in a
# legacy checkpoint are compatible only with the neutral behavior that was in
# effect before the feature existed.  Do not use these defaults to forgive a
# saved non-neutral value.
LEGACY_NEUTRAL_CONFIG_DEFAULTS = {
    "action_value_heads": False,
    "flat_action_value_heads": False,
    "sell_q_detach_encoder": False,
    "sell_raw_residual": False,
    "sell_raw_residual_width": 0,
    "buy_raw_residual": False,
    "buy_raw_residual_width": 0,
    "score_residual": False,
    "score_residual_width": 0,
    "score_feature_indices": (),
    "cross_sectional_supervised_context": False,
    "coin_history_context": False,
    "universe_context_width": 0,
    "mode_head": False,
    "boundary_sell_residual": False,
    "boundary_sell_feature_index": None,
    "sell_q_coef": 0.0,
    "sell_q_advantage_mix": 0.0,
    "sell_q_target": "mc",
    "sell_q_policy_coef": 0.0,
    "buy_q_coef": 0.0,
    "buy_q_advantage_mix": 0.0,
    "buy_allocation_q_coef": 0.0,
    "buy_allocation_q_advantage_mix": 0.0,
    "sell_aux_coef": 0.0,
    "sell_aux_positive_weight": 1.0,
    "sell_aux_horizon_hours": 0,
    "sell_aux_margin": 0.0,
    "sell_aux_min_return": None,
    "sell_aux_min_hold_hours": 0,
    "sell_aux_early_stop_return": -0.03,
    "sell_clear_aux_coef": 0.0,
    "sell_clear_aux_positive_weight": 1.0,
    "sell_clear_aux_exclude_conflict": False,
    "sell_clear_contrast_coef": 0.0,
    "sell_clear_contrast_margin": 0.25,
    "sell_clear_margin_floor_coef": 0.0,
    "sell_clear_margin_floor": -0.25,
    "sell_clear_margin_floor_max_return": None,
    "sell_clear_hold_margin_ceiling_coef": 0.0,
    "sell_clear_hold_margin_ceiling": -0.75,
    "sell_margin_align_coef": 0.0,
    "sell_margin_level_coef": 0.0,
    "sell_margin_level_scale": 1.0,
    "sell_margin_level_hold_opportunity_min": 0.0,
    "sell_margin_level_hold_weight": 1.0,
    "sell_margin_level_sell_avoid_loss_min": 0.0,
    "sell_margin_level_sell_weight": 1.0,
    "sell_margin_align_horizon_hours": 0,
    "sell_margin_align_target_mode": "continuous",
    "sell_margin_align_clear_hold_opportunity_min": 0.06,
    "sell_margin_align_clear_avoid_loss_min": 0.03,
    "sell_margin_align_clear_stop_return": -0.03,
    "sell_margin_align_hold_weight": 1.0,
    "sell_margin_align_scale": 0.06,
    "sell_margin_align_exclude_conflict": False,
    "sell_margin_conflict_hold_opportunity_min": 0.06,
    "sell_margin_conflict_avoid_loss_min": 0.03,
    "sell_margin_conflict_stop_return": -0.03,
    "sell_margin_conflict_uncertainty_coef": 0.0,
    "sell_margin_conflict_uncertainty_scale": 1.0,
    "sell_margin_subtype_confidence_coef": 0.0,
    "sell_target_aux_width": 0,
    "sell_target_policy_residual": False,
    "sell_target_policy_residual_width": 0,
    "sell_target_policy_residual_source": "score",
    "sell_target_aux_coef": 0.0,
    "train_sell_target_aux_only": False,
    "train_sell_target_policy_residual_only": False,
    "sell_clear_aux_horizon_hours": 0,
    "sell_clear_aux_hold_opportunity_min": 0.06,
    "sell_clear_aux_avoid_loss_min": 0.03,
    "sell_clear_aux_stop_return": -0.03,
    "sell_clear_aux_exclude_conflict": False,
    "buy_aux_coef": 0.0,
    "buy_aux_rank_only": False,
    "buy_aux_negative_coef": 0.0,
    "buy_aux_negative_buy_mass_target": 0.0,
    "buy_aux_negative_gate_feature": "none",
    "buy_aux_negative_gate_feature_index": -1,
    "buy_aux_negative_gate_min": float("-inf"),
    "buy_aux_negative_gate_max": float("inf"),
    "buy_aux_horizon_hours": 0,
    "buy_aux_min_return": 0.0,
    "failed_to_sell_margin_coef": 0.0,
    "failed_to_sell_margin_target": 0.0,
    "failed_to_sell_logit_margin_coef": 0.0,
    "failed_to_sell_logit_margin_target": 0.0,
    "failed_to_sell_hindsight_hours": 0,
    "failed_to_sell_min_hold_hours": 0,
    "turnover_penalty": 0.0,
    "sell_realization_reward_coef": 0.0,
    "held_policy_loss_weight": 1.0,
    "train_sell_head_only": False,
    "train_sell_residual_only": False,
    "train_buy_residual_only": False,
    "train_score_residual_only": False,
    "train_boundary_sell_residual_only": False,
    "score_residual_neutral_base": False,
    "buy_allocations": "0.25,0.50,0.75,1.00",
}
LEGACY_NEUTRAL_PPO_DEFAULTS = {
    "sell_q_coef": 0.0,
    "sell_q_advantage_mix": 0.0,
    "sell_q_target": "mc",
    "sell_q_policy_coef": 0.0,
    "buy_q_coef": 0.0,
    "buy_q_advantage_mix": 0.0,
    "buy_allocation_q_coef": 0.0,
    "buy_allocation_q_advantage_mix": 0.0,
    "sell_aux_coef": 0.0,
    "sell_aux_positive_weight": 1.0,
    "sell_clear_aux_coef": 0.0,
    "sell_clear_aux_positive_weight": 1.0,
    "sell_clear_contrast_coef": 0.0,
    "sell_clear_contrast_margin": 0.25,
    "sell_clear_margin_floor_coef": 0.0,
    "sell_clear_margin_floor": -0.25,
    "sell_clear_margin_floor_max_return": None,
    "sell_clear_hold_margin_ceiling_coef": 0.0,
    "sell_clear_hold_margin_ceiling": -0.75,
    "sell_margin_align_coef": 0.0,
    "sell_margin_align_exclude_conflict": False,
    "sell_margin_conflict_hold_opportunity_min": 0.06,
    "sell_margin_conflict_avoid_loss_min": 0.03,
    "sell_margin_conflict_stop_return": -0.03,
    "sell_margin_conflict_uncertainty_coef": 0.0,
    "sell_margin_conflict_uncertainty_scale": 1.0,
    "sell_margin_subtype_confidence_coef": 0.0,
    "sell_target_aux_coef": 0.0,
    "buy_aux_coef": 0.0,
    "buy_aux_rank_only": False,
    "buy_aux_negative_coef": 0.0,
    "buy_aux_negative_buy_mass_target": 0.0,
    "buy_aux_negative_gate_feature": "none",
    "buy_aux_negative_gate_feature_index": -1,
    "buy_aux_negative_gate_min": float("-inf"),
    "buy_aux_negative_gate_max": float("inf"),
    "failed_to_sell_margin_coef": 0.0,
    "failed_to_sell_margin_target": 0.0,
    "failed_to_sell_logit_margin_coef": 0.0,
    "failed_to_sell_logit_margin_target": 0.0,
    "failed_to_sell_hindsight_hours": 0,
    "failed_to_sell_min_hold_hours": 0,
    "held_policy_loss_weight": 1.0,
    "boundary_sell_residual": False,
    "boundary_sell_feature_index": None,
    "train_boundary_sell_residual_only": False,
}
RESUME_DERIVED_CONFIG_KEYS = {"trainable_parameters"}
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase1_shape_is_allowed(
    universe_size: int, max_slots: int, multicoin_diagnostic: bool,
) -> bool:
    """Keep Phase 1 single-slot while permitting an explicit full universe.

    ``universe_size=0`` is the selector's established spelling for every
    timestamp-aligned valid coin, never an empty universe. It remains a
    diagnostic only and does not relax the Phase-2 prerequisite gate.
    """
    if universe_size == 1 and max_slots == 1:
        return True
    return (
        multicoin_diagnostic
        and max_slots == 1
        and (universe_size > 1 or universe_size == 0)
    )


def save_checkpoint(
    path: Path, episode: int, model, optimizer, selector, config, metrics,
    training_coverage: dict,
) -> None:
    payload = {
        "spec_version": int(config.get("spec_version", SPEC_VERSION)),
        "episode": episode,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "selector_state": selector.get_state(),
        "config": config,
        "training_coverage": training_coverage,
        "last_episode_metrics": metrics,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def validate_resume_config(saved: dict, current: dict) -> None:
    """Reject a resume that changes experiment-defining configuration."""
    ignored = RESUME_MUTABLE_CONFIG_KEYS | {
        "source_sha256",  # checked separately with a precise error below
    } | RESUME_DERIVED_CONFIG_KEYS
    saved = dict(saved)
    current = dict(current)
    saved_legacy_missing = any(
        key not in saved for key in LEGACY_NEUTRAL_CONFIG_DEFAULTS
    )
    for key, default in LEGACY_NEUTRAL_CONFIG_DEFAULTS.items():
        saved.setdefault(key, default)
        current.setdefault(key, default)
    saved_ppo = dict(saved.get("ppo", {}))
    current_ppo = dict(current.get("ppo", {}))
    for key, default in LEGACY_NEUTRAL_PPO_DEFAULTS.items():
        saved_ppo.setdefault(key, default)
        current_ppo.setdefault(key, default)
    saved["ppo"], current["ppo"] = saved_ppo, current_ppo
    mismatches = {
        key: {"checkpoint": saved.get(key), "requested": current.get(key)}
        for key in sorted((set(saved) | set(current)) - ignored)
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Resume configuration mismatch: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )
    saved_sources = saved.get("source_sha256")
    current_sources = current.get("source_sha256")
    if saved_sources is not None and saved_sources != current_sources:
        changed_sources = {
            path for path in set(saved_sources) | set(current_sources)
            if saved_sources.get(path) != current_sources.get(path)
        }
        if (
            saved_legacy_missing
            and saved_sources == LEGACY_NEUTRAL_RESUME_SOURCE_SHA256
            and current_sources == APPROVED_NEUTRAL_RESUME_CURRENT_SOURCE_SHA256
        ):
            # Record the current hashes in the next checkpoint through normal
            # save logic. A different legacy bundle or any new tracked path is
            # still rejected below.
            return
        if (
            saved_sources == V41_FULL_UNIVERSE_SOURCE_SHA256
            and current_sources == V41_FULL_UNIVERSE_CURRENT_NEUTRAL_SOURCE_SHA256
            and is_v41_full_universe_neutral_config(saved)
            and is_v41_full_universe_neutral_config(current)
        ):
            # This exact source transition was replayed from immutable v41
            # episode 34 through 39 before enabling the bridge.  It is not a
            # generic hash bypass: any changed policy path, feature option,
            # environment mode, or later tracked-source edit falls through.
            return
        raise RuntimeError(
            "Training source changed since the checkpoint; start a new run or "
            "use --init-checkpoint for a controlled weights-only initialization"
        )


def write_config_once(path: Path, config: dict) -> None:
    serialized = json.dumps(config, indent=2) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise RuntimeError(f"Refusing to overwrite different experiment metadata: {path}")
        return
    path.write_text(serialized)


def expand_input_checkpoint(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
    source_columns: list[str], target_columns: list[str],
) -> None:
    """Load a compatible model while zero-initializing only new input columns."""
    if model.temporal_width != 0 or model.actor_mode != "shared":
        raise RuntimeError("Input expansion currently requires a feed-forward shared actor")
    if len(set(source_columns)) != len(source_columns) or len(set(target_columns)) != len(target_columns):
        raise RuntimeError("Observation columns must be unique for input expansion")
    missing = set(source_columns) - set(target_columns)
    if missing:
        raise RuntimeError(f"Input expansion removed source columns: {sorted(missing)}")
    target_state = model.state_dict()
    if set(source_state) != set(target_state):
        raise RuntimeError("Input expansion architecture keys differ")
    source_index = {column: index for index, column in enumerate(source_columns)}
    for key, target in target_state.items():
        source = source_state[key]
        if (
            key == "fc_in.weight"
            or (
                key.endswith(".weight")
                and source.ndim == 2
                and target.ndim == 2
                and source.shape[1] == len(source_columns)
                and target.shape[1] == len(target_columns)
            )
        ):
            if source.shape[0] != target.shape[0] or source.shape[1] != len(source_columns):
                raise RuntimeError(f"Source input layer does not match its observation schema: {key}")
            if target.shape[1] != len(target_columns):
                raise RuntimeError(f"Target input layer does not match its observation schema: {key}")
            expanded = torch.zeros_like(target)
            for target_column, column in enumerate(target_columns):
                if column in source_index:
                    expanded[:, target_column] = source[:, source_index[column]]
            target_state[key] = expanded
        elif source.shape == target.shape:
            target_state[key] = source
        else:
            raise RuntimeError(f"Input expansion shape mismatch: {key}")
    model.load_state_dict(target_state)


def initialize_zero_sell_residual(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Load a legacy policy while adding only a neutral sell residual."""
    result = model.load_state_dict(source_state, strict=False)
    expected_missing = {
        key for key in model.state_dict() if key.startswith("sell_raw_residual.")
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            "Sell-residual initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    probe = torch.zeros((2, model.input_dim), device=next(model.parameters()).device)
    if not torch.all(model.sell_raw_residual(probe) == 0.0):
        raise RuntimeError("New sell residual did not initialize neutrally")


def configure_residual_only_training(model: PPOTradingModel) -> None:
    """Freeze the proven policy and train its sell residual/optional critic."""
    if not model.sell_raw_residual_enabled:
        raise ValueError("Residual-only training requires --sell-raw-residual")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.sell_raw_residual.parameters():
        parameter.requires_grad_(True)
    if model.action_value_heads:
        for parameter in model.sell_q_head.parameters():
            parameter.requires_grad_(True)


def initialize_zero_action_value_head(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Add an exactly neutral action critic to a proven policy checkpoint."""
    result = model.load_state_dict(source_state, strict=False)
    expected = {"sell_q_head.weight", "sell_q_head.bias"}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Action-value initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if not torch.all(model.sell_q_head.weight == 0.0) or not torch.all(
        model.sell_q_head.bias == 0.0
    ):
        raise RuntimeError("New action-value head did not initialize neutrally")


def initialize_zero_flat_action_value_head(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Add an exactly neutral flat-state allocation critic to a proven policy."""
    result = model.load_state_dict(source_state, strict=False)
    expected = {"buy_allocation_q_head.weight", "buy_allocation_q_head.bias"}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Flat action-value initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if not torch.all(model.buy_allocation_q_head.weight == 0.0) or not torch.all(
        model.buy_allocation_q_head.bias == 0.0
    ):
        raise RuntimeError("New flat action-value head did not initialize neutrally")


def initialize_zero_action_value_and_sell_residual(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Add both neutral sell-learning paths without changing policy outputs."""
    result = model.load_state_dict(source_state, strict=False)
    expected = {"sell_q_head.weight", "sell_q_head.bias"} | {
        key for key in model.state_dict() if key.startswith("sell_raw_residual.")
    }
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Combined sell expansion architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    probe = torch.zeros((2, model.input_dim), device=next(model.parameters()).device)
    if not torch.all(model.sell_raw_residual(probe) == 0.0):
        raise RuntimeError("Combined sell residual did not initialize neutrally")
    if not torch.all(model.sell_q_head.weight == 0.0) or not torch.all(
        model.sell_q_head.bias == 0.0
    ):
        raise RuntimeError("Combined action-value head did not initialize neutrally")


def configure_sell_head_only_training(model: PPOTradingModel) -> None:
    """Preserve entries/trunk and train only AI sell policy plus its critic."""
    if not model.action_value_heads:
        raise ValueError("Sell-head-only training requires Sell-Q heads")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.sell_adapter, model.sell_head, model.sell_q_head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def configure_sell_policy_only_training(model: PPOTradingModel) -> None:
    """Fine-tune only the existing learned sell policy, without a Q head."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [model.sell_adapter, model.sell_head]
    if model.sell_encoder is not None:
        modules.append(model.sell_encoder)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def configure_sell_target_aux_only_training(model: PPOTradingModel) -> None:
    """Train only the detached sell-target predictor, never policy logits."""
    if not model.sell_target_aux_enabled or model.sell_target_aux_head is None:
        raise ValueError("Sell-target auxiliary-only training requires --sell-target-aux-width")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.sell_target_aux_head.parameters():
        parameter.requires_grad_(True)


def configure_sell_target_policy_residual_only_training(model: PPOTradingModel) -> None:
    """Train only the learned aux-to-policy sell-margin connection."""
    if (
        not model.sell_target_policy_residual_enabled
        or model.sell_target_policy_residual is None
    ):
        raise ValueError("Policy residual-only training requires --sell-target-policy-residual")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.sell_target_policy_residual.parameters():
        parameter.requires_grad_(True)


def configure_mode_head_only_training(model: PPOTradingModel) -> None:
    """Train only the portfolio-level NOOP/BUY/SELL mode selector."""
    if not model.mode_head_enabled or model.mode_head is None:
        raise ValueError("Mode-head-only training requires --mode-head")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.mode_head.parameters():
        parameter.requires_grad_(True)


def configure_boundary_sell_residual_only_training(model: PPOTradingModel) -> None:
    """Train only the boundary-gated sell-margin scalar."""
    if (
        not model.boundary_sell_residual_enabled
        or model.boundary_sell_logit_scale is None
    ):
        raise ValueError("Boundary sell residual-only training requires --boundary-sell-residual")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.boundary_sell_logit_scale.requires_grad_(True)


def configure_gated_buy_residual_only_training(model: PPOTradingModel) -> None:
    """Train only the feature-gated buy/noop scalar."""
    if (
        not model.gated_buy_residual_enabled
        or model.gated_buy_logit_scale is None
    ):
        raise ValueError("Gated buy residual-only training requires --gated-buy-residual")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gated_buy_logit_scale.requires_grad_(True)


def configure_gated_buy2_residual_only_training(model: PPOTradingModel) -> None:
    """Train only the second feature-gated buy/noop scalar."""
    if (
        not model.gated_buy2_residual_enabled
        or model.gated_buy2_logit_scale is None
    ):
        raise ValueError("Second gated buy residual-only training requires --gated-buy2-residual")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gated_buy2_logit_scale.requires_grad_(True)


def configure_gated_buy3_residual_only_training(model: PPOTradingModel) -> None:
    """Train only the third feature-gated buy/noop scalar."""
    if (
        not model.gated_buy3_residual_enabled
        or model.gated_buy3_logit_scale is None
    ):
        raise ValueError("Third gated buy residual-only training requires --gated-buy3-residual")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gated_buy3_logit_scale.requires_grad_(True)


def configure_gated_sell_residual_only_training(model: PPOTradingModel) -> None:
    """Train only the feature-gated sell/hold scalar."""
    if (
        not model.gated_sell_residual_enabled
        or model.gated_sell_logit_scale is None
    ):
        raise ValueError("Gated sell residual-only training requires --gated-sell-residual")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gated_sell_logit_scale.requires_grad_(True)


def configure_score_residual_only_training(model: PPOTradingModel) -> None:
    """Train only the supervised-score residual path."""
    if not model.score_residual_enabled:
        raise ValueError("Score-residual-only training requires --score-residual")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.score_residual.parameters():
        parameter.requires_grad_(True)


def initialize_separate_actor_from_shared(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Expand a shared-actor checkpoint to separate buy/sell encoders neutrally."""
    if model.actor_mode != "separate" or model.buy_encoder is None or model.sell_encoder is None:
        raise ValueError("Separate actor initialization requires actor_mode='separate'")
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if key in target and target[key].shape == value.shape:
            mapped[key] = value
    for prefix in ("buy_encoder", "sell_encoder"):
        for source_prefix, target_prefix in (
            ("fc_in", f"{prefix}.fc_in"),
            ("ln_in", f"{prefix}.ln_in"),
            ("fc_mid", f"{prefix}.fc_mid"),
            ("ln_mid", f"{prefix}.ln_mid"),
            ("res_blocks", f"{prefix}.res_blocks"),
            ("fc_out", f"{prefix}.fc_out"),
        ):
            for key, value in source_state.items():
                if key == source_prefix or key.startswith(source_prefix + "."):
                    mapped_key = target_prefix + key[len(source_prefix):]
                    if mapped_key in target and target[mapped_key].shape == value.shape:
                        mapped[mapped_key] = value
    required = {
        key for key in target
        if key.startswith("buy_encoder.") or key.startswith("sell_encoder.")
    }
    missing = sorted(required - set(mapped))
    if missing:
        raise RuntimeError(
            "Separate actor initialization missing compatible keys: "
            + ",".join(missing[:12])
        )
    result = model.load_state_dict(mapped, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected separate actor initialization keys: {result.unexpected_keys}"
        )


def initialize_sell_target_aux_head(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Load a checkpoint while adding only a detached sell-target predictor."""
    if not model.sell_target_aux_enabled:
        raise ValueError("Sell-target auxiliary initialization requires --sell-target-aux-width")
    result = model.load_state_dict(source_state, strict=False)
    expected = {
        key for key in model.state_dict() if key.startswith("sell_target_aux_head.")
    }
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Sell-target auxiliary initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def initialize_sell_target_policy_residual(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Load an aux checkpoint while adding a neutral sell-margin connection."""
    if not model.sell_target_policy_residual_enabled:
        raise ValueError("Sell-target policy residual initialization requires --sell-target-policy-residual")
    result = model.load_state_dict(source_state, strict=False)
    expected = {
        key for key in model.state_dict()
        if key.startswith("sell_target_policy_residual.")
    }
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Sell-target policy residual initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    final_weight = (
        model.sell_target_policy_residual[-1].weight
        if model.sell_target_policy_residual_width > 0
        else model.sell_target_policy_residual.weight
    )
    if not torch.all(final_weight == 0.0):
        raise RuntimeError("New sell-target policy residual did not initialize neutrally")


def initialize_zero_buy_residual(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Add a neutral learned buy-context path to a proven checkpoint."""
    result = model.load_state_dict(source_state, strict=False)
    expected = {
        key for key in model.state_dict() if key.startswith("buy_raw_residual.")
    }
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Buy-residual initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    probe = torch.zeros((2, model.input_dim), device=next(model.parameters()).device)
    if not torch.all(model.buy_raw_residual(probe) == 0.0):
        raise RuntimeError("New buy residual did not initialize neutrally")


def initialize_zero_score_residual(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Load a checkpoint while adding only a neutral supervised-score path."""
    if not model.score_residual_enabled:
        raise ValueError("Score-residual initialization requires --score-residual")
    result = model.load_state_dict(source_state, strict=False)
    expected = {
        key for key in model.state_dict() if key.startswith("score_residual.")
    }
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Score-residual initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if not torch.all(model.score_residual[-1].weight == 0.0):
        raise RuntimeError("New score residual did not initialize neutrally")


def initialize_zero_boundary_sell_residual(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Load a checkpoint while adding only a neutral boundary sell scalar."""
    if not model.boundary_sell_residual_enabled:
        raise ValueError("Boundary sell residual initialization requires --boundary-sell-residual")
    result = model.load_state_dict(source_state, strict=False)
    expected = {"boundary_sell_logit_scale"}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Boundary sell residual initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if not torch.all(model.boundary_sell_logit_scale == 0.0):
        raise RuntimeError("New boundary sell residual did not initialize neutrally")


def initialize_zero_gated_buy_residual(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Load a checkpoint while adding only a neutral gated buy scalar."""
    if not model.gated_buy_residual_enabled:
        raise ValueError("Gated buy residual initialization requires --gated-buy-residual")
    result = model.load_state_dict(source_state, strict=False)
    expected = {"gated_buy_logit_scale"}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Gated buy residual initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if not torch.all(model.gated_buy_logit_scale == 0.0):
        raise RuntimeError("New gated buy residual did not initialize neutrally")


def initialize_zero_gated_buy2_residual(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Load a checkpoint while adding only a second neutral gated buy scalar."""
    if not model.gated_buy2_residual_enabled:
        raise ValueError("Second gated buy residual initialization requires --gated-buy2-residual")
    result = model.load_state_dict(source_state, strict=False)
    expected = {"gated_buy2_logit_scale"}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Second gated buy residual initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if not torch.all(model.gated_buy2_logit_scale == 0.0):
        raise RuntimeError("New second gated buy residual did not initialize neutrally")


def initialize_zero_gated_buy3_residual(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Load a checkpoint while adding only a third neutral gated buy scalar."""
    if not model.gated_buy3_residual_enabled:
        raise ValueError("Third gated buy residual initialization requires --gated-buy3-residual")
    result = model.load_state_dict(source_state, strict=False)
    expected = {"gated_buy3_logit_scale"}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Third gated buy residual initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if not torch.all(model.gated_buy3_logit_scale == 0.0):
        raise RuntimeError("New third gated buy residual did not initialize neutrally")


def initialize_zero_gated_sell_residual(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
) -> None:
    """Load a checkpoint while adding only a neutral gated sell scalar."""
    if not model.gated_sell_residual_enabled:
        raise ValueError("Gated sell residual initialization requires --gated-sell-residual")
    result = model.load_state_dict(source_state, strict=False)
    expected = {"gated_sell_logit_scale"}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Gated sell residual initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if not torch.all(model.gated_sell_logit_scale == 0.0):
        raise RuntimeError("New gated sell residual did not initialize neutrally")


def init_checkpoint_spec_is_compatible(payload: dict, source_config: dict) -> bool:
    """Return whether a weights-only init checkpoint has a known compatible spec.

    ``--init-checkpoint`` can add one neutral experimental path at a time.  The
    original guard only accepted base spec-7 checkpoints, which was correct for
    first expansion but accidentally blocked safe chained residual experiments
    such as buy-residual -> buy+sell-residual.  This remains intentionally
    narrow: only base v7, action-value-stage, and residual-stage checkpoints
    whose saved config proves the corresponding experimental path exists are
    accepted here.  Some chained diagnostics add a neutral path on top of an
    action-value checkpoint while retaining spec 11 in the saved payload; the
    subsequent architecture/key checks still verify that the requested model is
    a real superset before any weights are loaded.
    """
    spec_version = int(payload.get("spec_version", -1))
    if spec_version == SPEC_VERSION:
        return True
    source_has_sell_residual = bool(source_config.get("sell_raw_residual", False))
    source_has_buy_residual = bool(source_config.get("buy_raw_residual", False))
    source_has_score_residual = bool(source_config.get("score_residual", False)) or bool(
        source_config.get("score_residual_enabled", False)
    )
    source_has_action_value = bool(source_config.get("action_value_heads", False))
    source_has_sell_aux = float(source_config.get("sell_aux_coef", 0.0) or 0.0) > 0.0
    source_has_sell_clear_aux = (
        float(source_config.get("sell_clear_aux_coef", 0.0) or 0.0) > 0.0
    )
    source_has_sell_target_aux = int(source_config.get("sell_target_aux_width", 0) or 0) > 0
    source_has_sell_target_policy_residual = bool(
        source_config.get(
            "sell_target_policy_residual",
            source_config.get("sell_target_policy_residual_enabled", False),
        )
    )
    if spec_version == ACTION_VALUE_SPEC_VERSION:
        return source_has_action_value
    if spec_version == SELL_RAW_RESIDUAL_SPEC_VERSION:
        return source_has_sell_residual
    if spec_version == BUY_RAW_RESIDUAL_SPEC_VERSION:
        return source_has_buy_residual
    if spec_version == SELL_AUX_SPEC_VERSION:
        return source_has_sell_aux or source_has_sell_clear_aux
    if spec_version == SELL_TARGET_AUX_SPEC_VERSION:
        return source_has_sell_target_aux
    if spec_version == SELL_TARGET_POLICY_RESIDUAL_SPEC_VERSION:
        return source_has_sell_target_policy_residual
    if spec_version == SUPERVISED_SCORE_RESIDUAL_SPEC_VERSION:
        return source_has_score_residual
    return False


def initialize_zero_mode_head(
    model: PPOTradingModel, source_state: dict[str, torch.Tensor],
    initial_buy_prob: float,
    initial_sell_prob: float,
) -> None:
    """Load a legacy policy while adding an explicit mode head.

    The added head starts with zero logits.  This is architecture-compatible
    but not policy-output neutral because hierarchical mode probabilities are
    intentionally a different action parameterization.
    """
    if not model.mode_head_enabled:
        raise ValueError("Mode-head initialization requires --mode-head")
    result = model.load_state_dict(source_state, strict=False)
    expected = {"mode_head.weight", "mode_head.bias"}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            "Mode-head initialization architecture mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    with torch.no_grad():
        model.mode_head.weight.zero_()
        model.mode_head.bias[0].fill_(float(np.log(1.0 - initial_buy_prob)))
        model.mode_head.bias[1].fill_(float(np.log(initial_buy_prob)))
        model.mode_head.bias[2].fill_(float(np.log(initial_sell_prob)))


def configure_buy_residual_only_training(model: PPOTradingModel) -> None:
    """Freeze the frontier and train only its learned buy/size context path."""
    if not model.buy_raw_residual_enabled:
        raise ValueError("Buy-residual-only training requires --buy-raw-residual")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.buy_raw_residual.parameters():
        parameter.requires_grad_(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("train_ready/features/final_causal_v2_supervised6"))
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--prerequisite-gate", type=Path)
    parser.add_argument(
        "--allow-same-phase-parent", action="store_true",
        help=(
            "Allow a passed same-phase Phase 3 checkpoint to initialize another "
            "Phase 3 run for slot-curriculum continuation"
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--episode-days", type=int, default=30)
    parser.add_argument(
        "--episode-days-schedule",
        help="Optional comma-separated hidden training horizons, e.g. 14,30,60",
    )
    parser.add_argument("--universe-size", type=int, default=8)
    parser.add_argument("--max-slots", type=int, default=4)
    parser.add_argument(
        "--buy-allocations", default="0.25,0.50,0.75,1.00",
        help="Four increasing AI-selectable cash fractions",
    )
    parser.add_argument(
        "--allocation-sizing-mode",
        choices=("action_fraction", "equal_empty_slot"),
        default="action_fraction",
        help="How BUY actions are sized. equal_empty_slot uses cash divided by remaining empty slots.",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--sell-head-width", type=int, default=0,
        help="Optional compact hidden adapter dedicated to sell/hold decisions",
    )
    parser.add_argument("--temporal-width", type=int, default=0)
    parser.add_argument("--actor-mode", choices=("shared", "separate"), default="shared")
    parser.add_argument("--state-value-heads", action="store_true")
    parser.add_argument(
        "--sell-raw-residual", action="store_true",
        help="Add a zero-initialized 104-weight learned sell-context path",
    )
    parser.add_argument(
        "--sell-raw-residual-width", type=int, default=0,
        help="Optional compact nonlinear width for the raw sell residual",
    )
    parser.add_argument(
        "--train-sell-residual-only", action="store_true",
        help="Freeze a proven policy and optimize only its learned sell residual",
    )
    parser.add_argument("--buy-raw-residual", action="store_true")
    parser.add_argument("--buy-raw-residual-width", type=int, default=0)
    parser.add_argument("--train-buy-residual-only", action="store_true")
    parser.add_argument(
        "--score-residual", action="store_true",
        help="Zero-initialized learned residual from selected supervised context into policy features",
    )
    parser.add_argument("--score-residual-width", type=int, default=16)
    parser.add_argument("--train-score-residual-only", action="store_true")
    parser.add_argument("--boundary-sell-residual", action="store_true")
    parser.add_argument("--train-boundary-sell-residual-only", action="store_true")
    parser.add_argument("--gated-buy-residual", action="store_true")
    parser.add_argument("--train-gated-buy-residual-only", action="store_true")
    parser.add_argument(
        "--gated-buy-feature", default="none",
        help="Observation column used to gate the learned BUY-vs-NOOP scalar.",
    )
    parser.add_argument(
        "--gated-buy-feature-min", type=float, default=float("-inf"),
        help="Minimum feature value for applying the gated buy residual.",
    )
    parser.add_argument(
        "--gated-buy-feature-max", type=float, default=float("inf"),
        help="Maximum feature value for applying the gated buy residual.",
    )
    parser.add_argument(
        "--gated-buy-feature2", default="none",
        help="Optional second observation column AND-gated with --gated-buy-feature.",
    )
    parser.add_argument(
        "--gated-buy-feature2-min", type=float, default=float("-inf"),
        help="Minimum second feature value for applying the gated buy residual.",
    )
    parser.add_argument(
        "--gated-buy-feature2-max", type=float, default=float("inf"),
        help="Maximum second feature value for applying the gated buy residual.",
    )
    parser.add_argument("--gated-buy2-residual", action="store_true")
    parser.add_argument("--train-gated-buy2-residual-only", action="store_true")
    parser.add_argument("--gated-buy2-feature", default="none")
    parser.add_argument("--gated-buy2-feature-min", type=float, default=float("-inf"))
    parser.add_argument("--gated-buy2-feature-max", type=float, default=float("inf"))
    parser.add_argument("--gated-buy2-feature2", default="none")
    parser.add_argument("--gated-buy2-feature2-min", type=float, default=float("-inf"))
    parser.add_argument("--gated-buy2-feature2-max", type=float, default=float("inf"))
    parser.add_argument("--gated-buy3-residual", action="store_true")
    parser.add_argument("--train-gated-buy3-residual-only", action="store_true")
    parser.add_argument("--gated-buy3-feature", default="none")
    parser.add_argument("--gated-buy3-feature-min", type=float, default=float("-inf"))
    parser.add_argument("--gated-buy3-feature-max", type=float, default=float("inf"))
    parser.add_argument("--gated-buy3-feature2", default="none")
    parser.add_argument("--gated-buy3-feature2-min", type=float, default=float("-inf"))
    parser.add_argument("--gated-buy3-feature2-max", type=float, default=float("inf"))
    parser.add_argument("--gated-sell-residual", action="store_true")
    parser.add_argument("--train-gated-sell-residual-only", action="store_true")
    parser.add_argument("--gated-sell-feature", default="none")
    parser.add_argument("--gated-sell-feature-min", type=float, default=float("-inf"))
    parser.add_argument("--gated-sell-feature-max", type=float, default=float("inf"))
    parser.add_argument("--gated-sell-feature2", default="none")
    parser.add_argument("--gated-sell-feature2-min", type=float, default=float("-inf"))
    parser.add_argument("--gated-sell-feature2-max", type=float, default=float("inf"))
    parser.add_argument(
        "--score-residual-neutral-base",
        action="store_true",
        help="Keep score columns neutral for the base encoder while the score residual sees real scores.",
    )
    parser.add_argument(
        "--cross-sectional-supervised-context", action="store_true",
        help="Append causal same-timestamp relative ranks of visible supervised scores",
    )
    parser.add_argument(
        "--coin-history-context", action="store_true",
        help="Append causal per-coin stored-history age/coverage context",
    )
    parser.add_argument(
        "--universe-context-width", type=int, default=0,
        help="Optional learned permutation-equivariant pooled visible-universe context width",
    )
    parser.add_argument(
        "--mode-head", action="store_true",
        help=(
            "Use a learned portfolio-level NOOP/BUY/SELL mode head before "
            "the learned coin/allocation conditional policy"
        ),
    )
    parser.add_argument("--sell-q-coef", type=float, default=0.0)
    parser.add_argument("--sell-q-advantage-mix", type=float, default=0.0)
    parser.add_argument("--sell-q-target", choices=("mc", "td0"), default="mc")
    parser.add_argument("--sell-q-policy-coef", type=float, default=0.0)
    parser.add_argument("--buy-q-coef", type=float, default=0.0)
    parser.add_argument("--buy-q-advantage-mix", type=float, default=0.0)
    parser.add_argument("--buy-allocation-q-coef", type=float, default=0.0)
    parser.add_argument("--buy-allocation-q-advantage-mix", type=float, default=0.0)
    parser.add_argument(
        "--sell-aux-coef", type=float, default=0.0,
        help="Training-only cross-entropy weight for causal held-state sell labels",
    )
    parser.add_argument(
        "--sell-aux-positive-weight", type=float, default=1.0,
        help="Class weight for positive SELL auxiliary labels; 1.0 is neutral",
    )
    parser.add_argument(
        "--sell-aux-horizon-hours", type=int, default=0,
        help="Future train-window horizon for sell auxiliary labels; zero disables labels",
    )
    parser.add_argument(
        "--sell-aux-margin", type=float, default=0.0,
        help="SELL label when current after-fee return is within this margin of the horizon best",
    )
    parser.add_argument(
        "--sell-aux-min-return", type=float,
        help="Optional minimum current after-fee return required for a positive SELL auxiliary label",
    )
    parser.add_argument(
        "--sell-aux-min-hold-hours", type=int, default=0,
        help=(
            "Optional training-only minimum hold window for sell auxiliary labels; "
            "inside this window the auxiliary prefers HOLD unless the position "
            "is below --sell-aux-early-stop-return"
        ),
    )
    parser.add_argument(
        "--sell-aux-early-stop-return", type=float, default=-0.03,
        help="After-fee return threshold that permits a SELL auxiliary label before min-hold expires",
    )
    parser.add_argument(
        "--sell-clear-aux-coef", type=float, default=0.0,
        help=(
            "Training-only auxiliary loss coefficient for clear held-state "
            "SELL/HOLD labels; ambiguous states are masked and no evaluation "
            "rule or action filter is created"
        ),
    )
    parser.add_argument(
        "--sell-clear-aux-positive-weight", type=float, default=1.0,
        help="Class weight for clear-state positive SELL labels; 1.0 is neutral",
    )
    parser.add_argument(
        "--sell-clear-contrast-coef", type=float, default=0.0,
        help=(
            "Training-only coefficient for relative clear-state sell-margin "
            "separation; unlike cross-entropy it targets SELL-vs-HOLD ranking "
            "without imposing an absolute exit threshold"
        ),
    )
    parser.add_argument(
        "--sell-clear-contrast-margin", type=float, default=0.25,
        help="Target mean SELL-label margin minus HOLD-label margin for clear-state contrast loss",
    )
    parser.add_argument(
        "--sell-clear-margin-floor-coef", type=float, default=0.0,
        help=(
            "Training-only coefficient that protects clear SELL-labeled held "
            "states from falling below an absolute SELL-HOLD logit margin. "
            "Applies only to SELL labels and creates no evaluation-time rule."
        ),
    )
    parser.add_argument(
        "--sell-clear-margin-floor", type=float, default=-0.25,
        help=(
            "Minimum SELL-minus-HOLD logit margin target for clear SELL labels "
            "when --sell-clear-margin-floor-coef is positive"
        ),
    )
    parser.add_argument(
        "--sell-clear-margin-floor-max-return", type=float, default=None,
        help=(
            "Optional training-only gate: apply clear SELL margin floor only "
            "when current after-fee position return is at or below this value. "
            "This does not create an evaluation-time sell rule."
        ),
    )
    parser.add_argument(
        "--sell-clear-hold-margin-ceiling-coef", type=float, default=0.0,
        help=(
            "Training-only coefficient that pushes clear HOLD-labeled held "
            "states below a SELL-HOLD logit margin ceiling. Applies only to "
            "HOLD labels and creates no evaluation-time hold rule."
        ),
    )
    parser.add_argument(
        "--sell-clear-hold-margin-ceiling", type=float, default=-0.75,
        help=(
            "Maximum SELL-minus-HOLD logit margin target for clear HOLD labels "
            "when --sell-clear-hold-margin-ceiling-coef is positive"
        ),
    )
    parser.add_argument(
        "--sell-margin-align-coef", type=float, default=0.0,
        help=(
            "Training-only coefficient for aligning relative SELL-HOLD logit "
            "margins with continuous future loss-avoidance versus hold-"
            "opportunity targets. This creates no evaluation-time rule."
        ),
    )
    parser.add_argument(
        "--sell-margin-align-horizon-hours", type=int, default=0,
        help="Future train-window horizon for continuous sell-margin alignment targets",
    )
    parser.add_argument(
        "--sell-margin-align-hold-weight", type=float, default=1.0,
        help="Weight applied to best future hold opportunity in the sell-margin alignment target",
    )
    parser.add_argument(
        "--sell-margin-align-scale", type=float, default=0.06,
        help="Return scale used before tanh-compressing sell-margin alignment targets",
    )
    parser.add_argument(
        "--sell-margin-align-target-mode",
        choices=("continuous", "clear_direction"),
        default="continuous",
        help=(
            "Training-only sell-margin target mode. 'continuous' preserves the "
            "legacy avoided-loss minus hold-opportunity target. 'clear_direction' "
            "masks ambiguous states, pushes HOLD for high best-hold opportunity, "
            "and pushes SELL for clear avoided-loss/current-loss states."
        ),
    )
    parser.add_argument(
        "--sell-margin-align-clear-hold-opportunity-min", type=float, default=0.06,
        help="Best-hold opportunity threshold for clear_direction HOLD targets",
    )
    parser.add_argument(
        "--sell-margin-align-clear-avoid-loss-min", type=float, default=0.03,
        help="Avoided-loss threshold for clear_direction SELL targets",
    )
    parser.add_argument(
        "--sell-margin-align-clear-stop-return", type=float, default=-0.03,
        help="Current after-fee return threshold for clear_direction SELL targets",
    )
    parser.add_argument(
        "--sell-margin-align-exclude-conflict", action="store_true",
        help=(
            "Skip clear_direction rows where high hold opportunity and "
            "loss-avoidance signals both fire."
        ),
    )
    parser.add_argument(
        "--sell-margin-level-coef", type=float, default=0.0,
        help=(
            "Training-only coefficient for absolute SELL-HOLD margin level "
            "alignment. Uses the same continuous sell-margin targets as "
            "relative alignment, but keeps the zero boundary instead of "
            "mean-centering. Creates no evaluation-time rule."
        ),
    )
    parser.add_argument(
        "--sell-margin-level-scale", type=float, default=1.0,
        help=(
            "Logit scale used in tanh(SELL-HOLD margin / scale) for "
            "sell-margin level alignment"
        ),
    )
    parser.add_argument(
        "--sell-margin-level-hold-opportunity-min", type=float, default=0.0,
        help=(
            "Training-only threshold for upweighting level-loss HOLD pressure "
            "when future best-hold opportunity is at least this amount"
        ),
    )
    parser.add_argument(
        "--sell-margin-level-hold-weight", type=float, default=1.0,
        help="Loss multiplier for high-opportunity held states in sell-margin level alignment",
    )
    parser.add_argument(
        "--sell-margin-level-sell-avoid-loss-min", type=float, default=0.0,
        help=(
            "Training-only threshold for upweighting level-loss SELL pressure "
            "when selling now avoids at least this future downside"
        ),
    )
    parser.add_argument(
        "--sell-margin-level-sell-weight", type=float, default=1.0,
        help="Loss multiplier for clear avoided-loss held states in sell-margin level alignment",
    )
    parser.add_argument(
        "--sell-margin-conflict-uncertainty-coef", type=float, default=0.0,
        help=(
            "Training-only conflict-aware confidence regularizer for held states where "
            "high-opportunity HOLD and loss-avoidance SELL criteria both apply. "
            "This pushes sell/hold logits toward neutrality on ambiguous states."
        ),
    )
    parser.add_argument(
        "--sell-margin-conflict-uncertainty-scale", type=float, default=1.0,
        help=(
            "Scale used before squaring tanh SELL-HOLD margins for conflict-state "
            "uncertainty regularization."
        ),
    )
    parser.add_argument(
        "--sell-margin-subtype-confidence-coef", type=float, default=0.0,
        help=(
            "Training-only BCE coefficient for held-state sell subtype confidence by "
            "whether state is avoid-loss-only or hold-opportunity-only."
        ),
    )
    parser.add_argument(
        "--sell-margin-conflict-hold-opportunity-min", type=float, default=0.06,
        help="Best-hold opportunity threshold for conflict state detection.",
    )
    parser.add_argument(
        "--sell-margin-conflict-avoid-loss-min", type=float, default=0.03,
        help="Avoided-loss threshold for conflict state detection.",
    )
    parser.add_argument(
        "--sell-margin-conflict-stop-return", type=float, default=-0.03,
        help=(
            "Current after-fee return threshold for conflict state detection when "
            "hold opportunity and sell criteria overlap."
        ),
    )
    parser.add_argument(
        "--sell-target-aux-width", type=int, default=0,
        help=(
            "Width of a detached raw-observation predictor for the continuous "
            "sell-margin target. Default 0 disables the head. The head is "
            "training context only and is not used by trade execution."
        ),
    )
    parser.add_argument(
        "--sell-target-aux-coef", type=float, default=0.0,
        help=(
            "Training-only MSE weight for the detached sell-target predictor. "
            "Uses sell-margin-align labels but does not alter policy logits."
        ),
    )
    parser.add_argument(
        "--train-sell-target-aux-only", action="store_true",
        help="Freeze the trader and optimize only the detached sell-target predictor",
    )
    parser.add_argument(
        "--sell-target-policy-residual", action="store_true",
        help=(
            "Add a zero-initialized learned residual from the detached "
            "sell-target prediction into the SELL-HOLD logit margin"
        ),
    )
    parser.add_argument(
        "--sell-target-policy-residual-width", type=int, default=0,
        help=(
            "Optional hidden width for the aux-to-policy residual. Zero keeps "
            "the one-scalar linear residual; positive values use a small MLP "
            "with zero-initialized final layer."
        ),
    )
    parser.add_argument(
        "--sell-target-policy-residual-source", choices=("score", "hidden"),
        default="score",
        help=(
            "Input source for the aux-to-policy residual. 'score' uses the "
            "detached scalar aux prediction; 'hidden' uses the detached hidden "
            "representation inside the causal auxiliary head."
        ),
    )
    parser.add_argument(
        "--train-sell-target-policy-residual-only", action="store_true",
        help="Freeze the trader and aux head; train only the aux-to-policy residual",
    )
    parser.add_argument(
        "--sell-clear-aux-horizon-hours", type=int, default=0,
        help="Future train-window horizon for clear-state sell labels; zero disables labels",
    )
    parser.add_argument(
        "--sell-clear-aux-hold-opportunity-min", type=float, default=0.06,
        help="Best-hold opportunity floor for a clear HOLD label",
    )
    parser.add_argument(
        "--sell-clear-aux-avoid-loss-min", type=float, default=0.03,
        help="Future downside avoided by selling now required for a clear SELL label",
    )
    parser.add_argument(
        "--sell-clear-aux-stop-return", type=float, default=-0.03,
        help="Current after-fee return threshold that permits a clear SELL label",
    )
    parser.add_argument(
        "--sell-clear-aux-exclude-conflict", action="store_true",
        help=(
            "Skip clear aux rows where high hold opportunity and "
            "loss-avoidance signals both fire."
        ),
    )
    parser.add_argument("--sell-q-detach-encoder", action="store_true")
    parser.add_argument("--train-sell-head-only", action="store_true")
    parser.add_argument("--train-sell-policy-only", action="store_true")
    parser.add_argument("--train-mode-head-only", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--execution-price-mode",
        choices=("current_close", "next_open"),
        default="current_close",
        help="Trade fill timing. current_close preserves legacy behavior; next_open observes t and fills at t+1 open.",
    )
    parser.add_argument(
        "--execution-cost-bps",
        type=float,
        default=0.0,
        help="Additional per-side execution cost in basis points for spread/slippage.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize model weights only from a compatible v7 stage")
    parser.add_argument(
        "--parent-checkpoint", type=Path,
        help="Passed prior-phase checkpoint whose weights start this curriculum",
    )
    parser.add_argument(
        "--expand-input-checkpoint", type=Path,
        help="Initialize matching columns from a narrower compatible checkpoint",
    )
    parser.add_argument(
        "--supervised-targets",
        default="opportunity_7d,drawdown_risk_7d,vol_expansion_risk_7d,future_max_return_7d,future_min_return_7d,future_vol_7d",
        help="Comma-separated percentile context signals, or 'none' for an ablation",
    )
    parser.add_argument(
        "--supervised-raw-targets", default="none",
        help=(
            "Optional bounded raw supervised probabilities to accompany percentile "
            "context; context only, never an action gate"
        ),
    )
    parser.add_argument(
        "--neutral-supervised-targets", default="none",
        help=(
            "Comma-separated included supervised scores to replace with neutral "
            "0.5 while preserving input dimension and initialization"
        ),
    )
    parser.add_argument(
        "--expose-episode-boundary", action="store_true",
        help=(
            "Expose artificial time-remaining/terminal-warning observations; "
            "disabled by default for deployment-consistent training"
        ),
    )
    parser.add_argument(
        "--position-dynamics-mode",
        choices=(
            "disabled", "neutral", "active", "neutral_fee", "active_fee",
            "neutral_memory", "active_memory", "neutral_memory_buy", "active_memory_buy",
        ),
        default="disabled",
        help="Include causal held-position dynamics, matched neutral values, or neither",
    )
    parser.add_argument(
        "--separate-head-memory", action="store_true",
        help=(
            "Mask explicit sell-memory columns from the buy head and explicit "
            "buy-memory columns from the sell head."
        ),
    )
    parser.add_argument("--initial-buy-prob", type=float, default=0.10)
    parser.add_argument("--initial-sell-prob", type=float, default=0.02)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="Credit-assignment horizon; 1.0 is undiscounted Monte Carlo within each episode",
    )
    parser.add_argument(
        "--held-gae-lambda", type=float,
        help=(
            "Optional longer GAE horizon only while a position is held; "
            "defaults to --gae-lambda"
        ),
    )
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--advantage-normalization",
        choices=("global", "state", "state_scale", "decision_scale"),
        default="global",
        help="Normalize PPO advantages globally or separately for flat/held states",
    )
    parser.add_argument(
        "--held-policy-loss-weight", type=float, default=1.0,
        help=(
            "Training-only multiplier for PPO policy loss on held-position "
            "timesteps; 1.0 preserves current behavior exactly"
        ),
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--drawdown-penalty", type=float, default=0.0,
        help="Penalty on incremental peak-to-trough drawdown; zero preserves pure telescoping PnL",
    )
    parser.add_argument(
        "--forced-sell-penalty", type=float, default=0.0,
        help=(
            "Training-only reward penalty per structural terminal liquidation; "
            "does not execute or force any normal sell"
        ),
    )
    parser.add_argument(
        "--turnover-penalty", type=float, default=0.0,
        help=(
            "Training-only reward penalty per executed buy or policy sell; "
            "does not change action legality or execute hidden trades"
        ),
    )
    parser.add_argument(
        "--sell-realization-reward-coef", type=float, default=0.0,
        help=(
            "Training-only reward coefficient applied to the realized after-fee "
            "return of AI manual sells. Zero preserves existing reward exactly; "
            "forced terminal liquidations do not receive this credit."
        ),
    )
    parser.add_argument(
        "--flat-buy-mass-penalty-coef", type=float, default=0.0,
        help=(
            "Training-only PPO loss coefficient penalizing aggregate BUY "
            "probability while flat; does not filter coins, ban buys, or alter evaluation"
        ),
    )
    parser.add_argument(
        "--flat-buy-mass-penalty-target", type=float, default=0.0,
        help=(
            "Optional hinge target for flat BUY mass; when positive, only "
            "BUY probability above this target is penalized, squared"
        ),
    )
    parser.add_argument(
        "--flat-buy-mass-penalty-gate-feature", default="none",
        help=(
            "Optional observation column name gating flat BUY mass penalty; "
            "use 'none' to apply it to every flat buy-allowed state."
        ),
    )
    parser.add_argument(
        "--flat-buy-mass-penalty-gate-min", type=float, default=float("-inf"),
        help="Minimum gated feature value for applying flat BUY mass penalty.",
    )
    parser.add_argument(
        "--flat-buy-mass-penalty-gate-max", type=float, default=float("inf"),
        help="Maximum gated feature value for applying flat BUY mass penalty.",
    )
    parser.add_argument(
        "--failed-to-sell-margin-coef", type=float, default=0.0,
        help=(
            "Training-only loss coefficient for hindsight failed-to-sell "
            "states from episodes ending in terminal liquidation. Pushes "
            "aggregate SELL mass above the strongest non-SELL mode."
        ),
    )
    parser.add_argument(
        "--failed-to-sell-margin-target", type=float, default=0.0,
        help="Target log SELL-mass margin over max(NOOP mass, BUY mass).",
    )
    parser.add_argument(
        "--failed-to-sell-logit-margin-coef", type=float, default=0.0,
        help=(
            "Training-only loss coefficient for failed-to-sell states that "
            "pushes the held coin SELL logit above NOOP and the strongest BUY "
            "logit. This directly targets deterministic exit competition."
        ),
    )
    parser.add_argument(
        "--failed-to-sell-logit-margin-target", type=float, default=0.0,
        help="Target held SELL logit margin over the strongest non-SELL logit.",
    )
    parser.add_argument(
        "--failed-to-sell-hindsight-hours", type=int, default=0,
        help=(
            "Number of final episode hours to mark as failed-to-sell examples "
            "when a held position reaches terminal forced liquidation."
        ),
    )
    parser.add_argument(
        "--failed-to-sell-min-hold-hours", type=int, default=0,
        help="Minimum held age for hindsight failed-to-sell labels.",
    )
    parser.add_argument(
        "--held-sell-mass-penalty-coef", type=float, default=0.0,
        help=(
            "Training-only PPO loss coefficient penalizing aggregate SELL "
            "probability while held; does not block or force sells"
        ),
    )
    parser.add_argument(
        "--held-sell-mass-penalty-target", type=float, default=0.0,
        help=(
            "Optional hinge target for held SELL mass; when positive, only "
            "SELL probability above this target is penalized, squared"
        ),
    )
    parser.add_argument(
        "--early-held-sell-mass-penalty-coef", type=float, default=0.0,
        help=(
            "Training-only PPO loss coefficient penalizing aggregate SELL "
            "probability during early held states that are not past the "
            "configured stop-return threshold; does not block or force sells"
        ),
    )
    parser.add_argument(
        "--early-held-sell-mass-penalty-target", type=float, default=0.0,
        help=(
            "Optional hinge target for early-held SELL mass; when positive, "
            "only SELL probability above this target is penalized, squared"
        ),
    )
    parser.add_argument(
        "--early-held-sell-mass-penalty-hours", type=int, default=0,
        help=(
            "Holding-age window in hours for early-held SELL-mass penalty; "
            "0 disables the mask even if coefficient is positive"
        ),
    )
    parser.add_argument(
        "--early-held-sell-mass-stop-return", type=float, default=-0.03,
        help=(
            "After-fee return threshold below which early-held SELL-mass "
            "penalty is not applied, allowing the policy to learn loss cuts"
        ),
    )
    parser.add_argument(
        "--buy-aux-coef", type=float, default=0.0,
        help=(
            "Training-only auxiliary loss coefficient for flat-state buy/no-buy "
            "selectivity labels; does not filter coins or execute trades"
        ),
    )
    parser.add_argument(
        "--buy-aux-rank-only", action="store_true",
        help=(
            "Apply buy auxiliary only to relative coin ranking conditional on "
            "BUY mass, without increasing aggregate BUY versus NOOP probability"
        ),
    )
    parser.add_argument(
        "--buy-aux-negative-coef", type=float, default=0.0,
        help=(
            "Training-only coefficient for negative buy auxiliary labels. "
            "When no visible coin clears --buy-aux-min-return, penalize "
            "aggregate BUY mass while flat."
        ),
    )
    parser.add_argument(
        "--buy-aux-negative-buy-mass-target", type=float, default=0.0,
        help=(
            "Optional hinge target for aggregate BUY mass on negative buy-aux "
            "states; zero penalizes all BUY mass."
        ),
    )
    parser.add_argument(
        "--buy-aux-negative-gate-feature", default="none",
        help=(
            "Optional observation column name gating negative buy-aux loss, "
            "for example market_regime_strength. Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--buy-aux-negative-gate-min", type=float, default=float("-inf"),
        help="Minimum gated feature value for applying negative buy-aux loss.",
    )
    parser.add_argument(
        "--buy-aux-negative-gate-max", type=float, default=float("inf"),
        help="Maximum gated feature value for applying negative buy-aux loss.",
    )
    parser.add_argument(
        "--buy-aux-horizon-hours", type=int, default=0,
        help=(
            "Future horizon used only for train-period buy auxiliary labels; "
            "0 disables the label source"
        ),
    )
    parser.add_argument(
        "--buy-aux-min-return", type=float, default=0.0,
        help=(
            "After-fee best-future return floor for labeling a flat state as "
            "BUY instead of NOOP in the training-only buy auxiliary"
        ),
    )
    parser.add_argument(
        "--time-limit-bootstrap", action="store_true",
        help=(
            "Bootstrap the critic at fixed data-window ends during training; "
            "terminal liquidation remains safety/evaluation accounting"
        ),
    )
    parser.add_argument(
        "--initial-held-episode-prob", type=float, default=0.0,
        help=(
            "Training-only probability of starting a Phase-1 episode from an "
            "already-open valid position, to improve learned sell/hold coverage"
        ),
    )
    parser.add_argument(
        "--initial-held-allocation", type=float, default=1.0,
        help="Allocation of the training-only initial holding when enabled",
    )
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument(
        "--phase1-multicoin-diagnostic", action="store_true",
        help=(
            "Allow a Phase-1 one-slot joint-policy diagnostic with multiple visible coins; "
            "does not unlock Phase 2 or permit more than one portfolio action per step"
        ),
    )
    parser.add_argument(
        "--year-balanced-selector", action="store_true",
        help="Interleave all valid Phase-1 calendar-year windows uniformly without filtering",
    )
    args = parser.parse_args()
    parent_expand_same_checkpoint = (
        args.parent_checkpoint is not None
        and args.expand_input_checkpoint is not None
        and args.parent_checkpoint.resolve() == args.expand_input_checkpoint.resolve()
    )
    initializers = [
        args.init_checkpoint,
        args.parent_checkpoint,
        None if parent_expand_same_checkpoint else args.expand_input_checkpoint,
    ]
    if sum(value is not None for value in initializers) > 1:
        parser.error("checkpoint initialization options are mutually exclusive")
    try:
        buy_allocations = tuple(float(value) for value in args.buy_allocations.split(","))
    except ValueError as error:
        parser.error(f"invalid --buy-allocations: {error}")
    if args.phase == 1 and not phase1_shape_is_allowed(
        args.universe_size, args.max_slots, args.phase1_multicoin_diagnostic,
    ):
        raise RuntimeError(
            "Phase 1 requires one coin/slot unless the explicit one-slot multi-coin diagnostic is enabled"
        )
    if args.phase >= 2:
        if args.prerequisite_gate is None or not args.prerequisite_gate.exists():
            raise RuntimeError("Phase 2/3 is locked until a prerequisite gate artifact is supplied")
        gate = json.loads(args.prerequisite_gate.read_text())
        gate_phase = gate.get("phase")
        gate_is_prior_phase = gate_phase == args.phase - 1
        gate_is_same_phase_curriculum = (
            args.allow_same_phase_parent and args.phase == 3 and gate_phase == args.phase
        )
        if (
            not (gate_is_prior_phase or gate_is_same_phase_curriculum)
            or gate.get("passed") is not True
        ):
            raise RuntimeError("The prerequisite phase gate did not pass")
        if args.parent_checkpoint is None:
            raise RuntimeError("Phase 2/3 requires --parent-checkpoint")
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "checkpoint_latest.pt"
    if checkpoint_path.exists() and not args.resume:
        raise RuntimeError(f"Run directory already has a checkpoint: {args.run_dir}")
    if args.episode_days_schedule and args.year_balanced_selector:
        parser.error("episode-days-schedule and year-balanced-selector are mutually exclusive")
    if args.episode_days_schedule:
        try:
            episode_horizons = tuple(
                int(value) for value in args.episode_days_schedule.split(",")
            )
        except ValueError as error:
            parser.error(f"invalid --episode-days-schedule: {error}")
        selector = MixedHorizonCoinSelector(
            args.data_dir, args.train_start, args.train_end, episode_horizons,
            args.universe_size, seed=args.seed,
            coin_history_context=args.coin_history_context,
        )
    else:
        selector_class = YearBalancedCoinSelector if args.year_balanced_selector else MultiCoinSelector
        selector = selector_class(
            args.data_dir, args.train_start, args.train_end, args.episode_days,
            args.universe_size, seed=args.seed,
            coin_history_context=args.coin_history_context,
        )
    probe_state = selector.get_state()
    _, probe_frames = selector.next_episode()
    supervised_targets = () if args.supervised_targets == "none" else tuple(args.supervised_targets.split(","))
    supervised_raw_targets = (
        () if args.supervised_raw_targets == "none"
        else tuple(args.supervised_raw_targets.split(","))
    )
    allowed_raw_targets = {
        "opportunity_7d", "drawdown_risk_7d", "vol_expansion_risk_7d",
        "profitable_after_fee_24h", "profitable_after_fee_72h", "profitable_after_fee_168h",
    }
    unknown_raw = set(supervised_raw_targets) - allowed_raw_targets
    if unknown_raw:
        raise ValueError("Raw supervised context requires bounded classification targets: " + ", ".join(sorted(unknown_raw)))
    neutral_supervised_targets = (
        () if args.neutral_supervised_targets == "none"
        else tuple(args.neutral_supervised_targets.split(","))
    )
    unknown_neutral = set(neutral_supervised_targets) - set(supervised_targets)
    if unknown_neutral:
        raise ValueError(
            "Neutralized supervised targets must also be included: "
            + ", ".join(sorted(unknown_neutral))
        )
    probe = MultiCoinTradingEnv(
        probe_frames, min(args.max_slots, len(probe_frames)),
        supervised_targets=supervised_targets,
        supervised_raw_targets=supervised_raw_targets,
        neutral_supervised_targets=neutral_supervised_targets,
        expose_episode_boundary=args.expose_episode_boundary,
        position_dynamics_mode=args.position_dynamics_mode,
        buy_allocations=buy_allocations,
        allocation_sizing_mode=args.allocation_sizing_mode,
        cross_sectional_supervised_context=args.cross_sectional_supervised_context,
        coin_history_context=args.coin_history_context,
        execution_price_mode=args.execution_price_mode,
        execution_cost_bps=args.execution_cost_bps,
    )
    selector.load_state(probe_state)
    input_dim = probe.input_dim
    held_feature_index = len(probe.feature_columns) + len(probe.cross_context_columns)
    score_feature_indices = tuple(
        probe.feature_columns.index(f"sup_{target}_pct")
        for target in supervised_targets
    )
    observation_columns = probe.feature_columns + probe.cross_context_columns + [
        "in_held", "allow_buy", "allow_sell", "position_allocation_fraction",
        "position_age_normalized", "steps_since_last_sell_normalized",
        "return_since_entry", "max_unrealized_return", "min_unrealized_return",
        "scan_progress_normalized", "auto_sell_flag", "time_remaining",
    ] + (
        position_dynamics_columns(args.position_dynamics_mode)
    ) + ["cash_fraction"]
    buy_memory_feature_indices: tuple[int, ...] = tuple()
    sell_memory_feature_indices: tuple[int, ...] = tuple()
    if args.separate_head_memory:
        explicit_sell_memory = {
            "memory_realizable_net_return",
            "memory_pnl_change_1h",
            "memory_mfe",
            "memory_mae",
            "memory_fade_from_mfe",
        }
        buy_memory_feature_indices = tuple(
            index for index, name in enumerate(observation_columns)
            if name.startswith("buy_memory_")
        )
        sell_memory_feature_indices = tuple(
            index for index, name in enumerate(observation_columns)
            if name in explicit_sell_memory
        )
    boundary_sell_feature_index = (
        observation_columns.index("auto_sell_flag")
        if args.boundary_sell_residual else None
    )
    buy_aux_negative_gate_feature_index = -1
    if args.buy_aux_negative_gate_feature != "none":
        if args.buy_aux_negative_gate_feature not in observation_columns:
            raise ValueError(
                "Unknown buy-aux negative gate feature: "
                f"{args.buy_aux_negative_gate_feature}"
            )
        buy_aux_negative_gate_feature_index = observation_columns.index(
            args.buy_aux_negative_gate_feature
        )
    flat_buy_mass_penalty_gate_feature_index = -1
    if args.flat_buy_mass_penalty_gate_feature != "none":
        if args.flat_buy_mass_penalty_gate_feature not in observation_columns:
            raise ValueError(
                "Unknown flat buy-mass penalty gate feature: "
                f"{args.flat_buy_mass_penalty_gate_feature}"
            )
        flat_buy_mass_penalty_gate_feature_index = observation_columns.index(
            args.flat_buy_mass_penalty_gate_feature
        )
    gated_buy_feature_index = None
    gated_buy_feature2_index = None
    gated_buy2_feature_index = None
    gated_buy2_feature2_index = None
    gated_buy3_feature_index = None
    gated_buy3_feature2_index = None
    gated_sell_feature_index = None
    gated_sell_feature2_index = None
    if args.gated_buy_residual:
        if args.gated_buy_feature not in observation_columns:
            raise ValueError(f"Unknown gated buy feature: {args.gated_buy_feature}")
        gated_buy_feature_index = observation_columns.index(args.gated_buy_feature)
        if args.gated_buy_feature2 != "none":
            if args.gated_buy_feature2 not in observation_columns:
                raise ValueError(f"Unknown second gated buy feature: {args.gated_buy_feature2}")
            gated_buy_feature2_index = observation_columns.index(args.gated_buy_feature2)
    if args.gated_buy2_residual:
        if args.gated_buy2_feature not in observation_columns:
            raise ValueError(f"Unknown second gated buy residual feature: {args.gated_buy2_feature}")
        gated_buy2_feature_index = observation_columns.index(args.gated_buy2_feature)
        if args.gated_buy2_feature2 != "none":
            if args.gated_buy2_feature2 not in observation_columns:
                raise ValueError(
                    f"Unknown second gated buy residual secondary feature: {args.gated_buy2_feature2}"
                )
            gated_buy2_feature2_index = observation_columns.index(args.gated_buy2_feature2)
    if args.gated_buy3_residual:
        if args.gated_buy3_feature not in observation_columns:
            raise ValueError(f"Unknown third gated buy residual feature: {args.gated_buy3_feature}")
        gated_buy3_feature_index = observation_columns.index(args.gated_buy3_feature)
        if args.gated_buy3_feature2 != "none":
            if args.gated_buy3_feature2 not in observation_columns:
                raise ValueError(
                    f"Unknown third gated buy residual secondary feature: {args.gated_buy3_feature2}"
                )
            gated_buy3_feature2_index = observation_columns.index(args.gated_buy3_feature2)
    if args.gated_sell_residual:
        if args.gated_sell_feature not in observation_columns:
            raise ValueError(f"Unknown gated sell feature: {args.gated_sell_feature}")
        gated_sell_feature_index = observation_columns.index(args.gated_sell_feature)
        if args.gated_sell_feature2 != "none":
            if args.gated_sell_feature2 not in observation_columns:
                raise ValueError(f"Unknown second gated sell feature: {args.gated_sell_feature2}")
            gated_sell_feature2_index = observation_columns.index(args.gated_sell_feature2)
    if args.score_residual and not score_feature_indices:
        raise ValueError("Score residual requires at least one supervised percentile context column")
    model = PPOTradingModel(
        input_dim, args.width, args.width, args.width,
        num_res_blocks=args.blocks, dropout=args.dropout,
        sell_head_width=args.sell_head_width,
        temporal_width=args.temporal_width,
        actor_mode=args.actor_mode,
        state_value_heads=args.state_value_heads,
        held_feature_index=held_feature_index,
        action_value_heads=(args.sell_q_coef > 0.0 or args.sell_q_advantage_mix > 0.0
                            or args.buy_q_coef > 0.0 or args.buy_q_advantage_mix > 0.0),
        flat_action_value_heads=(args.buy_allocation_q_coef > 0.0
                                 or args.buy_allocation_q_advantage_mix > 0.0),
        action_value_detach_encoder=args.sell_q_detach_encoder,
        sell_raw_residual=args.sell_raw_residual,
        sell_raw_residual_width=args.sell_raw_residual_width,
        buy_raw_residual=args.buy_raw_residual,
        buy_raw_residual_width=args.buy_raw_residual_width,
        score_residual=args.score_residual,
        score_residual_width=args.score_residual_width,
        score_feature_indices=score_feature_indices,
        score_residual_neutral_base=args.score_residual_neutral_base,
        boundary_sell_residual=args.boundary_sell_residual,
        boundary_sell_feature_index=boundary_sell_feature_index,
        gated_buy_residual=args.gated_buy_residual,
        gated_buy_feature_index=gated_buy_feature_index,
        gated_buy_feature_min=args.gated_buy_feature_min,
        gated_buy_feature_max=args.gated_buy_feature_max,
        gated_buy_feature2_index=gated_buy_feature2_index,
        gated_buy_feature2_min=args.gated_buy_feature2_min,
        gated_buy_feature2_max=args.gated_buy_feature2_max,
        gated_buy2_residual=args.gated_buy2_residual,
        gated_buy2_feature_index=gated_buy2_feature_index,
        gated_buy2_feature_min=args.gated_buy2_feature_min,
        gated_buy2_feature_max=args.gated_buy2_feature_max,
        gated_buy2_feature2_index=gated_buy2_feature2_index,
        gated_buy2_feature2_min=args.gated_buy2_feature2_min,
        gated_buy2_feature2_max=args.gated_buy2_feature2_max,
        gated_buy3_residual=args.gated_buy3_residual,
        gated_buy3_feature_index=gated_buy3_feature_index,
        gated_buy3_feature_min=args.gated_buy3_feature_min,
        gated_buy3_feature_max=args.gated_buy3_feature_max,
        gated_buy3_feature2_index=gated_buy3_feature2_index,
        gated_buy3_feature2_min=args.gated_buy3_feature2_min,
        gated_buy3_feature2_max=args.gated_buy3_feature2_max,
        gated_sell_residual=args.gated_sell_residual,
        gated_sell_feature_index=gated_sell_feature_index,
        gated_sell_feature_min=args.gated_sell_feature_min,
        gated_sell_feature_max=args.gated_sell_feature_max,
        gated_sell_feature2_index=gated_sell_feature2_index,
        gated_sell_feature2_min=args.gated_sell_feature2_min,
        gated_sell_feature2_max=args.gated_sell_feature2_max,
        universe_context_width=args.universe_context_width,
        mode_head=args.mode_head,
        sell_target_aux_width=args.sell_target_aux_width,
        sell_target_policy_residual=args.sell_target_policy_residual,
        sell_target_policy_residual_width=args.sell_target_policy_residual_width,
        sell_target_policy_residual_source=args.sell_target_policy_residual_source,
        buy_excluded_feature_indices=sell_memory_feature_indices,
        sell_excluded_feature_indices=buy_memory_feature_indices,
    ).to(device)
    if args.temporal_width < 0 or args.sequence_length < 1 or args.burn_in < 0:
        raise ValueError("Invalid temporal configuration")
    if not (0.0 < args.initial_buy_prob < 1.0 and 0.0 < args.initial_sell_prob < 1.0):
        raise ValueError("Initial action probabilities must be in (0,1)")
    if not (0.0 <= args.gae_lambda <= 1.0):
        raise ValueError("GAE lambda must be in [0,1]")
    held_gae_lambda = args.gae_lambda if args.held_gae_lambda is None else args.held_gae_lambda
    if not (0.0 <= held_gae_lambda <= 1.0):
        raise ValueError("held GAE lambda must be in [0,1]")
    if args.held_policy_loss_weight <= 0.0:
        raise ValueError("held-policy-loss-weight must be positive")
    if args.drawdown_penalty < 0.0:
        raise ValueError("drawdown penalty must be nonnegative")
    if args.forced_sell_penalty < 0.0:
        raise ValueError("forced-sell-penalty must be nonnegative")
    if args.turnover_penalty < 0.0:
        raise ValueError("turnover-penalty must be nonnegative")
    if args.execution_cost_bps < 0.0:
        raise ValueError("execution-cost-bps must be nonnegative")
    if args.sell_realization_reward_coef < 0.0:
        raise ValueError("sell-realization-reward-coef must be nonnegative")
    if args.flat_buy_mass_penalty_coef < 0.0:
        raise ValueError("flat-buy-mass-penalty-coef must be nonnegative")
    if not 0.0 <= args.flat_buy_mass_penalty_target < 1.0:
        raise ValueError("flat-buy-mass-penalty-target must be in [0, 1)")
    if args.flat_buy_mass_penalty_gate_min > args.flat_buy_mass_penalty_gate_max:
        raise ValueError("flat-buy-mass-penalty-gate-min must be <= gate max")
    if args.gated_buy_feature_min > args.gated_buy_feature_max:
        raise ValueError("gated-buy-feature-min must be <= gated-buy-feature-max")
    if args.gated_buy_feature2_min > args.gated_buy_feature2_max:
        raise ValueError("gated-buy-feature2-min must be <= gated-buy-feature2-max")
    if args.gated_buy_residual and args.gated_buy_feature == "none":
        raise ValueError("gated-buy-residual requires --gated-buy-feature")
    if args.gated_buy2_feature_min > args.gated_buy2_feature_max:
        raise ValueError("gated-buy2-feature-min must be <= gated-buy2-feature-max")
    if args.gated_buy2_feature2_min > args.gated_buy2_feature2_max:
        raise ValueError("gated-buy2-feature2-min must be <= gated-buy2-feature2-max")
    if args.gated_buy2_residual and args.gated_buy2_feature == "none":
        raise ValueError("gated-buy2-residual requires --gated-buy2-feature")
    if args.gated_buy3_feature_min > args.gated_buy3_feature_max:
        raise ValueError("gated-buy3-feature-min must be <= gated-buy3-feature-max")
    if args.gated_buy3_feature2_min > args.gated_buy3_feature2_max:
        raise ValueError("gated-buy3-feature2-min must be <= gated-buy3-feature2-max")
    if args.gated_buy3_residual and args.gated_buy3_feature == "none":
        raise ValueError("gated-buy3-residual requires --gated-buy3-feature")
    if args.gated_sell_feature_min > args.gated_sell_feature_max:
        raise ValueError("gated-sell-feature-min must be <= gated-sell-feature-max")
    if args.gated_sell_feature2_min > args.gated_sell_feature2_max:
        raise ValueError("gated-sell-feature2-min must be <= gated-sell-feature2-max")
    if args.gated_sell_residual and args.gated_sell_feature == "none":
        raise ValueError("gated-sell-residual requires --gated-sell-feature")
    if args.failed_to_sell_margin_coef < 0.0:
        raise ValueError("failed-to-sell-margin-coef must be nonnegative")
    if args.failed_to_sell_logit_margin_coef < 0.0:
        raise ValueError("failed-to-sell-logit-margin-coef must be nonnegative")
    if args.failed_to_sell_hindsight_hours < 0:
        raise ValueError("failed-to-sell-hindsight-hours must be nonnegative")
    if args.failed_to_sell_min_hold_hours < 0:
        raise ValueError("failed-to-sell-min-hold-hours must be nonnegative")
    if (
        args.failed_to_sell_margin_coef > 0.0
        or args.failed_to_sell_logit_margin_coef > 0.0
    ) and args.failed_to_sell_hindsight_hours <= 0:
        raise ValueError("failed-to-sell margin losses require positive failed-to-sell-hindsight-hours")
    if (
        args.failed_to_sell_margin_coef > 0.0
        and args.failed_to_sell_hindsight_hours <= 0
    ):
        raise ValueError("failed-to-sell-margin-coef requires positive failed-to-sell-hindsight-hours")
    if args.held_sell_mass_penalty_coef < 0.0:
        raise ValueError("held-sell-mass-penalty-coef must be nonnegative")
    if not 0.0 <= args.held_sell_mass_penalty_target < 1.0:
        raise ValueError("held-sell-mass-penalty-target must be in [0, 1)")
    if args.early_held_sell_mass_penalty_coef < 0.0:
        raise ValueError("early-held-sell-mass-penalty-coef must be nonnegative")
    if not 0.0 <= args.early_held_sell_mass_penalty_target < 1.0:
        raise ValueError("early-held-sell-mass-penalty-target must be in [0, 1)")
    if args.early_held_sell_mass_penalty_hours < 0:
        raise ValueError("early-held-sell-mass-penalty-hours must be nonnegative")
    if (
        args.early_held_sell_mass_penalty_coef > 0.0
        and args.early_held_sell_mass_penalty_hours <= 0
    ):
        raise ValueError("early-held-sell-mass-penalty-coef requires positive hours")
    if args.time_limit_bootstrap and args.drawdown_penalty != 0.0:
        raise ValueError("Time-limit bootstrap currently requires zero drawdown penalty")
    if not 0.0 <= args.initial_held_episode_prob <= 1.0:
        raise ValueError("initial-held-episode-prob must be in [0, 1]")
    if not 0.0 < args.initial_held_allocation <= 1.0:
        raise ValueError("initial-held-allocation must be in (0, 1]")
    if args.initial_held_episode_prob > 0.0 and (args.phase != 1 or args.max_slots != 1):
        raise ValueError("Initial-held curriculum is currently exact only for Phase 1 one-slot policies")
    if args.sell_q_coef < 0.0 or not 0.0 <= args.sell_q_advantage_mix <= 1.0:
        raise ValueError("Invalid sell-Q credit configuration")
    if args.sell_q_target == "td0" and args.sell_q_coef <= 0.0:
        raise ValueError("TD(0) Sell-Q target requires a positive sell-q-coef")
    if args.sell_q_policy_coef < 0.0:
        raise ValueError("sell-q-policy-coef must be nonnegative")
    if args.buy_q_coef < 0.0 or not 0.0 <= args.buy_q_advantage_mix <= 1.0:
        raise ValueError("Invalid Buy-Q credit configuration")
    if args.buy_q_advantage_mix > 0.0:
        if args.phase != 1 or args.universe_size != 1 or args.max_slots != 1:
            raise ValueError("Buy-Q policy credit is exact only for Phase 1")
        if args.buy_q_coef <= 0.0:
            raise ValueError("Buy-Q advantage mix requires a positive buy-q-coef")
    if args.buy_allocation_q_coef < 0.0 or not 0.0 <= args.buy_allocation_q_advantage_mix <= 1.0:
        raise ValueError("Invalid allocation-Q credit configuration")
    if args.buy_allocation_q_advantage_mix > 0.0:
        if args.phase != 1 or args.max_slots != 1:
            raise ValueError("Allocation-Q policy credit is exact only for Phase 1 one-slot policies")
        if args.buy_allocation_q_coef <= 0.0:
            raise ValueError("Allocation-Q advantage mix requires a positive coefficient")
    if args.sell_q_policy_coef > 0.0:
        if args.phase != 1 or args.max_slots != 1:
            raise ValueError("Sell-Q policy improvement is exact only for Phase 1 one-slot policies")
        if args.sell_q_coef <= 0.0:
            raise ValueError("Sell-Q policy improvement requires a positive sell-q-coef")
    if args.sell_aux_coef < 0.0:
        raise ValueError("sell-aux-coef must be nonnegative")
    if args.sell_aux_positive_weight <= 0.0:
        raise ValueError("sell-aux-positive-weight must be positive")
    if args.sell_aux_horizon_hours < 0:
        raise ValueError("sell-aux-horizon-hours must be nonnegative")
    if args.sell_aux_margin < 0.0:
        raise ValueError("sell-aux-margin must be nonnegative")
    if args.sell_aux_min_hold_hours < 0:
        raise ValueError("sell-aux-min-hold-hours must be nonnegative")
    if args.sell_aux_coef > 0.0:
        if args.phase != 1 or args.max_slots != 1:
            raise ValueError("Sell auxiliary labels are currently exact only for Phase 1 one-slot policies")
        if args.sell_aux_horizon_hours <= 0:
            raise ValueError("sell-aux-coef requires a positive sell-aux-horizon-hours")
    if args.sell_clear_aux_coef < 0.0:
        raise ValueError("sell-clear-aux-coef must be nonnegative")
    if args.sell_clear_aux_positive_weight <= 0.0:
        raise ValueError("sell-clear-aux-positive-weight must be positive")
    if args.sell_clear_contrast_coef < 0.0:
        raise ValueError("sell-clear-contrast-coef must be nonnegative")
    if args.sell_clear_contrast_margin < 0.0:
        raise ValueError("sell-clear-contrast-margin must be nonnegative")
    if args.sell_clear_margin_floor_coef < 0.0:
        raise ValueError("sell-clear-margin-floor-coef must be nonnegative")
    if args.sell_clear_hold_margin_ceiling_coef < 0.0:
        raise ValueError("sell-clear-hold-margin-ceiling-coef must be nonnegative")
    if args.sell_margin_align_coef < 0.0:
        raise ValueError("sell-margin-align-coef must be nonnegative")
    if args.sell_margin_level_coef < 0.0:
        raise ValueError("sell-margin-level-coef must be nonnegative")
    if args.sell_margin_level_scale <= 0.0:
        raise ValueError("sell-margin-level-scale must be positive")
    if args.sell_margin_level_hold_opportunity_min < 0.0:
        raise ValueError("sell-margin-level-hold-opportunity-min must be nonnegative")
    if args.sell_margin_level_hold_weight <= 0.0:
        raise ValueError("sell-margin-level-hold-weight must be positive")
    if args.sell_margin_level_sell_avoid_loss_min < 0.0:
        raise ValueError("sell-margin-level-sell-avoid-loss-min must be nonnegative")
    if args.sell_margin_level_sell_weight <= 0.0:
        raise ValueError("sell-margin-level-sell-weight must be positive")
    if args.sell_margin_conflict_uncertainty_coef < 0.0:
        raise ValueError("sell-margin-conflict-uncertainty-coef must be nonnegative")
    if args.sell_margin_conflict_uncertainty_scale <= 0.0:
        raise ValueError("sell-margin-conflict-uncertainty-scale must be positive")
    if args.sell_margin_subtype_confidence_coef < 0.0:
        raise ValueError("sell-margin-subtype-confidence-coef must be nonnegative")
    if args.sell_margin_conflict_hold_opportunity_min < 0.0:
        raise ValueError("sell-margin-conflict-hold-opportunity-min must be nonnegative")
    if args.sell_margin_conflict_avoid_loss_min < 0.0:
        raise ValueError("sell-margin-conflict-avoid-loss-min must be nonnegative")
    if args.sell_margin_align_horizon_hours < 0:
        raise ValueError("sell-margin-align-horizon-hours must be nonnegative")
    if args.sell_margin_align_clear_hold_opportunity_min < 0.0:
        raise ValueError("sell-margin-align-clear-hold-opportunity-min must be nonnegative")
    if args.sell_margin_align_clear_avoid_loss_min < 0.0:
        raise ValueError("sell-margin-align-clear-avoid-loss-min must be nonnegative")
    if args.sell_margin_align_hold_weight < 0.0:
        raise ValueError("sell-margin-align-hold-weight must be nonnegative")
    if args.sell_margin_align_scale <= 0.0:
        raise ValueError("sell-margin-align-scale must be positive")
    if args.sell_target_aux_width < 0:
        raise ValueError("sell-target-aux-width must be nonnegative")
    if args.sell_target_aux_coef < 0.0:
        raise ValueError("sell-target-aux-coef must be nonnegative")
    if args.sell_target_aux_coef > 0.0 and args.sell_target_aux_width <= 0:
        raise ValueError("sell-target-aux-coef requires --sell-target-aux-width")
    if args.train_sell_target_aux_only and args.sell_target_aux_width <= 0:
        raise ValueError("train-sell-target-aux-only requires --sell-target-aux-width")
    if args.sell_target_policy_residual and args.sell_target_aux_width <= 0:
        raise ValueError("sell-target-policy-residual requires --sell-target-aux-width")
    if args.sell_target_policy_residual_width < 0:
        raise ValueError("sell-target-policy-residual-width must be nonnegative")
    if args.sell_target_policy_residual_width > 0 and not args.sell_target_policy_residual:
        raise ValueError("sell-target-policy-residual-width requires --sell-target-policy-residual")
    if args.sell_target_policy_residual_source == "hidden" and not args.sell_target_policy_residual:
        raise ValueError("hidden sell-target-policy-residual-source requires --sell-target-policy-residual")
    if args.train_sell_target_policy_residual_only and not args.sell_target_policy_residual:
        raise ValueError("train-sell-target-policy-residual-only requires --sell-target-policy-residual")
    if args.sell_clear_aux_horizon_hours < 0:
        raise ValueError("sell-clear-aux-horizon-hours must be nonnegative")
    if args.sell_clear_aux_hold_opportunity_min < 0.0:
        raise ValueError("sell-clear-aux-hold-opportunity-min must be nonnegative")
    if args.sell_clear_aux_avoid_loss_min < 0.0:
        raise ValueError("sell-clear-aux-avoid-loss-min must be nonnegative")
    if (
        args.sell_clear_aux_coef > 0.0
        or args.sell_clear_contrast_coef > 0.0
        or args.sell_clear_margin_floor_coef > 0.0
        or args.sell_clear_hold_margin_ceiling_coef > 0.0
        or args.sell_margin_align_coef > 0.0
        or args.sell_margin_level_coef > 0.0
        or args.sell_target_aux_coef > 0.0
        or args.sell_margin_subtype_confidence_coef > 0.0
    ):
        if args.phase != 1 or args.max_slots != 1:
            raise ValueError("Clear sell auxiliary labels are currently exact only for Phase 1 one-slot policies")
        if (
            (
                args.sell_clear_aux_coef > 0.0
                or args.sell_clear_contrast_coef > 0.0
                or args.sell_clear_margin_floor_coef > 0.0
                or args.sell_clear_hold_margin_ceiling_coef > 0.0
            )
            and args.sell_clear_aux_horizon_hours <= 0
        ):
            raise ValueError("clear sell objectives require a positive sell-clear-aux-horizon-hours")
        if args.sell_margin_align_coef > 0.0 and args.sell_margin_align_horizon_hours <= 0:
            raise ValueError("sell-margin-align-coef requires a positive sell-margin-align-horizon-hours")
        if args.sell_margin_level_coef > 0.0 and args.sell_margin_align_horizon_hours <= 0:
            raise ValueError("sell-margin-level-coef requires a positive sell-margin-align-horizon-hours")
        if args.sell_target_aux_coef > 0.0 and args.sell_margin_align_horizon_hours <= 0:
            raise ValueError("sell-target-aux-coef requires a positive sell-margin-align-horizon-hours")
        if (
            args.sell_margin_subtype_confidence_coef > 0.0
            and args.sell_margin_align_horizon_hours <= 0
        ):
            raise ValueError(
                "sell-margin-subtype-confidence-coef requires a positive "
                "sell-margin-align-horizon-hours"
            )
    if args.buy_aux_coef < 0.0:
        raise ValueError("buy-aux-coef must be nonnegative")
    if args.buy_aux_negative_coef < 0.0:
        raise ValueError("buy-aux-negative-coef must be nonnegative")
    if not 0.0 <= args.buy_aux_negative_buy_mass_target < 1.0:
        raise ValueError("buy-aux-negative-buy-mass-target must be in [0, 1)")
    if args.buy_aux_negative_gate_min > args.buy_aux_negative_gate_max:
        raise ValueError("buy-aux-negative-gate-min must be <= gate max")
    if args.buy_aux_horizon_hours < 0:
        raise ValueError("buy-aux-horizon-hours must be nonnegative")
    if args.buy_aux_coef > 0.0 or args.buy_aux_negative_coef > 0.0:
        if args.phase != 1 or args.max_slots != 1:
            raise ValueError("Buy auxiliary labels are currently exact only for Phase 1 one-slot policies")
        if args.buy_aux_horizon_hours <= 0:
            raise ValueError("buy auxiliary losses require a positive buy-aux-horizon-hours")
    if sum(bool(value) for value in (
        args.train_sell_head_only,
        args.train_sell_policy_only,
        args.train_sell_residual_only,
        args.train_buy_residual_only,
        args.train_score_residual_only,
        args.train_boundary_sell_residual_only,
        args.train_gated_buy_residual_only,
        args.train_gated_buy2_residual_only,
        args.train_gated_buy3_residual_only,
        args.train_gated_sell_residual_only,
        args.train_mode_head_only,
        args.train_sell_target_aux_only,
        args.train_sell_target_policy_residual_only,
    )) > 1:
        raise ValueError("Choose only one train-*-only mode")
    if args.train_sell_head_only:
        configure_sell_head_only_training(model)
    if args.train_sell_policy_only:
        configure_sell_policy_only_training(model)
    if args.train_sell_target_aux_only:
        configure_sell_target_aux_only_training(model)
    if args.train_sell_target_policy_residual_only:
        configure_sell_target_policy_residual_only_training(model)
    if args.train_mode_head_only:
        configure_mode_head_only_training(model)
    if args.sell_raw_residual_width < 0 or (
        args.sell_raw_residual_width > 0 and not args.sell_raw_residual
    ):
        raise ValueError("Sell residual width requires --sell-raw-residual")
    if args.buy_raw_residual_width < 0 or (
        args.buy_raw_residual_width > 0 and not args.buy_raw_residual
    ):
        raise ValueError("Buy residual width requires --buy-raw-residual")
    if args.score_residual_width < 1 and args.score_residual:
        raise ValueError("Score residual width must be positive")
    if args.universe_context_width < 0:
        raise ValueError("Universe context width must be nonnegative")
    if args.universe_context_width > 0 and (
        args.universe_size == 1 or args.temporal_width > 0
    ):
        raise ValueError("Universe context requires a feed-forward multi-coin policy")
    if args.mode_head and (
        model.action_value_heads or model.flat_action_value_heads
        or args.temporal_width > 0
    ):
        raise ValueError("Mode-head diagnostic requires feed-forward PPO without auxiliary Q heads")
    if args.train_buy_residual_only:
        configure_buy_residual_only_training(model)
    if args.train_sell_residual_only:
        configure_residual_only_training(model)
    if args.train_score_residual_only:
        configure_score_residual_only_training(model)
    if args.train_boundary_sell_residual_only:
        configure_boundary_sell_residual_only_training(model)
    if args.train_gated_buy_residual_only:
        configure_gated_buy_residual_only_training(model)
    if args.train_gated_buy2_residual_only:
        configure_gated_buy2_residual_only_training(model)
    if args.train_gated_buy3_residual_only:
        configure_gated_buy3_residual_only_training(model)
    if args.train_gated_sell_residual_only:
        configure_gated_sell_residual_only_training(model)
    if args.time_limit_bootstrap and (
        args.temporal_width > 0 or args.sell_q_coef > 0.0 or args.buy_q_coef > 0.0
        or args.buy_allocation_q_coef > 0.0 or args.sell_q_advantage_mix > 0.0
        or args.sell_q_policy_coef > 0.0 or args.buy_allocation_q_advantage_mix > 0.0
        or args.sell_aux_coef > 0.0 or args.sell_clear_aux_coef > 0.0
        or args.sell_clear_contrast_coef > 0.0
        or args.sell_clear_margin_floor_coef > 0.0
        or args.sell_clear_hold_margin_ceiling_coef > 0.0
        or args.sell_margin_align_coef > 0.0
        or args.sell_target_aux_coef > 0.0
        or args.buy_aux_coef > 0.0
    ):
        raise ValueError("Time-limit bootstrap diagnostic requires feed-forward PPO without Sell-Q")
    if args.temporal_width > 0 and (
        args.sell_q_advantage_mix > 0.0 or args.sell_q_policy_coef > 0.0
        or args.buy_q_advantage_mix > 0.0 or args.buy_allocation_q_advantage_mix > 0.0
    ):
        raise ValueError("Sell-Q policy credit is feed-forward only")
    # Fee-aware exploration prior.  These are policy logits, never hidden
    # environment gates; PPO remains free to learn any buy/sell frequency.
    with torch.no_grad():
        model.buy_head.bias[:-1].fill_(
            float(np.log(args.initial_buy_prob / (len(model.buy_head.bias) - 1)))
        )
        model.buy_head.bias[-1].fill_(float(np.log(1.0 - args.initial_buy_prob)))
        model.sell_head.bias[0].fill_(float(np.log(args.initial_sell_prob)))
        model.sell_head.bias[1].fill_(float(np.log(1.0 - args.initial_sell_prob)))
        if model.mode_head_enabled:
            model.mode_head.weight.zero_()
            model.mode_head.bias[0].fill_(float(np.log(1.0 - args.initial_buy_prob)))
            model.mode_head.bias[1].fill_(float(np.log(args.initial_buy_prob)))
            model.mode_head.bias[2].fill_(float(np.log(args.initial_sell_prob)))
    ppo_config = JointPPOConfig(
        entropy_coef=args.entropy_coef, lr=args.learning_rate,
        gae_lambda=args.gae_lambda, held_gae_lambda=held_gae_lambda,
        advantage_normalization=args.advantage_normalization,
        sell_q_coef=args.sell_q_coef,
        sell_q_advantage_mix=args.sell_q_advantage_mix,
        sell_q_target=args.sell_q_target,
        sell_q_policy_coef=args.sell_q_policy_coef,
        buy_q_coef=args.buy_q_coef,
        buy_q_advantage_mix=args.buy_q_advantage_mix,
        buy_allocation_q_coef=args.buy_allocation_q_coef,
        buy_allocation_q_advantage_mix=args.buy_allocation_q_advantage_mix,
        buy_aux_coef=args.buy_aux_coef,
        buy_aux_rank_only=args.buy_aux_rank_only,
        buy_aux_negative_coef=args.buy_aux_negative_coef,
        buy_aux_negative_buy_mass_target=args.buy_aux_negative_buy_mass_target,
        buy_aux_negative_gate_feature_index=buy_aux_negative_gate_feature_index,
        buy_aux_negative_gate_min=args.buy_aux_negative_gate_min,
        buy_aux_negative_gate_max=args.buy_aux_negative_gate_max,
        flat_buy_mass_penalty_gate_feature_index=flat_buy_mass_penalty_gate_feature_index,
        flat_buy_mass_penalty_gate_min=args.flat_buy_mass_penalty_gate_min,
        flat_buy_mass_penalty_gate_max=args.flat_buy_mass_penalty_gate_max,
        sell_aux_coef=args.sell_aux_coef,
        sell_aux_positive_weight=args.sell_aux_positive_weight,
        sell_clear_aux_coef=args.sell_clear_aux_coef,
        sell_clear_aux_positive_weight=args.sell_clear_aux_positive_weight,
        sell_clear_contrast_coef=args.sell_clear_contrast_coef,
        sell_clear_contrast_margin=args.sell_clear_contrast_margin,
        sell_clear_margin_floor_coef=args.sell_clear_margin_floor_coef,
        sell_clear_margin_floor=args.sell_clear_margin_floor,
        sell_clear_margin_floor_max_return=args.sell_clear_margin_floor_max_return,
        sell_clear_hold_margin_ceiling_coef=args.sell_clear_hold_margin_ceiling_coef,
        sell_clear_hold_margin_ceiling=args.sell_clear_hold_margin_ceiling,
        sell_margin_align_coef=args.sell_margin_align_coef,
        sell_margin_align_clear_hold_opportunity_min=args.sell_margin_align_clear_hold_opportunity_min,
        sell_margin_align_clear_avoid_loss_min=args.sell_margin_align_clear_avoid_loss_min,
        sell_margin_align_clear_stop_return=args.sell_margin_align_clear_stop_return,
        sell_margin_level_coef=args.sell_margin_level_coef,
        sell_margin_level_scale=args.sell_margin_level_scale,
        sell_margin_level_hold_opportunity_min=args.sell_margin_level_hold_opportunity_min,
        sell_margin_level_hold_weight=args.sell_margin_level_hold_weight,
        sell_margin_level_sell_avoid_loss_min=args.sell_margin_level_sell_avoid_loss_min,
        sell_margin_level_sell_weight=args.sell_margin_level_sell_weight,
        sell_margin_conflict_uncertainty_coef=args.sell_margin_conflict_uncertainty_coef,
        sell_margin_conflict_uncertainty_scale=args.sell_margin_conflict_uncertainty_scale,
        sell_margin_subtype_confidence_coef=args.sell_margin_subtype_confidence_coef,
        sell_margin_conflict_hold_opportunity_min=args.sell_margin_conflict_hold_opportunity_min,
        sell_margin_conflict_avoid_loss_min=args.sell_margin_conflict_avoid_loss_min,
        sell_margin_conflict_stop_return=args.sell_margin_conflict_stop_return,
        sell_target_aux_coef=args.sell_target_aux_coef,
        flat_buy_mass_penalty_coef=args.flat_buy_mass_penalty_coef,
        flat_buy_mass_penalty_target=args.flat_buy_mass_penalty_target,
        failed_to_sell_margin_coef=args.failed_to_sell_margin_coef,
        failed_to_sell_margin_target=args.failed_to_sell_margin_target,
        failed_to_sell_logit_margin_coef=args.failed_to_sell_logit_margin_coef,
        failed_to_sell_logit_margin_target=args.failed_to_sell_logit_margin_target,
        held_sell_mass_penalty_coef=args.held_sell_mass_penalty_coef,
        held_sell_mass_penalty_target=args.held_sell_mass_penalty_target,
        early_held_sell_mass_penalty_coef=args.early_held_sell_mass_penalty_coef,
        early_held_sell_mass_penalty_target=args.early_held_sell_mass_penalty_target,
        early_held_sell_mass_penalty_hours=args.early_held_sell_mass_penalty_hours,
        early_held_sell_mass_stop_return=args.early_held_sell_mass_stop_return,
        held_policy_loss_weight=args.held_policy_loss_weight,
        sequence_length=args.sequence_length, burn_in=args.burn_in,
        value_coef=args.value_coef,
        update_epochs=args.update_epochs, target_kl=args.target_kl,
        minibatch_size=args.minibatch_size,
    )
    optimizer = Adam(model.parameters(), lr=ppo_config.lr)
    agent = JointPPOAgent(model, optimizer, ppo_config, device)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_dim": input_dim,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "device": str(device),
        "fee_rate": FEE_RATE,
        "spec_version": (
            MODE_HEAD_SPEC_VERSION
            if args.mode_head
            else SELL_TARGET_POLICY_RESIDUAL_SPEC_VERSION
            if model.sell_target_policy_residual_enabled
            else SELL_TARGET_AUX_SPEC_VERSION
            if model.sell_target_aux_enabled
            else SELL_AUX_SPEC_VERSION
            if (
                args.sell_aux_coef > 0.0 or args.sell_clear_aux_coef > 0.0
                or args.sell_clear_contrast_coef > 0.0
                or args.sell_clear_margin_floor_coef > 0.0
                or args.sell_clear_hold_margin_ceiling_coef > 0.0
                or args.sell_margin_align_coef > 0.0
                or args.sell_margin_level_coef > 0.0
            )
            else UNIVERSE_CONTEXT_SPEC_VERSION
            if model.universe_context_enabled
            else CROSS_SECTIONAL_SUPERVISED_CONTEXT_SPEC_VERSION
            if args.cross_sectional_supervised_context
            else SUPERVISED_SCORE_RESIDUAL_SPEC_VERSION
            if model.score_residual_enabled
            else BUY_ALLOCATION_Q_SPEC_VERSION
            if model.flat_action_value_heads
            else ACTION_VALUE_TD_SPEC_VERSION
            if model.action_value_heads and (
                args.sell_q_target == "td0"
                or args.sell_q_detach_encoder
                or args.train_sell_head_only
                or args.sell_q_policy_coef > 0.0
            )
            else ACTION_VALUE_SPEC_VERSION
            if model.action_value_heads
            else BUY_RAW_RESIDUAL_SPEC_VERSION
            if args.buy_raw_residual
            else SELL_RAW_RESIDUAL_SPEC_VERSION
            if args.sell_raw_residual
            else FACTORIZED_SPEC_VERSION
            if args.actor_mode != "shared" or args.state_value_heads
            else TEMPORAL_SPEC_VERSION if args.temporal_width > 0 else SPEC_VERSION
        ),
        "held_feature_index": held_feature_index,
        "action_value_heads": model.action_value_heads,
        "flat_action_value_heads": model.flat_action_value_heads,
        "mode_head": model.mode_head_enabled,
        "sell_target_aux_enabled": model.sell_target_aux_enabled,
        "sell_target_policy_residual_enabled": model.sell_target_policy_residual_enabled,
        "sell_target_policy_residual_width": model.sell_target_policy_residual_width,
        "sell_target_policy_residual_source": model.sell_target_policy_residual_source,
        "score_residual_enabled": model.score_residual_enabled,
        "score_feature_indices": score_feature_indices,
        "boundary_sell_residual": model.boundary_sell_residual_enabled,
        "boundary_sell_feature_index": boundary_sell_feature_index,
        "gated_buy_residual": model.gated_buy_residual_enabled,
        "gated_buy_feature_index": gated_buy_feature_index,
        "gated_buy_feature2_index": gated_buy_feature2_index,
        "gated_buy2_residual": model.gated_buy2_residual_enabled,
        "gated_buy2_feature_index": gated_buy2_feature_index,
        "gated_buy2_feature2_index": gated_buy2_feature2_index,
        "gated_buy3_residual": model.gated_buy3_residual_enabled,
        "gated_buy3_feature_index": gated_buy3_feature_index,
        "gated_buy3_feature2_index": gated_buy3_feature2_index,
        "gated_sell_residual": model.gated_sell_residual_enabled,
        "gated_sell_feature_index": gated_sell_feature_index,
        "gated_sell_feature2_index": gated_sell_feature2_index,
        "buy_excluded_feature_indices": sell_memory_feature_indices,
        "sell_excluded_feature_indices": buy_memory_feature_indices,
        "separate_head_memory": bool(args.separate_head_memory),
        "buy_aux_negative_gate_feature_index": buy_aux_negative_gate_feature_index,
        "flat_buy_mass_penalty_gate_feature_index": flat_buy_mass_penalty_gate_feature_index,
        "held_gae_lambda_resolved": held_gae_lambda,
        "ppo": vars(ppo_config),
        "observation_columns": observation_columns,
        "data_manifest_sha256": (
            sha256(args.data_dir / "manifest.json")
            if (args.data_dir / "manifest.json").exists() else None
        ),
        "source_sha256": {
            str(path): sha256(path) for path in (
                Path("rl/training/trainer_v7.py"), Path("rl/agent/joint_ppo.py"),
                Path("rl/agent/joint_policy.py"), Path("rl/agent/model.py"),
                Path("rl/env/multi_env.py"), Path("rl/env/multi_portfolio.py"),
                Path("rl/env/multi_selector.py"),
                Path("rl/env/coin_history_context.py"),
            )
        },
    }
    parent_payload = None
    parent_input_expansion = False
    if args.parent_checkpoint is not None:
        parent_payload = torch.load(
            args.parent_checkpoint, map_location=device, weights_only=False
        )
        parent_config = parent_payload.get("config", {})
        parent_input_expansion = (
            args.expand_input_checkpoint is not None
            and args.expand_input_checkpoint.resolve() == args.parent_checkpoint.resolve()
        )
        parent_phase = int(parent_config.get("phase", -1))
        parent_is_prior_phase = parent_phase == args.phase - 1
        parent_is_same_phase_curriculum = (
            args.allow_same_phase_parent and args.phase == 3 and parent_phase == args.phase
        )
        if not (parent_is_prior_phase or parent_is_same_phase_curriculum):
            raise RuntimeError("Parent checkpoint is not from the prior phase")
        if parent_is_same_phase_curriculum:
            parent_slots = int(parent_config.get("max_slots", -1))
            if parent_slots < 1 or parent_slots > args.max_slots:
                raise RuntimeError("Same-phase slot curriculum requires non-decreasing max slots")
        report_checkpoints = []
        for report_path in gate.get("reports", []):
            report = json.loads(Path(report_path).read_text())
            if report.get("checkpoint"):
                report_checkpoints.append(Path(report["checkpoint"]).resolve())
        if args.parent_checkpoint.resolve() not in report_checkpoints:
            raise RuntimeError("Parent checkpoint is not covered by the passed gate")
        parent_architecture = (
            input_dim if parent_input_expansion else int(parent_config.get("input_dim", -1)),
            int(parent_config.get("width", -1)),
            int(parent_config.get("blocks", -1)),
            int(parent_config.get("sell_head_width", 0)),
            int(parent_config.get("temporal_width", 0)),
            parent_config.get("actor_mode", "shared"),
            bool(parent_config.get("state_value_heads", False)),
            bool(parent_config.get("action_value_heads", False)),
            bool(parent_config.get("sell_q_detach_encoder", False)),
            bool(parent_config.get("sell_raw_residual", False)),
            int(parent_config.get("sell_raw_residual_width", 0)),
            bool(parent_config.get("buy_raw_residual", False)),
            int(parent_config.get("buy_raw_residual_width", 0)),
            bool(parent_config.get("mode_head", False)),
            int(parent_config.get("sell_target_aux_width", 0) or 0),
            bool(parent_config.get(
                "sell_target_policy_residual",
                parent_config.get("sell_target_policy_residual_enabled", False),
            )),
            int(parent_config.get("sell_target_policy_residual_width", 0) or 0),
            str(parent_config.get("sell_target_policy_residual_source", "score")),
        )
        current_architecture = (
            input_dim, args.width, args.blocks, args.sell_head_width,
            args.temporal_width, args.actor_mode, args.state_value_heads,
            model.action_value_heads,
            args.sell_q_detach_encoder,
            args.sell_raw_residual,
            args.sell_raw_residual_width,
            args.buy_raw_residual,
            args.buy_raw_residual_width,
            args.mode_head,
            args.sell_target_aux_width,
            args.sell_target_policy_residual,
            args.sell_target_policy_residual_width,
            args.sell_target_policy_residual_source,
        )
        if parent_architecture != current_architecture:
            raise RuntimeError("Architecture mismatch in parent checkpoint")
        if (
            not parent_input_expansion
            and parent_config.get("observation_columns") != config["observation_columns"]
        ):
            raise RuntimeError("Observation schema mismatch in parent checkpoint")
        if float(parent_config.get("fee_rate", -1.0)) != float(config["fee_rate"]):
            raise RuntimeError("Fee mismatch in parent checkpoint")
        if parent_config.get("buy_allocations", "0.25,0.50,0.75,1.00") != args.buy_allocations:
            raise RuntimeError("Allocation semantics mismatch in parent checkpoint")
        config["parent_provenance"] = {
            "path": str(args.parent_checkpoint),
            "sha256": sha256(args.parent_checkpoint),
            "phase": int(parent_config["phase"]),
            "episode": int(parent_payload["episode"]),
            "gate": str(args.prerequisite_gate),
            "gate_sha256": sha256(args.prerequisite_gate),
            "optimizer_transferred": False,
        }
    episode_start = 0
    coverage = {
        "episodes_seen": 0,
        "selector_windows": selector.window_count,
        "completed_selector_cycles": 0,
        "eligible_anchor_coins": len(selector.eligible_anchor_coins),
    }
    if args.resume:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("spec_version") != config["spec_version"] or payload.get("config", {}).get("input_dim") != input_dim:
            raise RuntimeError("Incompatible v7 checkpoint")
        validate_resume_config(payload["config"], config)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        selector.load_state(payload["selector_state"])
        if payload.get("python_random_state") is not None:
            random.setstate(payload["python_random_state"])
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.set_rng_state(payload["torch_random_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all([
                state.cpu() for state in payload["cuda_random_state_all"]
            ])
        episode_start = int(payload["episode"]) + 1
        saved_coverage = payload.get("training_coverage")
        if saved_coverage is not None:
            coverage.update(saved_coverage)
        else:
            # v7 checkpoints created before explicit coverage accounting still
            # have aligned episode numbering and deterministic selector state.
            coverage["episodes_seen"] = episode_start
            coverage["completed_selector_cycles"] = (
                episode_start // max(1, coverage["selector_windows"])
                if type(selector) is MultiCoinSelector
                else selector.completed_selector_cycles
            )
    elif parent_payload is not None:
        if parent_input_expansion:
            expand_input_checkpoint(
                model,
                parent_payload["model_state"],
                parent_payload["config"]["observation_columns"],
                config["observation_columns"],
            )
        else:
            model.load_state_dict(parent_payload["model_state"])
    elif args.expand_input_checkpoint:
        payload = torch.load(
            args.expand_input_checkpoint, map_location=device, weights_only=False
        )
        source = payload.get("config", {})
        architecture = (
            int(source.get("width", -1)), int(source.get("blocks", -1)),
            int(source.get("sell_head_width", 0)), int(source.get("temporal_width", 0)),
            source.get("actor_mode", "shared"), bool(source.get("state_value_heads", False)),
            bool(source.get("action_value_heads", False)),
            bool(source.get("sell_q_detach_encoder", False)),
            bool(source.get("sell_raw_residual", False)),
            int(source.get("sell_raw_residual_width", 0)),
            bool(source.get("buy_raw_residual", False)),
            int(source.get("buy_raw_residual_width", 0)),
            bool(source.get("mode_head", False)),
            int(source.get("sell_target_aux_width", 0) or 0),
            bool(source.get(
                "sell_target_policy_residual",
                source.get("sell_target_policy_residual_enabled", False),
            )),
            int(source.get("sell_target_policy_residual_width", 0) or 0),
            str(source.get("sell_target_policy_residual_source", "score")),
        )
        requested = (
            args.width, args.blocks, args.sell_head_width, args.temporal_width,
            args.actor_mode, args.state_value_heads,
            model.action_value_heads,
            args.sell_q_detach_encoder,
            args.sell_raw_residual,
            args.sell_raw_residual_width,
            args.buy_raw_residual,
            args.buy_raw_residual_width,
            args.mode_head,
            args.sell_target_aux_width,
            args.sell_target_policy_residual,
            args.sell_target_policy_residual_width,
            args.sell_target_policy_residual_source,
        )
        if architecture != requested:
            raise RuntimeError("Architecture mismatch in input-expansion checkpoint")
        expand_input_checkpoint(
            model, payload["model_state"], source["observation_columns"],
            config["observation_columns"],
        )
    elif args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        source = payload.get("config", {})
        if (
            not init_checkpoint_spec_is_compatible(payload, source)
            or int(source.get("input_dim", -1)) != input_dim
        ):
            raise RuntimeError("Incompatible v7 initialization checkpoint")
        if (
            int(source.get("width", -1)) != args.width
            or int(source.get("blocks", -1)) != args.blocks
            or int(source.get("sell_head_width", 0)) != args.sell_head_width
            or int(source.get("temporal_width", 0)) != args.temporal_width
            or bool(source.get("state_value_heads", False)) != args.state_value_heads
        ):
            raise RuntimeError("Architecture mismatch in v7 initialization checkpoint")
        source_has_action_value = bool(source.get("action_value_heads", False))
        source_has_flat_action_value = bool(source.get("flat_action_value_heads", False))
        source_has_residual = bool(source.get("sell_raw_residual", False))
        source_has_buy_residual = bool(source.get("buy_raw_residual", False))
        source_has_score_residual = bool(source.get("score_residual", False)) or bool(
            source.get("score_residual_enabled", False)
        )
        source_has_mode_head = bool(source.get("mode_head", False))
        source_has_boundary_sell_residual = bool(source.get("boundary_sell_residual", False))
        source_has_gated_buy_residual = bool(source.get("gated_buy_residual", False))
        source_has_gated_buy2_residual = bool(source.get("gated_buy2_residual", False))
        source_has_gated_buy3_residual = bool(source.get("gated_buy3_residual", False))
        source_has_gated_sell_residual = bool(source.get("gated_sell_residual", False))
        source_has_sell_target_aux = int(source.get("sell_target_aux_width", 0) or 0) > 0
        source_has_sell_target_policy_residual = bool(
            source.get(
                "sell_target_policy_residual",
                source.get("sell_target_policy_residual_enabled", False),
            )
        )
        adding_actor_separation = (
            source.get("actor_mode", "shared") == "shared"
            and args.actor_mode == "separate"
        )
        if source.get("actor_mode", "shared") != args.actor_mode and not adding_actor_separation:
            raise RuntimeError("Architecture mismatch in v7 initialization checkpoint")
        if source_has_action_value and not model.action_value_heads:
            raise RuntimeError("Cannot remove an action-value head during initialization")
        if source_has_flat_action_value and not model.flat_action_value_heads:
            raise RuntimeError("Cannot remove a flat action-value head during initialization")
        if source_has_residual and not args.sell_raw_residual:
            raise RuntimeError("Cannot remove a sell residual during initialization")
        if source_has_buy_residual and not args.buy_raw_residual:
            raise RuntimeError("Cannot remove a buy residual during initialization")
        if source_has_score_residual and not args.score_residual:
            raise RuntimeError("Cannot remove a score residual during initialization")
        if source_has_mode_head and not args.mode_head:
            raise RuntimeError("Cannot remove a mode head during initialization")
        if source_has_boundary_sell_residual and not args.boundary_sell_residual:
            raise RuntimeError("Cannot remove a boundary sell residual during initialization")
        if source_has_gated_buy_residual and not args.gated_buy_residual:
            raise RuntimeError("Cannot remove a gated buy residual during initialization")
        if source_has_gated_buy2_residual and not args.gated_buy2_residual:
            raise RuntimeError("Cannot remove a second gated buy residual during initialization")
        if source_has_gated_buy3_residual and not args.gated_buy3_residual:
            raise RuntimeError("Cannot remove a third gated buy residual during initialization")
        if source_has_gated_sell_residual and not args.gated_sell_residual:
            raise RuntimeError("Cannot remove a gated sell residual during initialization")
        if source_has_sell_target_aux and args.sell_target_aux_width <= 0:
            raise RuntimeError("Cannot remove a sell-target auxiliary head during initialization")
        if source_has_sell_target_policy_residual and not args.sell_target_policy_residual:
            raise RuntimeError("Cannot remove a sell-target policy residual during initialization")
        if (
            source_has_sell_target_policy_residual
            and int(source.get("sell_target_policy_residual_width", 0) or 0)
            != args.sell_target_policy_residual_width
        ):
            raise RuntimeError("Cannot change sell-target policy residual width during initialization")
        if (
            source_has_sell_target_policy_residual
            and str(source.get("sell_target_policy_residual_source", "score"))
            != args.sell_target_policy_residual_source
        ):
            raise RuntimeError("Cannot change sell-target policy residual source during initialization")
        if (
            source_has_sell_target_aux
            and int(source.get("sell_target_aux_width", 0) or 0) != args.sell_target_aux_width
        ):
            raise RuntimeError("Cannot change sell-target auxiliary width during initialization")
        adding_action_value = model.action_value_heads and not source_has_action_value
        adding_flat_action_value = model.flat_action_value_heads and not source_has_flat_action_value
        adding_residual = args.sell_raw_residual and not source_has_residual
        adding_buy_residual = args.buy_raw_residual and not source_has_buy_residual
        adding_score_residual = args.score_residual and not source_has_score_residual
        adding_mode_head = args.mode_head and not source_has_mode_head
        adding_boundary_sell_residual = (
            args.boundary_sell_residual and not source_has_boundary_sell_residual
        )
        adding_gated_buy_residual = args.gated_buy_residual and not source_has_gated_buy_residual
        adding_gated_buy2_residual = args.gated_buy2_residual and not source_has_gated_buy2_residual
        adding_gated_buy3_residual = args.gated_buy3_residual and not source_has_gated_buy3_residual
        adding_gated_sell_residual = args.gated_sell_residual and not source_has_gated_sell_residual
        adding_sell_target_aux = args.sell_target_aux_width > 0 and not source_has_sell_target_aux
        adding_sell_target_policy_residual = (
            args.sell_target_policy_residual
            and not source_has_sell_target_policy_residual
        )
        if sum(bool(value) for value in (
            adding_buy_residual, adding_action_value, adding_flat_action_value,
            adding_residual, adding_score_residual, adding_mode_head,
            adding_boundary_sell_residual, adding_actor_separation, adding_sell_target_aux,
            adding_sell_target_policy_residual, adding_gated_buy_residual,
            adding_gated_buy2_residual, adding_gated_buy3_residual,
            adding_gated_sell_residual,
        )) > 1:
            raise RuntimeError("Expand one experimental path at a time")
        if adding_buy_residual:
            initialize_zero_buy_residual(model, payload["model_state"])
        elif adding_actor_separation:
            initialize_separate_actor_from_shared(model, payload["model_state"])
        elif adding_sell_target_aux:
            initialize_sell_target_aux_head(model, payload["model_state"])
        elif adding_sell_target_policy_residual:
            initialize_sell_target_policy_residual(model, payload["model_state"])
        elif adding_score_residual:
            initialize_zero_score_residual(model, payload["model_state"])
        elif adding_mode_head:
            initialize_zero_mode_head(
                model, payload["model_state"],
                args.initial_buy_prob, args.initial_sell_prob,
            )
        elif adding_boundary_sell_residual:
            initialize_zero_boundary_sell_residual(model, payload["model_state"])
        elif adding_gated_buy_residual:
            initialize_zero_gated_buy_residual(model, payload["model_state"])
        elif adding_gated_buy2_residual:
            initialize_zero_gated_buy2_residual(model, payload["model_state"])
        elif adding_gated_buy3_residual:
            initialize_zero_gated_buy3_residual(model, payload["model_state"])
        elif adding_gated_sell_residual:
            initialize_zero_gated_sell_residual(model, payload["model_state"])
        elif adding_flat_action_value:
            initialize_zero_flat_action_value_head(model, payload["model_state"])
        elif adding_action_value and adding_residual:
            initialize_zero_action_value_and_sell_residual(
                model, payload["model_state"]
            )
        elif adding_action_value:
            initialize_zero_action_value_head(model, payload["model_state"])
        elif adding_residual:
            initialize_zero_sell_residual(model, payload["model_state"])
        else:
            model.load_state_dict(payload["model_state"])
    if args.resume:
        write_config_once(
            args.run_dir / f"invocation_resume_from_{episode_start - 1:06d}.json",
            config,
        )
    else:
        write_config_once(args.run_dir / "config.json", config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    metrics_path = args.run_dir / "metrics.jsonl"
    initial_path = args.run_dir / "checkpoint_initial.pt"
    if not args.resume and not initial_path.exists():
        save_checkpoint(
            initial_path, -1, model, optimizer, selector, config,
            {"kind": "pre_training_initialization"}, coverage,
        )
    update_buffer = JointEpisodeBuffer()
    for episode in range(episode_start, episode_start + args.episodes):
        coins, frames = selector.next_episode()
        env = MultiCoinTradingEnv(
            frames, max_slots=min(args.max_slots, len(frames)),
            starting_cash=STARTING_CASH, fee_rate=FEE_RATE,
            fee_round_decimals=FEE_ROUND_DECIMALS,
            supervised_targets=supervised_targets,
            supervised_raw_targets=supervised_raw_targets,
            neutral_supervised_targets=neutral_supervised_targets,
            expose_episode_boundary=args.expose_episode_boundary,
            position_dynamics_mode=args.position_dynamics_mode,
            drawdown_penalty=args.drawdown_penalty,
            forced_sell_penalty=args.forced_sell_penalty,
            turnover_penalty=args.turnover_penalty,
            sell_realization_reward_coef=args.sell_realization_reward_coef,
            sell_aux_horizon_hours=args.sell_aux_horizon_hours,
            sell_aux_margin=args.sell_aux_margin,
            sell_aux_min_return=args.sell_aux_min_return,
            sell_aux_min_hold_hours=args.sell_aux_min_hold_hours,
            sell_aux_early_stop_return=args.sell_aux_early_stop_return,
            sell_clear_aux_horizon_hours=args.sell_clear_aux_horizon_hours,
            sell_clear_aux_hold_opportunity_min=args.sell_clear_aux_hold_opportunity_min,
            sell_clear_aux_avoid_loss_min=args.sell_clear_aux_avoid_loss_min,
            sell_clear_aux_stop_return=args.sell_clear_aux_stop_return,
            sell_clear_aux_exclude_conflict=args.sell_clear_aux_exclude_conflict,
            sell_margin_align_horizon_hours=args.sell_margin_align_horizon_hours,
            sell_margin_align_target_mode=args.sell_margin_align_target_mode,
            sell_margin_align_clear_hold_opportunity_min=(
                args.sell_margin_align_clear_hold_opportunity_min
            ),
            sell_margin_align_clear_avoid_loss_min=(
                args.sell_margin_align_clear_avoid_loss_min
            ),
            sell_margin_align_clear_stop_return=args.sell_margin_align_clear_stop_return,
            sell_margin_align_exclude_conflict=args.sell_margin_align_exclude_conflict,
            sell_margin_align_hold_weight=args.sell_margin_align_hold_weight,
            sell_margin_align_scale=args.sell_margin_align_scale,
            buy_aux_horizon_hours=args.buy_aux_horizon_hours,
            buy_aux_min_return=args.buy_aux_min_return,
            buy_allocations=buy_allocations,
            allocation_sizing_mode=args.allocation_sizing_mode,
            cross_sectional_supervised_context=args.cross_sectional_supervised_context,
            coin_history_context=args.coin_history_context,
            execution_price_mode=args.execution_price_mode,
            execution_cost_bps=args.execution_cost_bps,
        )
        initial_allocation = (
            args.initial_held_allocation
            if random.random() < args.initial_held_episode_prob else None
        )
        obs = env.reset(
            initial_position_allocation=initial_allocation,
            initial_position_coin=coins[0] if initial_allocation is not None else None,
        )
        episode_steps = 0
        episode_training_reward = 0.0
        episode_raw_equity_reward = 0.0
        episode_drawdown_penalty = 0.0
        episode_forced_sell_penalty = 0.0
        episode_turnover_penalty = 0.0
        episode_sell_realization_reward = 0.0
        equity_curve, action_counts = [], {"buy": 0, "sell": 0, "noop": 0}
        started = time.perf_counter()
        done = False
        episode_buffer_start = len(update_buffer.obs)
        recurrent_hidden = model.initial_state(len(coins), device)
        while not done:
            held = env.held_mask()
            allow_buy = len(env.portfolio.positions) < env.portfolio.max_slots and env.portfolio.cash > 0.0
            hidden_before = recurrent_hidden
            output = agent.act(obs, recurrent_hidden)
            recurrent_hidden = output.pop("next_hidden")
            next_obs, reward, done, info = env.step(output, deterministic=False)
            training_reward = reward
            bootstrap_value = None
            time_limit_truncated = False
            if args.time_limit_bootstrap and done and info["time_limit_truncated"]:
                terminal_obs = info["time_limit_observation"].to(device)
                with torch.no_grad():
                    terminal_output, _ = model.forward_step(terminal_obs, recurrent_hidden)
                    bootstrap_value = float(terminal_output["value"].mean())
                training_reward = float(info["time_limit_raw_equity_reward"])
                time_limit_truncated = True
            update_buffer.add(
                obs, held, allow_buy, info["joint_action_index"], info["logp"],
                output["portfolio_value"], training_reward, done,
                recurrent_hidden=hidden_before, time_limit_truncated=time_limit_truncated,
                bootstrap_value=bootstrap_value,
                sell_aux_label=info.get("sell_aux_label"),
                sell_aux_mask=info.get("sell_aux_mask", False),
                sell_clear_aux_label=info.get("sell_clear_aux_label"),
                sell_clear_aux_mask=info.get("sell_clear_aux_mask", False),
                sell_margin_align_target=info.get("sell_margin_align_target"),
                sell_margin_align_mask=info.get("sell_margin_align_mask", False),
                sell_margin_best_hold_opportunity=info.get(
                    "sell_margin_align_best_hold_opportunity", 0.0,
                ),
                sell_margin_avoided_loss=info.get(
                    "sell_margin_align_avoided_loss", 0.0,
                ),
                sell_margin_current_return=info.get(
                    "sell_margin_align_current_return", 0.0,
                ),
                held_position_age=info.get("held_position_age", 0.0),
                held_position_return=info.get("held_position_return_after_fees", 0.0),
                buy_aux_target=info.get("buy_aux_target"),
                buy_aux_mask=info.get("buy_aux_mask", False),
            )
            if done and env.forced_sell_count > 0 and args.failed_to_sell_hindsight_hours > 0:
                update_buffer.mark_failed_to_sell(
                    episode_buffer_start,
                    lookback_steps=args.failed_to_sell_hindsight_hours,
                    min_hold_steps=args.failed_to_sell_min_hold_hours,
                )
            action_counts[info["action"]] += 1
            episode_training_reward += float(training_reward)
            episode_raw_equity_reward += float(info["raw_equity_reward"])
            episode_drawdown_penalty += float(info["drawdown_penalty_applied"])
            episode_forced_sell_penalty += float(info["forced_sell_penalty_applied"])
            episode_turnover_penalty += float(info["turnover_penalty_applied"])
            episode_sell_realization_reward += float(
                info["sell_realization_reward_applied"]
            )
            equity_curve.append(info["equity"])
            episode_steps += 1
            obs = next_obs
        episodes_completed = episode - episode_start + 1
        coverage["episodes_seen"] = int(coverage["episodes_seen"]) + 1
        coverage["completed_selector_cycles"] = (
            int(coverage["episodes_seen"]) // max(1, int(coverage["selector_windows"]))
            if type(selector) is MultiCoinSelector
            else selector.completed_selector_cycles
        )
        should_update = (
            episodes_completed % args.episodes_per_update == 0
            or episodes_completed == args.episodes
        )
        ppo_metrics = agent.update(update_buffer) if should_update else {}
        if should_update:
            update_buffer = JointEpisodeBuffer()
        equity = np.asarray(equity_curve, dtype=np.float64)
        peaks = np.maximum.accumulate(equity)
        pnl = np.asarray(env.trade_pnls, dtype=np.float64)
        gross_profit = float(pnl[pnl > 0].sum()) if pnl.size else 0.0
        gross_loss = float(-pnl[pnl < 0].sum()) if pnl.size else 0.0
        metrics = {
            "episode": episode,
            "episode_days": int(getattr(selector, "last_episode_days", args.episode_days)),
            "anchor_coin": coins[0],
            "coins": len(coins),
            "initial_holding": initial_allocation is not None,
            "return": float(equity[-1] / STARTING_CASH - 1.0),
            "training_reward_sum": episode_training_reward,
            "raw_equity_reward_sum": episode_raw_equity_reward,
            "drawdown_penalty_sum": episode_drawdown_penalty,
            "forced_sell_penalty_sum": episode_forced_sell_penalty,
            "turnover_penalty_sum": episode_turnover_penalty,
            "sell_realization_reward_sum": episode_sell_realization_reward,
            "max_drawdown": float(((equity - peaks) / np.maximum(peaks, 1e-12)).min()),
            "profit_factor": gross_profit / gross_loss if gross_loss else None,
            "expectancy_dollars": float(pnl.mean()) if pnl.size else 0.0,
            "trades": int(pnl.size),
            "manual_sells": env.manual_sell_count,
            "forced_sells": env.forced_sell_count,
            "actions": action_counts,
            "steps_per_second": episode_steps / max(time.perf_counter() - started, 1e-9),
            "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
            "ppo": ppo_metrics,
        }
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if should_update:
            save_checkpoint(
                checkpoint_path, episode, model, optimizer, selector, config,
                metrics, coverage,
            )
        if should_update and (episode + 1) % (args.episodes_per_update * 5) == 0:
            immutable = args.run_dir / f"checkpoint_episode_{episode:06d}.pt"
            if immutable.exists():
                raise RuntimeError(f"Refusing to overwrite immutable checkpoint: {immutable}")
            save_checkpoint(
                immutable, episode, model, optimizer, selector, config, metrics,
                coverage,
            )
        if should_update and episodes_completed == args.episodes:
            immutable = args.run_dir / f"checkpoint_episode_{episode:06d}.pt"
            if not immutable.exists():
                save_checkpoint(
                    immutable, episode, model, optimizer, selector, config,
                    metrics, coverage,
                )
        print(f"[EP {episode}] return={metrics['return']:+.3%} trades={metrics['trades']} manual={env.manual_sell_count} forced={env.forced_sell_count} sps={metrics['steps_per_second']:.1f}")


if __name__ == "__main__":
    main()
