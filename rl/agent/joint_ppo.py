"""PPO update for the v7 joint full-universe categorical policy."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from rl.agent.joint_policy import batched_joint_action_logits
from rl.config import BUY_ALLOCATIONS


@dataclass
class JointPPOConfig:
    # Undiscounted episodic finance objective: dense equity deltas telescope
    # exactly to final after-fee PnL, without making late losses/fees cheaper.
    gamma: float = 1.0
    gae_lambda: float = 0.95
    # Sell/hold consequences are mostly future avoided returns because marked
    # equity already includes the realizable exit fee.  This optional longer
    # horizon applies only while any position is held; flat buy/no-buy states
    # retain the lower-variance default above.
    held_gae_lambda: float = 0.95
    advantage_normalization: str = "global"
    sell_q_coef: float = 0.0
    sell_q_advantage_mix: float = 0.0
    sell_q_target: str = "mc"
    sell_q_policy_coef: float = 0.0
    buy_q_coef: float = 0.0
    buy_q_advantage_mix: float = 0.0
    buy_allocation_q_coef: float = 0.0
    buy_allocation_q_advantage_mix: float = 0.0
    sell_aux_coef: float = 0.0
    sell_aux_positive_weight: float = 1.0
    sell_clear_aux_coef: float = 0.0
    sell_clear_aux_positive_weight: float = 1.0
    sell_clear_contrast_coef: float = 0.0
    sell_clear_contrast_margin: float = 0.25
    sell_clear_margin_floor_coef: float = 0.0
    sell_clear_margin_floor: float = -0.25
    sell_clear_margin_floor_max_return: float | None = None
    sell_clear_hold_margin_ceiling_coef: float = 0.0
    sell_clear_hold_margin_ceiling: float = -0.75
    sell_margin_align_coef: float = 0.0
    sell_margin_align_clear_hold_opportunity_min: float = 0.06
    sell_margin_align_clear_avoid_loss_min: float = 0.03
    sell_margin_align_clear_stop_return: float = -0.03
    sell_margin_level_coef: float = 0.0
    sell_margin_level_scale: float = 1.0
    sell_margin_level_hold_opportunity_min: float = 0.0
    sell_margin_level_hold_weight: float = 1.0
    sell_margin_level_sell_avoid_loss_min: float = 0.0
    sell_margin_level_sell_weight: float = 1.0
    sell_margin_conflict_uncertainty_coef: float = 0.0
    sell_margin_conflict_uncertainty_scale: float = 1.0
    sell_margin_conflict_hold_opportunity_min: float = 0.0
    sell_margin_conflict_avoid_loss_min: float = 0.0
    sell_margin_conflict_stop_return: float = 0.0
    sell_margin_subtype_confidence_coef: float = 0.0
    sell_target_aux_coef: float = 0.0
    flat_buy_mass_penalty_coef: float = 0.0
    flat_buy_mass_penalty_target: float = 0.0
    flat_buy_mass_penalty_gate_feature_index: int = -1
    flat_buy_mass_penalty_gate_min: float = float("-inf")
    flat_buy_mass_penalty_gate_max: float = float("inf")
    failed_to_sell_margin_coef: float = 0.0
    failed_to_sell_margin_target: float = 0.0
    failed_to_sell_logit_margin_coef: float = 0.0
    failed_to_sell_logit_margin_target: float = 0.0
    held_sell_mass_penalty_coef: float = 0.0
    held_sell_mass_penalty_target: float = 0.0
    early_held_sell_mass_penalty_coef: float = 0.0
    early_held_sell_mass_penalty_target: float = 0.0
    early_held_sell_mass_penalty_hours: int = 0
    early_held_sell_mass_stop_return: float = -0.03
    buy_aux_coef: float = 0.0
    buy_aux_rank_only: bool = False
    buy_aux_negative_coef: float = 0.0
    buy_aux_negative_buy_mass_target: float = 0.0
    buy_aux_negative_gate_feature_index: int = -1
    buy_aux_negative_gate_min: float = float("-inf")
    buy_aux_negative_gate_max: float = float("inf")
    held_policy_loss_weight: float = 1.0
    sequence_length: int = 128
    burn_in: int = 32
    clip_eps: float = 0.15
    lr: float = 1e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    target_kl: float = 0.03
    eps: float = 1e-8


class JointEpisodeBuffer:
    def __init__(self):
        self.obs = []
        self.held = []
        self.coin_counts = []
        self.allow_buy = []
        self.actions = []
        self.logp = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.time_limit_truncations = []
        self.bootstrap_values = []
        self.recurrent_hidden = []
        self.sell_aux_labels = []
        self.sell_aux_masks = []
        self.sell_clear_aux_labels = []
        self.sell_clear_aux_masks = []
        self.sell_margin_align_targets = []
        self.sell_margin_align_masks = []
        self.sell_margin_best_hold_opportunities = []
        self.sell_margin_avoided_losses = []
        self.sell_margin_current_returns = []
        self.held_position_ages = []
        self.held_position_returns = []
        self.failed_to_sell_masks = []
        self.buy_aux_targets = []
        self.buy_aux_masks = []

    def add(
        self, obs, held, allow_buy, action, logp, value, reward, done,
        recurrent_hidden=None, time_limit_truncated=False, bootstrap_value=None,
        sell_aux_label=None, sell_aux_mask=False,
        sell_clear_aux_label=None, sell_clear_aux_mask=False,
        sell_margin_align_target=None, sell_margin_align_mask=False,
        sell_margin_best_hold_opportunity=0.0, sell_margin_avoided_loss=0.0,
        sell_margin_current_return=0.0,
        held_position_age=0.0, held_position_return=0.0,
        buy_aux_target=None, buy_aux_mask=False,
    ):
        self.obs.append(obs.detach().cpu())
        self.held.append(held.detach().cpu())
        self.coin_counts.append(int(obs.shape[0]))
        self.allow_buy.append(bool(allow_buy))
        self.actions.append(int(action))
        self.logp.append(logp.detach().cpu())
        self.values.append(value.detach().cpu())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.time_limit_truncations.append(float(time_limit_truncated))
        self.bootstrap_values.append(
            float(bootstrap_value) if bootstrap_value is not None else 0.0
        )
        self.recurrent_hidden.append(
            None if recurrent_hidden is None else recurrent_hidden.detach().cpu()
        )
        self.sell_aux_labels.append(
            float(sell_aux_label) if sell_aux_label is not None else 0.0
        )
        self.sell_aux_masks.append(bool(sell_aux_mask))
        self.sell_clear_aux_labels.append(
            float(sell_clear_aux_label) if sell_clear_aux_label is not None else 0.0
        )
        self.sell_clear_aux_masks.append(bool(sell_clear_aux_mask))
        self.sell_margin_align_targets.append(
            float(sell_margin_align_target) if sell_margin_align_target is not None else 0.0
        )
        self.sell_margin_align_masks.append(bool(sell_margin_align_mask))
        self.sell_margin_best_hold_opportunities.append(float(sell_margin_best_hold_opportunity))
        self.sell_margin_avoided_losses.append(float(sell_margin_avoided_loss))
        self.sell_margin_current_returns.append(float(sell_margin_current_return))
        self.held_position_ages.append(float(held_position_age))
        self.held_position_returns.append(float(held_position_return))
        self.failed_to_sell_masks.append(False)
        self.buy_aux_targets.append(
            int(buy_aux_target) if buy_aux_target is not None else -1
        )
        self.buy_aux_masks.append(bool(buy_aux_mask))

    def mark_failed_to_sell(
        self, start_index: int, lookback_steps: int, min_hold_steps: int = 0,
    ) -> int:
        """Mark hindsight held states from an episode ending in liquidation.

        This creates dense training signal from a single terminal forced sell:
        late held states become examples where deterministic SELL should beat
        the strongest non-SELL mode.  It does not affect environment behavior
        or evaluation accounting.
        """
        if lookback_steps <= 0:
            return 0
        stop = len(self.failed_to_sell_masks)
        begin = max(int(start_index), stop - int(lookback_steps))
        marked = 0
        for idx in range(begin, stop):
            held = self.held[idx]
            is_held = bool(torch.as_tensor(held).bool().any().item())
            if is_held and float(self.held_position_ages[idx]) >= float(min_hold_steps):
                if not self.failed_to_sell_masks[idx]:
                    marked += 1
                self.failed_to_sell_masks[idx] = True
        return marked


def sell_aux_labels_to_class_indices(labels: torch.Tensor) -> torch.Tensor:
    """Map SELL-positive environment labels to [SELL, HOLD] class indices."""
    return torch.where(
        labels > 0.5,
        torch.zeros_like(labels, dtype=torch.long),
        torch.ones_like(labels, dtype=torch.long),
    )


def normalized_margin_alignment_loss(
    margins: torch.Tensor, targets: torch.Tensor, eps: float,
) -> torch.Tensor:
    """Scale-free alignment loss for SELL-HOLD margins.

    This intentionally removes the minibatch mean from both tensors so the
    auxiliary objective learns relative sell/hold ordering rather than a hard
    absolute exit threshold.  Future-derived targets must only be supplied by
    training-time code.
    """
    if margins.numel() < 2:
        return margins.new_zeros(())
    target_std = targets.std(unbiased=False)
    if float(target_std.detach()) <= eps:
        return margins.new_zeros(())
    margin_z = (margins - margins.mean()) / (margins.std(unbiased=False) + eps)
    target_z = (targets - targets.mean()) / (target_std + eps)
    return F.mse_loss(margin_z, target_z)


def bounded_margin_level_loss(
    margins: torch.Tensor, targets: torch.Tensor, scale: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Absolute SELL-HOLD boundary loss for continuous training targets.

    Unlike ``normalized_margin_alignment_loss``, this keeps the target level:
    target 0 maps to a zero SELL-HOLD logit margin, positive values favor SELL,
    and negative values favor HOLD.  The tanh bound keeps the loss finite and
    avoids requesting unbounded logits from future-derived training labels.
    """
    if margins.numel() == 0:
        return margins.new_zeros(())
    loss = (torch.tanh(margins / float(scale)) - targets).square()
    if weights is not None:
        weights = weights.to(loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1e-8)
    return loss.mean()


def state_dependent_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    held_states: torch.Tensor,
    gamma: float,
    flat_lambda: float,
    held_lambda: float,
    time_limit_truncations: torch.Tensor | None = None,
    bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with optional critic bootstrapping at a fixed data-window limit."""
    advantages = torch.zeros_like(rewards)
    last_advantage = rewards.new_zeros(())
    for timestep in reversed(range(len(rewards))):
        not_done = 1.0 - dones[timestep]
        truncated = (
            time_limit_truncations[timestep]
            if time_limit_truncations is not None else rewards.new_zeros(())
        )
        if timestep + 1 < len(values) and not_done:
            next_value = values[timestep + 1]
        elif truncated and bootstrap_values is not None:
            next_value = bootstrap_values[timestep]
        else:
            next_value = values.new_zeros(())
        # A truncation bootstraps the one-step TD residual but does not carry
        # GAE recursively into a separately reset episode.
        delta = rewards[timestep] + gamma * next_value * (not_done + truncated) - values[timestep]
        lam = held_lambda if bool(held_states[timestep]) else flat_lambda
        last_advantage = delta + gamma * lam * not_done * last_advantage
        advantages[timestep] = last_advantage
    return advantages, advantages + values


def normalize_advantages(
    advantages: torch.Tensor,
    held_states: torch.Tensor,
    mode: str,
    eps: float,
    actions: torch.Tensor | None = None,
    coins: int | None = None,
) -> torch.Tensor:
    """Normalize globally or independently for flat and held decisions."""
    if mode == "global":
        return (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + eps
        )
    if mode == "state_scale":
        normalized = advantages.clone()
        for mask in (held_states.bool(), ~held_states.bool()):
            if mask.any():
                values = advantages[mask]
                normalized[mask] = values / (
                    values.std(unbiased=False) + eps
                )
        return normalized
    if mode == "decision_scale":
        if actions is None or coins is None:
            raise ValueError("decision_scale requires actions and coin count")
        first_sell = 1 + coins * len(BUY_ALLOCATIONS)
        groups = (
            actions.eq(0),
            actions.ge(1) & actions.lt(first_sell),
            actions.ge(first_sell),
        )
        normalized = advantages.clone()
        for mask in groups:
            if mask.any():
                values = advantages[mask]
                normalized[mask] = values / (
                    values.std(unbiased=False) + eps
                )
        return normalized
    if mode != "state":
        raise ValueError(
            "advantage normalization must be global, state, state_scale, or decision_scale"
        )
    normalized = advantages.clone()
    for mask in (held_states.bool(), ~held_states.bool()):
        if mask.any():
            values = advantages[mask]
            normalized[mask] = (values - values.mean()) / (
                values.std(unbiased=False) + eps
            )
    return normalized


def monte_carlo_returns(rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = rewards.new_zeros(())
    for timestep in reversed(range(len(rewards))):
        running = rewards[timestep] + (1.0 - dones[timestep]) * running
        returns[timestep] = running
    return returns


def one_step_td_returns(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Low-variance causal action target from the realized next state."""
    next_values = torch.cat((values[1:], values.new_zeros(1)))
    return rewards + gamma * (1.0 - dones) * next_values


def sell_q_credit(
    sell_q: torch.Tensor, buy_logits: torch.Tensor, sell_logits: torch.Tensor,
    held: torch.Tensor, allow_buy: torch.Tensor, actions: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return joint-policy-conditional sell advantage and chosen Q.

    For a held coin, every complete joint action other than selling that coin
    means continuing to hold it.  The baseline therefore uses the sell coin's
    marginal probability in the full NOOP/buy/sell categorical.  This reduces
    to the local sell/hold softmax in Phase 1, while remaining exact when later
    phases expose other positions and legal buys.
    """
    timesteps, coins, _ = sell_q.shape
    valid = torch.ones_like(held, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    if valid.shape != held.shape:
        raise ValueError("Valid mask shape must match held mask")
    held = held.bool() & valid
    sell_offset = 1 + coins * len(BUY_ALLOCATIONS)
    sell_indices = sell_offset + torch.arange(coins, device=actions.device)
    chose_sell = actions[:, None].eq(sell_indices[None, :])
    chosen_q = torch.where(chose_sell, sell_q[..., 0], sell_q[..., 1])
    joint_logits = batched_joint_action_logits(
        buy_logits, sell_logits, held, allow_buy, valid_mask=valid,
    )
    sell_probability = torch.softmax(joint_logits, dim=-1)[
        :, sell_offset : sell_offset + coins
    ]
    baseline = (
        sell_probability * sell_q[..., 0]
        + (1.0 - sell_probability) * sell_q[..., 1]
    )
    count = held.sum(dim=1).clamp_min(1)
    advantage = ((chosen_q - baseline) * held).sum(dim=1) / count
    chosen = (chosen_q * held).sum(dim=1) / count
    return advantage, chosen


def sell_q_expected_value(
    sell_q: torch.Tensor, buy_logits: torch.Tensor, sell_logits: torch.Tensor,
    held: torch.Tensor, allow_buy: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expected learned sell/hold value under the complete joint policy.

    In Phase 1 this is exactly the binary sell/hold expectation. The generic
    joint marginal keeps the formula well-defined for future multi-slot work,
    but policy improvement is intentionally restricted to the exact Phase-1
    case by the trainer.
    """
    _, coins, _ = sell_q.shape
    valid = torch.ones_like(held, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    if valid.shape != held.shape:
        raise ValueError("Valid mask shape must match held mask")
    held = held.bool() & valid
    sell_offset = 1 + coins * len(BUY_ALLOCATIONS)
    joint_logits = batched_joint_action_logits(
        buy_logits, sell_logits, held, allow_buy, valid_mask=valid,
    )
    sell_probability = torch.softmax(joint_logits, dim=-1)[
        :, sell_offset : sell_offset + coins
    ]
    expected = (
        sell_probability * sell_q[..., 0]
        + (1.0 - sell_probability) * sell_q[..., 1]
    )
    count = held.sum(dim=1).clamp_min(1)
    return (expected * held).sum(dim=1) / count


def buy_q_credit(
    action_q: torch.Tensor, buy_logits: torch.Tensor, sell_logits: torch.Tensor,
    held: torch.Tensor, allow_buy: torch.Tensor, actions: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact joint-policy buy/skip credit on flat coin states.

    The existing two-output auxiliary head is conditional on ``in_held``:
    output zero is BUY here (SELL while held), output one is SKIP here (HOLD
    while held).  Every joint action other than buying this coin means SKIP
    for that coin, so the full categorical buy marginal is the correct
    baseline even when later phases expose multiple coins.
    """
    timesteps, coins, _ = action_q.shape
    valid = torch.ones_like(held, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    if valid.shape != held.shape:
        raise ValueError("Valid mask shape must match held mask")
    buy_offset = 1
    buy_indices = buy_offset + torch.arange(coins, device=actions.device)[:, None] * len(BUY_ALLOCATIONS) + torch.arange(len(BUY_ALLOCATIONS), device=actions.device)
    chose_buy = actions[:, None, None].eq(buy_indices[None, :, :]).any(dim=-1)
    chosen_q = torch.where(chose_buy, action_q[..., 0], action_q[..., 1])
    joint_logits = batched_joint_action_logits(
        buy_logits, sell_logits, held, allow_buy, valid_mask=valid,
    )
    buy_probability = torch.softmax(joint_logits, dim=-1)[
        :, buy_offset : buy_offset + coins * len(BUY_ALLOCATIONS)
    ].reshape(timesteps, coins, len(BUY_ALLOCATIONS)).sum(dim=-1)
    baseline = buy_probability * action_q[..., 0] + (1.0 - buy_probability) * action_q[..., 1]
    eligible = (~held) & allow_buy[:, None] & valid
    count = eligible.sum(dim=1).clamp_min(1)
    advantage = ((chosen_q - baseline) * eligible).sum(dim=1) / count
    chosen = (chosen_q * eligible).sum(dim=1) / count
    return advantage, chosen


def buy_allocation_q_credit(
    allocation_q: torch.Tensor, buy_logits: torch.Tensor, sell_logits: torch.Tensor,
    held: torch.Tensor, allow_buy: torch.Tensor, actions: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Phase-1 credit for each learned buy allocation and SKIP.

    ``allocation_q`` is ordered as the four concrete allocations followed by
    SKIP, matching the buy policy head. At a flat Phase-1 state that local
    categorical is the complete joint decision (NOOP is SKIP), so the
    policy-weighted baseline is exact. The generic masking remains valid for
    diagnostics with more coins, but policy mixing is restricted to Phase 1.
    """
    timesteps, coins, actions_per_coin = allocation_q.shape
    allocations = len(BUY_ALLOCATIONS)
    if actions_per_coin != allocations + 1:
        raise ValueError("Allocation Q must contain every allocation plus SKIP")
    valid = torch.ones_like(held, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    if valid.shape != held.shape:
        raise ValueError("Valid mask shape must match held mask")
    buy_indices = (
        1 + torch.arange(coins, device=actions.device)[:, None] * allocations
        + torch.arange(allocations, device=actions.device)
    )
    selected = allocation_q[..., -1]
    for allocation_index in range(allocations):
        selected = torch.where(
            actions[:, None].eq(buy_indices[None, :, allocation_index]),
            allocation_q[..., allocation_index], selected,
        )
    joint_logits = batched_joint_action_logits(
        buy_logits, sell_logits, held, allow_buy, valid_mask=valid,
    )
    buy_probability = torch.softmax(joint_logits, dim=-1)[
        :, 1 : 1 + coins * allocations
    ].reshape(timesteps, coins, allocations)
    skip_probability = 1.0 - buy_probability.sum(dim=-1)
    baseline = (
        (buy_probability * allocation_q[..., :allocations]).sum(dim=-1)
        + skip_probability * allocation_q[..., -1]
    )
    eligible = (~held) & allow_buy[:, None] & valid
    count = eligible.sum(dim=1).clamp_min(1)
    advantage = ((selected - baseline) * eligible).sum(dim=1) / count
    chosen = (selected * eligible).sum(dim=1) / count
    return advantage, chosen


def center_scale_held_credit(
    credit: torch.Tensor, held_steps: torch.Tensor, eps: float,
) -> torch.Tensor:
    """Center and scale auxiliary policy credit only over eligible states."""
    normalized = torch.zeros_like(credit)
    if held_steps.any():
        eligible = credit[held_steps]
        normalized[held_steps] = (eligible - eligible.mean()) / (
            eligible.std(unbiased=False) + eps
        )
    return normalized


def center_scale_flat_credit(credit: torch.Tensor, flat_steps: torch.Tensor, eps: float) -> torch.Tensor:
    return center_scale_held_credit(credit, flat_steps, eps)


def held_policy_weights(held_states: torch.Tensor, weight: float) -> torch.Tensor:
    """Return per-timestep PPO policy weights; 1.0 is exactly neutral."""
    if weight <= 0.0:
        raise ValueError("held policy loss weight must be positive")
    weights = torch.ones_like(held_states, dtype=torch.float32)
    weights = weights.to(device=held_states.device)
    if weight != 1.0:
        weights = torch.where(
            held_states.bool(),
            weights.new_full((), float(weight)),
            weights,
        )
    return weights


def pad_variable_coin_batch(
    observations: list[torch.Tensor],
    held_masks: list[torch.Tensor],
    actions: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable visible-universe timesteps without making padded coins legal."""
    if len(observations) != len(held_masks) or len(observations) != len(actions):
        raise ValueError("Padded PPO inputs must have matching lengths")
    if not observations:
        raise ValueError("Cannot pad an empty PPO batch")
    feature_dim = observations[0].shape[1]
    max_coins = max(int(obs.shape[0]) for obs in observations)
    padded_obs = observations[0].new_zeros((len(observations), max_coins, feature_dim))
    padded_held = torch.zeros((len(observations), max_coins), dtype=torch.bool)
    valid = torch.zeros((len(observations), max_coins), dtype=torch.bool)
    padded_actions = torch.empty((len(observations),), dtype=torch.long)
    for index, (obs, held, action) in enumerate(zip(observations, held_masks, actions)):
        coins = int(obs.shape[0])
        if obs.shape[1] != feature_dim:
            raise ValueError("Observation feature dimensions differ")
        if held.shape[0] != coins:
            raise ValueError("Held mask length must match visible coins")
        padded_obs[index, :coins] = obs
        padded_held[index, :coins] = held.bool()
        valid[index, :coins] = True
        sell_offset = 1 + coins * len(BUY_ALLOCATIONS)
        if action >= sell_offset:
            coin_index = int(action - sell_offset)
            padded_actions[index] = 1 + max_coins * len(BUY_ALLOCATIONS) + coin_index
        else:
            padded_actions[index] = int(action)
    return padded_obs, padded_held, valid, padded_actions


class JointPPOAgent:
    def __init__(self, model, optimizer, config: JointPPOConfig, device: torch.device):
        self.model, self.optimizer, self.cfg, self.device = model, optimizer, config, device

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        self.model.eval()
        output, next_hidden = self.model.forward_step(obs.to(self.device), hidden)
        output["portfolio_value"] = output["value"].mean()
        output["next_hidden"] = next_hidden
        self.model.train()
        return output

    def _recurrent_chunks(self, dones: torch.Tensor) -> list[tuple[int, int, int]]:
        chunks = []
        episode_start = 0
        for index, done in enumerate(dones.tolist()):
            if done:
                episode_end = index + 1
                for start in range(episode_start, episode_end, self.cfg.sequence_length):
                    chunks.append((episode_start, start, min(episode_end, start + self.cfg.sequence_length)))
                episode_start = episode_end
        if episode_start < len(dones):
            for start in range(episode_start, len(dones), self.cfg.sequence_length):
                chunks.append((episode_start, start, min(len(dones), start + self.cfg.sequence_length)))
        return chunks

    def _update_recurrent(
        self, obs, held, allow_buy, actions, old_logp, returns, advantages, dones,
        hidden_states, policy_weights,
    ) -> dict[str, float]:
        cfg = self.cfg
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        updates = 0
        chunks = self._recurrent_chunks(dones)
        stop = False
        for _ in range(cfg.update_epochs):
            for chunk_index in torch.randperm(len(chunks)).tolist():
                episode_start, start, end = chunks[chunk_index]
                burn_start = max(episode_start, start - cfg.burn_in)
                hidden = hidden_states[burn_start].to(self.device)
                if burn_start < start:
                    with torch.no_grad():
                        _, hidden = self.model.forward_sequence(obs[burn_start:start], hidden)
                output, _ = self.model.forward_sequence(obs[start:end], hidden)
                buy, sell = output["buy_logits"], output["sell_logits"]
                value = output["value"].mean(dim=1)
                logits = batched_joint_action_logits(
                    buy, sell, held[start:end], allow_buy[start:end],
                    mode_logits=output.get("mode_logits"),
                )
                distribution = torch.distributions.Categorical(logits=logits)
                logp = distribution.log_prob(actions[start:end])
                legal_counts = torch.isfinite(logits).sum(dim=1).to(logits.dtype)
                entropy = (distribution.entropy() / legal_counts.log().clamp_min(1.0)).mean()
                ratio = torch.exp(logp - old_logp[start:end])
                surrogate1 = ratio * advantages[start:end]
                surrogate2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages[start:end]
                selected = torch.minimum(surrogate1, surrogate2)
                weights = policy_weights[start:end]
                policy_loss = -(selected * weights).sum() / weights.sum().clamp_min(cfg.eps)
                value_loss = F.mse_loss(value, returns[start:end])
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    kl = (old_logp[start:end] - logp).mean().abs()
                for key, val in (("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", entropy), ("kl", kl)):
                    totals[key] += float(val)
                updates += 1
                if float(kl) > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        return {key: value / max(1, updates) for key, value in totals.items()} | {"updates": updates}

    def update(self, buffer: JointEpisodeBuffer) -> dict[str, float]:
        cfg = self.cfg
        variable_coins = len(set(buffer.coin_counts)) > 1
        if variable_coins and self.model.is_recurrent:
            raise RuntimeError("Variable-coin PPO padding is feed-forward only")
        if variable_coins and (
            cfg.buy_q_coef > 0.0 or cfg.buy_q_advantage_mix > 0.0
        ):
            raise RuntimeError("Variable-coin PPO padding supports Sell-Q and Allocation-Q only")
        if variable_coins:
            obs, held, valid_mask, padded_actions = pad_variable_coin_batch(
                buffer.obs, buffer.held, buffer.actions,
            )
            obs = obs.to(self.device)
            held = held.to(self.device)
            valid_mask = valid_mask.to(self.device)
            actions = padded_actions.to(self.device)
        else:
            obs = torch.stack(buffer.obs).to(self.device)
            held = torch.stack(buffer.held).to(self.device)
            valid_mask = torch.ones_like(held, dtype=torch.bool, device=self.device)
            actions = torch.tensor(buffer.actions, dtype=torch.long, device=self.device)
        allow_buy = torch.tensor(buffer.allow_buy, dtype=torch.bool, device=self.device)
        old_logp = torch.stack(buffer.logp).to(self.device).view(-1)
        old_values = torch.stack(buffer.values).to(self.device).view(-1)
        rewards = torch.tensor(buffer.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(buffer.dones, dtype=torch.float32, device=self.device)
        time_limit_truncations = torch.tensor(
            buffer.time_limit_truncations, dtype=torch.float32, device=self.device,
        )
        bootstrap_values = torch.tensor(
            buffer.bootstrap_values, dtype=torch.float32, device=self.device,
        )
        sell_aux_labels = torch.tensor(
            buffer.sell_aux_labels, dtype=torch.float32, device=self.device,
        )
        sell_aux_masks = torch.tensor(
            buffer.sell_aux_masks, dtype=torch.bool, device=self.device,
        )
        sell_clear_aux_labels = torch.tensor(
            buffer.sell_clear_aux_labels, dtype=torch.float32, device=self.device,
        )
        sell_clear_aux_masks = torch.tensor(
            buffer.sell_clear_aux_masks, dtype=torch.bool, device=self.device,
        )
        sell_margin_align_targets = torch.tensor(
            buffer.sell_margin_align_targets, dtype=torch.float32, device=self.device,
        )
        sell_margin_align_masks = torch.tensor(
            buffer.sell_margin_align_masks, dtype=torch.bool, device=self.device,
        )
        sell_margin_best_hold_opportunities = torch.tensor(
            buffer.sell_margin_best_hold_opportunities, dtype=torch.float32, device=self.device,
        )
        sell_margin_avoided_losses = torch.tensor(
            buffer.sell_margin_avoided_losses, dtype=torch.float32, device=self.device,
        )
        sell_margin_current_returns = torch.tensor(
            buffer.sell_margin_current_returns, dtype=torch.float32, device=self.device,
        )
        buy_aux_targets = torch.tensor(
            buffer.buy_aux_targets, dtype=torch.long, device=self.device,
        )
        buy_aux_masks = torch.tensor(
            buffer.buy_aux_masks, dtype=torch.bool, device=self.device,
        )
        held_position_ages = torch.tensor(
            buffer.held_position_ages, dtype=torch.float32, device=self.device,
        )
        held_position_returns = torch.tensor(
            buffer.held_position_returns, dtype=torch.float32, device=self.device,
        )
        failed_to_sell_masks = torch.tensor(
            buffer.failed_to_sell_masks, dtype=torch.bool, device=self.device,
        )
        held_steps = (held & valid_mask).any(dim=1)
        policy_weights = held_policy_weights(held_steps, cfg.held_policy_loss_weight).to(self.device)
        held_advantages = advantages_for_metrics = None
        with torch.no_grad():
            advantages, returns = state_dependent_gae(
                rewards, old_values, dones, held_steps, cfg.gamma,
                cfg.gae_lambda, cfg.held_gae_lambda,
                time_limit_truncations=time_limit_truncations,
                bootstrap_values=bootstrap_values,
            )
            advantages = normalize_advantages(
                advantages, held_steps, cfg.advantage_normalization, cfg.eps,
                actions=actions, coins=held.shape[1],
            )
            if cfg.sell_q_target == "mc":
                q_returns = monte_carlo_returns(rewards, dones)
            elif cfg.sell_q_target == "td0":
                q_returns = one_step_td_returns(
                    rewards, old_values, dones, cfg.gamma,
                )
            else:
                raise ValueError("sell_q_target must be mc or td0")
            if cfg.sell_q_advantage_mix > 0.0:
                if not self.model.action_value_heads:
                    raise RuntimeError("Sell-Q credit requested without action-value heads")
                if self.model.is_recurrent:
                    raise RuntimeError("Sell-Q credit is currently feed-forward only")
                timesteps, coins, features = obs.shape
                credit_output = self.model(obs.reshape(-1, features))
                q_advantage, _ = sell_q_credit(
                    credit_output["sell_q"].reshape(timesteps, coins, -1),
                    credit_output["buy_logits"].reshape(timesteps, coins, -1),
                    credit_output["sell_logits"].reshape(timesteps, coins, -1),
                    held, allow_buy, actions, valid_mask=valid_mask,
                )
                held_steps = held.any(dim=1)
                if held_steps.any():
                    q_advantage = center_scale_held_credit(
                        q_advantage, held_steps, cfg.eps,
                    )
                    mix = cfg.sell_q_advantage_mix
                    advantages[held_steps] = (
                        (1.0 - mix) * advantages[held_steps]
                        + mix * q_advantage[held_steps]
                    )
            if cfg.buy_q_advantage_mix > 0.0:
                if not self.model.action_value_heads or self.model.is_recurrent:
                    raise RuntimeError("Buy-Q credit requires feed-forward action-value heads")
                timesteps, coins, features = obs.shape
                credit_output = self.model(obs.reshape(-1, features))
                q_advantage, _ = buy_q_credit(
                    credit_output["sell_q"].reshape(timesteps, coins, -1),
                    credit_output["buy_logits"].reshape(timesteps, coins, -1),
                    credit_output["sell_logits"].reshape(timesteps, coins, -1),
                    held, allow_buy, actions, valid_mask=valid_mask,
                )
                flat_steps = ~(held & valid_mask).any(dim=1)
                if flat_steps.any():
                    q_advantage = center_scale_flat_credit(q_advantage, flat_steps, cfg.eps)
                    mix = cfg.buy_q_advantage_mix
                    advantages[flat_steps] = (1.0 - mix) * advantages[flat_steps] + mix * q_advantage[flat_steps]
            if cfg.buy_allocation_q_advantage_mix > 0.0:
                if not self.model.flat_action_value_heads or self.model.is_recurrent:
                    raise RuntimeError("Allocation-Q credit requires feed-forward allocation-value heads")
                timesteps, coins, features = obs.shape
                credit_output = self.model(obs.reshape(-1, features))
                q_advantage, _ = buy_allocation_q_credit(
                    credit_output["buy_allocation_q"].reshape(timesteps, coins, -1),
                    credit_output["buy_logits"].reshape(timesteps, coins, -1),
                    credit_output["sell_logits"].reshape(timesteps, coins, -1),
                    held, allow_buy, actions, valid_mask=valid_mask,
                )
                flat_steps = ~(held & valid_mask).any(dim=1)
                if flat_steps.any():
                    q_advantage = center_scale_flat_credit(q_advantage, flat_steps, cfg.eps)
                    mix = cfg.buy_allocation_q_advantage_mix
                    advantages[flat_steps] = (1.0 - mix) * advantages[flat_steps] + mix * q_advantage[flat_steps]
            advantages_for_metrics = advantages.detach()
            held_advantages = advantages_for_metrics[held_steps]
            flat_advantages = advantages_for_metrics[~held_steps]
            credit_metrics = {
                "held_steps": int(held_steps.sum().item()),
                "flat_steps": int((~held_steps).sum().item()),
                "held_policy_loss_weight": float(cfg.held_policy_loss_weight),
                "held_advantage_mean": (
                    float(held_advantages.mean()) if held_advantages.numel() else 0.0
                ),
                "held_advantage_std": (
                    float(held_advantages.std(unbiased=False)) if held_advantages.numel() else 0.0
                ),
                "flat_advantage_mean": (
                    float(flat_advantages.mean()) if flat_advantages.numel() else 0.0
                ),
                "flat_advantage_std": (
                    float(flat_advantages.std(unbiased=False)) if flat_advantages.numel() else 0.0
                ),
                "policy_weight_mean": float(policy_weights.mean()),
            }
        if self.model.is_recurrent:
            if cfg.sell_aux_coef > 0.0:
                raise RuntimeError("Sell auxiliary loss is currently feed-forward only")
            if cfg.sell_clear_aux_coef > 0.0:
                raise RuntimeError("Clear-state sell auxiliary loss is currently feed-forward only")
            if cfg.sell_clear_contrast_coef > 0.0:
                raise RuntimeError("Clear-state sell contrast loss is currently feed-forward only")
            if cfg.sell_clear_margin_floor_coef > 0.0:
                raise RuntimeError("Clear-state sell margin floor loss is currently feed-forward only")
            if cfg.failed_to_sell_margin_coef > 0.0:
                raise RuntimeError("Failed-to-sell margin loss is currently feed-forward only")
            if cfg.failed_to_sell_logit_margin_coef > 0.0:
                raise RuntimeError("Failed-to-sell logit margin loss is currently feed-forward only")
            if cfg.sell_clear_hold_margin_ceiling_coef > 0.0:
                raise RuntimeError("Clear-state hold margin ceiling loss is currently feed-forward only")
            if cfg.sell_margin_align_coef > 0.0:
                raise RuntimeError("Sell-margin alignment loss is currently feed-forward only")
            if cfg.sell_margin_level_coef > 0.0:
                raise RuntimeError("Sell-margin level loss is currently feed-forward only")
            if cfg.sell_target_aux_coef > 0.0:
                raise RuntimeError("Sell-target auxiliary loss is currently feed-forward only")
            if cfg.buy_aux_coef > 0.0:
                raise RuntimeError("Buy auxiliary loss is currently feed-forward only")
            if cfg.buy_aux_negative_coef > 0.0:
                raise RuntimeError("Buy auxiliary negative loss is currently feed-forward only")
            if any(hidden is None for hidden in buffer.recurrent_hidden):
                raise RuntimeError("Recurrent PPO buffer is missing hidden states")
            metrics = self._update_recurrent(
                obs, held, allow_buy, actions, old_logp, returns, advantages, dones,
                buffer.recurrent_hidden, policy_weights,
            )
            return metrics | {"reward_sum": float(rewards.sum())} | credit_metrics
        timesteps, coins, features = obs.shape
        totals = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "q_loss": 0.0, "buy_q_loss": 0.0, "buy_allocation_q_loss": 0.0,
            "q_policy_loss": 0.0, "entropy": 0.0, "kl": 0.0,
            "sell_aux_loss": 0.0, "flat_buy_mass_penalty": 0.0,
            "sell_clear_aux_loss": 0.0, "sell_clear_contrast_loss": 0.0,
            "sell_clear_margin_floor_loss": 0.0,
            "sell_clear_hold_margin_ceiling_loss": 0.0,
            "sell_margin_align_loss": 0.0,
            "sell_margin_level_loss": 0.0,
            "sell_margin_subtype_confidence_loss": 0.0,
            "sell_target_aux_loss": 0.0,
            "sell_margin_conflict_loss": 0.0,
            "failed_to_sell_margin_loss": 0.0,
            "failed_to_sell_logit_margin_loss": 0.0,
            "held_sell_mass_penalty": 0.0,
            "early_held_sell_mass_penalty": 0.0,
            "buy_aux_loss": 0.0,
            "buy_aux_negative_loss": 0.0,
        }
        updates = 0
        stop = False
        for _ in range(cfg.update_epochs):
            permutation = torch.randperm(timesteps, device=self.device)
            for start in range(0, timesteps, cfg.minibatch_size):
                index = permutation[start : start + cfg.minibatch_size]
                if not len(index):
                    continue
                batch = len(index)
                if self.model.universe_context_enabled:
                    output = self.model.forward_group(obs[index])
                    buy = output["buy_logits"]
                    sell = output["sell_logits"]
                    sell_target_aux = output.get("sell_target_aux")
                    batch_valid = valid_mask[index]
                    value = (
                        output["value"] * batch_valid.to(output["value"].dtype)
                    ).sum(dim=1) / batch_valid.sum(dim=1).clamp_min(1).to(output["value"].dtype)
                else:
                    output = self.model(obs[index].reshape(-1, features))
                    buy = output["buy_logits"].reshape(batch, coins, -1)
                    sell = output["sell_logits"].reshape(batch, coins, -1)
                    sell_target_aux = (
                        output["sell_target_aux"].reshape(batch, coins)
                        if "sell_target_aux" in output else None
                    )
                    mode = (
                        output["mode_logits"].reshape(batch, coins, -1)
                        if "mode_logits" in output else None
                    )
                    values = output["value"].reshape(batch, coins)
                    batch_valid = valid_mask[index]
                    value = (
                        values * batch_valid.to(values.dtype)
                    ).sum(dim=1) / batch_valid.sum(dim=1).clamp_min(1).to(values.dtype)
                if self.model.universe_context_enabled:
                    mode = output.get("mode_logits")
                logits = batched_joint_action_logits(
                    buy, sell, held[index], allow_buy[index],
                    mode_logits=mode, valid_mask=valid_mask[index],
                )
                distribution = torch.distributions.Categorical(logits=logits)
                logp = distribution.log_prob(actions[index])
                legal_counts = torch.isfinite(logits).sum(dim=1).to(logits.dtype)
                normalizer = legal_counts.log().clamp_min(1.0)
                entropy = (distribution.entropy() / normalizer).mean()
                ratio = torch.exp(logp - old_logp[index])
                surrogate1 = ratio * advantages[index]
                surrogate2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages[index]
                selected = torch.minimum(surrogate1, surrogate2)
                weights = policy_weights[index]
                policy_loss = -(selected * weights).sum() / weights.sum().clamp_min(cfg.eps)
                value_loss = F.mse_loss(value, returns[index])
                q_loss = value.new_zeros(())
                buy_q_loss = value.new_zeros(())
                buy_allocation_q_loss = value.new_zeros(())
                q_policy_loss = value.new_zeros(())
                sell_aux_loss = value.new_zeros(())
                sell_clear_aux_loss = value.new_zeros(())
                sell_clear_contrast_loss = value.new_zeros(())
                sell_clear_margin_floor_loss = value.new_zeros(())
                sell_clear_hold_margin_ceiling_loss = value.new_zeros(())
                sell_margin_align_loss = value.new_zeros(())
                sell_margin_level_loss = value.new_zeros(())
                sell_margin_subtype_confidence_loss = value.new_zeros(())
                sell_target_aux_loss = value.new_zeros(())
                buy_aux_loss = value.new_zeros(())
                buy_aux_negative_loss = value.new_zeros(())
                flat_buy_mass_penalty = value.new_zeros(())
                failed_to_sell_margin_loss = value.new_zeros(())
                failed_to_sell_logit_margin_loss = value.new_zeros(())
                held_sell_mass_penalty = value.new_zeros(())
                early_held_sell_mass_penalty = value.new_zeros(())
                sell_margin_conflict_loss = value.new_zeros(())
                if cfg.sell_q_coef > 0.0:
                    _, chosen_q = sell_q_credit(
                        output["sell_q"].reshape(batch, coins, -1),
                        buy, sell, held[index], allow_buy[index], actions[index],
                        valid_mask=valid_mask[index],
                    )
                    q_mask = held[index].any(dim=1)
                    if q_mask.any():
                        q_loss = F.mse_loss(chosen_q[q_mask], q_returns[index][q_mask])
                if cfg.buy_q_coef > 0.0:
                    _, chosen_q = buy_q_credit(
                        output["sell_q"].reshape(batch, coins, -1),
                        buy, sell, held[index], allow_buy[index], actions[index],
                        valid_mask=valid_mask[index],
                    )
                    q_mask = ~(held[index] & valid_mask[index]).any(dim=1)
                    if q_mask.any():
                        buy_q_loss = F.mse_loss(chosen_q[q_mask], q_returns[index][q_mask])
                if cfg.buy_allocation_q_coef > 0.0:
                    _, chosen_q = buy_allocation_q_credit(
                        output["buy_allocation_q"].reshape(batch, coins, -1),
                        buy, sell, held[index], allow_buy[index], actions[index],
                        valid_mask=valid_mask[index],
                    )
                    q_mask = ~(held[index] & valid_mask[index]).any(dim=1)
                    if q_mask.any():
                        buy_allocation_q_loss = F.mse_loss(chosen_q[q_mask], q_returns[index][q_mask])
                if cfg.sell_q_policy_coef > 0.0:
                    if not self.model.action_value_heads:
                        raise RuntimeError("Sell-Q policy improvement requires action-value heads")
                    q_mask = held[index].any(dim=1)
                    if q_mask.any():
                        expected_q = sell_q_expected_value(
                            output["sell_q"].reshape(batch, coins, -1).detach(),
                            buy, sell, held[index], allow_buy[index],
                            valid_mask=valid_mask[index],
                        )
                        q_policy_loss = -expected_q[q_mask].mean()
                if cfg.buy_aux_coef > 0.0:
                    flat_mask = (
                        buy_aux_masks[index]
                        & (~(held[index] & valid_mask[index]).any(dim=1))
                        & allow_buy[index]
                    )
                    if flat_mask.any():
                        probabilities = distribution.probs
                        buy_start = 1
                        buy_end = 1 + coins * len(BUY_ALLOCATIONS)
                        buy_by_coin = probabilities[
                            flat_mask, buy_start:buy_end
                        ].reshape(-1, coins, len(BUY_ALLOCATIONS)).sum(dim=2)
                        target = buy_aux_targets[index][flat_mask]
                        target_probability = probabilities[flat_mask, 0].clone()
                        target_is_buy = target >= 0
                        if cfg.buy_aux_rank_only:
                            if target_is_buy.any():
                                conditional = buy_by_coin[target_is_buy] / buy_by_coin[
                                    target_is_buy
                                ].sum(dim=1, keepdim=True).clamp_min(cfg.eps)
                                rows = torch.arange(
                                    conditional.shape[0], device=self.device,
                                )
                                target_probability = conditional[
                                    rows, target[target_is_buy]
                                ]
                                buy_aux_loss = -torch.log(
                                    target_probability.clamp_min(cfg.eps)
                                ).mean()
                        elif target_is_buy.any():
                            rows = torch.arange(
                                target.numel(), device=self.device,
                            )[target_is_buy]
                            target_probability[target_is_buy] = buy_by_coin[
                                rows, target[target_is_buy]
                            ]
                            buy_aux_loss = -torch.log(
                                target_probability.clamp_min(cfg.eps)
                            ).mean()
                        else:
                            buy_aux_loss = -torch.log(
                                target_probability.clamp_min(cfg.eps)
                            ).mean()
                if cfg.buy_aux_negative_coef > 0.0:
                    flat_mask = (
                        buy_aux_masks[index]
                        & (~(held[index] & valid_mask[index]).any(dim=1))
                        & allow_buy[index]
                        & (buy_aux_targets[index] < 0)
                    )
                    if (
                        flat_mask.any()
                        and int(cfg.buy_aux_negative_gate_feature_index) >= 0
                    ):
                        feature_index = int(cfg.buy_aux_negative_gate_feature_index)
                        valid = valid_mask[index].to(obs.dtype)
                        gate_values = (
                            obs[index, :, feature_index] * valid
                        ).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                        flat_mask = flat_mask & (
                            gate_values >= float(cfg.buy_aux_negative_gate_min)
                        ) & (
                            gate_values <= float(cfg.buy_aux_negative_gate_max)
                        )
                    if flat_mask.any():
                        probabilities = distribution.probs
                        buy_start = 1
                        buy_end = 1 + coins * len(BUY_ALLOCATIONS)
                        buy_mass = probabilities[flat_mask, buy_start:buy_end].sum(dim=1)
                        if cfg.buy_aux_negative_buy_mass_target > 0.0:
                            excess = (
                                buy_mass - float(cfg.buy_aux_negative_buy_mass_target)
                            ).clamp_min(0.0)
                            buy_aux_negative_loss = excess.square().mean()
                        else:
                            buy_aux_negative_loss = buy_mass.mean()
                if cfg.sell_aux_coef > 0.0:
                    aux_mask = sell_aux_masks[index] & held[index].any(dim=1)
                    if aux_mask.any():
                        held_coin = held[index][aux_mask].to(sell.dtype)
                        selected_sell_logits = (
                            sell[aux_mask] * held_coin[..., None]
                        ).sum(dim=1) / held_coin.sum(dim=1, keepdim=True).clamp_min(1.0)
                        class_weight = selected_sell_logits.new_tensor([
                            cfg.sell_aux_positive_weight, 1.0,
                        ])
                        sell_aux_loss = F.cross_entropy(
                            selected_sell_logits,
                            sell_aux_labels_to_class_indices(
                                sell_aux_labels[index][aux_mask]
                            ),
                            weight=class_weight,
                        )
                if cfg.sell_clear_aux_coef > 0.0:
                    aux_mask = sell_clear_aux_masks[index] & held[index].any(dim=1)
                    if aux_mask.any():
                        held_coin = held[index][aux_mask].to(sell.dtype)
                        selected_sell_logits = (
                            sell[aux_mask] * held_coin[..., None]
                        ).sum(dim=1) / held_coin.sum(dim=1, keepdim=True).clamp_min(1.0)
                        class_weight = selected_sell_logits.new_tensor([
                            cfg.sell_clear_aux_positive_weight, 1.0,
                        ])
                        sell_clear_aux_loss = F.cross_entropy(
                            selected_sell_logits,
                            sell_aux_labels_to_class_indices(
                                sell_clear_aux_labels[index][aux_mask]
                            ),
                            weight=class_weight,
                        )
                if cfg.sell_clear_contrast_coef > 0.0:
                    aux_mask = sell_clear_aux_masks[index] & held[index].any(dim=1)
                    if aux_mask.any():
                        held_coin = held[index][aux_mask].to(sell.dtype)
                        selected_sell_logits = (
                            sell[aux_mask] * held_coin[..., None]
                        ).sum(dim=1) / held_coin.sum(dim=1, keepdim=True).clamp_min(1.0)
                        margins = selected_sell_logits[:, 0] - selected_sell_logits[:, 1]
                        labels = sell_clear_aux_labels[index][aux_mask]
                        positive = labels > 0.5
                        negative = ~positive
                        if positive.any() and negative.any():
                            separation = margins[positive].mean() - margins[negative].mean()
                            sell_clear_contrast_loss = (
                                cfg.sell_clear_contrast_margin - separation
                            ).clamp_min(0.0).square()
                if cfg.sell_clear_margin_floor_coef > 0.0:
                    aux_mask = sell_clear_aux_masks[index] & held[index].any(dim=1)
                    if aux_mask.any():
                        held_coin = held[index][aux_mask].to(sell.dtype)
                        selected_sell_logits = (
                            sell[aux_mask] * held_coin[..., None]
                        ).sum(dim=1) / held_coin.sum(dim=1, keepdim=True).clamp_min(1.0)
                        margins = selected_sell_logits[:, 0] - selected_sell_logits[:, 1]
                        labels = sell_clear_aux_labels[index][aux_mask]
                        positive = labels > 0.5
                        if cfg.sell_clear_margin_floor_max_return is not None:
                            positive = positive & (
                                held_position_returns[index][aux_mask]
                                <= float(cfg.sell_clear_margin_floor_max_return)
                            )
                        if positive.any():
                            sell_clear_margin_floor_loss = (
                                cfg.sell_clear_margin_floor - margins[positive]
                            ).clamp_min(0.0).square().mean()
                if cfg.sell_clear_hold_margin_ceiling_coef > 0.0:
                    aux_mask = sell_clear_aux_masks[index] & held[index].any(dim=1)
                    if aux_mask.any():
                        held_coin = held[index][aux_mask].to(sell.dtype)
                        selected_sell_logits = (
                            sell[aux_mask] * held_coin[..., None]
                        ).sum(dim=1) / held_coin.sum(dim=1, keepdim=True).clamp_min(1.0)
                        margins = selected_sell_logits[:, 0] - selected_sell_logits[:, 1]
                        labels = sell_clear_aux_labels[index][aux_mask]
                        negative = labels <= 0.5
                        if negative.any():
                            sell_clear_hold_margin_ceiling_loss = (
                                margins[negative] - cfg.sell_clear_hold_margin_ceiling
                            ).clamp_min(0.0).square().mean()
                held_any = held[index].any(dim=1)
                align_mask = sell_margin_align_masks[index] & held_any
                if align_mask.any():
                    held_coin = held[index][align_mask].to(sell.dtype)
                    selected_sell_logits = (
                        sell[align_mask] * held_coin[..., None]
                    ).sum(dim=1) / held_coin.sum(dim=1, keepdim=True).clamp_min(1.0)
                    selected_sell_margins = selected_sell_logits[:, 0] - selected_sell_logits[:, 1]
                    best_hold = sell_margin_best_hold_opportunities[index][align_mask]
                    avoided_loss = sell_margin_avoided_losses[index][align_mask]
                    current_returns = sell_margin_current_returns[index][align_mask]
                    hold_rows = (
                        best_hold >= float(cfg.sell_margin_align_clear_hold_opportunity_min)
                    )
                    avoid_rows = (
                        (current_returns <= float(cfg.sell_margin_align_clear_stop_return))
                        | (avoided_loss >= float(cfg.sell_margin_align_clear_avoid_loss_min))
                    )
                    conflict_rows = (
                        (best_hold >= float(cfg.sell_margin_conflict_hold_opportunity_min))
                        & (
                            (avoided_loss >= float(cfg.sell_margin_conflict_avoid_loss_min))
                            | (current_returns <= float(cfg.sell_margin_conflict_stop_return))
                        )
                    )
                    if cfg.sell_margin_conflict_uncertainty_coef > 0.0:
                        if conflict_rows.any():
                            sell_margin_conflict_loss = (torch.tanh(
                                selected_sell_margins[conflict_rows]
                                / float(cfg.sell_margin_conflict_uncertainty_scale)
                            )).square().mean()
                        trainable_mask = align_mask.clone()
                        trainable_mask[align_mask] = ~conflict_rows
                    else:
                        trainable_mask = align_mask
                    if cfg.sell_margin_subtype_confidence_coef > 0.0:
                        subtype_rows = hold_rows ^ avoid_rows
                        if subtype_rows.any():
                            subtype_targets = avoid_rows[subtype_rows].to(selected_sell_margins.dtype)
                            sell_margin_subtype_confidence_loss = F.binary_cross_entropy_with_logits(
                                selected_sell_margins[subtype_rows],
                                subtype_targets,
                            )
                else:
                    trainable_mask = align_mask
                if cfg.sell_margin_align_coef > 0.0:
                    aux_mask = trainable_mask
                    if aux_mask.any():
                        sell_margin_align_loss = normalized_margin_alignment_loss(
                            selected_sell_margins[aux_mask[align_mask]],
                            sell_margin_align_targets[index][aux_mask].to(selected_sell_margins.dtype),
                            cfg.eps,
                        )
                if cfg.sell_margin_level_coef > 0.0:
                    aux_mask = trainable_mask
                    if aux_mask.any():
                        level_weights = selected_sell_margins.new_ones(aux_mask.sum())
                        hold_mask = (
                            sell_margin_best_hold_opportunities[index][aux_mask]
                            >= float(cfg.sell_margin_level_hold_opportunity_min)
                        )
                        avoid_mask = (
                            sell_margin_avoided_losses[index][aux_mask]
                            >= float(cfg.sell_margin_level_sell_avoid_loss_min)
                        )
                        if cfg.sell_margin_level_hold_opportunity_min > 0.0:
                            level_weights = torch.where(
                                hold_mask,
                                level_weights * float(cfg.sell_margin_level_hold_weight),
                                level_weights,
                            )
                        if cfg.sell_margin_level_sell_avoid_loss_min > 0.0:
                            level_weights = torch.where(
                                avoid_mask,
                                level_weights * float(cfg.sell_margin_level_sell_weight),
                                level_weights,
                            )
                        sell_margin_level_loss = bounded_margin_level_loss(
                            selected_sell_margins[aux_mask[align_mask]],
                            sell_margin_align_targets[index][aux_mask].to(selected_sell_margins.dtype),
                            cfg.sell_margin_level_scale,
                            weights=level_weights,
                        )
                if cfg.sell_target_aux_coef > 0.0:
                    if sell_target_aux is None:
                        raise RuntimeError("Sell-target auxiliary loss requested without aux head")
                    aux_mask = sell_margin_align_masks[index] & held[index].any(dim=1)
                    if aux_mask.any():
                        held_coin = held[index][aux_mask].to(sell_target_aux.dtype)
                        selected_aux = (
                            sell_target_aux[aux_mask] * held_coin
                        ).sum(dim=1) / held_coin.sum(dim=1).clamp_min(1.0)
                        sell_target_aux_loss = F.mse_loss(
                            selected_aux,
                            sell_margin_align_targets[index][aux_mask].to(selected_aux.dtype),
                        )
                if cfg.flat_buy_mass_penalty_coef > 0.0:
                    flat_mask = ~(held[index] & valid_mask[index]).any(dim=1) & allow_buy[index]
                    if (
                        flat_mask.any()
                        and int(cfg.flat_buy_mass_penalty_gate_feature_index) >= 0
                    ):
                        feature_index = int(cfg.flat_buy_mass_penalty_gate_feature_index)
                        if feature_index >= obs[index].shape[-1]:
                            raise RuntimeError("Flat buy-mass gate feature index is out of range")
                        gate_values = obs[index][:, :, feature_index].masked_fill(
                            ~valid_mask[index], float("nan")
                        )
                        gate_mask = (
                            (gate_values >= float(cfg.flat_buy_mass_penalty_gate_min))
                            & (gate_values <= float(cfg.flat_buy_mass_penalty_gate_max))
                            & valid_mask[index]
                        ).any(dim=1)
                        flat_mask = flat_mask & gate_mask
                    if flat_mask.any():
                        probabilities = distribution.probs
                        buy_start = 1
                        buy_end = 1 + coins * len(BUY_ALLOCATIONS)
                        buy_mass = probabilities[flat_mask, buy_start:buy_end].sum(dim=1)
                        if cfg.flat_buy_mass_penalty_target > 0.0:
                            excess = (
                                buy_mass - cfg.flat_buy_mass_penalty_target
                            ).clamp_min(0.0)
                            flat_buy_mass_penalty = excess.square().mean()
                        else:
                            flat_buy_mass_penalty = buy_mass.mean()
                if cfg.failed_to_sell_margin_coef > 0.0:
                    hindsight_mask = (
                        failed_to_sell_masks[index]
                        & (held[index] & valid_mask[index]).any(dim=1)
                    )
                    if hindsight_mask.any():
                        probabilities = distribution.probs
                        buy_start = 1
                        buy_end = 1 + coins * len(BUY_ALLOCATIONS)
                        sell_start = buy_end
                        noop_mass = probabilities[hindsight_mask, 0]
                        buy_mass = probabilities[hindsight_mask, buy_start:buy_end].sum(dim=1)
                        sell_mass = probabilities[hindsight_mask, sell_start:].sum(dim=1)
                        non_sell_mass = torch.maximum(noop_mass, buy_mass)
                        margin = (
                            sell_mass.clamp_min(cfg.eps).log()
                            - non_sell_mass.clamp_min(cfg.eps).log()
                        )
                        failed_to_sell_margin_loss = (
                            float(cfg.failed_to_sell_margin_target) - margin
                        ).clamp_min(0.0).square().mean()
                if cfg.failed_to_sell_logit_margin_coef > 0.0:
                    hindsight_mask = (
                        failed_to_sell_masks[index]
                        & (held[index] & valid_mask[index]).any(dim=1)
                    )
                    if hindsight_mask.any():
                        buy_start = 1
                        buy_end = 1 + coins * len(BUY_ALLOCATIONS)
                        sell_start = buy_end
                        masked_logits = logits[hindsight_mask]
                        held_rows = (held[index] & valid_mask[index])[hindsight_mask]
                        selected_sell_logit = masked_logits[:, sell_start:].masked_fill(
                            ~held_rows, -torch.inf
                        ).amax(dim=1)
                        strongest_buy_logit = masked_logits[:, buy_start:buy_end].amax(dim=1)
                        strongest_non_sell_logit = torch.maximum(
                            masked_logits[:, 0], strongest_buy_logit,
                        )
                        logit_margin = selected_sell_logit - strongest_non_sell_logit
                        failed_to_sell_logit_margin_loss = (
                            float(cfg.failed_to_sell_logit_margin_target) - logit_margin
                        ).clamp_min(0.0).square().mean()
                if cfg.held_sell_mass_penalty_coef > 0.0:
                    held_mask_batch = (held[index] & valid_mask[index]).any(dim=1)
                    if held_mask_batch.any():
                        probabilities = distribution.probs
                        sell_start = 1 + coins * len(BUY_ALLOCATIONS)
                        sell_mass = probabilities[held_mask_batch, sell_start:].sum(dim=1)
                        if cfg.held_sell_mass_penalty_target > 0.0:
                            excess = (
                                sell_mass - cfg.held_sell_mass_penalty_target
                            ).clamp_min(0.0)
                            held_sell_mass_penalty = excess.square().mean()
                        else:
                            held_sell_mass_penalty = sell_mass.mean()
                if cfg.early_held_sell_mass_penalty_coef > 0.0:
                    held_mask_batch = (held[index] & valid_mask[index]).any(dim=1)
                    early_mask = (
                        held_mask_batch
                        & (held_position_ages[index] < float(cfg.early_held_sell_mass_penalty_hours))
                        & (held_position_returns[index] > cfg.early_held_sell_mass_stop_return)
                    )
                    if early_mask.any():
                        probabilities = distribution.probs
                        sell_start = 1 + coins * len(BUY_ALLOCATIONS)
                        sell_mass = probabilities[early_mask, sell_start:].sum(dim=1)
                        if cfg.early_held_sell_mass_penalty_target > 0.0:
                            excess = (
                                sell_mass - cfg.early_held_sell_mass_penalty_target
                            ).clamp_min(0.0)
                            early_held_sell_mass_penalty = excess.square().mean()
                        else:
                            early_held_sell_mass_penalty = sell_mass.mean()
                loss = (
                    policy_loss + cfg.value_coef * value_loss
                    + cfg.sell_q_coef * q_loss
                    + cfg.buy_q_coef * buy_q_loss
                    + cfg.buy_allocation_q_coef * buy_allocation_q_loss
                    + cfg.sell_q_policy_coef * q_policy_loss
                    + cfg.buy_aux_coef * buy_aux_loss
                    + cfg.buy_aux_negative_coef * buy_aux_negative_loss
                    + cfg.sell_aux_coef * sell_aux_loss
                    + cfg.sell_clear_aux_coef * sell_clear_aux_loss
                    + cfg.sell_clear_contrast_coef * sell_clear_contrast_loss
                    + cfg.sell_clear_margin_floor_coef * sell_clear_margin_floor_loss
                    + cfg.sell_clear_hold_margin_ceiling_coef * sell_clear_hold_margin_ceiling_loss
                    + cfg.sell_margin_align_coef * sell_margin_align_loss
                    + cfg.sell_margin_level_coef * sell_margin_level_loss
                    + cfg.sell_margin_subtype_confidence_coef * sell_margin_subtype_confidence_loss
                    + cfg.sell_target_aux_coef * sell_target_aux_loss
                    + cfg.sell_margin_conflict_uncertainty_coef * sell_margin_conflict_loss
                    + cfg.flat_buy_mass_penalty_coef * flat_buy_mass_penalty
                    + cfg.failed_to_sell_margin_coef * failed_to_sell_margin_loss
                    + cfg.failed_to_sell_logit_margin_coef * failed_to_sell_logit_margin_loss
                    + cfg.held_sell_mass_penalty_coef * held_sell_mass_penalty
                    + cfg.early_held_sell_mass_penalty_coef * early_held_sell_mass_penalty
                    - cfg.entropy_coef * entropy
                )
                if not loss.requires_grad:
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    kl = (old_logp[index] - logp).mean().abs()
                for key, val in (
                    ("policy_loss", policy_loss), ("value_loss", value_loss),
                    ("q_loss", q_loss), ("buy_q_loss", buy_q_loss), ("q_policy_loss", q_policy_loss),
                    ("buy_allocation_q_loss", buy_allocation_q_loss),
                    ("buy_aux_loss", buy_aux_loss),
                    ("buy_aux_negative_loss", buy_aux_negative_loss),
                    ("sell_aux_loss", sell_aux_loss),
                    ("sell_clear_aux_loss", sell_clear_aux_loss),
                    ("sell_clear_contrast_loss", sell_clear_contrast_loss),
                    ("sell_clear_margin_floor_loss", sell_clear_margin_floor_loss),
                    ("sell_clear_hold_margin_ceiling_loss", sell_clear_hold_margin_ceiling_loss),
                    ("sell_margin_align_loss", sell_margin_align_loss),
                    ("sell_margin_level_loss", sell_margin_level_loss),
                    ("sell_margin_subtype_confidence_loss", sell_margin_subtype_confidence_loss),
                    ("sell_target_aux_loss", sell_target_aux_loss),
                    ("sell_margin_conflict_loss", sell_margin_conflict_loss),
                    ("flat_buy_mass_penalty", flat_buy_mass_penalty),
                    ("failed_to_sell_margin_loss", failed_to_sell_margin_loss),
                    ("failed_to_sell_logit_margin_loss", failed_to_sell_logit_margin_loss),
                    ("held_sell_mass_penalty", held_sell_mass_penalty),
                    ("early_held_sell_mass_penalty", early_held_sell_mass_penalty),
                    ("entropy", entropy), ("kl", kl),
                ):
                    totals[key] += float(val)
                updates += 1
                if float(kl) > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        return {key: value / max(1, updates) for key, value in totals.items()} | {
            "reward_sum": float(rewards.sum()), "updates": updates,
            "sell_aux_labeled_steps": int(sell_aux_masks.sum().item()),
            "sell_aux_positive_rate": (
                float(sell_aux_labels[sell_aux_masks].mean().item())
                if sell_aux_masks.any() else 0.0
            ),
            "sell_clear_aux_labeled_steps": int(sell_clear_aux_masks.sum().item()),
            "sell_clear_aux_positive_rate": (
                float(sell_clear_aux_labels[sell_clear_aux_masks].mean().item())
                if sell_clear_aux_masks.any() else 0.0
            ),
            "sell_margin_align_labeled_steps": int(sell_margin_align_masks.sum().item()),
            "sell_margin_align_target_mean": (
                float(sell_margin_align_targets[sell_margin_align_masks].mean().item())
                if sell_margin_align_masks.any() else 0.0
            ),
            "sell_margin_align_target_std": (
                float(sell_margin_align_targets[sell_margin_align_masks].std(unbiased=False).item())
                if sell_margin_align_masks.any() else 0.0
            ),
            "sell_margin_best_hold_opportunity_mean": (
                float(sell_margin_best_hold_opportunities[sell_margin_align_masks].mean().item())
                if sell_margin_align_masks.any() else 0.0
            ),
            "sell_margin_avoided_loss_mean": (
                float(sell_margin_avoided_losses[sell_margin_align_masks].mean().item())
                if sell_margin_align_masks.any() else 0.0
            ),
            "sell_margin_current_return_mean": (
                float(sell_margin_current_returns[sell_margin_align_masks].mean().item())
                if sell_margin_align_masks.any() else 0.0
            ),
            "sell_target_aux_labeled_steps": int(sell_margin_align_masks.sum().item()),
            "failed_to_sell_labeled_steps": int(failed_to_sell_masks.sum().item()),
            "buy_aux_labeled_steps": int(buy_aux_masks.sum().item()),
            "buy_aux_positive_rate": (
                float((buy_aux_targets[buy_aux_masks] >= 0).to(torch.float32).mean().item())
                if buy_aux_masks.any() else 0.0
            ),
        } | credit_metrics
