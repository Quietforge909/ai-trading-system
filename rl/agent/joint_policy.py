"""Joint full-universe action distribution for one-action-per-timestep PPO."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rl.config import BUY_ALLOCATIONS


@dataclass(frozen=True)
class JointCandidate:
    kind: str
    coin_index: int = -1
    allocation_index: int = -1


def joint_action_logits(
    buy_logits: torch.Tensor,
    sell_logits: torch.Tensor,
    held_mask: torch.Tensor,
    allow_buy: bool,
    mode_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[JointCandidate]]:
    """Construct a categorical over noop, valid buys, and valid sells.

    Subtracting log(valid coin count) makes aggregate mode probability stable
    as the presented universe grows.  Scores are conditional log-odds against
    each head's native SKIP/HOLD alternative.
    """
    if buy_logits.dim() != 2 or sell_logits.dim() != 2:
        raise ValueError("Expected [coins, actions] logits")
    held_mask = held_mask.bool()
    if mode_logits is not None:
        if mode_logits.dim() != 2 or mode_logits.shape[-1] != 3:
            raise ValueError("Expected [coins, 3] mode logits")
        mode = mode_logits.mean(dim=0)
        # Exit-only mode head: preserve the proven flat BUY/NOOP selection
        # path and use the mode head only to decide whether held positions
        # should exit.  This avoids changing entry timing while still giving
        # deterministic evaluation a direct SELL-vs-NOOP control surface.
        values = [buy_logits.new_zeros(())]
        candidates = [JointCandidate("noop")]
        unheld = (~held_mask).nonzero(as_tuple=False).flatten().tolist()
        if allow_buy and unheld:
            correction = math.log(len(unheld))
            for coin_index in unheld:
                skip = buy_logits[coin_index, -1]
                for allocation_index in range(len(BUY_ALLOCATIONS)):
                    values.append(
                        buy_logits[coin_index, allocation_index] - skip - correction
                    )
                    candidates.append(JointCandidate("buy", coin_index, allocation_index))
        held = held_mask.nonzero(as_tuple=False).flatten().tolist()
        if held:
            raw = torch.stack([
                sell_logits[coin_index, 0] - sell_logits[coin_index, 1]
                for coin_index in held
            ])
            values.extend((
                mode[2] - mode[0] + raw - torch.logsumexp(raw, dim=0)
            ).unbind())
            candidates.extend(JointCandidate("sell", coin_index) for coin_index in held)
        return torch.stack(values), candidates

    values = [buy_logits.new_zeros(())]
    candidates = [JointCandidate("noop")]
    unheld = (~held_mask).nonzero(as_tuple=False).flatten().tolist()
    if allow_buy and unheld:
        correction = math.log(len(unheld))
        for coin_index in unheld:
            skip = buy_logits[coin_index, -1]
            for allocation_index in range(len(BUY_ALLOCATIONS)):
                values.append(buy_logits[coin_index, allocation_index] - skip - correction)
                candidates.append(JointCandidate("buy", coin_index, allocation_index))
    held = held_mask.nonzero(as_tuple=False).flatten().tolist()
    if held:
        correction = math.log(len(held))
        for coin_index in held:
            values.append(sell_logits[coin_index, 0] - sell_logits[coin_index, 1] - correction)
            candidates.append(JointCandidate("sell", coin_index))
    return torch.stack(values), candidates


def batched_joint_action_logits(
    buy_logits: torch.Tensor,
    sell_logits: torch.Tensor,
    held_mask: torch.Tensor,
    allow_buy: torch.Tensor,
    mode_logits: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized fixed-layout equivalent of :func:`joint_action_logits`.

    The output layout is NOOP, every coin/allocation BUY, then every coin
    SELL.  Illegal actions are negative infinity.  Removing those entries
    gives exactly the compact distribution produced by ``joint_action_logits``.
    A fixed layout lets PPO recompute a whole minibatch without a Python loop.
    """
    if buy_logits.dim() != 3 or sell_logits.dim() != 3:
        raise ValueError("Expected [batch, coins, actions] logits")
    if buy_logits.shape[:2] != sell_logits.shape[:2]:
        raise ValueError("Buy and sell batch shapes do not match")
    if held_mask.shape != buy_logits.shape[:2]:
        raise ValueError("Held mask shape does not match policy logits")
    batch, coins = buy_logits.shape[:2]
    held = held_mask.bool()
    valid = (
        torch.ones_like(held, dtype=torch.bool)
        if valid_mask is None else valid_mask.bool()
    )
    if valid.shape != held.shape:
        raise ValueError("Valid mask shape does not match policy logits")
    held = held & valid
    allow = allow_buy.bool().reshape(batch, 1, 1)
    if mode_logits is not None:
        if mode_logits.shape != (batch, coins, 3):
            raise ValueError("Mode logits shape must be [batch, coins, 3]")
        valid_count = valid.sum(dim=1).clamp_min(1).to(mode_logits.dtype)
        mode = (mode_logits * valid.unsqueeze(-1)).sum(dim=1) / valid_count.view(batch, 1)
        unheld = (~held) & valid
        unheld_count = unheld.sum(dim=1).clamp_min(1).to(buy_logits.dtype)
        raw_buy = (
            buy_logits[:, :, :-1]
            - buy_logits[:, :, -1:]
            - unheld_count.log().view(batch, 1, 1)
        ).reshape(batch, coins * len(BUY_ALLOCATIONS))
        buy_legal = (
            unheld.unsqueeze(-1).expand(batch, coins, len(BUY_ALLOCATIONS))
            & allow
        ).reshape(batch, coins * len(BUY_ALLOCATIONS))
        buy = raw_buy.masked_fill(~buy_legal, -torch.inf)
        raw_sell = sell_logits[:, :, 0] - sell_logits[:, :, 1]
        sell_raw_masked = raw_sell.masked_fill(~held, -torch.inf)
        sell = mode[:, 2:3] - mode[:, 0:1] + sell_raw_masked - torch.logsumexp(
            sell_raw_masked, dim=1, keepdim=True
        )
        sell = sell.masked_fill(~held, -torch.inf)
        noop = buy_logits.new_zeros((batch, 1))
        return torch.cat((noop, buy, sell), dim=1)

    unheld = (~held) & valid
    unheld_count = unheld.sum(dim=1).clamp_min(1).to(buy_logits.dtype)
    held_count = held.sum(dim=1).clamp_min(1).to(buy_logits.dtype)

    buy = buy_logits[:, :, :-1] - buy_logits[:, :, -1:] - unheld_count.log().view(batch, 1, 1)
    buy = buy.masked_fill(~(allow & unheld.unsqueeze(-1)), -torch.inf)
    sell = sell_logits[:, :, 0] - sell_logits[:, :, 1] - held_count.log().view(batch, 1)
    sell = sell.masked_fill(~held, -torch.inf)
    noop = buy_logits.new_zeros((batch, 1))
    return torch.cat((noop, buy.reshape(batch, coins * len(BUY_ALLOCATIONS)), sell), dim=1)


def fixed_action_index(candidate: JointCandidate, coins: int) -> int:
    """Map a valid compact candidate to the batched fixed action layout."""
    if candidate.kind == "noop":
        return 0
    if candidate.kind == "buy":
        return 1 + candidate.coin_index * len(BUY_ALLOCATIONS) + candidate.allocation_index
    if candidate.kind == "sell":
        return 1 + coins * len(BUY_ALLOCATIONS) + candidate.coin_index
    raise ValueError(f"Unknown joint candidate kind: {candidate.kind}")


def select_joint_action(
    buy_logits: torch.Tensor,
    sell_logits: torch.Tensor,
    held_mask: torch.Tensor,
    allow_buy: bool,
    deterministic: bool,
    mode_logits: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> tuple[JointCandidate, int, torch.Tensor, float]:
    logits, candidates = joint_action_logits(
        buy_logits, sell_logits, held_mask, allow_buy, mode_logits=mode_logits,
    )
    if deterministic:
        distribution = torch.distributions.Categorical(logits=logits)
        # Decide the portfolio-level mode by aggregate probability, then choose
        # the strongest concrete coin/size inside that mode.  A flat universe
        # of many tiny buy probabilities must not beat NOOP through cardinality,
        # and one valid buy need not individually exceed the entire buy mode.
        probabilities = distribution.probs
        kinds = ("noop", "buy", "sell")
        masses = torch.stack([
            probabilities.new_tensor(0.0) if not any(c.kind == kind for c in candidates)
            else probabilities[torch.tensor(
                [i for i, candidate in enumerate(candidates) if candidate.kind == kind],
                device=probabilities.device,
            )].sum()
            for kind in kinds
        ])
        selected_kind = kinds[int(torch.argmax(masses))]
        valid_indices = [i for i, candidate in enumerate(candidates) if candidate.kind == selected_kind]
        local = torch.argmax(probabilities[torch.tensor(valid_indices, device=probabilities.device)])
        index = torch.tensor(valid_indices[int(local)], device=probabilities.device)
    else:
        if temperature <= 0.0:
            raise ValueError("Sampling temperature must be positive")
        distribution = torch.distributions.Categorical(logits=logits / float(temperature))
        index = distribution.sample()
    candidate = candidates[int(index)]
    return (
        candidate,
        fixed_action_index(candidate, buy_logits.shape[0]),
        distribution.log_prob(index),
        float(distribution.probs[index]),
    )
