# rl/agent/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from rl.config import BUY_ACTION_DIM, SELL_ACTION_DIM


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.ln = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        x = x + h
        x = self.ln(x)
        return x


class ActorEncoder(nn.Module):
    """Independent policy encoder used to isolate flat and held gradients."""

    def __init__(
        self, input_dim: int, hidden1: int, hidden2: int, hidden3: int,
        num_res_blocks: int, dropout: float,
    ):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden1)
        self.ln_in = nn.LayerNorm(hidden1)
        self.fc_mid = nn.Linear(hidden1, hidden2)
        self.ln_mid = nn.LayerNorm(hidden2)
        self.res_blocks = nn.Sequential(*[
            ResidualMLPBlock(hidden2, dropout=dropout)
            for _ in range(num_res_blocks)
        ])
        self.fc_out = nn.Linear(hidden2, hidden3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.gelu(self.ln_in(self.fc_in(value)))
        value = F.gelu(self.ln_mid(self.fc_mid(value)))
        value = self.res_blocks(value)
        return F.gelu(self.fc_out(value))


class PPOTradingModel(nn.Module):
    """
    Shared trunk + two policy heads (buy/sell) + shared value head.
    Buy head: 5 logits  -> [buy25, buy50, buy75, buy100, skip]
    Sell head: 2 logits -> [sell, hold]
    Value head: 1 scalar
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 256,
        hidden2: int = 256,
        hidden3: int = 256,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        sell_head_width: int = 0,
        temporal_width: int = 0,
        actor_mode: str = "shared",
        state_value_heads: bool = False,
        held_feature_index: int | None = None,
        action_value_heads: bool = False,
        flat_action_value_heads: bool = False,
        action_value_detach_encoder: bool = False,
        sell_raw_residual: bool = False,
        sell_raw_residual_width: int = 0,
        buy_raw_residual: bool = False,
        buy_raw_residual_width: int = 0,
        score_residual: bool = False,
        score_residual_width: int = 0,
        score_feature_indices: tuple[int, ...] = (),
        score_residual_neutral_base: bool = False,
        universe_context_width: int = 0,
        mode_head: bool = False,
        boundary_sell_residual: bool = False,
        boundary_sell_feature_index: int | None = None,
        gated_buy_residual: bool = False,
        gated_buy_feature_index: int | None = None,
        gated_buy_feature_min: float = float("-inf"),
        gated_buy_feature_max: float = float("inf"),
        gated_buy_feature2_index: int | None = None,
        gated_buy_feature2_min: float = float("-inf"),
        gated_buy_feature2_max: float = float("inf"),
        gated_buy2_residual: bool = False,
        gated_buy2_feature_index: int | None = None,
        gated_buy2_feature_min: float = float("-inf"),
        gated_buy2_feature_max: float = float("inf"),
        gated_buy2_feature2_index: int | None = None,
        gated_buy2_feature2_min: float = float("-inf"),
        gated_buy2_feature2_max: float = float("inf"),
        gated_buy3_residual: bool = False,
        gated_buy3_feature_index: int | None = None,
        gated_buy3_feature_min: float = float("-inf"),
        gated_buy3_feature_max: float = float("inf"),
        gated_buy3_feature2_index: int | None = None,
        gated_buy3_feature2_min: float = float("-inf"),
        gated_buy3_feature2_max: float = float("inf"),
        gated_sell_residual: bool = False,
        gated_sell_feature_index: int | None = None,
        gated_sell_feature_min: float = float("-inf"),
        gated_sell_feature_max: float = float("inf"),
        gated_sell_feature2_index: int | None = None,
        gated_sell_feature2_min: float = float("-inf"),
        gated_sell_feature2_max: float = float("inf"),
        sell_target_aux_width: int = 0,
        sell_target_policy_residual: bool = False,
        sell_target_policy_residual_width: int = 0,
        sell_target_policy_residual_source: str = "score",
        buy_excluded_feature_indices: tuple[int, ...] = (),
        sell_excluded_feature_indices: tuple[int, ...] = (),
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.temporal_width = int(temporal_width)
        if actor_mode not in {"shared", "separate"}:
            raise ValueError("actor_mode must be shared or separate")
        if state_value_heads and held_feature_index is None:
            raise ValueError("state_value_heads requires held_feature_index")
        self.actor_mode = actor_mode
        self.state_value_heads = bool(state_value_heads)
        self.held_feature_index = held_feature_index
        self.action_value_heads = bool(action_value_heads)
        self.flat_action_value_heads = bool(flat_action_value_heads)
        self.action_value_detach_encoder = bool(action_value_detach_encoder)
        if self.action_value_detach_encoder and not (
            self.action_value_heads or self.flat_action_value_heads
        ):
            raise ValueError("Detached action value encoder requires action_value_heads")
        self.sell_raw_residual_enabled = bool(sell_raw_residual)
        self.sell_raw_residual_width = int(sell_raw_residual_width)
        self.buy_raw_residual_enabled = bool(buy_raw_residual)
        self.buy_raw_residual_width = int(buy_raw_residual_width)
        self.score_residual_enabled = bool(score_residual)
        self.score_residual_width = int(score_residual_width)
        self.score_feature_indices = tuple(int(index) for index in score_feature_indices)
        self.score_residual_neutral_base = bool(score_residual_neutral_base)
        self.universe_context_width = int(universe_context_width)
        self.universe_context_enabled = self.universe_context_width > 0
        self.mode_head_enabled = bool(mode_head)
        self.boundary_sell_residual_enabled = bool(boundary_sell_residual)
        self.boundary_sell_feature_index = (
            None if boundary_sell_feature_index is None
            else int(boundary_sell_feature_index)
        )
        self.gated_buy_residual_enabled = bool(gated_buy_residual)
        self.gated_buy_feature_index = (
            None if gated_buy_feature_index is None
            else int(gated_buy_feature_index)
        )
        self.gated_buy_feature_min = float(gated_buy_feature_min)
        self.gated_buy_feature_max = float(gated_buy_feature_max)
        self.gated_buy_feature2_index = (
            None if gated_buy_feature2_index is None
            else int(gated_buy_feature2_index)
        )
        self.gated_buy_feature2_min = float(gated_buy_feature2_min)
        self.gated_buy_feature2_max = float(gated_buy_feature2_max)
        self.gated_buy2_residual_enabled = bool(gated_buy2_residual)
        self.gated_buy2_feature_index = (
            None if gated_buy2_feature_index is None
            else int(gated_buy2_feature_index)
        )
        self.gated_buy2_feature_min = float(gated_buy2_feature_min)
        self.gated_buy2_feature_max = float(gated_buy2_feature_max)
        self.gated_buy2_feature2_index = (
            None if gated_buy2_feature2_index is None
            else int(gated_buy2_feature2_index)
        )
        self.gated_buy2_feature2_min = float(gated_buy2_feature2_min)
        self.gated_buy2_feature2_max = float(gated_buy2_feature2_max)
        self.gated_buy3_residual_enabled = bool(gated_buy3_residual)
        self.gated_buy3_feature_index = (
            None if gated_buy3_feature_index is None
            else int(gated_buy3_feature_index)
        )
        self.gated_buy3_feature_min = float(gated_buy3_feature_min)
        self.gated_buy3_feature_max = float(gated_buy3_feature_max)
        self.gated_buy3_feature2_index = (
            None if gated_buy3_feature2_index is None
            else int(gated_buy3_feature2_index)
        )
        self.gated_buy3_feature2_min = float(gated_buy3_feature2_min)
        self.gated_buy3_feature2_max = float(gated_buy3_feature2_max)
        self.gated_sell_residual_enabled = bool(gated_sell_residual)
        self.gated_sell_feature_index = (
            None if gated_sell_feature_index is None
            else int(gated_sell_feature_index)
        )
        self.gated_sell_feature_min = float(gated_sell_feature_min)
        self.gated_sell_feature_max = float(gated_sell_feature_max)
        self.gated_sell_feature2_index = (
            None if gated_sell_feature2_index is None
            else int(gated_sell_feature2_index)
        )
        self.gated_sell_feature2_min = float(gated_sell_feature2_min)
        self.gated_sell_feature2_max = float(gated_sell_feature2_max)
        self.sell_target_aux_width = int(sell_target_aux_width)
        self.sell_target_aux_enabled = self.sell_target_aux_width > 0
        self.sell_target_policy_residual_enabled = bool(sell_target_policy_residual)
        self.sell_target_policy_residual_width = int(sell_target_policy_residual_width)
        self.sell_target_policy_residual_source = str(sell_target_policy_residual_source)
        self.buy_excluded_feature_indices = tuple(
            int(index) for index in buy_excluded_feature_indices
        )
        self.sell_excluded_feature_indices = tuple(
            int(index) for index in sell_excluded_feature_indices
        )
        if self.sell_target_policy_residual_source not in {"score", "hidden"}:
            raise ValueError("sell_target_policy_residual_source must be score or hidden")
        if self.sell_raw_residual_width < 0:
            raise ValueError("sell_raw_residual_width must be nonnegative")
        if self.buy_raw_residual_width < 0:
            raise ValueError("buy_raw_residual_width must be nonnegative")
        if self.score_residual_width < 0:
            raise ValueError("score_residual_width must be nonnegative")
        if self.universe_context_width < 0:
            raise ValueError("universe_context_width must be nonnegative")
        if self.sell_target_aux_width < 0:
            raise ValueError("sell_target_aux_width must be nonnegative")
        if self.sell_target_policy_residual_width < 0:
            raise ValueError("sell_target_policy_residual_width must be nonnegative")
        if self.sell_target_policy_residual_enabled and not self.sell_target_aux_enabled:
            raise ValueError("sell_target_policy_residual requires sell_target_aux_width")
        if self.sell_target_policy_residual_width > 0 and not self.sell_target_policy_residual_enabled:
            raise ValueError("sell_target_policy_residual_width requires sell_target_policy_residual")
        if (
            self.sell_target_policy_residual_source == "hidden"
            and not self.sell_target_policy_residual_enabled
        ):
            raise ValueError("hidden sell-target residual source requires sell_target_policy_residual")
        if self.universe_context_enabled and self.temporal_width > 0:
            raise ValueError("Universe context is currently feed-forward only")
        if self.boundary_sell_residual_enabled and self.boundary_sell_feature_index is None:
            raise ValueError("Boundary sell residual requires a feature index")
        if (
            self.boundary_sell_feature_index is not None
            and not 0 <= self.boundary_sell_feature_index < self.input_dim
        ):
            raise ValueError("Boundary sell feature index is out of range")
        if self.gated_buy_residual_enabled and self.gated_buy_feature_index is None:
            raise ValueError("Gated buy residual requires a feature index")
        if (
            self.gated_buy_feature_index is not None
            and not 0 <= self.gated_buy_feature_index < self.input_dim
        ):
            raise ValueError("Gated buy feature index is out of range")
        if self.gated_buy_feature_min > self.gated_buy_feature_max:
            raise ValueError("Gated buy feature min must be <= max")
        if (
            self.gated_buy_feature2_index is not None
            and not 0 <= self.gated_buy_feature2_index < self.input_dim
        ):
            raise ValueError("Second gated buy feature index is out of range")
        if self.gated_buy_feature2_min > self.gated_buy_feature2_max:
            raise ValueError("Second gated buy feature min must be <= max")
        if self.gated_buy2_residual_enabled and self.gated_buy2_feature_index is None:
            raise ValueError("Second gated buy residual requires a feature index")
        if (
            self.gated_buy2_feature_index is not None
            and not 0 <= self.gated_buy2_feature_index < self.input_dim
        ):
            raise ValueError("Second gated buy residual feature index is out of range")
        if self.gated_buy2_feature_min > self.gated_buy2_feature_max:
            raise ValueError("Second gated buy residual feature min must be <= max")
        if (
            self.gated_buy2_feature2_index is not None
            and not 0 <= self.gated_buy2_feature2_index < self.input_dim
        ):
            raise ValueError("Second gated buy residual secondary feature index is out of range")
        if self.gated_buy2_feature2_min > self.gated_buy2_feature2_max:
            raise ValueError("Second gated buy residual secondary feature min must be <= max")
        if self.gated_buy3_residual_enabled and self.gated_buy3_feature_index is None:
            raise ValueError("Third gated buy residual requires a feature index")
        if (
            self.gated_buy3_feature_index is not None
            and not 0 <= self.gated_buy3_feature_index < self.input_dim
        ):
            raise ValueError("Third gated buy residual feature index is out of range")
        if self.gated_buy3_feature_min > self.gated_buy3_feature_max:
            raise ValueError("Third gated buy residual feature min must be <= max")
        if (
            self.gated_buy3_feature2_index is not None
            and not 0 <= self.gated_buy3_feature2_index < self.input_dim
        ):
            raise ValueError("Third gated buy residual secondary feature index is out of range")
        if self.gated_buy3_feature2_min > self.gated_buy3_feature2_max:
            raise ValueError("Third gated buy residual secondary feature min must be <= max")
        if self.gated_sell_residual_enabled and self.gated_sell_feature_index is None:
            raise ValueError("Gated sell residual requires a feature index")
        if (
            self.gated_sell_feature_index is not None
            and not 0 <= self.gated_sell_feature_index < self.input_dim
        ):
            raise ValueError("Gated sell feature index is out of range")
        if self.gated_sell_feature_min > self.gated_sell_feature_max:
            raise ValueError("Gated sell feature min must be <= max")
        if (
            self.gated_sell_feature2_index is not None
            and not 0 <= self.gated_sell_feature2_index < self.input_dim
        ):
            raise ValueError("Second gated sell feature index is out of range")
        if self.gated_sell_feature2_min > self.gated_sell_feature2_max:
            raise ValueError("Second gated sell feature min must be <= max")
        if self.score_residual_enabled and (
            not self.score_feature_indices or self.score_residual_width < 1
        ):
            raise ValueError("Score residual requires feature indices and positive width")
        self.temporal = (
            nn.GRU(self.input_dim, self.temporal_width)
            if self.temporal_width > 0 else None
        )
        self.fc_in = nn.Linear(input_dim + self.temporal_width, hidden1)
        self.ln_in = nn.LayerNorm(hidden1)

        self.fc_mid = nn.Linear(hidden1, hidden2)
        self.ln_mid = nn.LayerNorm(hidden2)

        self.res_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden2, dropout=dropout) for _ in range(num_res_blocks)]
        )

        self.fc_out = nn.Linear(hidden2, hidden3)

        encoder_input = input_dim + self.temporal_width
        self.buy_encoder = (
            ActorEncoder(
                encoder_input, hidden1, hidden2, hidden3,
                num_res_blocks, dropout,
            ) if actor_mode == "separate" else None
        )
        self.sell_encoder = (
            ActorEncoder(
                encoder_input, hidden1, hidden2, hidden3,
                num_res_blocks, dropout,
            ) if actor_mode == "separate" else None
        )

        # Heads
        self.buy_head = nn.Linear(hidden3, BUY_ACTION_DIM)
        self.buy_raw_residual = (
            nn.Sequential(
                nn.Linear(self.input_dim, self.buy_raw_residual_width),
                nn.LayerNorm(self.buy_raw_residual_width),
                nn.GELU(),
                nn.Linear(self.buy_raw_residual_width, BUY_ACTION_DIM, bias=False),
            )
            if self.buy_raw_residual_enabled and self.buy_raw_residual_width > 0
            else nn.Linear(self.input_dim, BUY_ACTION_DIM, bias=False)
            if self.buy_raw_residual_enabled else None
        )
        self.sell_adapter = (
            nn.Sequential(
                nn.Linear(hidden3, sell_head_width),
                nn.LayerNorm(sell_head_width),
                nn.GELU(),
            )
            if sell_head_width > 0 else nn.Identity()
        )
        self.sell_head = nn.Linear(
            sell_head_width if sell_head_width > 0 else hidden3,
            SELL_ACTION_DIM,
        )
        self.sell_raw_residual = (
            nn.Sequential(
                nn.Linear(self.input_dim, self.sell_raw_residual_width),
                nn.LayerNorm(self.sell_raw_residual_width),
                nn.GELU(),
                nn.Linear(self.sell_raw_residual_width, 1, bias=False),
            )
            if self.sell_raw_residual_enabled and self.sell_raw_residual_width > 0
            else nn.Linear(self.input_dim, 1, bias=False)
            if self.sell_raw_residual_enabled else None
        )
        self.value_head = nn.Linear(hidden3, 1)
        self.sell_q_head = nn.Linear(hidden3, SELL_ACTION_DIM) if action_value_heads else None
        # Flat-state action values keep the four learned allocation actions
        # distinct from SKIP. This head is auxiliary training credit only;
        # policy logits remain the sole execution authority.
        self.buy_allocation_q_head = (
            nn.Linear(hidden3, BUY_ACTION_DIM) if flat_action_value_heads else None
        )
        self.mode_head = nn.Linear(hidden3, 3) if self.mode_head_enabled else None
        self.boundary_sell_logit_scale = (
            nn.Parameter(torch.zeros(1))
            if self.boundary_sell_residual_enabled else None
        )
        self.gated_buy_logit_scale = (
            nn.Parameter(torch.zeros(1))
            if self.gated_buy_residual_enabled else None
        )
        self.gated_buy2_logit_scale = (
            nn.Parameter(torch.zeros(1))
            if self.gated_buy2_residual_enabled else None
        )
        self.gated_buy3_logit_scale = (
            nn.Parameter(torch.zeros(1))
            if self.gated_buy3_residual_enabled else None
        )
        self.gated_sell_logit_scale = (
            nn.Parameter(torch.zeros(1))
            if self.gated_sell_residual_enabled else None
        )
        self.flat_value_head = nn.Linear(hidden3, 1) if state_value_heads else None
        self.held_value_head = nn.Linear(hidden3, 1) if state_value_heads else None
        # Construct this after every established head so adding the optional
        # path leaves seeded base initialization bit-for-bit unchanged.
        # Scores remain ordinary causal context, never a legality or action
        # filter; the residual only expands learned representation capacity.
        if self.score_residual_enabled:
            # Module constructors initialize parameters eagerly. Restore the
            # RNG so this opt-in module cannot perturb the seeded base model
            # before _init_weights reaches its zero-initialized final layer.
            rng_state = torch.get_rng_state()
            self.score_residual = nn.Sequential(
                nn.Linear(len(self.score_feature_indices), self.score_residual_width),
                nn.LayerNorm(self.score_residual_width),
                nn.GELU(),
                nn.Linear(self.score_residual_width, hidden3, bias=False),
            )
            torch.set_rng_state(rng_state)
        else:
            self.score_residual = None
        if self.universe_context_enabled:
            # Save/restore eager-constructor RNG consumption so a zero
            # initialized optional context does not perturb the base network.
            rng_state = torch.get_rng_state()
            self.universe_context = nn.Sequential(
                nn.Linear(hidden3 * 3, self.universe_context_width),
                nn.LayerNorm(self.universe_context_width),
                nn.GELU(),
                nn.Linear(self.universe_context_width, hidden3, bias=False),
            )
            torch.set_rng_state(rng_state)
        else:
            self.universe_context = None
        if self.sell_target_aux_enabled:
            # This training-only predictor is deliberately independent of the
            # action logits.  Preserve seeded base initialization exactly; the
            # head is initialized after established modules and never feeds
            # buy/sell/value outputs.
            rng_state = torch.get_rng_state()
            self.sell_target_aux_head = nn.Sequential(
                nn.Linear(self.input_dim, self.sell_target_aux_width),
                nn.LayerNorm(self.sell_target_aux_width),
                nn.GELU(),
                nn.Linear(self.sell_target_aux_width, 1),
            )
            torch.set_rng_state(rng_state)
        else:
            self.sell_target_aux_head = None
        if self.sell_target_policy_residual_enabled:
            rng_state = torch.get_rng_state()
            residual_input_dim = (
                self.sell_target_aux_width
                if self.sell_target_policy_residual_source == "hidden"
                else 1
            )
            self.sell_target_policy_residual = (
                nn.Sequential(
                    nn.Linear(residual_input_dim, self.sell_target_policy_residual_width),
                    nn.LayerNorm(self.sell_target_policy_residual_width),
                    nn.GELU(),
                    nn.Linear(self.sell_target_policy_residual_width, 1, bias=False),
                )
                if self.sell_target_policy_residual_width > 0
                else nn.Linear(residual_input_dim, 1, bias=False)
            )
            torch.set_rng_state(rng_state)
        else:
            self.sell_target_policy_residual = None

        self._init_weights()

    def _init_weights(self):
        # Conservative init helps PPO stability
        optional_modules = set()
        if self.score_residual_enabled:
            optional_modules.update(self.score_residual.modules())
        if self.universe_context_enabled:
            optional_modules.update(self.universe_context.modules())
        if self.sell_target_aux_enabled:
            optional_modules.update(self.sell_target_aux_head.modules())
        if self.sell_target_policy_residual_enabled:
            optional_modules.update(self.sell_target_policy_residual.modules())
        for m in self.modules():
            if m in optional_modules:
                continue
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Smaller init for heads
        nn.init.orthogonal_(self.buy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.sell_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        if self.state_value_heads:
            nn.init.orthogonal_(self.flat_value_head.weight, gain=1.0)
            nn.init.orthogonal_(self.held_value_head.weight, gain=1.0)
        if self.action_value_heads:
            # The first PPO update uses ordinary GAE; learned Q credit enters
            # gradually from an exactly neutral zero prediction.
            nn.init.zeros_(self.sell_q_head.weight)
            nn.init.zeros_(self.sell_q_head.bias)
        if self.flat_action_value_heads:
            nn.init.zeros_(self.buy_allocation_q_head.weight)
            nn.init.zeros_(self.buy_allocation_q_head.bias)
        if self.mode_head_enabled:
            nn.init.orthogonal_(self.mode_head.weight, gain=0.01)
            nn.init.zeros_(self.mode_head.bias)
        if self.sell_raw_residual_enabled:
            # Add this path to a proven checkpoint without changing any
            # initial policy output.
            output_layer = (
                self.sell_raw_residual[-1]
                if self.sell_raw_residual_width > 0
                else self.sell_raw_residual
            )
            nn.init.zeros_(output_layer.weight)
        if self.buy_raw_residual_enabled:
            output_layer = (
                self.buy_raw_residual[-1]
                if self.buy_raw_residual_width > 0
                else self.buy_raw_residual
            )
            nn.init.zeros_(output_layer.weight)
        if self.score_residual_enabled:
            # Initialize the optional module only after every established
            # module/head has consumed its historical RNG sequence.
            for m in self.score_residual.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            nn.init.zeros_(self.score_residual[-1].weight)
        if self.universe_context_enabled:
            for m in self.universe_context.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            nn.init.zeros_(self.universe_context[-1].weight)
        if self.sell_target_aux_enabled:
            for m in self.sell_target_aux_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        if self.sell_target_policy_residual_enabled:
            if self.sell_target_policy_residual_width > 0:
                for m in self.sell_target_policy_residual.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight, gain=1.0)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)
                nn.init.zeros_(self.sell_target_policy_residual[-1].weight)
            else:
                nn.init.zeros_(self.sell_target_policy_residual.weight)

        # Four allocation categories must not make BUY four times likelier
        # than SKIP at initialization.  Start with P(BUY aggregate)=0.5 and
        # distribute that mass evenly across allocations.
        with torch.no_grad():
            self.buy_head.bias[:-1].fill_(math.log(0.125))
            self.buy_head.bias[-1].fill_(math.log(0.5))

    @property
    def is_recurrent(self) -> bool:
        return self.temporal is not None

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor | None:
        if not self.is_recurrent:
            return None
        reference = next(self.parameters())
        return torch.zeros(
            1, batch_size, self.temporal_width,
            device=device or reference.device, dtype=reference.dtype,
        )

    def _encode(
        self, model_input: torch.Tensor, raw_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def masked_input(indices: tuple[int, ...]) -> torch.Tensor:
            if not indices:
                return model_input
            masked = model_input.clone()
            masked[..., list(indices)] = 0.0
            return masked

        def encode_trunk(value: torch.Tensor) -> torch.Tensor:
            if self.score_residual_enabled and self.score_residual_neutral_base:
                value = value.clone()
                value[..., list(self.score_feature_indices)] = 0.5
            encoded = F.gelu(self.ln_in(self.fc_in(value)))
            encoded = F.gelu(self.ln_mid(self.fc_mid(encoded)))
            encoded = self.res_blocks(encoded)
            return F.gelu(self.fc_out(encoded))

        if self.score_residual_enabled and self.score_residual_neutral_base:
            model_input = model_input.clone()
            model_input[..., list(self.score_feature_indices)] = 0.5

        shared_excluded = tuple(sorted(
            set(self.buy_excluded_feature_indices)
            | set(self.sell_excluded_feature_indices)
        ))
        x = encode_trunk(masked_input(shared_excluded))
        buy_input = masked_input(self.buy_excluded_feature_indices)
        sell_input = masked_input(self.sell_excluded_feature_indices)
        if self.buy_encoder is not None:
            buy_features = self.buy_encoder(buy_input)
        elif self.buy_excluded_feature_indices:
            buy_features = encode_trunk(buy_input)
        else:
            buy_features = x
        if self.sell_encoder is not None:
            sell_features = self.sell_encoder(sell_input)
        elif self.sell_excluded_feature_indices:
            sell_features = encode_trunk(sell_input)
        else:
            sell_features = x
        if self.score_residual_enabled:
            score_delta = self.score_residual(raw_observation[..., list(self.score_feature_indices)])
            x = x + score_delta
            buy_features = buy_features + score_delta
            sell_features = sell_features + score_delta
        return x, buy_features, sell_features

    def _heads_from_features(
        self, x: torch.Tensor, buy_features: torch.Tensor,
        sell_features: torch.Tensor, raw_observation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        buy_observation = self._masked_observation(
            raw_observation, self.buy_excluded_feature_indices
        )
        sell_observation = self._masked_observation(
            raw_observation, self.sell_excluded_feature_indices
        )
        buy_logits = self.buy_head(buy_features)
        if self.buy_raw_residual_enabled:
            buy_logits = buy_logits + self.buy_raw_residual(buy_observation)
        if self.gated_buy_residual_enabled:
            feature = buy_observation[
                ..., self.gated_buy_feature_index:self.gated_buy_feature_index + 1
            ]
            gate = (
                (feature >= self.gated_buy_feature_min)
                & (feature <= self.gated_buy_feature_max)
            )
            if self.gated_buy_feature2_index is not None:
                feature2 = buy_observation[
                    ..., self.gated_buy_feature2_index:self.gated_buy_feature2_index + 1
                ]
                gate = gate & (
                    (feature2 >= self.gated_buy_feature2_min)
                    & (feature2 <= self.gated_buy_feature2_max)
                )
            gate = gate.to(buy_logits.dtype)
            residual = gate * self.gated_buy_logit_scale
            buy_logits = buy_logits + torch.cat(
                (
                    -0.5 * residual.expand(
                        *buy_logits.shape[:-1], buy_logits.shape[-1] - 1
                    ),
                    0.5 * residual,
                ),
                dim=-1,
            )
        if self.gated_buy2_residual_enabled:
            feature = buy_observation[
                ..., self.gated_buy2_feature_index:self.gated_buy2_feature_index + 1
            ]
            gate = (
                (feature >= self.gated_buy2_feature_min)
                & (feature <= self.gated_buy2_feature_max)
            )
            if self.gated_buy2_feature2_index is not None:
                feature2 = buy_observation[
                    ..., self.gated_buy2_feature2_index:self.gated_buy2_feature2_index + 1
                ]
                gate = gate & (
                    (feature2 >= self.gated_buy2_feature2_min)
                    & (feature2 <= self.gated_buy2_feature2_max)
                )
            residual = gate.to(buy_logits.dtype) * self.gated_buy2_logit_scale
            buy_logits = buy_logits + torch.cat(
                (
                    -0.5 * residual.expand(
                        *buy_logits.shape[:-1], buy_logits.shape[-1] - 1
                    ),
                    0.5 * residual,
                ),
                dim=-1,
            )
        if self.gated_buy3_residual_enabled:
            feature = buy_observation[
                ..., self.gated_buy3_feature_index:self.gated_buy3_feature_index + 1
            ]
            gate = (
                (feature >= self.gated_buy3_feature_min)
                & (feature <= self.gated_buy3_feature_max)
            )
            if self.gated_buy3_feature2_index is not None:
                feature2 = buy_observation[
                    ..., self.gated_buy3_feature2_index:self.gated_buy3_feature2_index + 1
                ]
                gate = gate & (
                    (feature2 >= self.gated_buy3_feature2_min)
                    & (feature2 <= self.gated_buy3_feature2_max)
                )
            residual = gate.to(buy_logits.dtype) * self.gated_buy3_logit_scale
            buy_logits = buy_logits + torch.cat(
                (
                    -0.5 * residual.expand(
                        *buy_logits.shape[:-1], buy_logits.shape[-1] - 1
                    ),
                    0.5 * residual,
                ),
                dim=-1,
            )
        sell_logits = self.sell_head(self.sell_adapter(sell_features))
        if self.sell_raw_residual_enabled:
            residual = self.sell_raw_residual(sell_observation)
            sell_logits = sell_logits + torch.cat(
                (0.5 * residual, -0.5 * residual), dim=-1
            )
        if self.boundary_sell_residual_enabled:
            boundary = sell_observation[..., self.boundary_sell_feature_index:self.boundary_sell_feature_index + 1]
            residual = boundary * self.boundary_sell_logit_scale
            sell_logits = sell_logits + torch.cat(
                (0.5 * residual, -0.5 * residual), dim=-1
            )
        if self.gated_sell_residual_enabled:
            feature = sell_observation[
                ..., self.gated_sell_feature_index:self.gated_sell_feature_index + 1
            ]
            gate = (
                (feature >= self.gated_sell_feature_min)
                & (feature <= self.gated_sell_feature_max)
            )
            if self.gated_sell_feature2_index is not None:
                feature2 = sell_observation[
                    ..., self.gated_sell_feature2_index:self.gated_sell_feature2_index + 1
                ]
                gate = gate & (
                    (feature2 >= self.gated_sell_feature2_min)
                    & (feature2 <= self.gated_sell_feature2_max)
                )
            residual = gate.to(sell_logits.dtype) * self.gated_sell_logit_scale
            sell_logits = sell_logits + torch.cat(
                (0.5 * residual, -0.5 * residual), dim=-1
            )
        sell_target_aux = None
        if self.sell_target_aux_enabled:
            sell_target_aux_hidden = self.sell_target_aux_head[:3](raw_observation)
            sell_target_aux = self.sell_target_aux_head[3](sell_target_aux_hidden).squeeze(-1)
            if self.sell_target_policy_residual_enabled:
                residual_input = (
                    sell_target_aux_hidden.detach()
                    if self.sell_target_policy_residual_source == "hidden"
                    else sell_target_aux.detach().unsqueeze(-1)
                )
                residual = self.sell_target_policy_residual(
                    residual_input
                )
                sell_logits = sell_logits + torch.cat(
                    (0.5 * residual, -0.5 * residual), dim=-1
                )
        if self.state_value_heads:
            held = raw_observation[..., self.held_feature_index] > 0.5
            flat_value = self.flat_value_head(x).squeeze(-1)
            held_value = self.held_value_head(x).squeeze(-1)
            value = torch.where(held, held_value, flat_value)
        else:
            value = self.value_head(x).squeeze(-1)  # (batch,)

        output = {"buy_logits": buy_logits, "sell_logits": sell_logits, "value": value}
        if self.mode_head_enabled:
            output["mode_logits"] = self.mode_head(x)
        if self.action_value_heads:
            q_features = x.detach() if self.action_value_detach_encoder else x
            output["sell_q"] = self.sell_q_head(q_features)
        if self.flat_action_value_heads:
            q_features = x.detach() if self.action_value_detach_encoder else x
            output["buy_allocation_q"] = self.buy_allocation_q_head(q_features)
        if self.sell_target_aux_enabled:
            output["sell_target_aux"] = sell_target_aux
        return output

    def _masked_observation(
        self, observation: torch.Tensor, indices: tuple[int, ...],
    ) -> torch.Tensor:
        if not indices:
            return observation
        masked = observation.clone()
        masked[..., list(indices)] = 0.0
        return masked

    def _heads(
        self, model_input: torch.Tensor, raw_observation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._heads_from_features(*self._encode(model_input, raw_observation), raw_observation)

    def forward_group(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Permutation-equivariant feed-forward evaluation of [batch, coins, features]."""
        if not self.universe_context_enabled:
            flat = self._heads(obs.reshape(-1, self.input_dim), obs.reshape(-1, self.input_dim))
            batch, coins = obs.shape[:2]
            output = {
                "buy_logits": flat["buy_logits"].reshape(batch, coins, -1),
                "sell_logits": flat["sell_logits"].reshape(batch, coins, -1),
                "value": flat["value"].reshape(batch, coins),
            }
            if "mode_logits" in flat:
                output["mode_logits"] = flat["mode_logits"].reshape(batch, coins, -1)
            if "sell_q" in flat:
                output["sell_q"] = flat["sell_q"].reshape(batch, coins, -1)
            if "buy_allocation_q" in flat:
                output["buy_allocation_q"] = flat["buy_allocation_q"].reshape(batch, coins, -1)
            if "sell_target_aux" in flat:
                output["sell_target_aux"] = flat["sell_target_aux"].reshape(batch, coins)
            return output
        if obs.dim() != 3 or obs.shape[-1] != self.input_dim:
            raise ValueError("Universe context expects [batch, coins, input_dim]")
        x, buy_features, sell_features = self._encode(obs, obs)
        mean = x.mean(dim=1, keepdim=True).expand_as(x)
        maximum = x.amax(dim=1, keepdim=True).expand_as(x)
        delta = self.universe_context(torch.cat((x, mean, maximum), dim=-1))
        return self._heads_from_features(
            x + delta, buy_features + delta, sell_features + delta, obs,
        )

    def forward_step(
        self, obs: torch.Tensor, hidden: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
        if self.is_recurrent:
            if hidden is None:
                hidden = self.initial_state(len(obs), obs.device)
            temporal, next_hidden = self.temporal(obs.unsqueeze(0), hidden)
            model_input = torch.cat((obs, temporal.squeeze(0)), dim=-1)
        else:
            next_hidden = None
            model_input = obs
        if self.universe_context_enabled:
            output = self.forward_group(obs.unsqueeze(0))
            output = {key: value.squeeze(0) for key, value in output.items()}
        else:
            output = self._heads(model_input, obs)
        if single:
            output = {key: value.squeeze(0) for key, value in output.items()}
        return output, next_hidden

    def forward_sequence(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        """Evaluate a causal [time, entities, features] sequence."""
        if obs.dim() != 3:
            raise ValueError("Expected [time, entities, features] observations")
        time, entities, features = obs.shape
        if features != self.input_dim:
            raise ValueError("Temporal observation width mismatch")
        if self.is_recurrent:
            if hidden is None:
                hidden = self.initial_state(entities, obs.device)
            temporal, next_hidden = self.temporal(obs, hidden)
            model_input = torch.cat((obs, temporal), dim=-1)
        else:
            next_hidden = None
            model_input = obs
        if self.universe_context_enabled:
            output = self.forward_group(obs)
            return output, next_hidden
        flat = self._heads(
            model_input.reshape(time * entities, -1),
            obs.reshape(time * entities, features),
        )
        output = {
            "buy_logits": flat["buy_logits"].reshape(time, entities, -1),
            "sell_logits": flat["sell_logits"].reshape(time, entities, -1),
            "value": flat["value"].reshape(time, entities),
        }
        if "mode_logits" in flat:
            output["mode_logits"] = flat["mode_logits"].reshape(time, entities, -1)
        if "sell_q" in flat:
            output["sell_q"] = flat["sell_q"].reshape(time, entities, -1)
        if "buy_allocation_q" in flat:
            output["buy_allocation_q"] = flat["buy_allocation_q"].reshape(time, entities, -1)
        if "sell_target_aux" in flat:
            output["sell_target_aux"] = flat["sell_target_aux"].reshape(time, entities)
        return output, next_hidden

    def forward(self, obs: torch.Tensor):
        """Feed-forward compatibility; recurrent callers should retain state."""
        return self.forward_step(obs, hidden=None)[0]
