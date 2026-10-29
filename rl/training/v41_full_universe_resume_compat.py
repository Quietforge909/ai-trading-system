"""Pinned neutral-resume bridge for the v41 full-universe PPO baseline.

The old source bundle is accepted only after an empirical continuation replay:
immutable episode 34 reproduced episodes 35--39 under the current neutral
path with identical actions, rewards, selector/RNG state, and model weights
within CUDA floating-point accumulation tolerance.  This is deliberately much
narrower than a generic source-hash override.
"""

V41_FULL_UNIVERSE_SOURCE_SHA256 = {
    "rl/training/trainer_v7.py": "70f2451fca7aaa941436ca79a79168f87039c5bdaacf8615a1d4b8bdbba749af",
    "rl/agent/joint_ppo.py": "e0521d7bd18aa43aa1036105bd167f234701a6388a3570a677a86258e3c8b171",
    "rl/agent/joint_policy.py": "95bff0afca043a70b778dc02d62213e6833eb858501c275fe2cd83ca0c5a18ef",
    "rl/agent/model.py": "76fb415e1d91a19fbc1603829923fdbaba82e504cd427d53a840de0b75b5e21b",
    "rl/env/multi_env.py": "b9e7b634f092782f23bcf9aecca75005572a7c3a5f49f1bee61ef741412f592b",
    "rl/env/multi_portfolio.py": "615a768b3ee52f186951dd7f7df7c3bbb6951304cd814f8bb9a74acccf39334e",
    "rl/env/multi_selector.py": "d211cf8c3d6d42213291317821a5db8ab59eb31d145d52bfb4154b6a1977afd0",
}

# Filled with the current trainer bundle hash after the bridge is added.  It
# intentionally lives outside trainer_v7.py to avoid a self-hash cycle.
V41_FULL_UNIVERSE_CURRENT_NEUTRAL_SOURCE_SHA256 = {
    "rl/training/trainer_v7.py": "770b55f8c0e90fccf34801912f1900a779d5b8b8ed3942391fed8e70b4eb94d0",
    "rl/agent/joint_ppo.py": "61b52c71b3722f1e2f697041fa5f1e6f558ef9b8c547282430e204d25e7a8696",
    "rl/agent/joint_policy.py": "95bff0afca043a70b778dc02d62213e6833eb858501c275fe2cd83ca0c5a18ef",
    "rl/agent/model.py": "cc6e4804c19819bde4efdaa56814345ddc785d87c22241c26de345c740c75e73",
    "rl/env/multi_env.py": "a320f360e89178ab937d1fb304dc5882fa7f86a20bd1949f6ba617c04973ddbd",
    "rl/env/multi_portfolio.py": "615a768b3ee52f186951dd7f7df7c3bbb6951304cd814f8bb9a74acccf39334e",
    "rl/env/multi_selector.py": "d211cf8c3d6d42213291317821a5db8ab59eb31d145d52bfb4154b6a1977afd0",
}


def is_v41_full_universe_neutral_config(config: dict) -> bool:
    """Only admit the configuration covered by the continuation replay."""
    return (
        config.get("phase") == 1
        and config.get("universe_size") == 0
        and config.get("max_slots") == 1
        and bool(config.get("phase1_multicoin_diagnostic", False))
        and config.get("input_dim") == 104
        and config.get("position_dynamics_mode") == "active"
        and config.get("supervised_targets")
        == "opportunity_7d,drawdown_risk_7d,vol_expansion_risk_7d,profitable_after_fee_72h,future_min_return_7d,future_vol_7d"
        and not bool(config.get("cross_sectional_supervised_context", False))
        and int(config.get("universe_context_width", 0)) == 0
        and not bool(config.get("score_residual", config.get("score_residual_enabled", False)))
        and not bool(config.get("action_value_heads", False))
        and not bool(config.get("flat_action_value_heads", False))
        and not bool(config.get("sell_raw_residual", False))
        and not bool(config.get("buy_raw_residual", False))
        and float(config.get("held_policy_loss_weight", 1.0)) == 1.0
    )
