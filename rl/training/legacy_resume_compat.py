"""Pinned source bundles permitted for one neutral legacy-resume bridge."""

# Exact source hashes saved by the legacy active-dynamics Phase-1 family.
LEGACY_NEUTRAL_RESUME_SOURCE_SHA256 = {
    "rl/agent/joint_policy.py": "95bff0afca043a70b778dc02d62213e6833eb858501c275fe2cd83ca0c5a18ef",
    "rl/agent/joint_ppo.py": "599b317e944c62ee1357dc834bcbd38031947b36cd1dcc5bb41ccf1a365bf8a8",
    "rl/agent/model.py": "30285ff748303101229de0936ebed187d085c2a53a5abd5db6e7f03bd8cd785d",
    "rl/env/multi_env.py": "258274b9c8d4e4ff413b09009565577e213817656469ff4012652d90b6b38c7d",
    "rl/env/multi_portfolio.py": "615a768b3ee52f186951dd7f7df7c3bbb6951304cd814f8bb9a74acccf39334e",
    "rl/env/multi_selector.py": "eef9df9bc15e49b54ce4a5abf0d74ca76b49f4029edaa5c4fe45fa1061c7c7bd",
    "rl/training/trainer_v7.py": "0ce2fbf5114b2c534f31b946a674afd749416f921f0b6c6a2516618d8e864403",
}

# Reviewed current neutral-path bundle. Keeping it outside trainer_v7.py
# avoids a self-hash; any later tracked-source edit makes a legacy resume fail.
APPROVED_NEUTRAL_RESUME_CURRENT_SOURCE_SHA256 = {
    "rl/agent/joint_policy.py": "95bff0afca043a70b778dc02d62213e6833eb858501c275fe2cd83ca0c5a18ef",
    "rl/agent/joint_ppo.py": "75847fe2a1d38ee64c5f738171b2020c1be46b80e3805dbcd5414f97efe734c4",
    "rl/agent/model.py": "de29fcab3d017f51c2c780345b0344c8ad9fff85bddf6f5d027f4bae034a0731",
    "rl/env/multi_env.py": "6389e1582413aebf2226e3ac21cb9e18a3d5e019e92a48428fc54518fb54313c",
    "rl/env/multi_portfolio.py": "615a768b3ee52f186951dd7f7df7c3bbb6951304cd814f8bb9a74acccf39334e",
    "rl/env/multi_selector.py": "d38f2db05fb836a97fba9c6dc838d740ea3db848beaa47ee615ff2f931ff450e",
    "rl/training/trainer_v7.py": "e2f35c0b63c7fab7d8e3b01b1af7a5c008d694be9785f94067cfc4e36babf539",
}
