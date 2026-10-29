# AI Trading System

An engineering portfolio of a Python cryptocurrency trading platform: market-data processing, causal features, supervised forecasting, a three-phase reinforcement-learning workflow, chronological evaluation and exchange execution.

**Maintainer:** [Quietforge909](https://github.com/Quietforge909)  
**Repository intent:** private code review and portfolio demonstration.

![System architecture](docs/architecture.svg)

## My contribution

I defined the system architecture, component connections, directory organization, training sequence and experiment workflow, and directed AI-assisted implementation and debugging. This repository presents that integration work with selected original source code and a small runnable demonstration.

## Start here

Run the offline accounting demonstration with Python 3.10 or later:

```bash
python3 -m examples.offline_demo
```

It exercises the original portfolio accounting code on two scripted price scenarios. It needs no API keys, downloads, GPU or third-party Python packages. The actions and prices are illustrative; this demonstration does not run a trained policy or measure trading performance.

For a short source walkthrough:

| Component | Original implementation |
|---|---|
| Gap-aware, causal feature pipeline | [features/pipeline_v3.py](features/pipeline_v3.py) |
| Walk-forward supervised training | [supervised/walk_forward_v3.py](supervised/walk_forward_v3.py) |
| PPO model and joint action distribution | [rl/agent/model.py](rl/agent/model.py), [rl/agent/joint_policy.py](rl/agent/joint_policy.py) |
| Three-phase trainer and prerequisite handling | [rl/training/trainer_v7.py](rl/training/trainer_v7.py) |
| Chronological evaluation and phase gates | [rl/evaluation/evaluate_phase1_v7.py](rl/evaluation/evaluate_phase1_v7.py), [rl/evaluation/phase_gate.py](rl/evaluation/phase_gate.py) |
| Account state and exchange execution | [live/state.py](live/state.py), [live/executor.py](live/executor.py) |
| Scheduled live orchestration | [live/live_daemon.py](live/live_daemon.py) |

## Engineering scope

- Stored OHLCV ingestion and feature generation with explicit handling of missing hourly observations.
- Walk-forward supervised features and a separate PPO decision model.
- Training phases 1, 2 and 3, checkpoint continuation and prerequisite-gate checks.
- BUY allocation, HOLD, voluntary SELL and account-state handling.
- Evaluation with fees, realized results, drawdown and forced-exit statistics.
- Kraken execution integration, a dry-run setting, order handling and Discord reporting.

The original execution adapter in this snapshot is **Kraken**; the later Coinbase rebuild is a separate project and is not included here.

## What this repository proves

It provides inspectable original implementation and an executable accounting demonstration. Source presence is not proof that every training phase passed its gates, that deployment was profitable, or that the system is production-ready. No live-profit or independent-validation claim is made by this portfolio export.

Full market datasets, trained weights, exchange account records, runtime state, credentials and the large experiment archive remain outside this repository. Consequently, full training and trained-policy replay require additional private artifacts and configuration; this is not a turnkey trained trading bot.

## Setup beyond the demonstration

The optional [requirements.txt](requirements.txt) lists libraries imported by the selected source. It is an inventory, not a locked environment or a claim that the full pipeline was reproduced in this export. The offline demonstration does not require it.

The [.env.example](.env.example) contains blank credential placeholders and keeps live trading disabled. The demonstration never loads it or imports an exchange client.

## Source preservation

The 47 original Python files were selected from a preserved July 19, 2026 snapshot and copied without modification. [docs/source-manifest.json](docs/source-manifest.json) records their hashes. Portfolio files, the diagram and the offline demonstration were added separately. No original project files or private data were changed.

```bash
python3 scripts/check_export.py
```

This verifies the source hashes and checks the export for several credential, personal-path and private-artifact patterns. It is a bounded local check, not a comprehensive security certification.
