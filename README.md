# AI Trading System

A Python project for processing crypto market data, training trading models and placing orders through Kraken.

Built by [Quietforge909](https://github.com/Quietforge909).

![System architecture](docs/architecture.svg)

## My role

I planned the project structure, decided how the parts connect and designed the training process. I used AI tools to help write and debug the code.

## Data

The original data folder contains **8,880,723 hourly rows across 265 trading pairs**, from August 2017 to July 2026. These are raw file counts, not a claim that every model trained on every row.

There is a [24-hour BTC sample](data-sample/BTCUSDT-first24h.csv) and a [count for each pair](data-sample/coverage.json). The columns are timestamp, open, high, low, close, volume and USD volume.

## Saved results

These two versions were tested on 200 sampled 30-day windows each, with 20 coins per window. Both used deterministic actions, seed 42, a fresh $1,000 simulated account per window and a 0.5% fee on each side of a trade.

The saved windows run from January 2021 to March 2026. They overlap training dates, so these are **development backtests, not independent validation**. The versions also have different training histories; this comparison does not isolate the effect of changing the slot count.

| Metric | Four-slot version (v344) | Six-slot version (v345) |
|---|---:|---:|
| Average return per 30-day window | +0.84% | +0.67% |
| Median return | 0.00% | 0.00% |
| Worst drawdown in a window | −34.48% | −29.58% |
| Closed trades | 41 | 53 |
| Winning trades | 68.29% | 69.81% |
| Profit factor | 8.14 | 4.97 |
| Profitable windows | 17 / 200 | 18 / 200 |
| Forced exits | 0 | 0 |

The six-slot version had lower worst drawdown, but also lower average return. Both traded infrequently. The windows can overlap, and these percentages are not the compounded return of one account.

The four-slot version also lost **6.16%** in a later full-market simulation: June 18–July 18, 2026, across 261 coins, with four closed trades. That test used next-candle opening prices, 0.5% fees and an additional 0.15% execution cost per side. It shows why the positive development results were not enough to establish reliable performance.

Saved records: [four-slot test](results/v344-200episodes.json), [six-slot test](results/v345-200episodes.json), [later 30-day test](results/v344-later-30days.json). Personal file paths were removed; the exports include hashes of the original reports. No exchange account records are included.

To recalculate the numbers from those records:

```bash
python3 scripts/verify_results.py
```

This checks the saved summaries. It does not rerun the models or establish that the original data pipeline was free of leakage.

## Try the demo

With Python 3.10 or later, run:

```bash
python3 -m examples.offline_demo
```

The demo buys and sells at sample prices using the original account calculations. It shows how fees affect a winning trade and a losing trade. It runs offline and needs no extra packages.

The prices and actions are scripted. This demo shows the accounting, not the performance of a trained model.

## The code

| Part | File |
|---|---|
| Process price data and handle missing candles | [features/pipeline_v3.py](features/pipeline_v3.py) |
| Train forecasts on earlier data and test on later data | [supervised/walk_forward_v3.py](supervised/walk_forward_v3.py) |
| Learn trading decisions with PPO | [rl/agent/model.py](rl/agent/model.py), [rl/agent/joint_policy.py](rl/agent/joint_policy.py) |
| Run the three training phases | [rl/training/trainer_v7.py](rl/training/trainer_v7.py) |
| Test trades and check whether a model can move to the next phase | [rl/evaluation/evaluate_phase1_v7.py](rl/evaluation/evaluate_phase1_v7.py), [rl/evaluation/phase_gate.py](rl/evaluation/phase_gate.py) |
| Track holdings and place orders | [live/state.py](live/state.py), [live/executor.py](live/executor.py) |
| Run scheduled trading cycles | [live/live_daemon.py](live/live_daemon.py) |

The system also includes position sizing, BUY/HOLD/SELL actions, fee calculations, saved checkpoints and Discord updates.

This is the original Kraken project. The newer Coinbase project is separate.

## What's included

This repo contains 47 unchanged Python files from the July 2026 project, plus the diagram, demo, small data sample and saved test results. The original code hashes are in [docs/source-manifest.json](docs/source-manifest.json).

Full datasets, trained models, account records, API keys and old experiment logs are kept out. Running the full system needs additional data and model files. The saved simulations are not live account returns or proof that all three training phases passed.

The libraries used by the original code are listed in [requirements.txt](requirements.txt). Their versions are not pinned. The demo runs without them.

The [.env.example](.env.example) has empty credential fields and keeps live trading off.

To check that the original files match their saved hashes and look for common private-file patterns:

```bash
python3 scripts/check_export.py
```

This is a basic export check, not a full security audit.
