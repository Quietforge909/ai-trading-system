"""Hard evaluation gate: Phase 2 is forbidden until Phase 1 passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MAX_FORCED_SELL_RATIO = 0.25


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=(1, 2), required=True)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    if any(report.get("phase") != args.phase for report in reports):
        raise RuntimeError("Report phase mismatch")
    metrics = [report["aggregate"] for report in reports]
    unique_periods = {(r["period"]["start"], r["period"]["end"]) for r in reports}
    unique_seeds = {r["policy"]["seed"] for r in reports}
    checks = {
        "at_least_two_regimes": len(unique_periods) >= 2,
        "at_least_three_action_seeds": len(unique_seeds) >= 3,
        "full_training_cycle": args.phase != 1 or all(
            (report.get("training_coverage") or {}).get(
                "completed_selector_cycles", 0
            ) >= 1
            for report in reports
        ),
        "broad_window_sampling": all(
            report["period"].get("windows_per_coin") is None
            or report["period"].get("windows_per_coin", 0) >= 3
            for report in reports
        ),
        "all_reports_profitable_mean": all(m["mean_return"] > 0.0 for m in metrics),
        "aggregate_median_nonnegative": float(np.median([m["median_return"] for m in metrics])) >= 0.0,
        "all_profit_factors_above_one": all((m["profit_factor"] or 0.0) > 1.0 for m in metrics),
        "all_expectancies_positive": all(m["expectancy_dollars"] > 0.0 for m in metrics),
        "worst_drawdown_below_35pct": min(m["worst_max_drawdown"] for m in metrics) >= -0.35,
        "forced_sell_ratio_below_25pct": max(
            m["forced_sell_ratio"] for m in metrics
        ) <= MAX_FORCED_SELL_RATIO,
        "every_report_trades": all(m["trades"] > 0 for m in metrics),
    }
    result = {
        "phase": args.phase, "passed": all(checks.values()), "checks": checks,
        "reports": [str(path) for path in args.reports],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
