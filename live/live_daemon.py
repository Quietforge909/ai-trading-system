from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from live.account import summarize_account
from live.config import LiveConfig, load_dotenv
from live.data_pipeline import build_kraken_execution_manifest, build_live_binance_dataset, build_live_kraken_dataset
from live.discord_reporter import build_report, build_status_report, send_report
from live.executor import execute_buy, execute_sell
from live.kraken_client import KrakenClient
from live.model_runner import decide
from live.state import BotState
from rl.evaluation.evaluate_kraken_realistic_v280 import (
    build_corrected_rules_by_symbol,
    load_pair_rules,
)


def seconds_until_next_cycle(minute: int) -> float:
    now = datetime.now(timezone.utc)
    target = now.replace(minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(hours=1)
    return max(0.0, (target - now).total_seconds())


def require_fresh_pipeline(config: LiveConfig, as_of: datetime) -> None:
    latest_closed_at = as_of + timedelta(hours=1)
    age_minutes = (datetime.now(timezone.utc) - latest_closed_at).total_seconds() / 60.0
    if age_minutes > config.max_data_age_minutes:
        raise RuntimeError(
            f"Processed data is stale: latest candle open is {as_of.isoformat()}, "
            f"closed at {latest_closed_at.isoformat()} "
            f"({age_minutes:.1f} minutes old), limit is {config.max_data_age_minutes} minutes"
        )


def checkpoint_label(checkpoint: Path) -> str:
    return f"{checkpoint.parent.name}/{checkpoint.name}"


def build_live_signal_dataset(config: LiveConfig):
    if config.signal_source == "kraken":
        return build_live_kraken_dataset(config.work_root)
    return build_live_binance_dataset(config.work_root)


def order_closed(result) -> bool:
    order = result.raw.get("order") if isinstance(result.raw, dict) else None
    return isinstance(order, dict) and order.get("status") == "closed"


def equal_slot_allocation(max_slots: int, used_slots: int) -> float:
    empty_slots = max(1, int(max_slots) - int(used_slots))
    return 1.0 / empty_slots


def restricted_symbols_path(config: LiveConfig) -> Path:
    return config.state_path.with_name("restricted_symbols.json")


def load_restricted_symbols(path: Path) -> set[str]:
    if not path.exists():
        return set()
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return {str(symbol).upper() for symbol in raw}
    if isinstance(raw, dict):
        return {str(symbol).upper() for symbol in raw.get("symbols", [])}
    return set()


def save_restricted_symbols(path: Path, symbols: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(symbols), indent=2) + "\n")


def run_startup_warmup(config: LiveConfig) -> None:
    if config.discord_enabled and config.discord_webhook_url:
        send_report(config.discord_webhook_url, build_status_report(
            "Startup data warmup started",
            (
                "No orders will be placed during this step.\n"
                f"Fetching newest {config.signal_source.upper()} candles, rebuilding features, and refreshing supervised scores "
                "so missed offline hours are recovered before the first live cycle."
            ),
        ))
    pipeline = build_live_signal_dataset(config)
    require_fresh_pipeline(config, pipeline.as_of)
    if config.discord_enabled and config.discord_webhook_url:
        wait_seconds = seconds_until_next_cycle(config.schedule_minute)
        next_cycle = datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
        send_report(config.discord_webhook_url, build_status_report(
            "Startup data warmup finished",
            (
                f"Latest processed candle UTC: `{pipeline.as_of.isoformat()}`\n"
                f"Next live decision cycle UTC: `{next_cycle.isoformat()}`"
            ),
        ))


def run_cycle(config: LiveConfig) -> None:
    config.require_private_api()
    cycle_started_at = datetime.now(timezone.utc)
    if config.discord_enabled and config.discord_webhook_url:
        send_report(config.discord_webhook_url, build_status_report(
            "Live cycle started",
            (
                f"Mode: **{'DRY RUN' if config.dry_run else 'LIVE'}**\n"
                f"Model: `{checkpoint_label(config.checkpoint)}`\n"
                f"Signal source: `{config.signal_source}`\n"
                f"Started UTC: `{cycle_started_at.isoformat()}`\n"
                f"Status: fetching account, {config.signal_source.upper()} candles/features/scores, then Kraken execution prices."
            ),
        ))

    client = KrakenClient(config.api_key, config.api_secret)
    state = BotState.load(config.state_path)
    errors: list[str] = []
    execution_results = []

    balances = client.balance()
    open_orders = client.open_orders()
    pipeline = build_live_signal_dataset(config)
    require_fresh_pipeline(config, pipeline.as_of)
    kraken_manifest = build_kraken_execution_manifest(config.work_root)
    source_rules = load_pair_rules(kraken_manifest)
    rules = build_corrected_rules_by_symbol(source_rules)
    restricted_symbols_file = restricted_symbols_path(config)
    restricted_symbols = load_restricted_symbols(restricted_symbols_file)
    allowed_symbols = (set(rules) - restricted_symbols) | set(state.holdings)

    managed_symbols = sorted(state.holdings)
    managed_pair_ids = [rules[symbol].pair_id for symbol in managed_symbols if symbol in rules]
    managed_tickers = client.ticker(managed_pair_ids) if managed_pair_ids else {}
    ticker_by_symbol = {
        symbol: managed_tickers.get(rules[symbol].pair_id, managed_tickers.get(rules[symbol].source_symbol))
        for symbol in managed_symbols
        if symbol in rules
    }
    account = summarize_account(
        balances,
        {k: v for k, v in ticker_by_symbol.items() if v},
        state.holdings,
        config.quote_asset,
        open_orders,
    )

    decision, meta = decide(
        checkpoint=config.checkpoint,
        data_dir=pipeline.scored_dir,
        state_holdings=state.holdings,
        rows=config.rows,
        min_visible_coins=config.min_visible_coins,
        max_slots=config.max_slots,
        buy_prob_threshold=config.buy_prob_threshold,
        device_name=config.model_device,
        allowed_symbols=allowed_symbols,
        excluded_symbols=restricted_symbols,
    )

    sell_waiting_for_fill = False
    if open_orders:
        errors.append("Open Kraken orders exist; skipping new orders this cycle.")
    else:
        if decision.action == "sell" and decision.symbol and decision.symbol in state.holdings:
            if decision.symbol not in rules:
                errors.append(f"AI selected sell {decision.symbol}, but it is not tradable on Kraken; skipping sell.")
            else:
                rule = rules[decision.symbol]
                balances = client.balance()
                latest_ticker = client.ticker([rule.pair_id])
                ticker = latest_ticker.get(rule.pair_id) or latest_ticker.get(rule.source_symbol)
                holding = dict(state.holdings[decision.symbol])
                available_base = float(balances.get(rule.base, 0.0))
                if available_base <= 0.0:
                    errors.append(
                        f"AI selected sell {decision.symbol}, but Kraken balance for {rule.base} is zero; skipping sell."
                    )
                    result = None
                else:
                    state_size = float(holding.get("size", 0.0))
                    holding["size"] = available_base
                    if abs(available_base - state_size) > 0.0:
                        errors.append(
                            f"Adjusted sell size for {decision.symbol} from state {state_size:.12g} "
                            f"to Kraken balance {available_base:.12g}."
                        )
                    result = execute_sell(client, config, rule, holding, ticker)
                if result is None:
                    pass
                else:
                    execution_results.append(result)
                    if result.reason == "trading_restricted" and decision.symbol:
                        restricted_symbols.add(decision.symbol)
                        save_restricted_symbols(restricted_symbols_file, restricted_symbols)
                    if result.dry_run or order_closed(result):
                        state.register_sell(decision.symbol)
                        balances = client.balance() if not config.dry_run else balances
                    elif result.submitted:
                        sell_waiting_for_fill = True
                        errors.append("Sell order submitted but not closed before timeout; skipping buy until Kraken finalizes it.")

        elif decision.action == "buy" and decision.symbol and state.used_slots() < config.max_slots:
            if decision.symbol not in rules:
                errors.append(f"AI selected buy {decision.symbol}, but it is not tradable on Kraken; skipping buy.")
            else:
                rule = rules[decision.symbol]
                ticker = client.ticker([rule.pair_id]).get(rule.pair_id)
                available_cash = float(balances.get(config.quote_asset, 0.0))
                allocation_fraction = equal_slot_allocation(config.max_slots, state.used_slots())
                result = execute_buy(
                    client,
                    config,
                    rule,
                    ticker,
                    available_cash,
                    allocation_fraction,
                )
                execution_results.append(result)
                if result.reason == "trading_restricted" and decision.symbol:
                    restricted_symbols.add(decision.symbol)
                    save_restricted_symbols(restricted_symbols_file, restricted_symbols)
                if result.dry_run or order_closed(result):
                    state.register_buy(
                        symbol=rule.corrected_symbol,
                        base_asset=rule.base,
                        size=float(result.volume or 0.0),
                        entry_price=float(result.price or 0.0),
                        cost_basis=float(result.notional or 0.0),
                        allocation_fraction=allocation_fraction,
                        timestamp=decision.timestamp,
                        txid=result.txid,
                    )
                elif result.submitted:
                    errors.append("Buy order submitted but not closed before timeout; state was not updated until a confirmed fill.")

    state.last_cycle_ts = datetime.now(timezone.utc).isoformat()
    state.last_action = {
        "action": decision.action,
        "symbol": decision.symbol,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "checkpoint_episode": meta.get("checkpoint_episode"),
    }
    state.save(config.state_path)

    final_account = account
    try:
        final_balances = client.balance()
        final_account = summarize_account(
            final_balances,
            {k: v for k, v in ticker_by_symbol.items() if v},
            state.holdings,
            config.quote_asset,
            client.open_orders(),
        )
    except Exception as exc:
        errors.append(f"Final account refresh failed: {exc}")

    if config.discord_enabled:
        payload = build_report(
            final_account,
            decision,
            execution_results,
            config.dry_run,
            checkpoint_label(config.checkpoint),
            errors,
        )
        send_report(config.discord_webhook_url, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    load_dotenv(args.env_file)
    config = LiveConfig.from_env()
    if config.discord_enabled and config.discord_webhook_url:
        wait_seconds = 0.0 if args.once else seconds_until_next_cycle(config.schedule_minute)
        next_cycle = datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
        send_report(config.discord_webhook_url, build_status_report(
            "Kraken AI trader daemon is running",
            (
                f"Mode: **{'DRY RUN' if config.dry_run else 'LIVE'}**\n"
                f"Model: `{checkpoint_label(config.checkpoint)}`\n"
                f"Signal source: `{config.signal_source}`\n"
                f"Slots: `{config.max_slots}`\n"
                f"Buy threshold: `{config.buy_prob_threshold}`\n"
                f"Next cycle UTC: `{next_cycle.isoformat()}`"
            ),
        ))
    if args.once:
        run_cycle(config)
        return
    try:
        run_startup_warmup(config)
    except Exception as exc:
        if config.discord_enabled and config.discord_webhook_url:
            send_report(config.discord_webhook_url, build_status_report(
                "Startup data warmup failed",
                f"`{type(exc).__name__}`: {exc}",
                ok=False,
            ))
        raise
    while True:
        time.sleep(seconds_until_next_cycle(config.schedule_minute))
        try:
            run_cycle(config)
        except Exception as exc:
            if config.discord_enabled and config.discord_webhook_url:
                payload = {
                    "username": "Kraken AI Trader",
                    "embeds": [{
                        "title": "Hourly live cycle failed",
                        "description": f"`{type(exc).__name__}`: {exc}",
                        "color": 0xd64545,
                    }],
                }
                send_report(config.discord_webhook_url, payload)
            else:
                raise


if __name__ == "__main__":
    main()
