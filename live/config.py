from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


@dataclass(frozen=True)
class LiveConfig:
    api_key: str
    api_secret: str
    discord_webhook_url: str
    live_trading_enabled: bool
    checkpoint: Path
    state_path: Path
    work_root: Path
    max_slots: int
    buy_prob_threshold: float
    side_cost_bps: float
    taker_fee_fallback: float
    max_spread_bps: float
    marketable_limit_bps: float
    min_visible_coins: int
    rows: int
    schedule_minute: int
    quote_asset: str
    min_cash_reserve: float
    min_order_usd: float
    order_timeout_seconds: int
    discord_enabled: bool
    model_device: str
    max_data_age_minutes: int
    signal_source: str

    @classmethod
    def from_env(cls) -> "LiveConfig":
        signal_source = os.getenv("LIVE_SIGNAL_SOURCE", "binance").strip().lower()
        if signal_source not in {"binance", "kraken"}:
            raise RuntimeError("LIVE_SIGNAL_SOURCE must be 'binance' or 'kraken'")
        return cls(
            api_key=os.getenv("KRAKEN_API_KEY", ""),
            api_secret=os.getenv("KRAKEN_API_SECRET", ""),
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
            live_trading_enabled=_bool_env("LIVE_TRADING_ENABLED", False),
            checkpoint=Path(os.getenv(
                "LIVE_CHECKPOINT",
                "experiments/phase3_smoke/"
                "experiments/phase1_credit_diag/"
                "w64_v396_v394e79_1slot_newestbinance_seed42/"
                "checkpoint_episode_000079.pt",
            )),
            state_path=Path(os.getenv("LIVE_STATE_PATH", "live/state/kraken_bot_state.json")),
            work_root=Path(os.getenv("LIVE_WORK_ROOT", "live/runtime")),
            max_slots=_int_env("LIVE_MAX_SLOTS", 1),
            buy_prob_threshold=_float_env("LIVE_BUY_PROB_THRESHOLD", 0.0),
            side_cost_bps=_float_env("LIVE_SIDE_COST_BPS", 15.0),
            taker_fee_fallback=_float_env("LIVE_TAKER_FEE_FALLBACK", 0.008),
            max_spread_bps=_float_env("LIVE_MAX_SPREAD_BPS", 150.0),
            marketable_limit_bps=_float_env("LIVE_MARKETABLE_LIMIT_BPS", 10.0),
            min_visible_coins=_int_env("LIVE_MIN_VISIBLE_COINS", 100),
            rows=_int_env("LIVE_ROWS", 719),
            schedule_minute=_int_env("LIVE_SCHEDULE_MINUTE", 1),
            quote_asset=os.getenv("LIVE_QUOTE_ASSET", "USD").upper(),
            min_cash_reserve=_float_env("LIVE_MIN_CASH_RESERVE", 0.0),
            min_order_usd=_float_env("LIVE_MIN_ORDER_USD", 10.0),
            order_timeout_seconds=_int_env("LIVE_ORDER_TIMEOUT_SECONDS", 45),
            discord_enabled=_bool_env("DISCORD_ENABLED", True),
            model_device=os.getenv("LIVE_MODEL_DEVICE", "cpu"),
            max_data_age_minutes=_int_env("LIVE_MAX_DATA_AGE_MINUTES", 90),
            signal_source=signal_source,
        )

    @property
    def dry_run(self) -> bool:
        return not self.live_trading_enabled

    def require_private_api(self) -> None:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("KRAKEN_API_KEY and KRAKEN_API_SECRET are required")


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
