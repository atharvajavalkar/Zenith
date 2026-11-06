"""Runtime trade execution settings (adjustable from dashboard)."""

import threading
from typing import Optional

from config import (
    MAX_TRADES_PER_BUCKET as _DEFAULT_MAX,
    TRADE_5M_MODE,
)

_lock = threading.Lock()
_max_trades_per_bucket: int = _DEFAULT_MAX
_trading_halted: bool = False
_exec_cadence: str = "5m" if TRADE_5M_MODE else "1m"

CADENCE_SECONDS = {"1m": 60, "3m": 180, "5m": 300}
CADENCE_ZONE_LABEL = {"1m": "1M/5M", "3m": "3M/5M", "5m": "3M/5M"}
VALID_CADENCES = frozenset(CADENCE_SECONDS)


def get_exec_cadence() -> str:
    with _lock:
        return _exec_cadence


def get_trade_cadence() -> str:
    return get_exec_cadence()


def set_exec_cadence(cadence: str) -> str:
    global _exec_cadence
    key = cadence.lower().strip()
    if key not in VALID_CADENCES:
        key = "5m"
    with _lock:
        _exec_cadence = key
    return _exec_cadence


def get_zone_timeframes() -> str:
    return CADENCE_ZONE_LABEL.get(get_exec_cadence(), "3M/5M")


def get_trade_bucket_seconds() -> int:
    return CADENCE_SECONDS.get(get_exec_cadence(), 300)


def get_max_trades_per_bucket() -> int:
    with _lock:
        return _max_trades_per_bucket


def get_max_trades_per_minute() -> int:
    return get_max_trades_per_bucket()


def is_trading_halted() -> bool:
    with _lock:
        return _trading_halted


def set_trading_halted(value: bool) -> bool:
    global _trading_halted
    with _lock:
        _trading_halted = value
    return _trading_halted


def set_max_trades_per_bucket(value: int) -> int:
    global _max_trades_per_bucket, _trading_halted
    clamped = max(0, min(15, int(value)))
    with _lock:
        _max_trades_per_bucket = clamped
        if clamped > 0:
            _trading_halted = False
    return _max_trades_per_bucket


def set_max_trades_per_minute(value: int) -> int:
    return set_max_trades_per_bucket(value)


def halt_all_trading() -> None:
    global _max_trades_per_bucket, _trading_halted
    with _lock:
        _max_trades_per_bucket = 0
        _trading_halted = True


def get_trade_interval_seconds() -> float:
    n = get_max_trades_per_bucket()
    if n <= 0:
        return 0.0
    return get_trade_bucket_seconds() / n


def bucket_log_time(sig: dict, cadence: Optional[str] = None) -> str:
    c = (cadence or get_exec_cadence()).lower()
    if c == "1m":
        return sig.get("LOG_TIME_1M") or sig.get("CANDLE_TIME", "")
    if c == "3m":
        return sig.get("LOG_TIME_3M") or sig.get("CANDLE_TIME", "")
    return sig.get("LOG_TIME_5M") or sig.get("CANDLE_TIME_5M") or sig.get("CANDLE_TIME", "")
