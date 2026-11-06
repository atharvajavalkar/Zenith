"""In-memory cache for latest signal and OHLCV candles."""

import asyncio
from typing import Any, Optional


class MarketCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._latest_signal: Optional[dict] = None
        self._candles: dict[str, list[dict]] = {}
        self._last_candle_time: Optional[str] = None
        self._worker_running: bool = False
        self._last_error: Optional[str] = None
        self._data_source: Optional[str] = None
        self._symbol: Optional[str] = None

    async def update(self, cycle: dict) -> bool:
        """Update cache. Returns True if candle_time changed."""
        async with self._lock:
            signal = cycle["signal"]
            candle_time = signal["CANDLE_TIME"]
            is_new = candle_time != self._last_candle_time
            self._latest_signal = signal
            self._candles = cycle["candles"]
            self._last_candle_time = candle_time
            self._last_error = None
            self._data_source = cycle.get("data_source") or signal.get("DATA_SOURCE")
            self._symbol = cycle.get("symbol") or signal.get("SYMBOL")
            return is_new

    async def set_error(self, message: str) -> None:
        async with self._lock:
            self._last_error = message

    async def set_worker_running(self, running: bool) -> None:
        async with self._lock:
            self._worker_running = running

    async def get_signal(self) -> Optional[dict]:
        async with self._lock:
            return self._latest_signal

    async def get_candles(self, timeframe: str, limit: int = 200) -> list[dict]:
        async with self._lock:
            key = timeframe.lower()
            bars = self._candles.get(key, [])
            return bars[-limit:]

    async def get_status(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "worker_running": self._worker_running,
                "last_candle_time": self._last_candle_time,
                "last_error": self._last_error,
                "has_signal": self._latest_signal is not None,
                "data_source": self._data_source,
                "symbol": self._symbol,
            }


market_cache = MarketCache()
