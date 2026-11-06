"""Hourly cache for macro + technical market bias."""

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

BIAS_TTL_SECONDS = 3600


class BiasCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._data: Optional[dict[str, Any]] = None
        self._updated_at: float = 0.0
        self._refreshing: bool = False
        self._last_error: Optional[str] = None

    async def get(self, force_refresh: bool = False) -> Optional[dict[str, Any]]:
        async with self._lock:
            stale = (time.time() - self._updated_at) >= BIAS_TTL_SECONDS
            if self._data is not None and not force_refresh and not stale:
                return self._data
        await self.refresh(force=force_refresh or stale or self._data is None)
        async with self._lock:
            return self._data

    async def refresh(self, force: bool = False) -> None:
        async with self._lock:
            if self._refreshing:
                return
            if not force and self._data and (time.time() - self._updated_at) < BIAS_TTL_SECONDS:
                return
            self._refreshing = True

        try:
            from analysis.market_analyst import run_market_analysis

            result = await asyncio.to_thread(run_market_analysis)
            async with self._lock:
                self._data = result
                self._updated_at = time.time()
                self._last_error = None
            logger.info(
                "Market bias updated: %s instruments · primary %s (%.1f%%)",
                len(result.get("instruments") or []),
                result["overall"]["label"],
                result["overall"]["confidence"],
            )
        except Exception as exc:
            logger.exception("Market bias refresh failed: %s", exc)
            async with self._lock:
                self._last_error = str(exc)
        finally:
            async with self._lock:
                self._refreshing = False

    async def invalidate(self) -> None:
        """Force next get() to re-fetch (e.g. after paper ↔ live switch)."""
        async with self._lock:
            self._data = None
            self._updated_at = 0.0
            self._last_error = None

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "has_data": self._data is not None,
                "updated_at": self._data.get("updated_at") if self._data else None,
                "refreshing": self._refreshing,
                "last_error": self._last_error,
                "ttl_seconds": BIAS_TTL_SECONDS,
            }


bias_cache = BiasCache()
