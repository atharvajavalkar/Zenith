"""TradingView OHLCV data fetching with retries and fallbacks (no login required)."""

import logging
import threading
import time
from typing import Optional, Tuple

import pandas as pd
from tvDatafeed import Interval, TvDatafeed

from config import (
    FETCH_BETWEEN_TF_DELAY,
    FETCH_DELAY_SECONDS,
    FETCH_RETRIES,
    FETCH_SOURCES,
    TV_PASSWORD,
    TV_USERNAME,
    TV_USE_LOGIN,
)

# Anonymous TradingView feed works without credentials — silence library nologin spam.
logging.getLogger("tvDatafeed.main").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

_INTERVAL_TO_KEY = {
    Interval.in_1_minute: "1m",
    Interval.in_3_minute: "3m",
    Interval.in_5_minute: "5m",
    Interval.in_15_minute: "15m",
    Interval.in_1_hour: "1h",
    Interval.in_4_hour: "4h",
}

_lock = threading.Lock()
_tv: Optional[TvDatafeed] = None
_active_source: Optional[Tuple[str, str]] = None
_client_logged = False


def _close_client(client: TvDatafeed) -> None:
    try:
        if client.ws is not None:
            client.ws.close()
    except Exception:
        pass
    client.ws = None


def _create_client() -> TvDatafeed:
    global _client_logged
    use_login = TV_USE_LOGIN and TV_USERNAME and TV_PASSWORD
    if use_login:
        if not _client_logged:
            logger.info("TradingView data: using account login")
            _client_logged = True
        return TvDatafeed(username=TV_USERNAME, password=TV_PASSWORD)
    if not _client_logged:
        logger.debug("TradingView data: anonymous feed (no login)")
        _client_logged = True
    return TvDatafeed()


def _get_or_create_client(force_new: bool = False) -> TvDatafeed:
    global _tv
    with _lock:
        if force_new and _tv is not None:
            _close_client(_tv)
            _tv = None
        if _tv is None:
            _tv = _create_client()
        return _tv


def get_tv_client() -> TvDatafeed:
    return _get_or_create_client()


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _try_fetch(
    symbol: str,
    exchange: str,
    interval: Interval,
    bars: int,
    force_new_client: bool = False,
) -> Optional[pd.DataFrame]:
    client = _get_or_create_client(force_new=force_new_client)
    try:
        df = client.get_hist(
            symbol=symbol, exchange=exchange, interval=interval, n_bars=bars
        )
    except Exception:
        _close_client(client)
        with _lock:
            global _tv
            _tv = None
        raise
    finally:
        _close_client(client)

    if df is None or df.empty:
        return None
    return _normalize_df(df)


def fetch_tv(interval: Interval, bars: int) -> pd.DataFrame:
    """Fetch OHLCV with retries, client reset on failure, and exchange fallbacks."""
    global _active_source

    sources = list(FETCH_SOURCES)
    if _active_source and _active_source in sources:
        sources.remove(_active_source)
        sources.insert(0, _active_source)

    last_error: Optional[Exception] = None
    force_new = False

    for attempt in range(1, FETCH_RETRIES + 1):
        for symbol, exchange in sources:
            try:
                logger.debug(
                    "Fetching %s:%s interval=%s bars=%s (attempt %s)",
                    exchange,
                    symbol,
                    interval,
                    bars,
                    attempt,
                )
                df = _try_fetch(symbol, exchange, interval, bars, force_new_client=force_new)
                force_new = False
                if df is not None and not df.empty:
                    _active_source = (symbol, exchange)
                    logger.info(
                        "Data OK: %s:%s (%s bars, attempt %s)",
                        exchange,
                        symbol,
                        len(df),
                        attempt,
                    )
                    return df
            except Exception as exc:
                last_error = exc
                force_new = True
                logger.debug(
                    "Fetch failed %s:%s attempt %s: %s",
                    exchange,
                    symbol,
                    attempt,
                    exc,
                )

            time.sleep(FETCH_DELAY_SECONDS)

        if attempt < FETCH_RETRIES:
            wait = FETCH_DELAY_SECONDS * attempt
            logger.debug("Retrying all sources in %.1fs...", wait)
            time.sleep(wait)

    msg = "TradingView feed returned no data for any source"
    if last_error:
        msg = f"{msg}: {last_error}"

    # Last-resort: OANDA REST candles (only if OANDA_API_TOKEN is set)
    try:
        from data.oanda import oanda_client

        if oanda_client.configured:
            key = _INTERVAL_TO_KEY.get(interval, "5m")
            logger.info("Falling back to OANDA candles for %s (legacy fallback)", key)
            return oanda_client.fetch_candles(key, min(bars, 5000))
    except Exception as exc:
        logger.debug("OANDA candle fallback failed: %s", exc)

    raise ValueError(msg)


def fetch_tv_with_pause(interval: Interval, bars: int) -> pd.DataFrame:
    """Fetch one timeframe then pause to avoid websocket rate limits."""
    df = fetch_tv(interval, bars)
    time.sleep(FETCH_BETWEEN_TF_DELAY)
    return df


_tick_cache: dict = {}


def get_tick_quote() -> dict:
    """Lightweight 1M quote for sub-second SL/TP monitoring (cached)."""
    from config import PRICE_FETCH_SECONDS

    now = time.time()
    if _tick_cache and now - _tick_cache.get("_ts", 0) < PRICE_FETCH_SECONDS:
        return _tick_cache

    try:
        df = fetch_tv(Interval.in_1_minute, 5)
        row = df.iloc[-1]
        _tick_cache.clear()
        _tick_cache.update(
            {
                "_ts": now,
                "price": round(float(row["close"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "time": str(df.index[-1]),
            }
        )
    except Exception as exc:
        logger.debug("Tick quote fetch failed, using cache: %s", exc)
        if not _tick_cache:
            return {}

    return _tick_cache
