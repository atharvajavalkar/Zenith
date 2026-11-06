"""
Mode-aware market data facade.

  trade_mode=demo|live → XM via MetaAPI (GOLD candles + ticks)
  trade_mode=paper     → TradingView (XM preferred, OANDA/FX fallback)

All signal, bias, tick, and chart candle fetches should go through here
so the dashboard stays consistent with the selected mode.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional, Tuple, Union

import pandas as pd
from tvDatafeed import Interval

logger = logging.getLogger(__name__)

_INTERVAL_TO_TF = {
    Interval.in_1_minute: "1m",
    Interval.in_3_minute: "3m",
    Interval.in_5_minute: "5m",
    Interval.in_15_minute: "15m",
    Interval.in_1_hour: "1h",
    Interval.in_4_hour: "4h",
}

_lock = threading.Lock()
_last_source: str = "tradingview"
_last_symbol: str = "XAUUSD"
_tick_cache: dict = {}


def get_trade_mode() -> str:
    try:
        import runtime_config

        return runtime_config.get_trade_mode()
    except Exception:
        from config import TRADE_MODE, VALID_TRADE_MODES

        m = str(TRADE_MODE).lower().strip()
        return m if m in VALID_TRADE_MODES else "paper"


def is_broker_mode(mode: Optional[str] = None) -> bool:
    try:
        import runtime_config

        return runtime_config.is_broker_mode(mode)
    except Exception:
        from config import is_broker_trade_mode

        return is_broker_trade_mode(mode)


def get_data_source() -> str:
    """
    Canonical feed for the current mode.

    Returns: "xm" | "tradingview"
    """
    if not is_broker_mode():
        return "tradingview"
    try:
        from data.broker import broker_client, broker_name

        if broker_name == "xm" and getattr(broker_client, "configured", False):
            return "xm"
    except Exception:
        pass
    return "tradingview"


def get_display_symbol() -> str:
    if get_data_source() == "xm":
        from config import MT5_SYMBOL

        return MT5_SYMBOL or "GOLD"
    return "XAUUSD"


def get_source_info() -> dict[str, str]:
    source = get_data_source()
    mode = get_trade_mode()
    symbol = get_display_symbol()
    if source == "xm":
        label = f"XM {symbol} ({mode} · MetaAPI)"
    else:
        label = "TradingView (XAUUSD)"
    return {
        "trade_mode": mode,
        "data_source": source,
        "symbol": symbol,
        "label": label,
    }


def last_fetch_source() -> str:
    with _lock:
        return _last_source


def on_mode_change(mode: str) -> None:
    """Clear tick/source caches when user switches paper ↔ demo ↔ live."""
    global _tick_cache, _last_source
    with _lock:
        _tick_cache = {}
        _last_source = "xm" if mode in ("demo", "live") else "tradingview"
    logger.info("Market data mode → %s (caches cleared)", mode)


def _interval_key(interval: Union[Interval, str]) -> Tuple[Any, str]:
    if isinstance(interval, str):
        key = interval.lower().strip()
        # map string back to Interval when possible for TV path
        rev = {v: k for k, v in _INTERVAL_TO_TF.items()}
        return rev.get(key, Interval.in_5_minute), key
    tf = _INTERVAL_TO_TF.get(interval)
    if not tf:
        raise ValueError(f"Unsupported interval: {interval}")
    return interval, tf


def fetch_ohlcv(
    interval: Union[Interval, str],
    bars: int,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV for signals / bias / charts.

    demo/live + XM → MetaAPI candles for `symbol` (default MT5_SYMBOL).
    Paper (or XM unavailable) → TradingView.
    """
    global _last_source, _last_symbol
    tv_interval, tf = _interval_key(interval)
    source = get_data_source()
    from config import MT5_SYMBOL

    sym = (symbol or MT5_SYMBOL or "GOLD").strip().upper()

    if source == "xm":
        try:
            from data.mt5_client import mt5_client

            df = mt5_client.fetch_candles(tf, bars, symbol=sym)
            with _lock:
                _last_source = "xm"
                _last_symbol = sym
            return df
        except Exception as exc:
            logger.warning(
                "XM MetaAPI %s %s candles failed (%s) — falling back to TradingView",
                sym,
                tf,
                exc,
            )

    from data.fetcher import fetch_tv_symbol

    df = fetch_tv_symbol(sym, tv_interval, bars)
    with _lock:
        _last_source = "tradingview"
        _last_symbol = sym
    return df


def fetch_ohlcv_with_pause(
    interval: Union[Interval, str],
    bars: int,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    from config import FETCH_BETWEEN_TF_DELAY

    df = fetch_ohlcv(interval, bars, symbol=symbol)
    delay = 0.2 if get_data_source() == "xm" else FETCH_BETWEEN_TF_DELAY
    time.sleep(delay)
    return df


def get_tick_quote(symbol: Optional[str] = None) -> dict:
    """
    Live mark price for dashboard / SL monitoring / multi-instrument fills.
    """
    from config import MT5_SYMBOL, PRICE_FETCH_SECONDS

    sym = (symbol or MT5_SYMBOL or "GOLD").strip().upper()
    now = time.time()
    ttl = max(PRICE_FETCH_SECONDS, 2.5) if get_data_source() == "xm" else PRICE_FETCH_SECONDS
    cache_key = f"tick:{sym}"
    with _lock:
        hit = _tick_cache.get(cache_key)
        if hit and now - hit.get("_ts", 0) < ttl:
            return dict(hit)

    source = get_data_source()

    if source == "xm":
        try:
            from data.mt5_client import mt5_client

            if mt5_client._in_backoff():  # noqa: SLF001
                with _lock:
                    hit = _tick_cache.get(cache_key)
                    if hit:
                        return dict(hit)

            quote = mt5_client.current_price(symbol=sym)
            if quote and quote.get("price") is not None:
                price = float(quote["price"])
                from strategy.instrument_meta import price_digits

                decimals = price_digits(sym)
                price = round(price, decimals)
                payload = {
                    "_ts": now,
                    "price": price,
                    "high": price,
                    "low": price,
                    "bid": quote.get("bid"),
                    "ask": quote.get("ask"),
                    "time": quote.get("time"),
                    "source": "xm",
                    "symbol": quote.get("symbol") or sym,
                    "stale": bool(quote.get("stale")),
                }
                with _lock:
                    _tick_cache[cache_key] = payload
                return dict(payload)
            logger.warning("MetaAPI current_price empty — no XM tick for %s", sym)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "backoff" in msg.lower():
                logger.warning("XM tick rate-limited: %s", exc)
            else:
                logger.warning("XM tick failed (%s): %s", sym, exc)
            with _lock:
                hit = _tick_cache.get(cache_key)
                if hit:
                    return dict(hit)
            return {}

    try:
        from data.fetcher import fetch_tv_symbol

        df = fetch_tv_symbol(sym, Interval.in_1_minute, 5)
        row = df.iloc[-1]
        from strategy.instrument_meta import price_digits

        decimals = price_digits(sym)
        payload = {
            "_ts": now,
            "price": round(float(row["close"]), decimals),
            "high": round(float(row["high"]), decimals),
            "low": round(float(row["low"]), decimals),
            "time": str(df.index[-1]),
            "source": "tradingview",
            "symbol": sym,
        }
        with _lock:
            _tick_cache[cache_key] = payload
        return dict(payload)
    except Exception as exc:
        logger.debug("TV tick failed (%s): %s", sym, exc)
        with _lock:
            hit = _tick_cache.get(cache_key)
            if hit:
                return dict(hit)
        return {}
