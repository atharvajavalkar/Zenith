"""
Historical XAUUSD OHLCV data.

Primary source  : local Dukascopy cache under data/hist/ (multi-year M1/M5)
Then           : TradingView via tvDatafeed (same feed as the live dashboard)
Fallback       : yfinance (5m ~60 days only)

Live MetaAPI candles are NOT routed through this module.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─── constants ────────────────────────────────────────────────────────────────
BARS_5M_PER_DAY = 300          # conservative upper bound (24 h × 12  + buffer)
TV_MAX_BARS     = 200_000      # hard cap on tvDatafeed request to avoid hangs


# ─── helpers ──────────────────────────────────────────────────────────────────

def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, keep OHLCV, ensure UTC DatetimeIndex."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    for drop_col in ("dividends", "stock splits", "capital gains", "symbol"):
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


# ─── TradingView / XM source ──────────────────────────────────────────────────

def _fetch_tv_5m(start: str, end: str) -> pd.DataFrame:
    """
    Fetch XAUUSD 5m bars from TradingView (XAUUSD/XM — same as live feed).

    tvDatafeed returns the LAST n_bars bars ending *now*.  We calculate how
    many bars we need to go back far enough to cover [start, end] and then
    filter the result.
    """
    from tvDatafeed import TvDatafeed, Interval
    from config import FETCH_SOURCES

    now       = datetime.now(timezone.utc)
    start_dt  = _to_utc(datetime.fromisoformat(start))
    end_dt    = _to_utc(datetime.fromisoformat(end))

    # How many bars back do we need?  Extra buffer for indicator warmup before `start`.
    days_back = max((now.date() - start_dt.date()).days + 5, 1)
    n_bars    = min(int(days_back * BARS_5M_PER_DAY * 1.5) + 2_000, TV_MAX_BARS)

    logger.info(
        "_fetch_tv_5m: requesting %d bars to cover %s → %s (%d days back)",
        n_bars, start, end, days_back,
    )

    # Prefer XM first (same feed as live trading), then other TV exchanges.
    sources_ordered = []
    known = set()
    for sym, exc in list(FETCH_SOURCES):
        pair = (sym, exc)
        if pair not in known:
            sources_ordered.append(pair)
            known.add(pair)
    for pair in (
        ("XAUUSD", "OANDA"),
        ("XAUUSD", "FX"),
        ("GOLD", "TVC"),
        ("XAUUSD", "FOREXCOM"),
        ("XAUUSD", "ICMARKETS"),
    ):
        if pair not in known:
            sources_ordered.append(pair)
            known.add(pair)

    last_err: Optional[Exception] = None
    for sym, exchange in sources_ordered:
        try:
            tv = TvDatafeed()
            raw = tv.get_hist(
                symbol=sym,
                exchange=exchange,
                interval=Interval.in_5_minute,
                n_bars=n_bars,
            )
            if raw is None or raw.empty:
                logger.debug("tvDatafeed %s:%s returned empty", exchange, sym)
                continue

            df = _normalise(raw)
            if df.empty:
                continue

            oldest = df.index[0]
            newest = df.index[-1]
            logger.info(
                "tvDatafeed %s:%s → %d bars  %s → %s",
                exchange, sym, len(df),
                str(oldest)[:16], str(newest)[:16],
            )

            # Filter to [start_dt, end_dt] — keep extra warmup bars before start
            mask = df.index <= end_dt
            df_out = df[mask]
            if df_out.empty:
                logger.warning("tvDatafeed: all bars are after requested end %s", end)
                continue

            # If oldest bar is still after our start, we didn't go back far enough
            if oldest > start_dt:
                logger.warning(
                    "tvDatafeed: oldest bar %s is after start %s — limited history",
                    str(oldest)[:16], start,
                )
                # Return what we have — caller will surface a useful error if range is empty
            return df_out

        except Exception as exc:
            last_err = exc
            logger.warning("tvDatafeed %s:%s 5m failed: %s", exchange, sym, exc)
            time.sleep(1)

    raise RuntimeError(
        f"TradingView 5m fetch failed for all sources. Last error: {last_err}"
    )


# ─── yfinance fallback ────────────────────────────────────────────────────────

def _fetch_yf_5m(start: str, end: str) -> pd.DataFrame:
    """yfinance 5m fetch — works for the last ~60 days only."""
    import yfinance as yf

    for sym in ("GC=F", "XAUUSD=X"):
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(start=start, end=end, interval="5m", auto_adjust=True)
            if df is not None and not df.empty:
                return _normalise(df)
        except Exception as exc:
            logger.warning("yfinance %s 5m failed: %s", sym, exc)

    raise RuntimeError(f"yfinance returned no 5m data for {start} → {end}")


# ─── public API ───────────────────────────────────────────────────────────────

def _try_local(symbol: str, start: str, end: str, tf: str = "5m") -> Optional[pd.DataFrame]:
    """Prefer Dukascopy cache. Soft coverage gate — still return partial local if present."""
    try:
        from data.hist_local import coverage_ok, has_local_hist, load_local_ohlcv

        if not has_local_hist(symbol, tf):
            return None
        df = load_local_ohlcv(symbol, start=start, end=end, tf=tf)
        if df is None or df.empty:
            return None
        # Accept partial overlap (hist may end a few days before "today")
        if not coverage_ok(symbol, start, end, tf=tf, min_coverage=0.20):
            logger.info(
                "local hist %s %s: partial coverage — using %d bars anyway",
                symbol,
                tf,
                len(df),
            )
        return _normalise(df)
    except Exception as exc:
        logger.debug("local hist %s %s skip: %s", symbol, tf, exc)
    return None


def fetch_xauusd(
    start: str,
    end: str,
    interval: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch XAUUSD OHLCV for [start, end].

    Always fetches 5m bars (to match the live 1m/3m/5m dashboard).
    Strategy:
      1. Local Dukascopy cache (data/hist/) when coverage is sufficient
      2. TradingView / XM
      3. yfinance GC=F / XAUUSD=X  (fallback — last 60 days only)

    Args:
        start:    ISO date string  e.g. "2026-01-01"
        end:      ISO date string  e.g. "2026-07-01"
        interval: Ignored — always uses 5m.  Kept for back-compat.
    """
    errors = []

    local = _try_local("GOLD", start, end, tf="5m")
    if local is not None:
        logger.info(
            "fetch_xauusd: local hist OK — %d bars  %s → %s",
            len(local), str(local.index[0])[:16], str(local.index[-1])[:16],
        )
        return local

    # ── 1. Try TradingView (XM) ───────────────────────────────────────────────
    try:
        df = _fetch_tv_5m(start, end)
        if df is not None and not df.empty:
            logger.info(
                "fetch_xauusd: TV source OK — %d bars  %s → %s",
                len(df), str(df.index[0])[:16], str(df.index[-1])[:16],
            )
            return df
    except Exception as exc:
        errors.append(f"TradingView: {exc}")
        logger.warning("fetch_xauusd: TV source failed: %s", exc)

    # ── 2. Try yfinance (last 60 days) ────────────────────────────────────────
    try:
        df = _fetch_yf_5m(start, end)
        if df is not None and not df.empty:
            logger.info(
                "fetch_xauusd: yfinance fallback OK — %d bars  %s → %s",
                len(df), str(df.index[0])[:16], str(df.index[-1])[:16],
            )
            return df
    except Exception as exc:
        errors.append(f"yfinance: {exc}")
        logger.warning("fetch_xauusd: yfinance fallback failed: %s", exc)

    raise RuntimeError(
        f"No 5m data available for {start} → {end}. "
        f"Run: python -m data.dukascopy_ingest --years 10 "
        f"or select a shorter TV/yfinance window. "
        f"Details: {' | '.join(errors)}"
    )


# Multi-instrument history (FX / crypto / metals) ───────────────────────────────

_YF_SYMBOLS: dict[str, tuple[str, ...]] = {
    "GOLD": ("GC=F", "XAUUSD=X"),
    "XAUUSD": ("GC=F", "XAUUSD=X"),
    "SILVER": ("SI=F", "XAGUSD=X"),
    "XAGUSD": ("SI=F", "XAGUSD=X"),
    "EURUSD": ("EURUSD=X",),
    "GBPUSD": ("GBPUSD=X",),
    "USDJPY": ("USDJPY=X",),
    "USDCHF": ("USDCHF=X",),
    "AUDUSD": ("AUDUSD=X",),
    "EURJPY": ("EURJPY=X",),
    "BTCUSD": ("BTC-USD", "BTCUSD=X"),
    "ETHUSD": ("ETH-USD",),
    "USOIL": ("CL=F",),
    "XTIUSD": ("CL=F",),
    "UKOIL": ("BZ=F",),
    "US30": ("YM=F", "^DJI"),
    "US500": ("ES=F", "^GSPC"),
    "NAS100": ("NQ=F", "^NDX"),
}


def _fetch_yf_symbol_5m(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    sym = str(symbol).upper().strip()
    yf_list = _YF_SYMBOLS.get(sym, (f"{sym}=X", sym))
    last_err: Optional[Exception] = None
    for ysym in yf_list:
        try:
            ticker = yf.Ticker(ysym)
            df = ticker.history(start=start, end=end, interval="5m", auto_adjust=True)
            if df is not None and not df.empty:
                return _normalise(df)
        except Exception as exc:
            last_err = exc
            logger.warning("yfinance %s 5m failed: %s", ysym, exc)
    raise RuntimeError(f"yfinance no 5m for {sym}: {last_err}")


def _fetch_tv_symbol_5m(symbol: str, start: str, end: str) -> pd.DataFrame:
    """TradingView history for any desk symbol (XM preferred)."""
    from tvDatafeed import TvDatafeed, Interval
    from data.fetcher import fetch_tv_symbol

    # tvDatafeed last-N-bars path via existing live helper, then date-filter
    now = datetime.now(timezone.utc)
    start_dt = _to_utc(datetime.fromisoformat(start))
    end_dt = _to_utc(datetime.fromisoformat(end))
    days_back = max((now.date() - start_dt.date()).days + 5, 1)
    n_bars = min(int(days_back * BARS_5M_PER_DAY * 1.5) + 2_000, TV_MAX_BARS)

    df = fetch_tv_symbol(symbol, Interval.in_5_minute, n_bars)
    if df is None or df.empty:
        raise RuntimeError(f"TV empty for {symbol}")
    df = _normalise(df)
    mask = (df.index >= start_dt) & (df.index <= end_dt + timedelta(days=1))
    out = df.loc[mask]
    if out.empty:
        # Keep warmup bars before start as well for backtests that filter later
        out = df[df.index <= end_dt + timedelta(days=1)]
    if out.empty:
        raise RuntimeError(f"TV {symbol} had no bars in range")
    logger.info(
        "TV %s 5m: %d bars requested → %d in window %s→%s",
        symbol, n_bars, len(out), start, end,
    )
    return out


def fetch_symbol_ohlcv(
    symbol: str,
    start: str,
    end: str,
    interval: Optional[str] = None,
) -> pd.DataFrame:
    """
    Multi-instrument OHLCV for backtests.
    Prefers local Dukascopy hist for all desk pairs (incl. GOLD).
    Remote TV/yfinance only as short-window fallback.
    """
    sym = str(symbol or "GOLD").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"

    local = _try_local(sym, start, end, tf="5m")
    if local is not None:
        logger.info("fetch_symbol_ohlcv local %s OK — %d bars", sym, len(local))
        return local

    # GOLD keeps the legacy remote chain when local missing
    if sym == "GOLD":
        return fetch_xauusd(start, end, interval=interval)

    errors: list[str] = []
    try:
        from data.hist_local import has_local_hist, local_span

        if not has_local_hist(sym, "5m"):
            span = local_span(sym, "5m")
            raise RuntimeError(
                f"No Dukascopy cache for {sym}. "
                f"Desk backtest hist covers GOLD/USDCHF/USDJPY/NAS100/UKOIL/AUDUSD/BTCUSD/ETHUSD. "
                f"Run: python -m data.dukascopy_ingest --symbols {sym} --years 10"
                + (f" (span={span})" if span else "")
            )
    except RuntimeError:
        raise
    except Exception:
        pass

    try:
        df = _fetch_tv_symbol_5m(sym, start, end)
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        errors.append(f"TradingView: {exc}")
        logger.warning("fetch_symbol_ohlcv TV %s failed: %s", sym, exc)

    try:
        df = _fetch_yf_symbol_5m(sym, start, end)
        if df is not None and not df.empty:
            logger.info("fetch_symbol_ohlcv yfinance %s OK — %d bars", sym, len(df))
            return df
    except Exception as exc:
        errors.append(f"yfinance: {exc}")
        logger.warning("fetch_symbol_ohlcv yf %s failed: %s", sym, exc)

    raise RuntimeError(
        f"No 5m data for {sym} ({start} → {end}). "
        f"Local hist missing/empty — run: python -m data.dukascopy_ingest --symbols {sym} --years 10. "
        f"Details: {' | '.join(errors)}"
    )


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Downsample OHLCV DataFrame to a coarser frequency."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna()
