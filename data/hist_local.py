"""
Local multi-year OHLCV cache for Strategy Lab / backtests.

Layout:
  data/hist/{SYMBOL}_1m.csv   — canonical Dukascopy M1 (UTC)
  data/hist/{SYMBOL}_{tf}.csv — optional pre-resampled (3m/5m/15m)

Live MetaAPI/TV path is untouched; labs prefer this when present.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / "data" / "hist"

# Desk symbol → Dukascopy datafeed instrument id
DUKA_INSTRUMENTS: dict[str, str] = {
    "GOLD": "XAUUSD",
    "XAUUSD": "XAUUSD",
    "USDCHF": "USDCHF",
    "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD",
    "NAS100": "USATECHIDXUSD",
    "UKOIL": "BRENTCMDUSD",
    "BTCUSD": "BTCUSD",
    "ETHUSD": "ETHUSD",
}

# Integer price → float divisor (native BID_candles_min_1.bi5)
DUKA_POINT_DIV: dict[str, float] = {
    "XAUUSD": 1000.0,
    "USDCHF": 100_000.0,
    "USDJPY": 1000.0,
    "AUDUSD": 100_000.0,
    "USATECHIDXUSD": 1000.0,
    "BRENTCMDUSD": 1000.0,
    "BTCUSD": 10.0,
    "ETHUSD": 10.0,
}

_TF_RULE = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
}

_cache: dict[str, pd.DataFrame] = {}


def hist_path(symbol: str, tf: str = "1m") -> Path:
    sym = str(symbol or "GOLD").upper().strip()
    if sym == "XAUUSD":
        sym = "GOLD"
    return HIST_DIR / f"{sym}_{tf}.csv"


def _parse_bound(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"hist CSV missing columns: {missing}")
    df = df[need].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def _read_csv(path: Path) -> pd.DataFrame:
    key = str(path.resolve())
    if key in _cache:
        return _cache[key]
    df = pd.read_csv(path, parse_dates=["time"])
    if "time" in df.columns:
        df = df.set_index("time")
    df = _normalise_frame(df)
    _cache[key] = df
    return df


def clear_hist_cache() -> None:
    _cache.clear()


def local_span(symbol: str, tf: str = "1m") -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    path = hist_path(symbol, tf)
    if not path.exists():
        # Fall back to 1m span
        if tf != "1m":
            return local_span(symbol, "1m")
        return None
    df = _read_csv(path)
    if df.empty:
        return None
    return df.index[0], df.index[-1]


def has_local_hist(symbol: str, tf: str = "5m") -> bool:
    if hist_path(symbol, tf).exists():
        return True
    if tf != "1m" and hist_path(symbol, "1m").exists():
        return True
    return False


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = _TF_RULE.get(tf)
    if rule is None:
        raise ValueError(f"unsupported tf {tf}")
    if tf == "1m":
        return df
    out = pd.DataFrame(
        {
            "open": df["open"].resample(rule).first(),
            "high": df["high"].resample(rule).max(),
            "low": df["low"].resample(rule).min(),
            "close": df["close"].resample(rule).last(),
            "volume": df["volume"].resample(rule).sum(),
        }
    ).dropna()
    return out


def load_local_ohlcv(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tf: str = "5m",
) -> Optional[pd.DataFrame]:
    """
    Load local hist for symbol/tf. Prefer exact CSV; else resample from 1m.
    Returns None if no usable cache.
    """
    tf = str(tf or "5m").lower().strip()
    exact = hist_path(symbol, tf)
    base_1m = hist_path(symbol, "1m")

    df: Optional[pd.DataFrame] = None
    try:
        if exact.exists():
            df = _read_csv(exact)
        elif tf != "1m" and base_1m.exists():
            df = _resample(_read_csv(base_1m), tf)
        else:
            return None
    except Exception as exc:
        logger.warning("load_local_ohlcv %s %s failed: %s", symbol, tf, exc)
        return None

    if df is None or df.empty:
        return None

    start_ts = _parse_bound(start)
    end_ts = _parse_bound(end)
    # Keep modest warmup before start for indicator labs
    if start_ts is not None:
        warm = start_ts - pd.Timedelta(days=5)
        df = df[df.index >= warm]
    if end_ts is not None:
        df = df[df.index <= end_ts + pd.Timedelta(days=1)]

    if df.empty:
        return None
    logger.info(
        "local hist %s %s: %d bars %s → %s",
        symbol,
        tf,
        len(df),
        str(df.index[0])[:16],
        str(df.index[-1])[:16],
    )
    return df


def save_ohlcv_csv(df: pd.DataFrame, symbol: str, tf: str) -> Path:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    path = hist_path(symbol, tf)
    out = _normalise_frame(df)
    dump = out.reset_index()
    time_col = dump.columns[0]
    dump = dump.rename(columns={time_col: "time"})
    dump.to_csv(path, index=False)
    clear_hist_cache()
    return path


def coverage_ok(
    symbol: str,
    start: str,
    end: str,
    tf: str = "5m",
    min_coverage: float = 0.5,
) -> bool:
    """True if local file covers most of [start, end]."""
    span = local_span(symbol, tf)
    if span is None:
        return False
    lo, hi = span
    start_ts = _parse_bound(start)
    end_ts = _parse_bound(end)
    if start_ts is None or end_ts is None:
        return True
    if hi < start_ts or lo > end_ts:
        return False
    overlap = (min(hi, end_ts) - max(lo, start_ts)).total_seconds()
    wanted = max((end_ts - start_ts).total_seconds(), 1.0)
    return (overlap / wanted) >= min_coverage
