"""Hyper-only micro fixed-range volume profile.

Uses the latest 30 completed/live 10-second candles (about five minutes),
24 price bins, and a 70% value area. This is intentionally separate from
Zenith's previous-session FRVP in analysis/volume_profile.py.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import pandas as pd

_LOCK = threading.Lock()
_last_profiles: dict[str, dict[str, Any]] = {}

DEFAULT_BARS = 30
MIN_BARS = 18
N_BINS = 24
VALUE_AREA_PCT = 0.70
POC_SKIP_ATR = 0.12


def build_micro_profile(
    df: pd.DataFrame,
    *,
    symbol: str,
    price: float,
    atr: float,
    bars: int = DEFAULT_BARS,
    timeframe: str = "10s",
) -> dict[str, Any]:
    """Build a rolling micro-FRVP from Hyper's seconds candles."""
    if df is None or len(df) < MIN_BARS:
        return {
            "available": False,
            "symbol": str(symbol).upper(),
            "bars_used": 0 if df is None else len(df),
            "summary": f"Micro FRVP warming ({0 if df is None else len(df)}/{MIN_BARS})",
        }

    from analysis.volume_profile import _profile_from_window

    window = df.tail(min(int(bars), len(df))).copy()
    profile = _profile_from_window(
        window,
        price=float(price),
        atr=float(atr),
        n_bins=N_BINS,
        value_area_pct=VALUE_AREA_PCT,
        symbol=symbol,
        source="hyper_micro_frvp",
        session_name="5m micro",
    )
    profile["symbol"] = str(symbol).upper()
    tf = str(timeframe).lower().strip()
    if tf.endswith("s") and tf[:-1].isdigit():
        bar_seconds = int(tf[:-1])
    elif tf.endswith("m") and tf[:-1].isdigit():
        bar_seconds = int(tf[:-1]) * 60
    else:
        bar_seconds = 10
    profile["timeframe"] = tf
    profile["range_bars"] = int(len(window))
    profile["range_seconds"] = int(len(window) * bar_seconds)
    profile["n_bins"] = N_BINS
    profile["value_area_pct"] = VALUE_AREA_PCT
    profile["updated_at"] = time.time()

    with _LOCK:
        _last_profiles[str(symbol).upper()] = dict(profile)
    return profile


def refresh_for_symbol(
    symbol: str,
    price: float,
    *,
    timeframe: str = "10s",
    bars: int = DEFAULT_BARS,
) -> dict[str, Any]:
    """Rebuild micro-FRVP from the live seconds-candle cache."""
    from indicators.technical import atr as _atr
    from strategy.hyper_ema import _fetch_frame, _tf_seconds

    tf = str(timeframe or "10s")
    # Fallback ATR when candles are still thin
    atr_fallback = max(abs(float(price)) * 0.0004, 1e-6)
    try:
        df = _fetch_frame(symbol, tf, max(bars, 40))
    except Exception:
        return {
            "available": False,
            "symbol": str(symbol).upper(),
            "summary": "Micro FRVP fetch failed",
        }
    if df is None or len(df) < MIN_BARS:
        return {
            "available": False,
            "symbol": str(symbol).upper(),
            "bars_used": 0 if df is None else len(df),
            "summary": f"Micro FRVP warming ({0 if df is None else len(df)}/{MIN_BARS})",
        }
    try:
        atr_series = _atr(df, min(14, max(5, len(df) // 2)))
        a = float(atr_series.iloc[-1])
        if not (a == a) or a <= 0:
            a = atr_fallback
    except Exception:
        a = atr_fallback
    return build_micro_profile(
        df,
        symbol=symbol,
        price=float(price),
        atr=a,
        bars=bars,
        timeframe=tf,
    )


def allows_signal(profile: dict[str, Any], signal: str) -> tuple[bool, str]:
    """Apply conservative POC/value-area alignment to an EMA signal."""
    if not profile.get("available"):
        return False, str(profile.get("summary") or "Micro FRVP warming")

    sig = str(signal).upper()
    dist = float(profile.get("dist_to_poc_atr") or 0.0)
    location = str(profile.get("location") or "UNKNOWN")

    # POC is a balance/magnet zone; avoid rapid chop around it.
    if profile.get("at_poc") or abs(dist) < POC_SKIP_ATR:
        return False, "At micro POC — balance/chop"

    if sig == "BUY":
        if dist <= 0:
            return False, "BUY blocked below micro POC"
        if location == "BELOW_VALUE":
            return False, "BUY blocked below micro VAL"
        return True, "BUY aligned above micro POC"

    if sig == "SELL":
        if dist >= 0:
            return False, "SELL blocked above micro POC"
        if location == "ABOVE_VALUE":
            return False, "SELL blocked above micro VAH"
        return True, "SELL aligned below micro POC"

    return False, "No BUY/SELL signal"


def snapshot() -> dict[str, Any]:
    with _LOCK:
        rows = [dict(v) for v in _last_profiles.values()]
    rows.sort(key=lambda row: str(row.get("symbol") or ""))
    return {"profiles": rows}
