"""XM session spread costs (price units) — measured MetaAPI candle.spread Jun–Aug 2026.

Sessions UTC: Asia 00–08 · London 08–13 · Overlap 13–16 · NY 16–21 · Off 21–24
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Union

import pandas as pd

# Avg spread in *price* (ask−bid). Source: analysis/xm_spreads_by_session.py
XM_SPREAD_PRICE: dict[str, dict[str, float]] = {
    "ETHUSD": {
        "Asia": 4.8219,
        "London": 4.8472,
        "Overlap": 4.8296,
        "NY": 4.8208,
        "Off": 4.8208,
        "ALL": 4.8278,
    },
    "BTCUSD": {
        "Asia": 49.589,
        "London": 49.722,
        "Overlap": 49.630,
        "NY": 49.583,
        "Off": 49.583,
        "ALL": 49.620,
    },
    "GOLD": {
        "Asia": 0.5180,
        "London": 0.4969,
        "Overlap": 0.4828,
        "NY": 0.4918,
        "Off": 0.5437,
        "ALL": 0.5055,
    },
    "NAS100": {
        "Asia": 2.6546,
        "London": 2.6528,
        "Overlap": 2.6556,
        "NY": 2.6573,
        "Off": 2.6556,
        "ALL": 2.6550,
    },
    "USDJPY": {
        "Asia": 0.02274,
        "London": 0.02232,
        "Overlap": 0.02201,
        "NY": 0.02266,
        "Off": 0.04615,
        "ALL": 0.02545,
    },
    "AUDUSD": {
        "Asia": 0.000230,
        "London": 0.000230,
        "Overlap": 0.000230,
        "NY": 0.000231,
        "Off": 0.000429,
        "ALL": 0.000255,
    },
    "USDCHF": {
        "Asia": 0.000240,
        "London": 0.000232,
        "Overlap": 0.000232,
        "NY": 0.000237,
        "Off": 0.000690,
        "ALL": 0.000292,
    },
    "UKOIL": {
        "Asia": 0.0500,
        "London": 0.0500,
        "Overlap": 0.0500,
        "NY": 0.0662,
        "Off": 0.1000,
        "ALL": 0.0563,
    },
}


def session_name(hour_utc: float) -> str:
    h = float(hour_utc)
    if 0 <= h < 8:
        return "Asia"
    if 8 <= h < 13:
        return "London"
    if 13 <= h < 16:
        return "Overlap"
    if 16 <= h < 21:
        return "NY"
    return "Off"


def spread_price(
    symbol: str,
    ts: Optional[Union[datetime, pd.Timestamp, float, int]] = None,
    *,
    hour_utc: Optional[float] = None,
) -> float:
    """Round-trip XM spread in price units for symbol at UTC time."""
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    table = XM_SPREAD_PRICE.get(sym)
    if not table:
        return 0.0
    if hour_utc is None and ts is not None:
        if isinstance(ts, (int, float)):
            hour_utc = float(ts)
        else:
            t = pd.Timestamp(ts)
            if t.tzinfo is not None:
                t = t.tz_convert("UTC")
            hour_utc = float(t.hour) + float(t.minute) / 60.0
    if hour_utc is None:
        return float(table["ALL"])
    return float(table.get(session_name(hour_utc), table["ALL"]))


def spread_r(symbol: str, risk_px: float, ts=None, *, hour_utc: Optional[float] = None) -> float:
    """Spread cost in R-multiples (full round-trip / risk distance)."""
    risk = float(risk_px or 0)
    if risk <= 0:
        return 0.0
    return float(spread_price(symbol, ts, hour_utc=hour_utc)) / risk


# Fine-tune v2 (XM-cost WF): min stop ≥ mult × session spread
# PASS/SOFT survivors: ETH 8 · GOLD 10 · NAS 8 · JPY 10 · UKOIL 1 (Off sit-out)
SPREAD_RISK_MULT: dict[str, float] = {
    "ETHUSD": 8.0,
    "BTCUSD": 8.0,
    "GOLD": 10.0,
    "NAS100": 8.0,
    "USDJPY": 10.0,
    "AUDUSD": 10.0,
    "USDCHF": 10.0,
    "UKOIL": 1.0,
}


def spread_risk_mult(symbol: Optional[str]) -> float:
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    return float(SPREAD_RISK_MULT.get(sym, 5.0))


def min_risk_with_spread(
    symbol: Optional[str],
    atr_risk: float,
    *,
    hour_utc: Optional[float] = None,
    floor_px: float = 0.0,
) -> float:
    """Live SL distance: max(ATR risk, pip floor, mult × XM session spread)."""
    sp = spread_price(str(symbol or ""), hour_utc=hour_utc)
    mult = spread_risk_mult(symbol)
    return max(float(atr_risk or 0), float(floor_px or 0), mult * sp)
