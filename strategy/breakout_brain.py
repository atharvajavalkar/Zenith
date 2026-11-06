"""Donchian-style breakout entry brain (UKOIL research upgrade).

Close breaks lookback high/low with candle confirmation · ATR stop · 2R.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def evaluate_breakout(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    sl_atr: float = 1.2,
) -> Optional[dict[str, Any]]:
    if df is None or len(df) < lookback + 5:
        return None
    need = ("close", "open", "high", "low")
    if any(c not in df.columns for c in need):
        return None

    row = df.iloc[-1]
    atr = None
    for key in ("ATR", "atr"):
        if key in df.columns:
            try:
                atr = float(row[key])
            except Exception:
                atr = None
            if atr and atr == atr and atr > 0:
                break
            atr = None
    if atr is None:
        return None
    c = float(row["close"])
    o = float(row["open"])
    window = df.iloc[-(lookback + 1) : -1]
    hh = float(window["high"].max())
    ll = float(window["low"].min())

    if c > hh and c > o:
        sl = c - sl_atr * atr
        if sl >= c:
            return None
        return {
            "signal": "BUY",
            "entry": c,
            "sl": sl,
            "atr": atr,
            "break_level": hh,
            "lookback": lookback,
        }
    if c < ll and c < o:
        sl = c + sl_atr * atr
        if sl <= c:
            return None
        return {
            "signal": "SELL",
            "entry": c,
            "sl": sl,
            "atr": atr,
            "break_level": ll,
            "lookback": lookback,
        }
    return None
