"""SMA20/50 cross entry brain (NAS100 research upgrade).

Cross of SMA20 through SMA50 + ATR stop · forced 2R downstream.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def evaluate_sma_2050_cross(
    df: pd.DataFrame,
    *,
    sl_atr: float = 1.0,
) -> Optional[dict[str, Any]]:
    if df is None or len(df) < 55:
        return None
    need = ("close", "open", "high", "low", "ATR")
    if any(c not in df.columns for c in need):
        return None

    if "SMA20" in df.columns and "SMA50" in df.columns:
        f0 = float(df["SMA20"].iloc[-2])
        s0 = float(df["SMA50"].iloc[-2])
        f1 = float(df["SMA20"].iloc[-1])
        s1 = float(df["SMA50"].iloc[-1])
    else:
        sma20 = df["close"].rolling(20).mean()
        sma50 = df["close"].rolling(50).mean()
        f0, s0 = float(sma20.iloc[-2]), float(sma50.iloc[-2])
        f1, s1 = float(sma20.iloc[-1]), float(sma50.iloc[-1])

    atr = float(df.iloc[-1]["ATR"])
    if not atr or atr != atr:
        return None
    c = float(df.iloc[-1]["close"])

    if f0 <= s0 and f1 > s1:
        sl = c - sl_atr * atr
        if sl >= c:
            return None
        return {"signal": "BUY", "entry": c, "sl": sl, "atr": atr, "sma_fast": f1, "sma_slow": s1}
    if f0 >= s0 and f1 < s1:
        sl = c + sl_atr * atr
        if sl <= c:
            return None
        return {"signal": "SELL", "entry": c, "sl": sl, "atr": atr, "sma_fast": f1, "sma_slow": s1}
    return None
