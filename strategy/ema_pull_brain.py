"""EMA20/50 pullback + displacement entry brain (research winner).

Research scan (14d/1m on composite book) beat the prior Zenith setup on
risk-normalized PnL and $/trade among high-volume alts. Live wiring only —
does not alter Hyper.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def evaluate_ema_2050_pull(
    df: pd.DataFrame,
    *,
    pull_atr: float = 1.0,
    disp_atr: float = 0.55,
    swing: int = 10,
    sl_pad_atr: float = 0.10,
) -> Optional[dict[str, Any]]:
    """Return BUY/SELL pack on the last bar, or None."""
    if df is None or len(df) < 55 + swing:
        return None

    row = df.iloc[-1]
    need = ("close", "open", "high", "low", "ATR")
    if any(c not in df.columns for c in need):
        return None

    # Prefer prepared EMA20/50; fall back to compute
    if "EMA20" in df.columns and "EMA50" in df.columns:
        ema_f = float(row["EMA20"])
        ema_s = float(row["EMA50"])
    else:
        close = df["close"]
        ema_f = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema_s = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

    atr = float(row["ATR"])
    if not atr or atr != atr:
        return None

    c = float(row["close"])
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    mid = 0.5 * (ema_f + ema_s)
    body = abs(c - o)
    rng = max(h - l, 1e-12)
    bull = ema_f > ema_s
    bear = ema_f < ema_s
    disp_buy = c > o and body >= disp_atr * atr and (c - l) / rng >= 0.75
    disp_sell = c < o and body >= disp_atr * atr and (h - c) / rng >= 0.75
    near_mid = abs(c - mid) <= pull_atr * atr

    tail = df.iloc[-(swing + 1) :]
    if bull and disp_buy and near_mid and c > ema_f:
        sl = float(tail["low"].min()) - sl_pad_atr * atr
        if sl >= c:
            return None
        return {
            "signal": "BUY",
            "entry": c,
            "sl": sl,
            "ema_fast": ema_f,
            "ema_slow": ema_s,
            "mid": mid,
            "atr": atr,
        }
    if bear and disp_sell and near_mid and c < ema_f:
        sl = float(tail["high"].max()) + sl_pad_atr * atr
        if sl <= c:
            return None
        return {
            "signal": "SELL",
            "entry": c,
            "sl": sl,
            "ema_fast": ema_f,
            "ema_slow": ema_s,
            "mid": mid,
            "atr": atr,
        }
    return None
