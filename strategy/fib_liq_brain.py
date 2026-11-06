"""GOLD-only Fib retracement + liquidity sweep entry brain.

Research winner (1m GC=F grid, $200/R):
  Impulse TF = 15m · Entry TF = 15m · Fib pocket 0.50–0.618 · 2R
  Sweep of zone/equal extreme → reclaim close → enter; SL beyond wick.

Wired only for GOLD / XAUUSD via GOLD_ENTRY_BRAIN=fib_liq_sweep.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from config import GOLD_FIB_LEVELS

SWING = 3
ZONE_ATR_MULT = 0.40
MIN_IMPULSE_ATR = 1.15
SL_PAD_ATR = 0.08
MIN_RISK_ATR = 0.12


def _is_swing_low(df: pd.DataFrame, i: int, n: int = SWING) -> bool:
    if i < n or i + n >= len(df):
        return False
    lo = float(df.iloc[i]["low"])
    return all(lo <= float(df.iloc[i + k]["low"]) for k in range(-n, n + 1) if k != 0)


def _is_swing_high(df: pd.DataFrame, i: int, n: int = SWING) -> bool:
    if i < n or i + n >= len(df):
        return False
    hi = float(df.iloc[i]["high"])
    return all(hi >= float(df.iloc[i + k]["high"]) for k in range(-n, n + 1) if k != 0)


def _atr(df: pd.DataFrame, i: int) -> float:
    row = df.iloc[i]
    if "ATR" in df.columns and row.get("ATR") == row.get("ATR") and float(row["ATR"] or 0) > 0:
        return float(row["ATR"])
    # fallback
    if i < 1:
        return max(float(row["high"]) - float(row["low"]), 1e-6)
    trs = []
    for j in range(max(1, i - 13), i + 1):
        h, l, pc = float(df.iloc[j]["high"]), float(df.iloc[j]["low"]), float(df.iloc[j - 1]["close"])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return max(sum(trs) / len(trs), 1e-6)


def _impulse_at(df: pd.DataFrame, i: int) -> Optional[dict[str, Any]]:
    atr = _atr(df, i)
    look = range(max(SWING, i - 60), i - SWING)
    lows = [(j, float(df.iloc[j]["low"])) for j in look if _is_swing_low(df, j)]
    highs = [(j, float(df.iloc[j]["high"])) for j in look if _is_swing_high(df, j)]
    if not lows or not highs:
        return None
    cands: list[dict[str, Any]] = []
    for lj, lv in reversed(lows):
        later = [(hj, hv) for hj, hv in highs if hj > lj]
        if not later:
            continue
        hj, hv = max(later, key=lambda x: x[1])
        if hv - lv >= MIN_IMPULSE_ATR * atr:
            cands.append({"side": "BUY", "a": lv, "b": hv, "bi": hj})
            break
    for hj, hv in reversed(highs):
        later = [(lj, lv) for lj, lv in lows if lj > hj]
        if not later:
            continue
        lj, lv = min(later, key=lambda x: x[1])
        if hv - lv >= MIN_IMPULSE_ATR * atr:
            cands.append({"side": "SELL", "a": hv, "b": lv, "bi": lj})
            break
    if not cands:
        return None
    return max(cands, key=lambda x: x["bi"])


def _fib_zone(imp: dict[str, Any], atr: float) -> tuple[float, float]:
    a, b = float(imp["a"]), float(imp["b"])
    levels = list(GOLD_FIB_LEVELS) or [0.50, 0.618]
    if imp["side"] == "BUY":
        fibs = [b - r * (b - a) for r in levels]
    else:
        fibs = [b + r * (a - b) for r in levels]
    pad = ZONE_ATR_MULT * atr
    return min(fibs) - pad, max(fibs) + pad


def evaluate_fib_liq_sweep(df_15m: pd.DataFrame) -> Optional[dict[str, Any]]:
    """Evaluate on last closed 15m bar. Returns BUY/SELL pack or None."""
    if df_15m is None or len(df_15m) < 80:
        return None
    need = ("open", "high", "low", "close")
    if any(c not in df_15m.columns for c in need):
        return None

    i = len(df_15m) - 1
    row = df_15m.iloc[i]
    h, l, c, o = (
        float(row["high"]),
        float(row["low"]),
        float(row["close"]),
        float(row["open"]),
    )
    atr = _atr(df_15m, i)
    imp = _impulse_at(df_15m, i)
    if not imp:
        return None

    zlo, zhi = _fib_zone(imp, atr)
    win = df_15m.iloc[max(0, i - 20) : i]
    if len(win) < 3:
        return None

    if imp["side"] == "BUY":
        liq = float(win["low"].min())
        swept = l < min(liq, zlo)
        reclaim = c > zlo and l <= zhi
        if not (swept and reclaim and c > o):
            return None
        entry = c
        sl = min(l, liq) - SL_PAD_ATR * atr
        risk = entry - sl
        if risk <= MIN_RISK_ATR * atr:
            return None
        return {
            "signal": "BUY",
            "entry": entry,
            "sl": sl,
            "atr": atr,
            "zone_lo": zlo,
            "zone_hi": zhi,
            "impulse": "bull",
            "fibs": list(GOLD_FIB_LEVELS),
        }

    liq = float(win["high"].max())
    swept = h > max(liq, zhi)
    reclaim = c < zhi and h >= zlo
    if not (swept and reclaim and c < o):
        return None
    entry = c
    sl = max(h, liq) + SL_PAD_ATR * atr
    risk = sl - entry
    if risk <= MIN_RISK_ATR * atr:
        return None
    return {
        "signal": "SELL",
        "entry": entry,
        "sl": sl,
        "atr": atr,
        "zone_lo": zlo,
        "zone_hi": zhi,
        "impulse": "bear",
        "fibs": list(GOLD_FIB_LEVELS),
    }
