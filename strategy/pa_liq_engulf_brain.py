"""PA: liquidity sweep → dual-TF EMA bias → 1m engulfing entry.

Per-symbol TFs via SYMBOL_PA_LIQ_TF / SYMBOL_PA_BIAS_TF:
  GOLD:    liq=3m · bias=15m+5m · RR=2
  USDCHF:  live trial liq=15m · bias=5m+1m · RR=4 (2m-horizon winner)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

SWING = 3
EQUAL_ATR = 0.25
LIQ_LOOKBACK = 80
SWEEP_WINDOW_MIN = 90
SL_PAD_ATR = 0.08
MIN_RISK_ATR = 0.12


def _atr_at(df: pd.DataFrame, i: int) -> float:
    row = df.iloc[i]
    for key in ("ATR", "atr"):
        if key in df.columns:
            try:
                v = float(row[key])
                if v == v and v > 0:
                    return v
            except Exception:
                pass
    if i < 1:
        return max(float(row["high"]) - float(row["low"]), 1e-6)
    trs = []
    for j in range(max(1, i - 13), i + 1):
        h, l, pc = (
            float(df.iloc[j]["high"]),
            float(df.iloc[j]["low"]),
            float(df.iloc[j - 1]["close"]),
        )
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return max(sum(trs) / len(trs), 1e-6)


def _ensure_emas(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if "ema20" not in out.columns and "EMA20" in out.columns:
        out = out.copy()
        out["ema20"] = out["EMA20"]
    if "ema50" not in out.columns and "EMA50" in out.columns:
        out = out.copy() if out is df else out
        out["ema50"] = out["EMA50"]
    if "ema20" not in out.columns or "ema50" not in out.columns:
        out = out.copy()
        out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    return out


def _bias_at(df: pd.DataFrame, ts: pd.Timestamp) -> int:
    """+1 bull, -1 bear, 0 chop — last bar at/before ts."""
    if df is None or len(df) < 55:
        return 0
    df = _ensure_emas(df)
    pos = df.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return 0
    row = df.iloc[pos]
    c = float(row["close"])
    e20 = float(row["ema20"])
    e50 = float(row["ema50"])
    if e20 > e50 and c > e20:
        return 1
    if e20 < e50 and c < e20:
        return -1
    return 0


def _swing_low(df: pd.DataFrame, i: int, n: int = SWING) -> bool:
    if i < n or i + n >= len(df):
        return False
    lo = float(df.iloc[i]["low"])
    return all(lo <= float(df.iloc[i + k]["low"]) for k in range(-n, n + 1) if k != 0)


def _swing_high(df: pd.DataFrame, i: int, n: int = SWING) -> bool:
    if i < n or i + n >= len(df):
        return False
    hi = float(df.iloc[i]["high"])
    return all(hi >= float(df.iloc[i + k]["high"]) for k in range(-n, n + 1) if k != 0)


def _equal_levels(levels: list[float], atr: float) -> list[float]:
    if not levels:
        return []
    tol = max(EQUAL_ATR * atr, 1e-9)
    levels = sorted(levels)
    clusters: list[list[float]] = [[levels[0]]]
    for v in levels[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def _is_engulf(row: pd.Series, prev: pd.Series, side: str) -> bool:
    o, c = float(row["open"]), float(row["close"])
    po, pc = float(prev["open"]), float(prev["close"])
    if side == "BUY":
        return c > o and pc < po and c >= po and o <= pc
    return c < o and pc > po and c <= po and o >= pc


def _recent_sweeps(df_liq: pd.DataFrame, *, end_i: Optional[int] = None) -> list[dict[str, Any]]:
    if df_liq is None or len(df_liq) < 60:
        return []
    end_i = len(df_liq) - 1 if end_i is None else end_i
    lows: list[tuple[int, float]] = []
    highs: list[tuple[int, float]] = []
    for i in range(SWING, min(end_i, len(df_liq) - SWING)):
        if _swing_low(df_liq, i):
            lows.append((i, float(df_liq.iloc[i]["low"])))
        if _swing_high(df_liq, i):
            highs.append((i, float(df_liq.iloc[i]["high"])))

    sweeps: list[dict[str, Any]] = []
    start = max(50, end_i - 40)
    for ei in range(start, end_i + 1):
        atr = _atr_at(df_liq, ei)
        row = df_liq.iloc[ei]
        h, l, c, o = (
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["open"]),
        )
        rlo = [lv for si, lv in lows if 3 <= ei - si <= LIQ_LOOKBACK]
        rhi = [hv for si, hv in highs if 3 <= ei - si <= LIQ_LOOKBACK]
        for liq in _equal_levels(rlo, atr):
            if l < liq - 0.02 * atr and c > liq and c > o:
                sl = l - SL_PAD_ATR * atr
                if abs(c - sl) > MIN_RISK_ATR * atr:
                    sweeps.append(
                        {
                            "t": df_liq.index[ei],
                            "side": "BUY",
                            "liq": liq,
                            "sl": sl,
                        }
                    )
                    break
        else:
            for liq in _equal_levels(rhi, atr):
                if h > liq + 0.02 * atr and c < liq and c < o:
                    sl = h + SL_PAD_ATR * atr
                    if abs(c - sl) > MIN_RISK_ATR * atr:
                        sweeps.append(
                            {
                                "t": df_liq.index[ei],
                                "side": "SELL",
                                "liq": liq,
                                "sl": sl,
                            }
                        )
                        break
    return sweeps


def evaluate_pa_liq_engulf(
    df_entry: pd.DataFrame,
    df_liq: pd.DataFrame,
    df_bias_a: pd.DataFrame,
    df_bias_b: pd.DataFrame,
    *,
    entry_tf_min: int = 1,
) -> Optional[dict[str, Any]]:
    """Evaluate on last closed entry bar.

    Grid winner call: evaluate_pa_liq_engulf(df_1m, df_3m, df_15m, df_5m)
    """
    if df_entry is None or df_liq is None or df_bias_a is None or df_bias_b is None:
        return None
    if len(df_entry) < 60 or len(df_liq) < 80:
        return None
    if len(df_bias_a) < 55 or len(df_bias_b) < 55:
        return None
    need = ("open", "high", "low", "close")
    if any(c not in df_entry.columns for c in need):
        return None

    i = len(df_entry) - 1
    if i < 1:
        return None
    row, prev = df_entry.iloc[i], df_entry.iloc[i - 1]
    ts = df_entry.index[i]
    atr = _atr_at(df_entry, i)
    max_bars = max(3, int(SWEEP_WINDOW_MIN / max(1, entry_tf_min)))

    b_a = _bias_at(df_bias_a, ts)
    b_b = _bias_at(df_bias_b, ts)
    if b_a == 0 or b_a != b_b:
        return None
    side = "BUY" if b_a > 0 else "SELL"
    if not _is_engulf(row, prev, side):
        return None

    sweeps = _recent_sweeps(df_liq)
    matched = None
    for sw in reversed(sweeps):
        if sw["side"] != side:
            continue
        sp = int(df_entry.index.searchsorted(sw["t"], side="left"))
        if sp >= len(df_entry):
            continue
        if sp <= i <= sp + max_bars:
            matched = sw
            break
    if matched is None:
        return None

    c = float(row["close"])
    if side == "BUY":
        sl = min(float(matched["sl"]), float(row["low"]) - SL_PAD_ATR * atr)
        if sl >= c or abs(c - sl) <= MIN_RISK_ATR * atr:
            return None
    else:
        sl = max(float(matched["sl"]), float(row["high"]) + SL_PAD_ATR * atr)
        if sl <= c or abs(c - sl) <= MIN_RISK_ATR * atr:
            return None

    return {
        "signal": side,
        "entry": c,
        "sl": float(sl),
        "atr": atr,
        "liq": float(matched["liq"]),
        "sweep_t": str(matched["t"])[:16],
        "bias_a": b_a,
        "bias_b": b_b,
        "stack": "liq=?|bias=15m+5m|entry=1m",
    }
