"""Live evaluators for Strategy Lab / Dukascopy deep-hunt winners (last-bar signals).

Brains: cta_10_40 · cta_50_200 · rsi · bb_1.5 · bb_2 · bb_2.5 · st_3 · roc_120 · volmom
Forced RR TP handled downstream via SYMBOL_RR.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def _atr(row: pd.Series, df: pd.DataFrame) -> Optional[float]:
    for key in ("ATR", "atr"):
        if key in df.columns:
            try:
                v = float(row[key])
                if v == v and v > 0:
                    return v
            except Exception:
                pass
    return None


def evaluate_roc_cross(
    df: pd.DataFrame,
    *,
    period: int = 120,
    min_abs: float = 0.02,
    sl_atr: float = 1.4,
) -> Optional[dict[str, Any]]:
    """ROC zero-cross with minimum magnitude (matches lab _collect_mom_roc)."""
    if df is None or len(df) < period + 5:
        return None
    close = df["close"]
    roc = close.pct_change(period)
    if pd.isna(roc.iloc[-1]) or pd.isna(roc.iloc[-2]):
        return None
    atr = _atr(df.iloc[-1], df)
    if atr is None:
        return None
    c = float(close.iloc[-1])
    r0, r1 = float(roc.iloc[-2]), float(roc.iloc[-1])
    if r0 <= 0 < r1 and abs(r1) >= min_abs:
        sl = c - sl_atr * atr
        if sl >= c:
            return None
        return {"signal": "BUY", "entry": c, "sl": sl, "atr": atr, "roc": r1}
    if r0 >= 0 > r1 and abs(r1) >= min_abs:
        sl = c + sl_atr * atr
        if sl <= c:
            return None
        return {"signal": "SELL", "entry": c, "sl": sl, "atr": atr, "roc": r1}
    return None


def evaluate_cta_cross(
    df: pd.DataFrame,
    *,
    fast: int = 10,
    slow: int = 40,
    sl_atr: float = 1.5,
) -> Optional[dict[str, Any]]:
    if df is None or len(df) < slow + 5:
        return None
    close = df["close"]
    f = close.ewm(span=fast, adjust=False).mean()
    s = close.ewm(span=slow, adjust=False).mean()
    atr = _atr(df.iloc[-1], df)
    if atr is None:
        return None
    c = float(close.iloc[-1])
    f0, s0 = float(f.iloc[-2]), float(s.iloc[-2])
    f1, s1 = float(f.iloc[-1]), float(s.iloc[-1])
    if f0 <= s0 and f1 > s1:
        sl = c - sl_atr * atr
        if sl >= c:
            return None
        return {"signal": "BUY", "entry": c, "sl": sl, "atr": atr}
    if f0 >= s0 and f1 < s1:
        sl = c + sl_atr * atr
        if sl <= c:
            return None
        return {"signal": "SELL", "entry": c, "sl": sl, "atr": atr}
    return None


def evaluate_rsi_mr(
    df: pd.DataFrame,
    *,
    period: int = 14,
    lo: float = 30.0,
    hi: float = 70.0,
    sl_atr: float = 1.3,
) -> Optional[dict[str, Any]]:
    if df is None or len(df) < period + 5:
        return None
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    if pd.isna(rsi.iloc[-1]) or pd.isna(rsi.iloc[-2]):
        return None
    atr = _atr(df.iloc[-1], df)
    if atr is None:
        return None
    c = float(df["close"].iloc[-1])
    r0, r1 = float(rsi.iloc[-2]), float(rsi.iloc[-1])
    if r0 < lo <= r1:
        sl = c - sl_atr * atr
        if sl >= c:
            return None
        return {"signal": "BUY", "entry": c, "sl": sl, "atr": atr, "rsi": r1}
    if r0 > hi >= r1:
        sl = c + sl_atr * atr
        if sl <= c:
            return None
        return {"signal": "SELL", "entry": c, "sl": sl, "atr": atr, "rsi": r1}
    return None


def evaluate_bb_mr(
    df: pd.DataFrame,
    *,
    std_mult: float = 2.0,
    period: int = 20,
    sl_atr: float = 1.2,
) -> Optional[dict[str, Any]]:
    if df is None or len(df) < period + 5:
        return None
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    lo = mid - std_mult * std
    hi = mid + std_mult * std
    row = df.iloc[-1]
    atr = _atr(row, df)
    if atr is None or pd.isna(lo.iloc[-1]) or pd.isna(hi.iloc[-1]):
        return None
    c, o = float(row["close"]), float(row["open"])
    if c < float(lo.iloc[-1]) and c > o:
        sl = c - sl_atr * atr
        if sl >= c:
            return None
        return {"signal": "BUY", "entry": c, "sl": sl, "atr": atr}
    if c > float(hi.iloc[-1]) and c < o:
        sl = c + sl_atr * atr
        if sl <= c:
            return None
        return {"signal": "SELL", "entry": c, "sl": sl, "atr": atr}
    return None


def evaluate_supertrend(
    df: pd.DataFrame,
    *,
    mult: float = 3.0,
    sl_atr: float = 1.5,
) -> Optional[dict[str, Any]]:
    if df is None or len(df) < 30:
        return None
    atr_s = df["ATR"] if "ATR" in df.columns else df.get("atr")
    if atr_s is None:
        return None
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * atr_s
    lower = hl2 - mult * atr_s
    direction = [1] * len(df)
    for i in range(1, len(df)):
        c = float(df["close"].iloc[i])
        if c > float(upper.iloc[i - 1]):
            direction[i] = 1
        elif c < float(lower.iloc[i - 1]):
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    if direction[-1] == direction[-2]:
        return None
    atr = _atr(df.iloc[-1], df)
    if atr is None:
        return None
    c = float(df["close"].iloc[-1])
    if direction[-1] == 1:
        sl = c - sl_atr * atr
        if sl >= c:
            return None
        return {"signal": "BUY", "entry": c, "sl": sl, "atr": atr}
    sl = c + sl_atr * atr
    if sl <= c:
        return None
    return {"signal": "SELL", "entry": c, "sl": sl, "atr": atr}


def evaluate_vol_scaled_mom(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    thr: float = 0.5,
    sl_atr: float = 1.5,
    symbol: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Vol-scaled momentum zero-cross (matches lab _collect_vol_scaled_mom)."""
    if df is None or len(df) < lookback + 5:
        return None
    close = df["close"]
    ret = close.pct_change(lookback)
    vol = close.pct_change().rolling(lookback).std()
    score = ret / vol.replace(0, pd.NA)
    if pd.isna(score.iloc[-1]) or pd.isna(score.iloc[-2]):
        return None
    atr = _atr(df.iloc[-1], df)
    if atr is None:
        return None
    c = float(close.iloc[-1])
    s0, s1 = float(score.iloc[-2]), float(score.iloc[-1])
    atr_risk = float(sl_atr) * float(atr)
    hour = None
    try:
        ts = pd.Timestamp(df.index[-1])
        hour = float(ts.tz_convert("UTC").hour) if ts.tzinfo else float(ts.hour)
    except Exception:
        hour = None
    try:
        from analysis.xm_session_costs import min_risk_with_spread

        risk = min_risk_with_spread(symbol or "NAS100", atr_risk, hour_utc=hour)
    except Exception:
        risk = atr_risk
    if s0 <= 0 < s1 and s1 > thr:
        sl = c - risk
        if sl >= c:
            return None
        return {"signal": "BUY", "entry": c, "sl": sl, "atr": atr, "score": s1}
    if s0 >= 0 > s1 and s1 < -thr:
        sl = c + risk
        if sl <= c:
            return None
        return {"signal": "SELL", "entry": c, "sl": sl, "atr": atr, "score": s1}
    return None
