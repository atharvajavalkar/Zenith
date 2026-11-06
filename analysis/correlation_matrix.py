"""Rolling return correlations across the instrument universe."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import INSTRUMENT_UNIVERSE

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "updated_at": None,
    "matrix": {},
    "pairs": {},
    "leaders": {},
}


def _returns(df: pd.DataFrame, n: int = 96) -> Optional[pd.Series]:
    if df is None or len(df) < 20 or "close" not in df.columns:
        return None
    close = df["close"].astype(float).tail(n)
    rets = close.pct_change().dropna()
    if len(rets) < 15:
        return None
    return rets


def compute_correlation_state(
    frames_by_symbol: dict[str, dict[str, pd.DataFrame]],
    *,
    tf: str = "5m",
    lookback: int = 96,
) -> dict[str, Any]:
    """
    Build pairwise pearson corr on TF returns.
    frames_by_symbol: {SYM: {"5m": df, "1m": df, ...}}
    """
    series: dict[str, pd.Series] = {}
    for sym, frames in (frames_by_symbol or {}).items():
        if not frames:
            continue
        df = frames.get(tf)
        if df is None:
            df = frames.get("5m")
        if df is None:
            df = frames.get("1m")
        r = _returns(df, lookback)
        if r is not None:
            series[str(sym).upper()] = r

    symbols = list(series.keys())
    matrix: dict[str, dict[str, float]] = {s: {} for s in symbols}
    pairs: dict[str, float] = {}

    for i, a in enumerate(symbols):
        matrix[a][a] = 1.0
        for b in symbols[i + 1 :]:
            aligned = pd.concat([series[a], series[b]], axis=1, join="inner").dropna()
            if len(aligned) < 12:
                corr = 0.0
            else:
                corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) or 0.0)
                if not np.isfinite(corr):
                    corr = 0.0
            corr = round(corr, 3)
            matrix[a][b] = corr
            matrix[b][a] = corr
            pairs[f"{a}:{b}"] = corr

    # Leader: GOLD vs FX / BTC — useful desk bias
    leaders: dict[str, Any] = {}
    gold = "GOLD" if "GOLD" in matrix else ("XAUUSD" if "XAUUSD" in matrix else None)
    if gold:
        for s in symbols:
            if s == gold:
                continue
            leaders[s] = {
                "vs_gold": matrix.get(gold, {}).get(s),
                "gold_direction_hint": None,
            }

    # Directional hint from last few gold returns
    if gold and gold in series:
        g_last = float(series[gold].tail(3).sum())
        g_dir = "BULLISH" if g_last > 0 else ("BEARISH" if g_last < 0 else "NEUTRAL")
        for s, row in leaders.items():
            c = row.get("vs_gold")
            if c is None:
                continue
            # Positive corr → follow gold; strong negative → fade gold
            if abs(c) < 0.25:
                hint = "NEUTRAL"
            elif c >= 0.25:
                hint = g_dir
            else:
                hint = "BEARISH" if g_dir == "BULLISH" else ("BULLISH" if g_dir == "BEARISH" else "NEUTRAL")
            row["gold_direction_hint"] = hint
            row["gold_move"] = round(g_last, 5)

    out = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tf": tf,
        "lookback": lookback,
        "symbols": symbols,
        "matrix": matrix,
        "pairs": pairs,
        "leaders": leaders,
    }
    with _LOCK:
        _STATE.clear()
        _STATE.update(out)
    return out


def get_correlation_state() -> dict[str, Any]:
    with _LOCK:
        return dict(_STATE)


def correlation_bias_for(symbol: str, signal: str) -> dict[str, Any]:
    """
    Soft trade bias from GOLD linkage.
    Returns {aligned: bool|None, score: float, mc_delta: float, detail: str}
    """
    sym = str(symbol or "").upper()
    with _LOCK:
        leaders = dict(_STATE.get("leaders") or {})
        matrix = dict(_STATE.get("matrix") or {})

    row = leaders.get(sym) or {}
    hint = row.get("gold_direction_hint")
    vs = row.get("vs_gold")
    if hint is None and "GOLD" in matrix and sym in matrix.get("GOLD", {}):
        vs = matrix["GOLD"][sym]

    score = 0.0
    aligned = None
    if signal in ("BUY", "SELL") and hint in ("BULLISH", "BEARISH"):
        aligned = (signal == "BUY" and hint == "BULLISH") or (
            signal == "SELL" and hint == "BEARISH"
        )
        strength = min(1.0, abs(float(vs or 0)) )
        score = (0.6 if aligned else -0.6) * strength

    return {
        "aligned": aligned,
        "score": round(score, 3),
        "mc_delta": round(score * 2.0, 2),  # ±pp soft
        "vs_gold": vs,
        "gold_hint": hint,
        "detail": f"corr vs GOLD={vs} hint={hint}",
    }
