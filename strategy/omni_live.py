"""Live Omni-horizon adaptive router (all-six survivors).

Ports analysis/lab_omnihorizon_adaptive_bt.py routing to last-bar live signals.
Only armed for the five XM all-horizon survivors — other (symbol, tf) → None.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy.adaptive_v3_brains import (
    _atr_val,
    _ensure_atr_col,
    _side_to_hit,
)

# Survivors from data/lab_omnihorizon_adaptive.json (all 7d→1y green, XM spreads)
OMNI_LIVE: dict[tuple[str, str], dict[str, Any]] = {
    ("GOLD", "3m"): {
        "look_fast": 120,
        "look_mid": 240,
        "look_slow": 600,
        "edge_min": 0.52,
        "hysteresis": 25,
        "min_agree": 2,
        "sl_atr": 2.0,
        "rr": 3.0,
    },
    ("GOLD", "5m"): {
        "look_fast": 80,
        "look_mid": 400,
        "look_slow": 600,
        "edge_min": 0.55,
        "hysteresis": 25,
        "min_agree": 2,
        "sl_atr": 2.0,
        "rr": 3.0,
    },
    ("BTCUSD", "5m"): {
        "look_fast": 80,
        "look_mid": 240,
        "look_slow": 600,
        "edge_min": 0.52,
        "hysteresis": 25,
        "min_agree": 2,
        "sl_atr": 2.0,
        "rr": 3.0,
    },
    ("BTCUSD", "3m"): {
        "look_fast": 120,
        "look_mid": 240,
        "look_slow": 600,
        "edge_min": 0.55,
        "hysteresis": 25,
        "min_agree": 2,
        "sl_atr": 2.5,
        "rr": 3.0,
    },
    ("UKOIL", "3m"): {
        "look_fast": 120,
        "look_mid": 240,
        "look_slow": 600,
        "edge_min": 0.52,
        "hysteresis": 25,
        "min_agree": 2,
        "sl_atr": 2.0,
        "rr": 3.0,
    },
}

LIVE_OMNI_SYMBOLS = frozenset({"GOLD", "BTCUSD", "UKOIL"})


def _norm_sym(symbol: Optional[str]) -> str:
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        return "GOLD"
    if sym in ("US100Cash", "USTEC", "NAS100Cash"):
        return "NAS100"
    if sym in ("BRENTCash", "UKOILCash"):
        return "UKOIL"
    return sym


def _sess_code(hour: np.ndarray) -> np.ndarray:
    out = np.zeros(len(hour), dtype=int)
    out[(hour >= 7) & (hour < 12)] = 1
    out[(hour >= 12) & (hour < 16)] = 3
    out[(hour >= 16) & (hour < 21)] = 2
    return out


def _feature_frame(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 120:
        return None
    d = _ensure_atr_col(df).copy()
    c = d["close"].astype(float)
    atr = d["atr"] if "atr" in d.columns else d["ATR"]
    atr = atr.astype(float).replace(0, np.nan)
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu = ret.rolling(40).mean()
    sig = ret.rolling(40).std().replace(0, np.nan)
    d["snr"] = (mu.abs() / sig).fillna(0.0)
    d["drift_sign"] = np.sign(mu.fillna(0.0))
    d["close"] = c
    ema = c.ewm(span=20, adjust=False).mean()
    resid = (c - ema) / atr
    kz = (resid - resid.rolling(60).mean()) / resid.rolling(60).std().replace(0, np.nan)
    d["kalman_z"] = kz.fillna(0.0)
    # Fast Hurst proxy (constant mid) — full R/S every bar is too slow for live
    d["hurst"] = 0.5
    idx = pd.DatetimeIndex(pd.to_datetime(d.index, utc=True, errors="coerce"))
    if getattr(idx, "tz", None) is None:
        # try coerce
        try:
            idx = pd.to_datetime(d.index, utc=True)
        except Exception:
            idx = None
    if idx is not None and getattr(idx, "tz", None) is not None:
        hours = idx.hour.astype(float).to_numpy()
        dows = idx.dayofweek.astype(float).to_numpy()
        mins = idx.minute.astype(float).to_numpy()
    else:
        hours = np.array(
            [float(getattr(ts, "hour", 12) or 12) for ts in d.index], dtype=float
        )
        dows = np.array(
            [float(getattr(ts, "dayofweek", 2) or 2) for ts in d.index], dtype=float
        )
        mins = np.array(
            [float(getattr(ts, "minute", 0) or 0) for ts in d.index], dtype=float
        )
    tod = (hours * 60.0 + mins) / 1440.0
    ang = 2.0 * math.pi * tod
    d["hour"] = hours
    d["dow"] = dows
    d["sin_tod"] = np.sin(ang)
    d["cos_tod"] = np.cos(ang)
    d["sess"] = _sess_code(hours)
    d["ret"] = ret
    return d


def _sides_session_mix(d: pd.DataFrame, thr_base: float, thr_ov: float) -> np.ndarray:
    snr = d["snr"].to_numpy(dtype=float)
    ds = d["drift_sign"].to_numpy(dtype=float)
    H = d["hurst"].to_numpy(dtype=float)
    kz = d["kalman_z"].to_numpy(dtype=float)
    sess = d["sess"].to_numpy(dtype=int)
    st = ds * (0.55 * snr + 0.45 * np.clip((H - 0.5) * 4.0, -1.0, 1.0))
    sf = -np.tanh(kz / 2.0)
    alpha = np.array([0.35, 0.75, 0.55, 0.45])[sess]
    score = alpha * st + (1.0 - alpha) * sf
    thr = np.where(sess == 3, thr_ov, thr_base)
    out = np.zeros(len(d), dtype=int)
    out[score >= thr] = 1
    out[score <= -thr] = -1
    return out


def _sides_circadian(d: pd.DataFrame, z0: float, z_sin: float) -> np.ndarray:
    kz = d["kalman_z"].to_numpy(dtype=float)
    zthr = z0 + z_sin * d["sin_tod"].to_numpy(dtype=float) + 0.2 * (
        d["dow"].to_numpy(dtype=float) / 7.0
    )
    out = np.zeros(len(d), dtype=int)
    out[kz <= -zthr] = 1
    out[kz >= zthr] = -1
    return out


def _sides_static(d: pd.DataFrame) -> np.ndarray:
    ds = d["drift_sign"].to_numpy(dtype=float)
    out = np.zeros(len(d), dtype=int)
    out[ds > 0] = 1
    out[ds < 0] = -1
    return out


def _rolling_edge(sides: np.ndarray, ret: np.ndarray, look: int) -> np.ndarray:
    n = len(sides)
    nxt = np.roll(ret, -1)
    nxt[-1] = 0.0
    active = sides != 0
    raw = np.full(n, np.nan, dtype=float)
    raw[active] = ((np.sign(sides[active]) * np.sign(nxt[active])) > 0).astype(float)
    return (
        pd.Series(raw)
        .rolling(look, min_periods=max(20, look // 5))
        .mean()
        .fillna(0.5)
        .to_numpy(dtype=float)
    )


def _omni_route_last(
    brain_sides: dict[str, np.ndarray],
    ret: np.ndarray,
    *,
    look_fast: int,
    look_mid: int,
    look_slow: int,
    edge_min: float,
    hysteresis: int,
    min_agree: int,
) -> tuple[int, str, dict[str, Any]]:
    names = list(brain_sides.keys())
    n = len(ret)
    edge_mat = np.column_stack(
        [
            0.45 * _rolling_edge(brain_sides[nm], ret, look_fast)
            + 0.35 * _rolling_edge(brain_sides[nm], ret, look_mid)
            + 0.20 * _rolling_edge(brain_sides[nm], ret, look_slow)
            for nm in names
        ]
    )
    best_i = np.argmax(edge_mat, axis=1)
    pick = best_i.copy()
    last = int(pick[0])
    hold = 0
    for i in range(n):
        if hold > 0 and edge_mat[i, last] >= edge_min - 0.02:
            pick[i] = last
            hold -= 1
        else:
            last = int(best_i[i])
            pick[i] = last
            hold = hysteresis

    i = n - 1
    if edge_mat[i, int(pick[i])] < edge_min:
        return 0, "sitout", {"reason": "edge_min", "edge": float(edge_mat[i, int(pick[i])])}
    nm = names[int(pick[i])]
    side = int(brain_sides[nm][i])
    if side == 0:
        return 0, nm, {"reason": "flat_brain"}
    agree = 0
    for j, other in enumerate(names):
        if edge_mat[i, j] < edge_min:
            continue
        if int(brain_sides[other][i]) == side:
            agree += 1
    if agree < min_agree:
        return 0, nm, {"reason": "disagree", "agree": agree}
    return side, nm, {"edge": float(edge_mat[i, int(pick[i])]), "agree": agree}


def evaluate_omni(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    tf: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Last-bar omni signal for wired survivors only."""
    sym = _norm_sym(symbol)
    tf_s = str(tf or "5m").strip().lower()
    cfg = OMNI_LIVE.get((sym, tf_s))
    if cfg is None:
        return None
    # Keep runtime light: last ~ max(look_slow)+200 bars
    need = int(cfg["look_slow"]) + 220
    if len(df) < 160:
        return None
    use = df.iloc[-min(len(df), need) :].copy()
    feat = _feature_frame(use)
    if feat is None:
        return None
    ret = feat["ret"].to_numpy(dtype=float)
    brain_sides = {
        "session_mix": _sides_session_mix(feat, 0.22, 0.35),
        "session_mix_strict": _sides_session_mix(feat, 0.32, 0.42),
        "circadian": _sides_circadian(feat, 1.7, 0.6),
        "static": _sides_static(feat),
    }
    side, active, meta = _omni_route_last(
        brain_sides,
        ret,
        look_fast=int(cfg["look_fast"]),
        look_mid=int(cfg["look_mid"]),
        look_slow=int(cfg["look_slow"]),
        edge_min=float(cfg["edge_min"]),
        hysteresis=int(cfg["hysteresis"]),
        min_agree=int(cfg["min_agree"]),
    )
    atr = _atr_val(feat)
    if atr is None or side == 0:
        return None
    hour = float(feat["hour"].iloc[-1])
    hit = _side_to_hit(
        side,
        float(feat["close"].iloc[-1]) if "close" in feat.columns else float(use["close"].iloc[-1]),
        float(atr),
        float(cfg["sl_atr"]),
        symbol=sym,
        hour_utc=hour,
    )
    if not hit:
        return None
    # ensure close present
    if "close" not in feat.columns:
        hit["entry"] = float(use["close"].iloc[-1])
    hit["brain_active"] = f"omni:{active}"
    hit["meta"] = {
        **meta,
        "omni_tf": tf_s,
        "omni_cfg": {k: cfg[k] for k in ("look_fast", "look_mid", "look_slow", "edge_min", "sl_atr", "rr")},
        "active_brain": active,
    }
    hit["rr"] = float(cfg["rr"])
    return hit
