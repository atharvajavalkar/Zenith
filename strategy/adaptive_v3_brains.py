"""Live evaluators for Adaptive v3 minute/day equations (last-bar signals).

Brains: session_mix · circadian_kalman_z · entropy_tod
Matches analysis/lab_adaptive_minute_day_v3_bt.py formulas on the exec TF.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

# Per-symbol SL ATR — widened for live XM (paper hid pip risk)
SL_ATR_BY_SYMBOL = {
    "ETHUSD": 2.0,
    "BTCUSD": 2.0,
    "GOLD": 2.0,
    "NAS100": 3.0,
    "USDJPY": 2.5,
    "AUDUSD": 2.5,
    "USDCHF": 2.5,
    "UKOIL": 2.0,
}


def _min_sl_px(symbol: Optional[str]) -> float:
    """Floor stop distance in price units (not ATR). FX ≈ 15–20 pips."""
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    if sym.endswith("JPY"):
        return 0.20  # 20 pips on JPY
    if sym in ("AUDUSD", "USDCHF", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD"):
        return 0.0020  # 20 pips
    try:
        from strategy.instrument_meta import min_stop_distance

        return float(min_stop_distance(sym))
    except Exception:
        if len(sym) == 6 and sym.isalpha():
            return 0.0015
        return 0.0


def _atr_val(df: pd.DataFrame) -> Optional[float]:
    row = df.iloc[-1]
    for key in ("ATR", "atr"):
        if key in df.columns:
            try:
                v = float(row[key])
                if v == v and v > 0:
                    return v
            except Exception:
                pass
    # fallback true range mean
    if len(df) < 15:
        return None
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    v = float(tr.rolling(14).mean().iloc[-1])
    return v if v == v and v > 0 else None


def _ensure_atr_col(df: pd.DataFrame) -> pd.DataFrame:
    if "atr" in df.columns or "ATR" in df.columns:
        out = df.copy()
        if "atr" not in out.columns and "ATR" in out.columns:
            out["atr"] = out["ATR"]
        return out
    out = df.copy()
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    c = out["close"].astype(float)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    return out


def _perm_entropy_last(ret: np.ndarray, win: int = 48, order: int = 3) -> float:
    if len(ret) < win:
        return 0.5
    seg = ret[-win:]
    if not np.isfinite(seg).all():
        return 0.5
    # bandt-pompe ordinal patterns
    from itertools import permutations

    pats = list(permutations(range(order)))
    counts = np.zeros(len(pats), dtype=float)
    for i in range(len(seg) - order + 1):
        w = seg[i : i + order]
        ranks = tuple(np.argsort(np.argsort(w)))
        try:
            counts[pats.index(ranks)] += 1
        except ValueError:
            continue
    s = counts.sum()
    if s <= 0:
        return 0.5
    p = counts / s
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum() / math.log(len(pats)))
    return ent


def _hurst_last(ret: np.ndarray, win: int = 128) -> float:
    if len(ret) < win:
        return 0.5
    x = ret[-win:]
    if not np.isfinite(x).all() or np.std(x) < 1e-12:
        return 0.5
    y = np.cumsum(x - x.mean())
    r = float(y.max() - y.min())
    s = float(np.std(x))
    if s <= 0 or r <= 0:
        return 0.5
    return math.log(r / s) / math.log(win)


def _features_last(df: pd.DataFrame) -> Optional[dict[str, Any]]:
    """Compute last-bar features needed by v3 adaptive brains."""
    if df is None or len(df) < 80:
        return None
    d = _ensure_atr_col(df)
    c = d["close"].astype(float)
    atr = d["atr"] if "atr" in d.columns else d["ATR"]
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rn = ret.to_numpy(dtype=float)

    mu = ret.rolling(40).mean()
    sig = ret.rolling(40).std().replace(0, np.nan)
    snr = float((mu.abs() / sig).fillna(0).iloc[-1])
    drift_sign = float(np.sign(mu.fillna(0).iloc[-1]))

    ema = c.ewm(span=20, adjust=False).mean()
    resid = (c - ema) / atr.replace(0, np.nan)
    kz_s = (resid - resid.rolling(60).mean()) / resid.rolling(60).std().replace(0, np.nan)
    kalman_z = float(kz_s.fillna(0).iloc[-1])

    pe = _perm_entropy_last(rn, 48, 3)
    H = _hurst_last(rn, 128)

    idx = d.index[-1]
    ts = pd.Timestamp(idx)
    if ts.tzinfo is None:
        # treat naive as UTC clock for session structure
        hour = float(ts.hour)
        minute = float(ts.minute)
        dow = float(ts.dayofweek)
    else:
        utc = ts.tz_convert("UTC")
        hour = float(utc.hour)
        minute = float(utc.minute)
        dow = float(utc.dayofweek)
    mod = hour * 60.0 + minute
    tod = mod / 1440.0
    ang = 2.0 * math.pi * tod
    if 7 <= hour < 12:
        sess = 1
    elif 12 <= hour < 16:
        sess = 3
    elif 16 <= hour < 21:
        sess = 2
    else:
        sess = 0

    return {
        "snr": snr,
        "drift_sign": drift_sign,
        "kalman_z": kalman_z,
        "perm_ent": pe,
        "hurst": H,
        "hour": hour,
        "dow": dow,
        "tod": tod,
        "sin_tod": math.sin(ang),
        "cos_tod": math.cos(ang),
        "sin_week": math.sin(2.0 * math.pi * (dow + tod) / 7.0),
        "cos_week": math.cos(2.0 * math.pi * (dow + tod) / 7.0),
        "sess": sess,
        "close": float(c.iloc[-1]),
    }


def _side_to_hit(
    side: int,
    close: float,
    atr: float,
    sl_atr: float,
    *,
    symbol: Optional[str] = None,
    hour_utc: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    if side == 0 or atr <= 0:
        return None
    floor = _min_sl_px(symbol)
    atr_risk = float(sl_atr) * float(atr)
    try:
        from analysis.xm_session_costs import min_risk_with_spread

        risk = min_risk_with_spread(
            symbol, atr_risk, hour_utc=hour_utc, floor_px=floor
        )
    except Exception:
        # Live-safe fallback: never use noise-sized stops
        risk = max(atr_risk, float(floor))
    if risk <= 0:
        return None
    if side > 0:
        sl = close - risk
        if sl >= close:
            return None
        return {
            "signal": "BUY",
            "entry": close,
            "sl": sl,
            "atr": atr,
            "side": 1,
            "sl_risk": risk,
            "sl_floor": floor,
        }
    sl = close + risk
    if sl <= close:
        return None
    return {
        "signal": "SELL",
        "entry": close,
        "sl": sl,
        "atr": atr,
        "side": -1,
        "sl_risk": risk,
        "sl_floor": floor,
    }


def _sl_for(symbol: Optional[str], default: float = 2.0) -> float:
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    return float(SL_ATR_BY_SYMBOL.get(sym, default))


def evaluate_session_mix(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    sl_atr: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Asia fade-lean / London trend / NY mix soft score."""
    feat = _features_last(df)
    atr = _atr_val(df)
    if feat is None or atr is None:
        return None
    st = feat["drift_sign"] * (
        0.55 * feat["snr"] + 0.45 * float(np.clip((feat["hurst"] - 0.5) * 4.0, -1.0, 1.0))
    )
    sf = -math.tanh(feat["kalman_z"] / 2.0)
    alpha = (0.35, 0.75, 0.55, 0.45)[int(feat["sess"])]
    score = alpha * st + (1.0 - alpha) * sf
    # Stricter than research — cut spam losing streaks on live XM
    thr = 0.42 if feat["sess"] == 3 else 0.32
    side = 1 if score >= thr else (-1 if score <= -thr else 0)
    hit = _side_to_hit(
        side,
        feat["close"],
        atr,
        sl_atr or _sl_for(symbol),
        symbol=symbol,
        hour_utc=feat["hour"],
    )
    if hit:
        hit["meta"] = {"score": score, "sess": feat["sess"], "thr": thr}
    return hit


def evaluate_circadian_kalman_z(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    sl_atr: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Kalman |z| threshold breathes with TOD + weekday."""
    feat = _features_last(df)
    atr = _atr_val(df)
    if feat is None or atr is None:
        return None
    # Higher base z — fewer weak fades
    zthr = 2.2 + 0.5 * feat["sin_tod"] + 0.25 * (feat["dow"] / 7.0)
    kz = feat["kalman_z"]
    side = 1 if kz <= -zthr else (-1 if kz >= zthr else 0)
    hit = _side_to_hit(
        side,
        feat["close"],
        atr,
        sl_atr or _sl_for(symbol),
        symbol=symbol,
        hour_utc=feat["hour"],
    )
    if hit:
        hit["meta"] = {"kalman_z": kz, "zthr": zthr}
    return hit


def evaluate_entropy_tod(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    sl_atr: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Trade with drift when perm-entropy below TOD-adapted ceiling."""
    feat = _features_last(df)
    atr = _atr_val(df)
    if feat is None or atr is None:
        return None
    pmax = 0.72 + 0.08 * feat["cos_tod"] + 0.03 * feat["sin_week"]
    ds = feat["drift_sign"]
    snr_ok = True
    try:
        # Require some drift strength so entropy alone doesn't spam
        snr_ok = float(feat.get("snr") or 0) >= 0.22
    except Exception:
        snr_ok = True
    side = int(ds) if (feat["perm_ent"] <= pmax and ds != 0 and snr_ok) else 0
    hit = _side_to_hit(
        side,
        feat["close"],
        atr,
        sl_atr or _sl_for(symbol),
        symbol=symbol,
        hour_utc=feat["hour"],
    )
    if hit:
        hit["meta"] = {"perm_ent": feat["perm_ent"], "pmax": pmax, "drift_sign": ds}
    return hit


def evaluate_adaptive_v3(
    df: pd.DataFrame,
    brain: str,
    *,
    symbol: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    from strategy.adaptive_v3_runtime import resolve_brain, should_sitout

    feat = _features_last(df)
    if feat is None:
        return None
    if should_sitout(symbol, feat["hour"]):
        return None

    primary = str(brain or "").strip().lower()
    active, route = resolve_brain(symbol, primary)
    b = active
    if b in ("session_mix", "adaptive_session_mix"):
        hit = evaluate_session_mix(df, symbol=symbol)
    elif b in ("circadian_kalman_z", "circadian_kalman", "adaptive_circadian_kalman_z"):
        hit = evaluate_circadian_kalman_z(df, symbol=symbol)
    elif b in ("entropy_tod", "adaptive_entropy_tod"):
        hit = evaluate_entropy_tod(df, symbol=symbol)
    else:
        hit = None
    if hit:
        hit["meta"] = {**(hit.get("meta") or {}), "route": route}
        hit["brain_active"] = b
    return hit
