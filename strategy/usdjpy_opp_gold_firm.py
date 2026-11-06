"""Live USDJPY opp-to-GOLD liquidity firm (lab_usdjpy_opp_gold_math_dist_deep_bt).

Winner:
  OPP_FORCE · liq=15m · GOLD bias 15m+5m · entry=1m
  · corr 15m/96 θ≤−0.25 · Fourier HF≤0.45 · RR=4

Side = opposite of GOLD dual EMA bias; JPY liq sweep + engulf;
require inverse corr + opposing GOLD return + low Fourier HF.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy.pa_liq_engulf_brain import (
    MIN_RISK_ATR,
    SL_PAD_ATR,
    _atr_at,
    _bias_at,
    _is_engulf,
    _recent_sweeps,
)

logger = logging.getLogger(__name__)

FIRM: dict[str, Any] = {
    "mode": "OPP_FORCE",
    "liq_tf": "15m",
    "gold_bias_a": "15m",
    "gold_bias_b": "5m",
    "entry_tf": "1m",
    "corr_tf": "15m",
    "corr_lb": 96,
    "corr_thresh": -0.25,
    "fourier_hf_max": 0.45,
    "rr": 4.0,
    "lab": "lab_usdjpy_opp_gold_math_dist_deep_bt",
    "lab_name": (
        "OPP_FORCE|liq=15m|bias=15m+5m|entry=1m|"
        "corr=15m/96/θ-0.25|math=hf_max0.45"
    ),
}

_GOLD_CACHE: dict[str, Any] = {"ts": 0.0, "frames": {}}
_GOLD_CACHE_TTL = 45.0  # seconds


def is_opp_gold_symbol(symbol: Optional[str]) -> bool:
    return str(symbol or "").upper().strip() in ("USDJPY", "JPY")


def firm_spec() -> dict[str, Any]:
    return dict(FIRM)


def _fourier_hf_last(df: pd.DataFrame, win: int = 64) -> float:
    if df is None or len(df) < win + 2 or "close" not in df.columns:
        return 0.5
    c = df["close"].astype(float)
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    seg = ret.to_numpy(dtype=float)[-win:]
    if not np.isfinite(seg).all():
        return 0.5
    seg = seg - float(seg.mean())
    spec = np.abs(np.fft.rfft(seg)) ** 2
    tot = float(spec.sum())
    if tot <= 0:
        return 0.5
    cut = max(2, len(spec) // 2)
    return float(spec[cut:].sum() / tot)


def _rolling_corr_last(
    gold_df: pd.DataFrame, jpy_df: pd.DataFrame, lookback: int
) -> float:
    if gold_df is None or jpy_df is None:
        return 0.0
    if len(gold_df) < lookback // 2 or len(jpy_df) < lookback // 2:
        return 0.0
    gc = gold_df["close"].astype(float)
    jc = jpy_df["close"].astype(float)
    gr = np.log(gc / gc.shift(1)).replace([np.inf, -np.inf], np.nan)
    jr = np.log(jc / jc.shift(1)).replace([np.inf, -np.inf], np.nan)
    aligned = pd.concat([gr, jr], axis=1, join="inner").dropna()
    if len(aligned) < max(20, lookback // 2):
        return 0.0
    window = aligned.iloc[-lookback:]
    c = float(window.iloc[:, 0].corr(window.iloc[:, 1]) or 0.0)
    return c if np.isfinite(c) else 0.0


def _gold_ret_sum_last(gold_df: pd.DataFrame, n: int = 3) -> float:
    if gold_df is None or len(gold_df) < n + 2:
        return 0.0
    c = gold_df["close"].astype(float)
    r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(r.iloc[-n:].sum())


def load_gold_frames(
    *,
    need_tfs: tuple[str, ...] = ("5m", "15m"),
    bars: int = 200,
) -> dict[str, pd.DataFrame]:
    """Cached GOLD OHLCV for bias/corr (TTL short)."""
    now = time.time()
    if (
        _GOLD_CACHE["frames"]
        and now - float(_GOLD_CACHE["ts"]) < _GOLD_CACHE_TTL
        and all(tf in _GOLD_CACHE["frames"] for tf in need_tfs)
    ):
        return {tf: _GOLD_CACHE["frames"][tf] for tf in need_tfs}

    from indicators.prepare import prepare_df

    frames: dict[str, pd.DataFrame] = {}
    try:
        from data.market_data import fetch_ohlcv

        for tf in need_tfs:
            raw = fetch_ohlcv(tf, bars, symbol="GOLD")
            if raw is None or len(raw) < 60:
                continue
            df = prepare_df(raw)
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            frames[tf] = df
    except Exception as exc:
        logger.debug("GOLD frame load failed: %s", exc)

    if frames:
        _GOLD_CACHE["ts"] = now
        _GOLD_CACHE["frames"].update(frames)
    return frames


def evaluate_opp_gold_force(
    df_entry: pd.DataFrame,
    df_liq: pd.DataFrame,
    df_gold_a: pd.DataFrame,
    df_gold_b: pd.DataFrame,
    df_gold_corr: pd.DataFrame,
    df_jpy_corr: pd.DataFrame,
    *,
    entry_tf_min: int = 1,
    spec: Optional[dict[str, Any]] = None,
) -> tuple[Optional[dict[str, Any]], str, dict[str, Any]]:
    """Return (hit|None, reason, meta)."""
    spec = spec or FIRM
    meta: dict[str, Any] = {"firm_formula": True, "firm_kind": "usdjpy_opp_gold"}

    if df_entry is None or df_liq is None:
        return None, "missing JPY frames", meta
    if df_gold_a is None or df_gold_b is None or df_gold_corr is None:
        return None, "missing GOLD frames", meta
    if len(df_entry) < 60 or len(df_liq) < 80:
        return None, "JPY bars short", meta
    if len(df_gold_a) < 55 or len(df_gold_b) < 55:
        return None, "GOLD bias short", meta

    i = len(df_entry) - 1
    if i < 1:
        return None, "no entry bar", meta
    row, prev = df_entry.iloc[i], df_entry.iloc[i - 1]
    ts = df_entry.index[i]
    atr = _atr_at(df_entry, i)
    max_bars = max(3, int(90 / max(1, entry_tf_min)))

    # GOLD dual bias → opposite JPY side
    b_a = _bias_at(df_gold_a, ts)
    b_b = _bias_at(df_gold_b, ts)
    meta["gold_bias_a"] = b_a
    meta["gold_bias_b"] = b_b
    if b_a == 0 or b_a != b_b:
        return None, f"GOLD bias not sync ({b_a}/{b_b})", meta
    side = "SELL" if b_a > 0 else "BUY"
    meta["htf_bias"] = -b_a
    meta["with_opp_gold"] = True

    if not _is_engulf(row, prev, side):
        return None, f"no {side} engulf", meta

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
        return None, "no matching JPY liq sweep", meta

    # Corr opp gate
    lb = int(spec.get("corr_lb") or 96)
    thresh = float(spec.get("corr_thresh") or -0.25)
    corr_v = _rolling_corr_last(df_gold_corr, df_jpy_corr, lb)
    meta["corr_vs_gold"] = round(corr_v, 4)
    if corr_v >= thresh:
        return None, f"corr {corr_v:.3f} >= {thresh}", meta

    gret = _gold_ret_sum_last(df_gold_corr, 3)
    meta["gold_ret3"] = round(gret, 6)
    if side == "BUY" and gret >= 0:
        return None, "GOLD not weak for JPY BUY", meta
    if side == "SELL" and gret <= 0:
        return None, "GOLD not strong for JPY SELL", meta

    # Fourier HF max (trend-spectrum)
    hf_max = float(spec.get("fourier_hf_max") or 0.45)
    hf = _fourier_hf_last(df_entry, 64)
    meta["fourier_hf"] = round(hf, 4)
    if hf > hf_max:
        return None, f"fourier_hf {hf:.3f} > {hf_max}", meta

    c = float(row["close"])
    if side == "BUY":
        sl = min(float(matched["sl"]), float(row["low"]) - SL_PAD_ATR * atr)
        if sl >= c or abs(c - sl) <= MIN_RISK_ATR * atr:
            return None, "SL invalid", meta
    else:
        sl = max(float(matched["sl"]), float(row["high"]) + SL_PAD_ATR * atr)
        if sl <= c or abs(c - sl) <= MIN_RISK_ATR * atr:
            return None, "SL invalid", meta

    hit = {
        "signal": side,
        "entry": c,
        "sl": float(sl),
        "atr": atr,
        "liq": float(matched["liq"]),
        "sweep_t": str(matched["t"])[:16],
        "gold_bias": b_a,
        "corr_vs_gold": corr_v,
        "fourier_hf": hf,
    }
    return hit, "opp-gold firm ok", meta
