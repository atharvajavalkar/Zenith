"""Live firm crypto formulas from deep hunts.

ETHUSD (lab_eth_withtrend_math_dist_deep_bt · robust pick):
  Donchian20 break · with-trend 15m+1h · Fourier HF≤0.40 (prefer trend spectrum)
  · sl=2.0×ATR · RR=4 · min_risk=$6
  Note: older ETH used HF≥0.35 (noise/MR). New polarity is HF max.

BTCUSD (lab_crypto_l3_fourier_hunt_bt):
  BB 2.5σ · sl=2.5×ATR · RR=2 · min_risk=$32 · HF≥0.40 · |z|≥1.5 · |kurt|<7 · eff≥0.25

Research wiring helpers — used by strategy.signal when CRYPTO_FIRM_FORMULA_ENABLED.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

# Firm specs (match hunt winners)
FIRM: dict[str, dict[str, Any]] = {
    "ETHUSD": {
        "family": "don_break",
        "don_n": 20,
        "bb_mult": None,
        "sl_atr": 2.0,
        "rr": 4.0,
        "min_risk_px": 6.0,
        # trend-spectrum: reject choppy high-HF bars
        "fourier_hf_max": 0.40,
        "fourier_hf_min": None,
        "fourier_lf_min": None,
        "abs_z_min": None,
        "kurt_max": None,
        "min_eff": None,
        "min_trend_str": None,
        "lab": "lab_eth_withtrend_math_dist_deep_bt",
        "lab_name": "don20_don_break_sl2_rr4_r6_fourier_hf_max0.4",
    },
    "ETH": {
        "family": "don_break",
        "don_n": 20,
        "bb_mult": None,
        "sl_atr": 2.0,
        "rr": 4.0,
        "min_risk_px": 6.0,
        "fourier_hf_max": 0.40,
        "fourier_hf_min": None,
        "fourier_lf_min": None,
        "abs_z_min": None,
        "kurt_max": None,
        "min_eff": None,
        "min_trend_str": None,
        "lab": "lab_eth_withtrend_math_dist_deep_bt",
        "lab_name": "don20_don_break_sl2_rr4_r6_fourier_hf_max0.4",
    },
    "BTCUSD": {
        "family": "bb_pull",
        "bb_mult": 2.5,
        "sl_atr": 2.5,
        "rr": 2.0,
        "min_risk_px": 32.0,
        "fourier_hf_min": 0.40,
        "fourier_hf_max": None,
        "abs_z_min": 1.5,
        "kurt_max": 7.0,
        "min_eff": 0.25,
    },
    "BTC": {
        "family": "bb_pull",
        "bb_mult": 2.5,
        "sl_atr": 2.5,
        "rr": 2.0,
        "min_risk_px": 32.0,
        "fourier_hf_min": 0.40,
        "fourier_hf_max": None,
        "abs_z_min": 1.5,
        "kurt_max": 7.0,
        "min_eff": 0.25,
    },
}


def is_firm_crypto_symbol(symbol: Optional[str]) -> bool:
    return str(symbol or "").upper().strip() in FIRM


def firm_spec(symbol: Optional[str]) -> Optional[dict[str, Any]]:
    return FIRM.get(str(symbol or "").upper().strip())


def _fourier_hf_last(ret: np.ndarray, win: int = 64) -> float:
    if len(ret) < win:
        return 0.5
    seg = ret[-win:]
    if not np.isfinite(seg).all():
        return 0.5
    seg = seg - float(seg.mean())
    spec = np.abs(np.fft.rfft(seg)) ** 2
    tot = float(spec.sum())
    if tot <= 0:
        return 0.5
    cut = max(2, len(spec) // 2)
    return float(spec[cut:].sum() / tot)


def last_bar_micro(df: pd.DataFrame) -> dict[str, float]:
    """Microstructure features on the last bar (for gates + trade features)."""
    if df is None or len(df) < 65:
        return {
            "fourier_hf": 0.5,
            "fourier_lf": 0.5,
            "ret_z": 0.0,
            "kurt60": 0.0,
            "eff": 0.0,
            "trend_str": 0.0,
        }
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_a = ret.to_numpy(dtype=float)
    mu = float(ret.iloc[-60:].mean())
    sd = float(ret.iloc[-60:].std()) or 1e-12
    ret_z = float((ret_a[-1] - mu) / sd)
    kurt60 = float(ret.iloc[-60:].kurt()) if len(ret) >= 60 else 0.0
    rng = float(h.iloc[-1] - l.iloc[-1]) or 1e-12
    eff = float(abs(float(c.iloc[-1]) - float(o.iloc[-1])) / rng)
    hf = _fourier_hf_last(ret_a, 64)
    ema8 = float(c.ewm(span=8, adjust=False).mean().iloc[-1])
    ema14 = float(c.ewm(span=14, adjust=False).mean().iloc[-1])
    atr = None
    for key in ("atr", "ATR"):
        if key in df.columns:
            try:
                atr = float(df[key].iloc[-1])
            except Exception:
                atr = None
            if atr and atr == atr and atr > 0:
                break
    atr = float(atr or 1e-9)
    trend_str = abs(ema8 - ema14) / atr
    return {
        "fourier_hf": hf,
        "fourier_lf": 1.0 - hf,
        "ret_z": ret_z,
        "kurt60": kurt60,
        "eff": min(1.0, max(0.0, eff)),
        "trend_str": float(trend_str),
    }


def passes_firm_filters(
    df: pd.DataFrame,
    symbol: Optional[str],
) -> tuple[bool, str, dict[str, float]]:
    """Return (ok, reason, micro features)."""
    spec = firm_spec(symbol)
    if spec is None:
        return True, "not firm crypto", {}
    micro = last_bar_micro(df)

    hf_max = spec.get("fourier_hf_max")
    if hf_max is not None and micro["fourier_hf"] > float(hf_max):
        return (
            False,
            f"fourier_hf {micro['fourier_hf']:.3f} > {hf_max}",
            micro,
        )
    hf_min = spec.get("fourier_hf_min")
    if hf_min is not None and micro["fourier_hf"] < float(hf_min):
        return (
            False,
            f"fourier_hf {micro['fourier_hf']:.3f} < {hf_min}",
            micro,
        )
    lf_min = spec.get("fourier_lf_min")
    if lf_min is not None and micro["fourier_lf"] < float(lf_min):
        return (
            False,
            f"fourier_lf {micro['fourier_lf']:.3f} < {lf_min}",
            micro,
        )
    z_min = spec.get("abs_z_min")
    if z_min is not None and abs(micro["ret_z"]) < float(z_min):
        return (
            False,
            f"|ret_z| {abs(micro['ret_z']):.3f} < {z_min}",
            micro,
        )
    k_max = spec.get("kurt_max")
    if k_max is not None and abs(micro["kurt60"]) > float(k_max):
        return (
            False,
            f"|kurt| {abs(micro['kurt60']):.2f} > {k_max}",
            micro,
        )
    eff_min = spec.get("min_eff")
    if eff_min is not None and micro["eff"] < float(eff_min):
        return False, f"eff {micro['eff']:.3f} < {eff_min}", micro
    t_min = spec.get("min_trend_str")
    if t_min is not None and micro["trend_str"] < float(t_min):
        return (
            False,
            f"trend_str {micro['trend_str']:.3f} < {t_min}",
            micro,
        )
    return True, "firm filters ok", micro


def apply_firm_sl(
    signal: str,
    entry: float,
    atr: float,
    symbol: Optional[str],
) -> Optional[float]:
    """Widen SL to firm sl_atr × ATR and enforce min_risk_px."""
    spec = firm_spec(symbol)
    if spec is None:
        return None
    sl_atr = float(spec["sl_atr"])
    min_risk = float(spec["min_risk_px"])
    a = max(float(atr or 0), 1e-9)
    risk = max(sl_atr * a, min_risk)
    side = str(signal or "").upper()
    if side == "BUY":
        return float(entry) - risk
    if side == "SELL":
        return float(entry) + risk
    return None
