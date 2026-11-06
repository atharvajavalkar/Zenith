"""Live firm FX formulas from lab_fx_fourier_vol_hunt_bt.

AUDUSD: 3m · BB 2.5σ · sl=2.0×ATR · RR=4 · min_risk=1.5pip · Fourier HF≥0.40
USDCHF: 1m · BB 2.5σ · sl=2.0×ATR · RR=4 · min_risk=1.5pip · Fourier HF≥0.40

Wired via strategy.signal BB path when FX_FIRM_FORMULA_ENABLED.
"""
from __future__ import annotations

from typing import Any, Optional

from strategy.crypto_firm_formula import last_bar_micro

# Firm specs (match hunt winners)
FIRM: dict[str, dict[str, Any]] = {
    "AUDUSD": {
        "bb_mult": 2.5,
        "sl_atr": 2.0,
        "rr": 4.0,
        "min_risk_px": 0.00015,
        "fourier_hf_min": 0.40,
        "scan_tf": "3m",
    },
    "USDCHF": {
        "bb_mult": 2.5,
        "sl_atr": 2.0,
        "rr": 4.0,
        "min_risk_px": 0.00015,
        "fourier_hf_min": 0.40,
        "scan_tf": "1m",
    },
}


def is_firm_fx_symbol(symbol: Optional[str]) -> bool:
    return str(symbol or "").upper().strip() in FIRM


def firm_spec(symbol: Optional[str]) -> Optional[dict[str, Any]]:
    return FIRM.get(str(symbol or "").upper().strip())


def passes_firm_filters(
    df,
    symbol: Optional[str],
) -> tuple[bool, str, dict[str, float]]:
    spec = firm_spec(symbol)
    if spec is None:
        return True, "not firm fx", {}
    micro = last_bar_micro(df)
    hf_min = spec.get("fourier_hf_min")
    if hf_min is not None and micro["fourier_hf"] < float(hf_min):
        return (
            False,
            f"fourier_hf {micro['fourier_hf']:.3f} < {hf_min}",
            micro,
        )
    return True, "fx firm filters ok", micro


def apply_firm_sl(
    signal: str,
    entry: float,
    atr: float,
    symbol: Optional[str],
) -> Optional[float]:
    spec = firm_spec(symbol)
    if spec is None:
        return None
    sl_atr = float(spec["sl_atr"])
    min_risk = float(spec["min_risk_px"])
    a = max(float(atr or 0), 1e-12)
    risk = max(sl_atr * a, min_risk)
    side = str(signal or "").upper()
    if side == "BUY":
        return float(entry) - risk
    if side == "SELL":
        return float(entry) + risk
    return None
