"""Structure-based take-profit targets (opposing swing / zone / HTF magnet).

Used as the preferred exit brain for GOLD/USOIL after walk-forward showed
structure TP beating fixed 1:2/1:3 grids on OOS expectancy and drawdown.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def structure_take_profit(
    direction: str,
    entry: float,
    sl: float,
    atr: float,
    *,
    zones: Optional[dict] = None,
    swing_high: Optional[float] = None,
    swing_low: Optional[float] = None,
    htf_high: Optional[float] = None,
    htf_low: Optional[float] = None,
    min_rr: float = 1.5,
    max_rr: float = 4.0,
) -> Optional[dict[str, Any]]:
    """Return TP at nearest opposing structure, clamped to [min_rr, max_rr]."""
    risk = abs(float(entry) - float(sl))
    if risk <= 1e-12 or atr <= 0:
        return None

    direction = str(direction).upper()
    candidates: list[float] = []

    if direction == "BUY":
        if zones:
            z = zones.get("s_buf_low")
            if z is not None and float(z) > entry:
                candidates.append(float(z))
        if swing_high is not None and float(swing_high) > entry:
            candidates.append(float(swing_high))
        if htf_high is not None and float(htf_high) > entry:
            candidates.append(float(htf_high))
        # Fallback magnets so we always have a structural stretch option
        candidates.extend([entry + 2.0 * risk, entry + 3.0 * atr])
        above = [c for c in candidates if c > entry]
        if not above:
            return None
        tp = min(above)
        rr = (tp - entry) / risk
        if rr < min_rr:
            tp = entry + min_rr * risk
            rr = min_rr
        if rr > max_rr:
            tp = entry + max_rr * risk
            rr = max_rr
    elif direction == "SELL":
        if zones:
            z = zones.get("d_buf_high")
            if z is not None and float(z) < entry:
                candidates.append(float(z))
        if swing_low is not None and float(swing_low) < entry:
            candidates.append(float(swing_low))
        if htf_low is not None and float(htf_low) < entry:
            candidates.append(float(htf_low))
        candidates.extend([entry - 2.0 * risk, entry - 3.0 * atr])
        below = [c for c in candidates if c < entry]
        if not below:
            return None
        tp = max(below)
        rr = (entry - tp) / risk
        if rr < min_rr:
            tp = entry - min_rr * risk
            rr = min_rr
        if rr > max_rr:
            tp = entry - max_rr * risk
            rr = max_rr
    else:
        return None

    return {
        "tp": float(tp),
        "rr": round(float(rr), 2),
        "source": "structure",
        "min_rr": float(min_rr),
        "max_rr": float(max_rr),
    }


def htf_extremes(df_htf: Optional[pd.DataFrame], lookback: int = 8) -> tuple[Optional[float], Optional[float]]:
    """Recent HTF high/low for structure magnets."""
    if df_htf is None or df_htf.empty or len(df_htf) < 2:
        return None, None
    win = df_htf.tail(max(2, lookback))
    return float(win["high"].max()), float(win["low"].min())


def structure_enabled_for(symbol: Optional[str]) -> bool:
    from config import STRUCTURE_TP_ENABLED, STRUCTURE_TP_SYMBOLS

    if not STRUCTURE_TP_ENABLED:
        return False
    if not symbol:
        return True
    sym = str(symbol).upper().strip()
    if not STRUCTURE_TP_SYMBOLS:
        return True
    return sym in STRUCTURE_TP_SYMBOLS


def resolve_structure_tp(
    direction: str,
    entry: float,
    sl: float,
    atr: float,
    *,
    symbol: Optional[str] = None,
    zones: Optional[dict] = None,
    swing_high: Optional[float] = None,
    swing_low: Optional[float] = None,
    htf_high: Optional[float] = None,
    htf_low: Optional[float] = None,
) -> Optional[float]:
    """Convenience: structure TP price or None when brain disabled / invalid."""
    from config import STRUCTURE_TP_MAX_RR, STRUCTURE_TP_MIN_RR

    if not structure_enabled_for(symbol):
        return None
    pack = structure_take_profit(
        direction,
        entry,
        sl,
        atr,
        zones=zones,
        swing_high=swing_high,
        swing_low=swing_low,
        htf_high=htf_high,
        htf_low=htf_low,
        min_rr=STRUCTURE_TP_MIN_RR,
        max_rr=STRUCTURE_TP_MAX_RR,
    )
    return None if not pack else float(pack["tp"])
