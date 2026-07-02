"""Supply and demand zone detection across timeframes."""

from typing import Optional

import pandas as pd

from config import BUFFER


def _zone_width(low: float, high: float) -> float:
    return max(high - low, 0.01)


def _pack_zone(
    demand_tf: str,
    d_low: float,
    d_high: float,
    supply_tf: str,
    s_low: float,
    s_high: float,
) -> dict:
    d_width = _zone_width(d_low, d_high)
    s_width = _zone_width(s_low, s_high)
    return {
        "demand_tf": demand_tf,
        "d_low": d_low,
        "d_high": d_high,
        "d_width": d_width,
        "d_buf_low": d_low - BUFFER * d_width,
        "d_buf_high": d_high + BUFFER * d_width,
        "supply_tf": supply_tf,
        "s_low": s_low,
        "s_high": s_high,
        "s_width": s_width,
        "s_buf_low": s_low - BUFFER * s_width,
        "s_buf_high": s_high + BUFFER * s_width,
    }


def _levels_from_recent(
    recent: pd.DataFrame, atr: float, demand_mult: float, supply_mult: float
) -> tuple[float, float, float, float]:
    d_low = float(recent["low"].min())
    d_high = d_low + atr * demand_mult
    s_high = float(recent["high"].max())
    s_low = s_high - atr * supply_mult
    return d_low, d_high, s_low, s_high


def _pick_demand(candidates: list[tuple[str, float, float]], price: float) -> tuple[str, float, float]:
    """Prefer 1M demand below price; else nearest valid zone."""
    valid = [z for z in candidates if z[2] < price]
    if not valid:
        return candidates[0]
    one_m = [z for z in valid if z[0] == "1M"]
    if one_m:
        return one_m[0]
    return max(valid, key=lambda z: z[2])


def _pick_supply(candidates: list[tuple[str, float, float]], price: float) -> tuple[str, float, float]:
    """Prefer 1M supply above price; else nearest valid zone."""
    valid = [z for z in candidates if z[1] > price]
    if not valid:
        return candidates[0]
    one_m = [z for z in valid if z[0] == "1M"]
    if one_m:
        return one_m[0]
    return min(valid, key=lambda z: z[1])


def _pick_demand_by_tf(
    candidates: list[tuple[str, float, float]], price: float, prefer: tuple[str, ...]
) -> tuple[str, float, float]:
    """Prefer demand below price in TF order (e.g. 5M then 3M)."""
    valid = [z for z in candidates if z[2] < price]
    if not valid:
        return candidates[0]
    for tf in prefer:
        hits = [z for z in valid if z[0] == tf]
        if hits:
            return hits[0]
    return max(valid, key=lambda z: z[2])


def _pick_supply_by_tf(
    candidates: list[tuple[str, float, float]], price: float, prefer: tuple[str, ...]
) -> tuple[str, float, float]:
    """Prefer supply above price in TF order (e.g. 5M then 3M)."""
    valid = [z for z in candidates if z[1] > price]
    if not valid:
        return candidates[0]
    for tf in prefer:
        hits = [z for z in valid if z[0] == tf]
        if hits:
            return hits[0]
    return min(valid, key=lambda z: z[1])


def get_zones(df_15m: pd.DataFrame, df_1h: pd.DataFrame, price: float) -> dict:
    """Macro context zones — 15M and 1H (bias, Graetz tube, major structure)."""
    recent_15 = df_15m.tail(48)
    recent_1h = df_1h.tail(48)

    atr_15 = float(df_15m.iloc[-1]["ATR"])
    atr_1h = float(df_1h.iloc[-1]["ATR"])

    d_low_15, d_high_15, s_low_15, s_high_15 = _levels_from_recent(recent_15, atr_15, 0.50, 0.50)
    d_low_1h, d_high_1h, s_low_1h, s_high_1h = _levels_from_recent(recent_1h, atr_1h, 0.35, 0.35)

    demands = [("15M", d_low_15, d_high_15), ("1H", d_low_1h, d_high_1h)]
    supplies = [("15M", s_low_15, s_high_15), ("1H", s_low_1h, s_high_1h)]

    d_tf, d_low, d_high = _pick_demand(demands, price)
    s_tf, s_low, s_high = _pick_supply(supplies, price)

    return _pack_zone(d_tf, d_low, d_high, s_tf, s_low, s_high)


def get_local_zones(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    price: float,
    df_3m: Optional[pd.DataFrame] = None,
    use_3m5m: bool = False,
) -> dict:
    """
    Micro execution zones.

    Production (use_3m5m): 5M preferred, 3M fallback — matches 5M execution cadence.
    Test mode: 1M preferred, 5M fallback.
    """
    if use_3m5m and df_3m is not None and len(df_3m) > 0:
        recent_3m = df_3m.tail(30)
        recent_5m = df_5m.tail(24)
        atr_3m = float(df_3m.iloc[-1]["ATR"])
        atr_5m = float(df_5m.iloc[-1]["ATR"])
        d_low_3m, d_high_3m, s_low_3m, s_high_3m = _levels_from_recent(
            recent_3m, atr_3m, 0.32, 0.32
        )
        d_low_5m, d_high_5m, s_low_5m, s_high_5m = _levels_from_recent(
            recent_5m, atr_5m, 0.35, 0.35
        )
        demands = [("5M", d_low_5m, d_high_5m), ("3M", d_low_3m, d_high_3m)]
        supplies = [("5M", s_low_5m, s_high_5m), ("3M", s_low_3m, s_high_3m)]
        prefer = ("5M", "3M")
        d_tf, d_low, d_high = _pick_demand_by_tf(demands, price, prefer)
        s_tf, s_low, s_high = _pick_supply_by_tf(supplies, price, prefer)
        return _pack_zone(d_tf, d_low, d_high, s_tf, s_low, s_high)

    recent_1m = df_1m.tail(36)
    recent_5m = df_5m.tail(24)
    atr_1m = float(df_1m.iloc[-1]["ATR"])
    atr_5m = float(df_5m.iloc[-1]["ATR"])
    d_low_1m, d_high_1m, s_low_1m, s_high_1m = _levels_from_recent(recent_1m, atr_1m, 0.30, 0.30)
    d_low_5m, d_high_5m, s_low_5m, s_high_5m = _levels_from_recent(recent_5m, atr_5m, 0.35, 0.35)
    demands = [("1M", d_low_1m, d_high_1m), ("5M", d_low_5m, d_high_5m)]
    supplies = [("1M", s_low_1m, s_high_1m), ("5M", s_low_5m, s_high_5m)]
    d_tf, d_low, d_high = _pick_demand(demands, price)
    s_tf, s_low, s_high = _pick_supply(supplies, price)
    return _pack_zone(d_tf, d_low, d_high, s_tf, s_low, s_high)


def nearest_opposing_liquidity(direction: str, entry: float, zones: dict) -> Optional[float]:
    """Nearest micro liquidity target for TP."""
    if direction == "BUY":
        liq = zones["s_buf_low"]
        return liq if liq > entry else None
    liq = zones["d_buf_high"]
    return liq if liq < entry else None
