"""Symbol-aware price digit helpers (GOLD=2, SILVER=3, US30=1, FX=5, …)."""

from __future__ import annotations

from typing import Any, Optional

from strategy.instrument_meta import price_digits as _digits


def price_digits(symbol: Optional[str] = None) -> int:
    return _digits(symbol)


def round_price(value: Any, symbol: Optional[str] = None) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), price_digits(symbol))
    except (TypeError, ValueError):
        return None
