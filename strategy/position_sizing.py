"""Lot size, P&L, and margin helpers — symbol-aware contract sizes for XM CFDs."""

from __future__ import annotations

from typing import Optional

from config import MARGIN_LEVERAGE, TRADE_LOT_SIZE
from strategy.instrument_meta import contract_units_per_lot as _contract_units


def contract_units_per_lot(symbol: Optional[str] = None) -> float:
    """
    XM CFD contract size per 1.00 lot.
      GOLD    → 100 oz
      SILVER  → 5000 oz
      USOIL   → 100 barrels
      US30    → 1 (point value)
      BTCUSD  → 1 BTC
      FX      → 100_000
    """
    return _contract_units(symbol)


def lot_to_units(lot_size: float | None = None, symbol: Optional[str] = None) -> float:
    lot = TRADE_LOT_SIZE if lot_size is None else float(lot_size)
    return lot * contract_units_per_lot(symbol)


def format_lot(lot_size: float | None = None) -> str:
    lot = TRADE_LOT_SIZE if lot_size is None else float(lot_size)
    return f"{lot:.2f}"


def resolve_lot_size(symbol: Optional[str] = None, capital: Optional[float] = None) -> float:
    """Capital + volatility-aware lot (live XM balance when available)."""
    from strategy.dynamic_lots import resolve_lot_size as _resolve

    return _resolve(symbol, capital)


def leg_pnl(
    signal: str,
    entry: float,
    price: float,
    lot_size: float | None = None,
    symbol: Optional[str] = None,
) -> float:
    direction = 1 if signal == "BUY" else -1
    units = lot_to_units(lot_size, symbol)
    return round((float(price) - float(entry)) * direction * units, 2)


def estimate_margin(
    price: float,
    lot_size: float | None = None,
    leverage: float | None = None,
    symbol: Optional[str] = None,
) -> float:
    """Notional / leverage — approximate margin (kept for settings / legacy)."""
    lev = MARGIN_LEVERAGE if leverage is None else float(leverage)
    units = lot_to_units(lot_size, symbol)
    return round(float(price) * units / lev, 2)
