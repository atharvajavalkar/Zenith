"""Lot size, P&L, and margin helpers for XAUUSD."""

from config import MARGIN_LEVERAGE, TRADE_LOT_SIZE, UNITS_PER_LOT


def lot_to_units(lot_size: float | None = None) -> float:
    lot = TRADE_LOT_SIZE if lot_size is None else float(lot_size)
    return lot * UNITS_PER_LOT


def format_lot(lot_size: float | None = None) -> str:
    lot = TRADE_LOT_SIZE if lot_size is None else float(lot_size)
    return f"{lot:.2f}"


def leg_pnl(signal: str, entry: float, price: float, lot_size: float | None = None) -> float:
    direction = 1 if signal == "BUY" else -1
    units = lot_to_units(lot_size)
    return round((float(price) - float(entry)) * direction * units, 2)


def estimate_margin(price: float, lot_size: float | None = None, leverage: float | None = None) -> float:
    """Notional / leverage — approximate margin required to hold the position."""
    lev = MARGIN_LEVERAGE if leverage is None else float(leverage)
    units = lot_to_units(lot_size)
    return round(float(price) * units / lev, 2)
