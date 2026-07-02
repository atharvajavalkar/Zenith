"""Win / loss / breakeven classification for closed trades."""

from __future__ import annotations

from typing import Any

# Match trailing_stop breakeven tolerance (~2 cents on gold)
BE_TOLERANCE = 0.02


def _get(row: Any, key: str, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def sl_at_breakeven(
    signal: str | None,
    entry: float | None,
    sl: float | None,
) -> bool:
    """Stop placed essentially at entry (not trailed into profit)."""
    if signal not in ("BUY", "SELL") or entry is None or sl is None:
        return False
    return abs(float(sl) - float(entry)) <= BE_TOLERANCE


def sl_trailed_profit(
    signal: str | None,
    entry: float | None,
    sl: float | None,
) -> bool:
    """Stop trailed beyond entry — counts as a win when stopped out."""
    if signal not in ("BUY", "SELL") or entry is None or sl is None:
        return False
    if sl_at_breakeven(signal, entry, sl):
        return False
    entry_f = float(entry)
    sl_f = float(sl)
    if signal == "BUY":
        return sl_f > entry_f + BE_TOLERANCE
    return sl_f < entry_f - BE_TOLERANCE


def is_breakeven_trade(row: Any) -> bool:
    """Stopped at SL at entry — neither win nor loss."""
    if _get(row, "signal") not in ("BUY", "SELL"):
        return False
    if _get(row, "status") != "closed":
        return False
    if _get(row, "pnl_status") != "closed_sl":
        return False
    return sl_at_breakeven(_get(row, "signal"), _get(row, "entry"), _get(row, "sl"))


def is_winning_trade(row: Any) -> bool:
    if is_breakeven_trade(row):
        return False
    if _get(row, "signal") not in ("BUY", "SELL"):
        return False
    if _get(row, "status") != "closed":
        return False

    pnl = _get(row, "pnl")
    if pnl is not None and float(pnl) > 0:
        return True

    pnl_status = _get(row, "pnl_status")
    if pnl_status == "closed_tp":
        return True

    if pnl_status == "closed_sl" and sl_trailed_profit(
        _get(row, "signal"), _get(row, "entry"), _get(row, "sl")
    ):
        return True

    return False


def is_losing_trade(row: Any) -> bool:
    if is_breakeven_trade(row):
        return False
    if _get(row, "signal") not in ("BUY", "SELL"):
        return False
    if _get(row, "status") != "closed":
        return False

    pnl = _get(row, "pnl")
    if pnl is not None and float(pnl) < 0:
        return True

    if _get(row, "pnl_status") == "closed_sl":
        return not sl_trailed_profit(
            _get(row, "signal"), _get(row, "entry"), _get(row, "sl")
        )

    return False


def outcome_label(row: Any) -> str:
    if is_breakeven_trade(row):
        return "breakeven"
    if is_winning_trade(row):
        return "win"
    if is_losing_trade(row):
        return "loss"
    return "flat"
