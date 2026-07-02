"""Execute trades — paper (simulated) or live via XM/MT5."""

import logging
from typing import Any

from config import AUTO_TRADE_ENABLED, TRADE_LOT_SIZE, TRADE_MODE, TRADE_UNITS
from strategy.position_sizing import format_lot
from data.oanda import oanda_client

logger = logging.getLogger(__name__)


def can_trade() -> tuple[bool, str]:
    from app.services.trade_settings import get_max_trades_per_minute, is_trading_halted

    if is_trading_halted():
        return False, "Trading halted — resume via trades/min dropdown or stop was triggered"

    if get_max_trades_per_minute() <= 0:
        return False, "Auto-trade paused (0 trades/min)"

    if not AUTO_TRADE_ENABLED:
        return False, "Auto-trade disabled (set AUTO_TRADE_ENABLED=true in .env)"

    if TRADE_MODE == "paper":
        return True, "Paper auto-trade active (simulated, no broker needed)"

    if not oanda_client.configured:
        return False, "Live mode requires broker credentials (XM/MT5 or OANDA)"
    return True, "Live auto-trade active via XM/MT5"


def _has_open_position(direction: str) -> bool:
    """Live mode only — paper mode stacks every setup."""
    if TRADE_MODE == "paper":
        return False
    return oanda_client.has_open_position(direction)


def _execute_paper(sig: dict) -> dict[str, Any]:
    signal = sig["SIGNAL"]
    entry, sl, tp = sig["ENTRY"], sig["SL"], sig["TP"]
    tp1 = sig.get("TP1")
    candle_time = sig.get("CANDLE_TIME", "")

    logger.info(
        "Paper trade: %s %s lot (%s units) @ %s SL=%s TP1=%s TP2=%s",
        signal,
        format_lot(),
        TRADE_UNITS,
        entry,
        sl,
        tp1,
        tp,
    )
    tp_line = f"TP1 {tp1}, TP2 {tp}" if tp1 else f"TP {tp}"
    return {
        "status": "executed",
        "message": f"Paper {signal} {format_lot()} lot — Entry {entry}, SL {sl}, {tp_line}",
        "signal": signal,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "candle_time": candle_time,
        "mode": "paper",
        "units": TRADE_UNITS,
        "lot_size": TRADE_LOT_SIZE,
    }


def execute_signal_trade(sig: dict, force: bool = False) -> dict[str, Any]:
    """Place a trade for BUY/SELL signals (paper or live)."""
    from app.services.trade_settings import is_trading_halted

    if is_trading_halted():
        return {
            "status": "skipped",
            "message": "Trading halted — use trades/min dropdown to resume",
        }

    signal = sig.get("SIGNAL", "NO TRADE")
    if signal not in ("BUY", "SELL"):
        return {"status": "skipped", "message": f"No trade signal ({signal})"}

    entry = sig.get("ENTRY")
    sl = sig.get("SL")
    tp = sig.get("TP")

    if entry is None or sl is None or tp is None:
        return {"status": "skipped", "message": "Signal missing entry/SL/TP"}

    if not force:
        ok, reason = can_trade()
        if not ok:
            return {"status": "skipped", "message": reason}

    if _has_open_position(signal):
        return {
            "status": "skipped",
            "message": f"Already have open {signal} position on broker",
        }

    use_paper = TRADE_MODE == "paper" or (
        force and TRADE_MODE != "live" and not oanda_client.configured
    )

    if use_paper:
        return _execute_paper(sig)

    try:
        candle_time = sig.get("CANDLE_TIME", "")
        response = oanda_client.place_market_order(
            direction=signal,
            sl=float(sl),
            tp=float(tp),
            client_tag=candle_time[:20] if candle_time else None,
        )
        tx = response.get("orderFillTransaction") or response.get(
            "orderCreateTransaction", {}
        )
        return {
            "status": "executed",
            "message": f"Live {signal} order placed on XM",
            "signal": signal,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "candle_time": candle_time,
            "mode": "live",
            "oanda_transaction_id": tx.get("id"),
            "oanda_response": response,
        }
    except Exception as exc:
        logger.exception("Trade execution failed: %s", exc)
        return {"status": "error", "message": str(exc), "signal": signal, "mode": "live"}
