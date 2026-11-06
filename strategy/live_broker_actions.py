"""
Execute live XM trade-management actions via MetaAPI.

Drives partial closes, breakeven SL, staged SL locks, and dynamic trailing
to match paper/backtest logic in trade_simulator.update_trade_on_bar().
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config import TRADE_LOT_SIZE
from strategy.trade_simulator import OPEN_STATUSES, rr_int, tp_fracs

logger = logging.getLogger(__name__)

# Minimum SL move before sending POSITION_MODIFY for trailing
def _sl_modify_min_delta(sl_value: Optional[float]) -> float:
    try:
        v = abs(float(sl_value or 0))
    except Exception:
        v = 0.0
    if v >= 50:  # metals / indices
        return 0.05
    if v >= 20:  # JPY pairs
        return 0.005
    return 0.00005  # major FX


def is_live_managed_row(row: Any) -> bool:
    """True when this open trade should be managed on the XM broker."""
    mode = getattr(row, "trade_mode", None)
    pos_id = getattr(row, "oanda_transaction_id", None)
    return mode in ("live", "demo") and bool(pos_id)


def row_to_trade_dict(row: Any) -> dict[str, Any]:
    """TradeRecord → dict for update_trade_on_bar()."""
    sym = getattr(row, "symbol", None)
    feats: Any = getattr(row, "features", None)
    if isinstance(feats, str):
        try:
            import json

            feats = json.loads(feats)
        except Exception:
            feats = {}
    if not isinstance(feats, dict):
        feats = {}
    if not sym:
        sym = feats.get("symbol") or feats.get("SYMBOL")
    entry_lot = feats.get("entry_lot") or feats.get("lot_size") or row.lot_size
    try:
        entry_lot = float(entry_lot or TRADE_LOT_SIZE)
    except Exception:
        entry_lot = float(TRADE_LOT_SIZE)
    return {
        "signal": row.signal,
        "symbol": (str(sym).upper() if sym else "GOLD"),
        "trail_lock": feats.get("trail_lock"),
        "entry": row.entry,
        "sl": row.sl,
        "tp": row.tp,
        "tp1": row.tp1,
        "tp2": row.tp2,
        "tp3": row.tp3,
        "rr_type": row.rr_type,
        "lot_size": row.lot_size or TRADE_LOT_SIZE,
        "entry_lot": entry_lot,
        "pnl_status": row.pnl_status or "open",
        "partial_pnl": row.partial_pnl or 0.0,
        "extreme_price": row.extreme_price,
        "pnl": row.pnl,
        "status": row.status,
        "message": row.message or "",
        "sl_trail_status": row.sl_trail_status,
    }


def apply_trade_dict_to_row(row: Any, trade: dict[str, Any], closed: bool) -> None:
    """Copy managed fields from trade dict back onto TradeRecord."""
    row.sl = trade.get("sl")
    row.tp = trade.get("tp")
    row.pnl_status = trade.get("pnl_status")
    row.partial_pnl = trade.get("partial_pnl")
    row.extreme_price = trade.get("extreme_price")
    row.pnl = trade.get("pnl")
    row.message = trade.get("message") or row.message
    row.sl_trail_status = trade.get("sl_trail_status")
    if closed:
        row.status = "closed"


def _partial_volume(lot: float, frac: float, broker_client: Any) -> float:
    vol = lot * frac
    if hasattr(broker_client, "round_volume"):
        return broker_client.round_volume(vol)
    return round(max(vol, 0.01), 2)


def _sl_changed(before: Optional[float], after: Optional[float]) -> bool:
    if before is None or after is None:
        return after is not None and before != after
    return abs(float(after) - float(before)) >= _sl_modify_min_delta(after)


_STAGE_ORDER = ("open", "partial", "partial2", "partial3")


def _stage_idx(status: Optional[str]) -> int:
    try:
        return _STAGE_ORDER.index(status or "open")
    except ValueError:
        return 0


def _tp_level_for_transition(before_status: str, after: dict[str, Any]) -> Optional[float]:
    """TP price that must be live-confirmed before booking a partial."""
    if before_status == "open":
        return after.get("tp1")
    if before_status == "partial":
        return after.get("tp2")
    if before_status == "partial2":
        return after.get("tp3")
    return None


def _live_price_confirms_tp(
    broker_client: Any,
    *,
    symbol: str,
    signal: str,
    tp_level: float,
) -> bool:
    """Reject phantom partials when the tick that triggered management already reversed."""
    try:
        from data.market_data import get_tick_quote

        q = get_tick_quote(symbol)
        px = q.get("price")
        if px is None:
            return False
        px = float(px)
        lv = float(tp_level)
        # Small fill slip only — never book TP while mark is still on the wrong side
        slip = max(abs(lv) * 0.00015, 0.15 if "ETH" in symbol.upper() else 0.01)
        if "BTC" in symbol.upper():
            slip = max(slip, 2.0)
        if signal == "BUY":
            return px >= lv - slip
        return px <= lv + slip
    except Exception as exc:
        logger.warning("TP confirm quote failed %s: %s", symbol, exc)
        return False


def _revert_after(before: dict[str, Any], after: dict[str, Any]) -> None:
    after["pnl_status"] = before.get("pnl_status")
    after["partial_pnl"] = before.get("partial_pnl")
    after["sl"] = before.get("sl")
    after["extreme_price"] = before.get("extreme_price")
    after["message"] = before.get("message")
    after["sl_trail_status"] = before.get("sl_trail_status")
    after["pnl"] = before.get("pnl")
    after["status"] = before.get("status", "executed")
    if before.get("exit_price") is None:
        after.pop("exit_price", None)


def apply_live_broker_actions(
    position_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    """
    Apply broker orders for state transitions computed by update_trade_on_bar().

    Returns True if DB state in `after` should be committed; False to retry next tick.
    On partial failure, reverts critical fields in `after` from `before`.
    """
    from data.broker import broker_client

    if not broker_client.configured:
        logger.warning("Broker not configured — skipping live actions for %s", position_id)
        return False

    before_status = before.get("pnl_status") or "open"
    after_status = after.get("pnl_status") or "open"
    # One stage per tick — same-bar OHLC can cascade open→partial3; live must not.
    if (
        before_status in _STAGE_ORDER
        and after_status in _STAGE_ORDER
        and _stage_idx(after_status) > _stage_idx(before_status) + 1
    ):
        logger.info(
            "Reject multi-stage live transition %s→%s on pos %s (retry next tick)",
            before_status,
            after_status,
            position_id,
        )
        _revert_after(before, after)
        return False
    # Partials must use original entry lot — lot_size may already be remaining after sync
    lot = float(
        after.get("entry_lot")
        or before.get("entry_lot")
        or after.get("lot_size")
        or TRADE_LOT_SIZE
    )
    rr_n = rr_int(after.get("rr_type"))
    sym = str(after.get("symbol") or after.get("SYMBOL") or "GOLD")
    signal = str(after.get("signal") or before.get("signal") or "")
    fracs = tp_fracs(rr_n, sym)

    try:
        # --- Partial booking + staged SL locks ---
        if before_status == "open" and after_status == "partial":
            tp_lv = _tp_level_for_transition(before_status, after)
            if tp_lv is None or not _live_price_confirms_tp(
                broker_client, symbol=sym, signal=signal, tp_level=float(tp_lv)
            ):
                logger.warning(
                    "Skip TP1 partial pos %s — mark not at TP1 (%.2f)",
                    position_id,
                    float(tp_lv or 0),
                )
                _revert_after(before, after)
                return False
            vol = _partial_volume(lot, fracs[0], broker_client)
            broker_client.partial_close(position_id, vol)
            broker_client.modify_position(
                position_id,
                stop_loss=after.get("sl"),
                take_profit=after.get("tp"),
            )
            logger.info("Live TP1 partial %.2f lot on pos %s", vol, position_id)

        elif before_status == "partial" and after_status == "partial2":
            tp_lv = _tp_level_for_transition(before_status, after)
            if tp_lv is None or not _live_price_confirms_tp(
                broker_client, symbol=sym, signal=signal, tp_level=float(tp_lv)
            ):
                logger.warning(
                    "Skip TP2 partial pos %s — mark not at TP2 (%.2f)",
                    position_id,
                    float(tp_lv or 0),
                )
                _revert_after(before, after)
                return False
            vol = _partial_volume(lot, fracs[1], broker_client)
            broker_client.partial_close(position_id, vol)
            broker_client.modify_position(position_id, stop_loss=after.get("sl"))
            logger.info("Live TP2 partial %.2f lot on pos %s", vol, position_id)

        elif before_status == "partial2" and after_status == "partial3":
            tp_lv = _tp_level_for_transition(before_status, after)
            if tp_lv is None or not _live_price_confirms_tp(
                broker_client, symbol=sym, signal=signal, tp_level=float(tp_lv)
            ):
                logger.warning(
                    "Skip TP3 partial pos %s — mark not at TP3 (%.2f)",
                    position_id,
                    float(tp_lv or 0),
                )
                _revert_after(before, after)
                return False
            vol = _partial_volume(lot, fracs[2], broker_client)
            broker_client.partial_close(position_id, vol)
            broker_client.modify_position(position_id, stop_loss=after.get("sl"))
            logger.info("Live TP3 partial %.2f lot on pos %s", vol, position_id)

        elif (
            before_status == after_status
            and after_status in ("partial", "partial2", "partial3")
            and _sl_changed(before.get("sl"), after.get("sl"))
        ):
            broker_client.modify_position(position_id, stop_loss=after.get("sl"))
            logger.info(
                "Live trail SL→%.2f on pos %s",
                float(after.get("sl") or 0),
                position_id,
            )

        elif after_status in ("closed_sl", "closed_tp", "closed_manual"):
            if hasattr(broker_client, "position_exists") and broker_client.position_exists(position_id):
                broker_client.close_position(position_id)
                logger.info("Live close pos %s (%s)", position_id, after_status)

        return True

    except Exception as exc:
        logger.error(
            "Live broker action failed pos=%s %s→%s: %s",
            position_id,
            before_status,
            after_status,
            exc,
        )
        # Revert state so we retry on the next tick
        _revert_after(before, after)
        return False
