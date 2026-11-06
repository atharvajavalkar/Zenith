"""Execute trades — paper (simulated) or live via XM/MT5 (MetaAPI) or OANDA."""

import logging
from typing import Any, Optional

from config import AUTO_TRADE_ENABLED, BLOCK_OPPOSITE_OPEN, TRADE_LOT_SIZE, TRADE_UNITS
from strategy.position_sizing import format_lot
from data.broker import broker_client, broker_name, order_id_from_response

logger = logging.getLogger(__name__)


def _trade_mode() -> str:
    try:
        import runtime_config
        return runtime_config.get_trade_mode()
    except Exception:
        from config import TRADE_MODE
        return TRADE_MODE


def _block_opposite_open() -> bool:
    try:
        import runtime_config
        return bool(runtime_config.get("block_opposite_open", BLOCK_OPPOSITE_OPEN))
    except Exception:
        return BLOCK_OPPOSITE_OPEN


def can_trade() -> tuple[bool, str]:
    from app.services.trade_settings import get_max_trades_per_minute, is_trading_halted

    if is_trading_halted():
        return False, "Trading halted — resume via trades/min dropdown or stop was triggered"

    if get_max_trades_per_minute() <= 0:
        return False, "Auto-trade paused (0 trades/min)"

    if not AUTO_TRADE_ENABLED:
        return False, "Auto-trade disabled (set AUTO_TRADE_ENABLED=true in .env)"

    mode = _trade_mode()
    if mode == "paper":
        return True, "Paper auto-trade active (simulated, no broker needed)"

    if mode == "demo":
        if not broker_client.configured:
            return False, (
                "Demo mode needs METAAPI_TOKEN + METAAPI_ACCOUNT_ID_DEMO in .env"
            )
        return True, f"XM DEMO auto-trade active via {broker_name.upper()}"

    if mode == "live":
        if not broker_client.configured:
            return False, (
                "Live mode needs METAAPI_TOKEN + METAAPI_ACCOUNT_ID_LIVE in .env"
            )
        return True, f"XM LIVE auto-trade active via {broker_name.upper()}"

    return False, f"Unknown trade mode: {mode}"


def _broker_skip_reason(
    signal: str, symbol: Optional[str] = None, hyper: bool = False
) -> Optional[str]:
    """
    Broker gates for demo/live — paper stacks freely (DB bucket limits only).
    Scoped per symbol when multi-instrument is on.

    Hyper-scalp bypasses the opposite-open block and per-symbol cap — it manages
    its own max_open / cooldown and needs to fire both directions rapidly.
    """
    if _trade_mode() == "paper" or hyper:
        return None

    from app.services.trade_settings import get_max_trades_per_bucket

    opposite = "SELL" if signal == "BUY" else "BUY"
    if _block_opposite_open() and broker_client.has_open_position(opposite, symbol=symbol):
        return f"Open {opposite} on {symbol or broker_name.upper()} — close before {signal}"

    max_per = get_max_trades_per_bucket()
    same_count = broker_client.count_positions(signal, symbol=symbol)
    if same_count >= max_per:
        label = symbol or broker_name.upper()
        return f"{label} {signal} cap ({same_count}/{max_per} open on symbol)"

    return None


def _execute_paper(sig: dict, lot: float) -> dict[str, Any]:
    signal = sig["SIGNAL"]
    entry, sl, tp = sig["ENTRY"], sig["SL"], sig["TP"]
    tp1 = sig.get("TP1")
    candle_time = sig.get("CANDLE_TIME", "")
    sym = str(sig.get("SYMBOL") or sig.get("INSTRUMENT") or "GOLD").upper()

    logger.info(
        "Paper trade: %s %s %.2f lot @ %s SL=%s TP1=%s TP2=%s",
        signal,
        sym,
        lot,
        entry,
        sl,
        tp1,
        tp,
    )
    tp_line = f"TP1 {tp1}, TP2 {tp}" if tp1 else f"TP {tp}"
    return {
        "status": "executed",
        "message": f"Paper {signal} {sym} {lot:.2f} lot — Entry {entry}, SL {sl}, {tp_line}",
        "signal": signal,
        "symbol": sym,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "candle_time": candle_time,
        "mode": "paper",
        "lot_size": lot,
    }


def execute_signal_trade(sig: dict, force: bool = False) -> dict[str, Any]:
    """Place a trade for BUY/SELL signals (paper or live)."""
    from app.services.trade_settings import is_trading_halted
    from strategy.dynamic_lots import sizing_snapshot
    from strategy.instrument_scanner import instrument_tradable

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

    sym = str(sig.get("SYMBOL") or sig.get("INSTRUMENT") or "GOLD").strip().upper()
    if not instrument_tradable(sym):
        return {
            "status": "skipped",
            "message": f"{sym} market closed",
            "market_open": False,
            "symbol": sym,
        }

    if not force:
        ok, reason = can_trade()
        if not ok:
            return {"status": "skipped", "message": reason}

    skip = _broker_skip_reason(signal, sym, hyper=bool(sig.get("HYPER_SCALP")))
    if skip:
        return {"status": "skipped", "message": skip}

    snap = sizing_snapshot(
        sym,
        entry=float(entry),
        sl=float(sl),
        tp=float(tp),
        rr=sig.get("RR"),
    )
    lot = float(snap["lot_size"])
    # Allow signal override only if explicitly set
    if sig.get("LOT_SIZE") is not None:
        try:
            lot = max(0.01, float(sig["LOT_SIZE"]))
        except Exception:
            pass

    mode = _trade_mode()
    use_paper = mode == "paper" or (
        force and mode not in ("demo", "live") and not broker_client.configured
    )
    if mode in ("demo", "live") and not broker_client.configured and not force:
        return {
            "status": "skipped",
            "message": (
                "METAAPI_ACCOUNT_ID_DEMO missing"
                if mode == "demo"
                else "METAAPI_ACCOUNT_ID_LIVE missing — add your live MetaAPI account UUID to .env"
            ),
            "mode": mode,
        }
    if mode in ("demo", "live") and not broker_client.configured and force:
        use_paper = True

    if use_paper:
        out = _execute_paper(sig, lot)
        out["sizing"] = snap
        return out

    try:
        candle_time = sig.get("CANDLE_TIME", "")
        with broker_client.using_symbol(sym):
            response = broker_client.place_market_order(
                direction=signal,
                sl=float(sl),
                tp=float(tp),
                entry=float(entry),
                lot_size=lot,
                tp1=float(sig["TP1"]) if sig.get("TP1") is not None else None,
                tp2=float(sig["TP2"]) if sig.get("TP2") is not None else None,
                tp3=float(sig["TP3"]) if sig.get("TP3") is not None else None,
                client_tag=candle_time[:20] if candle_time else None,
            )
        order_id = order_id_from_response(response)
        if not order_id:
            raise RuntimeError(
                f"XM {sym} order filled but no positionId returned — not logging Zenith-only trade"
            )
        rebased = (response or {}).get("_rebased") or {}
        live_entry = rebased.get("entry", entry)
        live_sl = rebased.get("sl", sl)
        live_tp = rebased.get("tp", tp)
        label = "DEMO" if mode == "demo" else "LIVE"
        return {
            "status": "executed",
            "message": (
                f"{label} {signal} {sym} {lot:.2f} lot on {broker_name.upper()} "
                f"pos {order_id} (TP~${snap.get('expected_profit_usd', 0):.0f} "
                f"/ SL~${snap.get('expected_loss_usd', 0):.0f} · {snap['asset_class']})"
            ),
            "signal": signal,
            "symbol": sym,
            "entry": live_entry,
            "sl": live_sl,
            "tp": live_tp,
            "tp1": rebased.get("tp1", sig.get("TP1")),
            "tp2": rebased.get("tp2", sig.get("TP2")),
            "tp3": rebased.get("tp3", sig.get("TP3")),
            "candle_time": candle_time,
            "mode": mode,
            "broker": broker_name,
            "order_id": order_id,
            "oanda_transaction_id": order_id,
            "broker_response": response,
            "price_rebase_offset": rebased.get("offset"),
            "lot_size": lot,
            "sizing": snap,
            "xm_synced": True,
        }
    except Exception as exc:
        logger.exception("Trade execution failed: %s", exc)
        return {"status": "error", "message": str(exc), "signal": signal, "mode": mode, "symbol": sym}
