"""Trade execution API."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

_ALGO_ROOT = Path(__file__).resolve().parents[3]
if str(_ALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGO_ROOT))

from config import (  # noqa: E402
    AUTO_TRADE_ENABLED,
    BIAS_GATE_ENABLED,
    BLOCK_OPPOSITE_OPEN,
    DIRECTION_LOCK_5M,
    MAX_TRADES_PER_5M,
    ML_GATE_ENABLED,
    MARGIN_LEVERAGE,
    TRADE_LOT_SIZE,
    TRADE_MODE,
    TRADE_UNITS,
    UNITS_PER_LOT,
)
from strategy.position_sizing import estimate_margin  # noqa: E402
from strategy.trade_executor import can_trade, execute_signal_trade  # noqa: E402
from strategy.trailing_stop import sl_trail_label  # noqa: E402

from app.core.database import get_session
from app.core.security import get_current_user
from app.models.trade_record import TradeRecord
from app.models.user import User
from app.services.market_cache import market_cache
from app.services.trade_settings import (
    bucket_log_time,
    get_exec_cadence,
    get_max_trades_per_bucket,
    get_max_trades_per_minute,
    get_trade_bucket_seconds,
    get_trade_cadence,
    get_trade_interval_seconds,
    get_zone_timeframes,
    is_trading_halted,
    set_exec_cadence,
    set_max_trades_per_bucket,
)
from app.services.trade_calibrator import trade_calibrator
from app.services.trade_close import close_all_open_trades
from app.services.trade_stats import get_trade_stats
from app.services.ws_manager import ws_manager
from pydantic import BaseModel, Field
from data.fetcher import get_tick_quote  # noqa: E402

router = APIRouter(prefix="/api/trades", tags=["trades"])


def _architecture_profile() -> dict:
    cadence = get_trade_cadence()
    return {
        "mode": cadence,
        "exec_cadence": cadence,
        "zone_timeframes": get_zone_timeframes(),
        "entry_source": f"{get_zone_timeframes()} supply & demand zones",
        "max_trades_per_5m": MAX_TRADES_PER_5M if cadence == "5m" else None,
        "trade_bucket_seconds": get_trade_bucket_seconds(),
        "ml_gate_enabled": ML_GATE_ENABLED,
        "bias_gate_enabled": BIAS_GATE_ENABLED,
        "direction_lock_5m": DIRECTION_LOCK_5M,
        "trade_5m_mode": cadence in ("3m", "5m"),
        "block_opposite_open": BLOCK_OPPOSITE_OPEN,
    }


class TradeSettingsUpdate(BaseModel):
    max_trades_per_minute: Optional[int] = Field(default=None, ge=0, le=10)
    exec_cadence: Optional[str] = None


@router.get("/gate-check")
async def gate_check(_: User = Depends(get_current_user)):
    """Live gate evaluation for the current signal — MC, ML prob, pass/fail reason."""
    from analysis.adaptive_guard import get_state
    from strategy.signal import build_stacked_trade

    sig = await market_cache.get_signal()
    if sig is None:
        return {"available": False, "reason": "Signal not available yet"}

    # Always return the raw MC from the signal regardless of direction
    mc_raw = sig.get("MARKOV_MC_PROB") or sig.get("SUCCESS_PERCENT")
    static_mc = sig.get("STATIC_MC_PROB")
    mc_delta = sig.get("MARKOV_MC_DELTA")
    signal_dir = sig.get("SIGNAL", "NO TRADE")
    adaptive = get_state()

    # Only run full calibrator eval when there's an actionable setup
    allowed = None
    ml_prob = None
    gate_reason = None
    features = {}

    if signal_dir in ("BUY", "SELL") and sig.get("ENTRY") is not None:
        try:
            quote = await asyncio.to_thread(get_tick_quote)
            live_px = float(quote.get("price") or sig.get("LIVE_PRICE") or 0)
            trade_sig = await asyncio.to_thread(build_stacked_trade, sig, live_px)
            allowed, ml_prob, gate_reason, features = await trade_calibrator.evaluate(sig, trade_sig)
        except Exception as exc:
            gate_reason = f"Eval error: {exc}"

    return {
        "available": True,
        "signal": signal_dir,
        "state": sig.get("STATE"),
        "mc_prob": round(mc_raw, 2) if mc_raw is not None else None,
        "static_mc_prob": round(static_mc, 2) if static_mc is not None else None,
        "mc_delta": round(mc_delta, 2) if mc_delta is not None else None,
        "ml_prob": round(ml_prob, 3) if ml_prob is not None else None,
        "allowed": allowed,
        "gate_reason": gate_reason,
        "mc_floor": adaptive.get("mc_floor"),
        "min_ml_prob": adaptive.get("min_ml_prob"),
        "adaptive_mode": adaptive.get("mode"),
        "entry": sig.get("ENTRY"),
        "sl": sig.get("SL"),
        "tp": sig.get("TP"),
    }


@router.get("/status")
async def trade_status(_: User = Depends(get_current_user)):
    ok, reason = can_trade()
    open_positions = []
    oanda_configured = False
    live_price = None
    try:
        from data.oanda import oanda_client

        oanda_configured = oanda_client.configured
        if oanda_configured:
            open_positions = await asyncio.to_thread(oanda_client.get_open_positions)
    except Exception as exc:
        return {
            "auto_trade_enabled": AUTO_TRADE_ENABLED,
            "oanda_configured": oanda_configured,
            "can_trade": False,
            "reason": str(exc),
            "open_positions": [],
        }

    try:
        quote = await asyncio.to_thread(get_tick_quote)
        live_price = quote.get("price")
    except Exception:
        pass

    margin_usd = estimate_margin(live_price, TRADE_LOT_SIZE) if live_price else None

    return {
        "auto_trade_enabled": AUTO_TRADE_ENABLED,
        "trade_mode": TRADE_MODE,
        "lot_size": TRADE_LOT_SIZE,
        "trade_units": TRADE_UNITS,
        "units_per_lot": UNITS_PER_LOT,
        "margin_leverage": MARGIN_LEVERAGE,
        "margin_usd": margin_usd,
        "max_trades_per_minute": get_max_trades_per_minute(),
        "max_trades_per_bucket": get_max_trades_per_bucket(),
        "exec_cadence": get_exec_cadence(),
        "trade_cadence": get_trade_cadence(),
        "zone_timeframes": get_zone_timeframes(),
        "trade_bucket_seconds": get_trade_bucket_seconds(),
        "trade_interval_seconds": get_trade_interval_seconds(),
        "trading_halted": is_trading_halted(),
        "architecture": _architecture_profile(),
        "oanda_configured": oanda_configured,
        "can_trade": ok,
        "reason": reason,
        "open_positions": open_positions,
    }


@router.get("/settings")
async def get_trade_settings(_: User = Depends(get_current_user)):
    n = get_max_trades_per_bucket()
    cadence = get_trade_cadence()
    return {
        "max_trades_per_minute": n,
        "max_trades_per_bucket": n,
        "exec_cadence": cadence,
        "trade_cadence": cadence,
        "zone_timeframes": get_zone_timeframes(),
        "trade_bucket_seconds": get_trade_bucket_seconds(),
        "trade_interval_seconds": get_trade_interval_seconds(),
        "auto_trade_enabled": AUTO_TRADE_ENABLED,
        "trading_halted": is_trading_halted(),
        "architecture": _architecture_profile(),
    }


@router.patch("/settings")
async def update_trade_settings(
    body: TradeSettingsUpdate,
    _: User = Depends(get_current_user),
):
    if body.exec_cadence is not None:
        cadence = body.exec_cadence.lower().strip()
        if cadence not in ("1m", "3m", "5m"):
            raise HTTPException(status_code=400, detail="exec_cadence must be 1m, 3m, or 5m")
        set_exec_cadence(cadence)

    if body.max_trades_per_minute is not None:
        set_max_trades_per_bucket(body.max_trades_per_minute)

    n = get_max_trades_per_bucket()
    interval = get_trade_interval_seconds()
    cadence = get_trade_cadence()
    cadence_min = cadence.replace("m", "")
    return {
        "max_trades_per_minute": n,
        "max_trades_per_bucket": n,
        "exec_cadence": cadence,
        "trade_cadence": cadence,
        "zone_timeframes": get_zone_timeframes(),
        "trade_interval_seconds": interval,
        "message": (
            "No trades"
            if n == 0
            else f"{cadence_min} min · up to {n} trade{'s' if n != 1 else ''} per candle"
        ),
    }


@router.get("/summary")
async def trade_summary(
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    return await get_trade_stats(session)


@router.get("/learning")
async def trade_learning(_: User = Depends(get_current_user)):
    """ML patterns + calibration insights from all historical trades."""
    return await trade_calibrator.get_insights()


@router.get("/history")
async def trade_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(15, ge=1, le=100),
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    total = await session.scalar(select(func.count()).select_from(TradeRecord)) or 0
    total_pages = max(1, (total + per_page - 1) // per_page) if total else 1
    page = min(page, total_pages)
    offset = (page - 1) * per_page

    rows = (
        await session.execute(
            select(TradeRecord)
            .order_by(TradeRecord.id.desc())
            .offset(offset)
            .limit(per_page)
        )
    ).scalars().all()
    return {
        "items": [r.to_dict() for r in rows],
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
    }


@router.post("/execute")
async def execute_trade(
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Manually execute the current live signal — TP1/trail/TP2 tracking fully active."""
    sig = await market_cache.get_signal()
    if sig is None:
        raise HTTPException(status_code=503, detail="No signal available yet")

    result = await asyncio.to_thread(execute_signal_trade, sig, True)

    bucket_log = bucket_log_time(sig, get_exec_cadence())

    signal = sig.get("SIGNAL", "NO TRADE")
    entry = sig.get("ENTRY")
    sl = sig.get("SL")
    tp = sig.get("TP")
    tp1 = sig.get("TP1")
    live_px = sig.get("LOG_PRICE_1M") or sig.get("LIVE_PRICE")
    lot = TRADE_LOT_SIZE
    margin = estimate_margin(float(entry), lot) if entry is not None else None

    from strategy.position_sizing import leg_pnl
    initial_pnl = leg_pnl(signal, entry, float(live_px), lot) if entry and live_px else 0.0

    trail_label = (
        sl_trail_label(signal, float(entry), float(sl), "open")
        if entry and sl else "Fixed SL"
    )

    record = TradeRecord(
        candle_time=sig.get("CANDLE_TIME", ""),
        log_time=bucket_log,
        signal=signal,
        entry=entry,
        sl=sl,
        tp=tp,
        tp1=tp1,
        tp2=sig.get("TP2"),
        tp3=sig.get("TP3"),
        rr_type=sig.get("RR"),
        current_price=live_px,
        pnl=initial_pnl,
        partial_pnl=0.0,
        lot_size=lot,
        margin_usd=margin,
        sl_trail_status=trail_label,
        pnl_status="open" if result["status"] == "executed" else None,
        status=result["status"],
        message=result["message"],
        oanda_transaction_id=result.get("oanda_transaction_id"),
    )
    session.add(record)
    await session.flush()

    return {"trade": record.to_dict(), "result": result}


@router.post("/stop-all")
async def stop_all_trades(
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Halt execution and close every open trade at the current market price."""
    from app.services.trade_settings import halt_all_trading

    quote = await asyncio.to_thread(get_tick_quote)
    exit_price = quote.get("price")
    if exit_price is None:
        raise HTTPException(status_code=503, detail="No live price available")

    if TRADE_MODE == "live":
        try:
            from data.oanda import oanda_client

            if oanda_client.configured:
                await asyncio.to_thread(oanda_client.close_all_positions)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Failed to close OANDA positions: {exc}"
            ) from exc

    closed_count = await close_all_open_trades(session, float(exit_price))
    halt_all_trading()

    if closed_count:
        await ws_manager.broadcast({"type": "trade_closed"})
        from app.services.learning_loop import on_trade_closed

        await on_trade_closed()

    stats = await get_trade_stats(session)
    return {
        "closed_count": closed_count,
        "exit_price": float(exit_price),
        "max_trades_per_minute": 0,
        "trading_halted": True,
        "open_trades": stats["open_trades"],
        "message": f"Stopped — closed {closed_count} open trade(s) at {float(exit_price):.2f}",
    }
