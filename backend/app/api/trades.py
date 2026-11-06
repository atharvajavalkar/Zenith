"""Trade execution API."""

import asyncio
import json
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
    trade_mode: Optional[str] = None


@router.get("/gate-check")
async def gate_check(_: User = Depends(get_current_user)):
    """Live gate evaluation for the current signal — MC, ML prob, pass/fail reason."""
    from analysis.adaptive_guard import get_state
    from analysis.instrument_rl import mc_floor_for, min_ml_for, overlays_for
    from config import INSTRUMENT_UNIVERSE
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
    desk_floor = float(adaptive.get("mc_floor") or 42.0)
    sym = str(sig.get("SYMBOL") or sig.get("INSTRUMENT") or "GOLD").upper()

    # Per-instrument effective floor = desk adaptive + RL/minute overlays
    try:
        symbol_mc_floor = float(mc_floor_for(sym, base=desk_floor))
        symbol_ml_min = float(min_ml_for(sym, base=adaptive.get("min_ml_prob")))
        ov = overlays_for(sym)
    except Exception:
        symbol_mc_floor = desk_floor
        symbol_ml_min = float(adaptive.get("min_ml_prob") or 0.5)
        ov = {}

    instrument_floors = {}
    try:
        for s in INSTRUMENT_UNIVERSE:
            instrument_floors[s] = {
                "mc_floor": round(float(mc_floor_for(s, base=desk_floor)), 1),
                "mc_delta": round(float(overlays_for(s).get("mc_floor_delta") or 0), 1),
                "min_ml_prob": round(float(min_ml_for(s, base=adaptive.get("min_ml_prob"))), 3),
            }
    except Exception:
        instrument_floors = {}

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
            # Ensure symbol is on trade_sig for per-pair floors
            trade_sig = dict(trade_sig or {})
            trade_sig.setdefault("SYMBOL", sym)
            allowed, ml_prob, gate_reason, features = await trade_calibrator.evaluate(sig, trade_sig)
        except Exception as exc:
            gate_reason = f"Eval error: {exc}"

    return {
        "available": True,
        "signal": signal_dir,
        "symbol": sym,
        "state": sig.get("STATE"),
        "mc_prob": round(mc_raw, 2) if mc_raw is not None else None,
        "static_mc_prob": round(static_mc, 2) if static_mc is not None else None,
        "mc_delta": round(mc_delta, 2) if mc_delta is not None else None,
        "ml_prob": round(ml_prob, 3) if ml_prob is not None else None,
        "allowed": allowed,
        "gate_reason": gate_reason,
        # Desk-wide adaptive base (protect mode) — same for everyone
        "desk_mc_floor": round(desk_floor, 1),
        "mc_floor": round(symbol_mc_floor, 1),  # effective for this symbol
        "min_ml_prob": round(symbol_ml_min, 3),
        "desk_min_ml_prob": adaptive.get("min_ml_prob"),
        "rl_mc_delta": ov.get("mc_floor_delta"),
        "instrument_floors": instrument_floors,
        "adaptive_mode": adaptive.get("mode"),
        "concept_quality": sig.get("CONCEPT_QUALITY"),
        "concept_reason": sig.get("CONCEPT_REASON"),
        "concepts": sig.get("CONCEPTS"),
        "entry": sig.get("ENTRY"),
        "sl": sig.get("SL"),
        "tp": sig.get("TP"),
    }


@router.get("/status")
async def trade_status(_: User = Depends(get_current_user)):
    from strategy.instrument_meta import get_desk_market_status

    market = get_desk_market_status()
    ok, reason = can_trade()
    open_positions = []
    broker_configured = False
    live_price = None
    auth = {"ok": True, "state": "ok", "message": ""}
    try:
        from data.broker import broker_client, broker_name

        broker_configured = broker_client.configured
        if hasattr(broker_client, "auth_status"):
            auth = broker_client.auth_status()
        if broker_configured and auth.get("ok"):
            open_positions = await asyncio.to_thread(broker_client.get_open_positions)
    except Exception as exc:
        return {
            "auto_trade_enabled": AUTO_TRADE_ENABLED,
            "broker": broker_name,
            "broker_configured": broker_configured,
            "oanda_configured": broker_configured,  # legacy alias
            "can_trade": False,
            "reason": str(exc),
            "open_positions": [],
            "market_open": market["is_open"],
            "market_status": market["status"],
            "market_label": market["label"],
            "market_message": market["message"],
            "market_next_open": market.get("next_open_display"),
            "market_server_time": market.get("server_time_display"),
            "open_symbols": market.get("open_symbols") or [],
            "crypto_symbols": market.get("crypto_symbols") or [],
        }

    try:
        quote = await asyncio.to_thread(get_tick_quote)
        live_price = quote.get("price")
    except Exception:
        pass

    margin_usd = None

    import runtime_config
    trade_mode = runtime_config.get_trade_mode()

    from config import MT5_SYMBOL
    from data.market_data import get_source_info
    from strategy.dynamic_lots import sizing_snapshot

    source_info = get_source_info()
    broker_account = None
    if runtime_config.is_broker_mode(trade_mode) and broker_configured:
        from data.broker import broker_account_summary
        try:
            broker_account = await asyncio.to_thread(broker_account_summary)
        except Exception:
            broker_account = None

    snap = await asyncio.to_thread(sizing_snapshot, MT5_SYMBOL)
    lot_size = float(snap.get("lot_size") or runtime_config.get("trade_lot_size", TRADE_LOT_SIZE))
    dynamic_on = bool(runtime_config.get("dynamic_lots_enabled", True))
    if live_price:
        margin_usd = estimate_margin(live_price, lot_size, symbol=MT5_SYMBOL)

    accounts = {}
    try:
        if hasattr(broker_client, "account_ids_available"):
            accounts = broker_client.account_ids_available()
    except Exception:
        accounts = {}

    return {
        "auto_trade_enabled": AUTO_TRADE_ENABLED,
        "trade_mode": trade_mode,
        "broker_accounts": accounts,
        "data_source": source_info["data_source"],
        "data_source_label": source_info["label"],
        "symbol": source_info["symbol"],
        "broker_symbol": MT5_SYMBOL,
        "lot_size": lot_size,
        "dynamic_lots": dynamic_on,
        "sizing_note": (
            f"${snap.get('capital', 0):,.0f} · {snap.get('asset_class', 'gold')}"
            if dynamic_on
            else None
        ),
        "sizing": snap if dynamic_on else None,
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
        "broker": broker_name,
        "broker_configured": broker_configured,
        "broker_auth": auth.get("state"),
        "broker_auth_ok": bool(auth.get("ok")),
        "broker_auth_message": auth.get("message"),
        "oanda_configured": broker_configured,  # legacy alias
        "can_trade": ok and bool(auth.get("ok", True)),
        "reason": auth.get("message") if auth.get("ok") is False else reason,
        "open_positions": open_positions,
        "broker_account": broker_account,
        "market_open": market["is_open"],
        "market_status": market["status"],
        "market_label": market["label"],
        "market_message": market["message"],
        "market_next_open": market.get("next_open_display"),
        "market_server_time": market.get("server_time_display"),
        "open_symbols": market.get("open_symbols") or [],
        "crypto_symbols": market.get("crypto_symbols") or [],
    }


@router.get("/broker-test")
async def broker_test(_: User = Depends(get_current_user)):
    """Verify the live broker connection — returns account balance/equity if reachable."""
    from data.broker import broker_client, broker_name

    if not broker_client.configured:
        return {
            "broker": broker_name,
            "configured": False,
            "ok": False,
            "message": (
                f"{broker_name.upper()} not configured for current mode. "
                "Set METAAPI_TOKEN + METAAPI_ACCOUNT_ID_DEMO / METAAPI_ACCOUNT_ID_LIVE in .env, then restart."
            ),
        }

    try:
        info = {}
        # MetaAPI exposes account-information; OANDA does not (skip gracefully)
        if hasattr(broker_client, "account_information"):
            info = await asyncio.to_thread(broker_client.account_information)
        positions = await asyncio.to_thread(broker_client.get_open_positions)
        return {
            "broker": broker_name,
            "configured": True,
            "ok": True,
            "message": f"Connected to {broker_name.upper()}",
            "account": {
                "balance": info.get("balance"),
                "equity": info.get("equity"),
                "currency": info.get("currency"),
                "leverage": info.get("leverage"),
                "server": info.get("server"),
                "name": info.get("name"),
            },
            "symbol": getattr(broker_client, "symbol", None),
            "open_positions": len(positions),
        }
    except Exception as exc:
        return {
            "broker": broker_name,
            "configured": True,
            "ok": False,
            "message": f"Connection failed: {exc}",
        }


@router.get("/settings")
async def get_trade_settings(_: User = Depends(get_current_user)):
    import runtime_config

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
        "trade_mode": runtime_config.get_trade_mode(),
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

    if body.trade_mode is not None:
        import runtime_config
        try:
            runtime_config.set_trade_mode(body.trade_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Force bias + next signal cycle onto the new feed
        try:
            from app.services.bias_cache import bias_cache

            await bias_cache.invalidate()
        except Exception:
            pass
        try:
            from app.services.market_cache import market_cache

            await market_cache.set_error("data source switched — refreshing…")
        except Exception:
            pass

    import runtime_config
    n = get_max_trades_per_bucket()
    interval = get_trade_interval_seconds()
    cadence = get_trade_cadence()
    cadence_min = cadence.replace("m", "")
    trade_mode = runtime_config.get_trade_mode()
    return {
        "max_trades_per_minute": n,
        "max_trades_per_bucket": n,
        "exec_cadence": cadence,
        "trade_cadence": cadence,
        "zone_timeframes": get_zone_timeframes(),
        "trade_interval_seconds": interval,
        "trade_mode": trade_mode,
        "message": (
            f"{trade_mode.upper()} · no trades"
            if n == 0
            else f"{trade_mode.upper()} · {cadence_min} min · up to {n} trade{'s' if n != 1 else ''} per candle"
        ),
    }


# Refresh XM PnL caches on summary at most this often (avoid MetaAPI 429).
_PNL_FORCE_INTERVAL_S = 20.0
_last_pnl_force_at = 0.0


@router.get("/summary")
async def trade_summary(
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    import time as _time

    import runtime_config
    from data.broker import broker_account_summary
    from data.market_data import get_source_info
    from app.services.broker_sync import sync_broker_throttled
    from app.services.desk_epoch import get_epoch_iso

    global _last_pnl_force_at

    trade_mode = runtime_config.get_trade_mode()
    xm_sync = None
    if runtime_config.is_broker_mode(trade_mode):
        try:
            xm_sync = await sync_broker_throttled(session)
        except Exception as exc:
            xm_sync = {"synced": False, "reason": str(exc)}

    stats = await get_trade_stats(session)
    source_info = get_source_info()
    stats["trade_mode"] = trade_mode
    stats["data_source"] = source_info["data_source"]
    stats["data_source_label"] = source_info["label"]
    stats["symbol"] = source_info["symbol"]
    stats["xm_sync"] = xm_sync
    stats["desk_reset_at"] = get_epoch_iso() or stats.get("desk_reset_at")

    # Session local P&L (Zenith closes after desk epoch)
    stats["session_realized_pnl"] = stats.get("realized_pnl")

    if runtime_config.is_broker_mode(trade_mode):
        now = _time.time()
        force_pnl = (now - _last_pnl_force_at) >= _PNL_FORCE_INTERVAL_S
        try:
            acct = await asyncio.to_thread(broker_account_summary, force=force_pnl)
            if force_pnl:
                _last_pnl_force_at = now
        except Exception:
            acct = None
        if acct:
            stats["broker_account"] = acct
            # Capital / equity / floating / open / realised — XM is source of truth
            stats["capital"] = acct.get("balance")
            stats["equity"] = acct.get("equity")
            broker_r = acct.get("history_pnl")
            if broker_r is None:
                broker_r = acct.get("realized_pnl")
            if broker_r is not None:
                stats["broker_realized_pnl"] = broker_r
                # balance − balance_at_reset (or full-account deals) — matches Capital
                stats["realized_pnl"] = broker_r
            if acct.get("lifetime_pnl_30d") is not None:
                stats["broker_lifetime_pnl"] = acct.get("lifetime_pnl_30d")
            if acct.get("floating_pnl") is not None:
                stats["unrealized_pnl"] = acct["floating_pnl"]
            stats["open_trades"] = acct.get("open_positions", stats.get("open_trades", 0))
            stats["xm_synced"] = True
            stats["pnl_forced"] = force_pnl
            stats["realized_source"] = acct.get("realized_source") or (
                "xm_since_desk_reset" if acct.get("realized_since") else "xm_history"
            )
            if acct.get("realized_since"):
                stats["realized_since"] = acct.get("realized_since")
            if acct.get("balance_at_reset") is not None:
                stats["balance_at_reset"] = acct.get("balance_at_reset")
        else:
            stats["xm_synced"] = False
            stats["realized_source"] = "zenith"
    else:
        stats["xm_synced"] = False
        stats["realized_source"] = "zenith"

    return stats


@router.get("/lot-desk")
async def lot_desk(
    _: User = Depends(get_current_user),
    capital: Optional[float] = Query(None, ge=50, le=5_000_000),
):
    """Capital + dynamic lot per universe symbol for the dashboard / backtest preview."""
    from strategy.lot_desk import lot_desk_payload

    return await asyncio.to_thread(lot_desk_payload, capital)


@router.post("/reset-desk")
async def reset_desk_stats_endpoint(_: User = Depends(get_current_user)):
    """Zero Won / BE / Lost and start a fresh desk session counter."""
    from app.services.desk_epoch import reset_desk_stats

    return await asyncio.to_thread(reset_desk_stats, reason="api")


@router.post("/repair-stops")
async def repair_stops(
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Rewrite missing/truncated SL-TP on open XM positions (and mirror into DB)."""
    from data.broker import broker_client
    from strategy.repair_stops import repair_open_stops

    result = await asyncio.to_thread(repair_open_stops, broker_client)
    # Mirror fixed stops into open Zenith rows
    if result.get("fixed"):
        for item in result["fixed"]:
            bid = str(item.get("id") or "")
            if not bid:
                continue
            row = (
                await session.execute(
                    select(TradeRecord).where(
                        TradeRecord.oanda_transaction_id == bid,
                        TradeRecord.pnl_status.in_(("open", "partial", "partial2", "partial3")),
                    )
                )
            ).scalar_one_or_none()
            if row:
                row.sl = item.get("sl")
                row.tp = item.get("tp")
        await session.commit()
    return result


@router.post("/sync-broker")
async def sync_broker(
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Pull open positions + P&L from XM and reconcile local trade records.

    Clears MetaAPI caches first so Realised / Floating are not stuck on stale
    values after a rate-limit backoff.
    """
    from app.services.broker_sync import (
        dedupe_broker_history,
        reconcile_closed_pnl_to_xm,
        sync_broker_positions,
    )
    from strategy.repair_stops import repair_open_stops
    from data.broker import broker_account_summary, broker_client

    # Drop stale realised/floating caches (often frozen after MetaAPI 429).
    # Prefer PnL-only clear so candle cache stays warm and we don't amplify 429s.
    clearer = getattr(broker_client, "clear_pnl_caches", None) or getattr(
        broker_client, "clear_caches", None
    )
    if clearer:
        await asyncio.to_thread(clearer)

    repair = await asyncio.to_thread(repair_open_stops, broker_client)
    dedupe = await dedupe_broker_history(session)
    result = await sync_broker_positions(session)
    recon = await reconcile_closed_pnl_to_xm(session, force=True)
    result["stop_repair"] = {
        "fixed_count": repair.get("fixed_count", 0),
        "failed": len(repair.get("failed") or []),
    }
    result["dedupe"] = dedupe
    result["pnl_reconcile"] = recon

    # force=True after sync — repair/sync may have refilled position cache only
    acct = await asyncio.to_thread(broker_account_summary, force=True)
    if acct:
        result["account"] = {
            "balance": acct.get("balance"),
            "equity": acct.get("equity"),
            "floating_pnl": acct.get("floating_pnl"),
            "realized_pnl": acct.get("realized_pnl") or acct.get("history_pnl"),
            "open_positions": acct.get("open_positions"),
            "realized_since": acct.get("realized_since"),
        }
        result["realized_pnl"] = result["account"]["realized_pnl"]
        result["floating_pnl"] = result["account"]["floating_pnl"]
        result["pnl_synced"] = True
    else:
        result["pnl_synced"] = False
    return result


@router.post("/dedupe-history")
async def dedupe_history(
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Remove duplicate XM position rows and fix fantasy P&L from MetaAPI deals."""
    from app.services.broker_sync import dedupe_broker_history

    return await dedupe_broker_history(session)


@router.post("/reset-history")
async def reset_history(
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Clear all trade history and re-import current XM open positions."""
    from app.services.broker_sync import reset_trade_history

    result = await reset_trade_history(session, sync_after=True)
    return {
        **result,
        "message": (
            f"Cleared {result.get('cleared', 0)} records. "
            f"Imported {result.get('synced', {}).get('imported', 0)} XM positions."
        ),
    }


@router.get("/learning")
async def trade_learning(_: User = Depends(get_current_user)):
    """ML patterns + calibration insights from all historical trades."""
    return await trade_calibrator.get_insights()


@router.get("/autopsy")
async def trade_autopsy_board(
    days: int = Query(90, ge=1, le=365),
    since_desk_reset: bool = Query(
        True, description="Only closes after desk reset (new brains)"
    ),
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Read-only per-pair autopsy: what's going wrong + efficiency 1–10.

    Defaults to the current desk session (post brain-wire), not rolling history.
    """
    from datetime import datetime, timedelta, timezone

    from config import INSTRUMENT_UNIVERSE
    from analysis.trade_autopsy_agent import build_dashboard_payload
    from analysis.trade_outcome import classify_broker_exit
    from app.services.desk_epoch import get_epoch, get_epoch_iso, row_after_epoch
    from sqlalchemy import or_

    ep = get_epoch() if since_desk_reset else None
    rolling = datetime.now(timezone.utc) - timedelta(days=days)
    since = ep if ep is not None else rolling
    if ep is not None and rolling > ep:
        # Still allow a shorter rolling window if caller asks, but never pre-epoch
        since = max(ep, rolling) if days < 14 else ep

    rows = (
        await session.execute(
            select(TradeRecord)
            .where(
                TradeRecord.created_at >= since,
                or_(
                    TradeRecord.status == "closed",
                    TradeRecord.pnl_status.like("closed%"),
                ),
            )
            .order_by(TradeRecord.created_at.desc())
            .limit(400)
        )
    ).scalars().all()

    if since_desk_reset and ep is not None:
        rows = [
            r
            for r in rows
            if row_after_epoch(getattr(r, "created_at", None), getattr(r, "log_time", None))
        ]

    # One-shot repair: legacy XM sync stamped everything closed_manual
    repaired = 0
    for r in rows:
        if (r.pnl_status or "") != "closed_manual":
            continue
        msg = str(r.message or "")
        if "Stopped at market" in msg:
            continue
        exit_px = r.current_price
        inferred = classify_broker_exit(
            r.signal,
            r.entry,
            r.sl,
            r.tp,
            exit_px,
            symbol=r._resolved_symbol() or r.symbol or "GOLD",
            tp1=r.tp1,
            tp2=r.tp2,
            tp3=r.tp3,
        )
        if inferred not in ("closed_sl", "closed_tp"):
            continue
        r.pnl_status = inferred
        bid = r.oanda_transaction_id or "synced"
        pnl_bit = f" · XM P&L {float(r.pnl):+.2f}" if r.pnl is not None else ""
        if inferred == "closed_sl":
            r.message = f"Stopped out on XM (pos {bid}){pnl_bit}"
            r.sl_trail_status = r.sl_trail_status or "SL hit"
        else:
            r.message = f"Take-profit on XM (pos {bid}){pnl_bit}"
            r.sl_trail_status = r.sl_trail_status or "TP hit"
        repaired += 1
    if repaired:
        await session.commit()

    payload_rows = []
    for r in rows:
        payload_rows.append(
            {
                "id": r.id,
                "symbol": r._resolved_symbol(),
                "signal": r.signal,
                "entry": r.entry,
                "sl": r.sl,
                "tp": r.tp,
                "tp1": r.tp1,
                "tp2": r.tp2,
                "tp3": r.tp3,
                "pnl": r.pnl,
                "pnl_status": r.pnl_status,
                "status": r.status,
                "message": r.message,
                "lot_size": r.lot_size,
                "margin_usd": r.margin_usd,
                "rr_type": r.rr_type,
                "trade_mode": r.trade_mode,
                "features": r.features,
                "candle_time": r.candle_time,
                "log_time": r.log_time,
                "created_at": r.created_at,
                "current_price": r.current_price,
            }
        )

    # Days spanned since desk reset (for UI label)
    span_days = days
    if ep is not None:
        span_days = max(
            1,
            int((datetime.now(timezone.utc) - ep).total_seconds() // 86400) + 1,
        )

    out = build_dashboard_payload(
        payload_rows,
        days=span_days,
        prefer_symbols=tuple(INSTRUMENT_UNIVERSE),
        desk_reset_at=get_epoch_iso() if since_desk_reset else None,
        since_desk_reset=bool(since_desk_reset and ep is not None),
    )
    out["repaired_labels"] = repaired
    return out


@router.get("/autopsy/proposals")
async def autopsy_proposals(_: User = Depends(get_current_user)):
    """Latest autopsy improve-agent proposals (research + soft-apply candidates)."""
    from analysis.autopsy_improve_agent import get_proposals
    from analysis.autopsy_soft import load as load_soft
    from analysis.autopsy_brain import load as load_brain

    payload = get_proposals()
    soft = load_soft()
    brain = load_brain()
    return {
        "updated_at": payload.get("updated_at"),
        "proposals": payload.get("proposals") or [],
        "runs": payload.get("runs") or [],
        "soft_active": {
            k: v
            for k, v in (soft.get("symbols") or {}).items()
            if v.get("active")
        },
        "brain_promoted": {
            k: v
            for k, v in (brain.get("symbols") or {}).items()
            if v.get("active")
        },
        "mode": payload.get("mode") or "idle",
        "note": payload.get("note"),
    }


@router.post("/autopsy/improve")
async def autopsy_improve_run(
    days: int = Query(14, ge=1, le=90),
    max_pairs: int = Query(5, ge=1, le=8),
    _: User = Depends(get_current_user),
):
    """Run agentic tip→A/B loop (paper OOS/WF/MC). Does not auto-apply live knobs."""
    from config import AUTOPSY_IMPROVE_ENABLED

    if not AUTOPSY_IMPROVE_ENABLED:
        raise HTTPException(
            status_code=403,
            detail="Autopsy improve agent disabled (set AUTOPSY_IMPROVE_ENABLED=true to re-enable)",
        )
    from analysis.autopsy_improve_agent import run_improve

    # yfinance + multi-symbol battery — run in thread so event loop stays alive
    payload = await asyncio.to_thread(
        run_improve,
        days=days,
        max_pairs=max_pairs,
        mc_paths=350,
        lookback_days=100,
    )
    proposed = [p for p in (payload.get("proposals") or []) if p.get("status") == "proposed"]
    return {
        "ok": True,
        "updated_at": payload.get("updated_at"),
        "n_proposals": len(payload.get("proposals") or []),
        "n_proposed": len(proposed),
        "proposals": payload.get("proposals") or [],
        "runs": payload.get("runs") or [],
        "note": payload.get("note"),
    }


@router.post("/autopsy/apply/{proposal_id}")
async def autopsy_apply_proposal(
    proposal_id: str,
    _: User = Depends(get_current_user),
):
    """Soft-apply → OOS/WF/MC vs previous → promote brain knobs or rollback soft."""
    from config import AUTOPSY_IMPROVE_ENABLED, AUTOPSY_SELF_HEAL_ENABLED

    if not (AUTOPSY_IMPROVE_ENABLED or AUTOPSY_SELF_HEAL_ENABLED):
        raise HTTPException(
            status_code=403,
            detail="Autopsy apply disabled — keeps live baselines untouched",
        )
    from analysis.autopsy_improve_agent import apply_proposal

    try:
        out = await asyncio.to_thread(apply_proposal, proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": bool(out.get("ok")), **out}


@router.post("/autopsy/self-heal")
async def autopsy_self_heal_run(
    force: bool = Query(False),
    _: User = Depends(get_current_user),
):
    """Run autopsy self-heal: research → soft-apply → promote/rollback → full stochastic battery."""
    from config import AUTOPSY_SELF_HEAL_ENABLED

    if not AUTOPSY_SELF_HEAL_ENABLED:
        raise HTTPException(
            status_code=403,
            detail="Autopsy self-heal disabled (set AUTOPSY_SELF_HEAL_ENABLED=true to re-enable)",
        )
    from analysis.autopsy_self_heal import run_self_heal

    out = await asyncio.to_thread(
        run_self_heal, force=force, days=14, auto_apply=True, max_apply=4
    )
    return out


@router.get("/autopsy/self-heal")
async def autopsy_self_heal_status(_: User = Depends(get_current_user)):
    from analysis.autopsy_self_heal import status

    return status()


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

    # Persist repaired FX P&L (legacy gold-units bug left $0 on closed majors)
    dirty = False
    for r in rows:
        if r.status != "closed" or r.signal not in ("BUY", "SELL"):
            continue
        fixed = r._repaired_pnl()
        if fixed is not None and abs(float(fixed) - float(r.pnl or 0)) >= 0.01:
            r.pnl = fixed
            dirty = True
    if dirty:
        await session.commit()

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

    sym = str(
        result.get("symbol")
        or sig.get("SYMBOL")
        or sig.get("INSTRUMENT")
        or "GOLD"
    ).upper().strip()
    from strategy.dynamic_lots import sizing_snapshot

    snap = result.get("sizing") or sizing_snapshot(sym)
    lot = float(result.get("lot_size") or snap.get("lot_size") or TRADE_LOT_SIZE)
    margin = estimate_margin(float(entry), lot, symbol=sym) if entry is not None else None

    from strategy.position_sizing import leg_pnl
    initial_pnl = leg_pnl(signal, entry, float(live_px), lot, symbol=sym) if entry and live_px else 0.0

    trail_label = (
        sl_trail_label(signal, float(entry), float(sl), "open")
        if entry and sl else "Fixed SL"
    )

    feats = {
        "symbol": sym,
        "SYMBOL": sym,
        "source": "manual_execute",
        "lot_size": lot,
        "capital": snap.get("capital"),
        "asset_class": snap.get("asset_class"),
        "trail_lock": snap.get("trail_lock"),
        "tp1_frac": snap.get("tp1_frac"),
    }

    record = TradeRecord(
        candle_time=sig.get("CANDLE_TIME", ""),
        log_time=bucket_log,
        signal=signal,
        symbol=sym,
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
        features=json.dumps(feats),
        pnl_status="open" if result["status"] == "executed" else None,
        status=result["status"],
        message=result["message"],
        oanda_transaction_id=result.get("oanda_transaction_id") or result.get("order_id"),
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

    import runtime_config
    if runtime_config.is_broker_mode():
        try:
            from data.broker import broker_client, broker_name

            if broker_client.configured:
                await asyncio.to_thread(broker_client.close_all_positions)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Failed to close {broker_name.upper()} positions: {exc}"
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
