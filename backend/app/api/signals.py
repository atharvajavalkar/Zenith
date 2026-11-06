"""Signal REST endpoints."""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

_ALGO_ROOT = Path(__file__).resolve().parents[3]
if str(_ALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGO_ROOT))

from config import TRADE_LOT_SIZE  # noqa: E402
from data.market_data import get_tick_quote  # noqa: E402
from strategy.position_sizing import estimate_margin, lot_to_units  # noqa: E402
from strategy.trailing_stop import sl_trail_label  # noqa: E402

from app.core.database import get_session
from app.core.security import get_current_user
from app.models.signal_snapshot import SignalSnapshot
from app.models.trade_record import TradeRecord
from app.models.user import User
from app.schemas.signals import SignalHistoryResponse, SignalResponse
from app.services.bias_cache import bias_cache
from app.services.market_cache import market_cache
from app.services.trade_settings import bucket_log_time, get_exec_cadence

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("/bias")
async def market_bias(_: User = Depends(get_current_user)):
    """Algo bias for the full instrument universe (physics + macro + news + profile)."""
    data = await bias_cache.get()
    if data is None:
        raise HTTPException(status_code=503, detail="Market bias not available yet")
    return data


@router.get("/tick")
async def live_tick(_: User = Depends(get_current_user)):
    """Lightweight live price for dashboard — polled every second."""
    quote = await asyncio.to_thread(get_tick_quote)
    sig = await market_cache.get_signal()
    price = quote.get("price")
    if price is None and sig:
        price = sig.get("LOG_PRICE_1M") or sig.get("LIVE_PRICE")

    rr_pct = None
    risk_usd = None
    reward_usd = None
    live_r = None
    if sig and sig.get("SIGNAL") in ("BUY", "SELL"):
        entry = sig.get("ENTRY")
        sl = sig.get("SL")
        tp = sig.get("TP")
        if entry is not None and sl is not None and tp is not None:
            units = lot_to_units()
            risk_pts = abs(float(entry) - float(sl))
            reward_pts = abs(float(tp) - float(entry))
            risk_usd = round(risk_pts * units, 2)
            reward_usd = round(reward_pts * units, 2)
            if risk_pts > 0:
                rr_pct = round(reward_pts / risk_pts * 100, 1)
                if price is not None:
                    move = (
                        float(price) - float(entry)
                        if sig["SIGNAL"] == "BUY"
                        else float(entry) - float(price)
                    )
                    live_r = round(move / risk_pts, 2)

    return {
        "price": price,
        "time": quote.get("time") or (sig.get("LOG_TIME_1M") if sig else None),
        "signal": sig.get("SIGNAL") if sig else None,
        "rr_pct": rr_pct,
        "risk_usd": risk_usd,
        "reward_usd": reward_usd,
        "live_r": live_r,
        "source": quote.get("source") or (sig.get("DATA_SOURCE") if sig else None),
        "symbol": quote.get("symbol") or (sig.get("SYMBOL") if sig else None),
        "bid": quote.get("bid"),
        "ask": quote.get("ask"),
    }


@router.get("/latest", response_model=SignalResponse)
async def latest_signal(_: User = Depends(get_current_user)):
    sig = await market_cache.get_signal()
    if sig is None:
        raise HTTPException(status_code=503, detail="Signal not available yet")
    return sig


@router.get("/history", response_model=SignalHistoryResponse)
async def signal_history(
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    signal_filter: Optional[str] = Query(None, alias="signal"),
):
    query = select(SignalSnapshot).order_by(SignalSnapshot.id.desc())
    count_query = select(func.count()).select_from(SignalSnapshot)

    if signal_filter:
        query = query.where(SignalSnapshot.signal == signal_filter.upper())
        count_query = count_query.where(SignalSnapshot.signal == signal_filter.upper())

    total = (await session.execute(count_query)).scalar_one()
    rows = (await session.execute(query.offset(offset).limit(limit))).scalars().all()

    return SignalHistoryResponse(
        items=[row.to_dict() for row in rows],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/opportunities")
async def opportunities(_: User = Depends(get_current_user)):
    """
    Ranked multi-instrument setups (same core algo across the universe).
    Used by the dashboard opportunity board.
    """
    from config import INSTRUMENT_UNIVERSE, MULTI_INSTRUMENT_ENABLED
    from analysis.opportunity_learn import record_opportunity
    from strategy.instrument_scanner import opportunity_summary, scan_universe

    if not MULTI_INSTRUMENT_ENABLED:
        return {
            "enabled": False,
            "universe": list(INSTRUMENT_UNIVERSE),
            "opportunities": [],
            "message": "MULTI_INSTRUMENT_ENABLED=false",
        }

    hits = await asyncio.to_thread(scan_universe)
    rows = opportunity_summary(hits)
    try:
        from analysis.opportunity_punch import annotate_opportunities

        rows = await asyncio.to_thread(annotate_opportunities, rows)
    except Exception:
        for r in rows:
            r["punchable"] = True
            r["status"] = "ready"
            r["block_reason"] = None
    for r in rows:
        try:
            record_opportunity(
                symbol=str(r.get("symbol") or ""),
                exec_tf=str(r.get("exec_tf") or ""),
                signal=str(r.get("signal") or ""),
                action="seen" if r.get("punchable", True) else "blocked",
                mc=float(r["mc"]) if r.get("mc") is not None else None,
                detail=r.get("block_reason"),
            )
        except Exception:
            pass
    ready = sum(1 for r in rows if r.get("punchable"))
    try:
        from config import SYMBOL_RR

        symbol_rr_map = {k: float(v) for k, v in SYMBOL_RR.items()}
        for r in rows:
            sym = str(r.get("symbol") or "").upper()
            if sym in symbol_rr_map:
                r["target_rr"] = symbol_rr_map[sym]
                r["target_rr_label"] = f"1:{symbol_rr_map[sym]:g}"
    except Exception:
        symbol_rr_map = {}
    return {
        "enabled": True,
        "universe": list(INSTRUMENT_UNIVERSE),
        "count": len(hits),
        "ready": ready,
        "blocked": len(rows) - ready,
        "opportunities": rows,
        "symbol_rr": symbol_rr_map,
        "note": "Board = brain setups. Auto only punches rows marked ready (gates pass).",
    }


class _OppEnterBody(BaseModel):
    symbol: str
    exec_tf: Optional[str] = None
    signal: Optional[str] = None
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    force: bool = True


@router.post("/opportunities/enter")
async def enter_opportunity(
    body: _OppEnterBody,
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    Punch a board row onto XM/demo using the live scanner payload for that
    symbol/TF (levels refreshed) + instrument RL lot sizing. Persists a
    TradeRecord tagged for per-instrument RL when the fill succeeds.
    """
    from analysis.opportunity_learn import record_opportunity
    from analysis.instrument_rl import get_active_arm, overlays_for
    from strategy.dynamic_lots import sizing_snapshot
    from strategy.instrument_scanner import scan_symbol
    from strategy.position_sizing import leg_pnl
    from strategy.trade_executor import execute_signal_trade

    sym = str(body.symbol).upper().strip()
    hits = await asyncio.to_thread(scan_symbol, sym)
    chosen = None
    for h in hits:
        if body.exec_tf and str(h.get("EXEC_TF") or "") != str(body.exec_tf):
            continue
        if body.signal and str(h.get("SIGNAL") or "") != str(body.signal).upper():
            continue
        if h.get("SIGNAL") in ("BUY", "SELL"):
            chosen = h
            break
    if chosen is None and hits:
        chosen = next((h for h in hits if h.get("SIGNAL") in ("BUY", "SELL")), None)
    if chosen is None:
        # Fall back to posted levels if scan no longer has the setup
        if body.signal in ("BUY", "SELL") and body.entry and body.sl and body.tp:
            chosen = {
                "SIGNAL": body.signal.upper(),
                "ENTRY": body.entry,
                "SL": body.sl,
                "TP": body.tp,
                "SYMBOL": sym,
                "INSTRUMENT": sym,
                "EXEC_TF": body.exec_tf or "5m",
                "CANDLE_TIME": "",
                "REASON": "Opportunity board manual enter (scan missed — using board levels)",
            }
        else:
            record_opportunity(
                symbol=sym,
                exec_tf=str(body.exec_tf or ""),
                signal=str(body.signal or ""),
                action="skipped",
                detail="setup no longer valid",
            )
            raise HTTPException(status_code=409, detail=f"No live {sym} setup to enter")

    chosen = dict(chosen)
    chosen["SYMBOL"] = sym
    chosen["INSTRUMENT"] = sym
    rl_arm = "balanced"
    try:
        rl_arm = get_active_arm(sym)
        chosen["RL_ARM"] = rl_arm
        chosen["RL_OVERLAYS"] = overlays_for(sym)
    except Exception:
        pass

    result = await asyncio.to_thread(execute_signal_trade, chosen, bool(body.force))
    action = "entered" if result.get("status") == "executed" else "blocked"
    record_opportunity(
        symbol=sym,
        exec_tf=str(chosen.get("EXEC_TF") or body.exec_tf or ""),
        signal=str(chosen.get("SIGNAL") or ""),
        action=action,
        mc=float(chosen["SUCCESS_PERCENT"]) if chosen.get("SUCCESS_PERCENT") is not None else None,
        detail=str(result.get("message") or "")[:160],
    )

    trade_dict = None
    if result.get("status") == "executed":
        signal = chosen.get("SIGNAL", result.get("signal", "NO TRADE"))
        entry = result.get("entry", chosen.get("ENTRY"))
        sl = result.get("sl", chosen.get("SL"))
        tp = result.get("tp", chosen.get("TP"))
        live_px = chosen.get("LOG_PRICE_1M") or chosen.get("LIVE_PRICE") or entry
        snap = result.get("sizing") or sizing_snapshot(
            sym,
            entry=float(entry) if entry is not None else None,
            sl=float(sl) if sl is not None else None,
            tp=float(tp) if tp is not None else None,
            rr=chosen.get("RR"),
        )
        lot = float(result.get("lot_size") or snap.get("lot_size") or TRADE_LOT_SIZE)
        margin = estimate_margin(float(entry), lot, symbol=sym) if entry is not None else None
        initial_pnl = (
            leg_pnl(signal, entry, float(live_px), lot, symbol=sym)
            if entry and live_px
            else 0.0
        )
        trail_label = (
            sl_trail_label(signal, float(entry), float(sl), "open")
            if entry and sl
            else "Fixed SL"
        )
        ov = chosen.get("RL_OVERLAYS") or {}
        feats = {
            "symbol": sym,
            "SYMBOL": sym,
            "source": "opportunity_board",
            "opportunity_enter": True,
            "exec_tf": chosen.get("EXEC_TF") or body.exec_tf,
            "lot_size": lot,
            "capital": snap.get("capital"),
            "asset_class": snap.get("asset_class"),
            "trail_lock": snap.get("trail_lock"),
            "tp1_frac": snap.get("tp1_frac"),
            "rl_arm": rl_arm,
            "rl_mc_delta": ov.get("mc_floor_delta"),
            "rl_aggression": ov.get("aggression"),
            "rl_risk_scale": ov.get("risk_scale"),
        }
        # Unique log_time so board punches never collide with auto worker
        bucket_log = f"{bucket_log_time(chosen, get_exec_cadence())}#opp{int(time.time()) % 100000:05d}"
        record = TradeRecord(
            candle_time=chosen.get("CANDLE_TIME", ""),
            log_time=bucket_log,
            signal=signal,
            symbol=sym,
            entry=entry,
            sl=sl,
            tp=tp,
            tp1=result.get("tp1", chosen.get("TP1")),
            tp2=result.get("tp2", chosen.get("TP2")),
            tp3=result.get("tp3", chosen.get("TP3")),
            rr_type=chosen.get("RR"),
            current_price=live_px,
            pnl=initial_pnl,
            partial_pnl=0.0,
            lot_size=lot,
            margin_usd=margin,
            sl_trail_status=trail_label,
            features=json.dumps(feats),
            pnl_status="open",
            status="executed",
            message=result.get("message", ""),
            oanda_transaction_id=result.get("oanda_transaction_id") or result.get("order_id"),
        )
        session.add(record)
        await session.flush()
        trade_dict = record.to_dict()

    return {
        "result": result,
        "trade": trade_dict,
        "signal": chosen.get("SIGNAL"),
        "symbol": sym,
        "exec_tf": chosen.get("EXEC_TF"),
        "rl_arm": rl_arm,
    }


@router.get("/status")
async def worker_status(_: User = Depends(get_current_user)):
    return await market_cache.get_status()
