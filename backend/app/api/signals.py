"""Signal REST endpoints."""

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

from data.fetcher import get_tick_quote  # noqa: E402
from strategy.position_sizing import lot_to_units  # noqa: E402

from app.core.database import get_session
from app.core.security import get_current_user
from app.models.signal_snapshot import SignalSnapshot
from app.models.user import User
from app.schemas.signals import SignalHistoryResponse, SignalResponse
from app.services.bias_cache import bias_cache
from app.services.market_cache import market_cache

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("/bias")
async def market_bias(_: User = Depends(get_current_user)):
    """Hourly macro + technical directional bias for XAU/USD (no trade levels)."""
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


@router.get("/status")
async def worker_status(_: User = Depends(get_current_user)):
    return await market_cache.get_status()
