"""Chart candle endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.security import get_current_user
from app.models.user import User
from app.schemas.signals import CandleBar, CandlesResponse
from app.services.market_cache import market_cache

VALID_TIMEFRAMES = {"1m", "5m", "15m", "1h", "4h"}

router = APIRouter(prefix="/api/chart", tags=["chart"])


@router.get("/candles", response_model=CandlesResponse)
async def get_candles(
    _: User = Depends(get_current_user),
    tf: str = Query("1m", description="Timeframe: 1m, 5m, 15m, 1h, 4h"),
    limit: int = Query(200, ge=1, le=500),
):
    tf_lower = tf.lower()
    if tf_lower not in VALID_TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Use: {VALID_TIMEFRAMES}")

    bars = await market_cache.get_candles(tf_lower, limit)
    if not bars:
        raise HTTPException(status_code=503, detail="Candle data not available yet")

    status = await market_cache.get_status()
    return CandlesResponse(
        timeframe=tf_lower,
        candles=[CandleBar(**b) for b in bars],
        data_source=status.get("data_source"),
        symbol=status.get("symbol"),
    )
