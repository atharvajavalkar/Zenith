"""Close open paper/live trade records at market."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import TP1_PARTIAL_FRACTION, TRADE_LOT_SIZE
from strategy.position_sizing import leg_pnl

from app.models.trade_record import TradeRecord

OPEN_STATUSES = ("open", "partial")


def calculate_pnl(
    signal: Optional[str],
    entry: Optional[float],
    price: Optional[float],
    lot_size: float | None = None,
    partial_pnl: float = 0.0,
    pnl_status: str | None = "open",
) -> float:
    if signal not in ("BUY", "SELL") or entry is None or price is None:
        return 0.0
    lot = lot_size or TRADE_LOT_SIZE
    runner_frac = 1.0 - TP1_PARTIAL_FRACTION if pnl_status == "partial" else 1.0
    runner = leg_pnl(signal, entry, price, lot) * runner_frac
    return round((partial_pnl or 0.0) + runner, 2)


async def close_all_open_trades(session: AsyncSession, exit_price: float) -> int:
    rows = (
        await session.execute(
            select(TradeRecord).where(TradeRecord.pnl_status.in_(OPEN_STATUSES))
        )
    ).scalars().all()
    if not rows:
        return 0

    for row in rows:
        row.current_price = exit_price
        row.pnl = calculate_pnl(
            row.signal,
            row.entry,
            exit_price,
            row.lot_size,
            row.partial_pnl or 0.0,
            row.pnl_status,
        )
        row.pnl_status = "closed_manual"
        row.status = "closed"
        row.sl_trail_status = "Closed"
        row.message = f"Stopped at market {exit_price:.2f}"

    await session.commit()
    return len(rows)
