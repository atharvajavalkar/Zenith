"""Aggregate trade performance statistics."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from analysis.trade_outcome import is_breakeven_trade, is_losing_trade, is_winning_trade
from app.models.trade_record import TradeRecord


def _is_trade_row(row: TradeRecord) -> bool:
    return row.signal in ("BUY", "SELL") and row.status in ("executed", "closed")


def _trade_rr_pct(row: TradeRecord) -> Optional[float]:
    if row.entry is None or row.sl is None or row.tp is None:
        return None
    risk = abs(float(row.entry) - float(row.sl))
    reward = abs(float(row.tp) - float(row.entry))
    if risk <= 0:
        return None
    return round(reward / risk * 100, 1)


def compute_trade_stats(rows: list[TradeRecord]) -> dict:
    trades = [r for r in rows if _is_trade_row(r)]
    closed = [r for r in trades if r.status == "closed"]
    open_rows = [
        r for r in trades
        if r.status == "executed" and r.pnl_status in ("open", "partial")
    ]

    wins = sum(1 for r in closed if is_winning_trade(r))
    losses = sum(1 for r in closed if is_losing_trade(r))
    breakevens = sum(1 for r in closed if is_breakeven_trade(r))

    realized_pnl = round(sum(r.pnl or 0 for r in closed), 2)
    unrealized_pnl = round(sum(r.pnl or 0 for r in open_rows), 2)

    open_rr_values = [v for r in open_rows if (v := _trade_rr_pct(r)) is not None]
    avg_open_rr_pct = (
        round(sum(open_rr_values) / len(open_rr_values), 1) if open_rr_values else None
    )

    return {
        "executed": len(trades),
        "open_trades": len(open_rows),
        "closed_trades": len(closed),
        "wins": wins,
        "losses": losses,
        "breakevens": breakevens,
        "net_pnl": round(realized_pnl + unrealized_pnl, 2),
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "avg_open_rr_pct": avg_open_rr_pct,
    }


async def get_trade_stats(session: AsyncSession) -> dict:
    rows = (await session.execute(select(TradeRecord))).scalars().all()
    return compute_trade_stats(rows)
