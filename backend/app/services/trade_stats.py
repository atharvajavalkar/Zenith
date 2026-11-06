"""Aggregate trade performance statistics (current desk session)."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from analysis.trade_outcome import is_breakeven_trade, is_losing_trade, is_winning_trade
from app.models.trade_record import TradeRecord
from app.services.desk_epoch import get_epoch_iso, row_after_epoch


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


def _dedupe_closed(closed: list[TradeRecord]) -> list[TradeRecord]:
    """One row per XM position id — prevents double-counted session PnL."""
    by_pid: dict[str, TradeRecord] = {}
    no_pid: list[TradeRecord] = []
    for r in closed:
        pid = str(r.oanda_transaction_id or "").strip()
        if not pid:
            no_pid.append(r)
            continue
        prev = by_pid.get(pid)
        if prev is None or (r.id or 0) < (prev.id or 0):
            by_pid[pid] = r  # keep oldest (first close)
    return no_pid + list(by_pid.values())


def compute_trade_stats(rows: list[TradeRecord]) -> dict:
    trades = [
        r
        for r in rows
        if _is_trade_row(r) and row_after_epoch(getattr(r, "created_at", None), getattr(r, "log_time", None))
    ]
    closed_raw = [r for r in trades if r.status == "closed"]
    closed = _dedupe_closed(closed_raw)
    open_raw = [
        r
        for r in trades
        if r.status == "executed" and r.pnl_status in ("open", "partial", "partial2", "partial3")
    ]
    # One open row per XM position id (native fill + sync import often double)
    open_rows = _dedupe_closed(open_raw)

    wins = sum(1 for r in closed if is_winning_trade(r))
    losses = sum(1 for r in closed if is_losing_trade(r))
    breakevens = sum(1 for r in closed if is_breakeven_trade(r))

    def _pnl(r: TradeRecord) -> float:
        # Prefer stamped XM deal cash (never leg_pnl fantasy on broker closes)
        try:
            feats = r.features
            if isinstance(feats, str):
                import json

                feats = json.loads(feats)
            if isinstance(feats, dict) and feats.get("pnl_source") == "xm_deal":
                if feats.get("xm_deal_pnl") is not None:
                    return float(feats["xm_deal_pnl"])
        except Exception:
            pass
        try:
            return float(r._repaired_pnl() if hasattr(r, "_repaired_pnl") else (r.pnl or 0))
        except Exception:
            return float(r.pnl or 0)

    realized_pnl = round(sum(_pnl(r) for r in closed), 2)
    unrealized_pnl = round(sum(_pnl(r) for r in open_rows), 2)

    open_rr_values = [v for r in open_rows if (v := _trade_rr_pct(r)) is not None]
    avg_open_rr_pct = (
        round(sum(open_rr_values) / len(open_rr_values), 1) if open_rr_values else None
    )

    return {
        "executed": len(trades),
        "open_trades": len(open_rows),
        "closed_trades": len(closed),
        "closed_dupes_dropped": max(0, len(closed_raw) - len(closed)),
        "wins": wins,
        "losses": losses,
        "breakevens": breakevens,
        "net_pnl": round(realized_pnl + unrealized_pnl, 2),
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "avg_open_rr_pct": avg_open_rr_pct,
        "desk_reset_at": get_epoch_iso(),
        "session_mode": True if get_epoch_iso() else False,
    }


async def get_trade_stats(session: AsyncSession) -> dict:
    rows = (await session.execute(select(TradeRecord))).scalars().all()
    return compute_trade_stats(rows)
