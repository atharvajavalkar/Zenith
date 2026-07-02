"""Refresh ML + Markov after every closed trade."""

from __future__ import annotations

import asyncio
import logging

from analysis.adaptive_guard import get_state, rebuild
from config import (
    ADAPTIVE_LEARNING,
    LOSS_PNL_PROTECT,
    LOSS_STREAK_REDUCE_AT,
    MAX_TRADES_PER_BUCKET,
    MC_FLOOR_BASE,
)
from app.services.markov_cache import markov_cache
from app.services.trade_calibrator import trade_calibrator
from app.services.trade_settings import get_max_trades_per_bucket, set_max_trades_per_bucket
from app.services.ws_manager import ws_manager

logger = logging.getLogger(__name__)

_refresh_lock = asyncio.Lock()
_pending = False


async def on_trade_closed() -> None:
    """Learn from the trade that just closed — Markov, ML, adaptive guard."""
    global _pending
    if not ADAPTIVE_LEARNING:
        return

    _pending = True

    async def _run() -> None:
        global _pending
        async with _refresh_lock:
            try:
                rows = await trade_calibrator.load_closed_rows()
                closed = [r for r in rows if r.get("status") == "closed"]
                adaptive = rebuild(
                    closed,
                    loss_pnl_protect=LOSS_PNL_PROTECT,
                    mc_floor_base=MC_FLOOR_BASE,
                )

                await asyncio.gather(
                    markov_cache.refresh(force=True),
                    trade_calibrator.refresh(force=True, rows=rows),
                )

                _apply_trade_pacing(adaptive)
                logger.info(
                    "Per-trade learning #%s: mode=%s realized=%.2f streak=%s mc_floor=%.0f",
                    adaptive.get("updates"),
                    adaptive.get("mode"),
                    adaptive.get("realized_pnl", 0),
                    adaptive.get("loss_streak"),
                    adaptive.get("mc_floor"),
                )
                await ws_manager.broadcast({"type": "learning_update", "adaptive": get_state()})
            except Exception as exc:
                logger.exception("Per-trade learning failed: %s", exc)
            finally:
                _pending = False

    asyncio.create_task(_run())


def _apply_trade_pacing(adaptive: dict) -> None:
    """Slow down stacking when losing; restore when recovering."""
    streak = adaptive.get("loss_streak", 0)
    mode = adaptive.get("mode", "normal")
    current = get_max_trades_per_bucket()
    if current <= 0:
        return

    if mode == "protect" and streak >= LOSS_STREAK_REDUCE_AT and current > 1:
        set_max_trades_per_bucket(current - 1)
        logger.warning(
            "Loss protection: reduced trades/candle %s → %s (streak %s, PnL %.2f)",
            current,
            current - 1,
            streak,
            adaptive.get("realized_pnl"),
        )
    elif mode == "normal" and streak == 0 and current < MAX_TRADES_PER_BUCKET:
        rolling = adaptive.get("rolling_win_rate")
        if rolling is not None and rolling >= 0.55:
            set_max_trades_per_bucket(min(MAX_TRADES_PER_BUCKET, current + 1))
