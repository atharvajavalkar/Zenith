"""Async service: learn Markov transitions from signal history + trade feedback."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from sqlalchemy import select

from analysis.markov_engine import build_markov_model, markov_store
from app.core.database import AsyncSessionLocal
from app.models.signal_snapshot import SignalSnapshot
from app.models.trade_record import TradeRecord

logger = logging.getLogger(__name__)

REFRESH_SECONDS = 3600


class MarkovCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._model: dict[str, Any] = markov_store.get()
        self._updated_at: float = 0.0
        self._last_error: Optional[str] = None

    async def refresh(self, force: bool = False) -> None:
        async with self._lock:
            if not force and (time.time() - self._updated_at) < REFRESH_SECONDS:
                return

        try:
            snapshots, trades = await asyncio.gather(
                self._load_snapshots(),
                self._load_trades(),
            )
            snap_dicts = [s.to_dict() for s in snapshots]
            trade_dicts = [t.to_dict() for t in trades]
            model = build_markov_model(snap_dicts, trade_dicts)
            markov_store.update(model)

            async with self._lock:
                self._model = model
                self._updated_at = time.time()
                self._last_error = None

            cal = model.get("calibration", {})
            logger.info(
                "Markov model refreshed: %s states, source=%s, cal samples=%s brier=%s",
                model.get("state_sequence_length", 0),
                model.get("transition_source"),
                cal.get("samples"),
                cal.get("brier"),
            )
        except Exception as exc:
            logger.exception("Markov refresh failed: %s", exc)
            async with self._lock:
                self._last_error = str(exc)

    async def get_insights(self) -> dict[str, Any]:
        async with self._lock:
            stale = (time.time() - self._updated_at) > REFRESH_SECONDS
        if stale:
            await self.refresh(force=True)
        async with self._lock:
            return {
                "markov": self._model,
                "updated_at": self._updated_at,
                "last_error": self._last_error,
            }

    @staticmethod
    async def _load_snapshots() -> list[SignalSnapshot]:
        async with AsyncSessionLocal() as session:
            return (
                await session.execute(
                    select(SignalSnapshot).order_by(SignalSnapshot.id.asc())
                )
            ).scalars().all()

    @staticmethod
    async def _load_trades() -> list[TradeRecord]:
        async with AsyncSessionLocal() as session:
            return (await session.execute(select(TradeRecord))).scalars().all()


markov_cache = MarkovCache()
