"""Async service: train ML calibrator from DB and gate new trades."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from sqlalchemy import select

from analysis.adaptive_guard import get_state, rebuild
from analysis.concept_engine import get_state as concept_get_state
from analysis.trade_learning import (
    LogisticCalibrator,
    analyze_patterns,
    evaluate_trade,
    extract_features_from_signal,
)
from config import (
    ADAPTIVE_LEARNING,
    BIAS_GATE_ENABLED,
    LOSS_PNL_PROTECT,
    MC_FLOOR_BASE,
    ML_GATE_ENABLED,
)
from app.core.database import AsyncSessionLocal
from app.services.markov_cache import markov_cache
from app.models.trade_record import TradeRecord

logger = logging.getLogger(__name__)

REFRESH_SECONDS = 300


class TradeCalibrator:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._model = LogisticCalibrator()
        self._patterns: dict[str, Any] = {}
        self._updated_at: float = 0.0
        self._last_error: Optional[str] = None
        self._closes_learned: int = 0

    async def load_all_rows(self) -> list[dict]:
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(select(TradeRecord))).scalars().all()
            return [r.to_dict() for r in rows]

    async def load_closed_rows(self) -> list[dict]:
        return await self.load_all_rows()

    async def refresh(self, force: bool = False, rows: Optional[list[dict]] = None) -> None:
        async with self._lock:
            if not force and (time.time() - self._updated_at) < REFRESH_SECONDS:
                return

        try:
            row_dicts = rows if rows is not None else await self.load_all_rows()
            closed = [d for d in row_dicts if d.get("status") == "closed"]
            # Research / gates learn only from the current desk session (new brains)
            try:
                from app.services.desk_epoch import filter_rows_since_epoch

                closed = filter_rows_since_epoch(closed)
                row_dicts = filter_rows_since_epoch(row_dicts)
            except Exception:
                pass

            if ADAPTIVE_LEARNING:
                rebuild(closed, loss_pnl_protect=LOSS_PNL_PROTECT, mc_floor_base=MC_FLOOR_BASE)

            model = LogisticCalibrator()
            model.fit(closed)
            patterns = analyze_patterns(row_dicts)

            async with self._lock:
                self._model = model
                self._patterns = patterns
                self._updated_at = time.time()
                self._last_error = None
                self._closes_learned = len(closed)

            logger.info(
                "Trade calibrator refreshed: %s closed, train WR %.1f%%, gated WR %.1f%%, threshold %.2f, mode=%s",
                patterns.get("total_closed", 0),
                (model.train_win_rate or 0) * 100,
                (model.filtered_win_rate or 0) * 100,
                model.threshold,
                get_state().get("mode"),
            )
        except Exception as exc:
            logger.exception("Trade calibrator refresh failed: %s", exc)
            async with self._lock:
                self._last_error = str(exc)

    async def on_trade_closed(self) -> None:
        """Immediate learn pass — called from learning_loop."""
        await self.refresh(force=True)

    async def evaluate(self, sig: dict, trade_sig: dict) -> tuple[bool, float, str, dict]:
        async with self._lock:
            if time.time() - self._updated_at > REFRESH_SECONDS:
                asyncio.create_task(self.refresh(force=True))
            model = self._model

        features = extract_features_from_signal(sig, trade_sig)
        features["exec_tf"] = trade_sig.get("EXEC_TF") or sig.get("EXEC_TF") or features.get("exec_tf")
        features["EXEC_TF"] = features["exec_tf"]
        features["mtf_independent"] = bool(
            trade_sig.get("MTF_INDEPENDENT") or sig.get("MTF_INDEPENDENT")
        )
        sym = str(
            trade_sig.get("SYMBOL")
            or trade_sig.get("INSTRUMENT")
            or sig.get("SYMBOL")
            or features.get("symbol")
            or "GOLD"
        ).upper()
        features["symbol"] = sym
        features["SYMBOL"] = sym
        try:
            from analysis.instrument_rl import get_active_arm, overlays_for

            features["rl_arm"] = get_active_arm(sym)
            features["rl_overlays"] = overlays_for(sym)
        except Exception:
            pass
        if not ML_GATE_ENABLED and not ADAPTIVE_LEARNING:
            return True, 0.55, "Gates off", features

        allowed, prob, reason = evaluate_trade(
            features, model, use_ml_gate=ML_GATE_ENABLED
        )
        return allowed, prob, reason, features

    async def get_insights(self) -> dict[str, Any]:
        async with self._lock:
            stale = not self._patterns
        if stale:
            await self.refresh(force=True)
        markov_data = await markov_cache.get_insights()
        try:
            from analysis.instrument_rl import get_state as rl_state

            instrument_rl = rl_state()
        except Exception:
            instrument_rl = {"instruments": {}, "total_updates": 0}
        try:
            from analysis.minute_learn import get_minute_state
            from analysis.correlation_matrix import get_correlation_state

            minute_learn = get_minute_state()
            correlation = get_correlation_state()
        except Exception:
            minute_learn = {}
            correlation = {}
        try:
            from analysis.candle_edge import snapshot as candle_edge_snap

            candle_edge = candle_edge_snap()
        except Exception:
            candle_edge = {"symbols": []}
        async with self._lock:
            return {
                "patterns": self._patterns,
                "model": self._model.to_dict(),
                "updated_at": self._updated_at,
                "last_error": self._last_error,
                "closes_learned": self._closes_learned,
                "adaptive": get_state(),
                "concepts": concept_get_state(),
                "instrument_rl": instrument_rl,
                "minute_learn": minute_learn,
                "candle_edge": candle_edge,
                "correlation": {
                    "pairs": correlation.get("pairs"),
                    "leaders": correlation.get("leaders"),
                    "updated_at": correlation.get("updated_at"),
                },
                "gates": {
                    "ml_gate_enabled": ML_GATE_ENABLED,
                    "bias_gate_enabled": BIAS_GATE_ENABLED,
                    "adaptive_learning": ADAPTIVE_LEARNING,
                    "instrument_rl": True,
                    "minute_candle_learn": True,
                    "candle_walk_forward": True,
                    "frvp_poc": True,
                    "correlation": True,
                },
                "markov": markov_data.get("markov"),
                "markov_updated_at": markov_data.get("updated_at"),
            }

    @staticmethod
    async def _load_trade_rows() -> list[TradeRecord]:
        async with AsyncSessionLocal() as session:
            return (await session.execute(select(TradeRecord))).scalars().all()

    @staticmethod
    def _row_to_dict(row: TradeRecord) -> dict:
        return row.to_dict()


trade_calibrator = TradeCalibrator()
