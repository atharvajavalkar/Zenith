"""FastAPI application entry point."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure algo core is on PYTHONPATH when running from backend/
_ALGO_ROOT = Path(__file__).resolve().parents[2]
if str(_ALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGO_ROOT))

from app.api.algo_settings import router as algo_settings_router
from app.api.auth import router as auth_router
from app.api.backtest import router as backtest_router
from app.api.chart import router as chart_router
from app.api.hyper_scalp import router as hyper_scalp_router
from app.api.news import router as news_router
from app.api.signals import router as signals_router
from app.api.hunt_live import router as hunt_live_router
from app.api.strategy_lab import router as strategy_lab_router
from app.api.trades import router as trades_router
from app.api.ws import router as ws_router
from app.core.config import settings
from app.core.database import init_db
from app.services.bias_cache import bias_cache
from app.services.hyper_scalp_worker import hyper_scalp_worker
from app.services.markov_cache import markov_cache
from app.services.signal_worker import signal_worker
from app.services.trade_calibrator import trade_calibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    signal_worker.start()
    hyper_scalp_worker.start()

    async def _bias_refresh_loop() -> None:
        await bias_cache.refresh(force=True)
        while True:
            await asyncio.sleep(3600)
            await bias_cache.refresh(force=True)

    async def _calibrator_refresh_loop() -> None:
        await trade_calibrator.refresh(force=True)
        await markov_cache.refresh(force=True)
        try:
            from analysis.instrument_rl import rebuild_from_trades

            rows = await trade_calibrator.load_closed_rows()
            closed = [r for r in rows if r.get("status") == "closed"]
            try:
                from app.services.desk_epoch import filter_rows_since_epoch

                closed = filter_rows_since_epoch(closed)
            except Exception:
                pass
            rl = await asyncio.to_thread(rebuild_from_trades, closed, ignore_epoch=False)
            logger.info(
                "Instrument RL bootstrapped: %s updates across %s symbols",
                rl.get("total_updates"),
                len(rl.get("instruments") or {}),
            )
        except Exception as exc:
            logger.warning("Instrument RL bootstrap skipped: %s", exc)
        while True:
            await asyncio.sleep(300)
            await trade_calibrator.refresh(force=True)
            await markov_cache.refresh(force=True)

    bias_task = asyncio.create_task(_bias_refresh_loop())
    calibrator_task = asyncio.create_task(_calibrator_refresh_loop())
    logger.info("%s v%s started", settings.APP_NAME, settings.APP_VERSION)
    yield
    bias_task.cancel()
    calibrator_task.cancel()
    for task in (bias_task, calibrator_task):
        try:
            await task
        except asyncio.CancelledError:
            pass
    await signal_worker.stop()
    await hyper_scalp_worker.stop()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(signals_router)
app.include_router(chart_router)
app.include_router(trades_router)
app.include_router(backtest_router)
app.include_router(algo_settings_router)
app.include_router(news_router)
app.include_router(hyper_scalp_router)
app.include_router(strategy_lab_router)
app.include_router(hunt_live_router)
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
    }
