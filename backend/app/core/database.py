"""Async database engine and session."""

from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.database_url,
    echo=settings.DEBUG,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    from app.models import signal_snapshot, trade_record, user  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_trade_record_columns(conn)
        await _widen_signal_snapshot_columns(conn)


async def _widen_signal_snapshot_columns(conn) -> None:
    """Widen columns whose content grew (e.g. dual-TP exit rule)."""
    alters = {
        "exit_rule": "TEXT",
        "state": "VARCHAR(64)",
        "context_4h": "VARCHAR(64)",
    }
    for name, column_type in alters.items():
        try:
            await conn.execute(
                text(f"ALTER TABLE signal_snapshots ALTER COLUMN {name} TYPE {column_type}")
            )
        except Exception:
            # SQLite (and others) may not support ALTER COLUMN TYPE — ignore.
            pass


async def _ensure_trade_record_columns(conn) -> None:
    """Add columns introduced after the first create_all run."""
    columns = {
        "log_time": "VARCHAR(64)",
        "current_price": "FLOAT",
        "pnl": "FLOAT",
        "pnl_status": "VARCHAR(24)",
        "extreme_price": "FLOAT",
        "tp1": "FLOAT",
        "tp2": "FLOAT",
        "tp3": "FLOAT",
        "rr_type": "VARCHAR(8)",
        "partial_pnl": "FLOAT",
        "features": "TEXT",
        "lot_size": "FLOAT",
        "margin_usd": "FLOAT",
        "sl_trail_status": "VARCHAR(64)",
        "trade_mode": "VARCHAR(8)",
        "symbol": "VARCHAR(32)",
    }
    for name, column_type in columns.items():
        try:
            await conn.execute(
                text(f"ALTER TABLE trade_records ADD COLUMN IF NOT EXISTS {name} {column_type}")
            )
        except Exception:
            # SQLite does not support IF NOT EXISTS for ADD COLUMN on older versions.
            try:
                await conn.execute(text(f"ALTER TABLE trade_records ADD COLUMN {name} {column_type}"))
            except Exception:
                pass
