"""Executed trade records."""

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from analysis.trade_outcome import outcome_label
from app.core.database import Base


class TradeRecord(Base):
    __tablename__ = "trade_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    candle_time: Mapped[str] = mapped_column(String(64), index=True)
    log_time: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    signal: Mapped[str] = mapped_column(String(16))
    entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rr_type: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    partial_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_status: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    oanda_transaction_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    extreme_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lot_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    margin_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sl_trail_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    features: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "candle_time": self.candle_time,
            "log_time": self.log_time,
            "signal": self.signal,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "rr_type": self.rr_type,
            "partial_pnl": self.partial_pnl,
            "current_price": self.current_price,
            "pnl": self.pnl,
            "pnl_status": self.pnl_status,
            "status": self.status,
            "message": self.message,
            "oanda_transaction_id": self.oanda_transaction_id,
            "extreme_price": self.extreme_price,
            "lot_size": self.lot_size,
            "margin_usd": self.margin_usd,
            "sl_trail_status": self.sl_trail_status,
            "features": json.loads(self.features) if self.features else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if self.status == "closed" and self.signal in ("BUY", "SELL"):
            d["outcome"] = outcome_label(self)
        return d
