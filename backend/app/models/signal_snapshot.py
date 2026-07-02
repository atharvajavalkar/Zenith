"""Persisted signal snapshot model."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SignalSnapshot(Base):
    __tablename__ = "signal_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    candle_time: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    live_time: Mapped[str] = mapped_column(String(32))
    live_price: Mapped[float] = mapped_column(Float)
    signal: Mapped[str] = mapped_column(String(16))
    bias_1h: Mapped[str] = mapped_column(String(16))
    bias_4h: Mapped[str] = mapped_column(String(16))
    context_4h: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(64))
    success_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rr: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    demand_tf: Mapped[str] = mapped_column(String(8))
    supply_tf: Mapped[str] = mapped_column(String(8))
    exit_rule: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    @classmethod
    def from_signal_dict(cls, sig: dict) -> "SignalSnapshot":
        return cls(
            candle_time=sig["CANDLE_TIME"],
            live_time=sig["LIVE_TIME"],
            live_price=sig["LIVE_PRICE"],
            signal=sig["SIGNAL"],
            bias_1h=sig["BIAS_1H"],
            bias_4h=sig["BIAS_4H"],
            context_4h=sig["CONTEXT_4H"],
            state=sig["STATE"],
            success_percent=sig["SUCCESS_PERCENT"],
            rr=sig["RR"],
            entry=sig["ENTRY"],
            sl=sig["SL"],
            tp=sig["TP"],
            demand_tf=sig["DEMAND_TF"],
            supply_tf=sig["SUPPLY_TF"],
            exit_rule=sig["EXIT"],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "candle_time": self.candle_time,
            "live_time": self.live_time,
            "live_price": self.live_price,
            "signal": self.signal,
            "bias_1h": self.bias_1h,
            "bias_4h": self.bias_4h,
            "context_4h": self.context_4h,
            "state": self.state,
            "success_percent": self.success_percent,
            "rr": self.rr,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "demand_tf": self.demand_tf,
            "supply_tf": self.supply_tf,
            "exit_rule": self.exit_rule,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
