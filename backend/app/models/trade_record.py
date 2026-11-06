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
    trade_mode: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    extreme_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lot_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    margin_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sl_trail_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    features: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def _resolved_symbol(self) -> Optional[str]:
        if self.symbol:
            return self.symbol
        if self.features:
            try:
                feats = json.loads(self.features) if isinstance(self.features, str) else self.features
                if isinstance(feats, dict):
                    return feats.get("symbol") or feats.get("SYMBOL") or feats.get("instrument")
            except Exception:
                pass
        return None

    def _repaired_pnl(self) -> Optional[float]:
        """
        Recompute P&L when stored value is ~0 but the trade clearly moved
        (legacy bug: FX used gold contract size → rounded to $0).

        Never invent PnL for XM-linked closes — broker deal cash is truth.
        """
        stored = self.pnl
        if self.signal not in ("BUY", "SELL") or self.entry is None:
            return stored

        # Prefer stamped XM deal cash from reconcile / close sync
        try:
            feats = (
                json.loads(self.features)
                if isinstance(self.features, str)
                else (self.features or {})
            )
            if isinstance(feats, dict) and feats.get("pnl_source") == "xm_deal":
                if feats.get("xm_deal_pnl") is not None:
                    return float(feats["xm_deal_pnl"])
        except Exception:
            pass

        msg = str(self.message or "")
        if "XM P&L" in msg:
            return stored
        # Any closed row tied to an XM position id — trust stored (reconciler keeps it true)
        if self.oanda_transaction_id and (
            self.status == "closed" or str(self.pnl_status or "").startswith("closed")
        ):
            return stored

        exit_px = self.current_price
        if exit_px is None:
            return stored
        is_closed = self.status == "closed" or (
            self.pnl_status or ""
        ).startswith("closed")
        if not is_closed and abs(float(stored or 0)) >= 0.01:
            return stored
        if abs(float(stored or 0)) >= 0.05:
            return stored
        if "Closed on XM" in msg or "Closed on METAAPI" in msg:
            return stored
        try:
            from strategy.position_sizing import leg_pnl

            lot = self.lot_size or 0.1
            if self.features:
                try:
                    feats = (
                        json.loads(self.features)
                        if isinstance(self.features, str)
                        else self.features
                    )
                    if isinstance(feats, dict) and feats.get("entry_lot"):
                        lot = float(feats["entry_lot"])
                except Exception:
                    pass
            fixed = leg_pnl(
                self.signal,
                float(self.entry),
                float(exit_px),
                lot,
                symbol=self._resolved_symbol() or "GOLD",
            )
            if abs(fixed) > abs(float(stored or 0)):
                return fixed
        except Exception:
            pass
        return stored

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "candle_time": self.candle_time,
            "log_time": self.log_time,
            "signal": self.signal,
            "symbol": self._resolved_symbol() or "GOLD",
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "rr_type": self.rr_type,
            "partial_pnl": self.partial_pnl,
            "current_price": self.current_price,
            "pnl": self._repaired_pnl(),
            "pnl_status": self.pnl_status,
            "status": self.status,
            "message": self.message,
            "oanda_transaction_id": self.oanda_transaction_id,
            "broker_position_id": self.oanda_transaction_id,
            "trade_mode": self.trade_mode,
            "extreme_price": self.extreme_price,
            "lot_size": self.lot_size,
            "sl_trail_status": self.sl_trail_status,
            "features": json.loads(self.features) if self.features else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if self.status == "closed" and self.signal in ("BUY", "SELL"):
            d["outcome"] = outcome_label(self)
        return d
