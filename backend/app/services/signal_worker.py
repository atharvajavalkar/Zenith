"""Background signal polling worker."""

import json
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

# Ensure algo core modules are importable
_ALGO_ROOT = Path(__file__).resolve().parents[3]
if str(_ALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGO_ROOT))

from config import (  # noqa: E402
    ADAPTIVE_LEARNING,
    AUTO_TRADE_ENABLED,
    BIAS_GATE_ENABLED,
    BLOCK_OPPOSITE_OPEN,
    ERROR_RETRY_SECONDS,
    MIN_BIAS_CONFIDENCE,
    ML_GATE_ENABLED,
    POLL_INTERVAL_SECONDS,
    RR_TP_FRACS,
    TICK_INTERVAL_SECONDS,
    TP1_PARTIAL_FRACTION,
    TRADE_LOT_SIZE,
)
from analysis.direction import bias_allows_trade  # noqa: E402
from analysis.trade_learning import extract_features_from_signal  # noqa: E402
from analysis.trade_outcome import is_breakeven_trade, sl_trailed_profit  # noqa: E402
from strategy.position_sizing import estimate_margin, leg_pnl  # noqa: E402
from strategy.trailing_stop import (  # noqa: E402
    apply_trailing_stop,
    sl_trail_label,
    trailing_status_message,
)
from data.fetcher import get_tick_quote  # noqa: E402
from strategy.signal import build_stacked_trade, run_signal_cycle  # noqa: E402
from strategy.trade_executor import execute_signal_trade  # noqa: E402

from app.core.database import AsyncSessionLocal
from app.models.signal_snapshot import SignalSnapshot
from app.models.trade_record import TradeRecord
from app.services.bias_cache import bias_cache
from app.services.market_cache import market_cache
from app.services.learning_loop import on_trade_closed
from app.services.trade_calibrator import trade_calibrator
from app.services.trade_settings import (
    bucket_log_time,
    get_max_trades_per_bucket,
    get_trade_cadence,
    get_trade_interval_seconds,
    is_trading_halted,
)
from app.services.ws_manager import ws_manager

logger = logging.getLogger(__name__)

OPEN_STATUSES = ("open", "partial", "partial2", "partial3")


class SignalWorker:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._tick_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._minute_bucket: Optional[str] = None
        self._trades_this_minute: int = 0
        self._last_trade_at: float = 0.0

    def _bucket_key(self, sig: dict) -> str:
        return bucket_log_time(sig, get_trade_cadence())

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run_loop())
            self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        for task in (self._task, self._tick_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _reset_minute_bucket(self, minute_key: str) -> None:
        if minute_key != self._minute_bucket:
            self._minute_bucket = minute_key
            self._trades_this_minute = 0
            self._last_trade_at = 0.0
            cadence = get_trade_cadence().upper()
            logger.info(
                "New %s bucket %s — up to %s trades",
                cadence,
                minute_key,
                get_max_trades_per_bucket(),
            )

    async def _has_opposite_open(self, direction: str) -> bool:
        if not BLOCK_OPPOSITE_OPEN:
            return False
        opposite = "SELL" if direction == "BUY" else "BUY"
        async with AsyncSessionLocal() as session:
            count = (
                await session.execute(
                    select(TradeRecord).where(
                        TradeRecord.pnl_status.in_(OPEN_STATUSES),
                        TradeRecord.signal == opposite,
                    )
                )
            ).scalars().all()
            return len(count) > 0

    async def _tick_loop(self) -> None:
        """SL/TP monitor + stacked trade entries on tick cadence."""
        logger.info(
            "Tick monitor started (interval=%ss)",
            TICK_INTERVAL_SECONDS,
        )
        while not self._stop_event.is_set():
            try:
                quote = await asyncio.to_thread(get_tick_quote)
                if quote.get("price") is not None:
                    await self._update_open_trades_from_quote(quote)
                    await self._maybe_execute_stacked_trade(quote)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Tick loop error: %s", exc)
            await asyncio.sleep(TICK_INTERVAL_SECONDS)

    async def _maybe_execute_stacked_trade(self, quote: dict) -> None:
        """Open trades on 3M/5M or 1M/5M zones — max N per candle bucket."""
        if not AUTO_TRADE_ENABLED or is_trading_halted():
            return

        sig = await market_cache.get_signal()
        if not sig or sig.get("SIGNAL") not in ("BUY", "SELL"):
            return

        bucket_key = self._bucket_key(sig)
        if not bucket_key:
            return

        max_per_bucket = get_max_trades_per_bucket()
        if max_per_bucket <= 0:
            return

        self._reset_minute_bucket(bucket_key)

        if self._trades_this_minute >= max_per_bucket:
            return

        now = time.time()
        trade_interval = get_trade_interval_seconds()
        if self._trades_this_minute > 0 and now - self._last_trade_at < trade_interval:
            return

        direction = sig["SIGNAL"]
        if await self._has_opposite_open(direction):
            logger.debug("Skipping %s — opposite direction still open", direction)
            return

        if BIAS_GATE_ENABLED:
            bias_data = await bias_cache.get()
            allowed, bias_reason = bias_allows_trade(direction, bias_data, MIN_BIAS_CONFIDENCE)
            if not allowed:
                logger.info("AI bias gate blocked: %s", bias_reason)
                return
        else:
            bias_reason = "bias gate off"

        trade_sig = build_stacked_trade(sig, float(quote["price"]))

        ml_features = extract_features_from_signal(sig, trade_sig)
        ml_features["zone"] = sig.get("DEMAND_TF") if direction == "BUY" else sig.get("SUPPLY_TF")
        if ML_GATE_ENABLED or ADAPTIVE_LEARNING:
            allowed, ml_prob, ml_reason, ml_features = await trade_calibrator.evaluate(sig, trade_sig)
            if not allowed:
                logger.info("Learning gate blocked trade: %s", ml_reason)
                return
            ml_features["bias_gate"] = bias_reason if BIAS_GATE_ENABLED else "off"

        result = await asyncio.to_thread(execute_signal_trade, trade_sig)
        if result.get("status") != "executed":
            return

        self._trades_this_minute += 1
        self._last_trade_at = now
        seq = self._trades_this_minute
        log_time = f"{bucket_key}#{seq:02d}"

        await self._persist_trade_log(result, trade_sig, log_time=log_time, features=ml_features)
        cadence = get_trade_cadence().upper()
        logger.info(
            "%s trade %s/%s: %s @ %s (bucket %s, zone %s/%s)",
            cadence,
            seq,
            max_per_bucket,
            trade_sig["SIGNAL"],
            trade_sig["ENTRY"],
            bucket_key,
            sig.get("DEMAND_TF"),
            sig.get("SUPPLY_TF"),
        )
        await ws_manager.broadcast(
            {
                "type": "trade_executed",
                "signal": trade_sig,
                "trade": result,
            }
        )

    async def _run_loop(self) -> None:
        await market_cache.set_worker_running(True)
        logger.info("Signal worker started (poll=%ss)", POLL_INTERVAL_SECONDS)
        while not self._stop_event.is_set():
            try:
                cycle = await asyncio.to_thread(run_signal_cycle)
                is_new = await market_cache.update(cycle)
                if is_new:
                    await self._persist_snapshot(cycle["signal"])

                sig = cycle["signal"]
                bucket_key = self._bucket_key(sig)
                if bucket_key:
                    self._reset_minute_bucket(bucket_key)

                quote = {
                    "price": sig.get("LOG_PRICE_1M") or sig.get("LIVE_PRICE"),
                    "high": sig.get("LOG_HIGH_1M"),
                    "low": sig.get("LOG_LOW_1M"),
                }
                if quote["price"] is not None:
                    await self._update_open_trades_from_quote(quote)

                await ws_manager.broadcast(
                    {
                        "type": "signal_update",
                        "signal": sig,
                    }
                )
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Signal worker error: %s", exc)
                await market_cache.set_error(str(exc))
                await asyncio.sleep(ERROR_RETRY_SECONDS)
        await market_cache.set_worker_running(False)
        logger.info("Signal worker stopped")

    async def _persist_snapshot(self, sig: dict) -> None:
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                select(SignalSnapshot).where(
                    SignalSnapshot.candle_time == sig["CANDLE_TIME"]
                )
            )
            if existing.scalar_one_or_none():
                return
            session.add(SignalSnapshot.from_signal_dict(sig))
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()

    async def _persist_trade_log(
        self, result: dict, sig: dict, log_time: Optional[str] = None, features: Optional[dict] = None
    ) -> None:
        async with AsyncSessionLocal() as session:
            log_time = log_time or sig.get("LOG_TIME_1M") or sig.get("CANDLE_TIME", "")
            existing = await session.execute(
                select(TradeRecord).where(TradeRecord.log_time == log_time)
            )
            if existing.scalar_one_or_none():
                return

            signal = sig.get("SIGNAL", result.get("signal", "NO TRADE"))
            status = result["status"]
            message = result.get("message", "")

            entry = sig.get("ENTRY")
            live_px = sig.get("LOG_PRICE_1M") or sig.get("LIVE_PRICE")
            lot = TRADE_LOT_SIZE
            margin = estimate_margin(float(entry), lot) if entry is not None else None

            rr_str = sig.get("RR")
            session.add(
                TradeRecord(
                    candle_time=sig.get("CANDLE_TIME", ""),
                    log_time=log_time,
                    signal=signal,
                    entry=entry,
                    sl=sig.get("SL"),
                    tp=sig.get("TP"),
                    tp1=sig.get("TP1"),
                    tp2=sig.get("TP2"),
                    tp3=sig.get("TP3"),
                    rr_type=rr_str,
                    current_price=live_px,
                    pnl=self._calculate_pnl(signal, entry, live_px, lot),
                    partial_pnl=0.0,
                    lot_size=lot,
                    margin_usd=margin,
                    sl_trail_status=sl_trail_label(
                        signal, float(entry), float(sig.get("SL")), "open"
                    ) if entry and sig.get("SL") else "Fixed SL",
                    features=json.dumps(features) if features else None,
                    pnl_status="open" if status == "executed" else None,
                    status=status,
                    message=message,
                    oanda_transaction_id=result.get("oanda_transaction_id"),
                )
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()

    # -----------------------------------------------------------------------
    # Multi-TP trade management helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _rr_int(row: TradeRecord) -> int:
        """Extract the integer RR from rr_type ('1:3' → 3).  Default 2 for legacy trades."""
        try:
            return int(str(row.rr_type or "1:2").split(":")[-1])
        except (ValueError, AttributeError):
            return 2

    @staticmethod
    def _tp_fracs(rr_n: int) -> list[float]:
        """Partial fractions for each TP stage as a fraction of the full position."""
        return RR_TP_FRACS.get(rr_n, RR_TP_FRACS[2])

    @staticmethod
    def _price_hit(signal: str, level: float, high: float, low: float) -> bool:
        if signal == "BUY":
            return float(high) >= float(level)
        return float(low) <= float(level)

    def _runner_fraction(self, row: TradeRecord) -> float:
        """Fraction of the original position still running (not yet booked)."""
        rr_n = self._rr_int(row)
        fracs = self._tp_fracs(rr_n)
        stage_idx = {"open": 0, "partial": 1, "partial2": 2, "partial3": 3}.get(
            row.pnl_status or "open", 0
        )
        booked = sum(fracs[:stage_idx])
        return max(0.0, round(1.0 - booked, 6))

    def _leg_pnl(self, signal: str, entry: float, price: float, lot_size: float | None = None) -> float:
        return leg_pnl(signal, entry, price, lot_size or TRADE_LOT_SIZE)

    def _total_pnl(self, row: TradeRecord, price: float) -> float:
        partial = row.partial_pnl or 0.0
        lot = row.lot_size or TRADE_LOT_SIZE
        runner = self._leg_pnl(row.signal, row.entry, price, lot) * self._runner_fraction(row)
        return round(partial + runner, 2)

    def _book_partial(
        self, row: TradeRecord, tp_price: float, frac: float, next_status: str, lot: float
    ) -> None:
        """Book `frac` of the full position at `tp_price` and advance the stage."""
        booked_pnl = leg_pnl(row.signal, row.entry, tp_price, lot) * frac
        row.partial_pnl = round((row.partial_pnl or 0.0) + booked_pnl, 2)
        row.pnl_status = next_status

    def _next_tp_target(self, row: TradeRecord) -> Optional[float]:
        """Return the TP level the trailing stop should move toward now."""
        rr_n = self._rr_int(row)
        if row.pnl_status == "partial":
            return row.tp2 if rr_n >= 3 else row.tp
        if row.pnl_status == "partial2":
            return row.tp3 if rr_n >= 4 else row.tp
        return row.tp

    async def _update_open_trades_from_quote(self, quote: dict) -> None:
        close_price = quote.get("price")
        if close_price is None:
            return
        high = quote.get("high", close_price)
        low = quote.get("low", close_price)

        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(TradeRecord).where(TradeRecord.pnl_status.in_(OPEN_STATUSES))
                )
            ).scalars().all()
            if not rows:
                return

            changed = False
            closed_any = False
            for row in rows:
                old_pnl = row.pnl
                rr_n = self._rr_int(row)
                fracs = self._tp_fracs(rr_n)

                # --- TP1 hit (all setups) ---
                if row.pnl_status == "open" and row.tp1 is not None:
                    if self._price_hit(row.signal, float(row.tp1), high, low):
                        lot = row.lot_size or TRADE_LOT_SIZE
                        self._book_partial(row, float(row.tp1), fracs[0], "partial", lot)
                        row.sl = row.entry  # SL → breakeven
                        pct = int(fracs[0] * 100)
                        row.message = (
                            f"TP1 hit @ {row.tp1:.2f} — {pct}% booked, SL → breakeven"
                        )
                        row.sl_trail_status = sl_trail_label(
                            row.signal, float(row.entry), float(row.entry), "partial"
                        )
                        changed = True

                # --- TP2 hit (1:3 and 1:4 only) ---
                if row.pnl_status == "partial" and rr_n >= 3 and row.tp2 is not None:
                    if self._price_hit(row.signal, float(row.tp2), high, low):
                        lot = row.lot_size or TRADE_LOT_SIZE
                        self._book_partial(row, float(row.tp2), fracs[1], "partial2", lot)
                        row.sl = float(row.tp1)  # SL → TP1 (lock 1R profit)
                        pct = int(fracs[1] * 100)
                        row.message = (
                            f"TP2 hit @ {row.tp2:.2f} — {pct}% booked, SL → TP1 ({row.tp1:.2f})"
                        )
                        row.sl_trail_status = sl_trail_label(
                            row.signal, float(row.entry), float(row.sl), "partial2"
                        )
                        changed = True

                # --- TP3 hit (1:4 only) ---
                if row.pnl_status == "partial2" and rr_n >= 4 and row.tp3 is not None:
                    if self._price_hit(row.signal, float(row.tp3), high, low):
                        lot = row.lot_size or TRADE_LOT_SIZE
                        self._book_partial(row, float(row.tp3), fracs[2], "partial3", lot)
                        row.sl = float(row.tp2)  # SL → TP2 (lock 2R profit)
                        pct = int(fracs[2] * 100)
                        row.message = (
                            f"TP3 hit @ {row.tp3:.2f} — {pct}% booked, SL → TP2 ({row.tp2:.2f})"
                        )
                        row.sl_trail_status = sl_trail_label(
                            row.signal, float(row.entry), float(row.sl), "partial3"
                        )
                        changed = True

                # --- Dynamic trailing on the runner (all partial stages) ---
                if row.pnl_status in ("partial", "partial2", "partial3"):
                    next_tp = self._next_tp_target(row)
                    if next_tp is not None:
                        new_sl, new_extreme, trail_changed = apply_trailing_stop(
                            row.signal,
                            row.entry,
                            next_tp,
                            row.sl,
                            row.extreme_price,
                            high,
                            low,
                            active=True,
                        )
                        if trail_changed or new_extreme != row.extreme_price:
                            row.sl = new_sl
                            row.extreme_price = new_extreme
                            changed = True
                            row.sl_trail_status = sl_trail_label(
                                row.signal,
                                float(row.entry),
                                float(new_sl),
                                row.pnl_status,
                                float(new_extreme) if new_extreme is not None else None,
                            )
                            row.message = trailing_status_message(
                                row.signal,
                                float(row.entry),
                                float(next_tp),
                                float(new_sl),
                                float(new_extreme),
                                partial=True,
                                stage=row.pnl_status,
                            )

                # --- SL / final TP exit check ---
                exit_status, exit_price = self._resolve_exit(
                    row.signal,
                    row.sl,
                    row.tp1,
                    row.tp,           # always the final TP
                    row.pnl_status,
                    low,
                    high,
                    close_price,
                )

                row.current_price = exit_price
                row.pnl = self._total_pnl(row, exit_price)

                if exit_status:
                    row.pnl_status = exit_status
                    row.status = "closed"
                    row.sl_trail_status = "Closed"
                    changed = True
                    closed_any = True
                    if exit_status == "closed_tp":
                        final_label = f"TP{rr_n}" if rr_n > 1 else "TP"
                        row.message = f"Closed at {final_label} {exit_price:.2f}"
                    elif exit_status == "closed_sl":
                        if is_breakeven_trade(row):
                            row.message = f"Closed at SL {row.sl:.2f} (breakeven)"
                        elif sl_trailed_profit(row.signal, row.entry, row.sl):
                            row.message = f"Closed at SL {row.sl:.2f} (trailed — win)"
                        else:
                            row.message = f"Closed at SL {row.sl:.2f}"
                elif row.pnl != old_pnl:
                    changed = True

                if not exit_status and row.pnl_status in OPEN_STATUSES and row.entry and row.sl:
                    row.sl_trail_status = sl_trail_label(
                        row.signal,
                        float(row.entry),
                        float(row.sl),
                        row.pnl_status,
                        float(row.extreme_price) if row.extreme_price is not None else None,
                    )

            if changed:
                await session.commit()
                if closed_any:
                    await ws_manager.broadcast({"type": "trade_closed"})
                    await on_trade_closed()

    @staticmethod
    def _hit_tp1(signal: str, tp1: float, high: float, low: float) -> bool:
        if signal == "BUY":
            return float(high) >= float(tp1)
        return float(low) <= float(tp1)

    @staticmethod
    def _resolve_exit(
        signal: Optional[str],
        sl: Optional[float],
        tp1: Optional[float],
        tp_final: Optional[float],
        pnl_status: Optional[str],
        low: Optional[float],
        high: Optional[float],
        close: Optional[float],
    ) -> tuple[Optional[str], Optional[float]]:
        """Check SL hit and final TP hit only.  Intermediate TP stages are handled separately."""
        if signal not in ("BUY", "SELL"):
            return None, close
        low_f = float(low) if low is not None else None
        high_f = float(high) if high is not None else None
        close_f = float(close) if close is not None else None
        sl_f = float(sl) if sl is not None else None
        tp1_f = float(tp1) if tp1 is not None else None
        tp_f = float(tp_final) if tp_final is not None else None

        if signal == "BUY":
            if sl_f is not None and low_f is not None and low_f <= sl_f:
                return "closed_sl", sl_f
            # Final TP hit when in last runner stage or no TP1 configured
            if tp_f is not None and high_f is not None and high_f >= tp_f:
                if pnl_status in ("partial", "partial2", "partial3"):
                    return "closed_tp", tp_f
                if pnl_status == "open" and tp1_f is None:
                    return "closed_tp", tp_f
        elif signal == "SELL":
            if sl_f is not None and high_f is not None and high_f >= sl_f:
                return "closed_sl", sl_f
            if tp_f is not None and low_f is not None and low_f <= tp_f:
                if pnl_status in ("partial", "partial2", "partial3"):
                    return "closed_tp", tp_f
                if pnl_status == "open" and tp1_f is None:
                    return "closed_tp", tp_f
        return None, close_f

    @staticmethod
    def _calculate_pnl(
        signal: Optional[str],
        entry: Optional[float],
        price: Optional[float],
        lot_size: float | None = None,
    ):
        if signal not in ("BUY", "SELL") or entry is None or price is None:
            return 0.0
        return leg_pnl(signal, entry, price, lot_size or TRADE_LOT_SIZE)


signal_worker = SignalWorker()
