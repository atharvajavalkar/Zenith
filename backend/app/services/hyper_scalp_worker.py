"""Hyper-scalp worker — sub-second cycle, parallel punches across INSTRUMENT_UNIVERSE."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Throttle heavy warm-up work so paper/TV mode cannot stall the whole loop.
_BIAS_REFRESH_SEC = 45.0
_VP_REFRESH_SEC = 8.0
_FLOW_PROBE_SEC = 12.0
_CANDLE_EDGE_SEC = 55.0  # ~every new 1m candle
_last_bias_ts: dict[str, float] = {}
_last_vp_ts: dict[str, float] = {}
_last_flow_ts: dict[str, float] = {}
_last_candle_edge_ts: float = 0.0


class HyperScalpWorker:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._busy = False

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run())
        logger.info("Hyper-scalp worker started")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Hyper-scalp worker stopped")

    async def _run(self) -> None:
        while not self._stop.is_set():
            cycle = 0.4
            try:
                from strategy.hyper_scalp import get_state

                st = get_state()
                cycle = max(0.2, float(st.get("cycle_sec") or 0.4))
                # Always warm ticks (builds the live seconds-candles), then punch
                await asyncio.to_thread(self._warm_ticks)
                if st.get("enabled"):
                    await self._cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Hyper-scalp cycle error: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=cycle)
                break
            except asyncio.TimeoutError:
                pass

    def _fast_quote(self, sym: str) -> dict[str, Any]:
        """Prefer MetaAPI ticks when configured — even in paper mode.

        Paper only means fills are simulated. Using XM prices keeps Hyper's
        10s candle builder fed at machine speed instead of hanging on TV.
        """
        try:
            from data.mt5_client import mt5_client

            if mt5_client.configured and not mt5_client._in_backoff():  # noqa: SLF001
                quote = mt5_client.current_price(symbol=sym)
                if quote and quote.get("price") is not None:
                    from strategy.instrument_meta import price_digits

                    dig = price_digits(sym)
                    px = round(float(quote["price"]), dig)
                    return {
                        "price": px,
                        "high": px,
                        "low": px,
                        "bid": quote.get("bid"),
                        "ask": quote.get("ask"),
                        "source": "xm",
                        "symbol": quote.get("symbol") or sym,
                    }
        except Exception as exc:
            logger.debug("Hyper XM quote %s failed: %s", sym, exc)
        from data.market_data import get_tick_quote

        return get_tick_quote(sym) or {}

    def _warm_ticks(self) -> None:
        try:
            from config import INSTRUMENT_UNIVERSE
            from strategy.hyper_scalp import get_state, record_tick
            from strategy.hyper_vp import refresh_for_symbol
            from strategy.hyper_momentum import punch_order, probe_momentum_board
            from strategy.hyper_flow import compute_flow_bias

            st = get_state()
            tf = str(st.get("ema_tf") or "1m")
            bias_tf = str(st.get("bias_tf") or "5m")
            momentum_settings = {
                "entry_tf": tf,
                "bias_tf": bias_tf,
                "disp_atr": st.get("disp_atr", 0.65),
                "pull_max_atr": st.get("pull_max_atr", 0.85),
                "tp_r": st.get("tp_r", 1.5),
                "tp1_r": st.get("tp1_r", 1.0),
                "tp1_frac": st.get("tp1_frac", 0.5),
                "cost_mult": st.get("cost_mult", 3.5),
                "min_range_bps": st.get("min_range_bps", 1.0),
                "swing_lookback": st.get("swing_lookback", 10),
            }
            now = time.time()
            for sym in punch_order(INSTRUMENT_UNIVERSE):
                try:
                    q = self._fast_quote(sym)
                    px = q.get("price")
                    if px is None:
                        continue
                    record_tick(sym, float(px))
                    last_vp = float(_last_vp_ts.get(sym) or 0)
                    if now - last_vp >= _VP_REFRESH_SEC:
                        refresh_for_symbol(sym, float(px), timeframe=tf)
                        _last_vp_ts[sym] = now
                    last_b = float(_last_bias_ts.get(sym) or 0)
                    if now - last_b >= _BIAS_REFRESH_SEC:
                        compute_flow_bias(
                            sym,
                            settings={"bias_tf": bias_tf},
                        )
                        _last_bias_ts[sym] = now
                    last_f = float(_last_flow_ts.get(sym) or 0)
                    if now - last_f >= _FLOW_PROBE_SEC:
                        probe_momentum_board(sym, price=float(px), settings=momentum_settings)
                        _last_flow_ts[sym] = now
                except Exception:
                    continue
            # Candle walk-forward even when Hyper is in standby — keeps knobs sharp
            global _last_candle_edge_ts
            if now - float(_last_candle_edge_ts or 0) >= _CANDLE_EDGE_SEC:
                try:
                    from analysis.candle_edge import run_candle_edge

                    run_candle_edge(candle_key=f"hyper-{int(now // 60)}")
                    _last_candle_edge_ts = now
                except Exception:
                    pass
        except Exception:
            pass

    async def _cycle(self) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            from config import INSTRUMENT_UNIVERSE
            from strategy.hyper_scalp import risk_guard_ok

            # Manage exits first — TP/SL (and safety time-cap) — serialize on the
            # shared MetaAPI client so symbols never stomp each other.
            await asyncio.to_thread(self._manage_exits)

            # Capital drawdown guard — auto-disarms on breach; no new punches
            if not await asyncio.to_thread(risk_guard_ok):
                return

            # Score-ranked punches — highest edge-per-cost first
            from strategy.hyper_momentum import score_rank

            for sym in score_rank(INSTRUMENT_UNIVERSE):
                await asyncio.to_thread(self._punch_symbol, sym)
        finally:
            self._busy = False

    def _punch_symbol(self, sym: str) -> None:
        from strategy.hyper_scalp import build_hyper_signal, mark_punched
        from strategy.instrument_scanner import instrument_tradable
        from strategy.trade_executor import execute_signal_trade

        try:
            if not instrument_tradable(sym):
                return
            q = self._fast_quote(sym)
            px = q.get("price")
            if px is None:
                return
            sig = build_hyper_signal(sym, float(px))
            if not sig:
                return
            result = execute_signal_trade(sig, force=True)
            mark_punched(sym, sig, result)
            status = str(result.get("status") or "")
            if status in ("ok", "filled", "executed", "open") or result.get("order_id"):
                logger.info(
                    "HYPER PUNCH %s %s lot=%s mode=%s dist=%s → %s",
                    sym,
                    sig.get("SIGNAL"),
                    sig.get("LOT_SIZE"),
                    sig.get("hyper_mode"),
                    sig.get("dist_atr"),
                    status or result.get("message"),
                )
        except Exception as exc:
            logger.debug("Hyper %s failed: %s", sym, exc)

    def _manage_exits(self) -> None:
        """Close open Hyper positions that hit TP or SL (or the safety time-cap).

        Runs every cycle. In paper mode this is the ONLY exit path (no broker
        holds the stops); in demo/live it backstops the broker's own TP/SL.
        """
        import time as _time

        from data.market_data import get_tick_quote
        from strategy.hyper_scalp import get_state, open_positions

        st = get_state()
        time_stop_on = int(st.get("time_stop_enabled") or 0)
        hard_cap = float(st.get("max_hold_sec") or 900)
        now = _time.time()

        for meta in open_positions():
            sym = str(meta.get("symbol") or "")
            sig = str(meta.get("signal") or "").upper()
            tp = float(meta.get("tp") or 0.0)
            sl = float(meta.get("sl") or 0.0)
            opened = float(meta.get("opened_at") or 0.0)
            hold = float(meta.get("max_hold_sec") or hard_cap)

            px: Optional[float] = None
            try:
                q = self._fast_quote(sym)
                if q and q.get("price") is not None:
                    px = float(q["price"])
            except Exception:
                px = None

            # TP1 partial + breakeven — book half at +1R, run the rest risk-free
            if px is not None and not meta.get("partial_done"):
                tp1 = float(meta.get("tp1") or 0.0)
                hit_tp1 = tp1 > 0 and (
                    (sig == "BUY" and px >= tp1) or (sig == "SELL" and px <= tp1)
                )
                if hit_tp1:
                    self._book_partial(meta, tp1)
                    continue  # re-evaluate the runner next cycle

            reason: Optional[str] = None
            exit_px: Optional[float] = px
            if px is not None:
                if sig == "BUY":
                    if tp > 0 and px >= tp:
                        reason, exit_px = "tp", tp
                    elif sl > 0 and px <= sl:
                        reason, exit_px = "sl", sl
                elif sig == "SELL":
                    if tp > 0 and px <= tp:
                        reason, exit_px = "tp", tp
                    elif sl > 0 and px >= sl:
                        reason, exit_px = "sl", sl

            # Forced timer (only if enabled) then hard safety cap
            if reason is None and opened:
                if time_stop_on and now - opened >= hold:
                    reason = "time_stop"
                elif now - opened >= max(hold, hard_cap):
                    reason = "max_hold"

            if reason:
                self._force_exit(meta, reason=reason, exit_px=exit_px)

    def _book_partial(self, meta: dict[str, Any], tp1_px: float) -> None:
        """Close tp1_frac of the position at TP1 and move the broker SL to entry."""
        from data.broker import broker_client
        from strategy.hyper_scalp import mark_partial

        tag = str(meta.get("tag") or "")
        sym = str(meta.get("symbol") or "")
        pid = meta.get("order_id")
        lot = float(meta.get("lot") or 0.0)
        frac = float(meta.get("tp1_frac") or 0.5)
        closed_lot = max(0.01, round(lot * frac, 2))
        if closed_lot >= lot:
            # Position too small to split — let the full TP/SL handle it
            return
        entry = float(meta.get("entry") or 0.0)
        try:
            import runtime_config

            if runtime_config.is_broker_mode() and broker_client.configured and pid:
                try:
                    if hasattr(broker_client, "partial_close"):
                        broker_client.partial_close(str(pid), closed_lot)
                except Exception as e1:
                    logger.warning("Hyper TP1 partial_close %s failed: %s", pid, e1)
                try:
                    if hasattr(broker_client, "modify_position") and entry > 0:
                        broker_client.modify_position(str(pid), stop_loss=entry)
                except Exception as e2:
                    logger.warning("Hyper TP1 SL→BE %s failed: %s", pid, e2)
            mark_partial(tag, exit_price=float(tp1_px), closed_lot=closed_lot)
            logger.info("HYPER TP1 %s booked %s lot @ %s · SL → BE", sym, closed_lot, tp1_px)
        except Exception as exc:
            logger.warning("Hyper TP1 %s failed: %s", sym, exc)

    def _force_exit(
        self,
        meta: dict[str, Any],
        reason: str = "tp",
        exit_px: Optional[float] = None,
    ) -> None:
        from data.broker import broker_client
        from data.market_data import get_tick_quote
        from strategy.hyper_scalp import clear_open

        tag = str(meta.get("tag") or "")
        sym = str(meta.get("symbol") or "")
        pid = meta.get("order_id")
        if exit_px is None:
            try:
                q = get_tick_quote(sym)
                if q and q.get("price") is not None:
                    exit_px = float(q["price"])
            except Exception:
                exit_px = None
        try:
            import runtime_config

            if runtime_config.is_broker_mode() and broker_client.configured:
                closed = False
                if pid and hasattr(broker_client, "close_position"):
                    try:
                        broker_client.close_position(str(pid))
                        closed = True
                    except Exception as e1:
                        logger.warning("Hyper close_position %s failed: %s — retry symbol close", pid, e1)
                if not closed and sym:
                    # Scoped close for this symbol only — never close_all across the desk
                    with broker_client.using_symbol(sym):
                        if hasattr(broker_client, "close_symbol_positions"):
                            broker_client.close_symbol_positions(sym)
                        elif hasattr(broker_client, "close_all_positions"):
                            # last resort, still inside using_symbol scope
                            broker_client.close_all_positions()
            clear_open(tag, reason=reason, exit_price=exit_px)
            logger.info("HYPER EXIT %s %s pid=%s px=%s", reason, sym, pid, exit_px)
        except Exception as e:
            logger.warning("Hyper exit failed %s (%s): %s", sym, reason, e)
            clear_open(tag, reason=f"{reason}_error", exit_price=exit_px)


hyper_scalp_worker = HyperScalpWorker()
