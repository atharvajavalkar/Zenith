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
    CRYPTO_L3_DEPTH_LEVELS,
    CRYPTO_L3_FAIL_OPEN,
    CRYPTO_L3_GATE_ENABLED,
    CRYPTO_L3_MAX_ADVERSE_IMBALANCE,
    CRYPTO_L3_MAX_SPREAD_BPS,
    CRYPTO_L3_MIN_RISK_SPREAD_MULT,
    CRYPTO_L3_MIN_TOP_NOTIONAL_USD,
    ERROR_RETRY_SECONDS,
    MAX_OPEN_PER_SYMBOL,
    MIN_BIAS_CONFIDENCE,
    ML_GATE_ENABLED,
    POLL_INTERVAL_LIVE_SECONDS,
    POLL_INTERVAL_SECONDS,
    TICK_INTERVAL_LIVE_SECONDS,
    TICK_INTERVAL_SECONDS,
    TRADE_LOT_SIZE,
)
from analysis.direction import bias_allows_trade  # noqa: E402
from analysis.trade_learning import extract_features_from_signal  # noqa: E402
from strategy.position_sizing import estimate_margin, leg_pnl  # noqa: E402
from strategy.trailing_stop import sl_trail_label  # noqa: E402
from strategy.trade_simulator import update_trade_on_bar  # noqa: E402
from strategy.live_broker_actions import (  # noqa: E402
    apply_live_broker_actions,
    apply_trade_dict_to_row,
    is_live_managed_row,
    row_to_trade_dict,
)
from data.fetcher import get_tick_quote  # noqa: E402
from strategy.signal import build_stacked_trade, run_signal_cycle, scan_mtf_signals  # noqa: E402
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
        self._mtf_cache: list = []
        self._mtf_cache_at: float = 0.0
        self._universe_cache: list = []
        self._universe_cache_at: float = 0.0
        self._last_frames: Optional[dict] = None
        self._eth_last_entry_at: float = 0.0  # wall-clock cooldown for ETH spam brake
        self._symbol_last_entry_at: dict[str, float] = {}  # anti-spam: same symbol multi-punch

    def _eth_cooldown_ok(self) -> bool:
        """Desk mastery C — ETH needs longer gap between new entries."""
        try:
            from config import ETH_ENTRY_COOLDOWN_BARS, POLL_INTERVAL_SECONDS

            bars = int(ETH_ENTRY_COOLDOWN_BARS or 12)
        except Exception:
            bars = 12
        # 5m bars → seconds (TRADE_5M_MODE)
        need = max(60.0, bars * 300.0)
        return (time.time() - float(self._eth_last_entry_at or 0)) >= need

    def _symbol_cooldown_ok(self, sym: str) -> bool:
        """Block stacking the same symbol within 90s (NAS/UKOIL triple-punch fix)."""
        last = float(self._symbol_last_entry_at.get(str(sym).upper(), 0) or 0)
        return (time.time() - last) >= 90.0

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

        try:
            import runtime_config
            if runtime_config.is_broker_mode():
                from data.broker import broker_client
                if broker_client.configured and broker_client.has_open_position(opposite):
                    return True
        except Exception:
            pass

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

    def _tick_interval(self) -> float:
        try:
            import runtime_config

            if runtime_config.is_broker_mode():
                return TICK_INTERVAL_LIVE_SECONDS
        except Exception:
            pass
        return TICK_INTERVAL_SECONDS

    def _poll_interval(self) -> float:
        try:
            import runtime_config

            if runtime_config.is_broker_mode():
                return float(POLL_INTERVAL_LIVE_SECONDS)
        except Exception:
            pass
        return float(POLL_INTERVAL_SECONDS)

    async def _tick_loop(self) -> None:
        """SL/TP monitor + stacked trade entries on tick cadence."""
        logger.info(
            "Tick monitor started (paper=%ss live=%ss)",
            TICK_INTERVAL_SECONDS,
            TICK_INTERVAL_LIVE_SECONDS,
        )
        self._sync_counter = 0
        while not self._stop_event.is_set():
            interval = self._tick_interval()
            try:
                quote = await asyncio.to_thread(get_tick_quote)
                if quote.get("price") is not None:
                    await self._update_open_trades_from_quote(quote)
                    await self._maybe_execute_stacked_trade(quote)
                    # Sync XM positions every ~5s so dashboard open board stays true
                    import runtime_config

                    if runtime_config.is_broker_mode():
                        self._sync_counter += 1
                        if self._sync_counter % max(1, int(5 / max(interval, 0.1))) == 0:
                            await self._sync_broker_positions()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Tick loop error: %s", exc)
            await asyncio.sleep(interval)

    async def _sync_broker_positions(self) -> None:
        from app.services.broker_sync import sync_broker_positions

        async with AsyncSessionLocal() as session:
            try:
                result = await sync_broker_positions(session)
                if result.get("imported") or result.get("closed_stale"):
                    logger.info("Broker sync: %s", result)
            except Exception as exc:
                logger.debug("Broker sync error: %s", exc)
                await session.rollback()

    async def _open_exec_tfs(self) -> set[str]:
        """Exec TFs that already have an open managed trade."""
        open_tfs: set[str] = set()
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(TradeRecord).where(TradeRecord.pnl_status.in_(OPEN_STATUSES))
                )
            ).scalars().all()
            for row in rows:
                feats = row.features
                if isinstance(feats, str):
                    try:
                        feats = json.loads(feats)
                    except Exception:
                        feats = {}
                if isinstance(feats, dict):
                    tf = feats.get("exec_tf") or feats.get("EXEC_TF")
                    if tf:
                        open_tfs.add(str(tf))
        return open_tfs

    async def _refresh_mtf_cache(self) -> list:
        now = time.time()
        if self._mtf_cache and now - self._mtf_cache_at < 20.0:
            return self._mtf_cache
        frames = self._last_frames
        signals = await asyncio.to_thread(scan_mtf_signals, frames)
        self._mtf_cache = signals
        self._mtf_cache_at = now
        return signals

    async def _open_symbol_tf_keys(self) -> set[str]:
        """Open positions keyed as SYMBOL:EXEC_TF."""
        keys: set[str] = set()
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(TradeRecord).where(TradeRecord.pnl_status.in_(OPEN_STATUSES))
                )
            ).scalars().all()
            for row in rows:
                feats = row.features
                if isinstance(feats, str):
                    try:
                        feats = json.loads(feats)
                    except Exception:
                        feats = {}
                if not isinstance(feats, dict):
                    feats = {}
                sym = str(
                    feats.get("symbol")
                    or feats.get("SYMBOL")
                    or getattr(row, "symbol", None)
                    or "GOLD"
                ).upper()
                tf = str(feats.get("exec_tf") or feats.get("EXEC_TF") or "5m")
                keys.add(f"{sym}:{tf}")
        return keys

    async def _refresh_universe_cache(self) -> list:
        now = time.time()
        # Universe scan is heavier — cache ~45s
        if getattr(self, "_universe_cache", None) and now - getattr(self, "_universe_cache_at", 0) < 45.0:
            return self._universe_cache
        from strategy.instrument_scanner import scan_universe

        signals = await asyncio.to_thread(scan_universe)
        self._universe_cache = signals
        self._universe_cache_at = now
        return signals

    async def _maybe_execute_stacked_trade(self, quote: dict) -> None:
        """Multi-instrument + MTF: take every valid setup across the universe."""
        if not AUTO_TRADE_ENABLED or is_trading_halted():
            return

        from analysis.correlation_groups import can_open_symbol
        from analysis.market_hours import is_gold_market_open
        from config import (
            MAX_CONCURRENT_INSTRUMENT_TRADES,
            MTF_SCAN_ENABLED,
            MULTI_INSTRUMENT_ENABLED,
            TF_HOP_ENABLED,
            TF_HOP_MAX_LEGS,
        )
        from data.market_data import get_tick_quote
        from strategy.mtf_scanner import TF_SL_ATR
        from strategy.monte_carlo import choose_minute_rr

        # Gold-only hours only when universe is metals-only
        if not MULTI_INSTRUMENT_ENABLED and not is_gold_market_open():
            return

        sig = await market_cache.get_signal()
        if not sig:
            return

        bucket_key = self._bucket_key(sig)
        if not bucket_key:
            return

        max_per_bucket = get_max_trades_per_bucket()
        if max_per_bucket <= 0:
            return

        self._reset_minute_bucket(bucket_key)
        # Raise bucket ceiling for multi-instrument concurrent fills
        bucket_cap = max(
            max_per_bucket,
            MAX_CONCURRENT_INSTRUMENT_TRADES if MULTI_INSTRUMENT_ENABLED else max_per_bucket,
        )
        if self._trades_this_minute >= bucket_cap:
            return

        trade_interval = get_trade_interval_seconds()
        if (
            not MTF_SCAN_ENABLED
            and not MULTI_INSTRUMENT_ENABLED
            and self._trades_this_minute > 0
            and time.time() - self._last_trade_at < trade_interval
        ):
            return

        open_keys = await self._open_symbol_tf_keys()
        if len(open_keys) >= MAX_CONCURRENT_INSTRUMENT_TRADES:
            return

        # Symbols already open (for correlation)
        open_syms = sorted({k.split(":", 1)[0] for k in open_keys})
        open_count_by_sym: dict[str, int] = {}
        for k in open_keys:
            s = k.split(":", 1)[0]
            open_count_by_sym[s] = open_count_by_sym.get(s, 0) + 1

        candidates: list = []
        if MULTI_INSTRUMENT_ENABLED:
            universe_hits = await self._refresh_universe_cache()
            for cand in universe_hits:
                sym = str(cand.get("SYMBOL") or cand.get("INSTRUMENT") or "").upper()
                tf = str(cand.get("EXEC_TF") or "5m")
                if not sym or f"{sym}:{tf}" in open_keys:
                    continue
                # One clean book per symbol — cuts stacked noise
                if open_count_by_sym.get(sym, 0) >= max(1, int(MAX_OPEN_PER_SYMBOL or 1)):
                    continue
                ok_corr, corr_reason = can_open_symbol(sym, open_syms)
                if not ok_corr and sym not in open_syms:
                    logger.info("Correlation skipped %s: %s", sym, corr_reason)
                    continue
                if sym in ("ETHUSD", "ETH") and not self._eth_cooldown_ok():
                    continue
                if not self._symbol_cooldown_ok(sym):
                    continue
                direction = cand.get("SIGNAL")
                if direction not in ("BUY", "SELL"):
                    continue
                try:
                    q = await asyncio.to_thread(get_tick_quote, sym)
                    px = float(q.get("price") or cand.get("ENTRY") or 0)
                except Exception:
                    px = float(cand.get("ENTRY") or 0)
                if px <= 0:
                    continue
                cand = dict(cand)
                atr = float(cand.get("ATR") or 1.0)
                from strategy.price_format import round_price
                from config import STRUCTURE_TP_MODE, is_forced_2r_brain

                cand["ENTRY"] = round_price(px, sym)
                # Preserve forced-brain / firm (crypto+FX) SL from the scanner
                # (bb firm formulas size SL to 2–2.5×ATR + min risk).
                brain = str(
                    cand.get("BRAIN")
                    or cand.get("ENTRY_BRAIN")
                    or cand.get("TP_SOURCE")
                    or ""
                )
                keep_sl = (
                    bool(cand.get("CRYPTO_FIRM"))
                    or bool(cand.get("FX_FIRM"))
                    or bool(cand.get("USDJPY_OPP_GOLD"))
                    or bool(cand.get("FIRM_FORMULA"))
                    or is_forced_2r_brain(brain)
                )
                prior_sl = cand.get("SL")
                if keep_sl and prior_sl is not None:
                    try:
                        cand["SL"] = round_price(float(prior_sl), sym)
                    except Exception:
                        keep_sl = False
                if not keep_sl:
                    sl_mult = float(TF_SL_ATR.get(tf, 1.1))
                    risk = atr * sl_mult
                    cand["SL"] = round_price(
                        px - risk if direction == "BUY" else px + risk, sym
                    )
                from analysis.structure_targets import resolve_structure_tp

                stp = resolve_structure_tp(
                    direction, px, float(cand["SL"]), atr, symbol=sym
                )
                best = choose_minute_rr(
                    direction,
                    px,
                    float(cand["SL"]),
                    cand.get("STATE") or "CHOP",
                    atr,
                    symbol=sym,
                    structure_tp=stp,
                    structure_mode=STRUCTURE_TP_MODE,
                )
                if best is None:
                    continue
                cand["TP"] = best["tp"]
                cand["TP1"] = best.get("tp1")
                cand["TP2"] = best.get("tp2")
                cand["TP3"] = best.get("tp3")
                rr_v = best.get("rr")
                cand["RR"] = (
                    f"1:{int(rr_v)}"
                    if rr_v is not None and float(rr_v) == int(float(rr_v))
                    else f"1:{float(rr_v):.2f}"
                )
                cand["TP_SOURCE"] = best.get("tp_source")
                cand["SUCCESS_PERCENT"] = best["mc_prob"]
                cand["LIVE_PRICE"] = cand["ENTRY"]
                cand["SYMBOL"] = sym
                candidates.append(cand)
        elif MTF_SCAN_ENABLED:
            mtf_sigs = await self._refresh_mtf_cache()
            prefer = ("4h", "1h", "15m", "5m", "3m", "1m")
            rank = {tf: i for i, tf in enumerate(prefer)}
            mtf_sigs = sorted(
                mtf_sigs, key=lambda s: rank.get(str(s.get("EXEC_TF") or ""), 99)
            )
            px = float(quote["price"])
            gold_open = {k.split(":", 1)[-1] for k in open_keys if k.startswith("GOLD:")}
            for cand in mtf_sigs:
                tf = str(cand.get("EXEC_TF") or "")
                if not tf or tf in gold_open:
                    continue
                direction = cand.get("SIGNAL")
                if direction not in ("BUY", "SELL"):
                    continue
                cand = dict(cand)
                atr = float(cand.get("ATR") or 1.0)
                sl_mult = float(TF_SL_ATR.get(tf, 1.1))
                risk = atr * sl_mult
                cand["ENTRY"] = px
                cand["SL"] = round(px - risk if direction == "BUY" else px + risk, 2)
                from analysis.structure_targets import resolve_structure_tp
                from config import STRUCTURE_TP_MODE

                stp = resolve_structure_tp(
                    direction, px, float(cand["SL"]), atr, symbol="GOLD"
                )
                best = choose_minute_rr(
                    direction,
                    px,
                    float(cand["SL"]),
                    cand.get("STATE") or "CHOP",
                    atr,
                    symbol="GOLD",
                    structure_tp=stp,
                    structure_mode=STRUCTURE_TP_MODE,
                )
                if best is None:
                    continue
                cand["TP"] = best["tp"]
                cand["TP1"] = best.get("tp1")
                cand["TP2"] = best.get("tp2")
                cand["TP3"] = best.get("tp3")
                rr_v = best.get("rr")
                cand["RR"] = (
                    f"1:{int(rr_v)}"
                    if rr_v is not None and float(rr_v) == int(float(rr_v))
                    else f"1:{float(rr_v):.2f}"
                )
                cand["TP_SOURCE"] = best.get("tp_source")
                cand["SUCCESS_PERCENT"] = best["mc_prob"]
                cand["LIVE_PRICE"] = px
                cand["SYMBOL"] = "GOLD"
                candidates.append(cand)
        elif TF_HOP_ENABLED and sig.get("SIGNAL") in ("BUY", "SELL"):
            from strategy.tf_hop import build_tf_hop_legs

            candidates = build_tf_hop_legs(
                sig,
                float(quote["price"]),
                max_legs=min(TF_HOP_MAX_LEGS, max_per_bucket),
                already_opened=self._trades_this_minute,
            )
            for c in candidates:
                c["SYMBOL"] = c.get("SYMBOL") or "GOLD"
        elif sig.get("SIGNAL") in ("BUY", "SELL"):
            if await self._has_opposite_open(sig["SIGNAL"]):
                return
            stacked = build_stacked_trade(sig, float(quote["price"]))
            if stacked:
                stacked["SYMBOL"] = "GOLD"
                candidates = [stacked]

        if not candidates:
            return

        bias_data = await bias_cache.get() if BIAS_GATE_ENABLED else None

        for trade_sig in candidates:
            if self._trades_this_minute >= bucket_cap:
                break
            if len(open_keys) >= MAX_CONCURRENT_INSTRUMENT_TRADES:
                break

            direction = trade_sig.get("SIGNAL")
            if direction not in ("BUY", "SELL"):
                continue
            sym = str(trade_sig.get("SYMBOL") or trade_sig.get("INSTRUMENT") or "GOLD").upper()
            tf = str(trade_sig.get("EXEC_TF") or "5m")
            key = f"{sym}:{tf}"
            if key in open_keys:
                continue

            ok_corr, corr_reason = can_open_symbol(sym, open_syms)
            if not ok_corr and sym not in open_syms:
                logger.info("Correlation skipped %s: %s", sym, corr_reason)
                continue

            # ETH spam brake (C)
            if sym in ("ETHUSD", "ETH") and not self._eth_cooldown_ok():
                logger.info("ETH cooldown — skipping new entry")
                continue

            # Same-symbol anti-stack (NAS/UKOIL punched 3× in seconds)
            if not self._symbol_cooldown_ok(sym):
                logger.info("Symbol cooldown — skipping stacked %s entry", sym)
                continue
            if any(s == sym for s in open_syms):
                logger.info("Symbol already open — skipping %s %s", sym, tf)
                continue

            if BIAS_GATE_ENABLED:
                allowed, bias_reason = bias_allows_trade(
                    direction, bias_data, MIN_BIAS_CONFIDENCE, symbol=sym
                )
                if not allowed:
                    logger.info("AI bias gate blocked %s %s: %s", sym, tf, bias_reason)
                    continue
            else:
                bias_reason = "bias gate off"

            ml_features = extract_features_from_signal(trade_sig, trade_sig)
            ml_features["zone"] = (
                trade_sig.get("DEMAND_TF") if direction == "BUY" else trade_sig.get("SUPPLY_TF")
            )
            ml_features["exec_tf"] = tf
            ml_features["EXEC_TF"] = tf
            ml_features["symbol"] = sym
            ml_features["SYMBOL"] = sym
            ml_features["mtf_independent"] = bool(trade_sig.get("MTF_INDEPENDENT"))
            if ML_GATE_ENABLED or ADAPTIVE_LEARNING:
                allowed, ml_prob, ml_reason, ml_features = await trade_calibrator.evaluate(
                    trade_sig, trade_sig
                )
                if not allowed:
                    logger.info(
                        "Learning gate blocked %s %s: %s",
                        sym,
                        tf,
                        ml_reason,
                    )
                    continue
                ml_features["bias_gate"] = bias_reason if BIAS_GATE_ENABLED else "off"
                ml_features["exec_tf"] = tf
                ml_features["symbol"] = sym
                ml_features["mtf_independent"] = bool(trade_sig.get("MTF_INDEPENDENT"))

            # Coinbase book gate — BTC/ETH only (L2 view of L3 engine)
            if CRYPTO_L3_GATE_ENABLED and sym in ("BTCUSD", "BTC", "ETHUSD", "ETH"):
                from data.l3_features import (
                    evaluate_crypto_l3_gate,
                    features_for_trade_log,
                )

                l3_ok, l3_reason, l3_feats = await asyncio.to_thread(
                    evaluate_crypto_l3_gate,
                    sym,
                    direction,
                    entry=trade_sig.get("ENTRY"),
                    sl=trade_sig.get("SL"),
                    max_spread_bps=CRYPTO_L3_MAX_SPREAD_BPS,
                    min_top_notional_usd=CRYPTO_L3_MIN_TOP_NOTIONAL_USD,
                    min_risk_spread_mult=CRYPTO_L3_MIN_RISK_SPREAD_MULT,
                    max_adverse_imbalance=CRYPTO_L3_MAX_ADVERSE_IMBALANCE,
                    depth_levels=CRYPTO_L3_DEPTH_LEVELS,
                    fail_open=CRYPTO_L3_FAIL_OPEN,
                )
                ml_features.update(features_for_trade_log(l3_feats, l3_reason))
                if not l3_ok:
                    logger.info("Crypto L3 gate blocked %s %s: %s", sym, tf, l3_reason)
                    continue

            result = await asyncio.to_thread(execute_signal_trade, trade_sig)
            if result.get("status") != "executed":
                logger.info(
                    "Execute skipped %s %s: %s",
                    sym,
                    tf,
                    result.get("message"),
                )
                continue

            self._trades_this_minute += 1
            self._last_trade_at = time.time()
            self._symbol_last_entry_at[sym] = self._last_trade_at
            if sym in ("ETHUSD", "ETH"):
                self._eth_last_entry_at = self._last_trade_at
            seq = self._trades_this_minute
            log_time = f"{bucket_key}#{seq:02d}"
            open_keys.add(key)
            if sym not in open_syms:
                open_syms.append(sym)

            await self._persist_trade_log(
                result, trade_sig, log_time=log_time, features=ml_features
            )
            # Immediately reconcile so open board matches XM
            try:
                await self._sync_broker_positions()
            except Exception:
                pass
            logger.info(
                "Universe trade %s/%s: %s %s %s @ %s SL=%s TP=%s XM=%s",
                seq,
                bucket_cap,
                sym,
                tf,
                trade_sig["SIGNAL"],
                trade_sig["ENTRY"],
                trade_sig.get("SL"),
                trade_sig.get("TP"),
                result.get("order_id"),
            )
            await ws_manager.broadcast(
                {"type": "trade_executed", "signal": trade_sig, "trade": result}
            )

    async def _run_loop(self) -> None:
        await market_cache.set_worker_running(True)
        logger.info(
            "Signal worker started (paper poll=%ss live poll=%ss)",
            POLL_INTERVAL_SECONDS,
            POLL_INTERVAL_LIVE_SECONDS,
        )
        while not self._stop_event.is_set():
            try:
                cycle = await asyncio.to_thread(run_signal_cycle)
                self._last_frames = cycle.pop("frames", None)
                is_new = await market_cache.update(cycle)
                if is_new:
                    await self._persist_snapshot(cycle["signal"])
                    self._mtf_cache_at = 0.0  # refresh MTF scan on new candle
                    self._universe_cache_at = 0.0

                # Continuous learning: every TF candle + prior-session FRVP/POC + correlations
                try:
                    from analysis.minute_learn import run_minute_learn

                    candle_key = str(
                        (cycle.get("signal") or {}).get("LOG_TIME_1M")
                        or (cycle.get("signal") or {}).get("CANDLE_TIME")
                        or ""
                    )
                    await asyncio.to_thread(
                        run_minute_learn,
                        primary_frames=self._last_frames,
                        primary_symbol=str(
                            (cycle.get("signal") or {}).get("SYMBOL") or "GOLD"
                        ),
                        primary_signal=cycle.get("signal"),
                        candle_key=candle_key or None,
                    )
                except Exception as exc:
                    logger.debug("minute_learn: %s", exc)

                # Live-brain retrospection on each new 1m: paper-sim GOLD fib /
                # book EMA at exec-TF close → soft knobs only (close RL stays truth)
                if is_new:
                    try:
                        from analysis.candle_edge import run_candle_edge

                        ck = str(
                            (cycle.get("signal") or {}).get("LOG_TIME_1M")
                            or (cycle.get("signal") or {}).get("CANDLE_TIME")
                            or ""
                        )
                        await asyncio.to_thread(run_candle_edge, candle_key=ck or None)
                    except Exception as exc:
                        logger.debug("candle_edge: %s", exc)

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
                await asyncio.sleep(self._poll_interval())
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

            entry = result.get("entry", sig.get("ENTRY"))
            live_px = sig.get("LOG_PRICE_1M") or sig.get("LIVE_PRICE") or result.get("entry")

            rr_str = sig.get("RR")
            import runtime_config
            mode = runtime_config.get_trade_mode()
            broker_id = result.get("order_id") or result.get("oanda_transaction_id")
            if not broker_id:
                br = result.get("broker_response") or {}
                broker_id = br.get("positionId") or br.get("orderId") or br.get("id")

            # Demo/live: never persist Zenith-only ghosts without an XM position id
            if mode in ("demo", "live") and status == "executed" and not broker_id:
                logger.error(
                    "Refusing to persist %s trade without XM position id: %s",
                    mode,
                    message,
                )
                return

            sym = (
                (features or {}).get("symbol")
                or (features or {}).get("SYMBOL")
                or result.get("symbol")
                or sig.get("SYMBOL")
                or sig.get("INSTRUMENT")
                or "GOLD"
            )
            sym = str(sym).upper().strip()

            from strategy.dynamic_lots import sizing_snapshot

            snap = result.get("sizing") or sizing_snapshot(sym)
            lot = float(result.get("lot_size") or snap.get("lot_size") or TRADE_LOT_SIZE)
            if features is None:
                features = {}
            features = dict(features)
            features["symbol"] = sym
            features["SYMBOL"] = sym
            features["lot_size"] = lot
            features["entry_lot"] = lot
            features["capital"] = snap.get("capital")
            features["asset_class"] = snap.get("asset_class")
            features["trail_lock"] = snap.get("trail_lock")
            features["tp1_frac"] = snap.get("tp1_frac")
            features["sizing_aggression"] = snap.get("aggression")
            try:
                from analysis.instrument_rl import get_active_arm, overlays_for

                features["rl_arm"] = get_active_arm(sym)
                ov = overlays_for(sym)
                features["rl_mc_delta"] = ov.get("mc_floor_delta")
                features["rl_aggression"] = ov.get("aggression")
                features["rl_risk_scale"] = ov.get("risk_scale")
                features["opportunity_enter"] = True
                features["source"] = features.get("source") or "opportunity_auto"
                features["exec_tf"] = sig.get("EXEC_TF") or features.get("exec_tf")
                if sig.get("MANAGE_TF"):
                    features["manage_tf"] = sig.get("MANAGE_TF")
                if sig.get("ENTRY_TF"):
                    features["entry_tf"] = sig.get("ENTRY_TF")
                if sig.get("EXIT_PATH"):
                    features["exit_path"] = sig.get("EXIT_PATH")
            except Exception:
                pass

            if status == "executed":
                try:
                    from analysis.opportunity_learn import record_opportunity

                    record_opportunity(
                        symbol=sym,
                        exec_tf=str(sig.get("EXEC_TF") or features.get("exec_tf") or ""),
                        signal=str(signal),
                        action="entered",
                        mc=float(sig.get("SUCCESS_PERCENT") or 0) or None,
                        detail="auto worker",
                    )
                except Exception:
                    pass

            margin = estimate_margin(float(entry), lot, symbol=sym) if entry is not None else None

            session.add(
                TradeRecord(
                    candle_time=sig.get("CANDLE_TIME", ""),
                    log_time=log_time,
                    signal=signal,
                    symbol=sym,
                    entry=entry,
                    sl=result.get("sl", sig.get("SL")),
                    tp=result.get("tp", sig.get("TP")),
                    tp1=result.get("tp1", sig.get("TP1")),
                    tp2=result.get("tp2", sig.get("TP2")),
                    tp3=result.get("tp3", sig.get("TP3")),
                    rr_type=rr_str,
                    current_price=live_px,
                    pnl=self._calculate_pnl(signal, entry, live_px, lot, symbol=sym),
                    partial_pnl=0.0,
                    lot_size=lot,
                    margin_usd=margin,
                    sl_trail_status=sl_trail_label(
                        signal, float(entry), float(result.get("sl") or sig.get("SL")), "open"
                    ) if entry and (result.get("sl") or sig.get("SL")) else "Fixed SL",
                    features=json.dumps(features) if features else None,
                    pnl_status="open" if status == "executed" else None,
                    status=status,
                    message=message,
                    oanda_transaction_id=str(broker_id) if broker_id else None,
                    trade_mode=result.get("mode") or mode,
                )
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()

    async def _update_open_trades_from_quote(self, quote: dict) -> None:
        """Manage open trades using each row's own XM symbol tick (not GOLD-only)."""
        from data.market_data import get_tick_quote as get_sym_quote

        import runtime_config
        live_mode = runtime_config.is_broker_mode()

        # If anything looks open locally, refresh against XM first (catches flat account)
        if live_mode:
            async with AsyncSessionLocal() as session:
                has_open = (
                    await session.execute(
                        select(TradeRecord.id)
                        .where(TradeRecord.pnl_status.in_(OPEN_STATUSES))
                        .limit(1)
                    )
                ).scalar_one_or_none()
            if has_open is not None:
                await self._sync_broker_positions()

        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(TradeRecord).where(TradeRecord.pnl_status.in_(OPEN_STATUSES))
                )
            ).scalars().all()
            if not rows:
                return

            default_sym = str((quote or {}).get("symbol") or "GOLD").upper()
            quotes: dict[str, dict] = {}
            if quote and quote.get("price") is not None:
                quotes[default_sym] = quote

            changed = False
            closed_any = False
            for row in rows:
                feats = row.features
                if isinstance(feats, str):
                    try:
                        feats = json.loads(feats)
                    except Exception:
                        feats = {}
                if not isinstance(feats, dict):
                    feats = {}
                sym = str(
                    row.symbol
                    or feats.get("symbol")
                    or feats.get("SYMBOL")
                    or default_sym
                ).upper()

                if sym not in quotes:
                    try:
                        quotes[sym] = await asyncio.to_thread(get_sym_quote, sym)
                    except Exception:
                        quotes[sym] = quote or {}

                q = quotes.get(sym) or {}
                close_price = q.get("price")
                if close_price is None:
                    continue
                high = q.get("high", close_price)
                low = q.get("low", close_price)

                old_pnl = row.pnl
                trade = row_to_trade_dict(row)
                before = dict(trade)

                closed = update_trade_on_bar(
                    trade,
                    float(high),
                    float(low),
                    float(close_price),
                    row.lot_size,
                )

                if live_mode and is_live_managed_row(row):
                    ok = await asyncio.to_thread(
                        apply_live_broker_actions,
                        str(row.oanda_transaction_id),
                        before,
                        trade,
                    )
                    if not ok:
                        continue
                    # If local logic closed but XM still holds the position, keep open
                    if closed and trade.get("pnl_status", "").startswith("closed"):
                        still_open = await asyncio.to_thread(
                            _broker_position_exists, str(row.oanda_transaction_id)
                        )
                        if still_open:
                            logger.warning(
                                "Local close blocked — XM pos %s still open",
                                row.oanda_transaction_id,
                            )
                            trade["pnl_status"] = before.get("pnl_status") or "open"
                            trade["status"] = before.get("status") or "executed"
                            trade["message"] = before.get("message") or trade.get("message")
                            trade["sl"] = before.get("sl")
                            trade["partial_pnl"] = before.get("partial_pnl")
                            trade["pnl"] = before.get("pnl")
                            closed = False
                    # Floating must match XM — local leg_pnl uses wrong CFD multipliers
                    if not closed:
                        xm_float = await asyncio.to_thread(
                            _xm_open_profit, str(row.oanda_transaction_id)
                        )
                        if xm_float is not None:
                            trade["pnl"] = xm_float
                    # Local bar close confirmed XM-flat → stamp deal cash (not leg_pnl)
                    elif closed and row.oanda_transaction_id:
                        xm_deal = await asyncio.to_thread(
                            _xm_closed_deal_pnl, str(row.oanda_transaction_id)
                        )
                        if xm_deal and xm_deal.get("pnl") is not None:
                            trade["pnl"] = float(xm_deal["pnl"])
                            if xm_deal.get("exit_price") is not None:
                                trade["exit_price"] = float(xm_deal["exit_price"])
                            trade["message"] = (
                                f"Closed on XM (pos {row.oanda_transaction_id}) "
                                f"· XM P&L {float(xm_deal['pnl']):+.2f}"
                            )

                apply_trade_dict_to_row(row, trade, closed)
                row.current_price = trade.get("exit_price") or float(close_price)
                if not row.symbol:
                    row.symbol = sym
                if (
                    closed
                    and live_mode
                    and row.oanda_transaction_id
                    and trade.get("pnl") is not None
                ):
                    # Persist pnl_source so stats never "repair" with fantasy math
                    try:
                        feats = dict(feats) if isinstance(feats, dict) else {}
                        feats["xm_deal_pnl"] = float(trade["pnl"])
                        feats["pnl_source"] = "xm_deal"
                        row.features = json.dumps(feats)
                    except Exception:
                        pass

                if closed or trade != before or row.pnl != old_pnl:
                    changed = True
                if closed:
                    closed_any = True

            if changed:
                await session.commit()
                if closed_any:
                    await ws_manager.broadcast({"type": "trade_closed"})
                    await on_trade_closed()

    @staticmethod
    def _calculate_pnl(
        signal: Optional[str],
        entry: Optional[float],
        price: Optional[float],
        lot_size: float | None = None,
        symbol: Optional[str] = None,
    ):
        if signal not in ("BUY", "SELL") or entry is None or price is None:
            return 0.0
        return leg_pnl(signal, entry, price, lot_size or TRADE_LOT_SIZE, symbol=symbol)


signal_worker = SignalWorker()


def _broker_position_exists(position_id: str) -> bool:
    """True when XM still holds this position id."""
    if not position_id:
        return False
    try:
        from data.broker import broker_client

        if hasattr(broker_client, "position_exists"):
            return bool(broker_client.position_exists(position_id))
        for p in broker_client.get_open_positions() or []:
            pid = str(p.get("id") or p.get("positionId") or "")
            if pid == str(position_id):
                return True
    except Exception:
        return True  # fail-safe: don't mark closed if unsure
    return False


def _xm_closed_deal_pnl(position_id: str) -> dict | None:
    """XM realised deal cash for a closed position id."""
    if not position_id:
        return None
    try:
        from data.broker import broker_client

        if hasattr(broker_client, "position_closed_pnl"):
            return broker_client.position_closed_pnl(str(position_id), days=90)
    except Exception:
        return None
    return None


def _xm_open_profit(position_id: str) -> float | None:
    """XM floating profit for an open position id, or None if unavailable."""
    if not position_id:
        return None
    try:
        from data.broker import broker_client

        for p in broker_client.get_open_positions() or []:
            pid = str(p.get("id") or p.get("positionId") or "")
            if pid == str(position_id) and p.get("profit") is not None:
                return round(float(p["profit"]), 2)
    except Exception:
        return None
    return None
