"""
Bar-by-bar backtest engine.

Fetches historical OHLCV from TradingView, resamples to all required
timeframes, then runs _compute_signal() at every 5-minute bar in the
selected date range — exactly as the live worker does.

Session time is injected via a ContextVar so each bar is evaluated in its
correct historical trading session (London / NY / etc.).
"""
from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

BARS_PER_DAY_5M = 290      # conservative estimate (market ~24h XAUUSD)
MAX_FETCH_BARS = 12_000    # hard cap for tvDatafeed request
# prepare_df rolling(50) on 4h → first valid row after ~50×4h; fetch extra calendar days
_PREPARE_ROLLING = 50
_MIN_4H_PREPARED = 8


def warmup_days_for_backtest() -> int:
    """Calendar days of OHLCV to fetch *before* the selected range.

    Enough for 4h rolling(50) + volume-profile lookback (~288×5m ≈ 1 day)
    so the *first bar of the selected range* can already trade.
    """
    hours_needed = (_PREPARE_ROLLING + _MIN_4H_PREPARED) * 4
    # +7: weekends/gaps; profile lookback is tiny vs 4h warmup but keep cushion
    return max(18, int(hours_needed / 24) + 7)


def signal_slices_ready(
    p5m: pd.DataFrame,
    p15m: pd.DataFrame,
    p1h: pd.DataFrame,
    p4h: pd.DataFrame,
) -> bool:
    """True when every TF slice has enough prepared data for _compute_signal."""
    if len(p5m) < 2:
        return False
    for frame in (p5m, p15m, p1h, p4h):
        if frame.empty:
            return False
        last = frame.iloc[-1]
        for col in ("ATR", "EMA9", "EMA21", "RSI"):
            v = last.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return False
    return True


# ── helpers ──────────────────────────────────────────────────────────────────

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna()


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _margin(entry: float, lot: float, leverage: int = 1000) -> float:
    return round(entry * lot * 100 / leverage, 2)


# ── main engine ──────────────────────────────────────────────────────────────

def run_signal_backtest(
    start_date: str,
    end_date: str,
    lot_size: float = 0.1,
    mc_floor: float = 42.0,
    rr_mode: str = "auto",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    capital: float = 1000.0,
    symbol: str = "GOLD",
    dynamic_lots: bool = True,
) -> Dict[str, Any]:
    """
    Run the live signal algorithm on historical 5m OHLCV data.

    Uses SYMBOL_ENTRY_BRAIN / SYMBOL_SCAN_TFS for the selected symbol
    (same map as the live desk).

    Args:
        start_date: ISO date string (e.g. "2026-06-01")
        end_date:   ISO date string (e.g. "2026-06-30")
        lot_size:   Lot size for simulated trades
        mc_floor:   Minimum MC probability to take a trade (%)
        progress_cb: Optional callable(done, total, message) for progress updates
        capital:    Starting capital for equity protect / bank target
        symbol:     Desk symbol (GOLD, USDJPY, …)
        dynamic_lots: Use live lot_for_profit_target (+ SYMBOL_ALLOCATION)

    Returns:
        Dict with keys: trades, stats, data_range, bars_scanned, error
    """
    # Lazy imports — engine only loaded when a simulation is requested
    from analysis.session import set_backtest_bar_time
    from analysis.adaptive_guard import rebuild as ag_rebuild, check as ag_check
    from config import (
        ENTRY_COOLDOWN_BARS,
        LOSS_PNL_PROTECT,
        MC_FLOOR_BASE,
        symbol_entry_brain,
        symbol_rr,
    )
    from data.historical import fetch_symbol_ohlcv, resample_ohlcv
    from indicators.prepare import prepare_df
    from strategy.instrument_scanner import resolve_entry_scan_tfs
    from strategy.signal import _compute_signal

    def _progress(done: int, total: int, msg: str = "") -> None:
        if progress_cb:
            try:
                progress_cb(done, total, msg)
            except Exception:
                pass

    backtest_symbol = str(symbol or "GOLD").upper().strip()
    if backtest_symbol in ("XAUUSD", "XAU"):
        backtest_symbol = "GOLD"
    entry_brain = symbol_entry_brain(backtest_symbol)
    scan_tfs = resolve_entry_scan_tfs(backtest_symbol)
    pair_rr = symbol_rr(backtest_symbol, default=4.0)

    # ── 1. Date range ─────────────────────────────────────────────────────────
    try:
        start_dt = _parse_utc(start_date)
        end_dt = _parse_utc(end_date).replace(hour=23, minute=59, second=59)
    except (ValueError, TypeError) as exc:
        return {"error": f"Invalid date: {exc}", "trades": [], "stats": {}, "bars_scanned": 0}

    days = max((end_dt - start_dt).days + 1, 1)
    logger.info(
        "Backtest: %s · %s @ %s · %d days (%s → %s)",
        backtest_symbol,
        entry_brain,
        "/".join(scan_tfs),
        days,
        start_date,
        end_date,
    )
    _progress(
        0,
        1,
        f"Fetching {backtest_symbol} · {entry_brain} @ {'/'.join(scan_tfs)}…",
    )

    # ── 2. Fetch base data (local Dukascopy hist preferred, then TV/YF) ───────
    # Fetch extra history before range start so 4h indicators are warm at bar 0
    warmup_days = warmup_days_for_backtest()
    warmup_start = (start_dt - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    logger.info("Backtest warmup: %d calendar days before %s", warmup_days, start_date)
    fetch_end    = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        df_5m_raw = fetch_symbol_ohlcv(
            backtest_symbol, warmup_start, fetch_end, interval="5m"
        )
    except Exception as exc:
        return {"error": f"Data fetch failed: {exc}", "trades": [], "stats": {}, "bars_scanned": 0}

    if df_5m_raw is None or df_5m_raw.empty:
        return {"error": "No historical data returned", "trades": [], "stats": {}, "bars_scanned": 0}

    available_from = str(df_5m_raw.index[0])[:10]
    available_to   = str(df_5m_raw.index[-1])[:10]

    # Check overlap
    in_range_mask = (df_5m_raw.index >= start_dt) & (df_5m_raw.index <= end_dt)
    in_range_idx  = df_5m_raw.index[in_range_mask]

    if len(in_range_idx) == 0:
        return {
            "error": (
                f"No data available for {start_date} → {end_date}. "
                f"yfinance returned data for {available_from} → {available_to}. "
                f"For 5m data, yfinance only goes back ~60 days. "
                f"For older dates use 1h interval (up to 2 years)."
            ),
            "trades": [], "stats": {},
            "bars_scanned": 0,
            "data_range": {"from": available_from, "to": available_to},
        }

    # ── 3. Resample & prepare indicators ONCE ────────────────────────────────
    _progress(0, 100, "Preparing indicators on historical data…")
    try:
        # Prefer shared historical resampler when available (same as candles job)
        try:
            df_15m_raw = resample_ohlcv(df_5m_raw, "15min")
            df_1h_raw = resample_ohlcv(df_5m_raw, "1h")
            df_4h_raw = resample_ohlcv(df_5m_raw, "4h")
            df_3m_raw = resample_ohlcv(df_5m_raw, "3min")
        except Exception:
            df_15m_raw = _resample_ohlcv(df_5m_raw, "15min")
            df_1h_raw = _resample_ohlcv(df_5m_raw, "1h")
            df_4h_raw = _resample_ohlcv(df_5m_raw, "4h")
            df_3m_raw = _resample_ohlcv(df_5m_raw, "3min")

        full_5m  = prepare_df(df_5m_raw.copy())
        full_15m = prepare_df(df_15m_raw.copy())
        full_1h  = prepare_df(df_1h_raw.copy())
        full_4h  = prepare_df(df_4h_raw.copy())
        full_3m = (
            prepare_df(df_3m_raw.copy())
            if df_3m_raw is not None and not df_3m_raw.empty
            else None
        )
    except Exception as exc:
        logger.exception("prepare_df failed: %s", exc)
        return {"error": f"Indicator preparation failed: {exc}", "trades": [], "stats": {}, "bars_scanned": 0}

    # ── 4. Bar-by-bar replay ──────────────────────────────────────────────────
    total_bars   = len(in_range_idx)
    simulated_trades: List[Dict] = []
    open_trade: Optional[Dict] = None
    cooldown_until_step = -1
    closed_sim: List[Dict] = []
    start_capital = float(capital or 1000.0)
    base_mc = float(mc_floor if mc_floor is not None else MC_FLOOR_BASE)
    from analysis.concept_engine import rebuild as concept_rebuild
    concept_rebuild([])  # walk-forward from cold start for this timeline
    bars_not_ready = 0
    bars_evaluated = 0
    first_signal_bar: Optional[int] = None
    first_signal_time: Optional[str] = None
    first_trade_bar: Optional[int] = None
    first_trade_time: Optional[str] = None

    for step, bar_time in enumerate(in_range_idx):
        if step % 50 == 0:
            _progress(step, total_bars, f"Scanning bar {step}/{total_bars}…")

        bar_pos = full_5m.index.get_loc(bar_time)

        # Slice each TF "up to and including" this bar
        p5m  = full_5m.iloc[:bar_pos + 1]
        p15m = full_15m[full_15m.index <= bar_time]
        p1h  = full_1h[full_1h.index <= bar_time]
        p4h  = full_4h[full_4h.index <= bar_time]

        if not signal_slices_ready(p5m, p15m, p1h, p4h):
            bars_not_ready += 1
            continue

        bar_day = str(bar_time)[:10]

        # ── manage open trade (live-style partials + trailing) ────────────
        if open_trade is not None:
            bar = df_5m_raw.loc[bar_time]
            h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])
            from strategy.trade_simulator import update_trade_on_bar

            if update_trade_on_bar(open_trade, h, l, c, open_trade.get("lot_size")):
                open_trade["exit_time"] = str(bar_time)[:16]
                open_trade["status"] = "closed"
                open_trade["candle_time"] = str(bar_time)[:19]
                simulated_trades.append(open_trade)
                closed_sim.append(dict(open_trade))
                ag_rebuild(
                    closed_sim,
                    loss_pnl_protect=LOSS_PNL_PROTECT,
                    mc_floor_base=base_mc,
                    start_capital=start_capital,
                    as_of_day=bar_day,
                )
                from analysis.concept_engine import rebuild as concept_rebuild
                concept_rebuild(closed_sim)
                open_trade = None
                cooldown_until_step = step + ENTRY_COOLDOWN_BARS
                continue

        if open_trade is not None:
            continue

        if step < cooldown_until_step:
            continue

        adaptive = ag_rebuild(
            closed_sim,
            loss_pnl_protect=LOSS_PNL_PROTECT,
            mc_floor_base=base_mc,
            start_capital=start_capital,
            as_of_day=bar_day,
        )
        if adaptive.get("equity_allow") is False:
            continue
        eff_mc_floor = float(adaptive.get("mc_floor") or base_mc)

        # ── inject bar time so session + XM hours use historical time ─────
        set_backtest_bar_time(bar_time.to_pydatetime())

        # ── live signal engine + pair brain / scan TF (MTF when mapped) ───
        p3m = full_3m[full_3m.index <= bar_time] if full_3m is not None else None
        frames = {
            "1m": None,
            "3m": p3m if p3m is not None and len(p3m) >= 40 else None,
            "5m": p5m,
            "15m": p15m,
            "1h": p1h,
            "4h": p4h,
        }
        try:
            from config import MTF_SCAN_ENABLED
            from strategy.mtf_scanner import scan_all_timeframes

            result = None
            if MTF_SCAN_ENABLED and scan_tfs:
                hits = scan_all_timeframes(
                    frames,
                    only_tfs=tuple(scan_tfs),
                    symbol=backtest_symbol,
                )
                result = hits[0] if hits else None
            if result is None:
                result = _compute_signal(
                    df_1m=p5m,
                    df_3m=p3m if p3m is not None else p5m,
                    df_5m=p5m,
                    df_15m=p15m,
                    df_1h=p1h,
                    df_4h=p4h,
                    symbol=backtest_symbol,
                )
            if result:
                result["SYMBOL"] = backtest_symbol
                result["INSTRUMENT"] = backtest_symbol
        except Exception:
            logger.debug("_compute_signal error at %s: %s", bar_time, traceback.format_exc()[-200:])
            continue
        finally:
            set_backtest_bar_time(None)   # always clear

        bars_evaluated += 1

        from strategy.trade_simulator import apply_rr_mode

        result = apply_rr_mode(result, rr_mode)
        if result is None:
            continue

        signal = result.get("SIGNAL")
        if signal not in ("BUY", "SELL"):
            continue

        if first_signal_bar is None:
            first_signal_bar = step
            first_signal_time = str(bar_time)[:16]

        # ── MC + adaptive gates (same spirit as live XM) ──────────────────
        mc_prob = float(result.get("SUCCESS_PERCENT") or result.get("MARKOV_MC_PROB") or 0)
        buy_s = float(result.get("BUY_SCORE") or 0)
        sell_s = float(result.get("SELL_SCORE") or 0)
        feats = {
            "STATE": result.get("STATE"),
            "mc_prob": mc_prob,
            "markov_mc_prob": float(result.get("MARKOV_MC_PROB") or mc_prob),
            "score_margin": abs(buy_s - sell_s),
            "hour_utc": int(bar_time.hour) if hasattr(bar_time, "hour") else None,
        }
        ok_adapt, _reason = ag_check(feats)
        if not ok_adapt:
            continue
        if mc_prob < eff_mc_floor:
            continue

        entry_px = result.get("ENTRY")
        sl_px    = result.get("SL")
        tp_px    = result.get("TP")
        if entry_px is None or sl_px is None:
            continue

        from strategy.trade_simulator import new_open_trade

        if first_trade_bar is None:
            first_trade_bar = step
            first_trade_time = str(bar_time)[:16]

        trade_lot = float(lot_size)
        if dynamic_lots:
            try:
                from strategy.dynamic_lots import lot_for_profit_target

                eq = start_capital + sum(float(t.get("pnl") or 0) for t in closed_sim)
                sized = lot_for_profit_target(
                    backtest_symbol,
                    entry=float(entry_px),
                    sl=float(sl_px),
                    tp=float(tp_px) if tp_px else None,
                    rr=result.get("RR") or pair_rr,
                    capital=max(50.0, eq),
                )
                trade_lot = float(sized["lot_size"])
            except Exception:
                trade_lot = float(lot_size)

        open_trade = new_open_trade(
            signal=signal,
            entry=float(entry_px),
            sl=float(sl_px),
            tp=float(tp_px) if tp_px else None,
            tp1=result.get("TP1"),
            tp2=result.get("TP2"),
            tp3=result.get("TP3"),
            rr_type=result.get("RR"),
            lot_size=trade_lot,
            mc_prob=mc_prob,
            margin_usd=_margin(float(entry_px), trade_lot),
            entry_time=str(bar_time)[:16],
            candle_time=str(bar_time)[:19],
            symbol=backtest_symbol,
            SYMBOL=backtest_symbol,
            exec_tf=result.get("EXEC_TF"),
            features={
                "STATE": result.get("STATE"),
                "mc_prob": mc_prob,
                "concepts": result.get("CONCEPTS") or {},
                "concept_quality": result.get("CONCEPT_QUALITY"),
                "smc_score": result.get("SMC_SCORE"),
                "score_margin": abs(buy_s - sell_s),
                "vp_location": result.get("VP_LOCATION"),
                "vp_aligned": result.get("VP_ALIGNED"),
                "vp_in_value": result.get("VP_IN_VALUE"),
                "symbol": backtest_symbol,
                "entry_brain": entry_brain,
            },
        )
        open_trade["candle_index"] = step

    # Close any trade still open at end of range
    if open_trade is not None:
        from strategy.trade_simulator import close_at_market

        last_bar = df_5m_raw[df_5m_raw.index <= end_dt]
        if not last_bar.empty:
            exit_px = float(last_bar.iloc[-1]["close"])
            close_at_market(open_trade, exit_px, str(last_bar.index[-1])[:16])
        simulated_trades.append(open_trade)

    _progress(total_bars, total_bars, "Done.")

    # ── 5. Stats ──────────────────────────────────────────────────────────────
    from strategy.trade_simulator import stats_from_trades

    outcome_stats = stats_from_trades(simulated_trades)

    entries = [t["entry"] for t in simulated_trades if t.get("entry")]
    margins = [t["margin_usd"] for t in simulated_trades if t.get("margin_usd")]

    stats = {
        "executed":      len(simulated_trades),
        "closed":        outcome_stats["closed"],
        "wins":          outcome_stats["wins"],
        "losses":        outcome_stats["losses"],
        "breakevens":    outcome_stats["breakevens"],
        "realized_pnl":  outcome_stats["realized_pnl"],
        "win_rate":      outcome_stats["win_rate"],
        "lot_size":      lot_size,
        "trade_units":   round(lot_size * 100),
        "first_price":   round(entries[0], 2) if entries else None,
        "last_price":    round(entries[-1], 2) if entries else None,
        "avg_margin_usd": round(sum(margins) / len(margins), 2) if margins else None,
        "avg_leverage":  1000,
        "rr_mode":       rr_mode,
        "bars_evaluated": bars_evaluated,
        "bars_not_ready": bars_not_ready,
        "first_signal_bar": first_signal_bar,
        "first_signal_time": first_signal_time,
        "first_trade_bar": first_trade_bar,
        "first_trade_time": first_trade_time,
        "first_trade_pct": (
            round(100.0 * first_trade_bar / max(total_bars - 1, 1), 1)
            if first_trade_bar is not None and total_bars > 0
            else None
        ),
        "trades_from_first_setup": True,
        "symbol": backtest_symbol,
        "entry_brain": entry_brain,
        "scan_tfs": list(scan_tfs),
        "symbol_rr": pair_rr,
    }

    logger.info(
        "Backtest complete: %s %s · %d bars, %d trades (%d W / %d L), PnL $%.2f",
        backtest_symbol,
        entry_brain,
        total_bars,
        len(simulated_trades),
        outcome_stats["wins"],
        outcome_stats["losses"],
        outcome_stats["realized_pnl"],
    )

    return {
        "trades":       simulated_trades,
        "stats":        stats,
        "data_range":   {"from": available_from, "to": available_to},
        "bars_scanned": total_bars,
        "symbol":       backtest_symbol,
        "entry_brain":  entry_brain,
        "scan_tfs":     list(scan_tfs),
        "error":        None,
    }
