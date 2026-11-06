"""Backtesting engine — replay closed trades & time-machine dashboard snapshots."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select

from app.core.security import get_current_user
from app.core.database import AsyncSessionLocal
from app.models.trade_record import TradeRecord

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# ── In-memory simulation job store ───────────────────────────────────────────
_jobs: Dict[str, dict] = {}   # job_id → {status, progress, total, message, result, error}


class BacktestParams(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_balance: float = 10000.0
    signal_filter: Optional[str] = None   # "BUY" | "SELL" | None = both
    rr_filter: Optional[str] = None       # "1:2" | "1:3" | "1:4" | None = all
    sl_cap: float = 10.0                  # max SL risk in USD
    lot_size: float = 0.1


def _parse_dt(val) -> Optional[datetime]:
    if val is None:
        return None
    try:
        s = str(val).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _trade_time(t: dict) -> Optional[datetime]:
    return _parse_dt(t.get("created_at") or t.get("log_time") or t.get("candle_time"))


def _risk_usd(t: dict) -> Optional[float]:
    entry = t.get("entry")
    sl = t.get("sl")
    lot = t.get("lot_size") or 0.1
    if entry is not None and sl is not None:
        return abs(float(entry) - float(sl)) * float(lot) * 100
    return None


@router.post("/run")
async def run_backtest(
    params: BacktestParams,
    _: dict = Depends(get_current_user),
):
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(TradeRecord))).scalars().all()
        all_trades = [r.to_dict() for r in rows]

    closed = [t for t in all_trades if t.get("status") == "closed" and t.get("signal") in ("BUY", "SELL")]

    # Date filters
    start_dt = _parse_dt(params.start_date)
    end_dt = _parse_dt(params.end_date)
    if start_dt or end_dt:
        filtered = []
        for t in closed:
            tt = _trade_time(t)
            if tt is None:
                continue
            if start_dt and tt < start_dt:
                continue
            if end_dt and tt > end_dt:
                continue
            filtered.append(t)
        closed = filtered

    # Direction filter
    if params.signal_filter in ("BUY", "SELL"):
        closed = [t for t in closed if t.get("signal") == params.signal_filter]

    # RR filter
    if params.rr_filter:
        closed = [t for t in closed if t.get("rr_type") == params.rr_filter]

    # SL cap filter
    closed = [t for t in closed if (_risk_usd(t) or 0) <= params.sl_cap or _risk_usd(t) is None]

    # Sort chronologically
    closed.sort(key=lambda t: _trade_time(t) or datetime.min.replace(tzinfo=timezone.utc))

    # ── Compute equity curve ──────────────────────────────────────────────────
    equity = params.initial_balance
    peak_equity = equity
    max_drawdown = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = losses = breakevens = 0
    best_trade = worst_trade = 0.0

    # daily equity: { "YYYY-MM-DD": final_equity }
    daily: dict[str, float] = {}
    monthly: dict[str, dict] = {}   # "YYYY-MM" → {pnl, trades, wins}
    by_signal: dict[str, dict] = {}
    by_rr: dict[str, dict] = {}
    by_hour: dict[int, dict] = {}

    result_trades = []

    for t in closed:
        pnl = float(t.get("pnl") or 0)
        equity += pnl

        if pnl > 1:
            wins += 1
            gross_profit += pnl
        elif pnl < -1:
            losses += 1
            gross_loss += pnl  # negative
        else:
            breakevens += 1

        if equity > peak_equity:
            peak_equity = equity
        dd = equity - peak_equity
        if dd < max_drawdown:
            max_drawdown = dd

        if pnl > best_trade:
            best_trade = pnl
        if pnl < worst_trade:
            worst_trade = pnl

        tt = _trade_time(t)
        day_key = tt.strftime("%Y-%m-%d") if tt else "unknown"
        mon_key = tt.strftime("%Y-%m") if tt else "unknown"
        hour_val = tt.hour if tt else None

        daily[day_key] = round(equity, 2)

        m = monthly.setdefault(mon_key, {"pnl": 0.0, "trades": 0, "wins": 0})
        m["pnl"] += pnl
        m["trades"] += 1
        if pnl > 1:
            m["wins"] += 1

        sig = t.get("signal", "OTHER")
        s = by_signal.setdefault(sig, {"trades": 0, "wins": 0, "pnl": 0.0})
        s["trades"] += 1
        s["pnl"] += pnl
        if pnl > 1:
            s["wins"] += 1

        rr = t.get("rr_type") or "unknown"
        r = by_rr.setdefault(rr, {"trades": 0, "wins": 0, "pnl": 0.0})
        r["trades"] += 1
        r["pnl"] += pnl
        if pnl > 1:
            r["wins"] += 1

        if hour_val is not None:
            h = by_hour.setdefault(hour_val, {"trades": 0, "wins": 0, "pnl": 0.0})
            h["trades"] += 1
            h["pnl"] += pnl
            if pnl > 1:
                h["wins"] += 1

        result_trades.append({
            "id": t.get("id"),
            "time": day_key + ("T" + tt.strftime("%H:%M") if tt else ""),
            "signal": t.get("signal"),
            "entry": t.get("entry"),
            "sl": t.get("sl"),
            "tp": t.get("tp"),
            "exit_price": t.get("exit_price"),
            "pnl": round(pnl, 2),
            "rr_type": t.get("rr_type"),
            "result": "win" if pnl > 1 else ("loss" if pnl < -1 else "be"),
        })

    total = len(closed)
    decisive = wins + losses
    win_rate = wins / decisive if decisive else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else (999.0 if gross_profit > 0 else 0.0)
    expectancy = (equity - params.initial_balance) / total if total else 0.0

    # Build equity curve list sorted by date
    equity_curve = [
        {"time": k, "value": v}
        for k, v in sorted(daily.items())
        if k != "unknown"
    ]

    # Monthly list sorted
    monthly_list = [
        {
            "month": k,
            "pnl": round(v["pnl"], 2),
            "trades": v["trades"],
            "win_rate": round(v["wins"] / v["trades"], 3) if v["trades"] else 0,
        }
        for k, v in sorted(monthly.items())
        if k != "unknown"
    ]

    # Best hours (≥5 trades, sorted by win rate)
    best_hours = sorted(
        [
            {"hour": h, "trades": v["trades"], "win_rate": round(v["wins"] / v["trades"], 3), "pnl": round(v["pnl"], 2)}
            for h, v in by_hour.items()
            if v["trades"] >= 5
        ],
        key=lambda x: x["win_rate"],
        reverse=True,
    )[:6]

    return {
        "status": "completed",
        "total_input_trades": len(all_trades),
        "summary": {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": losses,
            "breakeven_trades": breakevens,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(equity - params.initial_balance, 2),
            "initial_balance": params.initial_balance,
            "final_balance": round(equity, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown / params.initial_balance * 100, 2),
            "profit_factor": round(profit_factor, 3),
            "avg_win": round(gross_profit / wins, 2) if wins else 0,
            "avg_loss": round(abs(gross_loss) / losses, 2) if losses else 0,
            "best_trade": round(best_trade, 2),
            "worst_trade": round(worst_trade, 2),
            "expectancy": round(expectancy, 2),
        },
        "equity_curve": equity_curve,
        "monthly": monthly_list,
        "by_signal": {
            k: {
                "trades": v["trades"],
                "win_rate": round(v["wins"] / v["trades"], 3) if v["trades"] else 0,
                "pnl": round(v["pnl"], 2),
            }
            for k, v in by_signal.items()
        },
        "by_rr": {
            k: {
                "trades": v["trades"],
                "win_rate": round(v["wins"] / v["trades"], 3) if v["trades"] else 0,
                "pnl": round(v["pnl"], 2),
            }
            for k, v in by_rr.items()
        },
        "best_hours": best_hours,
        "recent_trades": result_trades[-50:],
    }


@router.get("/summary")
async def backtest_summary(_: dict = Depends(get_current_user)):
    """Quick overview without running a full backtest."""
    return await run_backtest(BacktestParams(), _)


# ── Time-machine replay endpoint ──────────────────────────────────────────────

@router.get("/replay")
async def replay_snapshot(
    start_date: Optional[str] = Query(None, description="Range start (inclusive)"),
    end_date: Optional[str] = Query(None, description="Range end (inclusive, end-of-day)"),
    _: dict = Depends(get_current_user),
):
    """
    Dashboard snapshot for a date range.
    - Stats (P&L, WR, wins/losses) are for trades IN the range.
    - ML model + adaptive guard are retrained on ALL trades up to end_date
      (mirrors what the algorithm actually knew at that point).
    """
    from analysis.adaptive_guard import rebuild as ag_rebuild
    from analysis.trade_learning import LogisticCalibrator, analyze_patterns
    from config import LOSS_PNL_PROTECT, MC_FLOOR_BASE

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(TradeRecord))).scalars().all()
        all_trades = [r.to_dict() for r in rows]

    def _to_dt_eod(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            return None

    def _to_dt_sod(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.replace(hour=0, minute=0, second=0)
        except ValueError:
            return None

    start_dt = _to_dt_sod(start_date)
    end_dt = _to_dt_eod(end_date)

    def in_range(t: dict) -> bool:
        tt = _parse_dt(t.get("created_at") or t.get("log_time"))
        if tt is None:
            return False
        if start_dt and tt < start_dt:
            return False
        if end_dt and tt > end_dt:
            return False
        return True

    def before_end(t: dict) -> bool:
        if end_dt is None:
            return True
        tt = _parse_dt(t.get("created_at") or t.get("log_time"))
        return tt is not None and tt <= end_dt

    all_closed = [
        t for t in all_trades
        if t.get("status") == "closed" and t.get("signal") in ("BUY", "SELL")
    ]
    all_closed.sort(key=lambda t: _parse_dt(t.get("created_at") or t.get("log_time")) or datetime.min.replace(tzinfo=timezone.utc))

    # Trades in the selected range → used for stats display
    range_trades = [t for t in all_closed if in_range(t)]

    # All trades up to end_date → used to train ML + guard (what the algo knew then)
    training_trades = [t for t in all_closed if before_end(t)]

    # ── Stats (range only) ────────────────────────────────────────────────────
    wins = sum(1 for t in range_trades if (t.get("pnl") or 0) > 1)
    losses = sum(1 for t in range_trades if (t.get("pnl") or 0) < -1)
    bes = len(range_trades) - wins - losses
    realized = round(sum(float(t.get("pnl") or 0) for t in range_trades), 2)
    decisive = wins + losses
    win_rate = round(wins / decisive, 4) if decisive else 0.0

    # Real prices and margin from the range trades
    entries = [float(t["entry"]) for t in range_trades if t.get("entry") is not None]
    margins = [float(t["margin_usd"]) for t in range_trades if t.get("margin_usd") is not None]
    lots = [float(t["lot_size"]) for t in range_trades if t.get("lot_size") is not None]
    leverages = []
    for t in range_trades:
        entry = t.get("entry")
        lot = t.get("lot_size")
        margin = t.get("margin_usd")
        if entry and lot and margin and margin > 0:
            lev = round((float(entry) * float(lot) * 100) / float(margin))
            leverages.append(lev)

    first_price = entries[0] if entries else None
    last_price = entries[-1] if entries else None
    avg_margin = round(sum(margins) / len(margins), 2) if margins else None
    avg_lot = round(sum(lots) / len(lots), 3) if lots else 0.1
    avg_leverage = round(sum(leverages) / len(leverages)) if leverages else 1000

    stats = {
        "executed": len(range_trades),
        "wins": wins,
        "losses": losses,
        "breakevens": bes,
        "realized_pnl": realized,
        "unrealized_pnl": 0.0,
        "lot_size": avg_lot,
        "trade_units": round(avg_lot * 100),
        "win_rate": win_rate,
        # price & margin from actual trades
        "first_price": round(first_price, 2) if first_price else None,
        "last_price": round(last_price, 2) if last_price else None,
        "avg_margin_usd": avg_margin,
        "avg_leverage": avg_leverage,
    }

    # ── Adaptive guard (trained on all up to end_date) ────────────────────────
    adaptive = ag_rebuild(training_trades, loss_pnl_protect=LOSS_PNL_PROTECT, mc_floor_base=MC_FLOOR_BASE)

    # ── ML calibrator (trained on all up to end_date) ─────────────────────────
    cal = LogisticCalibrator()
    cal.fit(training_trades)
    model = cal.to_dict()

    # ── Patterns (range only for relevance) ───────────────────────────────────
    patterns = analyze_patterns(range_trades)

    # ── Last signal in the range ──────────────────────────────────────────────
    last_signal = None
    for t in reversed(range_trades):
        feats = t.get("features") or {}
        if isinstance(feats, str):
            try:
                feats = json.loads(feats)
            except Exception:
                feats = {}
        if feats and feats.get("STATE"):
            last_signal = {
                "SIGNAL": t.get("signal") or "NO TRADE",
                "STATE": feats.get("STATE"),
                "SUCCESS_PERCENT": feats.get("SUCCESS_PERCENT") or feats.get("mc_prob"),
                "MARKOV_MC_PROB": feats.get("MARKOV_MC_PROB") or feats.get("markov_mc_prob"),
                "ENTRY": t.get("entry"),
                "SL": t.get("sl"),
                "TP": t.get("tp"),
                "RR": t.get("rr_type"),
                "SMC_SCORE": feats.get("SMC_SCORE") or feats.get("smc_score"),
                "BUY_SCORE": feats.get("BUY_SCORE"),
                "SELL_SCORE": feats.get("SELL_SCORE"),
                "RISK_USD": _risk_usd(t),
                "CANDLE_TIME": str(t.get("created_at") or t.get("log_time") or "")[:19],
            }
            break

    # ── Recent trades in range ────────────────────────────────────────────────
    recent = []
    for t in reversed(range_trades[-50:]):
        tt = _parse_dt(t.get("created_at") or t.get("log_time"))
        pnl = float(t.get("pnl") or 0)
        recent.append({
            "id": t.get("id"),
            "signal": t.get("signal"),
            "time": str(tt)[:16] if tt else "—",
            "entry": t.get("entry"),
            "sl": t.get("sl"),
            "tp": t.get("tp"),
            "tp1": t.get("tp1"),
            "tp2": t.get("tp2"),
            "tp3": t.get("tp3"),
            "exit_price": t.get("exit_price"),
            "pnl": round(pnl, 2),
            "rr_type": t.get("rr_type"),
            "sl_trail_status": t.get("sl_trail_status"),
            "status": t.get("status"),
            "note": t.get("note"),
            "result": "win" if pnl > 1 else ("loss" if pnl < -1 else "be"),
        })

    range_label = (
        f"{start_date or 'beginning'} → {end_date or 'today'}"
    )
    return {
        "range": range_label,
        "start_date": start_date or "",
        "end_date": end_date or "",
        "total_trades_in_db": len(all_trades),
        "training_trades_count": len(training_trades),
        "range_trades_count": len(range_trades),
        "stats": stats,
        "adaptive": adaptive,
        "model": model,
        "patterns": patterns,
        "last_signal": last_signal,
        "recent_trades": recent,
    }


# ── Live-replay simulation (bar-by-bar signal engine on historical OHLCV) ────

class SimulateParams(BaseModel):
    start_date: str
    end_date: str
    lot_size: float = 0.1
    mc_floor: float = 42.0
    rr_mode: str = "auto"   # auto | 1:2 | 1:3 | 1:4
    capital: float = 1000.0
    symbol: str = "GOLD"
    dynamic_lots: bool = True


def _run_simulation_sync(job_id: str, params: SimulateParams) -> None:
    """Runs in a thread pool — updates _jobs[job_id] throughout."""
    from analysis.backtest_engine import run_signal_backtest

    def _progress(done: int, total: int, msg: str = "") -> None:
        pct = int(done / total * 100) if total > 0 else 0
        _jobs[job_id].update({"progress": pct, "message": msg})

    try:
        result = run_signal_backtest(
            start_date=params.start_date,
            end_date=params.end_date,
            lot_size=params.lot_size,
            mc_floor=params.mc_floor,
            rr_mode=params.rr_mode,
            progress_cb=_progress,
            capital=params.capital,
            symbol=params.symbol or "GOLD",
            dynamic_lots=bool(params.dynamic_lots),
        )
        if result.get("error"):
            _jobs[job_id].update({"status": "error", "error": result["error"], "progress": 100})
        else:
            _jobs[job_id].update({
                "status": "done",
                "progress": 100,
                "message": "Simulation complete",
                "result": result,
            })
    except Exception as exc:
        logger.exception("Simulation job %s failed", job_id)
        _jobs[job_id].update({"status": "error", "error": str(exc), "progress": 100})


@router.post("/simulate/start")
async def start_simulation(
    params: SimulateParams,
    _: dict = Depends(get_current_user),
):
    """
    Start a bar-by-bar live-signal backtest in the background.
    Returns a job_id to poll for progress/results.
    """
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Starting…",
        "result": None,
        "error": None,
        "start_date": params.start_date,
        "end_date": params.end_date,
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_simulation_sync, job_id, params)
    return {"job_id": job_id, "status": "running"}


@router.get("/simulate/status/{job_id}")
async def simulation_status(
    job_id: str,
    _: dict = Depends(get_current_user),
):
    """Poll for simulation progress and results."""
    job = _jobs.get(job_id)
    if job is None:
        return {"status": "not_found", "error": f"Job {job_id} not found"}
    return job


# ── Visual replay endpoint — candles + signals for chart playback ─────────────

def _run_candles_job(
    job_id: str,
    start_date: str,
    end_date: str,
    mc_floor: float,
    lot_size_arg: float = 0.1,
    rr_mode: str = "auto",
    capital: float = 1000.0,
    symbol: str = "GOLD",
    dynamic_lots: bool = True,
) -> None:
    """
    Background job: fetch OHLCV + run algorithm, store candles+events in _jobs.
    """
    from datetime import timedelta
    from config import ENTRY_COOLDOWN_BARS, LOSS_PNL_PROTECT, MC_FLOOR_BASE
    from data.historical import fetch_symbol_ohlcv, resample_ohlcv
    from analysis.session import set_backtest_bar_time
    from analysis.adaptive_guard import rebuild as ag_rebuild, check as ag_check
    from analysis.concept_engine import (
        rebuild as concept_rebuild,
        get_ranking as concept_ranking,
        get_state as concept_state,
        snapshot as concept_snapshot,
        restore as concept_restore,
    )
    from analysis.backtest_engine import (
        warmup_days_for_backtest,
        signal_slices_ready,
        _parse_utc,
        _margin,
    )
    from indicators.prepare import prepare_df
    from strategy.signal import _compute_signal
    from strategy.trade_simulator import apply_rr_mode
    import traceback

    def _prog(pct: int, msg: str):
        _jobs[job_id].update({"progress": pct, "message": msg})

    try:
        _prog(2, "Fetching historical OHLCV…")
        start_dt = _parse_utc(start_date)
        end_dt   = _parse_utc(end_date).replace(hour=23, minute=59, second=59)
        days     = max((end_dt - start_dt).days + 1, 1)

        # Always use 5m — same as live dashboard (1m/3m/5m timeframes)
        # Extra days before range so 4h rolling indicators are warm at bar 0
        interval = "5m"
        warmup_days = warmup_days_for_backtest()
        warmup_start = (start_dt - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        fetch_end    = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            df_raw = fetch_symbol_ohlcv(
                symbol or "GOLD", warmup_start, fetch_end, interval=interval
            )
        except Exception as fetch_exc:
            _jobs[job_id].update({"status": "error", "error": str(fetch_exc)})
            return
        if df_raw is None or df_raw.empty:
            _jobs[job_id].update({"status": "error", "error": f"No data returned for {symbol}"})
            return

        warmup_bars = int((df_raw.index < start_dt).sum())
        if warmup_bars < 400:
            logger.warning(
                "Backtest: only %d warmup bars before %s — fetch may be too shallow; "
                "try a more recent preset or shorter range",
                warmup_bars, start_date,
            )

        avail_from = str(df_raw.index[0])[:10]
        avail_to   = str(df_raw.index[-1])[:10]
        # Clamp range end to last available hist bar (cache may lag "today")
        data_end = df_raw.index[-1]
        if end_dt > data_end.to_pydatetime():
            end_dt = data_end.to_pydatetime()
            logger.info(
                "Backtest %s: clamped end to hist %s (requested %s)",
                symbol,
                avail_to,
                end_date,
            )

        in_range_mask = (df_raw.index >= start_dt) & (df_raw.index <= end_dt)
        in_range_idx  = df_raw.index[in_range_mask]
        if len(in_range_idx) == 0:
            _jobs[job_id].update({
                "status": "error",
                "error": (
                    f"No bars found for {symbol} {start_date} → {end_date}. "
                    f"Local Dukascopy hist available: {avail_from} → {avail_to}. "
                    f"Pick a range inside that window (Last 30d / 60d usually works)."
                ),
            })
            return

        _prog(10, "Preparing indicators…")
        # 5m base → resample to 15m / 1h / 4h (same multi-TF structure as live)
        df_15m = resample_ohlcv(df_raw, "15min")
        df_1h  = resample_ohlcv(df_raw, "1h")
        df_4h  = resample_ohlcv(df_raw, "4h")

        full_5m  = prepare_df(df_raw.copy())
        full_15m = prepare_df(df_15m.copy())
        full_1h  = prepare_df(df_1h.copy())
        full_4h  = prepare_df(df_4h.copy())

        # Use full_5m's index as source of truth (prepare_df may drop NaN rows)
        # This prevents KeyError when get_loc() can't find a bar dropped by dropna()
        in_range_mask_5m = (full_5m.index >= start_dt) & (full_5m.index <= end_dt)
        in_range_idx     = full_5m.index[in_range_mask_5m]

        # Serialise candles using the prepared index (guaranteed to exist)
        candles = []
        for ts, row in full_5m[in_range_mask_5m].iterrows():
            candles.append({
                "time":   int(ts.timestamp()),
                "open":   round(float(row["open"]), 2),
                "high":   round(float(row["high"]), 2),
                "low":    round(float(row["low"]), 2),
                "close":  round(float(row["close"]), 2),
                "volume": round(float(row.get("volume", 0)), 0),
            })

        total = len(in_range_idx)
        events     = []   # {candle_index, type, ...}
        signal_log = []   # per-bar evaluation log
        cooldown_until_step = -1
        closed_sim: list = []
        lot_size   = lot_size_arg   # from capital input / dynamic desk
        backtest_symbol = str(symbol or "GOLD").upper().strip()
        _prog(8, f"Loaded {backtest_symbol} history…")
        start_capital = float(capital or 1000.0)
        base_mc = float(mc_floor if mc_floor is not None else MC_FLOOR_BASE)
        # Fresh concept learning for this timeline (walk-forward inside the range)
        _concept_snap = concept_snapshot()
        concept_rebuild([])
        from config import (
            TF_HOP_ENABLED,
            TF_HOP_MAX_LEGS,
            ENTRY_COOLDOWN_BARS as _ECOOL,
            MTF_SCAN_ENABLED,
            MTF_MAX_CONCURRENT,
            MTF_SCAN_TFS,
            symbol_entry_brain,
            symbol_rr,
        )
        from strategy.instrument_scanner import resolve_entry_scan_tfs

        hop_max = int(MTF_MAX_CONCURRENT or 8)
        open_trades: list = []
        MAX_LOG    = 8000  # cap to avoid huge payloads
        # Last seen candle time per TF — only evaluate a TF when its bar just closed
        last_tf_bar: dict[str, object] = {}
        scan_tfs = resolve_entry_scan_tfs(backtest_symbol)
        entry_brain = symbol_entry_brain(backtest_symbol)
        pair_rr = symbol_rr(backtest_symbol, default=4.0)

        # Extra resamples for independent MTF holds
        df_3m_bt = resample_ohlcv(df_raw, "3min")
        full_3m = prepare_df(df_3m_bt.copy()) if df_3m_bt is not None and not df_3m_bt.empty else None
        # 1m not available from 5m history — scanner skips missing frames
        bars_not_ready = 0
        bars_evaluated = 0
        first_signal_bar = None
        first_signal_time = None
        first_trade_bar = None
        first_trade_time = None

        _prog(
            15,
            f"Running {backtest_symbol} · {entry_brain} @ {'/'.join(scan_tfs)} · {total} bars…",
        )

        for step, bar_time in enumerate(in_range_idx):
            if step % 100 == 0:
                pct = 15 + int(step / total * 75)
                _prog(pct, f"Scanning bar {step}/{total}…")

            bar_pos = full_5m.index.get_loc(bar_time)

            p5m  = full_5m.iloc[:bar_pos + 1]
            p15m = full_15m[full_15m.index <= bar_time]
            p1h  = full_1h[full_1h.index <= bar_time]
            p4h  = full_4h[full_4h.index <= bar_time]

            if not signal_slices_ready(p5m, p15m, p1h, p4h):
                bars_not_ready += 1
                continue

            bar_ts = int(bar_time.timestamp())
            bar_day = str(bar_time)[:10]
            bar = full_5m.loc[bar_time]
            h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])

            # Manage all open TF-hop legs (partials + trailing)
            still_open = []
            any_closed = False
            from strategy.trade_simulator import update_trade_on_bar

            for ot in open_trades:
                # Use each trade's own desk-sized lot (not the fallback preview)
                if update_trade_on_bar(ot, h, l, c, ot.get("lot_size")):
                    ot["exit_time"] = bar_ts
                    ot["status"] = "closed"
                    ot["candle_time"] = str(bar_time)[:19]
                    events.append({**ot, "type": "trade_close", "candle_index": step})
                    closed_sim.append(dict(ot))
                    any_closed = True
                else:
                    still_open.append(ot)
            open_trades = still_open
            if any_closed:
                ag_rebuild(
                    closed_sim,
                    loss_pnl_protect=LOSS_PNL_PROTECT,
                    mc_floor_base=base_mc,
                    start_capital=start_capital,
                    as_of_day=bar_day,
                )
                concept_rebuild(closed_sim)
                if not open_trades:
                    cooldown_until_step = step + _ECOOL

            # Room for more independent TF holds?
            if len(open_trades) >= hop_max:
                continue

            if step < cooldown_until_step and not open_trades:
                continue

            # Live parity: adaptive + equity bank/protect/harvest before new entries
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

            # Slice frames up to this bar for independent MTF scan
            p3m = full_3m[full_3m.index <= bar_time] if full_3m is not None else None
            frames = {
                "1m": None,  # not available on 5m historical feed
                "3m": p3m if p3m is not None and len(p3m) >= 40 else None,
                "5m": p5m,
                "15m": p15m,
                "1h": p1h,
                "4h": p4h,
            }

            from strategy.mtf_scanner import scan_all_timeframes, open_tfs as mtf_open_tfs

            # Live parity: only the pair's SYMBOL_SCAN_TFS (deep-hunt map)
            scan_universe = tuple(scan_tfs) if scan_tfs else tuple(MTF_SCAN_TFS)
            due_tfs = []
            for tf in scan_universe:
                fr = frames.get(tf)
                if fr is None or len(fr) < 40:
                    continue
                tf_last = fr.index[-1]
                if last_tf_bar.get(tf) == tf_last:
                    continue
                last_tf_bar[tf] = tf_last
                due_tfs.append(tf)

            set_backtest_bar_time(bar_time.to_pydatetime())
            try:
                # Dashboard-style signal for the log — wired to pair brain
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
                    result["ENTRY_BRAIN"] = entry_brain
                mtf_hits = []
                if MTF_SCAN_ENABLED and due_tfs:
                    mtf_hits = scan_all_timeframes(
                        frames,
                        only_tfs=tuple(due_tfs),
                        symbol=backtest_symbol,
                    )
                    for h in mtf_hits:
                        h["SYMBOL"] = backtest_symbol
                        h["INSTRUMENT"] = backtest_symbol
                        h["ENTRY_BRAIN"] = entry_brain
                elif not MTF_SCAN_ENABLED:
                    # Fallback: single result only
                    if result and result.get("SIGNAL") in ("BUY", "SELL"):
                        result["EXEC_TF"] = (scan_universe[0] if scan_universe else "5m")
                        mtf_hits = [result]
            except Exception:
                continue
            finally:
                set_backtest_bar_time(None)

            bars_evaluated += 1

            result = apply_rr_mode(result, rr_mode) if result else None
            if result is None:
                continue

            signal  = result.get("SIGNAL")
            mc_prob = float(result.get("SUCCESS_PERCENT") or result.get("MARKOV_MC_PROB") or 0)
            markov_prob = float(result.get("MARKOV_MC_PROB") or 0)

            # ── Raw indicator values from the prepared DF ─────────────────────
            row = p5m.iloc[-1]
            def _f(col, dec=2):
                v = row.get(col)
                return round(float(v), dec) if v is not None and str(v) != 'nan' else None

            # ── Construct "why" explanation ────────────────────────────────────
            buy_score  = result.get("BUY_SCORE")  or result.get("buy_score")
            sell_score = result.get("SELL_SCORE") or result.get("sell_score")
            smc_score  = result.get("SMC_SCORE")  or result.get("smc_score")
            state      = result.get("STATE", "")
            reason_raw = result.get("REASON") or result.get("reason") or ""

            mtf_dirs = [f"{h.get('EXEC_TF')}:{h.get('SIGNAL')}" for h in mtf_hits]
            if signal in ("BUY", "SELL") or mtf_hits:
                entry_px = result.get("ENTRY")
                sl_px    = result.get("SL")
                tp_px    = result.get("TP")
                rr       = result.get("RR") or ""
                dir_word = "demand zone" if signal == "BUY" else "supply zone"
                score_w  = buy_score if signal == "BUY" else sell_score
                why = (
                    f"5m={signal or 'FLAT'}. MTF hits={mtf_dirs or 'none'}. "
                    f"State={state}. Score={score_w}. SMC={smc_score}. "
                    f"MC={mc_prob:.0f}%. RR={rr}."
                )
                if first_signal_bar is None and (signal in ("BUY", "SELL") or mtf_hits):
                    first_signal_bar = step
                    first_signal_time = str(bar_time)[:16]
            else:
                why = reason_raw or f"No valid setup. Buy={buy_score} Sell={sell_score} MC={mc_prob:.0f}%"
                entry_px = result.get("ENTRY")
                sl_px = result.get("SL")
                tp_px = result.get("TP")

            # ── Build log entry ────────────────────────────────────────────────
            bar_row  = full_5m.loc[bar_time]
            decision = "NO SIGNAL"
            if mtf_hits:
                decision = "TRADE"
            elif signal in ("BUY", "SELL"):
                entry_px = result.get("ENTRY")
                sl_px    = result.get("SL")
                tp_px    = result.get("TP")
                if entry_px and sl_px:
                    decision = "BLOCKED_MC" if mc_prob < eff_mc_floor else "TRADE"

            if len(signal_log) < MAX_LOG:
                signal_log.append({
                    "time":        bar_ts,
                    "open":        round(float(bar_row["open"]),  2),
                    "high":        round(float(bar_row["high"]),  2),
                    "low":         round(float(bar_row["low"]),   2),
                    "close":       round(float(bar_row["close"]), 2),
                    "signal":      signal or "NO TRADE",
                    "decision":    decision,
                    "mc_prob":     round(mc_prob, 1),
                    "markov_prob": round(markov_prob, 1),
                    "state":       state,
                    "atr":         _f("ATR"),
                    "rsi":         _f("RSI", 1),
                    "ema9":        _f("EMA9"),
                    "ema21":       _f("EMA21"),
                    "atr_ratio":   _f("ATR_RATIO"),
                    "macd_hist":   _f("MACD_HIST"),
                    "buy_score":   round(float(buy_score), 2)  if buy_score  is not None else None,
                    "sell_score":  round(float(sell_score), 2) if sell_score is not None else None,
                    "smc_score":   round(float(smc_score), 2)  if smc_score  is not None else None,
                    "entry":       round(float(entry_px), 2) if signal in ("BUY","SELL") and entry_px else None,
                    "sl":          round(float(sl_px),    2) if signal in ("BUY","SELL") and sl_px    else None,
                    "tp":          round(float(tp_px),    2) if signal in ("BUY","SELL") and tp_px    else None,
                    "why":         why,
                    "vp_location": result.get("VP_LOCATION"),
                    "candle_index": step,
                })

            # ── Open independent MTF trades (one hold per TF) ─────────────────
            if not mtf_hits:
                if mc_prob >= 20:
                    events.append({
                        "type":         "signal",
                        "candle_index": step,
                        "time":         bar_ts,
                        "signal":       "NO TRADE",
                        "mc_prob":      round(mc_prob, 1),
                        "reason":       why,
                    })
                continue

            from strategy.trade_simulator import new_open_trade

            occupied = mtf_open_tfs(open_trades)
            for hit in mtf_hits:
                if len(open_trades) >= hop_max:
                    break
                hit = apply_rr_mode(hit, rr_mode) or hit
                h_signal = hit.get("SIGNAL")
                if h_signal not in ("BUY", "SELL"):
                    continue
                exec_tf = str(hit.get("EXEC_TF") or "5m")
                if exec_tf in occupied:
                    continue

                h_entry = hit.get("ENTRY")
                h_sl = hit.get("SL")
                h_tp = hit.get("TP")
                h_mc = float(hit.get("SUCCESS_PERCENT") or hit.get("MARKOV_MC_PROB") or 0)
                if h_entry is None or h_sl is None:
                    continue

                buy_s = float(hit.get("BUY_SCORE") or buy_score or 0)
                sell_s = float(hit.get("SELL_SCORE") or sell_score or 0)
                feats = {
                    "STATE": hit.get("STATE") or state,
                    "mc_prob": h_mc,
                    "markov_mc_prob": float(hit.get("MARKOV_MC_PROB") or h_mc),
                    "score_margin": abs(buy_s - sell_s),
                    "hour_utc": int(bar_time.hour) if hasattr(bar_time, "hour") else None,
                    "exec_tf": exec_tf,
                    "concepts": hit.get("CONCEPTS") or {},
                    "concept_quality": hit.get("CONCEPT_QUALITY"),
                }
                ok_adapt, adapt_reason = ag_check(feats)
                if not ok_adapt:
                    events.append({
                        "type": "signal", "candle_index": step, "time": bar_ts,
                        "signal": h_signal, "mc_prob": round(h_mc, 1), "blocked": True,
                        "reason": f"[{exec_tf}] {adapt_reason}",
                        "entry": round(float(h_entry), 2), "sl": round(float(h_sl), 2),
                        "tp": round(float(h_tp), 2) if h_tp else None,
                    })
                    continue
                if h_mc < eff_mc_floor:
                    events.append({
                        "type": "signal", "candle_index": step, "time": bar_ts,
                        "signal": h_signal, "mc_prob": round(h_mc, 1), "blocked": True,
                        "reason": f"[{exec_tf}] MC {h_mc:.0f}% < floor {eff_mc_floor:.0f}%",
                        "entry": round(float(h_entry), 2), "sl": round(float(h_sl), 2),
                        "tp": round(float(h_tp), 2) if h_tp else None,
                    })
                    continue

                if first_trade_bar is None:
                    first_trade_bar = step
                    first_trade_time = str(bar_time)[:16]

                trade_lot = lot_size
                if dynamic_lots:
                    try:
                        from strategy.dynamic_lots import lot_for_profit_target

                        # Same desk sizer as live: alloc + TARGET_PROFIT + MAX_RISK
                        eq = start_capital + sum(float(t.get("pnl") or 0) for t in closed_sim)
                        sized = lot_for_profit_target(
                            backtest_symbol,
                            entry=float(h_entry),
                            sl=float(h_sl),
                            tp=float(h_tp) if h_tp else None,
                            rr=hit.get("RR") or pair_rr,
                            capital=max(50.0, eq),
                        )
                        trade_lot = float(sized["lot_size"])
                    except Exception:
                        trade_lot = lot_size

                ot = new_open_trade(
                    signal=h_signal,
                    entry=float(h_entry),
                    sl=float(h_sl),
                    tp=float(h_tp) if h_tp else None,
                    tp1=hit.get("TP1"),
                    tp2=hit.get("TP2"),
                    tp3=hit.get("TP3"),
                    rr_type=hit.get("RR"),
                    lot_size=trade_lot,
                    mc_prob=h_mc,
                    type="trade_open",
                    candle_index=step,
                    time=bar_ts,
                    entry_time=bar_ts,
                    margin_usd=_margin(float(h_entry), trade_lot),
                    features={
                        **(feats or {}),
                        "symbol": backtest_symbol,
                        "SYMBOL": backtest_symbol,
                        "entry_brain": entry_brain,
                        "brain": hit.get("BRAIN") or entry_brain,
                    },
                    candle_time=str(bar_time)[:19],
                    symbol=backtest_symbol,
                    SYMBOL=backtest_symbol,
                    entry_brain=entry_brain,
                    brain=hit.get("BRAIN") or entry_brain,
                    exec_tf=exec_tf,
                    mtf_independent=True,
                )
                open_trades.append(ot)
                occupied.add(exec_tf)
                events.append({**ot})

        # Close any still-open MTF holds at range end
        if open_trades:
            from strategy.trade_simulator import close_at_market

            last = full_5m[full_5m.index <= end_dt]
            if not last.empty:
                ep = float(last.iloc[-1]["close"])
                ets = int(last.index[-1].timestamp())
                for ot in open_trades:
                    close_at_market(ot, ep, ets)
                    events.append({**ot, "type": "trade_close", "candle_index": total - 1})
            open_trades = []

        # Quick stats (same outcome rules as live)
        from strategy.trade_simulator import stats_from_trades

        trade_opens = [e for e in events if e.get("type") == "trade_open"]
        closed_trades = [e for e in events if e.get("type") == "trade_close" and e.get("status") == "closed"]
        outcome_stats = stats_from_trades(closed_trades)
        wins = outcome_stats["wins"]
        losses = outcome_stats["losses"]
        realized = outcome_stats["realized_pnl"]
        open_lots = [
            float(e["lot_size"])
            for e in trade_opens
            if e.get("lot_size") is not None
        ]
        avg_lot = round(sum(open_lots) / len(open_lots), 2) if open_lots else round(float(lot_size), 2)
        try:
            from strategy.dynamic_lots import lot_for_profit_target as _lpt

            desk_preview = _lpt(backtest_symbol, capital=start_capital, rr=pair_rr)
        except Exception:
            desk_preview = {}

        _prog(100, "Done")
        _jobs[job_id].update({
            "status":   "done",
            "result": {
                "candles":       candles,
                "events":        events,
                "signal_log":    signal_log,
                "data_range":    {"from": avail_from, "to": avail_to},
                "bars_scanned":  total,
                "stats": {
                    "executed":     len(trade_opens),
                    "closed":       outcome_stats["closed"],
                    "wins":         wins,
                    "losses":       losses,
                    "breakevens":   outcome_stats["breakevens"],
                    "realized_pnl": realized,
                    "win_rate":     outcome_stats["win_rate"],
                    "lot_size":     avg_lot,
                    "lot_size_min": round(min(open_lots), 2) if open_lots else avg_lot,
                    "lot_size_max": round(max(open_lots), 2) if open_lots else avg_lot,
                    "desk_lot_preview": desk_preview.get("lot_size"),
                    "allocation_mult": desk_preview.get("allocation_mult"),
                    "dynamic_lots": bool(dynamic_lots),
                    "start_capital": start_capital,
                    "first_price":  candles[0]["open"] if candles else None,
                    "last_price":   candles[-1]["close"] if candles else None,
                    "avg_margin_usd": round(float(trade_opens[0]["margin_usd"]), 2) if trade_opens else None,
                    "avg_leverage": 1000,
                    "trade_units":  round(avg_lot * 100),
                    "rr_mode":      rr_mode,
                    "bars_evaluated": bars_evaluated,
                    "bars_not_ready": bars_not_ready,
                    "first_signal_bar": first_signal_bar,
                    "first_signal_time": first_signal_time,
                    "first_trade_bar": first_trade_bar,
                    "first_trade_time": first_trade_time,
                    "first_trade_pct": (
                        round(100.0 * first_trade_bar / max(total - 1, 1), 1)
                        if first_trade_bar is not None and total > 0
                        else None
                    ),
                    "trades_from_first_setup": True,
                    "warmup_bars": warmup_bars,
                    "concept_ranking": concept_ranking(),
                    "concept_message": concept_state().get("message"),
                    "mtf_scan": bool(MTF_SCAN_ENABLED),
                    "mtf_tfs": list(scan_tfs),
                    "mtf_max_concurrent": hop_max,
                    "symbol": backtest_symbol,
                    "entry_brain": entry_brain,
                    "scan_tfs": list(scan_tfs),
                    "symbol_rr": pair_rr,
                },
            },
        })
        concept_restore(_concept_snap)

    except Exception as exc:
        logger.exception("Candles job %s failed", job_id)
        try:
            if "_concept_snap" in locals() and _concept_snap is not None:
                concept_restore(_concept_snap)
        except Exception:
            pass
        _jobs[job_id].update({"status": "error", "error": str(exc), "progress": 100})


@router.post("/candles/start")
async def start_candles_job(
    params: SimulateParams,
    _: dict = Depends(get_current_user),
):
    """Start a candles+events computation job for visual replay."""
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {
        "status": "running", "progress": 0, "message": "Starting…",
        "result": None, "error": None,
        "start_date": params.start_date, "end_date": params.end_date,
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_candles_job,
        job_id,
        params.start_date,
        params.end_date,
        params.mc_floor,
        params.lot_size,
        params.rr_mode,
        params.capital,
        params.symbol,
        params.dynamic_lots,
    )
    return {"job_id": job_id}


@router.get("/candles/status/{job_id}")
async def candles_job_status(
    job_id: str,
    _: dict = Depends(get_current_user),
):
    job = _jobs.get(job_id)
    if job is None:
        return {"status": "not_found"}
    return job
