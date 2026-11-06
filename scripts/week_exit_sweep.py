#!/usr/bin/env python3
"""Last-week exit/lot sweep across INSTRUMENT_UNIVERSE.

1) Cache bar signals once per symbol (expensive)
2) Replay RR × management modes on desk lots
3) Scale PnL for lot multipliers (linear)

Writes JSON to /tmp/week_exit_sweep.json and prints a markdown table.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UNIVERSE = (
    "GOLD",
    "EURUSD",
    "USDCHF",
    "BTCUSD",
    "ETHUSD",
    "USOIL",
    "SILVER",
    "GBPUSD",
)

# RR overlay × exit management
MODES = [
    ("rr12_fixed", "1:2", "fixed"),
    ("rr13_fixed", "1:3", "fixed"),
    ("rr14_fixed", "1:4", "fixed"),
    ("auto_fixed", "auto", "fixed"),
    ("auto_partials_be", "auto", "partials_be"),
    ("auto_partials_trail", "auto", "partials_trail"),  # live-style
    ("rr12_partials_trail", "1:2", "partials_trail"),
    ("rr13_partials_trail", "1:3", "partials_trail"),
]

LOT_MULTS = (0.5, 1.0, 1.5)
MC_FLOOR = 42.0
CAPITAL = 9928.0
OUT_PATH = Path("/tmp/week_exit_sweep.json")


def _week_range() -> tuple[str, str]:
    """Last completed Mon–Fri week ending before today (UTC)."""
    today = datetime.now(timezone.utc).date()
    # Saturday Jul 18 2026 → Fri Jul 17; Mon Jul 13
    weekday = today.weekday()  # Mon=0
    last_fri = today - timedelta(days=(weekday - 4) % 7 or 7)
    last_mon = last_fri - timedelta(days=4)
    return last_mon.isoformat(), last_fri.isoformat()


def _desk_lot(symbol: str) -> float:
    from strategy.dynamic_lots import lot_for_profit_target

    snap = lot_for_profit_target(symbol, capital=CAPITAL)
    return float(snap.get("lot_size") or 0.01)


def _prepare_frames(df_raw: pd.DataFrame):
    from analysis.backtest_engine import _resample_ohlcv
    from indicators.prepare import prepare_df

    full_5m = prepare_df(df_raw.copy())
    full_15m = prepare_df(_resample_ohlcv(df_raw, "15min").copy())
    full_1h = prepare_df(_resample_ohlcv(df_raw, "1h").copy())
    full_4h = prepare_df(_resample_ohlcv(df_raw, "4h").copy())
    return full_5m, full_15m, full_1h, full_4h


def cache_signals(
    symbol: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    from analysis.backtest_engine import signal_slices_ready, warmup_days_for_backtest
    from analysis.session import set_backtest_bar_time
    from data.historical import fetch_symbol_ohlcv
    from strategy.signal import _compute_signal

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    warmup = warmup_days_for_backtest()
    warmup_start = (start_dt - timedelta(days=warmup)).strftime("%Y-%m-%d")
    fetch_end = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    t0 = time.time()
    df_raw = fetch_symbol_ohlcv(symbol, warmup_start, fetch_end, interval="5m")
    if df_raw is None or df_raw.empty:
        raise RuntimeError(f"no OHLCV for {symbol}")

    full_5m, full_15m, full_1h, full_4h = _prepare_frames(df_raw)
    in_range = full_5m.index[(full_5m.index >= start_dt) & (full_5m.index <= end_dt)]
    raw_5m = df_raw.reindex(full_5m.index).ffill()

    signals: list[Optional[dict]] = []
    bars_meta: list[dict] = []
    evaluated = 0
    for step, bar_time in enumerate(in_range):
        bar_pos = full_5m.index.get_loc(bar_time)
        p5m = full_5m.iloc[: bar_pos + 1]
        p15m = full_15m[full_15m.index <= bar_time]
        p1h = full_1h[full_1h.index <= bar_time]
        p4h = full_4h[full_4h.index <= bar_time]
        row = raw_5m.loc[bar_time] if bar_time in raw_5m.index else full_5m.loc[bar_time]
        bars_meta.append(
            {
                "t": str(bar_time),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
            }
        )
        if not signal_slices_ready(p5m, p15m, p1h, p4h):
            signals.append(None)
            continue
        set_backtest_bar_time(bar_time.to_pydatetime())
        try:
            result = _compute_signal(
                df_1m=p5m,
                df_3m=p5m,
                df_5m=p5m,
                df_15m=p15m,
                df_1h=p1h,
                df_4h=p4h,
                symbol=symbol,
            )
            evaluated += 1
            if result.get("SIGNAL") in ("BUY", "SELL"):
                # keep compact payload
                signals.append(
                    {
                        "SIGNAL": result.get("SIGNAL"),
                        "ENTRY": result.get("ENTRY"),
                        "SL": result.get("SL"),
                        "TP": result.get("TP"),
                        "TP1": result.get("TP1"),
                        "TP2": result.get("TP2"),
                        "TP3": result.get("TP3"),
                        "RR": result.get("RR"),
                        "SUCCESS_PERCENT": result.get("SUCCESS_PERCENT"),
                        "MARKOV_MC_PROB": result.get("MARKOV_MC_PROB"),
                        "STATE": result.get("STATE"),
                        "ATR": result.get("ATR"),
                        "BUY_SCORE": result.get("BUY_SCORE"),
                        "SELL_SCORE": result.get("SELL_SCORE"),
                        "TP_SOURCE": result.get("TP_SOURCE"),
                        "BRAIN": result.get("BRAIN"),
                    }
                )
            else:
                signals.append(None)
        except Exception:
            signals.append(None)
        finally:
            set_backtest_bar_time(None)
        if step and step % 200 == 0:
            print(f"  {symbol}: cached {step}/{len(in_range)} bars…", flush=True)

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "bars": bars_meta,
        "signals": signals,
        "evaluated": evaluated,
        "fetch_sec": round(time.time() - t0, 1),
        "desk_lot": _desk_lot(symbol),
    }


def _replay(
    cache: dict[str, Any],
    rr_mode: str,
    mgmt_mode: str,
    lot: float,
    mc_floor: float = MC_FLOOR,
) -> dict[str, Any]:
    from config import ENTRY_COOLDOWN_BARS
    from strategy.trade_simulator import (
        apply_rr_mode,
        close_at_market,
        new_open_trade,
        stats_from_trades,
        update_trade_on_bar,
    )

    bars = cache["bars"]
    signals = cache["signals"]
    open_trade = None
    cooldown_until = -1
    trades: list[dict] = []

    for step, meta in enumerate(bars):
        h, l, c = meta["h"], meta["l"], meta["c"]
        if open_trade is not None:
            if update_trade_on_bar(open_trade, h, l, c, lot, mgmt_mode=mgmt_mode):
                open_trade["exit_time"] = meta["t"][:16]
                open_trade["status"] = "closed"
                trades.append(open_trade)
                open_trade = None
                cooldown_until = step + int(ENTRY_COOLDOWN_BARS or 3)
            continue

        if step < cooldown_until:
            continue

        raw = signals[step]
        if not raw:
            continue
        result = apply_rr_mode(dict(raw), rr_mode)
        if result is None:
            continue
        signal = result.get("SIGNAL")
        if signal not in ("BUY", "SELL"):
            continue
        mc = float(result.get("SUCCESS_PERCENT") or result.get("MARKOV_MC_PROB") or 0)
        if mc < mc_floor:
            continue
        entry = result.get("ENTRY")
        sl = result.get("SL")
        tp = result.get("TP")
        if entry is None or sl is None:
            continue

        open_trade = new_open_trade(
            signal=signal,
            entry=float(entry),
            sl=float(sl),
            tp=float(tp) if tp else None,
            tp1=result.get("TP1"),
            tp2=result.get("TP2"),
            tp3=result.get("TP3"),
            rr_type=result.get("RR"),
            lot_size=lot,
            mc_prob=mc,
            entry_time=meta["t"][:16],
            symbol=cache["symbol"],
            SYMBOL=cache["symbol"],
            mgmt_mode=mgmt_mode,
            brain=result.get("BRAIN"),
            tp_source=result.get("TP_SOURCE"),
        )

    if open_trade is not None:
        close_at_market(open_trade, bars[-1]["c"], bars[-1]["t"][:16])
        trades.append(open_trade)

    st = stats_from_trades(trades)
    rr_hist: dict[str, int] = {}
    brains: dict[str, int] = {}
    for t in trades:
        rr_hist[str(t.get("rr_type") or "?")] = rr_hist.get(str(t.get("rr_type") or "?"), 0) + 1
        brains[str(t.get("brain") or "?")] = brains.get(str(t.get("brain") or "?"), 0) + 1

    return {
        "trades": st["closed"],
        "wins": st["wins"],
        "losses": st["losses"],
        "breakevens": st["breakevens"],
        "win_rate": st["win_rate"],
        "pnl": st["realized_pnl"],
        "rr_hist": rr_hist,
        "brains": brains,
        "lot": lot,
    }


def main() -> int:
    start, end = _week_range()
    print(f"Week exit sweep {start} → {end}  capital=${CAPITAL:.0f}", flush=True)

    rows: list[dict] = []
    caches: dict[str, Any] = {}

    for sym in UNIVERSE:
        print(f"\n=== {sym}: caching signals ===", flush=True)
        try:
            caches[sym] = cache_signals(sym, start, end)
            print(
                f"  bars={len(caches[sym]['bars'])} eval={caches[sym]['evaluated']} "
                f"desk_lot={caches[sym]['desk_lot']} t={caches[sym]['fetch_sec']}s",
                flush=True,
            )
        except Exception as exc:
            print(f"  FAIL {sym}: {exc}", flush=True)
            traceback.print_exc()
            caches[sym] = None

    for sym, cache in caches.items():
        if not cache:
            for mode_name, rr_mode, mgmt in MODES:
                rows.append(
                    {
                        "symbol": sym,
                        "mode": mode_name,
                        "rr_mode": rr_mode,
                        "mgmt": mgmt,
                        "lot_mult": 1.0,
                        "lot": None,
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "win_rate": 0,
                        "pnl": 0,
                        "error": "no_data",
                    }
                )
            continue

        desk = float(cache["desk_lot"])
        for mode_name, rr_mode, mgmt in MODES:
            base = _replay(cache, rr_mode, mgmt, desk)
            for mult in LOT_MULTS:
                # PnL scales linearly with lot for this simulator
                scale = mult
                rows.append(
                    {
                        "symbol": sym,
                        "mode": mode_name,
                        "rr_mode": rr_mode,
                        "mgmt": mgmt,
                        "lot_mult": mult,
                        "lot": round(desk * mult, 4),
                        "trades": base["trades"],
                        "wins": base["wins"],
                        "losses": base["losses"],
                        "breakevens": base["breakevens"],
                        "win_rate": base["win_rate"],
                        "pnl": round(float(base["pnl"]) * scale, 2),
                        "rr_hist": base["rr_hist"] if mult == 1.0 else None,
                        "brains": base["brains"] if mult == 1.0 else None,
                    }
                )
            print(
                f"  {sym} {mode_name}: n={base['trades']} wr={base['win_rate']:.0f}% "
                f"pnl=${base['pnl']:.0f} @ desk lot",
                flush=True,
            )

    # Aggregate mode totals at desk lots (mult=1)
    agg: dict[str, dict] = {}
    for r in rows:
        if r.get("lot_mult") != 1.0 or r.get("error"):
            continue
        a = agg.setdefault(
            r["mode"],
            {"mode": r["mode"], "rr_mode": r["rr_mode"], "mgmt": r["mgmt"], "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        )
        a["trades"] += int(r["trades"])
        a["wins"] += int(r["wins"])
        a["losses"] += int(r["losses"])
        a["pnl"] += float(r["pnl"])

    for a in agg.values():
        d = a["wins"] + a["losses"]
        a["win_rate"] = round(100.0 * a["wins"] / d, 1) if d else 0.0
        a["pnl"] = round(a["pnl"], 2)

    payload = {
        "range": {"start": start, "end": end},
        "capital": CAPITAL,
        "mc_floor": MC_FLOOR,
        "note": (
            "Yahoo/TV 5m proxies — not exact XM ticks. "
            "Lot-desk 1:2 is a sizing assumption; live RR can still be structure/auto. "
            "Signals cached once per symbol; exits replayed."
        ),
        "aggregates": sorted(agg.values(), key=lambda x: -x["pnl"]),
        "rows": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH}", flush=True)

    print("\n### Aggregate (desk lots, all symbols)\n")
    print("| mode | rr | mgmt | trades | WR% | PnL $ |")
    print("|---|---|---|---:|---:|---:|")
    for a in payload["aggregates"]:
        print(
            f"| {a['mode']} | {a['rr_mode']} | {a['mgmt']} | "
            f"{a['trades']} | {a['win_rate']} | {a['pnl']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
