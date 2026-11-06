#!/usr/bin/env python3
"""Backtest GOLD with live ENTRY_BRAIN (ema_20_50_pull) — 14d + 1m."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RISK_USD = 200.0
MC_FLOOR = 36.0
OUT = Path("/tmp/gold_ema_bt.json")


def _lot(symbol: str, entry: float, sl: float) -> float:
    from strategy.position_sizing import leg_pnl

    p1 = abs(float(leg_pnl("BUY", float(entry), float(sl), 1.0, symbol=symbol)))
    if p1 < 1e-12:
        return 0.01
    return max(0.01, round(RISK_USD / p1, 2))


def run_window(start: str, end: str) -> dict[str, Any]:
    from analysis.backtest_engine import (
        _resample_ohlcv,
        signal_slices_ready,
        warmup_days_for_backtest,
    )
    from analysis.session import set_backtest_bar_time
    from config import ENTRY_BRAIN, ENTRY_COOLDOWN_BARS
    from data.historical import fetch_symbol_ohlcv
    from indicators.prepare import prepare_df
    from strategy.signal import _compute_signal
    from strategy.trade_simulator import (
        apply_rr_mode,
        close_at_market,
        new_open_trade,
        stats_from_trades,
        update_trade_on_bar,
    )

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    warmup = warmup_days_for_backtest()
    ws = (start_dt - timedelta(days=warmup)).strftime("%Y-%m-%d")
    fe = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    t0 = time.time()
    df_raw = fetch_symbol_ohlcv("GOLD", ws, fe, interval="5m")
    if df_raw is None or df_raw.empty:
        raise RuntimeError("no GOLD OHLCV")

    full_5m = prepare_df(df_raw.copy())
    full_15m = prepare_df(_resample_ohlcv(df_raw, "15min").copy())
    full_1h = prepare_df(_resample_ohlcv(df_raw, "1h").copy())
    full_4h = prepare_df(_resample_ohlcv(df_raw, "4h").copy())
    raw = df_raw.reindex(full_5m.index).ffill()
    in_range = full_5m.index[(full_5m.index >= start_dt) & (full_5m.index <= end_dt)]

    open_trade = None
    cooldown = -1
    trades: list[dict] = []
    signals_seen = 0

    for step, bar_time in enumerate(in_range):
        bar_pos = full_5m.index.get_loc(bar_time)
        row = raw.loc[bar_time] if bar_time in raw.index else full_5m.loc[bar_time]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])

        if open_trade is not None:
            lot = float(open_trade.get("lot_size") or 0.01)
            if update_trade_on_bar(open_trade, h, l, c, lot, mgmt_mode="partials_trail"):
                open_trade["exit_time"] = str(bar_time)[:16]
                open_trade["status"] = "closed"
                trades.append(open_trade)
                open_trade = None
                cooldown = step + int(ENTRY_COOLDOWN_BARS or 3)
            continue
        if step < cooldown:
            continue

        p5m = full_5m.iloc[: bar_pos + 1]
        p15m = full_15m[full_15m.index <= bar_time]
        p1h = full_1h[full_1h.index <= bar_time]
        p4h = full_4h[full_4h.index <= bar_time]
        if not signal_slices_ready(p5m, p15m, p1h, p4h):
            continue

        set_backtest_bar_time(bar_time.to_pydatetime())
        try:
            r = _compute_signal(
                df_1m=p5m,
                df_3m=p5m,
                df_5m=p5m,
                df_15m=p15m,
                df_1h=p1h,
                df_4h=p4h,
                symbol="GOLD",
            )
        finally:
            set_backtest_bar_time(None)

        if r.get("SIGNAL") not in ("BUY", "SELL"):
            continue
        signals_seen += 1
        r = apply_rr_mode(dict(r), "auto")
        if r is None:
            continue
        mc = float(r.get("SUCCESS_PERCENT") or r.get("MARKOV_MC_PROB") or 0)
        if mc < MC_FLOOR:
            continue
        entry, sl, tp = r.get("ENTRY"), r.get("SL"), r.get("TP")
        if entry is None or sl is None:
            continue
        lot = _lot("GOLD", float(entry), float(sl))
        open_trade = new_open_trade(
            signal=r["SIGNAL"],
            entry=float(entry),
            sl=float(sl),
            tp=float(tp) if tp else None,
            tp1=r.get("TP1"),
            tp2=r.get("TP2"),
            tp3=r.get("TP3"),
            rr_type=r.get("RR"),
            lot_size=lot,
            mc_prob=mc,
            entry_time=str(bar_time)[:16],
            symbol="GOLD",
            SYMBOL="GOLD",
            mgmt_mode="partials_trail",
            brain=r.get("BRAIN"),
        )
        if step and step % 400 == 0:
            print(f"  {start}→{end}: {step}/{len(in_range)}", flush=True)

    if open_trade is not None:
        last = in_range[-1]
        row = raw.loc[last] if last in raw.index else full_5m.loc[last]
        close_at_market(open_trade, float(row["close"]), str(last)[:16])
        trades.append(open_trade)

    st = stats_from_trades(trades)
    return {
        "start": start,
        "end": end,
        "bars": len(in_range),
        "signals": signals_seen,
        "trades": st["closed"],
        "wins": st["wins"],
        "losses": st["losses"],
        "breakevens": st["breakevens"],
        "win_rate": st["win_rate"],
        "pnl": st["realized_pnl"],
        "sec": round(time.time() - t0, 1),
        "entry_brain": ENTRY_BRAIN,
        "sample": [
            {
                "t": t.get("entry_time"),
                "side": t.get("signal"),
                "pnl": t.get("pnl"),
                "rr": t.get("rr_type"),
                "brain": t.get("brain"),
            }
            for t in trades[:8]
        ],
    }


def main() -> int:
    from config import ENTRY_BRAIN

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    windows = [
        ("14d", (end - timedelta(days=13)).isoformat(), end.isoformat()),
        ("1m", (end - timedelta(days=29)).isoformat(), end.isoformat()),
    ]
    print(f"GOLD EMA backtest brain={ENTRY_BRAIN}", windows, flush=True)
    results = {}
    for label, a, b in windows:
        print(f"\n=== {label} {a}→{b} ===", flush=True)
        results[label] = run_window(a, b)
        r = results[label]
        print(
            f"  bars={r['bars']} signals={r['signals']} trades={r['trades']} "
            f"wr={100*float(r['win_rate']):.0f}% pnl=${r['pnl']:.0f} t={r['sec']}s",
            flush=True,
        )

    payload = {
        "symbol": "GOLD",
        "risk_usd": RISK_USD,
        "mc_floor": MC_FLOOR,
        "entry_brain": ENTRY_BRAIN,
        "windows": results,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT}", flush=True)
    print("\n| window | trades | WR | PnL @$200R |", flush=True)
    print("|---|---:|---:|---:|", flush=True)
    for label, _, _ in windows:
        r = results[label]
        print(
            f"| {label} | {r['trades']} | {100*float(r['win_rate']):.0f}% | {r['pnl']} |",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
