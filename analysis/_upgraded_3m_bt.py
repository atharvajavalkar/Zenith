#!/usr/bin/env python3
"""RESEARCH — 3-month paper BT on upgraded per-symbol brains.

Uses 1h yfinance proxy (same evaluators as live SYMBOL_ENTRY_BRAIN).
$200/R · forced 2R · no live wiring.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import (  # noqa: E402
    BRAINS,
    BOOK,
    RISK_USD,
    YF,
    _fetch_1h,
    _prep,
    _stats,
    monte_carlo,
    run_breakout_trades,
    run_ema_trades,
    run_fib_trades,
    run_sma_trades,
)

OUT = Path("/tmp/upgraded_3m_bt.json")


def gen_trades(symbol: str, df):
    brain = BRAINS.get(symbol.upper(), "ema_20_50_pull")
    if brain == "fib_liq":
        return brain, run_fib_trades(df)
    if brain == "sma_20_50_cross":
        return brain, run_sma_trades(df)
    if brain == "breakout_20":
        return brain, run_breakout_trades(df)
    pull, disp = (0.75, 0.70) if brain == "ema_20_50_tight" else (1.0, 0.55)
    return brain, run_ema_trades(df, pull_atr=pull, disp_atr=disp)


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=91)).isoformat()
    end_s = end.isoformat()
    print(f"Upgraded 3m BT {start} → {end_s}  ($200/R · 1h proxy)", flush=True)

    rows = []
    desk_pnl = 0.0
    desk_n = 0
    for sym in BOOK:
        try:
            raw = _fetch_1h(sym, start, end_s)
            df = _prep(raw)
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if len(df) < 120:
                print(f"  SKIP {sym}: {len(df)} bars", flush=True)
                continue
            brain, trades = gen_trades(sym, df)
            st = _stats(trades)
            mc = monte_carlo(trades)
            desk_pnl += st["pnl"]
            desk_n += st["n"]
            row = {
                "symbol": sym,
                "brain": brain,
                "bars": len(df),
                **st,
                "monte_carlo": mc,
            }
            rows.append(row)
            mc_s = (
                f"MC P(profit)={mc['prob_profit']}% medR={mc['median_R']}"
                if mc.get("paths")
                else "MC n/a"
            )
            print(
                f"  {sym:7s} {brain:18s} n={st['n']:3d} wr={st['wr']:5.1f}% "
                f"E[R]={st['exp_r']:+.3f} pnl=${st['pnl']:+.0f}  {mc_s}",
                flush=True,
            )
        except Exception as exc:
            print(f"  FAIL {sym}: {exc}", flush=True)
            rows.append({"symbol": sym, "error": str(exc)})

    payload = {
        "window": {"start": start, "end": end_s, "tf": "1h_proxy", "months": 3},
        "risk_usd": RISK_USD,
        "tickers": YF,
        "brains": BRAINS,
        "results": rows,
        "desk": {"trades": desk_n, "pnl": round(desk_pnl, 2)},
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(
        f"\nDESK 3m: trades={desk_n} pnl=${desk_pnl:+.0f}  → {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
