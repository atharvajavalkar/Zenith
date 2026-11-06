#!/usr/bin/env python3
"""Re-rank fine-tune cells requiring n>=20 on 1y."""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.xm_cost_finetune_v2 import (  # noqa: E402
    BRAINS,
    MAX_SPR,
    OFF_SIT,
    RISK_MULTS,
    _battery,
    _entries_spread_aware,
    _i0_for_days,
    _score,
    _simulate,
)
from analysis.xm_cost_wired_battery import (  # noqa: E402
    _calendar,
    _load,
    _sides_for_brain,
    _summarize,
    _window_trades,
)
from config import symbol_entry_brain, symbol_rr, symbol_scan_tfs  # noqa: E402

LOSER = ["BTCUSD", "AUDUSD", "USDCHF", "ETHUSD"]


def main() -> int:
    for sym in LOSER:
        live = symbol_entry_brain(sym)
        rr = float(symbol_rr(sym) or 2)
        tf = (symbol_scan_tfs(sym) or ["5m"])[0]
        print(f"\n=== {sym} live={live} tf={tf} RR={rr} (n>=20) ===", flush=True)
        df = _load(sym, tf)
        cal = _calendar(df)
        i0 = _i0_for_days(df, 365)
        i1 = len(df)
        sides_map = {b: _sides_for_brain(sym, b, df, cal) for b in BRAINS}
        ranked = []
        for brain, rmult, max_spr, off in product(BRAINS, RISK_MULTS, MAX_SPR, OFF_SIT):
            entries = _entries_spread_aware(
                df, sides_map[brain], sym, risk_mult=float(rmult)
            )
            trades = _simulate(
                df, entries, rr, sym, max_spread_r=max_spr, off_sit=bool(off)
            )
            w1 = _summarize(_window_trades(df, trades, 365))
            if w1["n"] < 20:
                continue
            bat = _battery(_window_trades(df, trades, 365), i0, i1)
            cell = {
                "brain": brain,
                "risk_mult": rmult,
                "max_spread_r": max_spr,
                "off_sitout": off,
                "windows": {"1y": w1, "3y": _summarize(trades)},
                "battery_1y": bat,
            }
            ranked.append(cell)
        ranked.sort(key=_score, reverse=True)
        if not ranked:
            print("  none", flush=True)
            continue
        for c in ranked[:8]:
            w = c["windows"]["1y"]
            bat = c["battery_1y"]
            print(
                f"  {c['brain']:<20} mult={c['risk_mult']:<4} "
                f"maxSp={str(c['max_spread_r']):<5} off={str(c['off_sitout']):<5} "
                f"{bat['verdict']:<5} wf={bat['wf_green']}/4 "
                f"1y ${w['pnl']:>9,.0f} E[R]={w['exp_r']:+.3f} "
                f"n={w['n']} spR={w['avg_spread_r']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
