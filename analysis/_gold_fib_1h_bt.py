#!/usr/bin/env python3
"""GOLD fib 0.50–0.618 — 15m vs 1h OOS / walk-forward / Monte Carlo.

Uses same fib+liq logic as live (0.5 tap in the pocket). Report only.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import (  # noqa: E402
    MC_PATHS,
    RISK_USD,
    _fetch_1h,
    _prep,
    _stats,
    monte_carlo,
    run_fib_trades,
    verdict,
)

OUT = Path("/tmp/gold_fib_15m_vs_1h.json")
FIBS = (0.50, 0.618)


def battery(label: str, df, *, fibs=FIBS) -> dict:
    n = len(df)
    print(f"  {label}: bars={n} fibs={fibs}", flush=True)

    def gen(start_i=0, end_i=None, **kw):
        use_fibs = kw.pop("fibs", fibs)
        return run_fib_trades(df, start_i=start_i, end_i=end_i, fibs=use_fibs, **kw)

    full_tr = gen()
    full = _stats(full_tr)
    split = int(n * 0.70)
    oos = _stats(gen(start_i=split))
    is_st = _stats(gen(end_i=split - 1))

    wf = []
    for k in range(4):
        test_start = int(n * (0.40 + 0.15 * k))
        test_end = int(n * (0.55 + 0.15 * k)) - 1
        if test_end <= test_start or test_start < 80:
            continue
        wf.append(
            {
                "fold": k + 1,
                "train": _stats(gen(end_i=test_start - 1)),
                "oos": _stats(gen(start_i=test_start, end_i=test_end)),
            }
        )

    torture = {
        "spread_0.15R": _stats(gen(spread_r=0.15)),
        "late_1bar": _stats(gen(late_bars=1)),
        "skip_30pct": _stats(gen(skip_p=0.30, rng=random.Random(11))),
        "fib_50_only": _stats(gen(fibs=(0.50,))),
    }
    mc = monte_carlo(full_tr, paths=MC_PATHS)
    v = verdict(full, oos, wf, torture, mc)
    print(
        f"    full E[R]={full['exp_r']} n={full['n']} | "
        f"OOS E[R]={oos['exp_r']} n={oos['n']} | "
        f"MC P(profit)={mc.get('prob_profit')}% | {v}",
        flush=True,
    )
    return {
        "label": label,
        "fibs": list(fibs),
        "bars": n,
        "full": full,
        "in_sample": is_st,
        "oos": oos,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "verdict": v,
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"GOLD fib 0.50–0.618 battery {start}→{end_s} ($200/R · yfinance 1h proxy)",
        flush=True,
    )
    print(
        "Note: 1h proxy = native 1h fib. 15m fib approximated on same 1h series "
        "(coarser than live 15m) — compare directionally.",
        flush=True,
    )

    raw = _fetch_1h("GOLD", start, end_s)
    df = _prep(raw)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_s).replace(hour=23, minute=59, tzinfo=timezone.utc)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    if len(df) < 200:
        print(f"FAIL: only {len(df)} bars", flush=True)
        return 1

    # 1h native fib pocket (0.50–0.618) + 0.50-only tap stress
    r_1h = battery("GOLD_fib_1h_50_618", df, fibs=FIBS)
    r_50 = battery("GOLD_fib_1h_tap50_only", df, fibs=(0.50,))

    results = [r_1h, r_50]
    payload = {
        "window": {"start": start, "end": end_s, "tf": "1h_yfinance"},
        "risk_usd": RISK_USD,
        "mc_paths": MC_PATHS,
        "live_wire": "FIB_EXEC_TF=15m,1h · GOLD_FIB_LEVELS=0.50,0.618",
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### GOLD FIB — OOS / WF / MC\n", flush=True)
    print(
        "| Variant | Full E[R] | OOS E[R] | n(full/oos) | WF green | MC P(profit) | Verdict |",
        flush=True,
    )
    print("|---|---:|---:|---:|---:|---:|---|", flush=True)
    for r in results:
        wf_g = sum(1 for f in r["walk_forward"] if f["oos"]["exp_r"] > 0)
        wf_n = len(r["walk_forward"])
        mc = r["monte_carlo"]
        print(
            f"| {r['label']} | {r['full']['exp_r']} | {r['oos']['exp_r']} | "
            f"{r['full']['n']}/{r['oos']['n']} | {wf_g}/{wf_n} | "
            f"{mc.get('prob_profit')}% | **{r['verdict']}** |",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    print(
        "\nLive: GOLD fib now allowed on 15m AND 1h (0.50–0.618 pocket).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
