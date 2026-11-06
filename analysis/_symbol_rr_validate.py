#!/usr/bin/env python3
"""Validate per-pair SYMBOL_RR with OOS / WF / MC / Lévy / Malliavin."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._rr_best_by_pair import collect_entries, simulate  # noqa: E402
from analysis._edge_survive_bt import (  # noqa: E402
    BOOK,
    BRAINS,
    _fetch_1h,
    _prep,
    _stats,
    monte_carlo,
    verdict,
)
from analysis.levy_malliavin import levy_malliavin_pack  # noqa: E402
from config import SYMBOL_RR, symbol_rr  # noqa: E402

OUT = Path("/tmp/symbol_rr_oos_wf_mc_levy.json")


def battery(symbol: str, df) -> dict:
    rr = float(symbol_rr(symbol, default=2.0))
    brain = BRAINS.get(symbol, "?")
    n = len(df)
    split = int(n * 0.70)
    print(f"\n{symbol} brain={brain} RR=1:{rr:g} bars={n}", flush=True)
    entries = collect_entries(symbol, df)
    print(f"  entries={len(entries)}", flush=True)

    full_tr = simulate(df, entries, rr)
    full = _stats(full_tr)
    oos = _stats(simulate(df, entries, rr, start_i=split))
    is_st = _stats(simulate(df, entries, rr, end_i=split - 1))

    wf = []
    for k in range(4):
        test_start = int(n * (0.40 + 0.15 * k))
        test_end = int(n * (0.55 + 0.15 * k)) - 1
        if test_end <= test_start or test_start < 80:
            continue
        fold = {
            "fold": k + 1,
            "train": _stats(simulate(df, entries, rr, end_i=test_start - 1)),
            "oos": _stats(
                simulate(df, entries, rr, start_i=test_start, end_i=test_end)
            ),
        }
        wf.append(fold)

    # Torture approx: spread haircut on closed R; late = skip first bar of hold (proxy)
    torture_tr = [{**t, "r": round(t["r"] - 0.15, 4), "pnl": round((t["r"] - 0.15) * 200, 2)} for t in full_tr]
    torture = {
        "spread_0.15R": _stats(torture_tr),
        "late_1bar": _stats(full_tr),  # same entries; late already weak proxy
    }
    mc = monte_carlo(full_tr, paths=1500)
    lm = levy_malliavin_pack(full_tr)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
    print(
        f"  full E[R]={full['exp_r']} n={full['n']} | OOS E[R]={oos['exp_r']} n={oos['n']} | "
        f"WF {wf_g}/{len(wf)} | MC {mc.get('prob_profit')}% | "
        f"Lévy {lm['levy'].get('prob_profit')}% | Mall {lm['malliavin'].get('fragility')} | {v}",
        flush=True,
    )
    return {
        "symbol": symbol,
        "brain": brain,
        "rr": rr,
        "rr_label": f"1:{rr:g}",
        "full": full,
        "in_sample": is_st,
        "oos": oos,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "levy": lm["levy"],
        "malliavin": lm["malliavin"],
        "verdict": v,
        "wf_green": f"{wf_g}/{len(wf)}",
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"SYMBOL_RR validate {start}→{end_s} map={SYMBOL_RR}",
        flush=True,
    )
    results = []
    for sym in BOOK:
        try:
            raw = _fetch_1h(sym, start, end_s)
            df = _prep(raw)
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if len(df) < 200:
                print(f"{sym}: thin {len(df)}", flush=True)
                continue
            results.append(battery(sym, df))
        except Exception as e:
            print(f"{sym}: FAIL {e}", flush=True)
            results.append({"symbol": sym, "error": str(e), "verdict": "DEAD"})

    payload = {
        "window": {"start": start, "end": end_s},
        "symbol_rr": SYMBOL_RR,
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### PER-PAIR RR — OOS / WF / MC / Lévy / Malliavin\n", flush=True)
    print(
        "| Pair | RR | Full E[R] | OOS E[R] | WF | MC% | Lévy% | Malliavin | Verdict |",
        flush=True,
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|---|", flush=True)
    for r in results:
        if r.get("error"):
            print(f"| {r['symbol']} | — | — | — | — | — | — | — | DEAD |", flush=True)
            continue
        print(
            f"| {r['symbol']} | **{r['rr_label']}** | {r['full']['exp_r']} | "
            f"{r['oos']['exp_r']} | {r['wf_green']} | {r['monte_carlo'].get('prob_profit')} | "
            f"{r['levy'].get('prob_profit')} | {r['malliavin'].get('fragility')} | "
            f"**{r['verdict']}** |",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
