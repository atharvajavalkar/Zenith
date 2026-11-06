#!/usr/bin/env python3
"""RESEARCH — Full battery matching current live TF policy. No config writes.

  GOLD/USDJPY: fib on 1h proxy (15m/1h live)
  BTC/ETH: 1h proxy (MTF live)
  USDCHF: pure 5m
  NAS100/UKOIL/AUDUSD: HTF entry + 5m exit
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import (  # noqa: E402
    BOOK,
    BRAINS,
    RISK_USD,
    YF,
    _fetch_1h,
    _prep,
    _stats,
    monte_carlo,
    verdict,
)
from analysis._rr_best_by_pair import collect_entries, simulate  # noqa: E402
from analysis._htf_trend_5m_exit_bt import (  # noqa: E402
    _fetch_5m,
    simulate_htf_entry_5m_exit,
)
from analysis.levy_malliavin import levy_malliavin_pack  # noqa: E402
from analysis.stochastic_pack import stochastic_pack  # noqa: E402
from config import SYMBOL_RR, symbol_rr  # noqa: E402

OUT = Path("/tmp/live_policy_full_battery.json")

# Modes matching live SYMBOL_SCAN_TFS / HTF_ENTRY_5M_EXIT
PURE_5M = {"USDCHF"}
HTF_5M_EXIT = {"NAS100", "UKOIL", "AUDUSD"}


def _clip(df, start: str, end: str, warm_days: int = 0):
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) - timedelta(
        days=warm_days
    )
    end_dt = datetime.fromisoformat(end).replace(hour=23, minute=59, tzinfo=timezone.utc)
    return df[(df.index >= start_dt) & (df.index <= end_dt)]


def _battery_from_trades(symbol, df_for_ito, full_tr, n_htf, split, rr, mode):
    def slice_tr(start_i=0, end_i=None):
        end_i = end_i if end_i is not None else 10**9
        return [t for t in full_tr if start_i <= int(t["i"]) <= end_i]

    # Re-sim for folds when we have a gen; for pure lists filter by i
    full = _stats(full_tr)
    oos = _stats(slice_tr(start_i=split))
    is_st = _stats(slice_tr(end_i=split - 1))
    wf = []
    for k in range(4):
        test_start = int(n_htf * (0.40 + 0.15 * k))
        test_end = int(n_htf * (0.55 + 0.15 * k)) - 1
        if test_end <= test_start or test_start < 80:
            continue
        wf.append(
            {
                "fold": k + 1,
                "train": _stats(slice_tr(end_i=test_start - 1)),
                "oos": _stats(slice_tr(start_i=test_start, end_i=test_end)),
            }
        )
    torture = {
        "spread_0.15R": _stats(
            [
                {
                    **t,
                    "r": round(t["r"] - 0.15, 4),
                    "pnl": round((t["r"] - 0.15) * RISK_USD, 2),
                }
                for t in full_tr
            ]
        )
    }
    mc = monte_carlo(full_tr, paths=1200)
    lm = levy_malliavin_pack(full_tr)
    stoch = stochastic_pack(df_for_ito, full_tr)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
    return {
        "symbol": symbol,
        "mode": mode,
        "brain": BRAINS.get(symbol, "?"),
        "rr": rr,
        "rr_label": f"1:{rr:g}",
        "full": full,
        "oos": oos,
        "in_sample": is_st,
        "walk_forward": wf,
        "wf_green": f"{wf_g}/{len(wf)}",
        "monte_carlo": mc,
        "levy": lm["levy"],
        "malliavin": lm["malliavin"],
        "brownian": stoch["brownian"],
        "ito": stoch["ito"],
        "martingale": stoch["martingale"],
        "hjm": stoch["hjm"],
        "verdict": v,
        "pnl_usd": full.get("pnl"),
        "oos_pnl_usd": oos.get("pnl"),
    }


def run_pair(symbol: str, start_htf: str, start_5: str, end_s: str) -> dict:
    rr = float(symbol_rr(symbol, default=2.0))
    if symbol in PURE_5M:
        raw = _fetch_5m(symbol, start_5, end_s)
        df = _prep(_clip(raw, start_5, end_s))
        if len(df) < 300:
            return {"symbol": symbol, "error": f"thin5m:{len(df)}", "verdict": "DEAD"}
        entries = collect_entries(symbol, df)
        n = len(df)
        split = int(n * 0.70)
        full_tr = simulate(df, entries, rr)
        print(f"\n{symbol} mode=pure_5m RR=1:{rr:g} bars={n} entries={len(entries)}", flush=True)
        row = _battery_from_trades(symbol, df, full_tr, n, split, rr, "pure_5m")
    elif symbol in HTF_5M_EXIT:
        df_h = _prep(_clip(_fetch_1h(symbol, start_htf, end_s), start_5, end_s, warm_days=20))
        df_5 = _prep(_clip(_fetch_5m(symbol, start_5, end_s), start_5, end_s))
        if len(df_h) < 200 or len(df_5) < 300:
            return {
                "symbol": symbol,
                "error": f"thin h={len(df_h)} 5={len(df_5)}",
                "verdict": "DEAD",
            }
        entries = collect_entries(symbol, df_h)
        t0, t1 = df_5.index[0], df_5.index[-1]
        entries = [e for e in entries if t0 <= df_h.index[int(e["ei"])] <= t1]
        n = len(df_h)
        split = int(n * 0.70)
        full_tr = simulate_htf_entry_5m_exit(df_h, df_5, entries, rr)
        print(
            f"\n{symbol} mode=htf+5m_exit RR=1:{rr:g} htf={n} 5m={len(df_5)} entries={len(entries)}",
            flush=True,
        )
        row = _battery_from_trades(symbol, df_5, full_tr, n, split, rr, "htf_5m_exit")
    else:
        df = _prep(_clip(_fetch_1h(symbol, start_htf, end_s), start_htf, end_s))
        if len(df) < 200:
            return {"symbol": symbol, "error": f"thin1h:{len(df)}", "verdict": "DEAD"}
        entries = collect_entries(symbol, df)
        n = len(df)
        split = int(n * 0.70)
        full_tr = simulate(df, entries, rr)
        print(f"\n{symbol} mode=htf_1h RR=1:{rr:g} bars={n} entries={len(entries)}", flush=True)
        row = _battery_from_trades(symbol, df, full_tr, n, split, rr, "htf_1h")

    f, o = row["full"], row["oos"]
    print(
        f"  n={f['n']} E[R]={f['exp_r']} pnl=${f['pnl']} | OOS E[R]={o['exp_r']} pnl=${o['pnl']} | "
        f"WF {row['wf_green']} | MC={row['monte_carlo'].get('prob_profit')}% "
        f"Lévy={row['levy'].get('prob_profit')}% BM={row['brownian'].get('prob_profit')}% | "
        f"Itô={row['ito'].get('verdict')} Mart={row['martingale'].get('verdict')} "
        f"HJM={row['hjm'].get('verdict')} Mall={row['malliavin'].get('fragility')} | {row['verdict']}",
        flush=True,
    )
    return row


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_htf = (end - timedelta(days=182)).isoformat()
    start_5 = (end - timedelta(days=55)).isoformat()
    end_s = end.isoformat()
    print(
        f"LIVE-POLICY battery {end_s} · HTF={start_htf} · 5m={start_5} · "
        f"${RISK_USD}/R · RR={SYMBOL_RR}",
        flush=True,
    )
    results = []
    for sym in BOOK:
        try:
            results.append(run_pair(sym, start_htf, start_5, end_s))
        except Exception as e:
            print(f"{sym}: FAIL {e}", flush=True)
            results.append({"symbol": sym, "error": str(e), "verdict": "DEAD", "pnl_usd": 0})

    ok = [r for r in results if not r.get("error")]
    total_pnl = sum(float(r.get("pnl_usd") or 0) for r in ok)
    total_oos = sum(float(r.get("oos_pnl_usd") or 0) for r in ok)
    total_n = sum(int((r.get("full") or {}).get("n") or 0) for r in ok)

    payload = {
        "window": {"htf_start": start_htf, "m5_start": start_5, "end": end_s},
        "policy": {
            "pure_5m": sorted(PURE_5M),
            "htf_5m_exit": sorted(HTF_5M_EXIT),
            "htf_1h": [s for s in BOOK if s not in PURE_5M and s not in HTF_5M_EXIT],
        },
        "symbol_rr": SYMBOL_RR,
        "results": results,
        "totals": {
            "trades": total_n,
            "pnl_usd": round(total_pnl, 2),
            "oos_pnl_usd": round(total_oos, 2),
            "survives": sum(1 for r in ok if r.get("verdict") == "SURVIVES"),
            "fragile": sum(1 for r in ok if r.get("verdict") == "FRAGILE"),
            "dead": sum(1 for r in results if r.get("verdict") == "DEAD"),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### LIVE POLICY — FULL TESTS\n", flush=True)
    print(
        "| Pair | Mode | RR | n | Full E[R] | Full PnL | OOS PnL | WF | MC% | Lévy% | BM% | Itô | Mart | HJM | Mall | Verdict |",
        flush=True,
    )
    print(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
        flush=True,
    )
    for r in results:
        if r.get("error"):
            print(f"| {r['symbol']} | — | — | — | — | — | — | — | — | — | — | — | — | — | — | DEAD |", flush=True)
            continue
        f = r["full"]
        print(
            f"| {r['symbol']} | {r['mode']} | {r['rr_label']} | {f['n']} | {f['exp_r']} | "
            f"${f['pnl']:+.0f} | ${r['oos']['pnl']:+.0f} | {r['wf_green']} | "
            f"{r['monte_carlo'].get('prob_profit')} | {r['levy'].get('prob_profit')} | "
            f"{r['brownian'].get('prob_profit')} | {r['ito'].get('verdict')} | "
            f"{r['martingale'].get('verdict')} | {r['hjm'].get('verdict')} | "
            f"{r['malliavin'].get('fragility')} | **{r['verdict']}** |",
            flush=True,
        )
    print("\n### TOTALS\n", flush=True)
    print(f"| Trades | {total_n} |", flush=True)
    print(f"| **Full PnL** | **${total_pnl:+.2f}** |", flush=True)
    print(f"| **OOS PnL** | **${total_oos:+.2f}** |", flush=True)
    print(
        f"| Verdicts | S={payload['totals']['survives']} F={payload['totals']['fragile']} D={payload['totals']['dead']} |",
        flush=True,
    )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
