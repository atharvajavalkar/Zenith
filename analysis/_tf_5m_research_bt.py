#!/usr/bin/env python3
"""RESEARCH ONLY — Unify analysis on 5m (same TF for all pairs). No live wiring.

Compares desk brains + current SYMBOL_RR on native 5m bars vs prior 1h proxy.
Full pack: OOS / WF / MC / Lévy / Malliavin / Brownian / Itô / Martingale / HJM.
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
    BOOK,
    BRAINS,
    RISK_USD,
    YF,
    _prep,
    _stats,
    monte_carlo,
    verdict,
)
from analysis._rr_best_by_pair import collect_entries, simulate  # noqa: E402
from analysis.levy_malliavin import levy_malliavin_pack  # noqa: E402
from analysis.stochastic_pack import stochastic_pack  # noqa: E402
from config import SYMBOL_RR, symbol_rr  # noqa: E402

OUT = Path("/tmp/tf_5m_research_battery.json")
# yfinance 5m ≈ 60d max — stay inside
LOOKBACK_DAYS = 55


def _fetch_5m(symbol: str, start: str, end: str):
    """Prefer historical.fetch_symbol_ohlcv (TV→YF); fallback yfinance."""
    try:
        from data.historical import fetch_symbol_ohlcv

        df = fetch_symbol_ohlcv(symbol, start, end, interval="5m")
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        print(f"  {symbol}: historical 5m fail ({exc}) — trying yfinance", flush=True)

    import yfinance as yf

    warm = (datetime.fromisoformat(start) - timedelta(days=2)).strftime("%Y-%m-%d")
    end_p = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    last = None
    for ticker in (YF.get(symbol, symbol), f"{symbol}=X", symbol):
        try:
            df = yf.Ticker(ticker).history(
                start=warm, end=end_p, interval="5m", auto_adjust=True
            )
            if df is not None and not df.empty:
                df = df.rename(columns=str.lower)
                df = df[[c for c in ("open", "high", "low", "close") if c in df.columns]].dropna()
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                return df
        except Exception as e:
            last = e
    raise RuntimeError(f"no 5m {symbol}: {last}")


def battery(symbol: str, df) -> dict:
    rr = float(symbol_rr(symbol, default=2.0))
    brain = BRAINS.get(symbol, "?")
    n = len(df)
    split = int(n * 0.70)
    print(f"\n{symbol} 5m brain={brain} RR=1:{rr:g} bars={n}", flush=True)
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
        wf.append(
            {
                "fold": k + 1,
                "train": _stats(simulate(df, entries, rr, end_i=test_start - 1)),
                "oos": _stats(
                    simulate(df, entries, rr, start_i=test_start, end_i=test_end)
                ),
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
    stoch = stochastic_pack(df, full_tr)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)

    print(
        f"  n={full['n']} E[R]={full['exp_r']} pnl=${full['pnl']} | "
        f"OOS E[R]={oos['exp_r']} | WF {wf_g}/{len(wf)} | "
        f"MC={mc.get('prob_profit')}% Lévy={lm['levy'].get('prob_profit')}% "
        f"BM={stoch['brownian'].get('prob_profit')}% | "
        f"Itô={stoch['ito'].get('verdict')} Mart={stoch['martingale'].get('verdict')} "
        f"HJM={stoch['hjm'].get('verdict')} Mall={lm['malliavin'].get('fragility')} | {v}",
        flush=True,
    )
    return {
        "symbol": symbol,
        "tf": "5m",
        "brain": brain,
        "rr": rr,
        "rr_label": f"1:{rr:g}",
        "bars": n,
        "entries": len(entries),
        "full": full,
        "in_sample": is_st,
        "oos": oos,
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


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=LOOKBACK_DAYS)).isoformat()
    end_s = end.isoformat()
    print(
        f"RESEARCH 5m unified TF {start}→{end_s} · $/{RISK_USD}/R · "
        f"SYMBOL_RR={SYMBOL_RR} · NO LIVE CHANGES",
        flush=True,
    )

    results = []
    for sym in BOOK:
        try:
            raw = _fetch_5m(sym, start, end_s)
            df = _prep(raw)
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if len(df) < 300:
                print(f"{sym}: thin 5m bars={len(df)}", flush=True)
                results.append(
                    {"symbol": sym, "error": f"thin:{len(df)}", "verdict": "DEAD", "pnl_usd": 0}
                )
                continue
            results.append(battery(sym, df))
        except Exception as e:
            print(f"{sym}: FAIL {e}", flush=True)
            results.append({"symbol": sym, "error": str(e), "verdict": "DEAD", "pnl_usd": 0})

    ok = [r for r in results if not r.get("error")]
    total_pnl = sum(float(r.get("pnl_usd") or 0) for r in ok)
    total_oos_pnl = sum(float(r.get("oos_pnl_usd") or 0) for r in ok)
    total_n = sum(int((r.get("full") or {}).get("n") or 0) for r in ok)

    payload = {
        "note": "Research only — live desk NOT changed. Unified 5m analysis TF.",
        "window": {"start": start, "end": end_s, "tf": "5m"},
        "risk_usd": RISK_USD,
        "symbol_rr": SYMBOL_RR,
        "results": results,
        "totals": {
            "pairs": len(ok),
            "trades": total_n,
            "pnl_usd": round(total_pnl, 2),
            "oos_pnl_usd": round(total_oos_pnl, 2),
            "survives": sum(1 for r in ok if r.get("verdict") == "SURVIVES"),
            "fragile": sum(1 for r in ok if r.get("verdict") == "FRAGILE"),
            "dead": sum(1 for r in results if r.get("verdict") == "DEAD"),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### 5m UNIFIED ANALYSIS — PER PAIR (research)\n", flush=True)
    print(
        "| Pair | RR | n | Full E[R] | Full PnL | OOS E[R] | OOS PnL | WF | MC% | Lévy% | BM% | Itô | Mart | HJM | Mall | Verdict |",
        flush=True,
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
        flush=True,
    )
    for r in results:
        if r.get("error"):
            print(
                f"| {r['symbol']} | — | — | — | — | — | — | — | — | — | — | — | — | — | — | DEAD |",
                flush=True,
            )
            continue
        f, o = r["full"], r["oos"]
        print(
            f"| {r['symbol']} | {r['rr_label']} | {f['n']} | {f['exp_r']} | "
            f"${f['pnl']:+.0f} | {o['exp_r']} | ${o['pnl']:+.0f} | {r['wf_green']} | "
            f"{r['monte_carlo'].get('prob_profit')} | {r['levy'].get('prob_profit')} | "
            f"{r['brownian'].get('prob_profit')} | {r['ito'].get('verdict')} | "
            f"{r['martingale'].get('verdict')} | {r['hjm'].get('verdict')} | "
            f"{r['malliavin'].get('fragility')} | **{r['verdict']}** |",
            flush=True,
        )

    print("\n### TOTALS (5m research · $200/R)\n", flush=True)
    print(f"| Metric | Value |", flush=True)
    print(f"|---|---|", flush=True)
    print(f"| Pairs tested | {payload['totals']['pairs']} |", flush=True)
    print(f"| Total trades | {total_n} |", flush=True)
    print(f"| **Full-sample PnL** | **${total_pnl:+.2f}** |", flush=True)
    print(f"| **OOS PnL** | **${total_oos_pnl:+.2f}** |", flush=True)
    print(
        f"| Verdicts | S={payload['totals']['survives']} "
        f"F={payload['totals']['fragile']} D={payload['totals']['dead']} |",
        flush=True,
    )
    print(f"\nWrote {OUT} — live config untouched.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
