#!/usr/bin/env python3
"""RESEARCH ONLY — HTF trend entries + 5m TP/SL path. No live wiring.

Trend / entry: current brains on 1h (proxy for 15m/1h fib & MTF structure).
Exit management: walk SL/TP on native 5m bars after entry timestamp.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

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
from analysis._rr_best_by_pair import collect_entries  # noqa: E402
from analysis.levy_malliavin import levy_malliavin_pack  # noqa: E402
from analysis.stochastic_pack import stochastic_pack  # noqa: E402
from config import SYMBOL_RR, symbol_rr  # noqa: E402

OUT = Path("/tmp/htf_trend_5m_exit_battery.json")
LOOKBACK_5M = 55  # yfinance 5m cap


def _fetch_5m(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        from data.historical import fetch_symbol_ohlcv

        df = fetch_symbol_ohlcv(symbol, start, end, interval="5m")
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        print(f"  {symbol}: TV/hist 5m fail ({exc}) — yfinance", flush=True)

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
                df = df[
                    [c for c in ("open", "high", "low", "close") if c in df.columns]
                ].dropna()
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                return df
        except Exception as e:
            last = e
    raise RuntimeError(f"no 5m {symbol}: {last}")


def simulate_htf_entry_5m_exit(
    df_htf: pd.DataFrame,
    df_5m: pd.DataFrame,
    entries: list[dict],
    rr: float,
    *,
    start_i: int = 0,
    end_i: Optional[int] = None,
) -> list[dict]:
    """Entries from HTF indices; exits resolved on 5m OHLC path."""
    end_i = end_i if end_i is not None else len(df_htf) - 1
    trades: list[dict] = []
    busy_until_htf = -1

    # Precompute 5m search helpers
    t5 = df_5m.index

    for e in entries:
        ei = int(e["ei"])
        if ei < start_i or ei > end_i or ei < busy_until_htf:
            continue
        side = e["side"]
        entry = float(e["entry"])
        sl0 = float(e["sl"])
        risk = abs(entry - sl0)
        if risk <= 0:
            continue
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk

        entry_t = df_htf.index[ei]
        # First 5m bar at/after HTF entry close
        pos = int(t5.searchsorted(entry_t, side="left"))
        if pos >= len(df_5m) - 1:
            continue

        exit_r = None
        exit_j = None
        for j in range(pos + 1, len(df_5m)):
            row = df_5m.iloc[j]
            h, l = float(row["high"]), float(row["low"])
            hit_sl = (side == "BUY" and l <= sl) or (side == "SELL" and h >= sl)
            hit_tp = (side == "BUY" and h >= tp) or (side == "SELL" and l <= tp)
            if hit_sl or hit_tp:
                exit_r = -1.0 if hit_sl else float(rr)
                if hit_sl and hit_tp:
                    exit_r = -1.0  # conservative
                exit_j = j
                break
        if exit_r is None:
            continue

        trades.append(
            {
                "i": ei,
                "entry_t": str(entry_t)[:16],
                "side": side,
                "r": round(exit_r, 4),
                "pnl": round(exit_r * RISK_USD, 2),
                "hold_h": round(
                    (df_5m.index[exit_j] - entry_t).total_seconds() / 3600.0, 2
                ),
            }
        )
        # Cooldown on HTF: skip until HTF bar covering exit time
        exit_t = df_5m.index[exit_j]
        busy_until_htf = int(df_htf.index.searchsorted(exit_t, side="left")) + 2

    return trades


def battery(symbol: str, df_htf: pd.DataFrame, df_5m: pd.DataFrame) -> dict:
    rr = float(symbol_rr(symbol, default=2.0))
    brain = BRAINS.get(symbol, "?")
    n = len(df_htf)
    split = int(n * 0.70)
    print(
        f"\n{symbol} HTF-trend+5m-exit brain={brain} RR=1:{rr:g} "
        f"htf_bars={n} 5m_bars={len(df_5m)}",
        flush=True,
    )
    entries = collect_entries(symbol, df_htf)
    # Only entries that fall inside 5m coverage
    t5_min, t5_max = df_5m.index[0], df_5m.index[-1]
    entries = [
        e
        for e in entries
        if t5_min <= df_htf.index[int(e["ei"])] <= t5_max
    ]
    print(f"  HTF entries in 5m window={len(entries)}", flush=True)

    def gen(start_i=0, end_i=None):
        return simulate_htf_entry_5m_exit(
            df_htf, df_5m, entries, rr, start_i=start_i, end_i=end_i
        )

    full_tr = gen()
    full = _stats(full_tr)
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
    # Itô/price tests on 5m path (exit TF); martingale/HJM/BM on trade R
    stoch = stochastic_pack(df_5m, full_tr)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
    holds = [t.get("hold_h") for t in full_tr if t.get("hold_h") is not None]
    med_hold = float(np.median(holds)) if holds else None

    print(
        f"  n={full['n']} E[R]={full['exp_r']} pnl=${full['pnl']} | "
        f"OOS E[R]={oos['exp_r']} pnl=${oos['pnl']} | WF {wf_g}/{len(wf)} | "
        f"MC={mc.get('prob_profit')}% Lévy={lm['levy'].get('prob_profit')}% "
        f"BM={stoch['brownian'].get('prob_profit')}% | "
        f"Itô={stoch['ito'].get('verdict')} Mart={stoch['martingale'].get('verdict')} "
        f"HJM={stoch['hjm'].get('verdict')} Mall={lm['malliavin'].get('fragility')} | "
        f"med_hold={med_hold}h | {v}",
        flush=True,
    )
    return {
        "symbol": symbol,
        "mode": "htf_trend_5m_exit",
        "brain": brain,
        "rr": rr,
        "rr_label": f"1:{rr:g}",
        "htf_bars": n,
        "m5_bars": len(df_5m),
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
        "median_hold_h": med_hold,
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    # HTF for trend: ~6m; 5m exits only available ~55d — overlap window = 5m span
    start_5 = (end - timedelta(days=LOOKBACK_5M)).isoformat()
    start_htf = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"RESEARCH HTF-trend + 5m TP/SL · trend HTF from {start_htf} · "
        f"5m exits {start_5}→{end_s} · ${RISK_USD}/R · NO LIVE CHANGES",
        flush=True,
    )
    print(f"SYMBOL_RR={SYMBOL_RR}", flush=True)

    results = []
    for sym in BOOK:
        try:
            raw_h = _fetch_1h(sym, start_htf, end_s)
            df_h = _prep(raw_h)
            raw_5 = _fetch_5m(sym, start_5, end_s)
            df_5 = _prep(raw_5)
            start_dt = datetime.fromisoformat(start_5).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            # Keep HTF covering 5m window (+ warmup before)
            warm = start_dt - timedelta(days=20)
            df_h = df_h[(df_h.index >= warm) & (df_h.index <= end_dt)]
            df_5 = df_5[(df_5.index >= start_dt) & (df_5.index <= end_dt)]
            if len(df_h) < 200 or len(df_5) < 300:
                print(
                    f"{sym}: thin htf={len(df_h)} 5m={len(df_5)}",
                    flush=True,
                )
                results.append(
                    {
                        "symbol": sym,
                        "error": f"thin htf={len(df_h)} 5m={len(df_5)}",
                        "verdict": "DEAD",
                        "pnl_usd": 0,
                        "oos_pnl_usd": 0,
                    }
                )
                continue
            results.append(battery(sym, df_h, df_5))
        except Exception as e:
            print(f"{sym}: FAIL {e}", flush=True)
            results.append(
                {
                    "symbol": sym,
                    "error": str(e),
                    "verdict": "DEAD",
                    "pnl_usd": 0,
                    "oos_pnl_usd": 0,
                }
            )

    ok = [r for r in results if not r.get("error")]
    total_pnl = sum(float(r.get("pnl_usd") or 0) for r in ok)
    total_oos = sum(float(r.get("oos_pnl_usd") or 0) for r in ok)
    total_n = sum(int((r.get("full") or {}).get("n") or 0) for r in ok)

    payload = {
        "note": (
            "Research only. Trend/entry on HTF (1h proxy for current brains); "
            "TP/SL filled on 5m path. Live desk NOT changed."
        ),
        "window": {
            "htf_start": start_htf,
            "m5_start": start_5,
            "end": end_s,
        },
        "risk_usd": RISK_USD,
        "symbol_rr": SYMBOL_RR,
        "results": results,
        "totals": {
            "pairs": len(ok),
            "trades": total_n,
            "pnl_usd": round(total_pnl, 2),
            "oos_pnl_usd": round(total_oos, 2),
            "survives": sum(1 for r in ok if r.get("verdict") == "SURVIVES"),
            "fragile": sum(1 for r in ok if r.get("verdict") == "FRAGILE"),
            "dead": sum(1 for r in results if r.get("verdict") == "DEAD"),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### HTF TREND + 5m TP/SL — PER PAIR\n", flush=True)
    print(
        "| Pair | RR | n | Full E[R] | Full PnL | OOS E[R] | OOS PnL | WF | MC% | Lévy% | BM% | Itô | Mart | HJM | Mall | med hold | Verdict |",
        flush=True,
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---|",
        flush=True,
    )
    for r in results:
        if r.get("error"):
            print(
                f"| {r['symbol']} | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | DEAD |",
                flush=True,
            )
            continue
        f, o = r["full"], r["oos"]
        mh = r.get("median_hold_h")
        mh_s = f"{mh:.1f}h" if mh is not None else "—"
        print(
            f"| {r['symbol']} | {r['rr_label']} | {f['n']} | {f['exp_r']} | "
            f"${f['pnl']:+.0f} | {o['exp_r']} | ${o['pnl']:+.0f} | {r['wf_green']} | "
            f"{r['monte_carlo'].get('prob_profit')} | {r['levy'].get('prob_profit')} | "
            f"{r['brownian'].get('prob_profit')} | {r['ito'].get('verdict')} | "
            f"{r['martingale'].get('verdict')} | {r['hjm'].get('verdict')} | "
            f"{r['malliavin'].get('fragility')} | {mh_s} | **{r['verdict']}** |",
            flush=True,
        )

    print("\n### TOTALS\n", flush=True)
    print("| Metric | Value |", flush=True)
    print("|---|---|", flush=True)
    print(f"| Pairs | {payload['totals']['pairs']} |", flush=True)
    print(f"| Trades | {total_n} |", flush=True)
    print(f"| **Full PnL** | **${total_pnl:+.2f}** |", flush=True)
    print(f"| **OOS PnL** | **${total_oos:+.2f}** |", flush=True)
    print(
        f"| Verdicts | S={payload['totals']['survives']} "
        f"F={payload['totals']['fragile']} D={payload['totals']['dead']} |",
        flush=True,
    )
    print(f"\nWrote {OUT} — live untouched.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
