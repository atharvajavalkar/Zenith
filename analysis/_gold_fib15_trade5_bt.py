#!/usr/bin/env python3
"""RESEARCH ONLY — GOLD fib on 15m, take trade on 5m. No live tweaks.

Arms (same GOLD window, RR=live SYMBOL_RR=4):
  1. fib_15m_trade_5m  — impulse+fib zone on 15m; sweep/reclaim entry + SL/TP on 5m
  2. fib_15m_native    — same fib logic entry+exit on 15m (control TF)
  3. fib_1h_current    — current 1h proxy (desk battery control)

Full pack: OOS / WF / MC / Lévy / Malliavin / BM / Itô / Martingale / HJM.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import (  # noqa: E402
    RISK_USD,
    _fetch_1h,
    _fib_zone,
    _impulse,
    _prep,
    _stats,
    monte_carlo,
    verdict,
)
from analysis._htf_trend_5m_exit_bt import _fetch_5m  # noqa: E402
from analysis._rr_best_by_pair import _collect_fib, simulate  # noqa: E402
from analysis.levy_malliavin import levy_malliavin_pack  # noqa: E402
from analysis.stochastic_pack import stochastic_pack  # noqa: E402
from config import symbol_rr  # noqa: E402

OUT = Path("/tmp/gold_fib15_trade5_battery.json")
RR = float(symbol_rr("GOLD", default=4.0))
FIBS = (0.50, 0.618)
MAX_IMP_AGE_H = 48.0


def _resample(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    m = {"5m": "5min", "15m": "15min", "1h": "1h"}[rule]
    o = df5["open"].resample(m).first()
    h = df5["high"].resample(m).max()
    l = df5["low"].resample(m).min()
    c = df5["close"].resample(m).last()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    return _prep(out)


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, tzinfo=timezone.utc
    )
    return df[(df.index >= start_dt) & (df.index <= end_dt)].copy()


def collect_fib_15m_trade_5m(df_15: pd.DataFrame, df_5: pd.DataFrame) -> list[dict]:
    """Fib impulse/zone from 15m; entry trigger on 5m bars into that zone."""
    cache: list[Any] = [None] * len(df_15)
    for i in range(40, len(df_15)):
        atr = float(df_15.iloc[i]["atr"] or 1)
        cache[i] = _impulse(df_15, i, atr)

    entries: list[dict] = []
    # Walk 5m; map each bar to latest closed 15m
    for ei in range(60, len(df_5) - 1):
        ts = df_5.index[ei]
        pos15 = int(df_15.index.searchsorted(ts, side="right")) - 1
        if pos15 < 40:
            continue
        imp = cache[pos15]
        if not imp:
            continue
        if (ts - imp["end_t"]).total_seconds() > MAX_IMP_AGE_H * 3600:
            continue

        row = df_5.iloc[ei]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        o = float(row["open"])
        atr5 = float(row["atr"] or 0) or max(h - l, 1e-6)
        # Zone sized with 15m ATR (structure scale)
        atr15 = float(df_15.iloc[pos15]["atr"] or atr5) or atr5
        zlo, zhi = _fib_zone(imp, atr15, FIBS)

        # Liquidity reference on recent 5m
        win = df_5.iloc[max(0, ei - 24) : ei]
        signal = None
        if imp["side"] == "BUY":
            liq = float(win["low"].min()) if len(win) else l
            if l < min(liq, zlo) and c > zlo and l <= zhi and c > o:
                signal = "BUY"
                entry, sl = c, min(l, liq) - 0.08 * atr5
        else:
            liq = float(win["high"].max()) if len(win) else h
            if h > max(liq, zhi) and c < zhi and h >= zlo and c < o:
                signal = "SELL"
                entry, sl = c, max(h, liq) + 0.08 * atr5
        if not signal:
            continue
        risk = abs(entry - sl)
        if risk <= 0.12 * atr5:
            continue
        entries.append(
            {
                "ei": ei,
                "side": signal,
                "entry": float(entry),
                "sl": float(sl),
                "entry_t": ts,
                "htf_i": pos15,
            }
        )
    return entries


def simulate_on_df(
    df: pd.DataFrame,
    entries: list[dict],
    rr: float,
    *,
    index_key: str = "ei",
) -> list[dict]:
    """One-at-a-time SL/TP on the given frame (5m or 15m)."""
    trades: list[dict] = []
    busy_until = -1
    for e in entries:
        ei = int(e[index_key] if index_key in e else e["ei"])
        # For 15m-control entries, ei is on that frame; for hybrid, ei is 5m
        if ei < busy_until:
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

        exit_r = None
        exit_j = ei
        for j in range(ei + 1, len(df)):
            row = df.iloc[j]
            h, l = float(row["high"]), float(row["low"])
            hit_sl = (side == "BUY" and l <= sl) or (side == "SELL" and h >= sl)
            hit_tp = (side == "BUY" and h >= tp) or (side == "SELL" and l <= tp)
            if hit_sl or hit_tp:
                exit_r = -1.0 if hit_sl else float(rr)
                if hit_sl and hit_tp:
                    exit_r = -1.0
                exit_j = j
                break
        if exit_r is None:
            continue
        trades.append(
            {
                "i": int(e.get("htf_i", ei)),  # fold on structure TF when present
                "ei5": ei,
                "entry_t": str(e["entry_t"])[:16],
                "side": side,
                "r": round(exit_r, 4),
                "pnl": round(exit_r * RISK_USD, 2),
                "hold_h": round(
                    (df.index[exit_j] - df.index[ei]).total_seconds() / 3600.0, 2
                ),
            }
        )
        busy_until = exit_j + 2
    return trades


def _battery(
    full_tr: list[dict],
    df_for_ito: pd.DataFrame,
    n_fold: int,
    arm: str,
    label: str,
) -> dict:
    split = int(n_fold * 0.70)

    def slice_tr(start_i=0, end_i=None):
        end_i = end_i if end_i is not None else 10**9
        return [t for t in full_tr if start_i <= int(t["i"]) <= end_i]

    full = _stats(full_tr)
    oos = _stats(slice_tr(start_i=split))
    wf = []
    for k in range(4):
        test_start = int(n_fold * (0.40 + 0.15 * k))
        test_end = int(n_fold * (0.55 + 0.15 * k)) - 1
        if test_end <= test_start or test_start < 40:
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
        "arm": arm,
        "label": label,
        "rr": RR,
        "full": full,
        "oos": oos,
        "wf_green": f"{wf_g}/{len(wf)}",
        "walk_forward": wf,
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
    # 5m yfinance ~60d; keep ~55d honest window
    start_5 = (end - timedelta(days=55)).isoformat()
    end_s = end.isoformat()
    # 1h control uses longer warm history for fair "current" reference
    start_1h = (end - timedelta(days=182)).isoformat()

    print(
        f"GOLD fib15→trade5 battery · 5m/15m window {start_5}→{end_s} · "
        f"1h control {start_1h}→{end_s} · RR=1:{RR:g} · ${RISK_USD}/R · NO LIVE TWEAK",
        flush=True,
    )

    raw5 = _fetch_5m("GOLD", start_5, end_s)
    df_5 = _resample(_clip(raw5, start_5, end_s), "5m")
    # Prefer resample from same 5m for alignment; also try native if thin
    df_15 = _resample(_clip(raw5, start_5, end_s), "15m")
    print(f"5m bars={len(df_5)}  15m bars={len(df_15)}", flush=True)

    results = []

    # --- Arm 1: fib 15m → trade 5m ---
    e_hyb = collect_fib_15m_trade_5m(df_15, df_5)
    tr_hyb = simulate_on_df(df_5, e_hyb, RR)
    print(
        f"\nfib_15m_trade_5m entries={len(e_hyb)} fills={len(tr_hyb)}",
        flush=True,
    )
    r1 = _battery(tr_hyb, df_5, len(df_15), "fib_15m_trade_5m", "fib on 15m, entry+exit on 5m")
    r1["raw_entries"] = len(e_hyb)
    r1["fills"] = len(tr_hyb)
    f, o = r1["full"], r1["oos"]
    print(
        f"  n={f['n']} E[R]={f['exp_r']} pnl=${f['pnl']} | OOS E[R]={o['exp_r']} pnl=${o['pnl']} | "
        f"WF {r1['wf_green']} | MC={r1['monte_carlo'].get('prob_profit')}% "
        f"Lévy={r1['levy'].get('prob_profit')}% | {r1['verdict']}",
        flush=True,
    )
    results.append(r1)

    # --- Arm 2: fib 15m native ---
    e_15 = _collect_fib(df_15, fibs=FIBS)
    tr_15 = simulate(df_15, e_15, RR)
    print(f"\nfib_15m_native entries={len(e_15)} fills={len(tr_15)}", flush=True)
    r2 = _battery(tr_15, df_15, len(df_15), "fib_15m_native", "fib entry+exit on 15m")
    r2["raw_entries"] = len(e_15)
    r2["fills"] = len(tr_15)
    f, o = r2["full"], r2["oos"]
    print(
        f"  n={f['n']} E[R]={f['exp_r']} pnl=${f['pnl']} | OOS E[R]={o['exp_r']} pnl=${o['pnl']} | "
        f"WF {r2['wf_green']} | MC={r2['monte_carlo'].get('prob_profit')}% "
        f"Lévy={r2['levy'].get('prob_profit')}% | {r2['verdict']}",
        flush=True,
    )
    results.append(r2)

    # --- Arm 3: current 1h fib proxy (same ~55d slice for apples-to-apples) ---
    raw1 = _fetch_1h("GOLD", start_5, end_s)
    df_1h = _prep(_clip(raw1, start_5, end_s))
    e_1h = _collect_fib(df_1h, fibs=FIBS)
    tr_1h = simulate(df_1h, e_1h, RR)
    print(
        f"\nfib_1h_current (same 55d window) entries={len(e_1h)} fills={len(tr_1h)} bars={len(df_1h)}",
        flush=True,
    )
    r3 = _battery(tr_1h, df_1h, len(df_1h), "fib_1h_current", "current 1h fib proxy (55d)")
    r3["raw_entries"] = len(e_1h)
    r3["fills"] = len(tr_1h)
    f, o = r3["full"], r3["oos"]
    print(
        f"  n={f['n']} E[R]={f['exp_r']} pnl=${f['pnl']} | OOS E[R]={o['exp_r']} pnl=${o['pnl']} | "
        f"WF {r3['wf_green']} | MC={r3['monte_carlo'].get('prob_profit')}% "
        f"Lévy={r3['levy'].get('prob_profit')}% | {r3['verdict']}",
        flush=True,
    )
    results.append(r3)

    # --- Arm 4: longer 1h control (desk-style ~182d) for context ---
    raw1L = _fetch_1h("GOLD", start_1h, end_s)
    df_1hL = _prep(_clip(raw1L, start_1h, end_s))
    e_1hL = _collect_fib(df_1hL, fibs=FIBS)
    tr_1hL = simulate(df_1hL, e_1hL, RR)
    print(
        f"\nfib_1h_long (182d desk proxy) entries={len(e_1hL)} fills={len(tr_1hL)} bars={len(df_1hL)}",
        flush=True,
    )
    r4 = _battery(
        tr_1hL, df_1hL, len(df_1hL), "fib_1h_long", "current 1h fib proxy (182d)"
    )
    r4["raw_entries"] = len(e_1hL)
    r4["fills"] = len(tr_1hL)
    f, o = r4["full"], r4["oos"]
    print(
        f"  n={f['n']} E[R]={f['exp_r']} pnl=${f['pnl']} | OOS E[R]={o['exp_r']} pnl=${o['pnl']} | "
        f"WF {r4['wf_green']} | MC={r4['monte_carlo'].get('prob_profit')}% "
        f"Lévy={r4['levy'].get('prob_profit')}% | {r4['verdict']}",
        flush=True,
    )
    results.append(r4)

    payload = {
        "note": "RESEARCH ONLY — no live config changes",
        "window_5m_15m": {"start": start_5, "end": end_s},
        "window_1h_long": {"start": start_1h, "end": end_s},
        "rr": RR,
        "risk_usd": RISK_USD,
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### GOLD: FIB 15m → TRADE 5m vs CURRENT\n", flush=True)
    print(
        "| Arm | Window | Entries | n | Full E[R] | Full PnL | OOS PnL | WF | MC% | Lévy% | BM% | Itô | Mart | HJM | Mall | Verdict |",
        flush=True,
    )
    print(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
        flush=True,
    )
    for r in results:
        win = "55d" if r["arm"] != "fib_1h_long" else "182d"
        f = r["full"]
        print(
            f"| {r['arm']} | {win} | {r.get('raw_entries')} | {f['n']} | {f['exp_r']} | "
            f"${f['pnl']:+.0f} | ${r['oos']['pnl']:+.0f} | {r['wf_green']} | "
            f"{r['monte_carlo'].get('prob_profit')} | {r['levy'].get('prob_profit')} | "
            f"{r['brownian'].get('prob_profit')} | {r['ito'].get('verdict')} | "
            f"{r['martingale'].get('verdict')} | {r['hjm'].get('verdict')} | "
            f"{r['malliavin'].get('fragility')} | **{r['verdict']}** |",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
