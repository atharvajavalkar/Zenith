#!/usr/bin/env python3
"""RESEARCH — GOLD style bake-off vs current live fib setup.

Arms (same 1h GOLD window, fixed RR=4 = live SYMBOL_RR):
  1. fib_current     — live fib_liq_sweep proxy (control)
  2. price_action    — pin / engulf at swing + EMA trend
  3. liquidity_sweep — equal-H/L or prior swing grab → reclaim (no fib)
  4. trendline       — pivot trendline break + close confirm
  5. breakout        — Donchian 20 (UKOIL brain)

Full battery: OOS / WF / MC / Lévy / Malliavin / BM / Itô / Martingale / HJM.
No live wiring.
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
    RISK_USD,
    SWING,
    _fetch_1h,
    _prep,
    _stats,
    _swing_high,
    _swing_low,
    monte_carlo,
    verdict,
)
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_breakout,
    _collect_fib,
    simulate,
)
from analysis.levy_malliavin import levy_malliavin_pack  # noqa: E402
from analysis.stochastic_pack import stochastic_pack  # noqa: E402
from config import symbol_rr  # noqa: E402

OUT = Path("/tmp/gold_style_compare.json")
RR = float(symbol_rr("GOLD", default=4.0))
EQUAL_TOL_ATR = 0.20
SL_PAD = 0.08


def _collect_price_action(df: pd.DataFrame) -> list[dict]:
    """Pin bar or engulfing at a confirmed swing, with EMA20/50 trend."""
    entries: list[dict] = []
    for i in range(60, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        atr = float(row["atr"] or 0) or 1e-6
        o, h, l, c = (
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        )
        po, ph, pl, pc = (
            float(prev["open"]),
            float(prev["high"]),
            float(prev["low"]),
            float(prev["close"]),
        )
        ema20 = float(row["ema20"])
        ema50 = float(row["ema50"])
        body = abs(c - o)
        rng = max(h - l, 1e-12)
        upper = h - max(c, o)
        lower = min(c, o) - l

        bull_pin = lower >= 2.0 * body and lower >= 0.55 * rng and c > o
        bear_pin = upper >= 2.0 * body and upper >= 0.55 * rng and c < o
        bull_eng = c > o and pc < po and c >= po and o <= pc and body >= 0.35 * atr
        bear_eng = c < o and pc > po and c <= po and o >= pc and body >= 0.35 * atr

        # swing context in last ~12 bars
        win = df.iloc[max(0, i - 12) : i + 1]
        near_lo = l <= float(win["low"].min()) + 0.15 * atr
        near_hi = h >= float(win["high"].max()) - 0.15 * atr

        signal = None
        if ema20 > ema50 and near_lo and (bull_pin or bull_eng):
            signal = "BUY"
            entry, sl = c, l - SL_PAD * atr
        elif ema20 < ema50 and near_hi and (bear_pin or bear_eng):
            signal = "SELL"
            entry, sl = c, h + SL_PAD * atr
        if not signal:
            continue
        if abs(entry - sl) <= 0.12 * atr:
            continue
        entries.append(
            {
                "ei": i,
                "side": signal,
                "entry": float(entry),
                "sl": float(sl),
                "entry_t": df.index[i],
            }
        )
    return entries


def _collect_liquidity_sweep(df: pd.DataFrame) -> list[dict]:
    """Pure liq grab: wick through prior swing / equal H-L, close reclaims."""
    entries: list[dict] = []
    # precompute swings
    swing_lows: list[tuple[int, float]] = []
    swing_highs: list[tuple[int, float]] = []
    for i in range(SWING, len(df) - SWING):
        if _swing_low(df, i):
            swing_lows.append((i, float(df.iloc[i]["low"])))
        if _swing_high(df, i):
            swing_highs.append((i, float(df.iloc[i]["high"])))

    for ei in range(40, len(df) - 1):
        row = df.iloc[ei]
        atr = float(row["atr"] or 0) or 1e-6
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        o = float(row["open"])
        # recent swing levels (not including current)
        recent_lo = [lv for si, lv in swing_lows if 5 <= ei - si <= 80]
        recent_hi = [lv for si, lv in swing_highs if 5 <= ei - si <= 80]
        if not recent_lo and not recent_hi:
            continue

        signal = None
        # BUY: sweep below a swing low (or equal lows cluster), close back above
        if recent_lo:
            liq = min(recent_lo)
            # equal lows: ≥2 levels within tol of the deepest
            equals = sum(1 for lv in recent_lo if abs(lv - liq) <= EQUAL_TOL_ATR * atr)
            if l < liq - 0.02 * atr and c > liq and c > o and equals >= 1:
                signal = "BUY"
                entry, sl = c, l - SL_PAD * atr
        # SELL: sweep above swing high
        if signal is None and recent_hi:
            liq = max(recent_hi)
            equals = sum(1 for lv in recent_hi if abs(lv - liq) <= EQUAL_TOL_ATR * atr)
            if h > liq + 0.02 * atr and c < liq and c < o and equals >= 1:
                signal = "SELL"
                entry, sl = c, h + SL_PAD * atr

        if not signal:
            continue
        if abs(entry - sl) <= 0.12 * atr:
            continue
        entries.append(
            {
                "ei": ei,
                "side": signal,
                "entry": float(entry),
                "sl": float(sl),
                "entry_t": df.index[ei],
            }
        )
    return entries


def _collect_trendline(df: pd.DataFrame) -> list[dict]:
    """Two-pivot trendline: break close + next-bar hold / retest entry."""
    entries: list[dict] = []
    lows: list[tuple[int, float]] = []
    highs: list[tuple[int, float]] = []
    for i in range(SWING, len(df) - SWING):
        if _swing_low(df, i):
            lows.append((i, float(df.iloc[i]["low"])))
        if _swing_high(df, i):
            highs.append((i, float(df.iloc[i]["high"])))

    for ei in range(80, len(df) - 1):
        row = df.iloc[ei]
        atr = float(row["atr"] or 0) or 1e-6
        c = float(row["close"])
        o = float(row["open"])
        h, l = float(row["high"]), float(row["low"])

        # ascending support from last 2 swing lows before ei
        lo_piv = [(si, lv) for si, lv in lows if si < ei - 2]
        hi_piv = [(si, lv) for si, lv in highs if si < ei - 2]
        signal = None

        if len(lo_piv) >= 2:
            (i1, y1), (i2, y2) = lo_piv[-2], lo_piv[-1]
            if i2 > i1 and y2 > y1:  # rising support
                slope = (y2 - y1) / (i2 - i1)
                line = y2 + slope * (ei - i2)
                # prior bar below/on line, this bar closes above → bullish break
                prev_c = float(df.iloc[ei - 1]["close"])
                prev_line = y2 + slope * (ei - 1 - i2)
                if prev_c <= prev_line + 0.05 * atr and c > line + 0.05 * atr and c > o:
                    signal = "BUY"
                    entry = c
                    sl = min(l, line) - SL_PAD * atr

        if signal is None and len(hi_piv) >= 2:
            (i1, y1), (i2, y2) = hi_piv[-2], hi_piv[-1]
            if i2 > i1 and y2 < y1:  # falling resistance
                slope = (y2 - y1) / (i2 - i1)
                line = y2 + slope * (ei - i2)
                prev_c = float(df.iloc[ei - 1]["close"])
                prev_line = y2 + slope * (ei - 1 - i2)
                if prev_c >= prev_line - 0.05 * atr and c < line - 0.05 * atr and c < o:
                    signal = "SELL"
                    entry = c
                    sl = max(h, line) + SL_PAD * atr

        if not signal:
            continue
        if abs(entry - sl) <= 0.15 * atr:
            continue
        # risk cap — avoid absurd trendline distance
        if abs(entry - sl) > 2.5 * atr:
            continue
        entries.append(
            {
                "ei": ei,
                "side": signal,
                "entry": float(entry),
                "sl": float(sl),
                "entry_t": df.index[ei],
            }
        )
    return entries


COLLECTORS = {
    "fib_current": ("live fib_liq (control)", _collect_fib),
    "price_action": ("pin/engulf + EMA trend", _collect_price_action),
    "liquidity_sweep": ("swing/equal H-L sweep reclaim", _collect_liquidity_sweep),
    "trendline": ("pivot TL break confirm", _collect_trendline),
    "breakout": ("Donchian 20", _collect_breakout),
}


def _battery(df: pd.DataFrame, full_tr: list[dict], name: str, label: str) -> dict:
    n = len(df)
    split = int(n * 0.70)

    def slice_tr(start_i=0, end_i=None):
        end_i = end_i if end_i is not None else 10**9
        return [t for t in full_tr if start_i <= int(t["i"]) <= end_i]

    full = _stats(full_tr)
    oos = _stats(slice_tr(start_i=split))
    wf = []
    for k in range(4):
        test_start = int(n * (0.40 + 0.15 * k))
        test_end = int(n * (0.55 + 0.15 * k)) - 1
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
    stoch = stochastic_pack(df, full_tr)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
    return {
        "arm": name,
        "label": label,
        "rr": RR,
        "rr_label": f"1:{RR:g}",
        "entries": len(full_tr),  # filled; overwritten below with raw count
        "full": full,
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
    start = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"GOLD style compare {start}→{end_s} · 1h · RR=1:{RR:g} · ${RISK_USD}/R",
        flush=True,
    )
    raw = _fetch_1h("GOLD", start, end_s)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_s).replace(
        hour=23, minute=59, tzinfo=timezone.utc
    )
    df = _prep(raw[(raw.index >= start_dt) & (raw.index <= end_dt)])
    print(f"bars={len(df)}", flush=True)

    results = []
    for name, (label, collector) in COLLECTORS.items():
        entries = collector(df)
        trades = simulate(df, entries, RR)
        print(
            f"\n{name} ({label}) entries={len(entries)} fills={len(trades)}",
            flush=True,
        )
        row = _battery(df, trades, name, label)
        row["raw_entries"] = len(entries)
        row["fills"] = len(trades)
        f, o = row["full"], row["oos"]
        print(
            f"  n={f['n']} E[R]={f['exp_r']} pnl=${f['pnl']} | "
            f"OOS E[R]={o['exp_r']} pnl=${o['pnl']} | WF {row['wf_green']} | "
            f"MC={row['monte_carlo'].get('prob_profit')}% "
            f"Lévy={row['levy'].get('prob_profit')}% "
            f"BM={row['brownian'].get('prob_profit')}% | "
            f"Itô={row['ito'].get('verdict')} Mart={row['martingale'].get('verdict')} "
            f"HJM={row['hjm'].get('verdict')} Mall={row['malliavin'].get('fragility')} | "
            f"{row['verdict']}",
            flush=True,
        )
        results.append(row)

    # Rank by OOS pnl then full E[R]
    ranked = sorted(
        results,
        key=lambda r: (
            float(r.get("oos_pnl_usd") or -1e9),
            float((r.get("full") or {}).get("exp_r") or -1e9),
        ),
        reverse=True,
    )
    control = next((r for r in results if r["arm"] == "fib_current"), None)

    payload = {
        "window": {"start": start, "end": end_s, "tf": "1h"},
        "rr": RR,
        "risk_usd": RISK_USD,
        "results": results,
        "rank_by_oos_pnl": [r["arm"] for r in ranked],
        "control": "fib_current",
        "beats_control": [
            r["arm"]
            for r in results
            if control
            and r["arm"] != "fib_current"
            and float(r.get("oos_pnl_usd") or -1e9)
            > float(control.get("oos_pnl_usd") or 0)
            and r.get("verdict") in ("SURVIVES", "FRAGILE")
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### GOLD STYLE BAKE-OFF vs CURRENT FIB\n", flush=True)
    print(
        "| Style | Entries | n | Full E[R] | Full PnL | OOS PnL | WF | MC% | Lévy% | BM% | Itô | Mart | HJM | Mall | Verdict |",
        flush=True,
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
        flush=True,
    )
    for r in results:
        f = r["full"]
        mark = " ← LIVE" if r["arm"] == "fib_current" else ""
        print(
            f"| {r['arm']}{mark} | {r['raw_entries']} | {f['n']} | {f['exp_r']} | "
            f"${f['pnl']:+.0f} | ${r['oos']['pnl']:+.0f} | {r['wf_green']} | "
            f"{r['monte_carlo'].get('prob_profit')} | {r['levy'].get('prob_profit')} | "
            f"{r['brownian'].get('prob_profit')} | {r['ito'].get('verdict')} | "
            f"{r['martingale'].get('verdict')} | {r['hjm'].get('verdict')} | "
            f"{r['malliavin'].get('fragility')} | **{r['verdict']}** |",
            flush=True,
        )
    print("\n### RANK (OOS PnL)\n", flush=True)
    for i, r in enumerate(ranked, 1):
        print(
            f"{i}. {r['arm']}: OOS ${r['oos_pnl_usd']:+.0f} · Full ${r['pnl_usd']:+.0f} · {r['verdict']}",
            flush=True,
        )
    print(f"\nBeats fib control on OOS: {payload['beats_control'] or 'none'}", flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
