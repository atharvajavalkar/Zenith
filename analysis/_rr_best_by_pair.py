#!/usr/bin/env python3
"""RESEARCH — Best fixed RR per desk pair (live brains, 1h proxy).

Caches entries once per symbol, then resimulates exits for RR ∈ {1.5,2,2.4,3,4}.
Score = OOS E[R] ×3 + full E[R] ×1.5 + MC P(profit).
"""
from __future__ import annotations

import json
import random
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
    SWING,
    _fetch_1h,
    _fib_zone,
    _impulse,
    _prep,
    _stats,
    monte_carlo,
)

OUT = Path("/tmp/rr_best_by_pair.json")
RRS = (1.5, 2.0, 2.4, 3.0, 4.0)


def _collect_fib(df: pd.DataFrame, fibs=(0.50, 0.618)) -> list[dict]:
    entries: list[dict] = []
    cache = [None] * len(df)
    for i in range(40, len(df)):
        atr = float(df.iloc[i]["atr"] or 1)
        cache[i] = _impulse(df, i, atr)
    for ei in range(40, len(df) - 1):
        ts = df.index[ei]
        row = df.iloc[ei]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        atr = float(row["atr"] or 0) or max(h - l, 1e-6)
        imp = cache[ei]
        if not imp or (ts - imp["end_t"]).total_seconds() > 48 * 3600:
            continue
        zlo, zhi = _fib_zone(imp, atr, fibs)
        win = df.iloc[max(0, ei - 20) : ei]
        signal = None
        if imp["side"] == "BUY":
            liq = float(win["low"].min()) if len(win) else l
            if l < min(liq, zlo) and c > zlo and l <= zhi and c > float(row["open"]):
                signal = "BUY"
                entry, sl = c, min(l, liq) - 0.08 * atr
        else:
            liq = float(win["high"].max()) if len(win) else h
            if h > max(liq, zhi) and c < zhi and h >= zlo and c < float(row["open"]):
                signal = "SELL"
                entry, sl = c, max(h, liq) + 0.08 * atr
        if not signal:
            continue
        risk = abs(entry - sl)
        if risk <= 0.12 * atr:
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


def _collect_ema(df: pd.DataFrame, *, pull_atr: float, disp_atr: float) -> list[dict]:
    from strategy.ema_pull_brain import evaluate_ema_2050_pull

    entries: list[dict] = []
    for i in range(60, len(df) - 1):
        hit = evaluate_ema_2050_pull(
            df.iloc[: i + 1], pull_atr=pull_atr, disp_atr=disp_atr
        )
        if not hit:
            continue
        entry = float(hit["entry"])
        sl = float(hit["sl"])
        if abs(entry - sl) <= 0:
            continue
        entries.append(
            {
                "ei": i,
                "side": hit["signal"],
                "entry": entry,
                "sl": sl,
                "entry_t": df.index[i],
            }
        )
    return entries


def _collect_sma(df: pd.DataFrame) -> list[dict]:
    from strategy.sma_cross_brain import evaluate_sma_2050_cross

    entries: list[dict] = []
    for i in range(60, len(df) - 1):
        hit = evaluate_sma_2050_cross(df.iloc[: i + 1])
        if not hit:
            continue
        entry = float(hit["entry"])
        sl = float(hit["sl"])
        if abs(entry - sl) <= 0:
            continue
        entries.append(
            {
                "ei": i,
                "side": hit["signal"],
                "entry": entry,
                "sl": sl,
                "entry_t": df.index[i],
            }
        )
    return entries


def _collect_breakout(df: pd.DataFrame, lookback: int = 20) -> list[dict]:
    from strategy.breakout_brain import evaluate_breakout

    entries: list[dict] = []
    for i in range(lookback + 10, len(df) - 1):
        hit = evaluate_breakout(df.iloc[: i + 1], lookback=lookback)
        if not hit:
            continue
        entry = float(hit["entry"])
        sl = float(hit["sl"])
        if abs(entry - sl) <= 0:
            continue
        entries.append(
            {
                "ei": i,
                "side": hit["signal"],
                "entry": entry,
                "sl": sl,
                "entry_t": df.index[i],
            }
        )
    return entries


def collect_entries(symbol: str, df: pd.DataFrame) -> list[dict]:
    brain = BRAINS.get(symbol.upper(), "ema_20_50_pull")
    if brain == "fib_liq":
        return _collect_fib(df)
    if brain == "sma_20_50_cross":
        return _collect_sma(df)
    if brain == "breakout_20":
        return _collect_breakout(df)
    pull = 0.75 if brain == "ema_20_50_tight" else 1.0
    disp = 0.70 if brain == "ema_20_50_tight" else 0.55
    return _collect_ema(df, pull_atr=pull, disp_atr=disp)


def simulate(
    df: pd.DataFrame,
    entries: list[dict],
    rr: float,
    *,
    start_i: int = 0,
    end_i: Optional[int] = None,
) -> list[dict]:
    """Replay cached entries with given RR; one position at a time."""
    end_i = end_i if end_i is not None else len(df) - 1
    trades: list[dict] = []
    busy_until = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < start_i or ei > end_i or ei < busy_until:
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
        exit_i = ei
        for j in range(ei + 1, min(end_i, len(df) - 1) + 1):
            row = df.iloc[j]
            h, l = float(row["high"]), float(row["low"])
            hit_sl = (side == "BUY" and l <= sl) or (side == "SELL" and h >= sl)
            hit_tp = (side == "BUY" and h >= tp) or (side == "SELL" and l <= tp)
            if hit_sl or hit_tp:
                exit_r = -1.0 if hit_sl else float(rr)
                # if both same bar, conservative: SL first
                if hit_sl and hit_tp:
                    exit_r = -1.0
                exit_i = j
                break
        if exit_r is None:
            continue
        trades.append(
            {
                "i": ei,
                "entry_t": str(e["entry_t"])[:16],
                "side": side,
                "r": round(exit_r, 4),
                "pnl": round(exit_r * RISK_USD, 2),
            }
        )
        busy_until = exit_i + 2
    return trades


def score_row(full: dict, oos: dict, mc: dict) -> float:
    if full.get("n", 0) < 4:
        return -999.0
    s = float(oos.get("exp_r") or 0) * 3.0
    s += float(full.get("exp_r") or 0) * 1.5
    s += (float(mc.get("prob_profit") or 0) / 100.0)
    if oos.get("n", 0) >= 3 and (oos.get("exp_r") or 0) > 0:
        s += 0.5
    if full.get("n", 0) < 8:
        s -= 0.3
    return round(s, 4)


def sweep_symbol(symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    brain = BRAINS.get(symbol.upper(), "?")
    n = len(df)
    split = int(n * 0.70)
    print(f"\n{symbol} brain={brain} bars={n} — collecting entries…", flush=True)
    entries = collect_entries(symbol, df)
    print(f"  cached entries={len(entries)}", flush=True)
    rows = []
    for rr in RRS:
        full_tr = simulate(df, entries, rr)
        oos_tr = simulate(df, entries, rr, start_i=split)
        full = _stats(full_tr)
        oos = _stats(oos_tr)
        mc = monte_carlo(full_tr, paths=1000)
        sc = score_row(full, oos, mc)
        rows.append(
            {
                "rr": rr,
                "label": f"1:{rr:g}",
                "full": full,
                "oos": oos,
                "monte_carlo": mc,
                "score": sc,
            }
        )
        print(
            f"  1:{rr:g}  full E[R]={full['exp_r']:+.3f} n={full['n']} wr={full['wr']}% | "
            f"OOS E[R]={oos['exp_r']:+.3f} n={oos['n']} | "
            f"MC {mc.get('prob_profit', 0)}% | score={sc}",
            flush=True,
        )
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    best = ranked[0]
    return {
        "symbol": symbol,
        "brain": brain,
        "bars": n,
        "entries": len(entries),
        "best_rr": best["rr"],
        "best_label": best["label"],
        "best_score": best["score"],
        "best_full_exp_r": best["full"]["exp_r"],
        "best_oos_exp_r": best["oos"]["exp_r"],
        "best_wr": best["full"]["wr"],
        "best_mc_prob_profit": best["monte_carlo"].get("prob_profit"),
        "sweep": rows,
        "ranked": [r["label"] for r in ranked],
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"RR sweep {start}→{end_s} · candidates {list(RRS)} · live brains · 1h yfinance",
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
                print(f"{sym}: skip thin bars={len(df)}", flush=True)
                continue
            results.append(sweep_symbol(sym, df))
        except Exception as e:
            print(f"{sym}: FAIL {e}", flush=True)

    payload = {
        "window": {"start": start, "end": end_s, "tf": "1h"},
        "rr_candidates": list(RRS),
        "results": results,
        "summary": [
            {
                "symbol": r["symbol"],
                "brain": r["brain"],
                "best_rr": r["best_label"],
                "full_E_R": r["best_full_exp_r"],
                "oos_E_R": r["best_oos_exp_r"],
                "wr": r["best_wr"],
                "mc_profit_pct": r["best_mc_prob_profit"],
                "ranked": r["ranked"],
            }
            for r in results
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### BEST RR BY PAIR\n", flush=True)
    print(
        "| Pair | Brain | Best RR | Full E[R] | OOS E[R] | WR% | MC P(profit) | Ranked |",
        flush=True,
    )
    print("|---|---|---:|---:|---:|---:|---:|---|", flush=True)
    for r in results:
        ranked = " > ".join(r["ranked"])
        print(
            f"| {r['symbol']} | {r['brain']} | **{r['best_label']}** | "
            f"{r['best_full_exp_r']} | {r['best_oos_exp_r']} | {r['best_wr']} | "
            f"{r['best_mc_prob_profit']}% | {ranked} |",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
