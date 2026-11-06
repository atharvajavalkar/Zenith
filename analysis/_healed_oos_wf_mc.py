#!/usr/bin/env python3
"""RESEARCH — OOS / Walk-forward / Monte Carlo / Kelly for the HEALED desk.

Uses live SYMBOL_ENTRY_BRAIN map + autopsy_brain_promoted EMA knobs
(+ soft-heal pull tighten proxy when soft overlays are active).

No live wiring changes — report only.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import (  # noqa: E402
    BOOK,
    BRAINS,
    MC_PATHS,
    RISK_USD,
    _fetch_1h,
    _prep,
    _stats,
    battery_for,
    monte_carlo,
    run_breakout_trades,
    run_ema_trades,
    run_fib_trades,
    run_sma_trades,
    verdict,
)

OUT = Path("/tmp/healed_oos_wf_mc.json")


def _healed_ema_kw(symbol: str, brain: str) -> dict[str, float]:
    """Baseline + promoted brain knobs; soft overlays → slightly tighter pull proxy."""
    if brain == "ema_20_50_tight":
        kw = {"pull_atr": 0.75, "disp_atr": 0.70}
    elif brain.startswith("ema_"):
        kw = {"pull_atr": 1.0, "disp_atr": 0.55}
    else:
        return {}
    try:
        from analysis.autopsy_brain import live_ema_knobs

        promo = live_ema_knobs(symbol, brain)
        if promo.get("pull_atr") is not None:
            kw["pull_atr"] = float(promo["pull_atr"])
        if promo.get("disp_atr") is not None:
            kw["disp_atr"] = float(promo["disp_atr"])
    except Exception:
        pass
    try:
        from analysis.autopsy_soft import overlays_for

        soft = overlays_for(symbol)
        if soft:
            # Soft MC raise → fewer/cleaner entries in paper (pull tighter)
            mc_d = float(soft.get("mc_floor_delta") or 0)
            if mc_d >= 2.0:
                kw["pull_atr"] = max(0.55, float(kw["pull_atr"]) - 0.08)
                kw["disp_atr"] = min(0.85, float(kw.get("disp_atr", 0.55)) + 0.05)
            elif mc_d >= 1.0:
                kw["pull_atr"] = max(0.55, float(kw["pull_atr"]) - 0.05)
    except Exception:
        pass
    return kw


def kelly_from_trades(trades: list[dict]) -> dict[str, Any]:
    """Half-Kelly from R sequence (wins vs losses)."""
    if len(trades) < 5:
        return {"kelly": None, "half_kelly": None, "note": "too_few"}
    rs = [float(t["r"]) for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    if not wins or not losses:
        return {"kelly": None, "half_kelly": None, "note": "one_sided"}
    p = len(wins) / len(rs)
    q = 1.0 - p
    avg_w = float(np.mean(wins))
    avg_l = abs(float(np.mean(losses)))
    if avg_l <= 1e-9:
        return {"kelly": None, "half_kelly": None, "note": "no_loss_mag"}
    b = avg_w / avg_l  # payoff ratio
    f = (p * b - q) / b
    f = float(f)
    return {
        "p_win": round(p, 3),
        "avg_win_r": round(avg_w, 3),
        "avg_loss_r": round(avg_l, 3),
        "b_payoff": round(b, 3),
        "kelly": round(max(-1.0, min(1.0, f)), 3),
        "half_kelly": round(max(0.0, min(0.5, f * 0.5)), 3),
        "note": "ok" if f > 0 else "negative_edge",
    }


def healed_battery(symbol: str, df) -> dict[str, Any]:
    """Like battery_for but injects healed EMA knobs."""
    brain = BRAINS.get(symbol.upper(), "ema_20_50_pull")
    ema_kw = _healed_ema_kw(symbol, brain)
    n = len(df)
    print(f"  {symbol}: bars={n} brain={brain} healed_kw={ema_kw or '—'}", flush=True)

    def gen(start_i=0, end_i=None, **kw):
        m = {**ema_kw, **kw}
        if brain == "fib_liq":
            return run_fib_trades(df, start_i=start_i, end_i=end_i, **m)
        if brain == "sma_20_50_cross":
            return run_sma_trades(df, start_i=start_i, end_i=end_i, **m)
        if brain == "breakout_20":
            return run_breakout_trades(df, start_i=start_i, end_i=end_i, **m)
        pull = float(m.pop("pull_atr", 1.0))
        disp = float(m.pop("disp_atr", 0.55))
        return run_ema_trades(
            df, start_i=start_i, end_i=end_i, pull_atr=pull, disp_atr=disp, **m
        )

    import random

    full_tr = gen()
    full = _stats(full_tr)
    split = int(n * 0.70)
    oos_tr = gen(start_i=split)
    oos = _stats(oos_tr)
    is_st = _stats(gen(end_i=split - 1))

    wf = []
    for k in range(4):
        test_start = int(n * (0.40 + 0.15 * k))
        test_end = int(n * (0.55 + 0.15 * k)) - 1
        if test_end <= test_start or test_start < 80:
            continue
        test_tr = gen(start_i=test_start, end_i=test_end)
        fold = {
            "fold": k + 1,
            "train": _stats(gen(end_i=test_start - 1)),
            "oos": _stats(test_tr),
        }
        wf.append(fold)

    torture = {
        "spread_0.15R": _stats(gen(spread_r=0.15)),
        "late_1bar": _stats(gen(late_bars=1)),
        "skip_30pct": _stats(gen(skip_p=0.30, rng=random.Random(7))),
    }
    mc = monte_carlo(full_tr, paths=MC_PATHS)
    kelly = kelly_from_trades(full_tr)
    v = verdict(full, oos, wf, torture, mc)

    print(
        f"    full E[R]={full['exp_r']} n={full['n']} | "
        f"OOS E[R]={oos['exp_r']} | Kelly={kelly.get('kelly')} half={kelly.get('half_kelly')} | "
        f"{v}",
        flush=True,
    )
    return {
        "symbol": symbol,
        "brain": brain,
        "healed_kw": ema_kw,
        "bars": n,
        "full": full,
        "in_sample": is_st,
        "oos": oos,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "kelly": kelly,
        "verdict": v,
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"HEALED edge battery {start} → {end_s}  "
        f"($200/R · 1h · promoted+soft knobs)",
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
                print(f"  SKIP {sym}: {len(df)} bars", flush=True)
                continue
            results.append(healed_battery(sym, df))
        except Exception as exc:
            print(f"  FAIL {sym}: {exc}", flush=True)
            results.append({"symbol": sym, "error": str(exc), "verdict": "DEAD"})

    payload = {
        "mode": "healed_algo",
        "window": {"start": start, "end": end_s, "tf": "1h_proxy"},
        "risk_usd": RISK_USD,
        "mc_paths": MC_PATHS,
        "results": results,
        "summary": {
            "survives": [r["symbol"] for r in results if r.get("verdict") == "SURVIVES"],
            "fragile": [r["symbol"] for r in results if r.get("verdict") == "FRAGILE"],
            "dead": [r["symbol"] for r in results if r.get("verdict") == "DEAD"],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### HEALED DESK — OOS / WF / MC / KELLY\n", flush=True)
    print(
        "| Symbol | Brain | Full E[R] | OOS E[R] | WF green | MC P(profit) | Kelly | Half-K | Verdict |",
        flush=True,
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---|", flush=True)
    for r in results:
        if r.get("error"):
            print(f"| {r['symbol']} | — | — | — | — | — | — | — | DEAD |", flush=True)
            continue
        wf_g = sum(1 for f in r.get("walk_forward") or [] if f["oos"]["exp_r"] > 0)
        wf_n = len(r.get("walk_forward") or [])
        mc = r.get("monte_carlo") or {}
        k = r.get("kelly") or {}
        print(
            f"| {r['symbol']} | {r['brain']} | {r['full']['exp_r']} | "
            f"{r['oos']['exp_r']} | {wf_g}/{wf_n} | "
            f"{mc.get('prob_profit', '—')}% | {k.get('kelly', '—')} | "
            f"{k.get('half_kelly', '—')} | **{r['verdict']}** |",
            flush=True,
        )
    print(
        f"\nSURVIVES: {payload['summary']['survives']}\n"
        f"FRAGILE:  {payload['summary']['fragile']}\n"
        f"DEAD:     {payload['summary']['dead']}\n"
        f"Wrote {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
