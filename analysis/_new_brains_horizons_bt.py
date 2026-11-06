#!/usr/bin/env python3
"""Test CURRENT live brains across 1..10y and 15y windows.

Uses daily OHLC (only TF with honest multi-year depth on free feeds).
Also runs 1h for 1y+2y when available.

READ-ONLY — no config / brain changes.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from analysis._edge_survive_bt import (  # noqa: E402
    BOOK,
    RISK_USD,
    _fetch_1h,
    _prep,
    _stats,
    monte_carlo,
    verdict,
)
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_breakout,
    _collect_ema,
    _collect_fib,
    _collect_sma,
    simulate,
)
from analysis.strategy_lab import (  # noqa: E402
    _clip,
    _collect_cta,
    _collect_rsi,
    _collect_supertrend,
    _fetch_daily,
)
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402
from config import symbol_entry_brain, symbol_rr  # noqa: E402

OUT = Path(ROOT) / "data" / "new_brains_horizons.json"
HORIZONS_Y = list(range(1, 11)) + [15]  # 1..10 and 15


def _collector(brain: str) -> Callable:
    b = brain.lower().strip()
    if b in ("fib_liq_sweep", "fib_liq"):
        return _collect_fib
    if b in ("ema_20_50_tight", "ema_tight"):
        return lambda d: _collect_ema(d, pull_atr=0.75, disp_atr=0.70)
    if b in ("ema_20_50_pull", "ema_pull"):
        return lambda d: _collect_ema(d, pull_atr=1.0, disp_atr=0.55)
    if b == "sma_20_50_cross":
        return _collect_sma
    if b == "breakout_20":
        return lambda d: _collect_breakout(d, 20)
    if b == "cta_10_40":
        return lambda d: _collect_cta(d, 10, 40)
    if b in ("rsi", "rsi_mr"):
        return _collect_rsi
    if b in ("bb_1.5", "bb_1_5"):
        return lambda d: _bb_std(d, 1.5)
    if b in ("bb_2", "bb_2.0"):
        return lambda d: _bb_std(d, 2.0)
    if b in ("st_3", "supertrend_3"):
        return lambda d: _collect_supertrend(d, 3.0)
    raise KeyError(brain)


def _battery(df, entries, rr: float) -> dict[str, Any]:
    n = len(df)
    if n < 80 or not entries:
        return {
            "bars": n,
            "entries": len(entries),
            "full": {"n": 0, "pnl": 0.0, "exp_r": 0.0, "wr": 0.0},
            "oos": {"n": 0, "pnl": 0.0, "exp_r": 0.0, "wr": 0.0},
            "monte_carlo": {},
            "verdict": "DEAD",
            "wf_green": 0,
            "wf_n": 0,
            "note": "too_few_bars_or_entries",
        }

    def gen(start_i=0, end_i=None, **kw):
        spread_r = float(kw.get("spread_r") or 0)
        skip_p = float(kw.get("skip_p") or 0)
        rng = kw.get("rng") or random.Random(0)
        tr = simulate(df, entries, rr, start_i=start_i, end_i=end_i)
        if spread_r > 0:
            tr = [
                {
                    **t,
                    "r": round(float(t["r"]) - spread_r, 4),
                    "pnl": round((float(t["r"]) - spread_r) * RISK_USD, 2),
                }
                for t in tr
            ]
        if skip_p > 0 and tr:
            tr = [t for t in tr if rng.random() >= skip_p]
        return tr

    full_tr = gen()
    full = _stats(full_tr)
    split = int(n * 0.70)
    oos = _stats(gen(start_i=split))
    wf = []
    for k in range(4):
        ts = int(n * (0.40 + 0.15 * k))
        te = int(n * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 60:
            continue
        wf.append(
            {
                "fold": k + 1,
                "train": _stats(gen(end_i=ts - 1)),
                "oos": _stats(gen(start_i=ts, end_i=te)),
            }
        )
    torture = {
        "spread_0.15R": _stats(gen(spread_r=0.15)),
        "skip_30pct": _stats(gen(skip_p=0.30, rng=random.Random(7))),
    }
    mc = monte_carlo(full_tr)
    v = verdict(full, oos, wf, torture, mc)
    return {
        "bars": n,
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "entries": len(entries),
        "full": full,
        "oos": oos,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": sum(1 for f in wf if f["oos"]["exp_r"] > 0),
        "wf_n": len(wf),
    }


def _load_daily(sym: str, start: str, end: str):
    raw = _fetch_daily(sym, start, end)
    df = _prep(_clip(raw, start, end))
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(hour=23, minute=59, tzinfo=timezone.utc)
    return df[(df.index >= start_dt) & (df.index <= end_dt)]


def main() -> int:
    t0 = time.time()
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    end_s = end.isoformat()
    brains = {s: symbol_entry_brain(s) for s in BOOK}
    print(
        f"NEW BRAINS multi-horizon · daily 1..10y + 15y · ${RISK_USD}/R\n"
        f"Brains: {brains}\n"
        f"Note: daily = only honest multi-year TF on free feeds (not 1m/5m).\n",
        flush=True,
    )

    # Preload max daily once per symbol (15y)
    start_15 = (end - timedelta(days=int(15 * 365.25))).isoformat()
    daily_full: dict[str, Any] = {}
    for sym in BOOK:
        try:
            daily_full[sym] = _load_daily(sym, start_15, end_s)
            print(
                f"  {sym} daily: {len(daily_full[sym])} bars "
                f"{daily_full[sym].index[0].date()}→{daily_full[sym].index[-1].date()}",
                flush=True,
            )
        except Exception as e:
            print(f"  {sym} daily FAIL: {e}", flush=True)
            daily_full[sym] = None

    by_horizon: dict[str, Any] = {}
    for years in HORIZONS_Y:
        start = (end - timedelta(days=int(years * 365.25))).isoformat()
        print(f"\n===== {years}y daily ({start}→{end_s}) =====", flush=True)
        rows = []
        for sym in BOOK:
            brain = brains[sym]
            rr = float(symbol_rr(sym, default=2.0))
            df_all = daily_full.get(sym)
            if df_all is None or len(df_all) < 100:
                rows.append({"symbol": sym, "brain": brain, "error": "no_data", "verdict": "DEAD"})
                continue
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            df = df_all[df_all.index >= start_dt]
            if len(df) < 100:
                rows.append(
                    {
                        "symbol": sym,
                        "brain": brain,
                        "error": f"only {len(df)} bars",
                        "verdict": "DEAD",
                    }
                )
                continue
            try:
                entries = _collector(brain)(df)
                bat = _battery(df, entries, rr)
            except Exception as e:
                rows.append({"symbol": sym, "brain": brain, "error": str(e), "verdict": "DEAD"})
                continue
            rows.append({"symbol": sym, "brain": brain, "rr": rr, "tf": "1d", **bat})
            print(
                f"  {sym:8} {brain:18} n={bat['full']['n']:4} "
                f"PnL=${bat['full']['pnl']:10,.0f} OOS=${bat['oos']['pnl']:9,.0f} "
                f"E[R]={bat['oos']['exp_r']:+.3f} {bat['verdict']}",
                flush=True,
            )

        ok = [r for r in rows if r.get("full")]
        s = sum(1 for r in rows if r.get("verdict") == "SURVIVES")
        f = sum(1 for r in rows if r.get("verdict") == "FRAGILE")
        d = sum(1 for r in rows if r.get("verdict") == "DEAD")
        full_pnl = sum(float(r["full"]["pnl"]) for r in ok)
        oos_pnl = sum(float(r["oos"]["pnl"]) for r in ok)
        mean_oos_er = (
            float(np.mean([r["oos"]["exp_r"] for r in ok])) if ok else None
        )
        # Fit score: survives weight + positive desk OOS
        fit = s * 1.0 + f * 0.35 - d * 0.5
        if oos_pnl > 0:
            fit += 1.0
        if full_pnl > 0:
            fit += 0.5
        fit_score = float(np.clip(round(1 + fit, 1), 1.0, 10.0))

        by_horizon[str(years)] = {
            "years": years,
            "start": start,
            "end": end_s,
            "tf": "1d",
            "results": rows,
            "desk": {
                "survives": s,
                "fragile": f,
                "dead": d,
                "full_pnl": round(full_pnl, 2),
                "oos_pnl": round(oos_pnl, 2),
                "mean_oos_exp_r": round(mean_oos_er, 4) if mean_oos_er is not None else None,
                "fit_score_1_to_10": fit_score,
                "fits": fit_score >= 5.0 and oos_pnl > 0 and s >= 3,
            },
        }
        desk = by_horizon[str(years)]["desk"]
        print(
            f"  → Desk {s}S/{f}F/{d}D · Full ${full_pnl:,.0f} · OOS ${oos_pnl:,.0f} · "
            f"fit {fit_score}/10 · {'FITS' if desk['fits'] else 'DOES NOT FIT'}",
            flush=True,
        )

    # Extra: 1h for 1y and 2y if data allows
    hourly = {}
    for years in (1, 2):
        start = (end - timedelta(days=int(years * 365.25))).isoformat()
        # Yahoo 1h ~730d max
        if years > 2:
            continue
        print(f"\n===== {years}y 1h ({start}→{end_s}) =====", flush=True)
        rows = []
        for sym in BOOK:
            brain = brains[sym]
            rr = float(symbol_rr(sym, default=2.0))
            try:
                raw = _fetch_1h(sym, start, end_s)
                df = _prep(_clip(raw, start, end_s))
                start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
                end_dt = datetime.fromisoformat(end_s).replace(
                    hour=23, minute=59, tzinfo=timezone.utc
                )
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                if len(df) < 200:
                    raise RuntimeError(f"only {len(df)} bars")
                entries = _collector(brain)(df)
                bat = _battery(df, entries, rr)
                rows.append({"symbol": sym, "brain": brain, "rr": rr, "tf": "1h", **bat})
                print(
                    f"  {sym:8} n={bat['full']['n']:4} PnL=${bat['full']['pnl']:10,.0f} "
                    f"OOS=${bat['oos']['pnl']:9,.0f} {bat['verdict']}",
                    flush=True,
                )
            except Exception as e:
                rows.append({"symbol": sym, "brain": brain, "error": str(e), "verdict": "DEAD"})
        ok = [r for r in rows if r.get("full")]
        s = sum(1 for r in rows if r.get("verdict") == "SURVIVES")
        f = sum(1 for r in rows if r.get("verdict") == "FRAGILE")
        d = sum(1 for r in rows if r.get("verdict") == "DEAD")
        full_pnl = sum(float(r["full"]["pnl"]) for r in ok)
        oos_pnl = sum(float(r["oos"]["pnl"]) for r in ok)
        fit = s * 1.0 + f * 0.35 - d * 0.5 + (1 if oos_pnl > 0 else 0) + (0.5 if full_pnl > 0 else 0)
        fit_score = float(np.clip(round(1 + fit, 1), 1.0, 10.0))
        hourly[str(years)] = {
            "years": years,
            "tf": "1h",
            "results": rows,
            "desk": {
                "survives": s,
                "fragile": f,
                "dead": d,
                "full_pnl": round(full_pnl, 2),
                "oos_pnl": round(oos_pnl, 2),
                "fit_score_1_to_10": fit_score,
                "fits": fit_score >= 5.0 and oos_pnl > 0 and s >= 3,
            },
        }
        print(
            f"  → 1h desk {s}S/{f}F/{d}D · OOS ${oos_pnl:,.0f} · fit {fit_score}/10",
            flush=True,
        )

    # Summary table
    fits_list = []
    print("\n" + "=" * 95, flush=True)
    print("### NEW BRAINS FIT BY HORIZON (daily)\n", flush=True)
    print(
        f"{'Years':>6} {'S/F/D':>8} {'Full PnL':>12} {'OOS PnL':>12} {'Fit':>5} {'Fits?':>8}",
        flush=True,
    )
    print("-" * 60, flush=True)
    for years in HORIZONS_Y:
        desk = by_horizon[str(years)]["desk"]
        tag = "YES" if desk["fits"] else "NO"
        fits_list.append(desk["fits"])
        print(
            f"{years:6} {desk['survives']}/{desk['fragile']}/{desk['dead']:>2} "
            f"${desk['full_pnl']:11,.0f} ${desk['oos_pnl']:11,.0f} "
            f"{desk['fit_score_1_to_10']:5.1f} {tag:>8}",
            flush=True,
        )

    n_fit = sum(1 for x in fits_list if x)
    overall = (
        "fits_most_horizons"
        if n_fit >= 7
        else "fits_some"
        if n_fit >= 3
        else "does_not_fit_multi_year"
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "new_brains_horizons_readonly",
        "brains": brains,
        "risk_usd": RISK_USD,
        "daily_horizons": by_horizon,
        "hourly_horizons": hourly,
        "summary": {
            "horizons_tested": HORIZONS_Y,
            "horizons_that_fit": [
                y for y in HORIZONS_Y if by_horizon[str(y)]["desk"]["fits"]
            ],
            "horizons_fit_count": n_fit,
            "overall": overall,
            "note": (
                "Daily TF used for 1–15y (intraday history unavailable that far). "
                "Brains were selected on short 1m/3m/5m/15m — multi-year daily is a portability stress test."
            ),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n→ Overall: {overall} ({n_fit}/{len(HORIZONS_Y)} horizons fit)", flush=True)
    print(f"Wrote {OUT} in {payload['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
