#!/usr/bin/env python3
"""RESEARCH — Strategy Lab winners on 5m only. OOS / WF / torture / MC.

Does NOT change live brains.
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
    _prep,
    _stats,
    monte_carlo,
    verdict,
)
from analysis._htf_trend_5m_exit_bt import _fetch_5m  # noqa: E402
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_sma,
    simulate,
)
from analysis.strategy_lab import (  # noqa: E402
    _collect_cta,
    _collect_mom_roc,
    _collect_vol_scaled_mom,
)
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402
from config import symbol_rr  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_winners_5m.json"

LAB_WINNERS = {
    "GOLD": "roc_60",
    "USDCHF": "volmom",
    "USDJPY": "sma_cross",
    "NAS100": "roc_120",
    "UKOIL": "roc_120",
    "AUDUSD": "bb_2",
    "BTCUSD": "cta_50_200",
    "ETHUSD": "roc_60",
}


def _collector(sid: str) -> Callable:
    if sid == "roc_60":
        return lambda d: _collect_mom_roc(d, 60)
    if sid == "roc_120":
        return lambda d: _collect_mom_roc(d, 120)
    if sid == "volmom":
        return _collect_vol_scaled_mom
    if sid == "sma_cross":
        return _collect_sma
    if sid == "bb_2":
        return lambda d: _bb_std(d, 2.0)
    if sid == "cta_50_200":
        return lambda d: _collect_cta(d, 50, 200)
    raise KeyError(sid)


def _battery(symbol: str, df, *, brain: str, rr: float) -> dict[str, Any]:
    n = len(df)
    entries = _collector(brain)(df)
    print(
        f"  {symbol} brain={brain} rr={rr:g} bars={n} entries={len(entries)}",
        flush=True,
    )

    def gen(start_i=0, end_i=None, **kw):
        spread_r = float(kw.get("spread_r") or 0)
        skip_p = float(kw.get("skip_p") or 0)
        sl_widen = float(kw.get("sl_widen") or 1.0)
        rng = kw.get("rng") or random.Random(0)
        tr = simulate(df, entries, rr, start_i=start_i, end_i=end_i)
        if sl_widen != 1.0:
            out = []
            for t in tr:
                r = float(t["r"])
                if r < 0:
                    r = -1.0 * sl_widen
                out.append({**t, "r": round(r, 4), "pnl": round(r * RISK_USD, 2)})
            tr = out
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
    is_st = _stats(gen(end_i=split - 1))

    wf = []
    for k in range(4):
        test_start = int(n * (0.40 + 0.15 * k))
        test_end = int(n * (0.55 + 0.15 * k)) - 1
        if test_end <= test_start or test_start < 120:
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
        "skip_30pct": _stats(gen(skip_p=0.30, rng=random.Random(7))),
        "sl_widen_20pct": _stats(gen(sl_widen=1.20)),
    }
    mc = monte_carlo(full_tr)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
    print(
        f"    full n={full['n']} E[R]={full['exp_r']} wr={full['wr']}% | "
        f"OOS n={oos['n']} E[R]={oos['exp_r']} | WF {wf_g}/{len(wf)} | "
        f"MC {mc.get('prob_profit')}% → {v}",
        flush=True,
    )
    return {
        "symbol": symbol,
        "brain": brain,
        "rr": rr,
        "tf": "5m",
        "bars": n,
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "entries": len(entries),
        "full": full,
        "in_sample": is_st,
        "oos": oos,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": wf_g,
        "wf_n": len(wf),
    }


def _score_desk(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    s = sum(1 for r in ok if r.get("verdict") == "SURVIVES")
    f = sum(1 for r in ok if r.get("verdict") == "FRAGILE")
    d = sum(1 for r in ok if r.get("verdict") == "DEAD")
    oos_er = [float(r["oos"]["exp_r"]) for r in ok if r.get("oos")]
    mc_p = [
        float(r["monte_carlo"]["prob_profit"])
        for r in ok
        if (r.get("monte_carlo") or {}).get("prob_profit") is not None
    ]
    score = 1.0 + 0.9 * s + 0.35 * f - 0.7 * d
    if oos_er:
        score += float(np.clip(np.mean(oos_er) * 4.0, -1.5, 2.0))
    if mc_p:
        score += float(np.clip((np.mean(mc_p) - 50.0) / 25.0, -1.0, 1.5))
    return {
        "survives": s,
        "fragile": f,
        "dead": d,
        "mean_oos_exp_r": round(float(np.mean(oos_er)), 4) if oos_er else None,
        "mean_mc_prob": round(float(np.mean(mc_p)), 1) if mc_p else None,
        "score_1_to_10": float(np.clip(round(score, 1), 1.0, 10.0)),
    }


def main() -> int:
    t0 = time.time()
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    # Max practical 5m depth (TV/YF); ~120 calendar days
    start = (end - timedelta(days=120)).isoformat()
    end_s = end.isoformat()
    print(
        f"LAB winners · 5m only · {start} → {end_s} · ${RISK_USD}/R · live SYMBOL_RR\n",
        flush=True,
    )

    rows: list[dict] = []
    for sym in BOOK:
        rr = float(symbol_rr(sym, default=2.0))
        brain = LAB_WINNERS[sym]
        try:
            raw = _fetch_5m(sym, start, end_s)
            df = _prep(raw.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna())
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if len(df) < 400:
                raise RuntimeError(f"only {len(df)} bars")
            rows.append(_battery(sym, df, brain=brain, rr=rr))
        except Exception as exc:
            print(f"  FAIL {sym}: {exc}", flush=True)
            rows.append(
                {"symbol": sym, "brain": brain, "error": str(exc), "verdict": "DEAD"}
            )

    desk = _score_desk(rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "lab_winners_5m_only",
        "window": {"start": start, "end": end_s, "tf": "5m", "days": 120},
        "risk_usd": RISK_USD,
        "lab_winners": LAB_WINNERS,
        "results": rows,
        "desk": desk,
        "elapsed_s": round(time.time() - t0, 1),
        "note": "Research only — does not change live SYMBOL_ENTRY_BRAIN",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### LAB WINNERS · 5m ONLY\n", flush=True)
    print(
        f"{'Pair':8} {'Brain':14} {'Verdict':9} {'Full n':>7} {'E[R]':>7} "
        f"{'OOS n':>6} {'OOS E[R]':>8} {'WF':>5} {'MC%':>6}",
        flush=True,
    )
    print("-" * 85, flush=True)
    for r in rows:
        if r.get("error"):
            print(f"{r['symbol']:8} {(r.get('brain') or '—'):14} DEAD      —", flush=True)
            continue
        mc = r.get("monte_carlo") or {}
        print(
            f"{r['symbol']:8} {r['brain']:14} {r['verdict']:9} "
            f"{r['full']['n']:7} {r['full']['exp_r']:7.3f} "
            f"{r['oos']['n']:6} {r['oos']['exp_r']:8.3f} "
            f"{r['wf_green']}/{r['wf_n']:>3} {mc.get('prob_profit', '—')!s:>6}",
            flush=True,
        )
    print(
        f"\nDesk {desk['survives']}S/{desk['fragile']}F/{desk['dead']}D "
        f"· score {desk['score_1_to_10']}/10 · mean OOS E[R]={desk['mean_oos_exp_r']} "
        f"· MC={desk['mean_mc_prob']}%",
        flush=True,
    )
    print(f"Wrote {OUT} in {payload['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
