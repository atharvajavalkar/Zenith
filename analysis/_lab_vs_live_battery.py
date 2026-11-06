#!/usr/bin/env python3
"""RESEARCH — Live brains vs Strategy-Lab winners. Honest OOS/WF/torture/MC.

Entry-cache + simulate (same as Strategy Lab), 1h ~700d, live SYMBOL_RR.
No dense/bootstrap. Does NOT change live brains.
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
    _collect_mom_roc,
    _collect_vol_scaled_mom,
    _fetch_daily,
)
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402
from config import SYMBOL_ENTRY_BRAIN, symbol_rr  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_vs_live_battery.json"

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

LIVE_ALIAS = {
    "fib_liq_sweep": "fib_liq",
    "fib_liq": "fib_liq",
    "ema_20_50_tight": "ema_20_50_tight",
    "ema_20_50_pull": "ema_20_50_pull",
    "sma_20_50_cross": "sma_20_50_cross",
    "breakout_20": "breakout_20",
}


def _live_brain(sym: str) -> str:
    raw = (SYMBOL_ENTRY_BRAIN.get(sym) or "ema_20_50_pull").lower()
    return LIVE_ALIAS.get(raw, raw)


def _collector(sid: str) -> Callable:
    if sid == "roc_60":
        return lambda d: _collect_mom_roc(d, 60)
    if sid == "roc_120":
        return lambda d: _collect_mom_roc(d, 120)
    if sid == "volmom":
        return _collect_vol_scaled_mom
    if sid in ("sma_cross", "sma_20_50_cross"):
        return _collect_sma
    if sid == "bb_2":
        return lambda d: _bb_std(d, 2.0)
    if sid == "cta_50_200":
        return lambda d: _collect_cta(d, 50, 200)
    if sid == "fib_liq":
        return _collect_fib
    if sid == "breakout_20":
        return lambda d: _collect_breakout(d, 20)
    if sid == "ema_20_50_tight":
        return lambda d: _collect_ema(d, pull_atr=0.75, disp_atr=0.70)
    if sid == "ema_20_50_pull":
        return lambda d: _collect_ema(d, pull_atr=1.0, disp_atr=0.55)
    raise KeyError(sid)


def _load_frame(sym: str, tf: str, start: str, end_s: str):
    if tf == "1h":
        raw = _fetch_1h(sym, start, end_s)
        df = _prep(raw)
    else:
        raw = _fetch_daily(sym, start, end_s)
        df = _prep(_clip(raw, start, end_s))
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_s).replace(hour=23, minute=59, tzinfo=timezone.utc)
    return df[(df.index >= start_dt) & (df.index <= end_dt)]


def _battery(symbol: str, df, *, brain: str, rr: float, mode: str) -> dict[str, Any]:
    n = len(df)
    entries = _collector(brain)(df)
    print(
        f"  [{mode}] {symbol} brain={brain} rr={rr:g} bars={n} entries={len(entries)}",
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
        "mode": mode,
        "brain": brain,
        "rr": rr,
        "bars": n,
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
    score = 1.0
    score += 0.9 * s
    score += 0.35 * f
    score -= 0.7 * d
    if oos_er:
        score += float(np.clip(np.mean(oos_er) * 4.0, -1.5, 2.0))
    if mc_p:
        score += float(np.clip((np.mean(mc_p) - 50.0) / 25.0, -1.0, 1.5))
    score = float(np.clip(round(score, 1), 1.0, 10.0))
    return {
        "survives": s,
        "fragile": f,
        "dead": d,
        "mean_oos_exp_r": round(float(np.mean(oos_er)), 4) if oos_er else None,
        "mean_mc_prob": round(float(np.mean(mc_p)), 1) if mc_p else None,
        "score_1_to_10": score,
    }


def _run_tf(tf: str, days: int) -> dict[str, Any]:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=days)).isoformat()
    end_s = end.isoformat()
    min_bars = 250 if tf == "1h" else 400
    print(
        f"\n===== TF={tf} window {start}→{end_s} ({days}d) · ${RISK_USD}/R =====\n",
        flush=True,
    )

    live_rows: list[dict] = []
    lab_rows: list[dict] = []
    head_to_head: list[dict] = []

    for sym in BOOK:
        rr = float(symbol_rr(sym, default=2.0))
        try:
            df = _load_frame(sym, tf, start, end_s)
            if len(df) < min_bars:
                raise RuntimeError(f"only {len(df)} bars")
        except Exception as exc:
            print(f"  FAIL {sym}: {exc}", flush=True)
            err = {"symbol": sym, "error": str(exc), "verdict": "DEAD"}
            live_rows.append(err)
            lab_rows.append(err)
            continue

        live_b = _live_brain(sym)
        lab_b = LAB_WINNERS[sym]
        live = _battery(sym, df, brain=live_b, rr=rr, mode="live")
        lab = _battery(sym, df, brain=lab_b, rr=rr, mode="lab")
        live_rows.append(live)
        lab_rows.append(lab)

        rank = {"SURVIVES": 2, "FRAGILE": 1, "DEAD": 0}
        lv, rv = rank.get(live["verdict"], 0), rank.get(lab["verdict"], 0)
        if rv > lv:
            better = "lab"
        elif lv > rv:
            better = "live"
        else:
            lo = float(live["oos"]["exp_r"])
            la = float(lab["oos"]["exp_r"])
            if la > lo + 0.02:
                better = "lab"
            elif lo > la + 0.02:
                better = "live"
            else:
                better = "tie"
        head_to_head.append(
            {
                "symbol": sym,
                "rr": rr,
                "live_brain": live_b,
                "lab_brain": lab_b,
                "live_verdict": live["verdict"],
                "lab_verdict": lab["verdict"],
                "live_oos_exp_r": live["oos"]["exp_r"],
                "lab_oos_exp_r": lab["oos"]["exp_r"],
                "live_full_n": live["full"]["n"],
                "lab_full_n": lab["full"]["n"],
                "live_mc": (live.get("monte_carlo") or {}).get("prob_profit"),
                "lab_mc": (lab.get("monte_carlo") or {}).get("prob_profit"),
                "better": better,
            }
        )

    live_desk = _score_desk(live_rows)
    lab_desk = _score_desk(lab_rows)
    lab_wins = sum(1 for h in head_to_head if h["better"] == "lab")
    live_wins = sum(1 for h in head_to_head if h["better"] == "live")
    ties = sum(1 for h in head_to_head if h["better"] == "tie")

    if lab_desk["score_1_to_10"] >= live_desk["score_1_to_10"] + 1.0 and lab_wins >= 5:
        note = "switch_recommended"
    elif lab_desk["score_1_to_10"] > live_desk["score_1_to_10"] and lab_wins >= live_wins:
        note = "selective_switch"
    elif lab_desk["dead"] >= 3:
        note = "do_not_switch"
    else:
        note = "hold_live"

    print(
        f"\n### HEAD-TO-HEAD {tf}\n",
        flush=True,
    )
    print(
        f"{'Pair':8} {'Live':18} {'Lab':14} {'L-ver':9} {'Lab-ver':9} "
        f"{'L-OOS':>7} {'Lab-OOS':>8} {'Ln':>4} {'Labn':>5} {'Better':7}",
        flush=True,
    )
    print("-" * 100, flush=True)
    for h in head_to_head:
        print(
            f"{h['symbol']:8} {h['live_brain']:18} {h['lab_brain']:14} "
            f"{h['live_verdict']:9} {h['lab_verdict']:9} "
            f"{h['live_oos_exp_r']:7.3f} {h['lab_oos_exp_r']:8.3f} "
            f"{h['live_full_n']:4} {h['lab_full_n']:5} {h['better']:7}",
            flush=True,
        )
    print(
        f"\nLIVE {tf}  {live_desk['survives']}S/{live_desk['fragile']}F/{live_desk['dead']}D "
        f"· {live_desk['score_1_to_10']}/10 · OOS E[R]={live_desk['mean_oos_exp_r']} "
        f"· MC={live_desk['mean_mc_prob']}%",
        flush=True,
    )
    print(
        f"LAB  {tf}  {lab_desk['survives']}S/{lab_desk['fragile']}F/{lab_desk['dead']}D "
        f"· {lab_desk['score_1_to_10']}/10 · OOS E[R]={lab_desk['mean_oos_exp_r']} "
        f"· MC={lab_desk['mean_mc_prob']}%",
        flush=True,
    )
    print(
        f"Pair wins: lab={lab_wins} live={live_wins} tie={ties} → {note}",
        flush=True,
    )
    return {
        "tf": tf,
        "window": {"start": start, "end": end_s, "days": days},
        "head_to_head": head_to_head,
        "live": {"results": live_rows, "desk": live_desk},
        "lab": {"results": lab_rows, "desk": lab_desk},
        "comparison": {
            "lab_pair_wins": lab_wins,
            "live_pair_wins": live_wins,
            "ties": ties,
            "live_score": live_desk["score_1_to_10"],
            "lab_score": lab_desk["score_1_to_10"],
            "switch_recommendation": note,
        },
    }


def main() -> int:
    t0 = time.time()
    # Daily ~15y = Strategy Lab native TF; 1h ~700d = live-proxy portability
    daily = _run_tf("1d", int(15 * 365.25))
    hourly = _run_tf("1h", 700)

    # Final score: weight daily 60% (where lab was selected) + 1h 40% (live portability)
    live_blend = round(
        0.6 * daily["comparison"]["live_score"] + 0.4 * hourly["comparison"]["live_score"],
        1,
    )
    lab_blend = round(
        0.6 * daily["comparison"]["lab_score"] + 0.4 * hourly["comparison"]["lab_score"],
        1,
    )
    if lab_blend >= live_blend + 1.0 and daily["comparison"]["lab_pair_wins"] >= 5:
        final_note = "switch_recommended"
    elif lab_blend > live_blend and hourly["comparison"]["lab_score"] >= 5.0:
        final_note = "selective_switch"
    elif hourly["comparison"]["lab_score"] < 4.0 or hourly["lab"]["desk"]["dead"] >= 4:
        final_note = "do_not_switch"
    else:
        final_note = "hold_live"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "risk_usd": RISK_USD,
        "lab_winners": LAB_WINNERS,
        "live_brains": {s: _live_brain(s) for s in BOOK},
        "daily_15y": daily,
        "hourly_700d": hourly,
        "final": {
            "live_score_1_to_10": live_blend,
            "lab_score_1_to_10": lab_blend,
            "recommendation": final_note,
            "note": (
                "Blend = 60% daily-15y (lab selection TF) + 40% 1h-700d (live portability). "
                "Do not switch if lab dies on 1h."
            ),
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print("\n### FINAL 1–10 (blend)\n", flush=True)
    print(f"LIVE brains: {live_blend}/10", flush=True)
    print(f"LAB winners: {lab_blend}/10", flush=True)
    print(f"Recommendation: {final_note}", flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
