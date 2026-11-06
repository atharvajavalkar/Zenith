#!/usr/bin/env python3
"""Omni-horizon adaptive router — XM session spreads.

Why the deep hunt clustered on a few brains
-------------------------------------------
The deep hunt ranked cells by (1y + 180d + 90d). That crowns long-horizon
specialists. Short windows (7d/15d/30d) are noisy and usually FAIL even when
1y is PASS — so Nebula heat concentrates on circadian / baseline / session_mix
variants that print on 1y, not systems that stay green on every horizon.

This module builds a *different* objective:
  · Maintain a diverse brain pool (not 80 parametric twins)
  · Each bar, score brains on rolling multi-scale edge (fast/mid/slow)
  · Route to the best brain with hysteresis; sit out on disagreement / weak edge
  · Select params by multi-horizon survival (how many windows are green), not 1y PnL

Does NOT wire live. Writes:
  data/lab_omnihorizon_adaptive.json
  data/lab_omnihorizon_adaptive.progress.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.lab_adaptive_minute_day_v3_bt import _calendar  # noqa: E402
from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    BOOK,
    _classify_regime,
    _sides_kalman,
    _sides_snr,
    _static_sides_for_winner,
    _window_trades,
)
from analysis.lab_adaptive_xm_deep_hunt_bt import (  # noqa: E402
    RISK_USD,
    WINDOWS,
    _entries,
    _load,
    _simulate,
    _summarize,
    mk_circadian,
    mk_entropy,
    mk_ou_fade,
    mk_session_mix,
    mk_vol_clock,
    mk_volmom,
    mk_weekday_regime,
)
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_omnihorizon_adaptive.json"
PROGRESS = Path(ROOT) / "data" / "lab_omnihorizon_adaptive.progress.json"
LOG = Path(ROOT) / "data" / "lab_omnihorizon_adaptive.log"

# Focus TFs that matter for desk + scalp without another 40h run
TFS = ("5m", "15m", "3m")
TF_YEARS = {"5m": 5.0, "15m": 5.0, "3m": 3.0}

# Diverse pool — one archetype each (not parametric clones)
POOL_BUILDERS = (
    ("session_mix", lambda: mk_session_mix(0.22, 0.35)),
    ("session_mix_strict", lambda: mk_session_mix(0.32, 0.42)),
    ("circadian", lambda: mk_circadian(1.7, 0.6)),
    ("circadian_tight", lambda: mk_circadian(2.2, 0.5)),
    ("entropy", lambda: mk_entropy(0.78, 0.0)),
    ("entropy_snr", lambda: mk_entropy(0.72, 0.22)),
    ("ou_fade", lambda: mk_ou_fade(1.8)),
    ("vol_clock", lambda: mk_vol_clock()),
    ("weekday", lambda: mk_weekday_regime()),
    ("volmom", lambda: mk_volmom(0.5, 20)),
    ("snr", lambda: mk_snr_wrap(0.40)),
    ("kalman_fade", lambda: mk_kalman_wrap(2.0)),
)


def mk_snr_wrap(thr: float):
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        return _sides_snr(df, thr)

    return fn


def mk_kalman_wrap(z: float):
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        return _sides_kalman(df, z)

    return fn


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _rolling_edge(sides: np.ndarray, ret: np.ndarray, look: int) -> np.ndarray:
    """Rolling mean of sign(side)*sign(next_ret) when side!=0; else 0.5 neutral."""
    n = len(sides)
    hit = np.full(n, 0.5, dtype=float)
    nxt = np.roll(ret, -1)
    nxt[-1] = 0.0
    active = sides != 0
    raw = np.zeros(n, dtype=float)
    raw[active] = (np.sign(sides[active]) * np.sign(nxt[active])) > 0
    # expand to series for rolling
    s = pd.Series(np.where(active, raw, np.nan))
    # only average active bars
    roll = s.rolling(look, min_periods=max(20, look // 5)).mean()
    # fill with 0.5 when insufficient
    out = roll.fillna(0.5).to_numpy(dtype=float)
    return out


def _omni_route(
    brain_sides: dict[str, np.ndarray],
    ret: np.ndarray,
    *,
    look_fast: int,
    look_mid: int,
    look_slow: int,
    edge_min: float,
    hysteresis: int,
    min_agree: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pick sides adaptively; flat when weak / disagree."""
    names = list(brain_sides.keys())
    n = len(ret)
    edges = {}
    for name, sides in brain_sides.items():
        ef = _rolling_edge(sides, ret, look_fast)
        em = _rolling_edge(sides, ret, look_mid)
        es = _rolling_edge(sides, ret, look_slow)
        edges[name] = 0.45 * ef + 0.35 * em + 0.20 * es

    edge_mat = np.column_stack([edges[nm] for nm in names])
    best_i = np.argmax(edge_mat, axis=1)
    best_e = edge_mat[np.arange(n), best_i]

    # hysteresis
    pick = best_i.copy()
    last = int(pick[0])
    hold = 0
    for i in range(n):
        if hold > 0 and edge_mat[i, last] >= edge_min - 0.02:
            pick[i] = last
            hold -= 1
        else:
            last = int(best_i[i])
            pick[i] = last
            hold = hysteresis

    out = np.zeros(n, dtype=int)
    usage = {nm: 0 for nm in names}
    sitout = 0
    for i in range(n):
        if best_e[i] < edge_min:
            sitout += 1
            continue
        nm = names[int(pick[i])]
        side = int(brain_sides[nm][i])
        if side == 0:
            sitout += 1
            continue
        # multi-brain agreement on sign among top edges
        agree = 0
        for j, other in enumerate(names):
            if edge_mat[i, j] < edge_min:
                continue
            s2 = int(brain_sides[other][i])
            if s2 == side:
                agree += 1
        if agree < min_agree:
            sitout += 1
            continue
        out[i] = side
        usage[nm] += 1

    meta = {
        "usage": usage,
        "sitout_bars": sitout,
        "active_bars": int((out != 0).sum()),
    }
    return out, meta


def _horizon_survival(wins: dict[str, dict]) -> dict[str, Any]:
    keys = ["7d", "15d", "30d", "90d", "180d", "1y"]
    green = []
    for k in keys:
        w = wins.get(k) or {}
        pnl = float(w.get("pnl") or 0)
        er = float(w.get("exp_r") or 0)
        ok = pnl > 0 and er > 0 and int(w.get("n") or 0) >= 3
        if ok:
            green.append(k)
    return {
        "green": green,
        "n_green": len(green),
        "all_six": len(green) == 6,
        "score": (
            len(green),
            float((wins.get("30d") or {}).get("pnl") or -1e9),
            float((wins.get("90d") or {}).get("pnl") or -1e9),
            float((wins.get("1y") or {}).get("pnl") or -1e9),
        ),
    }


def _eval_sides(
    df: pd.DataFrame,
    sides: np.ndarray,
    sym: str,
    *,
    sl_atr: float,
    rr: float,
    gap: int,
) -> dict[str, Any]:
    entries = _entries(df, sides, sym, sl_atr=float(sl_atr), min_gap=int(gap))
    trades = _simulate(df, entries, float(rr), sym)
    wins = {}
    for wname, days in WINDOWS:
        if days is None:
            tw = trades
        else:
            tw = _window_trades(df, trades, days)
        wins[wname] = _summarize(tw)
    surv = _horizon_survival(wins)
    return {
        "windows": wins,
        "survival": surv,
        "n_entries": len(entries),
        "n_trades": len(trades),
        "_score": surv["score"],
    }


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    t0 = time.time()
    _log("=== Omni-horizon adaptive router (XM spreads) ===")
    _log("Objective: maximize multi-window green count, not 1y PnL alone\n")

    # param grid — keep lean
    route_grid = list(
        product(
            (80, 120),
            (240, 400),
            (600,),
            (0.52, 0.55),
            (25,),
            (2, 3),
        )
    )  # 2*2*1*2*1*2 = 16
    sl_grid = (2.0, 2.5, 3.0)
    rr_grid = (3.0, 4.0)
    gap_grid = (2, 3)

    per_pair: dict[str, Any] = {}
    desk_surv = {k: 0 for k, _ in WINDOWS if k != "full"}

    for pi, sym in enumerate(BOOK):
        per_pair[sym] = {"by_tf": {}, "best": None}
        for tf in TFS:
            years = TF_YEARS[tf]
            _log(f"\n######## {sym} @ {tf} · {years}y ########")
            _progress(
                {
                    "status": "running",
                    "pair": sym,
                    "tf": tf,
                    "pct": round(100 * (pi * len(TFS) + TFS.index(tf)) / (len(BOOK) * len(TFS)), 1),
                    "elapsed_sec": round(time.time() - t0, 1),
                }
            )
            try:
                df = _load(sym, tf, years)
            except Exception as exc:
                per_pair[sym]["by_tf"][tf] = {"error": str(exc), "done": True}
                _log(f"  skip: {exc}")
                continue

            # enrich already via _load path in deep hunt — _load returns enriched
            static, label, _ = _static_sides_for_winner(df, "")
            regime = _classify_regime(df)
            cal = _calendar(df)
            ctx = {
                "sym": sym,
                "static": static,
                "regime": regime,
                "cal": cal,
                "v2_preferred_sides": static,
                "family": "trend",
            }
            ret = df["close"].pct_change().fillna(0).to_numpy(dtype=float)

            brain_sides: dict[str, np.ndarray] = {"static": static.astype(int)}
            for name, builder in POOL_BUILDERS:
                try:
                    brain_sides[name] = builder()(df, ctx).astype(int)
                except Exception as exc:
                    _log(f"  brain {name} fail: {exc}")

            results = []
            live_sl = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))
            # Stage 1 — route search at fixed SL/RR/gap (survival objective)
            route_hits = []
            fixed_sl, fixed_rr, fixed_gap = 2.5, 3.0, 3
            for lf, lm, ls, emin, hyst, magree in route_grid:
                sides, meta = _omni_route(
                    brain_sides,
                    ret,
                    look_fast=lf,
                    look_mid=lm,
                    look_slow=ls,
                    edge_min=emin,
                    hysteresis=hyst,
                    min_agree=magree,
                )
                if int((sides != 0).sum()) < 30:
                    continue
                cell = _eval_sides(
                    df, sides, sym, sl_atr=fixed_sl, rr=fixed_rr, gap=fixed_gap
                )
                cell.update(
                    {
                        "sides": sides,
                        "route": {
                            "look_fast": lf,
                            "look_mid": lm,
                            "look_slow": ls,
                            "edge_min": emin,
                            "hysteresis": hyst,
                            "min_agree": magree,
                        },
                        "usage": meta["usage"],
                        "sitout_bars": meta["sitout_bars"],
                        "active_bars": meta["active_bars"],
                    }
                )
                route_hits.append(cell)
            if not route_hits:
                per_pair[sym]["by_tf"][tf] = {"done": True, "error": "no_routes"}
                continue
            route_hits.sort(key=lambda c: c["_score"], reverse=True)
            top_routes = route_hits[:4]

            # Stage 2 — SL/RR/gap fine-tune on top routes only
            for base in top_routes:
                sides = base["sides"]
                for sl_atr, rr, gap in product(sl_grid, rr_grid, gap_grid):
                    if sym in ("AUDUSD", "USDCHF", "USDJPY") and sl_atr < 2.0:
                        continue
                    cell = _eval_sides(df, sides, sym, sl_atr=sl_atr, rr=rr, gap=gap)
                    rt = base["route"]
                    cell.update(
                        {
                            "id": (
                                f"omni|f{rt['look_fast']}|m{rt['look_mid']}|s{rt['look_slow']}"
                                f"|e{rt['edge_min']}|h{rt['hysteresis']}|a{rt['min_agree']}"
                                f"|sl{sl_atr}|rr{rr}|g{gap}"
                            ),
                            "route": rt,
                            "usage": base["usage"],
                            "sitout_bars": base["sitout_bars"],
                            "active_bars": base["active_bars"],
                            "sl_atr": sl_atr,
                            "rr": rr,
                            "min_gap": gap,
                            "live_sl_ref": live_sl,
                            "tf": tf,
                        }
                    )
                    results.append(cell)
                # free sides ref
                base.pop("sides", None)

            if not results:
                per_pair[sym]["by_tf"][tf] = {"done": True, "error": "no_cells"}
                continue

            results.sort(key=lambda c: c["_score"], reverse=True)
            # also find best all_six if any
            all_six = [c for c in results if c["survival"]["all_six"]]
            best = all_six[0] if all_six else results[0]
            top = []
            for c in results[:15]:
                w1 = c["windows"]["1y"]
                top.append(
                    {
                        "id": c["id"],
                        "n_green": c["survival"]["n_green"],
                        "green": c["survival"]["green"],
                        "all_six": c["survival"]["all_six"],
                        "verdict_1y": w1.get("verdict"),
                        "pnl_1y": w1.get("pnl"),
                        "pnl_30d": c["windows"]["30d"].get("pnl"),
                        "pnl_90d": c["windows"]["90d"].get("pnl"),
                        "pnl_7d": c["windows"]["7d"].get("pnl"),
                        "usage": c["usage"],
                        "sl_atr": c["sl_atr"],
                        "rr": c["rr"],
                        "min_gap": c["min_gap"],
                    }
                )

            block = {
                "done": True,
                "n_bars": len(df),
                "n_brains": len(brain_sides),
                "n_cells": len(results),
                "n_all_six": len(all_six),
                "best": {
                    "id": best["id"],
                    "route": best["route"],
                    "usage": best["usage"],
                    "survival": best["survival"],
                    "sl_atr": best["sl_atr"],
                    "rr": best["rr"],
                    "min_gap": best["min_gap"],
                    "windows": {
                        k: {
                            "verdict": v.get("verdict"),
                            "pnl": v.get("pnl"),
                            "exp_r": v.get("exp_r"),
                            "n": v.get("n"),
                            "oos_exp_r": v.get("oos_exp_r"),
                            "wf_green": v.get("wf_green"),
                        }
                        for k, v in best["windows"].items()
                    },
                },
                "top15": top,
            }
            per_pair[sym]["by_tf"][tf] = block
            g = best["survival"]["n_green"]
            _log(
                f"  BEST green={g}/6 all_six={best['survival']['all_six']} "
                f"1y={best['windows']['1y'].get('verdict')} "
                f"${best['windows']['1y'].get('pnl')} · {best['id']}"
            )
            _log(f"  usage={best['usage']}")

            # track desk-ish
            if best["survival"]["all_six"]:
                for k in best["survival"]["green"]:
                    desk_surv[k] = desk_surv.get(k, 0) + 1

        # pair-level best = max green across TFs
        cands = []
        for tf, block in per_pair[sym]["by_tf"].items():
            b = block.get("best")
            if b:
                cands.append((b["survival"]["n_green"], b.get("windows", {}).get("1y", {}).get("pnl") or 0, tf, b))
        if cands:
            cands.sort(reverse=True)
            per_pair[sym]["best"] = {"tf": cands[0][2], **cands[0][3]}

    # summary table
    survivors = []
    for sym, row in per_pair.items():
        for tf, block in (row.get("by_tf") or {}).items():
            b = block.get("best")
            if not b:
                continue
            survivors.append(
                {
                    "symbol": sym,
                    "tf": tf,
                    "n_green": b["survival"]["n_green"],
                    "all_six": b["survival"]["all_six"],
                    "green": b["survival"]["green"],
                    "pnl_1y": (b.get("windows") or {}).get("1y", {}).get("pnl"),
                    "pnl_30d": (b.get("windows") or {}).get("30d", {}).get("pnl"),
                    "id": b.get("id"),
                    "usage": b.get("usage"),
                }
            )
    survivors.sort(key=lambda r: (-int(r["n_green"]), -float(r["pnl_1y"] or 0)))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "risk_usd": RISK_USD,
        "objective": "maximize multi-horizon green count (7d..1y) under XM session spreads",
        "why_not_deep_hunt": (
            "Deep hunt ranked by 1y+180d+90d, so Nebula heat clustered on long-horizon "
            "specialists that still FAIL 7d/15d/30d. Omni router switches brains by "
            "rolling multi-scale edge and scores survival across all six windows."
        ),
        "pool": [n for n, _ in POOL_BUILDERS] + ["static"],
        "tfs": list(TFS),
        "per_pair": per_pair,
        "leaderboard": survivors,
        "all_six_count": sum(1 for r in survivors if r["all_six"]),
        "note": "Research only — not live-wired.",
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress({"status": "done", "pct": 100, "elapsed_sec": payload["elapsed_sec"]})
    _log(
        f"\nDONE · {payload['elapsed_sec']}s · all_six={payload['all_six_count']}/{len(survivors)}"
    )
    for r in survivors[:12]:
        _log(
            f"  {r['symbol']:7} {r['tf']:3} green={r['n_green']}/6 "
            f"all={r['all_six']} 1y=${r['pnl_1y']} 30d=${r['pnl_30d']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
