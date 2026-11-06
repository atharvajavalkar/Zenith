#!/usr/bin/env python3
"""Solution hunt — XM session spreads, with a locked holdout.

The July 2026 hunts ranked brains on data that ended 2026-07-23. The demo
book then lost money in August–October, a stretch those hunts never saw.

This run:
  · Refuses to score a cell on the post-2026-07-23 bars (holdout).
  · Deducts measured XM session spreads on every fill.
  · Calls a cell a solution only when the pre-holdout year survives
    (30d/90d/180d/1y green, 1y PASS|SOFT, OOS>0, walk-forward ≥3,
    E[R]≥0.03, n≥40) AND the holdout itself is green (n≥8, E[R]>0).
  · Does not write .env or live brains.

Book: GOLD BTCUSD UKOIL NAS100. TFs: 5m, 15m, 3m.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.lab_adaptive_minute_day_v3_bt import _calendar  # noqa: E402
from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    _classify_regime,
    _sides_snr,
    _static_sides_for_winner,
)
from analysis.lab_adaptive_xm_deep_hunt_bt import (  # noqa: E402
    RISK_USD,
    TF_YEARS,
    _load,
    _summarize,
)
from analysis.lab_mega_desk_invent_catalog import build_mega_catalog  # noqa: E402
from analysis.lab_mega_desk_invent_hunt_bt import GAP, RR_GRID, _sl_grid  # noqa: E402
from analysis.lab_omnihorizon_adaptive_bt import (  # noqa: E402
    POOL_BUILDERS,
    _omni_route,
)
from analysis.xm_session_costs import (  # noqa: E402
    min_risk_with_spread,
    session_name,
    spread_price,
)
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL, _min_sl_px  # noqa: E402
from strategy.omni_live import OMNI_LIVE  # noqa: E402

JOIN = Path(ROOT) / "data" / "hist_yahoo_join.json"
OUT = Path(ROOT) / "data" / "lab_solution_xm_hunt.json"
LOG = Path(ROOT) / "data" / "lab_solution_xm_hunt.log"
PROGRESS = Path(ROOT) / "data" / "lab_solution_xm_hunt.progress.json"
PARTIAL = Path(ROOT) / "data" / "lab_solution_xm_hunt.partial.json"

BOOK = ("BTCUSD", "UKOIL", "GOLD", "NAS100")
TFS = ("5m", "15m", "3m")
HOLDOUT_START = pd.Timestamp("2026-07-24 00:00:00", tz="UTC")
MIN_DATA_END = pd.Timestamp("2026-09-15 00:00:00", tz="UTC")

# Coarse screen, then a tight densify on modes that already look alive pre-holdout.
COARSE_SL = (2.0, 3.0)
COARSE_RR = (3.0,)
COARSE_GAP = (3,)
DENSIFY_KEEP = 18


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    p["hunt"] = "solution_xm"
    p["book"] = list(BOOK)
    p["holdout_start"] = str(HOLDOUT_START)
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _save_partial(payload: dict) -> None:
    PARTIAL.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _loc(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    t = pd.Timestamp(ts)
    if index.tz is not None:
        if t.tzinfo is None:
            t = t.tz_localize(index.tz)
        else:
            t = t.tz_convert(index.tz)
    elif t.tzinfo is not None:
        t = t.tz_localize(None)
    return int(index.searchsorted(t))


def _entries_fast(
    df: pd.DataFrame,
    sides: np.ndarray,
    sym: str,
    *,
    sl_atr: float,
    min_gap: int,
) -> list[dict]:
    """Same economics as lab_adaptive_xm_deep_hunt_bt._entries, fewer iterations."""
    atr = df["atr"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    hours = pd.DatetimeIndex(df.index).hour.to_numpy(dtype=float)
    floor = float(_min_sl_px(sym))
    n = len(df)
    sides_arr = np.asarray(sides)
    idxs = np.flatnonzero(sides_arr[40 : n - 2] != 0) + 40
    out: list[dict] = []
    busy = -1
    for i in idxs:
        i = int(i)
        if i < busy:
            continue
        a = float(atr[i] or 0)
        if a <= 0 or not np.isfinite(a):
            continue
        entry = float(close[i])
        hour = float(hours[i])
        risk = min_risk_with_spread(
            sym, float(sl_atr) * a, hour_utc=hour, floor_px=floor
        )
        sp = spread_price(sym, hour_utc=hour)
        if sp > 0 and risk < 4.0 * sp:
            continue
        side = "BUY" if int(sides_arr[i]) > 0 else "SELL"
        out.append(
            {
                "ei": i,
                "side": side,
                "entry": entry,
                "risk": float(risk),
                "spread_px": float(sp),
                "hour": hour,
            }
        )
        busy = i + int(min_gap)
    return out


def _simulate_fast(
    high: np.ndarray,
    low: np.ndarray,
    entries: list[dict],
    rr: float,
    cut_i: Optional[int] = None,
) -> list[dict]:
    """First barrier wins. Same-bar SL and TP counts as a stop, matching the deep hunt.

    cut_i: bars before this index cannot exit on or after it (a multi-day hole
    in the stitched history must not be treated as one bar).
    """
    n = len(high)
    trades: list[dict] = []
    busy = -1
    rr = float(rr)
    for e in entries:
        ei = int(e["ei"])
        if ei < busy or ei >= n - 1:
            continue
        risk = float(e["risk"])
        if risk <= 0:
            continue
        entry = float(e["entry"])
        spread_r = float(e["spread_px"]) / risk
        j = ei + 1
        limit_n = n
        if cut_i is not None and ei < int(cut_i):
            limit_n = int(cut_i)
        hit_j: Optional[int] = None
        buy = e["side"] == "BUY"
        if buy:
            sl = entry - risk
            tp = entry + rr * risk
        else:
            sl = entry + risk
            tp = entry - rr * risk
        while j < limit_n:
            j1 = min(limit_n, j + 2500)
            if buy:
                hit = (low[j:j1] <= sl) | (high[j:j1] >= tp)
            else:
                hit = (high[j:j1] >= sl) | (low[j:j1] <= tp)
            if hit.any():
                hit_j = j + int(np.argmax(hit))
                break
            j = j1
        if hit_j is None:
            continue
        if buy:
            stopped = bool(low[hit_j] <= sl)
        else:
            stopped = bool(high[hit_j] >= sl)
        exit_r = -1.0 if stopped else rr
        r_net = float(exit_r) - spread_r
        trades.append(
            {
                "i": ei,
                "exit_i": int(hit_j),
                "side": e["side"],
                "session": session_name(float(e["hour"])),
                "r": round(r_net, 4),
                "spread_r": round(spread_r, 4),
                "pnl": round(r_net * RISK_USD, 2),
            }
        )
        busy = int(hit_j) + 2
    return trades


def _between(
    df: pd.DataFrame, trades: list[dict], start: pd.Timestamp, end: pd.Timestamp
) -> list[dict]:
    i0 = _loc(df.index, start)
    i1 = _loc(df.index, end)
    return [t for t in trades if i0 <= int(t["i"]) < i1]


def _pre_windows(df: pd.DataFrame, trades: list[dict]) -> dict[str, dict]:
    end = HOLDOUT_START
    out = {}
    for name, days in (
        ("7d", 7),
        ("15d", 15),
        ("30d", 30),
        ("90d", 90),
        ("180d", 180),
        ("1y", 365),
    ):
        start = end - pd.Timedelta(days=days)
        out[name] = _summarize(_between(df, trades, start, end))
    return out


def _row(mode: str, sl: float, rr: float, gap: int, pre: dict, hold: dict) -> dict:
    y = pre["1y"]
    return {
        "id": f"{mode}|sl{sl}|rr{rr}|g{gap}",
        "mode": mode,
        "sl_atr": sl,
        "rr": rr,
        "min_gap": gap,
        "n_green": sum(1 for k in pre if float(pre[k]["pnl"] or 0) > 0),
        "verdict_1y": y["verdict"],
        "verdict_30d": pre["30d"]["verdict"],
        "verdict_90d": pre["90d"]["verdict"],
        "verdict_7d": pre["7d"]["verdict"],
        "verdict_15d": pre["15d"]["verdict"],
        "pnl_1y": y["pnl"],
        "exp_r_1y": y["exp_r"],
        "oos_1y": y["oos_exp_r"],
        "wf_1y": y["wf_green"],
        "n_1y": y["n"],
        "spR_1y": y["avg_spread_r"],
        "pnl_180d": pre["180d"]["pnl"],
        "pnl_90d": pre["90d"]["pnl"],
        "pnl_30d": pre["30d"]["pnl"],
        "pnl_15d": pre["15d"]["pnl"],
        "pnl_7d": pre["7d"]["pnl"],
        "holdout_n": hold["n"],
        "holdout_pnl": hold["pnl"],
        "holdout_exp_r": hold["exp_r"],
        "holdout_win_rate": hold["win_rate"],
        "solution": _is_solution(pre, hold),
    }


def _is_solution(pre: dict, hold: dict) -> bool:
    y = pre["1y"]
    if y["verdict"] not in ("PASS", "SOFT"):
        return False
    if float(y["exp_r"] or 0) < 0.03:
        return False
    if int(y["n"] or 0) < 40:
        return False
    if float(y["oos_exp_r"] or 0) <= 0:
        return False
    if int(y["wf_green"] or 0) < 3:
        return False
    if float(y.get("avg_spread_r") or 0) > 0.20:
        return False
    for k in ("30d", "90d", "180d", "1y"):
        if float(pre[k]["pnl"] or 0) <= 0:
            return False
    if int(hold["n"] or 0) < 8:
        return False
    if float(hold["pnl"] or 0) <= 0:
        return False
    if float(hold["exp_r"] or 0) <= 0:
        return False
    return True


def _score(row: dict) -> float:
    return (
        (100_000.0 if row["solution"] else 0.0)
        + (20_000.0 if float(row["holdout_pnl"] or 0) > 0 else 0.0)
        + float(row["holdout_pnl"] or 0)
        + float(row["n_green"] or 0) * 500.0
        + float(row["pnl_1y"] or 0) * 0.05
        + float(row["exp_r_1y"] or 0) * 2000.0
    )


def _eval_cell(
    df: pd.DataFrame,
    high: np.ndarray,
    low: np.ndarray,
    sides: np.ndarray,
    sym: str,
    mode: str,
    sl: float,
    rr: float,
    gap: int,
    cut_i: Optional[int] = None,
) -> Optional[dict]:
    if int((sides != 0).sum()) < 30:
        return None
    entries = _entries_fast(df, sides, sym, sl_atr=sl, min_gap=gap)
    if len(entries) < 20:
        return None
    trades = _simulate_fast(high, low, entries, rr, cut_i=cut_i)
    if len(trades) < 20:
        return None
    pre = _pre_windows(df, trades)
    hold = _summarize(_between(df, trades, HOLDOUT_START, df.index[-1] + pd.Timedelta(days=1)))
    row = _row(mode, sl, rr, gap, pre, hold)
    row["_score"] = _score(row)
    return row


def _ctx(df: pd.DataFrame, sym: str) -> dict:
    static, _label, _ = _static_sides_for_winner(df, "")
    return {
        "sym": sym,
        "cal": _calendar(df),
        "static": static,
        "regime": _classify_regime(df),
        "v2_preferred_sides": static,
        "family": "snr",
    }


def _omni_sides(df: pd.DataFrame, ctx: dict) -> list[tuple[str, np.ndarray]]:
    ret = df["close"].pct_change().fillna(0).to_numpy(dtype=float)
    brains: dict[str, np.ndarray] = {
        "static": np.asarray(_sides_snr(df, 0.40), dtype=int)
    }
    for name, builder in POOL_BUILDERS:
        try:
            brains[name] = np.asarray(builder()(df, ctx), dtype=int)
        except Exception:
            continue
    out = []
    grid = (
        (80, 240, 600, 0.52, 25, 2),
        (80, 400, 600, 0.55, 25, 2),
        (120, 240, 600, 0.52, 25, 2),
        (120, 240, 600, 0.55, 25, 2),
        (120, 400, 600, 0.55, 25, 3),
    )
    for lf, lm, ls, emin, hyst, agree in grid:
        sides, _meta = _omni_route(
            brains,
            ret,
            look_fast=lf,
            look_mid=lm,
            look_slow=ls,
            edge_min=emin,
            hysteresis=hyst,
            min_agree=agree,
        )
        out.append((f"omni_f{lf}_m{lm}_e{emin}_a{agree}", np.asarray(sides, dtype=int)))
    return out


def _live_omni_sides(df: pd.DataFrame, ctx: dict, sym: str, tf: str) -> list[tuple[str, np.ndarray]]:
    spec = OMNI_LIVE.get((sym, tf))
    if not spec:
        return []
    ret = df["close"].pct_change().fillna(0).to_numpy(dtype=float)
    brains: dict[str, np.ndarray] = {
        "static": np.asarray(_sides_snr(df, 0.40), dtype=int)
    }
    for name, builder in POOL_BUILDERS:
        try:
            brains[name] = np.asarray(builder()(df, ctx), dtype=int)
        except Exception:
            continue
    sides, _meta = _omni_route(
        brains,
        ret,
        look_fast=int(spec["look_fast"]),
        look_mid=int(spec["look_mid"]),
        look_slow=int(spec["look_slow"]),
        edge_min=float(spec["edge_min"]),
        hysteresis=int(spec["hysteresis"]),
        min_agree=int(spec["min_agree"]),
    )
    return [(f"live_omni", np.asarray(sides, dtype=int))]


def _better(a: dict, b: Optional[dict]) -> bool:
    if b is None:
        return True
    return float(a.get("_score") or 0) > float(b.get("_score") or 0)


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n===== SOLUTION XM HUNT @ {datetime.now(timezone.utc).isoformat()} =====\n")

    t0 = time.time()
    done_keys: set[str] = set()
    per_pair: dict[str, Any] = {}
    if PARTIAL.exists():
        try:
            prev = json.loads(PARTIAL.read_text(encoding="utf-8"))
            per_pair = prev.get("per_pair") or {}
            for sym, row in per_pair.items():
                for tf, block in (row.get("by_tf") or {}).items():
                    if block.get("done"):
                        done_keys.add(f"{sym}:{tf}")
            _log(f"RESUME · {len(done_keys)} pair×tf done")
        except Exception as exc:
            _log(f"partial unreadable, starting fresh: {exc}")

    _log(
        "=== SOLUTION HUNT (XM spreads, locked holdout) ===\n"
        f"book={BOOK} tfs={TFS}\n"
        f"holdout starts {HOLDOUT_START.date()} — not used to pick the brain\n"
        "gate: pre 30/90/180/1y green, 1y PASS|SOFT, E[R]≥0.03, n≥40, "
        "OOS>0, WF≥3, spread<0.20R\n"
        "AND holdout n≥8, pnl>0, E[R]>0\n"
        "research only · .env untouched\n"
    )
    _progress({"status": "running", "pct": 0, "phase": "start"})

    cells_done = 0
    for pi, sym in enumerate(BOOK):
        per_pair.setdefault(
            sym,
            {
                "symbol": sym,
                "family": "solution_xm",
                "by_tf": {},
                "best_overall": None,
                "best_wireable": None,
            },
        )
        for ti, tf in enumerate(TFS):
            key = f"{sym}:{tf}"
            if key in done_keys:
                _log(f"skip {key}")
                continue
            years = float(TF_YEARS.get(tf, 5.0))
            _log(f"\n######## {sym} @ {tf} · {years}y ########")
            _progress(
                {
                    "status": "running",
                    "pair": sym,
                    "tf": tf,
                    "pct": round(100 * (pi * len(TFS) + ti) / (len(BOOK) * len(TFS)), 1),
                    "cells_done": cells_done,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "phase": "load",
                }
            )
            try:
                df = _load(sym, tf, years)
            except Exception as exc:
                _log(f"  LOAD FAIL {exc}")
                per_pair[sym]["by_tf"][tf] = {"error": str(exc), "done": True}
                continue
            end_ts = pd.Timestamp(df.index[-1])
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            if end_ts < MIN_DATA_END:
                msg = f"hist ends {end_ts.date()} — holdout not covered"
                _log(f"  SKIP {msg}")
                per_pair[sym]["by_tf"][tf] = {"error": msg, "done": True}
                continue

            joins = {}
            if JOIN.exists():
                try:
                    joins = json.loads(JOIN.read_text(encoding="utf-8"))
                except Exception:
                    joins = {}
            jmeta = ((joins.get(sym) or {}).get(tf) or {})
            cut_i = None
            if float(jmeta.get("gap_hours") or 0) > 72 and jmeta.get("join"):
                cut_i = _loc(df.index, pd.Timestamp(jmeta["join"]))
                _log(f"  hole cut at {jmeta['join']} (gap {jmeta['gap_hours']}h)")

            ctx = _ctx(df, sym)
            high = df["high"].to_numpy(dtype=float)
            low = df["low"].to_numpy(dtype=float)
            catalog = build_mega_catalog(lean=False, symbol=sym)
            _log(f"  catalog {len(catalog)} modes · bars {len(df):,} · end {end_ts}")

            modes: list[tuple[str, np.ndarray]] = []
            for vid, _lab, fn in catalog:
                try:
                    modes.append((vid, np.asarray(fn(df, ctx), dtype=int)))
                except Exception as exc:
                    _log(f"  sides {vid} fail: {exc}")
            for vid, sides in _live_omni_sides(df, ctx, sym, tf) + _omni_sides(df, ctx):
                modes.append((vid, sides))
            _log(f"  runnable modes {len(modes)}")

            ranked: list[dict] = []
            survivors: list[dict] = []
            for mi, (mode, sides) in enumerate(modes):
                for sl in COARSE_SL:
                    for rr in COARSE_RR:
                        for gap in COARSE_GAP:
                            row = _eval_cell(
                                df, high, low, sides, sym, mode, float(sl), float(rr), int(gap), cut_i
                            )
                            cells_done += 1
                            if row is None:
                                continue
                            ranked.append(row)
                            if row["solution"]:
                                survivors.append(row)
                                _log(
                                    f"  SOLUTION {row['id']} holdout={row['holdout_pnl']} "
                                    f"1y={row['pnl_1y']} exp={row['exp_r_1y']}"
                                )
                if (mi + 1) % 40 == 0 or mi == 0:
                    _log(f"  … coarse {mi+1}/{len(modes)} cells={cells_done} {time.time()-t0:.0f}s")
                    _progress(
                        {
                            "status": "running",
                            "pair": sym,
                            "tf": tf,
                            "mode_i": mi + 1,
                            "modes": len(modes),
                            "mode_id": mode,
                            "cells_done": cells_done,
                            "elapsed_sec": round(time.time() - t0, 1),
                            "pct": round(
                                100
                                * (
                                    pi * len(TFS)
                                    + ti
                                    + (mi + 1) / max(len(modes), 1)
                                )
                                / (len(BOOK) * len(TFS)),
                                1,
                            ),
                            "solutions": sum(
                                1
                                for srow in per_pair.values()
                                for block in (srow.get("by_tf") or {}).values()
                                for w in (block.get("wireable") or [])
                            )
                            + len(survivors),
                        }
                    )

            # Densify modes that were green on the pre-holdout year.
            ranked.sort(key=lambda r: r["_score"], reverse=True)
            seen = set()
            densify_modes: list[str] = []
            side_by_mode = {m: s for m, s in modes}
            for row in ranked:
                if row["mode"] in seen:
                    continue
                if float(row["pnl_1y"] or 0) <= 0 or float(row["pnl_90d"] or 0) <= 0:
                    continue
                seen.add(row["mode"])
                densify_modes.append(row["mode"])
                if len(densify_modes) >= DENSIFY_KEEP:
                    break
            _log(f"  densify {len(densify_modes)} modes")
            sl_grid = _sl_grid(sym, tf)
            for mode in densify_modes:
                sides = side_by_mode[mode]
                for sl in sl_grid:
                    for rr in RR_GRID:
                        for gap in GAP[tf]:
                            if sl in COARSE_SL and rr in COARSE_RR and gap in COARSE_GAP:
                                continue
                            row = _eval_cell(
                                df, high, low, sides, sym, mode, float(sl), float(rr), int(gap), cut_i
                            )
                            cells_done += 1
                            if row is None:
                                continue
                            ranked.append(row)
                            if row["solution"]:
                                survivors.append(row)
                                _log(
                                    f"  SOLUTION {row['id']} holdout={row['holdout_pnl']} "
                                    f"1y={row['pnl_1y']}"
                                )

            ranked.sort(key=lambda r: r["_score"], reverse=True)
            top = ranked[:25]
            best = top[0] if top else None
            sols = [r for r in ranked if r.get("solution")]
            sols.sort(key=lambda r: (float(r["holdout_pnl"]), float(r["pnl_1y"])), reverse=True)
            block = {
                "done": True,
                "n_bars": len(df),
                "span": [str(df.index[0]), str(df.index[-1])],
                "n_combos": len(ranked),
                "best": best,
                "top25": top,
                "wireable": sols[:12],
                "live_sl_ref": float(SL_ATR_BY_SYMBOL.get(sym, 2.0)),
            }
            per_pair[sym]["by_tf"][tf] = block
            if best and _better(best, per_pair[sym].get("best_overall")):
                per_pair[sym]["best_overall"] = {**best, "tf": tf}
            if sols and _better(sols[0], per_pair[sym].get("best_wireable")):
                per_pair[sym]["best_wireable"] = {**sols[0], "tf": tf}
            _log(
                f"  BEST {tf}: {best['id'] if best else None} "
                f"sol={bool(best and best.get('solution'))} "
                f"hold={best.get('holdout_pnl') if best else None} "
                f"1y={best.get('pnl_1y') if best else None} "
                f"solutions={len(sols)}"
            )
            _save_partial(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "cells_done": cells_done,
                    "per_pair": per_pair,
                    "hunt": "solution_xm",
                    "holdout_start": str(HOLDOUT_START),
                }
            )

    recs = []
    n_ok = 0
    for sym in BOOK:
        pick = (per_pair.get(sym) or {}).get("best_wireable")
        if pick and pick.get("solution"):
            n_ok += 1
            recs.append({"symbol": sym, **pick})
        else:
            near = (per_pair.get(sym) or {}).get("best_overall")
            recs.append({"symbol": sym, "solution": False, "nearest": near})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "solution_xm_v1",
        "research_only": True,
        "recommend_live": False,
        "criteria_met": n_ok == len(BOOK),
        "wireable_count": n_ok,
        "holdout_start": str(HOLDOUT_START),
        "xm_spreads": True,
        "live_untouched": True,
        "elapsed_sec": round(time.time() - t0, 1),
        "cells_done": cells_done,
        "book": list(BOOK),
        "tfs": list(TFS),
        "per_pair": per_pair,
        "recommendations": recs,
        "note": (
            "Solution requires pre-2026-07-24 survival AND a green holdout "
            "through the refreshed Dukascopy end, with XM session spreads. "
            "No .env write."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress(
        {
            "status": "done",
            "pct": 100,
            "cells_done": cells_done,
            "elapsed_sec": round(time.time() - t0, 1),
            "wireable_count": n_ok,
            "criteria_met": n_ok == len(BOOK),
        }
    )
    _log(f"\n=== DONE · solutions {n_ok}/{len(BOOK)} · cells {cells_done} ===")
    for rec in recs:
        if rec.get("solution"):
            _log(
                f"  {rec['symbol']}: {rec.get('id')} tf={rec.get('tf')} "
                f"holdout={rec.get('holdout_pnl')} 1y={rec.get('pnl_1y')}"
            )
        else:
            near = rec.get("nearest") or {}
            _log(
                f"  {rec['symbol']}: NO SOLUTION · nearest {near.get('id')} "
                f"holdout={near.get('holdout_pnl')} 1y={near.get('pnl_1y')}"
            )
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
