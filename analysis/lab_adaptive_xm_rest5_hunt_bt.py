#!/usr/bin/env python3
"""REST-5 deep hunt — AUDUSD · USDCHF · ETHUSD · NAS100 · USDJPY.

Goal: find all-horizon survivors for the pairs that missed omni live wire,
using *different* adaptive brains than the prior deep/omni catalogs.

  · New routers: dual-agree, hurst gate, London/NY-only, Off-flat FX,
    omni-style multi-brain with fresh param grids, session-strict FX
  · Score by n_green (7d→1y) first — not 1y-only crowns
  · XM session spreads · multi-TF · resume-safe
  · Does NOT touch live GOLD/BTCUSD/UKOIL wiring

Outputs:
  data/lab_adaptive_xm_rest5_hunt.json
  data/lab_adaptive_xm_rest5_hunt.progress.json
  data/lab_adaptive_xm_rest5_hunt.partial.json
  data/lab_adaptive_xm_rest5_hunt.log
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.lab_adaptive_minute_day_v3_bt import _calendar  # noqa: E402
from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    _classify_regime,
    _sides_bimod,
    _sides_kalman,
    _sides_perm,
    _sides_snr,
    _static_sides_for_winner,
    _window_trades,
)
from analysis.lab_adaptive_regime_v2_bt import _family, _gated_sides, _switch_sides  # noqa: E402
from analysis.lab_adaptive_xm_deep_hunt_bt import (  # noqa: E402
    GAP_GRID_BY_TF,
    RISK_USD,
    RR_GRID,
    SL_GRID_BY_TF,
    TF_YEARS,
    TFS,
    WINDOWS,
    _entries,
    _load,
    _rank,
    _simulate,
    _summarize,
    mk_circadian,
    mk_entropy,
    mk_fourier_trend,
    mk_liq_only,
    mk_ou_fade,
    mk_rolling_edge_gate,
    mk_session_book_switch,
    mk_session_mix,
    mk_snr,
    mk_vol_clock,
    mk_volmom,
    mk_weekday_regime,
)
from analysis.lab_omnihorizon_adaptive_bt import _omni_route  # noqa: E402
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL  # noqa: E402

REST_BOOK = ("AUDUSD", "USDCHF", "ETHUSD", "NAS100", "USDJPY")

OUT = Path(ROOT) / "data" / "lab_adaptive_xm_rest5_hunt.json"
LOG = Path(ROOT) / "data" / "lab_adaptive_xm_rest5_hunt.log"
PROGRESS = Path(ROOT) / "data" / "lab_adaptive_xm_rest5_hunt.progress.json"
PARTIAL = Path(ROOT) / "data" / "lab_adaptive_xm_rest5_hunt.partial.json"
HUNT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"

SideFn = Callable[[pd.DataFrame, dict[str, Any]], np.ndarray]

# FX-friendly wider stops (XM pip risk)
SL_GRID_FX = {
    "1m": (2.5, 3.0),
    "3m": (2.5, 3.0, 3.5),
    "5m": (2.0, 2.5, 3.0, 3.5),
    "15m": (2.0, 2.5, 3.0, 3.5),
}
FX_SYMS = frozenset({"AUDUSD", "USDCHF", "USDJPY"})


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    p["hunt"] = "rest5"
    p["book"] = list(REST_BOOK)
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _save_partial(payload: dict) -> None:
    PARTIAL.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ── New adaptive brains (not in prior deep catalog clones) ───────────────────


def mk_dual_agree(a: SideFn, b: SideFn) -> SideFn:
    """Only trade when two brains agree on sign."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sa = a(df, ctx)
        sb = b(df, ctx)
        out = np.zeros(len(df), dtype=int)
        m = (sa != 0) & (sa == sb)
        out[m] = sa[m]
        return out

    return fn


def mk_hurst_gate(base: SideFn, *, mode: str = "trend", thr: float = 0.55) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sides = base(df, ctx)
        H = df["hurst"].to_numpy(dtype=float) if "hurst" in df.columns else np.full(len(df), 0.5)
        out = sides.copy()
        if mode == "trend":
            out[H < thr] = 0
        else:
            out[H > (1.0 - thr + 0.05)] = 0
            # mean-rev: prefer low hurst
            out[H > thr] = 0
        return out

    return fn


def mk_london_ny_only(base: SideFn) -> SideFn:
    """Liquid FX hours only: 7–21 UTC."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sides = base(df, ctx)
        hour = ctx["cal"]["hour"]
        out = sides.copy()
        out[(hour < 7) | (hour >= 21)] = 0
        return out

    return fn


def mk_off_flat(base: SideFn) -> SideFn:
    """Force flat 21–24 UTC (XM Off blowouts)."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sides = base(df, ctx)
        hour = ctx["cal"]["hour"]
        out = sides.copy()
        out[hour >= 21] = 0
        return out

    return fn


def mk_asia_fade_london_trend() -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        cal = ctx["cal"]
        k = _sides_kalman(df, 2.0)
        s = ctx["static"]
        snr = _sides_snr(df, 0.45)
        out = np.zeros(len(df), dtype=int)
        asia = cal["sess"] == 0
        lon = cal["sess"] == 1
        ny = (cal["sess"] == 2) | (cal["sess"] == 3)
        out[asia] = k[asia]
        out[lon] = np.where(snr[lon] != 0, snr[lon], s[lon])
        out[ny] = s[ny]
        out[cal["hour"] >= 21] = 0
        return out

    return fn


def mk_omni_pool(
    *,
    look_fast: int,
    look_mid: int,
    look_slow: int,
    edge_min: float,
    hysteresis: int,
    min_agree: int,
    pool: str = "core",
) -> SideFn:
    """Fresh omni routers — different looks / edges than prior all-six hunt."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        builders = {
            "sm": mk_session_mix(0.22, 0.35),
            "sms": mk_session_mix(0.32, 0.42),
            "cir": mk_circadian(1.7, 0.6),
            "cir_t": mk_circadian(2.4, 0.4),
            "ent": mk_entropy(0.72, 0.22),
            "ou": mk_ou_fade(1.8),
            "snr": mk_snr(0.45),
            "static": lambda d, c: c["static"],
        }
        if pool == "core":
            names = ("sm", "sms", "cir", "static")
        elif pool == "fade":
            names = ("cir", "cir_t", "ou", "sms")
        else:  # wide
            names = ("sm", "sms", "cir", "ent", "snr", "static")
        brain_sides = {nm: builders[nm](df, ctx) for nm in names}
        ret = df["close"].pct_change().fillna(0).to_numpy(dtype=float)
        sides, _meta = _omni_route(
            brain_sides,
            ret,
            look_fast=look_fast,
            look_mid=look_mid,
            look_slow=look_slow,
            edge_min=edge_min,
            hysteresis=hysteresis,
            min_agree=min_agree,
        )
        return sides

    return fn


def mk_tod_blend(z0: float = 2.0) -> SideFn:
    """Kalman fade strength modulated by TOD sinusoid + weekday."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        cal = ctx["cal"]
        kz = df["kalman_z"].to_numpy(dtype=float)
        thr = z0 + 0.7 * cal["sin_tod"] + 0.3 * cal["cos_week"]
        out = np.zeros(len(df), dtype=int)
        out[kz <= -thr] = 1
        out[kz >= thr] = -1
        return out

    return fn


def _build_rest_variants(*, lean: bool = False) -> list[tuple[str, str, SideFn]]:
    out: list[tuple[str, str, SideFn]] = []

    # Compact prior winners (anchors) + NEW brains
    anchors = [
        ("session_mix_t0.28_o0.38", "session_mix", mk_session_mix(0.28, 0.38)),
        ("session_mix_t0.32_o0.42", "session_mix", mk_session_mix(0.32, 0.42)),
        ("circadian_z2.2_s0.5", "circadian", mk_circadian(2.2, 0.5)),
        ("circadian_z1.7_s0.6", "circadian", mk_circadian(1.7, 0.6)),
        ("entropy_p0.72_s0.22", "entropy", mk_entropy(0.72, 0.22)),
        ("snr_0.45", "snr", mk_snr(0.45)),
        ("vol_clock", "vol_clock", mk_vol_clock()),
        ("weekday_regime_prior", "weekday", mk_weekday_regime()),
        ("session_book_switch", "session_book", mk_session_book_switch()),
        ("ou_fade_z1.8", "ou_fade", mk_ou_fade(1.8)),
        ("fourier_lf0.55_s0.35", "fourier", mk_fourier_trend(0.55, 0.35)),
        ("volmom_t0.5_l20", "volmom", mk_volmom(0.5, 20)),
    ]
    out.extend(anchors)

    # Dual-agree
    sm = mk_session_mix(0.28, 0.38)
    cir = mk_circadian(2.0, 0.6)
    ent = mk_entropy(0.72, 0.18)
    out.append(("agree_sm_cir", "agree", mk_dual_agree(sm, cir)))
    out.append(("agree_sm_ent", "agree", mk_dual_agree(sm, ent)))
    out.append(("agree_cir_ent", "agree", mk_dual_agree(cir, ent)))

    # Hurst gates
    out.append(("hurst_trend_sm", "hurst", mk_hurst_gate(sm, mode="trend", thr=0.52)))
    out.append(("hurst_mr_cir", "hurst", mk_hurst_gate(cir, mode="mr", thr=0.48)))

    # Session filters
    out.append(("lonny_sm", "liquid_hours", mk_london_ny_only(sm)))
    out.append(("lonny_cir", "liquid_hours", mk_london_ny_only(cir)))
    out.append(("offflat_sm", "off_flat", mk_off_flat(sm)))
    out.append(("offflat_cir", "off_flat", mk_off_flat(cir)))
    out.append(("asia_fade_lon_trend", "session_book", mk_asia_fade_london_trend()))
    out.append(("tod_blend_z2.0", "circadian", mk_tod_blend(2.0)))
    out.append(("tod_blend_z2.4", "circadian", mk_tod_blend(2.4)))

    # Edge gates on new bases
    out.append(
        (
            "agree_sm_cir_edge",
            "edgegate",
            mk_rolling_edge_gate(mk_dual_agree(sm, cir), 500, 0.02),
        )
    )
    out.append(
        (
            "lonny_sm_edge",
            "edgegate",
            mk_rolling_edge_gate(mk_london_ny_only(sm), 400, 0.0),
        )
    )

    if lean:
        # lean omni grid for 1m
        for lf, em, ag in ((80, 0.50, 2), (120, 0.55, 2)):
            out.append(
                (
                    f"omni_core_f{lf}_e{em}_a{ag}",
                    "omni",
                    mk_omni_pool(
                        look_fast=lf,
                        look_mid=240,
                        look_slow=500,
                        edge_min=em,
                        hysteresis=20,
                        min_agree=ag,
                        pool="core",
                    ),
                )
            )
        return out

    # Full omni / router grids — different from prior omni survivors (pruned)
    omni_grid = [
        ("core", 80, 240, 500, 0.50, 20, 2),
        ("core", 120, 320, 600, 0.52, 25, 2),
        ("core", 100, 200, 480, 0.55, 15, 2),
        ("core", 60, 240, 720, 0.48, 30, 1),
        ("fade", 80, 240, 500, 0.52, 20, 2),
        ("fade", 120, 320, 600, 0.55, 25, 2),
        ("fade", 100, 200, 480, 0.50, 15, 1),
        ("wide", 100, 240, 600, 0.55, 25, 2),
        ("wide", 80, 320, 720, 0.52, 20, 2),
        ("wide", 160, 320, 600, 0.58, 30, 2),
    ]
    for pool, lf, lm, ls, em, hy, ag in omni_grid:
        out.append(
            (
                f"omni_{pool}_f{lf}_m{lm}_s{ls}_e{em}_h{hy}_a{ag}",
                "omni",
                mk_omni_pool(
                    look_fast=lf,
                    look_mid=lm,
                    look_slow=ls,
                    edge_min=em,
                    hysteresis=hy,
                    min_agree=ag,
                    pool=pool,
                ),
            )
        )

    # Extra parametric session/circadian not heavily used before
    for tb, to in ((0.25, 0.40), (0.35, 0.45), (0.40, 0.50)):
        out.append((f"session_mix_t{tb:.2f}_o{to:.2f}", "session_mix", mk_session_mix(tb, to)))
    for z0, zs in ((1.8, 0.9), (2.4, 0.3), (2.8, 0.5)):
        out.append((f"circadian_z{z0:.1f}_s{zs:.1f}", "circadian", mk_circadian(z0, zs)))
    for p0, sn in ((0.68, 0.25), (0.80, 0.15)):
        out.append((f"entropy_p{p0:.2f}_s{sn:.2f}", "entropy", mk_entropy(p0, sn)))

    return out


def _n_green(wins: dict[str, dict]) -> int:
    n = 0
    for k in ("7d", "15d", "30d", "90d", "180d", "1y"):
        w = wins.get(k) or {}
        try:
            if float(w.get("pnl") or 0) > 0:
                n += 1
        except Exception:
            pass
    return n


def _cell_score(wins: dict[str, dict]) -> float:
    """All-horizon first; then quality on 1y/90d."""
    ng = _n_green(wins)

    def _q(wname: str, w: float) -> float:
        sm = wins.get(wname) or {}
        v = sm.get("verdict")
        rank = 2.0 if v == "PASS" else (1.0 if v == "SOFT" else 0.0)
        return w * (
            rank * 1000.0
            + float(sm.get("pnl") or 0) * 0.01
            + float(sm.get("exp_r") or 0) * 50.0
            + float(sm.get("oos_exp_r") or 0) * 40.0
            + float(sm.get("wf_green") or 0) * 20.0
        )

    return (
        ng * 10_000.0
        + _q("1y", 1.0)
        + _q("90d", 0.5)
        + _q("30d", 0.35)
        + _q("15d", 0.25)
        + _q("7d", 0.15)
    )


def _top_row(c: dict) -> dict:
    w = c["windows"]
    return {
        "id": c["id"],
        "mode": c["mode"],
        "sl_atr": c["sl_atr"],
        "rr": c["rr"],
        "min_gap": c["min_gap"],
        "n_green": _n_green(w),
        "all_six": _n_green(w) >= 6,
        "verdict_1y": w["1y"]["verdict"],
        "verdict_30d": w["30d"]["verdict"],
        "verdict_90d": w["90d"]["verdict"],
        "verdict_7d": w["7d"]["verdict"],
        "verdict_15d": w["15d"]["verdict"],
        "pnl_1y": w["1y"]["pnl"],
        "exp_r_1y": w["1y"]["exp_r"],
        "oos_1y": w["1y"]["oos_exp_r"],
        "wf_1y": w["1y"]["wf_green"],
        "n_1y": w["1y"]["n"],
        "spR_1y": w["1y"]["avg_spread_r"],
        "pnl_180d": w["180d"]["pnl"],
        "pnl_90d": w["90d"]["pnl"],
        "pnl_30d": w["30d"]["pnl"],
        "pnl_15d": w["15d"]["pnl"],
        "pnl_7d": w["7d"]["pnl"],
    }


def main() -> int:
    LOG.write_text("", encoding="utf-8") if not LOG.exists() else None
    t0 = time.time()
    hunt = json.loads(HUNT.read_text(encoding="utf-8")) if HUNT.exists() else {}
    per_hunt = hunt.get("per_pair") or {}
    variants_full = _build_rest_variants(lean=False)
    variants_lean = _build_rest_variants(lean=True)

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
            _log(f"RESUME · {len(done_keys)} pair×tf already done")
        except Exception:
            pass

    est = 0
    for tf in TFS:
        nv = len(variants_lean if tf == "1m" else variants_full) + 3
        for sym in REST_BOOK:
            slg = SL_GRID_FX[tf] if sym in FX_SYMS else SL_GRID_BY_TF[tf]
            est += nv * len(slg) * len(RR_GRID) * len(GAP_GRID_BY_TF[tf])

    _log(
        f"=== REST-5 XM adaptive hunt (Nebula) ===\n"
        f"book={REST_BOOK}\n"
        f"live untouched: GOLD BTCUSD UKOIL\n"
        f"tfs={TFS} full_variants={len(variants_full)} lean_1m={len(variants_lean)}\n"
        f"est_cells≈{est:,} · score=n_green(7d→1y) first · XM spreads ON\n"
    )
    _progress({"status": "running", "pct": 0, "cells_est": est, "modes": len(variants_full)})

    cells_done = 0
    for pi, sym in enumerate(REST_BOOK):
        wmeta = (per_hunt.get(sym) or {}).get("winner") or {}
        wid = str(wmeta.get("id") or "")
        fam = _family(wid)
        per_pair.setdefault(
            sym,
            {
                "symbol": sym,
                "static_id": wid,
                "family": fam,
                "by_tf": {},
                "best_overall": None,
            },
        )

        for tf in TFS:
            key = f"{sym}:{tf}"
            if key in done_keys:
                _log(f"skip {key} (resume)")
                continue
            years = TF_YEARS[tf]
            _log(f"\n######## {sym} @ {tf} · {years}y ########")
            _progress(
                {
                    "status": "running",
                    "pct": round(
                        100 * (pi * len(TFS) + list(TFS).index(tf)) / (len(REST_BOOK) * len(TFS)),
                        1,
                    ),
                    "pair": sym,
                    "tf": tf,
                    "cells_done": cells_done,
                    "cells_est": est,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "modes": len(variants_lean if tf == "1m" else variants_full),
                }
            )
            try:
                df = _load(sym, tf, years)
            except Exception as exc:
                _log(f"  LOAD FAIL: {exc}")
                per_pair[sym]["by_tf"][tf] = {"error": str(exc), "done": True}
                _save_partial({"per_pair": per_pair, "elapsed_sec": round(time.time() - t0, 1)})
                continue

            static, label, _ = _static_sides_for_winner(df, wid)
            regime = _classify_regime(df)
            cal = _calendar(df)
            ctx = {
                "sym": sym,
                "static": static,
                "regime": regime,
                "cal": cal,
                "v2_preferred_sides": static,
                "family": fam,
                "label": label,
            }
            live_sl = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))
            variants = variants_lean if tf == "1m" else variants_full
            alt_trend = (
                _sides_bimod(df, 0.55)
                if sym in ("ETHUSD", "BTCUSD", "NAS100")
                else _sides_snr(df, 0.40)
            )
            alt_fade = _sides_kalman(df, 2.2)
            mode_ids: list[tuple[str, str, Any]] = [
                ("static", "static winner", static),
                ("gated", "gated", _gated_sides(static, regime, fam)),
                (
                    "switch",
                    "switch",
                    _switch_sides(static, regime, fam, alt_trend, alt_fade),
                ),
            ]
            for vid, vlab, vfn in variants:
                try:
                    mode_ids.append((vid, vlab, vfn(df, ctx)))
                except Exception as exc:
                    _log(f"  sides {vid} fail: {exc}")

            sl_grid = SL_GRID_FX[tf] if sym in FX_SYMS else SL_GRID_BY_TF[tf]
            gap_grid = GAP_GRID_BY_TF[tf]
            results: list[dict] = []

            for mi, (mode, _desc, sides) in enumerate(mode_ids):
                if sides is None:
                    continue
                try:
                    sides_arr = np.asarray(sides, dtype=int)
                except Exception as exc:
                    _log(f"  skip {mode}: {exc}")
                    continue
                if int((sides_arr != 0).sum()) < 20:
                    continue
                for sl_atr in sl_grid:
                    for gap in gap_grid:
                        entries = _entries(
                            df, sides_arr, sym, sl_atr=float(sl_atr), min_gap=int(gap)
                        )
                        for rr in RR_GRID:
                            trades = _simulate(df, entries, float(rr), sym)
                            wins = {}
                            for wname, days in WINDOWS:
                                if days is None:
                                    tw = trades
                                else:
                                    tw = _window_trades(df, trades, days)
                                wins[wname] = _summarize(tw)
                            cell = {
                                "id": f"{mode}|sl{sl_atr}|rr{rr}|g{gap}",
                                "mode": mode,
                                "sl_atr": sl_atr,
                                "rr": rr,
                                "min_gap": gap,
                                "tf": tf,
                                "windows": wins,
                                "n_entries": len(entries),
                            }
                            cell["_score"] = _cell_score(wins)
                            results.append(cell)
                            cells_done += 1
                if (mi + 1) % 8 == 0 or mi == 0:
                    _log(
                        f"  … {mi+1}/{len(mode_ids)} modes · cells={cells_done} · "
                        f"{time.time()-t0:.0f}s · {mode}"
                    )
                    _progress(
                        {
                            "status": "running",
                            "pair": sym,
                            "tf": tf,
                            "mode_i": mi + 1,
                            "modes": len(mode_ids),
                            "mode_id": mode,
                            "mode_family": str(mode).split("_")[0],
                            "cells_done": cells_done,
                            "cells_est": est,
                            "elapsed_sec": round(time.time() - t0, 1),
                            "pct": round(
                                100
                                * (
                                    pi * len(TFS)
                                    + list(TFS).index(tf)
                                    + (mi + 1) / max(len(mode_ids), 1)
                                )
                                / (len(REST_BOOK) * len(TFS)),
                                1,
                            ),
                        }
                    )

            results.sort(key=lambda c: c["_score"], reverse=True)
            top = [_top_row(c) for c in results[:30]]
            best = top[0] if top else None
            all_six = [t for t in top if t.get("all_six")]
            per_pair[sym]["by_tf"][tf] = {
                "done": True,
                "n_bars": len(df),
                "span": [str(df.index[0]), str(df.index[-1])],
                "n_combos": len(results),
                "best": best,
                "top25": top[:25],
                "all_six": all_six[:10],
                "live_sl_ref": live_sl,
            }
            cand = best
            cur = per_pair[sym].get("best_overall")
            if cand and (
                cur is None
                or (cand.get("n_green") or 0) > (cur.get("n_green") or 0)
                or (
                    (cand.get("n_green") or 0) == (cur.get("n_green") or 0)
                    and float(cand.get("pnl_1y") or 0) > float(cur.get("pnl_1y") or 0)
                )
            ):
                per_pair[sym]["best_overall"] = {**cand, "tf": tf}

            _log(
                f"  BEST {tf}: {best['id'] if best else None} "
                f"green={best.get('n_green') if best else 0}/6 "
                f"1y ${best['pnl_1y'] if best else 0:,.0f} "
                f"all_six_in_top={len(all_six)}"
            )
            _save_partial(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "cells_done": cells_done,
                    "per_pair": per_pair,
                    "hunt": "rest5",
                }
            )

    survivors = []
    recs = []
    for sym in REST_BOOK:
        row = per_pair.get(sym) or {}
        b = row.get("best_overall")
        if not b:
            continue
        recs.append({"symbol": sym, **b})
        # collect any all_six across tfs
        for tf, block in (row.get("by_tf") or {}).items():
            for a in block.get("all_six") or []:
                survivors.append({"symbol": sym, "tf": tf, **a})

    survivors.sort(key=lambda x: (-int(x.get("n_green") or 0), -float(x.get("pnl_1y") or 0)))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "adaptive_xm_rest5_hunt",
        "mode": "adaptive_xm_rest5_hunt",
        "research_only": True,
        "recommend_live": bool(survivors),
        "risk_usd": RISK_USD,
        "elapsed_sec": round(time.time() - t0, 1),
        "cells_done": cells_done,
        "tfs": list(TFS),
        "book": list(REST_BOOK),
        "live_untouched": ["GOLD", "BTCUSD", "UKOIL"],
        "n_variants": len(variants_full),
        "windows": [w for w, _ in WINDOWS],
        "survivors_all_six": survivors[:20],
        "recommendations": recs,
        "per_pair": per_pair,
        "note": (
            "Rest-5 Nebula hunt — new adaptive brains + omni routers for "
            "AUDUSD/USDCHF/ETHUSD/NAS100/USDJPY. Live desk (GOLD/BTC/UKOIL) untouched."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _save_partial(payload)
    _progress(
        {
            "status": "done",
            "pct": 100,
            "cells_done": cells_done,
            "cells_est": est,
            "elapsed_sec": round(time.time() - t0, 1),
            "survivors": len(survivors),
        }
    )
    _log(
        f"\n=== DONE · {time.time()-t0:.0f}s · cells={cells_done} · "
        f"all_six survivors={len(survivors)} ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
