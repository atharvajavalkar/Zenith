#!/usr/bin/env python3
"""RESEARCH ONLY — ETHUSD 1y: noise-aware SL method vs live bb_1.5.

Method (from live BE autopsy):
  1. Live micro-stops (~1.2×ATR on bb_1.5@1m) sit inside noise + spread → −EV.
  2. Widen SL to k×ATR (k≥2) and/or widen BB σ so stops clear microstructure.
  3. Spread-cost gate: skip if risk_px < MIN_RISK_SPREAD_MULT × spread_px
     (refuse −EV noise lottery).
  4. Score with R-based W/L (no fake BE): E[R], WR vs 1/(1+RR), 8-test battery.

Does NOT change live SYMBOL_ENTRY_BRAIN.
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

from analysis._edge_survive_bt import RISK_USD, _prep  # noqa: E402
from analysis.lab_agentic_1y_battery import _battery_from_trades  # noqa: E402
from analysis.lab_gold_pa_liq_engulf_bt import WINDOW_DAYS  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import _simulate_fast  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_eth_noise_sl_method_bt.json"
LOG = Path(ROOT) / "data" / "lab_eth_noise_sl_method_bt.log"

SYMBOL = "ETHUSD"
# Live wired
LIVE = {"bb_mult": 1.5, "sl_atr": 1.2, "rr": 4.0, "tf": "1m"}
# Grid: band width × stop width
BB_MULTS = (1.5, 2.0, 2.5, 3.0)
SL_ATRS = (1.2, 2.0, 2.5, 3.0)
RRS = (2.0, 2.4, 3.0, 4.0)
# XM crypto CFD rough half-spread proxy ($)
SPREAD_PX = 1.5
MIN_RISK_SPREAD_MULT = 3.0  # need risk ≥ 3× spread ≈ 0.33R start cost max


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _load_eth_1m() -> pd.DataFrame:
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    clear_hist_cache()
    if not has_local_hist(SYMBOL, "1m"):
        raise RuntimeError(f"no local hist {SYMBOL} 1m")
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    df = load_local_ohlcv(
        SYMBOL, tf="1m", start=start.isoformat(), end=end.isoformat()
    )
    if df is None or len(df) < 400:
        raise RuntimeError(f"{SYMBOL} 1m too short")
    need = [c for c in ("open", "high", "low", "close") if c in df.columns]
    df = _prep(df[need].dropna())
    data_end = df.index[-1]
    data_start = data_end - pd.Timedelta(days=WINDOW_DAYS)
    df = df[df.index >= data_start]
    _log(
        f"load {SYMBOL} 1m: {len(df):,} bars "
        f"{df.index[0].date()}→{df.index[-1].date()}"
    )
    return df


def _bb_entries(
    df: pd.DataFrame,
    *,
    std_mult: float,
    sl_atr: float,
    min_risk_px: Optional[float] = None,
) -> tuple[list[dict], int]:
    """BB mean-reversion entries matching strategy.lab_brains.evaluate_bb_mr."""
    out = df.copy()
    mid = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    lo, hi = mid - std_mult * std, mid + std_mult * std
    entries: list[dict] = []
    skipped_thin = 0
    for i in range(30, len(out) - 1):
        row = out.iloc[i]
        atr = float(row["atr"] or 0) or 1e-6
        c, o = float(row["close"]), float(row["open"])
        side = None
        if c < float(lo.iloc[i]) and c > o:
            side = "BUY"
            sl = c - sl_atr * atr
        elif c > float(hi.iloc[i]) and c < o:
            side = "SELL"
            sl = c + sl_atr * atr
        if side is None:
            continue
        risk = abs(c - sl)
        if risk <= 0:
            continue
        if min_risk_px is not None and risk < min_risk_px:
            skipped_thin += 1
            continue
        entries.append(
            {
                "ei": i,
                "side": side,
                "entry": c,
                "sl": sl,
                "entry_t": out.index[i],
                "atr": atr,
                "risk_px": risk,
            }
        )
    return entries, skipped_thin


def _r_class_stats(trades: list[dict], rr: float) -> dict[str, Any]:
    """R-based W/L/BE (no absolute $ BE fiction)."""
    if not trades:
        return {
            "n": 0,
            "w": 0,
            "l": 0,
            "be": 0,
            "wr": 0.0,
            "be_wr_needed": round(100.0 / (1.0 + rr), 1),
            "exp_r": 0.0,
            "mean_risk_px": 0.0,
            "p_start_r_after_spread": 0.0,
        }
    rs = np.array([float(t["r"]) for t in trades], dtype=float)
    w = int((rs > 0.25).sum())
    l = int((rs < -0.25).sum())
    be = int(len(rs) - w - l)
    risks = [float(t.get("risk_px") or 0) for t in trades if t.get("risk_px")]
    mean_risk = float(np.mean(risks)) if risks else 0.0
    # barrier start after paying spread once
    x0 = -SPREAD_PX / mean_risk if mean_risk > 0 else -0.3
    p_fair = (x0 + 1.0) / (rr + 1.0)  # no-drift barrier approx
    return {
        "n": len(rs),
        "w": w,
        "l": l,
        "be": be,
        "wr": round(100.0 * w / len(rs), 1),
        "be_wr_needed": round(100.0 / (1.0 + rr), 1),
        "exp_r": round(float(rs.mean()), 4),
        "mean_risk_px": round(mean_risk, 3),
        "spread_start_r": round(float(x0), 3),
        "fair_barrier_p_tp_pct": round(100.0 * p_fair, 1),
        "edge_vs_fair_pp": round(100.0 * w / len(rs) - 100.0 * p_fair, 1),
    }


def _pack(
    name: str,
    df: pd.DataFrame,
    entries: list[dict],
    rr: float,
    *,
    note: str = "",
    skipped_thin: int = 0,
) -> dict[str, Any]:
    tr = _simulate_fast(df, entries, rr)
    # attach risk_px from entries by matching ei
    by_i = {int(e["ei"]): e for e in entries}
    for t in tr:
        e = by_i.get(int(t["i"]))
        if e:
            t["risk_px"] = float(e.get("risk_px") or abs(e["entry"] - e["sl"]))
    bat = _battery_from_trades(tr, len(df))
    full = bat.get("full") or {}
    rstats = _r_class_stats(tr, rr)
    cell = {
        "name": name,
        "note": note,
        "rr": rr,
        "entries_raw": len(entries),
        "skipped_thin_risk": skipped_thin,
        "trades": full.get("n", 0),
        "pnl": full.get("pnl", 0.0),
        "exp_r": full.get("exp_r", 0.0),
        "win_rate": full.get("win_rate", 0.0),
        "verdict": bat.get("verdict"),
        "wf_green": bat.get("wf_green"),
        "wf_n": bat.get("wf_n"),
        "mc_prob_profit": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "r_class": rstats,
        "tests": {
            k: {
                "n": ((bat.get("tests") or {}).get(k) or {}).get("n"),
                "pnl": ((bat.get("tests") or {}).get(k) or {}).get("pnl"),
                "exp_r": ((bat.get("tests") or {}).get(k) or {}).get("exp_r"),
                "win_rate": ((bat.get("tests") or {}).get(k) or {}).get("win_rate"),
            }
            for k in (
                "1_full",
                "2_oos70",
                "3_wf_fold_1",
                "4_wf_fold_2",
                "5_wf_fold_3",
                "6_wf_fold_4",
                "7_torture_spread",
                "8_torture_skip",
            )
        },
        "span": {
            "bars": len(df),
            "start": str(df.index[0].date()) if len(df) else None,
            "end": str(df.index[-1].date()) if len(df) else None,
        },
    }
    return cell


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass

    _log("=== ETHUSD noise-aware SL method · Dukascopy 1y · RESEARCH ===")
    _log("Live SYMBOL_ENTRY_BRAIN NOT changed.")
    _log(
        f"spread_px=${SPREAD_PX} min_risk={MIN_RISK_SPREAD_MULT}×spread "
        f"(${SPREAD_PX * MIN_RISK_SPREAD_MULT:.1f}) · ${RISK_USD}/R"
    )

    df = _load_eth_1m()
    cells: list[dict[str, Any]] = []

    # --- Live baseline (no gate) ---
    live_e, _ = _bb_entries(
        df, std_mult=LIVE["bb_mult"], sl_atr=LIVE["sl_atr"], min_risk_px=None
    )
    live_cell = _pack(
        "LIVE_bb_1.5_sl1.2",
        df,
        live_e,
        LIVE["rr"],
        note="wired live: bb_1.5 @ 1m, sl=1.2ATR, RR=4, no spread gate",
    )
    cells.append(live_cell)
    _log(
        f"LIVE bb1.5/sl1.2 RR4: n={live_cell['trades']} pnl=${live_cell['pnl']:,.0f} "
        f"E[R]={live_cell['exp_r']} WR={live_cell['win_rate']}% "
        f"risk$={live_cell['r_class']['mean_risk_px']} "
        f"fairP={live_cell['r_class']['fair_barrier_p_tp_pct']}% "
        f"edge={live_cell['r_class']['edge_vs_fair_pp']}pp "
        f"{live_cell['verdict']}"
    )

    # --- Method grid: widen band × widen SL @ RR=4, with spread gate ---
    min_risk = SPREAD_PX * MIN_RISK_SPREAD_MULT
    core: list[dict[str, Any]] = []
    for bb in BB_MULTS:
        for sl_atr in SL_ATRS:
            name = f"M_bb{bb:g}_sl{sl_atr:g}_rr4"
            ents, skipped = _bb_entries(
                df, std_mult=bb, sl_atr=sl_atr, min_risk_px=min_risk
            )
            cell = _pack(
                name,
                df,
                ents,
                4.0,
                note=(
                    f"method: bb={bb}σ sl={sl_atr}ATR RR=4 "
                    f"gate risk≥{min_risk:.1f}$"
                ),
                skipped_thin=skipped,
            )
            cells.append(cell)
            core.append(cell)
            _log(
                f"{name}: n={cell['trades']} pnl=${cell['pnl']:,.0f} "
                f"E[R]={cell['exp_r']} WR={cell['win_rate']}% "
                f"risk$={cell['r_class']['mean_risk_px']} "
                f"skip_thin={skipped} {cell['verdict']} "
                f"WF={cell['wf_green']}/{cell['wf_n']} "
                f"MC={cell['mc_prob_profit']}%"
            )

    # RR sweep on top-3 core configs by E[R]
    top_core = sorted(core, key=lambda c: float(c.get("exp_r") or -99), reverse=True)[:3]
    for base in top_core:
        # parse bb/sl from name M_bbX_slY_rr4
        parts = base["name"].replace("M_bb", "").replace("_rr4", "").split("_sl")
        bb = float(parts[0])
        sl_atr = float(parts[1])
        ents, skipped = _bb_entries(
            df, std_mult=bb, sl_atr=sl_atr, min_risk_px=min_risk
        )
        for rr in RRS:
            if abs(rr - 4.0) < 1e-9:
                continue
            name = f"M_bb{bb:g}_sl{sl_atr:g}_rr{rr:g}"
            cell = _pack(
                name,
                df,
                ents,
                rr,
                note=f"RR sweep on top core bb={bb} sl={sl_atr}",
                skipped_thin=skipped,
            )
            cells.append(cell)
            _log(
                f"{name}: n={cell['trades']} pnl=${cell['pnl']:,.0f} "
                f"E[R]={cell['exp_r']} WR={cell['win_rate']}% {cell['verdict']}"
            )

    # Rank: SURVIVES first, then E[R], then pnl
    def _rank(c: dict) -> tuple:
        return (
            1 if c.get("verdict") == "SURVIVES" else 0,
            float(c.get("exp_r") or -99),
            float(c.get("pnl") or -1e18),
        )

    ranked = sorted(cells, key=_rank, reverse=True)
    best = ranked[0]
    best_surv = next((c for c in ranked if c["verdict"] == "SURVIVES"), None)

    # Compare method@live RR to live
    method_live_like = next(
        (
            c
            for c in cells
            if c["name"] == "M_bb1.5_sl1.2_rr4"
        ),
        None,
    )
    # Preferred prescription from autopsy: bb≥2.5 and/or sl≥2.5ATR
    preferred = next(
        (c for c in cells if c["name"] == "M_bb2.5_sl2.5_rr4"),
        None,
    )
    preferred3 = next(
        (c for c in cells if c["name"] == "M_bb2.5_sl3_rr4"),
        None,
    )

    summary = {
        "live": {
            "name": live_cell["name"],
            "n": live_cell["trades"],
            "pnl": live_cell["pnl"],
            "exp_r": live_cell["exp_r"],
            "win_rate": live_cell["win_rate"],
            "verdict": live_cell["verdict"],
            "r_class": live_cell["r_class"],
        },
        "best_overall": {
            "name": best["name"],
            "n": best["trades"],
            "pnl": best["pnl"],
            "exp_r": best["exp_r"],
            "win_rate": best["win_rate"],
            "verdict": best["verdict"],
            "r_class": best["r_class"],
            "mc_prob_profit": best["mc_prob_profit"],
            "wf_green": best["wf_green"],
        },
        "best_survives": None
        if best_surv is None
        else {
            "name": best_surv["name"],
            "n": best_surv["trades"],
            "pnl": best_surv["pnl"],
            "exp_r": best_surv["exp_r"],
            "win_rate": best_surv["win_rate"],
            "verdict": best_surv["verdict"],
            "r_class": best_surv["r_class"],
            "mc_prob_profit": best_surv["mc_prob_profit"],
        },
        "method_gated_same_as_live": None
        if method_live_like is None
        else {
            "name": method_live_like["name"],
            "n": method_live_like["trades"],
            "pnl": method_live_like["pnl"],
            "exp_r": method_live_like["exp_r"],
            "verdict": method_live_like["verdict"],
            "r_class": method_live_like["r_class"],
            "skipped_thin_risk": method_live_like["skipped_thin_risk"],
        },
        "preferred_bb2.5_sl2.5_rr4": None
        if preferred is None
        else {
            "n": preferred["trades"],
            "pnl": preferred["pnl"],
            "exp_r": preferred["exp_r"],
            "win_rate": preferred["win_rate"],
            "verdict": preferred["verdict"],
            "r_class": preferred["r_class"],
            "mc_prob_profit": preferred["mc_prob_profit"],
            "wf_green": preferred["wf_green"],
        },
        "preferred_bb2.5_sl3_rr4": None
        if preferred3 is None
        else {
            "n": preferred3["trades"],
            "pnl": preferred3["pnl"],
            "exp_r": preferred3["exp_r"],
            "win_rate": preferred3["win_rate"],
            "verdict": preferred3["verdict"],
            "r_class": preferred3["r_class"],
            "mc_prob_profit": preferred3["mc_prob_profit"],
        },
        "delta_preferred_minus_live_pnl": None
        if preferred is None
        else round(float(preferred["pnl"]) - float(live_cell["pnl"]), 2),
        "top10": [
            {
                "name": c["name"],
                "pnl": c["pnl"],
                "exp_r": c["exp_r"],
                "win_rate": c["win_rate"],
                "verdict": c["verdict"],
                "n": c["trades"],
                "mean_risk_px": c["r_class"]["mean_risk_px"],
                "edge_vs_fair_pp": c["r_class"]["edge_vs_fair_pp"],
            }
            for c in ranked[:10]
        ],
        "survives_count": sum(1 for c in cells if c.get("verdict") == "SURVIVES"),
        "fragile_count": sum(1 for c in cells if c.get("verdict") == "FRAGILE"),
        "dead_count": sum(1 for c in cells if c.get("verdict") == "DEAD"),
    }

    _log("\n### SUMMARY")
    _log(
        f"LIVE: n={summary['live']['n']} pnl=${summary['live']['pnl']:,.0f} "
        f"E[R]={summary['live']['exp_r']} {summary['live']['verdict']}"
    )
    _log(
        f"BEST: {summary['best_overall']['name']} "
        f"n={summary['best_overall']['n']} pnl=${summary['best_overall']['pnl']:,.0f} "
        f"E[R]={summary['best_overall']['exp_r']} {summary['best_overall']['verdict']}"
    )
    if summary["best_survives"]:
        _log(
            f"BEST SURVIVES: {summary['best_survives']['name']} "
            f"pnl=${summary['best_survives']['pnl']:,.0f} "
            f"E[R]={summary['best_survives']['exp_r']}"
        )
    else:
        _log("BEST SURVIVES: none")
    if summary["preferred_bb2.5_sl2.5_rr4"]:
        p = summary["preferred_bb2.5_sl2.5_rr4"]
        _log(
            f"PREFERRED bb2.5/sl2.5/RR4: n={p['n']} pnl=${p['pnl']:,.0f} "
            f"E[R]={p['exp_r']} WR={p['win_rate']}% {p['verdict']} "
            f"Δ vs live ${summary['delta_preferred_minus_live_pnl']:,.0f}"
        )
    _log(
        f"verdicts: SURVIVES={summary['survives_count']} "
        f"FRAGILE={summary['fragile_count']} DEAD={summary['dead_count']}"
    )
    _log("TOP10:")
    for row in summary["top10"]:
        _log(
            f"  {row['name']}: pnl=${row['pnl']:,.0f} E[R]={row['exp_r']} "
            f"WR={row['win_rate']}% n={row['n']} risk$={row['mean_risk_px']} "
            f"edge={row['edge_vs_fair_pp']}pp {row['verdict']}"
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "symbol": SYMBOL,
        "window_days": WINDOW_DAYS,
        "method": {
            "spread_px": SPREAD_PX,
            "min_risk_spread_mult": MIN_RISK_SPREAD_MULT,
            "min_risk_px": min_risk,
            "bb_mults": list(BB_MULTS),
            "sl_atrs": list(SL_ATRS),
            "rrs": list(RRS),
            "risk_usd": RISK_USD,
        },
        "live_baseline": LIVE,
        "summary": summary,
        "cells": cells,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": (
            "Refuse thin risk (spread gate) + widen SL/bands. "
            "No system wins every trade; goal is E[R]>0 after costs + SURVIVES battery."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"\nWrote {OUT} in {payload['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
