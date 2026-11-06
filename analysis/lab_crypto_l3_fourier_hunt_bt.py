#!/usr/bin/env python3
"""RESEARCH ONLY — Crypto efficiency hunt: Dukascopy + L3 proxies + Fourier/dist.

1. Baseline live brains (ETH bb_1.5 RR4, BTC bb_2.5 RR2.4)
2. Derive formulas from:
   - L3/XM microstructure gates (spread floor, min risk) calibrated on live Coinbase book
   - Fourier spectral regime (high-freq power → mean-revert OK)
   - Return distribution gates (z-score extremes, kurtosis filter)
3. Full-history battery → pick clear winner per symbol + desk
4. Horizon validate winner: 1w / 1m / 2m

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
from analysis.strategy_lab_duka_deep_hunt import _simulate_fast  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_crypto_l3_fourier_hunt_bt.json"
LOG = Path(ROOT) / "data" / "lab_crypto_l3_fourier_hunt_bt.log"

WINDOW_DAYS = 365
HORIZONS = (("1w", 7), ("1m", 30), ("2m", 60))

# Live wired (.env)
LIVE = {
    "ETHUSD": {"bb_mult": 1.5, "sl_atr": 1.2, "rr": 4.0, "xm_floor": 1.5},
    "BTCUSD": {"bb_mult": 2.5, "sl_atr": 1.2, "rr": 2.4, "xm_floor": 8.0},
}

# L3-calibrated (matches live CRYPTO_L3 gate)
L3_RISK_MULT = 4.0


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _load_1m(sym: str) -> pd.DataFrame:
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    clear_hist_cache()
    if not has_local_hist(sym, "1m"):
        raise RuntimeError(f"no hist {sym} 1m")
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = end - pd.Timedelta(days=WINDOW_DAYS + 5)
    df = load_local_ohlcv(sym, tf="1m", start=start.isoformat(), end=end.isoformat())
    if df is None or len(df) < 1000:
        raise RuntimeError(f"{sym} too short")
    need = [c for c in ("open", "high", "low", "close") if c in df.columns]
    df = _prep(df[need].dropna())
    # volume optional
    if "volume" not in df.columns:
        df["volume"] = 0.0
    data_end = df.index[-1]
    df = df[df.index >= data_end - pd.Timedelta(days=WINDOW_DAYS)]
    _log(f"load {sym}: {len(df):,} bars {df.index[0].date()}→{df.index[-1].date()}")
    return df


def _enrich_micro(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV proxies for historical L3 (no Tardis history)."""
    out = df.copy()
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    o = out["open"].astype(float)
    rng = (h - l).replace(0, np.nan)
    out["range_px"] = rng
    out["range_bps"] = (rng / c) * 10_000.0
    out["eff"] = ((c - o).abs() / rng).clip(0, 1).fillna(0)
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ret"] = ret
    # rolling distribution
    out["ret_z"] = (ret - ret.rolling(60).mean()) / ret.rolling(60).std().replace(0, np.nan)
    out["ret_z"] = out["ret_z"].fillna(0.0)
    out["kurt60"] = ret.rolling(60).kurt().fillna(0.0)
    out["vol60"] = ret.rolling(60).std().fillna(0.0)
    # Fourier: fraction of power in high-frequency bins (mean-revert friendly)
    # window 64 — vectorized via rolling apply on blocks every bar is slow;
    # compute on stride then ffill.
    out["fourier_hf"] = _rolling_fourier_hf(ret.to_numpy(), win=64, stride=8)
    return out


def _rolling_fourier_hf(ret: np.ndarray, win: int = 64, stride: int = 8) -> np.ndarray:
    n = len(ret)
    out = np.full(n, np.nan, dtype=float)
    half = win // 2
    for i in range(win, n, stride):
        seg = ret[i - win : i]
        if not np.isfinite(seg).all():
            continue
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg)) ** 2
        if spec.sum() <= 0:
            continue
        # high-freq = upper half of positive freqs
        cut = max(2, len(spec) // 2)
        hf = float(spec[cut:].sum() / spec.sum())
        out[i] = hf
    # forward-fill stride gaps
    s = pd.Series(out).ffill().fillna(0.5)
    return s.to_numpy(dtype=float)


def _bb_mask_entries(
    df: pd.DataFrame,
    *,
    std_mult: float,
    sl_atr: float,
    min_risk_px: float = 0.0,
    fourier_hf_min: Optional[float] = None,
    fourier_hf_max: Optional[float] = None,
    abs_z_min: Optional[float] = None,
    kurt_max: Optional[float] = None,
    min_eff: Optional[float] = None,
    max_range_bps: Optional[float] = None,
) -> list[dict]:
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    lo = mid - std_mult * std
    hi = mid + std_mult * std
    atr = df["atr"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    lo_a = lo.to_numpy(dtype=float)
    hi_a = hi.to_numpy(dtype=float)
    fh = df["fourier_hf"].to_numpy(dtype=float) if "fourier_hf" in df.columns else None
    rz = df["ret_z"].to_numpy(dtype=float) if "ret_z" in df.columns else None
    kurt = df["kurt60"].to_numpy(dtype=float) if "kurt60" in df.columns else None
    eff = df["eff"].to_numpy(dtype=float) if "eff" in df.columns else None
    rbps = df["range_bps"].to_numpy(dtype=float) if "range_bps" in df.columns else None

    entries: list[dict] = []
    for i in range(60, len(df) - 1):
        a = float(atr[i] or 0) or 1e-6
        c, o = float(close[i]), float(open_[i])
        if not np.isfinite(lo_a[i]) or not np.isfinite(hi_a[i]):
            continue
        side = None
        if c < lo_a[i] and c > o:
            side = "BUY"
            sl = c - sl_atr * a
        elif c > hi_a[i] and c < o:
            side = "SELL"
            sl = c + sl_atr * a
        if side is None:
            continue
        risk = abs(c - sl)
        if risk <= 0 or risk < min_risk_px:
            continue
        if fourier_hf_min is not None and fh is not None and fh[i] < fourier_hf_min:
            continue
        if fourier_hf_max is not None and fh is not None and fh[i] > fourier_hf_max:
            continue
        if abs_z_min is not None and rz is not None and abs(rz[i]) < abs_z_min:
            continue
        if kurt_max is not None and kurt is not None and abs(kurt[i]) > kurt_max:
            continue
        if min_eff is not None and eff is not None and eff[i] < min_eff:
            continue
        if max_range_bps is not None and rbps is not None and rbps[i] > max_range_bps:
            continue
        entries.append(
            {
                "ei": i,
                "side": side,
                "entry": c,
                "sl": sl,
                "entry_t": df.index[i],
                "risk_px": risk,
            }
        )
    return entries


def _pack(name: str, df: pd.DataFrame, entries: list, rr: float, note: str = "") -> dict:
    tr = _simulate_fast(df, entries, rr)
    bat = _battery_from_trades(tr, len(df))
    full = bat.get("full") or {}
    return {
        "name": name,
        "note": note,
        "rr": rr,
        "entries_raw": len(entries),
        "trades": full.get("n", 0),
        "pnl": full.get("pnl", 0.0),
        "exp_r": full.get("exp_r", 0.0),
        "win_rate": full.get("win_rate", 0.0),
        "verdict": bat.get("verdict"),
        "wf_green": bat.get("wf_green"),
        "wf_n": bat.get("wf_n"),
        "mc_prob_profit": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "oos_exp_r": ((bat.get("oos") or {}).get("exp_r")),
        "oos_pnl": ((bat.get("oos") or {}).get("pnl")),
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
                "7_torture_spread",
                "8_torture_skip",
            )
        },
    }


def _horizon_stats(df: pd.DataFrame, entries: list, rr: float, days: int) -> dict:
    if df is None or len(df) == 0:
        return {"n": 0, "pnl": 0, "exp_r": 0, "win_rate": 0}
    end = df.index[-1]
    start = end - pd.Timedelta(days=days)
    # map times to indices
    start_i = int(df.index.searchsorted(start))
    tr = _simulate_fast(df, entries, rr, start_i=start_i)
    if not tr:
        return {"n": 0, "pnl": 0.0, "exp_r": 0.0, "win_rate": 0.0, "start": str(start.date()), "end": str(end.date())}
    rs = np.array([t["r"] for t in tr], dtype=float)
    return {
        "n": len(rs),
        "pnl": round(float(rs.sum() * RISK_USD), 2),
        "exp_r": round(float(rs.mean()), 4),
        "win_rate": round(100.0 * float((rs > 0).mean()), 1),
        "start": str(df.index[start_i].date()) if start_i < len(df) else None,
        "end": str(end.date()),
    }


def _rank_key(c: dict) -> tuple:
    return (
        1 if c.get("verdict") == "SURVIVES" else 0,
        float(c.get("exp_r") or -99),
        float(c.get("oos_exp_r") or -99),
        float(c.get("pnl") or -1e18),
    )


def _live_l3_calibration() -> dict[str, Any]:
    cal = {"source": "coinbase_live_or_defaults", "symbols": {}}
    try:
        from data.l3_features import get_crypto_book_features

        for sym in ("ETHUSD", "BTCUSD"):
            f = get_crypto_book_features(sym)
            cal["symbols"][sym] = {
                "ok": f.get("ok"),
                "spread_bps": f.get("spread_bps"),
                "spread_px": f.get("spread_px"),
                "imbalance": f.get("imbalance"),
                "bid_notional": f.get("bid_notional_usd"),
                "ask_notional": f.get("ask_notional_usd"),
                "xm_floor": LIVE[sym]["xm_floor"],
                "min_risk_px": L3_RISK_MULT * LIVE[sym]["xm_floor"],
            }
    except Exception as exc:
        cal["error"] = str(exc)
        for sym in ("ETHUSD", "BTCUSD"):
            cal["symbols"][sym] = {
                "ok": False,
                "xm_floor": LIVE[sym]["xm_floor"],
                "min_risk_px": L3_RISK_MULT * LIVE[sym]["xm_floor"],
            }
    return cal


def _candidate_specs(sym: str) -> list[dict[str, Any]]:
    live = LIVE[sym]
    min_risk = L3_RISK_MULT * float(live["xm_floor"])
    # sl_atr so typical ATR*sl clears min_risk — also fixed atr multiples
    specs = [
        {
            "name": f"LIVE_{sym}",
            "bb_mult": live["bb_mult"],
            "sl_atr": live["sl_atr"],
            "rr": live["rr"],
            "min_risk_px": 0.0,
            "note": "wired live brain",
        },
        {
            "name": f"L3GATE_bb{live['bb_mult']:g}_sl2_rr{live['rr']:g}",
            "bb_mult": live["bb_mult"],
            "sl_atr": 2.0,
            "rr": live["rr"],
            "min_risk_px": min_risk,
            "note": f"L3/XM gate risk≥{min_risk} + sl=2ATR",
        },
        {
            "name": f"L3GATE_bb{live['bb_mult']:g}_sl2.5_rr{live['rr']:g}",
            "bb_mult": live["bb_mult"],
            "sl_atr": 2.5,
            "rr": live["rr"],
            "min_risk_px": min_risk,
            "note": f"L3/XM gate risk≥{min_risk} + sl=2.5ATR",
        },
        {
            "name": f"FOURIER_hf_bb{live['bb_mult']:g}_sl2",
            "bb_mult": live["bb_mult"],
            "sl_atr": 2.0,
            "rr": live["rr"],
            "min_risk_px": min_risk,
            "fourier_hf_min": 0.35,
            "note": "Fourier HF power≥0.35 (mean-revert regime) + L3 risk",
        },
        {
            "name": f"FOURIER_hf45_bb{live['bb_mult']:g}_sl2.5",
            "bb_mult": live["bb_mult"],
            "sl_atr": 2.5,
            "rr": live["rr"],
            "min_risk_px": min_risk,
            "fourier_hf_min": 0.45,
            "note": "Fourier HF≥0.45 + sl2.5 + L3 risk",
        },
        {
            "name": f"DIST_z2_bb{live['bb_mult']:g}_sl2",
            "bb_mult": live["bb_mult"],
            "sl_atr": 2.0,
            "rr": live["rr"],
            "min_risk_px": min_risk,
            "abs_z_min": 1.5,
            "kurt_max": 8.0,
            "note": "|ret_z|≥1.5 & |kurt|<8 + L3 risk",
        },
        {
            "name": f"DIST_z2_kurt_bb{live['bb_mult']:g}_sl2.5",
            "bb_mult": live["bb_mult"],
            "sl_atr": 2.5,
            "rr": live["rr"],
            "min_risk_px": min_risk,
            "abs_z_min": 2.0,
            "kurt_max": 6.0,
            "note": "|ret_z|≥2 & |kurt|<6 + L3 risk",
        },
        {
            "name": f"COMBO_fourier_dist_l3",
            "bb_mult": live["bb_mult"],
            "sl_atr": 2.5,
            "rr": live["rr"],
            "min_risk_px": min_risk,
            "fourier_hf_min": 0.40,
            "abs_z_min": 1.5,
            "kurt_max": 7.0,
            "min_eff": 0.25,
            "note": "Fourier+dist+eff+L3 — firm candidate",
        },
        {
            "name": f"COMBO_strict",
            "bb_mult": max(live["bb_mult"], 2.0),
            "sl_atr": 3.0,
            "rr": live["rr"],
            "min_risk_px": min_risk * 1.25,
            "fourier_hf_min": 0.45,
            "abs_z_min": 2.0,
            "kurt_max": 5.0,
            "min_eff": 0.35,
            "max_range_bps": 40.0,
            "note": "Strict combo — fewer trades, quality",
        },
    ]
    # RR variants on combo
    for rr in (2.0, 2.4, 3.0, 4.0):
        if abs(rr - live["rr"]) < 1e-9:
            continue
        specs.append(
            {
                "name": f"COMBO_fourier_dist_l3_rr{rr:g}",
                "bb_mult": live["bb_mult"],
                "sl_atr": 2.5,
                "rr": rr,
                "min_risk_px": min_risk,
                "fourier_hf_min": 0.40,
                "abs_z_min": 1.5,
                "kurt_max": 7.0,
                "min_eff": 0.25,
                "note": f"combo RR sweep {rr}",
            }
        )
    return specs


def run_symbol(sym: str) -> dict[str, Any]:
    _log(f"\n===== {sym} =====")
    df = _enrich_micro(_load_1m(sym))
    cells = []
    entry_cache: dict[str, list] = {}

    for spec in _candidate_specs(sym):
        key = json.dumps(
            {k: spec.get(k) for k in (
                "bb_mult", "sl_atr", "min_risk_px", "fourier_hf_min",
                "fourier_hf_max", "abs_z_min", "kurt_max", "min_eff", "max_range_bps",
            )},
            sort_keys=True,
        )
        if key not in entry_cache:
            entry_cache[key] = _bb_mask_entries(
                df,
                std_mult=float(spec["bb_mult"]),
                sl_atr=float(spec["sl_atr"]),
                min_risk_px=float(spec.get("min_risk_px") or 0),
                fourier_hf_min=spec.get("fourier_hf_min"),
                fourier_hf_max=spec.get("fourier_hf_max"),
                abs_z_min=spec.get("abs_z_min"),
                kurt_max=spec.get("kurt_max"),
                min_eff=spec.get("min_eff"),
                max_range_bps=spec.get("max_range_bps"),
            )
        ents = entry_cache[key]
        cell = _pack(spec["name"], df, ents, float(spec["rr"]), note=spec.get("note", ""))
        cell["spec"] = {k: v for k, v in spec.items() if k != "note"}
        cells.append(cell)
        _log(
            f"  {cell['name']}: n={cell['trades']} pnl=${cell['pnl']:,.0f} "
            f"E[R]={cell['exp_r']} WR={cell['win_rate']}% "
            f"OOS_E[R]={cell['oos_exp_r']} {cell['verdict']} "
            f"MC={cell['mc_prob_profit']}%"
        )

    ranked = sorted(cells, key=_rank_key, reverse=True)
    winner = ranked[0]
    live_cell = next(c for c in cells if c["name"].startswith("LIVE_"))

    # rebuild winner entries for horizons
    wspec = winner["spec"]
    wents = _bb_mask_entries(
        df,
        std_mult=float(wspec["bb_mult"]),
        sl_atr=float(wspec["sl_atr"]),
        min_risk_px=float(wspec.get("min_risk_px") or 0),
        fourier_hf_min=wspec.get("fourier_hf_min"),
        fourier_hf_max=wspec.get("fourier_hf_max"),
        abs_z_min=wspec.get("abs_z_min"),
        kurt_max=wspec.get("kurt_max"),
        min_eff=wspec.get("min_eff"),
        max_range_bps=wspec.get("max_range_bps"),
    )
    horizons = {}
    for tag, days in HORIZONS:
        horizons[tag] = _horizon_stats(df, wents, float(wspec["rr"]), days)
        h = horizons[tag]
        _log(
            f"  HORIZON {tag}: n={h['n']} pnl=${h['pnl']:,.0f} "
            f"E[R]={h['exp_r']} WR={h['win_rate']}% ({h.get('start')}→{h.get('end')})"
        )

    # also horizon on LIVE for compare
    live_ents = _bb_mask_entries(
        df,
        std_mult=float(LIVE[sym]["bb_mult"]),
        sl_atr=float(LIVE[sym]["sl_atr"]),
        min_risk_px=0.0,
    )
    live_horizons = {
        tag: _horizon_stats(df, live_ents, float(LIVE[sym]["rr"]), days)
        for tag, days in HORIZONS
    }

    return {
        "symbol": sym,
        "live": live_cell,
        "winner": {
            "name": winner["name"],
            "spec": winner["spec"],
            "note": winner["note"],
            "full": {
                "n": winner["trades"],
                "pnl": winner["pnl"],
                "exp_r": winner["exp_r"],
                "win_rate": winner["win_rate"],
                "verdict": winner["verdict"],
                "oos_exp_r": winner["oos_exp_r"],
                "mc_prob_profit": winner["mc_prob_profit"],
            },
            "horizons": horizons,
        },
        "live_horizons": live_horizons,
        "delta_pnl_winner_minus_live": round(
            float(winner["pnl"]) - float(live_cell["pnl"]), 2
        ),
        "top5": [
            {
                "name": c["name"],
                "pnl": c["pnl"],
                "exp_r": c["exp_r"],
                "win_rate": c["win_rate"],
                "verdict": c["verdict"],
                "oos_exp_r": c["oos_exp_r"],
                "n": c["trades"],
            }
            for c in ranked[:5]
        ],
        "cells": cells,
        "span": {
            "bars": len(df),
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
        },
    }


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass

    _log("=== CRYPTO L3 + Fourier + Dist hunt · Dukascopy 1y · RESEARCH ===")
    _log("Live SYMBOL_ENTRY_BRAIN NOT changed.")
    cal = _live_l3_calibration()
    _log(f"L3 calibration: {json.dumps(cal, indent=2)}")

    results = []
    for sym in ("ETHUSD", "BTCUSD"):
        try:
            results.append(run_symbol(sym))
        except Exception as exc:
            _log(f"FAIL {sym}: {exc}")
            results.append({"symbol": sym, "error": str(exc)})

    # Desk pick: sum winner full pnl vs live
    desk = {"winner_pnl": 0.0, "live_pnl": 0.0, "picks": {}}
    for r in results:
        if r.get("error"):
            continue
        desk["winner_pnl"] += float(r["winner"]["full"]["pnl"])
        desk["live_pnl"] += float(r["live"]["pnl"])
        desk["picks"][r["symbol"]] = r["winner"]["name"]

    # Horizon desk totals for winners
    desk_horizons = {}
    for tag, _days in HORIZONS:
        wp = lp = 0.0
        for r in results:
            if r.get("error"):
                continue
            wp += float(r["winner"]["horizons"][tag]["pnl"])
            lp += float(r["live_horizons"][tag]["pnl"])
        desk_horizons[tag] = {
            "winner_pnl": round(wp, 2),
            "live_pnl": round(lp, 2),
            "delta": round(wp - lp, 2),
        }

    _log("\n### DESK SUMMARY")
    _log(f"picks: {desk['picks']}")
    _log(
        f"full 1y winner Σpnl=${desk['winner_pnl']:,.0f} vs live Σpnl=${desk['live_pnl']:,.0f} "
        f"Δ=${desk['winner_pnl']-desk['live_pnl']:,.0f}"
    )
    for tag, h in desk_horizons.items():
        _log(
            f"horizon {tag}: winner=${h['winner_pnl']:,.0f} live=${h['live_pnl']:,.0f} "
            f"Δ=${h['delta']:,.0f}"
        )

    for r in results:
        if r.get("error"):
            continue
        w = r["winner"]
        _log(
            f"\n{r['symbol']} WINNER={w['name']} E[R]={w['full']['exp_r']} "
            f"pnl=${w['full']['pnl']:,.0f} {w['full']['verdict']}"
        )
        _log(f"  formula: {w['note']}")
        _log(f"  spec: {w['spec']}")
        for tag in ("1w", "1m", "2m"):
            h = w["horizons"][tag]
            lh = r["live_horizons"][tag]
            _log(
                f"  {tag}: winner n={h['n']} E[R]={h['exp_r']} ${h['pnl']:,.0f} | "
                f"live n={lh['n']} E[R]={lh['exp_r']} ${lh['pnl']:,.0f}"
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "window_days": WINDOW_DAYS,
        "risk_usd": RISK_USD,
        "l3_calibration": cal,
        "l3_risk_mult": L3_RISK_MULT,
        "method": {
            "data": "Dukascopy 1m OHLCV",
            "l3": "Live Coinbase book calibrates XM spread floors; hist uses OHLCV micro proxies + risk gate",
            "fourier": "rolling rFFT 64 — high-freq power fraction",
            "dist": "ret_z + kurtosis60 gates",
        },
        "desk": {
            **desk,
            "horizons": desk_horizons,
        },
        "symbols": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"\nWrote {OUT} in {payload['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
