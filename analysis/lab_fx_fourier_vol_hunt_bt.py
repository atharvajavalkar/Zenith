#!/usr/bin/env python3
"""RESEARCH ONLY — AUDUSD / USDCHF scalp-intraday hunt.

Stack:
  • Dukascopy OHLCV (tick volume) on 1m / 3m / 5m / 15m
  • TradingView volume overlay only if Duka vol>0 rate is weak
  • Fourier HF regime + return-distribution gates (z / kurtosis)
  • Relative volume gate (vol / 60-bar mean)

Protocol:
  1. Train/select on trailing 6m and 1y (joint rank)
  2. Validate winners on 1w / 2w / 1m
  3. Compare vs live brains (AUDUSD bb_2, USDCHF treated as bb baseline + CTA)

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

OUT = Path(ROOT) / "data" / "lab_fx_fourier_vol_hunt_bt.json"
LOG = Path(ROOT) / "data" / "lab_fx_fourier_vol_hunt_bt.log"

SYMBOLS = ("AUDUSD", "USDCHF")
TFS = ("1m", "3m", "5m", "15m")
TRAIN_WINDOWS = (("6m", 183), ("1y", 365))
VAL_HORIZONS = (("1w", 7), ("2w", 14), ("1m", 30))

# Live desk baselines (.env) — USDCHF is PA live; BB used as MR proxy for hunt
LIVE = {
    "AUDUSD": {"brain": "bb_2", "bb_mult": 2.0, "sl_atr": 1.2, "rr": 4.0},
    "USDCHF": {"brain": "pa_liq_engulf→bb_proxy", "bb_mult": 2.0, "sl_atr": 1.2, "rr": 4.0},
}

# FX min stop distance (price) — ~1.5–2.5 pips
MIN_RISK_PIP = {
    "AUDUSD": 0.00015,
    "USDCHF": 0.00015,
}


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _rolling_fourier_hf(ret: np.ndarray, win: int = 64, stride: int = 8) -> np.ndarray:
    n = len(ret)
    out = np.full(n, np.nan, dtype=float)
    for i in range(win, n, stride):
        seg = ret[i - win : i]
        if not np.isfinite(seg).all():
            continue
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg)) ** 2
        if spec.sum() <= 0:
            continue
        cut = max(2, len(spec) // 2)
        out[i] = float(spec[cut:].sum() / spec.sum())
    return pd.Series(out).ffill().fillna(0.5).to_numpy(dtype=float)


def _maybe_overlay_tv_volume(df: pd.DataFrame, sym: str, tf: str) -> tuple[pd.DataFrame, str]:
    """Keep Duka volume unless sparse; then try TradingView merge."""
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    vol = out["volume"].astype(float)
    rate = float((vol > 0).mean()) if len(vol) else 0.0
    if rate >= 0.45 and float(vol.median()) > 0:
        return out, f"duka_volume ok rate={rate:.1%}"

    try:
        from data.fetcher import fetch_tv_symbol

        bars = min(50_000, max(5_000, len(out) + 500))
        # TV intervals: 1 / 3 / 5 / 15
        interval = {"1m": "1", "3m": "3", "5m": "5", "15m": "15"}.get(tf, "5")
        tv = fetch_tv_symbol(sym, interval=interval, bars=bars)
        if tv is None or tv.empty or "volume" not in tv.columns:
            return out, f"duka_volume weak rate={rate:.1%}; TV unavailable"
        tv = tv.copy()
        if not isinstance(tv.index, pd.DatetimeIndex):
            tv.index = pd.to_datetime(tv.index, utc=True)
        tv.index = tv.index.tz_localize(None) if tv.index.tz is not None else tv.index
        out_idx = out.index.tz_localize(None) if getattr(out.index, "tz", None) else out.index
        aligned = tv["volume"].reindex(out_idx, method="nearest", tolerance=pd.Timedelta(tf))
        if aligned.notna().mean() < 0.3:
            return out, f"duka_volume weak rate={rate:.1%}; TV align poor"
        out["volume"] = aligned.fillna(out["volume"]).astype(float)
        return out, f"tv_volume overlay (duka rate was {rate:.1%})"
    except Exception as exc:
        return out, f"duka_volume weak rate={rate:.1%}; TV err={exc}"


def _load_tf(sym: str, tf: str, days: int = 370) -> tuple[pd.DataFrame, str]:
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    clear_hist_cache()
    if not has_local_hist(sym, tf):
        raise RuntimeError(f"no hist {sym} {tf}")
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    # hist files end ~2026-07-23 — use file end as anchor
    raw = load_local_ohlcv(
        sym,
        tf=tf,
        start=(end - pd.Timedelta(days=days + 30)).isoformat(),
        end=end.isoformat(),
    )
    if raw is None or len(raw) < 800:
        raise RuntimeError(f"{sym} {tf} too short")
    need = [c for c in ("open", "high", "low", "close", "volume") if c in raw.columns]
    df = raw[need].copy()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = _prep(df[["open", "high", "low", "close"]].dropna())
    # restore volume aligned to prep index
    vol = raw["volume"].reindex(df.index).fillna(0.0) if "volume" in raw.columns else 0.0
    df["volume"] = vol
    df, vol_note = _maybe_overlay_tv_volume(df, sym, tf)
    data_end = df.index[-1]
    df = df[df.index >= data_end - pd.Timedelta(days=days)]
    _log(f"load {sym} {tf}: {len(df):,} bars {df.index[0].date()}→{df.index[-1].date()} · {vol_note}")
    return df, vol_note


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    o = out["open"].astype(float)
    v = out["volume"].astype(float) if "volume" in out.columns else pd.Series(0.0, index=out.index)
    rng = (h - l).replace(0, np.nan)
    out["range_px"] = rng
    out["range_bps"] = (rng / c) * 10_000.0
    out["eff"] = ((c - o).abs() / rng).clip(0, 1).fillna(0)
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ret"] = ret
    out["ret_z"] = ((ret - ret.rolling(60).mean()) / ret.rolling(60).std().replace(0, np.nan)).fillna(0.0)
    out["kurt60"] = ret.rolling(60).kurt().fillna(0.0)
    out["vol60"] = ret.rolling(60).std().fillna(0.0)
    out["fourier_hf"] = _rolling_fourier_hf(ret.to_numpy(), win=64, stride=8)
    vmean = v.rolling(60).mean().replace(0, np.nan)
    out["rel_vol"] = (v / vmean).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _slice_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    end = df.index[-1]
    return df[df.index >= end - pd.Timedelta(days=days)].copy()


def _bb_entries(
    df: pd.DataFrame,
    *,
    std_mult: float,
    sl_atr: float,
    min_risk_px: float = 0.0,
    fourier_hf_min: Optional[float] = None,
    abs_z_min: Optional[float] = None,
    kurt_max: Optional[float] = None,
    min_eff: Optional[float] = None,
    rel_vol_min: Optional[float] = None,
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
    fh = df["fourier_hf"].to_numpy(dtype=float)
    rz = df["ret_z"].to_numpy(dtype=float)
    kurt = df["kurt60"].to_numpy(dtype=float)
    eff = df["eff"].to_numpy(dtype=float)
    rbps = df["range_bps"].to_numpy(dtype=float)
    rv = df["rel_vol"].to_numpy(dtype=float)

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
        if fourier_hf_min is not None and fh[i] < fourier_hf_min:
            continue
        if abs_z_min is not None and abs(rz[i]) < abs_z_min:
            continue
        if kurt_max is not None and abs(kurt[i]) > kurt_max:
            continue
        if min_eff is not None and eff[i] < min_eff:
            continue
        if rel_vol_min is not None and rv[i] < rel_vol_min:
            continue
        if max_range_bps is not None and rbps[i] > max_range_bps:
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


def _cta_entries(
    df: pd.DataFrame,
    *,
    fast: int = 10,
    slow: int = 40,
    sl_atr: float = 1.5,
    min_risk_px: float = 0.0,
    fourier_hf_min: Optional[float] = None,
    rel_vol_min: Optional[float] = None,
) -> list[dict]:
    c = df["close"].astype(float)
    ema_f = c.ewm(span=fast, adjust=False).mean()
    ema_s = c.ewm(span=slow, adjust=False).mean()
    atr = df["atr"].to_numpy(dtype=float)
    close = c.to_numpy(dtype=float)
    ef = ema_f.to_numpy(dtype=float)
    es = ema_s.to_numpy(dtype=float)
    fh = df["fourier_hf"].to_numpy(dtype=float)
    rv = df["rel_vol"].to_numpy(dtype=float)
    entries: list[dict] = []
    for i in range(slow + 5, len(df) - 1):
        if not (np.isfinite(ef[i]) and np.isfinite(es[i]) and np.isfinite(ef[i - 1])):
            continue
        side = None
        if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
            side = "BUY"
        elif ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
            side = "SELL"
        if side is None:
            continue
        a = float(atr[i] or 0) or 1e-6
        px = float(close[i])
        sl = px - sl_atr * a if side == "BUY" else px + sl_atr * a
        risk = abs(px - sl)
        if risk < min_risk_px:
            continue
        if fourier_hf_min is not None and fh[i] < fourier_hf_min:
            continue
        if rel_vol_min is not None and rv[i] < rel_vol_min:
            continue
        entries.append(
            {
                "ei": i,
                "side": side,
                "entry": px,
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
        "mc_prob_profit": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "oos_exp_r": ((bat.get("oos") or {}).get("exp_r")),
        "oos_pnl": ((bat.get("oos") or {}).get("pnl")),
    }


def _horizon_stats(df: pd.DataFrame, entries: list, rr: float, days: int) -> dict:
    if df is None or len(df) == 0:
        return {"n": 0, "pnl": 0.0, "exp_r": 0.0, "win_rate": 0.0}
    end = df.index[-1]
    start = end - pd.Timedelta(days=days)
    start_i = int(df.index.searchsorted(start))
    tr = _simulate_fast(df, entries, rr, start_i=start_i)
    if not tr:
        return {
            "n": 0,
            "pnl": 0.0,
            "exp_r": 0.0,
            "win_rate": 0.0,
            "start": str(start.date()),
            "end": str(end.date()),
        }
    rs = np.array([t["r"] for t in tr], dtype=float)
    return {
        "n": len(rs),
        "pnl": round(float(rs.sum() * RISK_USD), 2),
        "exp_r": round(float(rs.mean()), 4),
        "win_rate": round(100.0 * float((rs > 0).mean()), 1),
        "start": str(df.index[start_i].date()) if start_i < len(df) else None,
        "end": str(end.date()),
    }


def _candidate_specs(sym: str) -> list[dict[str, Any]]:
    live = LIVE[sym]
    mr = MIN_RISK_PIP[sym]
    specs: list[dict[str, Any]] = [
        {
            "name": f"LIVE_bb{live['bb_mult']:g}_rr{live['rr']:g}",
            "family": "bb",
            "bb_mult": live["bb_mult"],
            "sl_atr": live["sl_atr"],
            "rr": live["rr"],
            "min_risk_px": 0.0,
            "note": f"live-like {live['brain']}",
        },
        {
            "name": "BB_fourier_hf35_sl1.5",
            "family": "bb",
            "bb_mult": 2.0,
            "sl_atr": 1.5,
            "rr": 2.0,
            "min_risk_px": mr,
            "fourier_hf_min": 0.35,
            "note": "MR + Fourier HF≥0.35 · scalp RR2",
        },
        {
            "name": "BB_fourier_hf40_vol_sl2",
            "family": "bb",
            "bb_mult": 2.0,
            "sl_atr": 2.0,
            "rr": 2.4,
            "min_risk_px": mr,
            "fourier_hf_min": 0.40,
            "rel_vol_min": 1.1,
            "note": "Fourier + rel_vol≥1.1",
        },
        {
            "name": "BB_dist_z15_kurt_sl2",
            "family": "bb",
            "bb_mult": 2.0,
            "sl_atr": 2.0,
            "rr": 3.0,
            "min_risk_px": mr,
            "abs_z_min": 1.5,
            "kurt_max": 7.0,
            "note": "|z|≥1.5 & |kurt|<7",
        },
        {
            "name": "BB_COMBO_fourier_dist_vol",
            "family": "bb",
            "bb_mult": 2.0,
            "sl_atr": 2.0,
            "rr": 2.4,
            "min_risk_px": mr,
            "fourier_hf_min": 0.40,
            "abs_z_min": 1.5,
            "kurt_max": 7.0,
            "min_eff": 0.25,
            "rel_vol_min": 1.0,
            "note": "firm combo Fourier+dist+vol+eff",
        },
        {
            "name": "BB_COMBO_strict_rr2",
            "family": "bb",
            "bb_mult": 2.5,
            "sl_atr": 2.5,
            "rr": 2.0,
            "min_risk_px": mr * 1.3,
            "fourier_hf_min": 0.45,
            "abs_z_min": 2.0,
            "kurt_max": 5.0,
            "min_eff": 0.30,
            "rel_vol_min": 1.2,
            "max_range_bps": 12.0,
            "note": "strict quality scalp",
        },
        {
            "name": "BB15_fourier_vol_rr2",
            "family": "bb",
            "bb_mult": 1.5,
            "sl_atr": 1.5,
            "rr": 2.0,
            "min_risk_px": mr,
            "fourier_hf_min": 0.35,
            "rel_vol_min": 1.0,
            "note": "tighter BB1.5 scalp",
        },
        {
            "name": "BB25_fourier_rr4",
            "family": "bb",
            "bb_mult": 2.5,
            "sl_atr": 2.0,
            "rr": 4.0,
            "min_risk_px": mr,
            "fourier_hf_min": 0.40,
            "note": "wide BB runner RR4",
        },
        {
            "name": "CTA_10_40_liveish",
            "family": "cta",
            "fast": 10,
            "slow": 40,
            "sl_atr": 1.5,
            "rr": 3.0,
            "min_risk_px": mr,
            "note": "CTA baseline",
        },
        {
            "name": "CTA_fourier_vol",
            "family": "cta",
            "fast": 10,
            "slow": 40,
            "sl_atr": 2.0,
            "rr": 2.4,
            "min_risk_px": mr,
            "fourier_hf_min": 0.35,
            "rel_vol_min": 1.1,
            "note": "CTA + Fourier + volume",
        },
    ]
    # RR sweep on firm combo
    for rr in (2.0, 3.0, 4.0):
        specs.append(
            {
                "name": f"BB_COMBO_rr{rr:g}",
                "family": "bb",
                "bb_mult": 2.0,
                "sl_atr": 2.0,
                "rr": rr,
                "min_risk_px": mr,
                "fourier_hf_min": 0.40,
                "abs_z_min": 1.5,
                "kurt_max": 7.0,
                "min_eff": 0.25,
                "rel_vol_min": 1.0,
                "note": f"combo RR sweep {rr}",
            }
        )
    return specs


def _entries_for_spec(df: pd.DataFrame, spec: dict) -> list[dict]:
    if spec.get("family") == "cta":
        return _cta_entries(
            df,
            fast=int(spec.get("fast") or 10),
            slow=int(spec.get("slow") or 40),
            sl_atr=float(spec["sl_atr"]),
            min_risk_px=float(spec.get("min_risk_px") or 0),
            fourier_hf_min=spec.get("fourier_hf_min"),
            rel_vol_min=spec.get("rel_vol_min"),
        )
    return _bb_entries(
        df,
        std_mult=float(spec["bb_mult"]),
        sl_atr=float(spec["sl_atr"]),
        min_risk_px=float(spec.get("min_risk_px") or 0),
        fourier_hf_min=spec.get("fourier_hf_min"),
        abs_z_min=spec.get("abs_z_min"),
        kurt_max=spec.get("kurt_max"),
        min_eff=spec.get("min_eff"),
        rel_vol_min=spec.get("rel_vol_min"),
        max_range_bps=spec.get("max_range_bps"),
    )


def _joint_rank_key(row: dict) -> tuple:
    """Prefer specs that survive both 6m and 1y with positive expectancy."""
    w6 = row.get("windows", {}).get("6m") or {}
    w1 = row.get("windows", {}).get("1y") or {}
    surv = int(w6.get("verdict") == "SURVIVES") + int(w1.get("verdict") == "SURVIVES")
    er = float(w6.get("exp_r") or -99) + float(w1.get("exp_r") or -99)
    oos = float(w6.get("oos_exp_r") or -99) + float(w1.get("oos_exp_r") or -99)
    pnl = float(w6.get("pnl") or -1e18) + float(w1.get("pnl") or -1e18)
    return (surv, er, oos, pnl)


def run_symbol_tf(sym: str, tf: str, df_full: pd.DataFrame, vol_note: str) -> dict[str, Any]:
    _log(f"\n===== {sym} {tf} =====")
    cells = []
    for spec in _candidate_specs(sym):
        win_pack = {}
        for wtag, wdays in TRAIN_WINDOWS:
            dfw = _enrich(_slice_days(df_full, wdays))
            ents = _entries_for_spec(dfw, spec)
            # remap ei is already on sliced df
            cell = _pack(spec["name"], dfw, ents, float(spec["rr"]), note=spec.get("note", ""))
            win_pack[wtag] = cell
        row = {
            "name": spec["name"],
            "note": spec.get("note", ""),
            "spec": {k: v for k, v in spec.items() if k not in ("note",)},
            "windows": {
                "6m": {k: win_pack["6m"][k] for k in (
                    "trades", "pnl", "exp_r", "win_rate", "verdict", "oos_exp_r", "mc_prob_profit"
                )},
                "1y": {k: win_pack["1y"][k] for k in (
                    "trades", "pnl", "exp_r", "win_rate", "verdict", "oos_exp_r", "mc_prob_profit"
                )},
            },
        }
        cells.append(row)
        _log(
            f"  {spec['name']}: "
            f"6m n={row['windows']['6m']['trades']} E[R]={row['windows']['6m']['exp_r']} "
            f"{row['windows']['6m']['verdict']} | "
            f"1y n={row['windows']['1y']['trades']} E[R]={row['windows']['1y']['exp_r']} "
            f"{row['windows']['1y']['verdict']}"
        )

    ranked = sorted(cells, key=_joint_rank_key, reverse=True)
    winner = ranked[0]
    live_row = next(c for c in cells if c["name"].startswith("LIVE_"))

    # Validate winner + live on short horizons using 1y enriched frame
    df1y = _enrich(_slice_days(df_full, 365))
    wents = _entries_for_spec(df1y, winner["spec"])
    lents = _entries_for_spec(df1y, live_row["spec"])
    horizons = {}
    live_horizons = {}
    for tag, days in VAL_HORIZONS:
        horizons[tag] = _horizon_stats(df1y, wents, float(winner["spec"]["rr"]), days)
        live_horizons[tag] = _horizon_stats(df1y, lents, float(live_row["spec"]["rr"]), days)
        h, lh = horizons[tag], live_horizons[tag]
        _log(
            f"  VAL {tag}: winner n={h['n']} E[R]={h['exp_r']} ${h['pnl']:,.0f} | "
            f"live n={lh['n']} E[R]={lh['exp_r']} ${lh['pnl']:,.0f}"
        )

    return {
        "symbol": sym,
        "tf": tf,
        "volume_note": vol_note,
        "span": {
            "bars": len(df_full),
            "start": str(df_full.index[0].date()),
            "end": str(df_full.index[-1].date()),
        },
        "live": live_row,
        "winner": {
            "name": winner["name"],
            "note": winner["note"],
            "spec": winner["spec"],
            "windows": winner["windows"],
            "horizons": horizons,
        },
        "live_horizons": live_horizons,
        "top5": [
            {
                "name": c["name"],
                "windows": c["windows"],
                "note": c["note"],
            }
            for c in ranked[:5]
        ],
        "cells": cells,
    }


def _pick_best_tf(sym_results: list[dict]) -> dict:
    """Best TF for a symbol by joint 6m+1y rank of its winner."""
    def key(r: dict) -> tuple:
        if r.get("error"):
            return (-1, -99, -99, -1e18)
        w = r["winner"]
        return _joint_rank_key({"windows": w["windows"]})

    return max(sym_results, key=key)


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass

    _log("=== FX Fourier + Dist + Volume hunt · AUDUSD/USDCHF · RESEARCH ===")
    _log("Scalp/intraday TFs: 1m 3m 5m 15m · train 6m+1y · val 1w/2w/1m")
    _log("Live SYMBOL_ENTRY_BRAIN NOT changed.")

    all_results: list[dict] = []
    by_symbol: dict[str, list] = {s: [] for s in SYMBOLS}

    for sym in SYMBOLS:
        for tf in TFS:
            try:
                df, vol_note = _load_tf(sym, tf, days=370)
                r = run_symbol_tf(sym, tf, df, vol_note)
                all_results.append(r)
                by_symbol[sym].append(r)
            except Exception as exc:
                _log(f"FAIL {sym} {tf}: {exc}")
                err = {"symbol": sym, "tf": tf, "error": str(exc)}
                all_results.append(err)
                by_symbol[sym].append(err)

    picks = {}
    desk_horizons = {tag: {"winner_pnl": 0.0, "live_pnl": 0.0} for tag, _ in VAL_HORIZONS}
    firm = []

    _log("\n### PER-SYMBOL BEST TF")
    for sym in SYMBOLS:
        best = _pick_best_tf(by_symbol[sym])
        if best.get("error"):
            _log(f"{sym}: ALL FAILED")
            continue
        w = best["winner"]
        picks[sym] = {"tf": best["tf"], "name": w["name"], "spec": w["spec"], "note": w["note"]}
        _log(
            f"{sym} BEST={best['tf']} · {w['name']} · "
            f"6m E[R]={w['windows']['6m']['exp_r']} {w['windows']['6m']['verdict']} · "
            f"1y E[R]={w['windows']['1y']['exp_r']} {w['windows']['1y']['verdict']}"
        )
        for tag, _ in VAL_HORIZONS:
            h = w["horizons"][tag]
            lh = best["live_horizons"][tag]
            desk_horizons[tag]["winner_pnl"] += float(h["pnl"])
            desk_horizons[tag]["live_pnl"] += float(lh["pnl"])
            _log(
                f"  {tag}: winner ${h['pnl']:,.0f} E[R]={h['exp_r']} n={h['n']} | "
                f"live ${lh['pnl']:,.0f} E[R]={lh['exp_r']} n={lh['n']}"
            )
        # firm = survives both windows and short-horizon mean E[R] > 0
        surv_both = (
            w["windows"]["6m"].get("verdict") == "SURVIVES"
            and w["windows"]["1y"].get("verdict") == "SURVIVES"
        )
        short_er = np.mean([float(w["horizons"][t]["exp_r"] or 0) for t, _ in VAL_HORIZONS])
        if surv_both and short_er > 0:
            firm.append(picks[sym])

    for tag in desk_horizons:
        desk_horizons[tag]["winner_pnl"] = round(desk_horizons[tag]["winner_pnl"], 2)
        desk_horizons[tag]["live_pnl"] = round(desk_horizons[tag]["live_pnl"], 2)
        desk_horizons[tag]["delta"] = round(
            desk_horizons[tag]["winner_pnl"] - desk_horizons[tag]["live_pnl"], 2
        )

    _log("\n### DESK SHORT-HORIZON (best TF picks)")
    for tag, h in desk_horizons.items():
        _log(
            f"{tag}: winner=${h['winner_pnl']:,.0f} live=${h['live_pnl']:,.0f} Δ=${h['delta']:,.0f}"
        )
    _log(f"FIRM (survive 6m+1y & short E[R]>0): {firm}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "risk_usd": RISK_USD,
        "method": {
            "data": "Dukascopy OHLCV; TV volume only if Duka sparse",
            "fourier": "rolling rFFT 64 — high-freq power fraction",
            "dist": "ret_z60 + kurtosis60",
            "volume": "rel_vol = volume / SMA60(volume)",
            "train": "joint rank on 6m + 1y",
            "validate": "1w / 2w / 1m on winner",
            "style": "scalping / intraday",
        },
        "live_baselines": LIVE,
        "picks": picks,
        "firm_candidates": firm,
        "desk_horizons": desk_horizons,
        "results": all_results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"\nWrote {OUT} in {payload['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
