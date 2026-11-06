#!/usr/bin/env python3
"""RESEARCH ONLY — ETHUSD: BB mean-revert vs EMA8/14 trend scalp + HTF sync.

Question: is a trend brain (EMA 8/14 scalp) more efficient when 15m + 1h
are in sync, vs live bb_1.5 mean-reversion (buys dips into shorts)?

Protocol (same battery as FX/crypto hunts):
  • Dukascopy OHLCV
  • Train on trailing 6m + 1y (joint rank)
  • Validate winners on 1w / 2w / 1m
  • Exec TFs: 1m / 3m / 5m (scalp / intraday)

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

OUT = Path(ROOT) / "data" / "lab_eth_trend_ema_htf_sync_bt.json"
LOG = Path(ROOT) / "data" / "lab_eth_trend_ema_htf_sync_bt.log"

SYMBOL = "ETHUSD"
EXEC_TFS = ("1m", "3m", "5m")
TRAIN_WINDOWS = (("6m", 183), ("1y", 365))
VAL_HORIZONS = (("1w", 7), ("2w", 14), ("1m", 30))


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _load_tf(sym: str, tf: str, days: int = 400) -> pd.DataFrame:
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    clear_hist_cache()
    if not has_local_hist(sym, tf):
        raise RuntimeError(f"no hist {sym} {tf}")
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    raw = load_local_ohlcv(
        sym,
        tf=tf,
        start=(end - pd.Timedelta(days=days + 40)).isoformat(),
        end=end.isoformat(),
    )
    if raw is None or len(raw) < 500:
        raise RuntimeError(f"{sym} {tf} too short")
    df = _prep(raw[["open", "high", "low", "close"]].dropna())
    data_end = df.index[-1]
    df = df[df.index >= data_end - pd.Timedelta(days=days)]
    _log(f"load {sym} {tf}: {len(df):,} bars {df.index[0].date()}→{df.index[-1].date()}")
    return df


def _slice_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    end = df.index[-1]
    return df[df.index >= end - pd.Timedelta(days=days)].copy()


def _htf_bias_series(df_htf: pd.DataFrame, mode: str) -> pd.Series:
    """+1 bull, -1 bear, 0 flat/unknown — aligned to HTF index."""
    o = df_htf["open"].astype(float)
    c = df_htf["close"].astype(float)
    if mode == "candle":
        # last completed bar colour
        sig = pd.Series(0, index=df_htf.index, dtype=int)
        sig = sig.mask(c > o, 1).mask(c < o, -1)
        return sig
    # EMA 8/14 on HTF
    ema8 = c.ewm(span=8, adjust=False).mean()
    ema14 = c.ewm(span=14, adjust=False).mean()
    sig = pd.Series(0, index=df_htf.index, dtype=int)
    sig = sig.mask(ema8 > ema14, 1).mask(ema8 < ema14, -1)
    return sig


def _align_htf_to_exec(
    exec_index: pd.DatetimeIndex,
    htf_sig: pd.Series,
) -> np.ndarray:
    """Forward-fill HTF bias onto exec bars (use last closed HTF state)."""
    # shift 1 so we don't peek the forming HTF candle
    s = htf_sig.shift(1)
    aligned = s.reindex(exec_index, method="ffill")
    return aligned.fillna(0).astype(int).to_numpy()


def _bb_entries(
    df: pd.DataFrame,
    *,
    std_mult: float,
    sl_atr: float,
    min_risk_px: float = 0.0,
    require_htf: Optional[np.ndarray] = None,
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
        if require_htf is not None:
            b = int(require_htf[i])
            if side == "BUY" and b != 1:
                continue
            if side == "SELL" and b != -1:
                continue
        risk = abs(c - sl)
        if risk < min_risk_px:
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


def _ema_trend_entries(
    df: pd.DataFrame,
    *,
    fast: int = 8,
    slow: int = 14,
    sl_atr: float = 1.5,
    min_risk_px: float = 0.0,
    mode: str = "cross",  # cross | pullback
    htf_sync: np.ndarray,
) -> list[dict]:
    c = df["close"].astype(float)
    ema_f = c.ewm(span=fast, adjust=False).mean()
    ema_s = c.ewm(span=slow, adjust=False).mean()
    atr = df["atr"].to_numpy(dtype=float)
    close = c.to_numpy(dtype=float)
    ef = ema_f.to_numpy(dtype=float)
    es = ema_s.to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    entries: list[dict] = []
    start = max(slow + 5, 60)
    for i in range(start, len(df) - 1):
        if not (np.isfinite(ef[i]) and np.isfinite(es[i]) and np.isfinite(ef[i - 1])):
            continue
        bias = int(htf_sync[i])
        if bias == 0:
            continue
        side = None
        if mode == "cross":
            if ef[i - 1] <= es[i - 1] and ef[i] > es[i] and bias == 1:
                side = "BUY"
            elif ef[i - 1] >= es[i - 1] and ef[i] < es[i] and bias == -1:
                side = "SELL"
        else:  # pullback: ribbon aligned + touch fast EMA with continuation candle
            a = float(atr[i] or 0) or 1e-6
            if bias == 1 and ef[i] > es[i]:
                # bull pullback: low tags ema8, close back above
                if low[i] <= ef[i] + 0.05 * a and close[i] > ef[i] and close[i] > float(df["open"].iloc[i]):
                    side = "BUY"
            elif bias == -1 and ef[i] < es[i]:
                if high[i] >= ef[i] - 0.05 * a and close[i] < ef[i] and close[i] < float(df["open"].iloc[i]):
                    side = "SELL"
        if side is None:
            continue
        a = float(atr[i] or 0) or 1e-6
        px = float(close[i])
        sl = px - sl_atr * a if side == "BUY" else px + sl_atr * a
        risk = abs(px - sl)
        if risk < min_risk_px:
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


def _joint_key(row: dict) -> tuple:
    w6 = row.get("windows", {}).get("6m") or {}
    w1 = row.get("windows", {}).get("1y") or {}
    surv = int(w6.get("verdict") == "SURVIVES") + int(w1.get("verdict") == "SURVIVES")
    er = float(w6.get("exp_r") or -99) + float(w1.get("exp_r") or -99)
    oos = float(w6.get("oos_exp_r") or -99) + float(w1.get("oos_exp_r") or -99)
    pnl = float(w6.get("pnl") or -1e18) + float(w1.get("pnl") or -1e18)
    return (surv, er, oos, pnl)


def _build_htf_sync(
    exec_df: pd.DataFrame,
    df_15: pd.DataFrame,
    df_1h: pd.DataFrame,
    mode: str,
) -> np.ndarray:
    b15 = _htf_bias_series(df_15, mode)
    b1h = _htf_bias_series(df_1h, mode)
    a15 = _align_htf_to_exec(exec_df.index, b15)
    a1h = _align_htf_to_exec(exec_df.index, b1h)
    # both must agree and be non-zero
    out = np.zeros(len(exec_df), dtype=int)
    agree = (a15 == a1h) & (a15 != 0)
    out[agree] = a15[agree]
    return out


def _candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        # --- baselines: live-like MR ---
        {
            "name": "LIVE_bb1.5_rr4",
            "family": "bb",
            "bb_mult": 1.5,
            "sl_atr": 1.2,
            "rr": 4.0,
            "min_risk_px": 0.0,
            "htf": None,
            "note": "live ETH bb_1.5 mean-revert (no HTF sync)",
        },
        {
            "name": "LIVE_bb1.5_firm_sl2_rr4",
            "family": "bb",
            "bb_mult": 1.5,
            "sl_atr": 2.0,
            "rr": 4.0,
            "min_risk_px": 6.0,
            "htf": None,
            "note": "firm widen SL only — still MR",
        },
        {
            "name": "BB_mr_with_htf_ema_sync",
            "family": "bb",
            "bb_mult": 1.5,
            "sl_atr": 2.0,
            "rr": 4.0,
            "min_risk_px": 6.0,
            "htf": "ema",
            "note": "MR but only WITH 15m+1h EMA sync (no counter-trend)",
        },
        # --- trend EMA 8/14 ---
        {
            "name": "EMA814_cross_htf_candle_sl1.5_rr2",
            "family": "ema",
            "mode": "cross",
            "sl_atr": 1.5,
            "rr": 2.0,
            "min_risk_px": 6.0,
            "htf": "candle",
            "note": "EMA8/14 cross + 15m&1h candle sync · scalp RR2",
        },
        {
            "name": "EMA814_cross_htf_ema_sl1.5_rr2",
            "family": "ema",
            "mode": "cross",
            "sl_atr": 1.5,
            "rr": 2.0,
            "min_risk_px": 6.0,
            "htf": "ema",
            "note": "EMA8/14 cross + 15m&1h EMA sync · scalp RR2",
        },
        {
            "name": "EMA814_cross_htf_ema_sl2_rr2.4",
            "family": "ema",
            "mode": "cross",
            "sl_atr": 2.0,
            "rr": 2.4,
            "min_risk_px": 6.0,
            "htf": "ema",
            "note": "EMA cross + HTF EMA sync · wider SL",
        },
        {
            "name": "EMA814_cross_htf_ema_sl2_rr4",
            "family": "ema",
            "mode": "cross",
            "sl_atr": 2.0,
            "rr": 4.0,
            "min_risk_px": 6.0,
            "htf": "ema",
            "note": "EMA cross + HTF EMA sync · RR4",
        },
        {
            "name": "EMA814_pull_htf_ema_sl1.5_rr2",
            "family": "ema",
            "mode": "pullback",
            "sl_atr": 1.5,
            "rr": 2.0,
            "min_risk_px": 6.0,
            "htf": "ema",
            "note": "EMA8/14 pullback with trend + HTF sync",
        },
        {
            "name": "EMA814_pull_htf_candle_sl2_rr2",
            "family": "ema",
            "mode": "pullback",
            "sl_atr": 2.0,
            "rr": 2.0,
            "min_risk_px": 6.0,
            "htf": "candle",
            "note": "pullback + candle sync",
        },
        {
            "name": "EMA814_pull_htf_ema_sl2_rr2.4",
            "family": "ema",
            "mode": "pullback",
            "sl_atr": 2.0,
            "rr": 2.4,
            "min_risk_px": 6.0,
            "htf": "ema",
            "note": "pullback firm candidate",
        },
        {
            "name": "EMA814_cross_NO_htf_sl1.5_rr2",
            "family": "ema",
            "mode": "cross",
            "sl_atr": 1.5,
            "rr": 2.0,
            "min_risk_px": 6.0,
            "htf": None,
            "note": "ablation: EMA cross WITHOUT HTF sync",
        },
    ]
    return specs


def _entries_for(
    spec: dict,
    exec_df: pd.DataFrame,
    sync_candle: np.ndarray,
    sync_ema: np.ndarray,
) -> list[dict]:
    htf = spec.get("htf")
    if htf == "candle":
        sync = sync_candle
    elif htf == "ema":
        sync = sync_ema
    else:
        sync = None

    if spec["family"] == "bb":
        return _bb_entries(
            exec_df,
            std_mult=float(spec["bb_mult"]),
            sl_atr=float(spec["sl_atr"]),
            min_risk_px=float(spec.get("min_risk_px") or 0),
            require_htf=sync,
        )
    # ema with no htf → allow all bars (fake sync all ± from ribbon)
    if sync is None:
        # unrestricted: build sync from local ribbon so crosses always pass bias check
        c = exec_df["close"].astype(float)
        ef = c.ewm(span=8, adjust=False).mean()
        es = c.ewm(span=14, adjust=False).mean()
        sync = np.where(ef > es, 1, np.where(ef < es, -1, 0)).astype(int)
    return _ema_trend_entries(
        exec_df,
        fast=8,
        slow=14,
        sl_atr=float(spec["sl_atr"]),
        min_risk_px=float(spec.get("min_risk_px") or 0),
        mode=str(spec.get("mode") or "cross"),
        htf_sync=sync,
    )


def run_exec_tf(
    exec_tf: str,
    df_exec_full: pd.DataFrame,
    df_15_full: pd.DataFrame,
    df_1h_full: pd.DataFrame,
) -> dict[str, Any]:
    _log(f"\n===== ETHUSD exec={exec_tf} =====")
    cells = []
    for spec in _candidate_specs():
        win_pack = {}
        for wtag, wdays in TRAIN_WINDOWS:
            ex = _slice_days(df_exec_full, wdays)
            d15 = _slice_days(df_15_full, wdays + 5)
            d1h = _slice_days(df_1h_full, wdays + 5)
            sync_c = _build_htf_sync(ex, d15, d1h, "candle")
            sync_e = _build_htf_sync(ex, d15, d1h, "ema")
            ents = _entries_for(spec, ex, sync_c, sync_e)
            cell = _pack(spec["name"], ex, ents, float(spec["rr"]), note=spec.get("note", ""))
            win_pack[wtag] = cell
        row = {
            "name": spec["name"],
            "note": spec.get("note", ""),
            "spec": {k: v for k, v in spec.items() if k != "note"},
            "windows": {
                "6m": {
                    k: win_pack["6m"][k]
                    for k in (
                        "trades",
                        "pnl",
                        "exp_r",
                        "win_rate",
                        "verdict",
                        "oos_exp_r",
                        "mc_prob_profit",
                    )
                },
                "1y": {
                    k: win_pack["1y"][k]
                    for k in (
                        "trades",
                        "pnl",
                        "exp_r",
                        "win_rate",
                        "verdict",
                        "oos_exp_r",
                        "mc_prob_profit",
                    )
                },
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

    ranked = sorted(cells, key=_joint_key, reverse=True)
    winner = ranked[0]
    live = next(c for c in cells if c["name"] == "LIVE_bb1.5_rr4")

    # short-horizon validate on 1y frame
    ex1 = _slice_days(df_exec_full, 365)
    d15 = _slice_days(df_15_full, 370)
    d1h = _slice_days(df_1h_full, 370)
    sync_c = _build_htf_sync(ex1, d15, d1h, "candle")
    sync_e = _build_htf_sync(ex1, d15, d1h, "ema")
    wents = _entries_for(winner["spec"], ex1, sync_c, sync_e)
    lents = _entries_for(live["spec"], ex1, sync_c, sync_e)
    horizons = {}
    live_horizons = {}
    for tag, days in VAL_HORIZONS:
        horizons[tag] = _horizon_stats(ex1, wents, float(winner["spec"]["rr"]), days)
        live_horizons[tag] = _horizon_stats(ex1, lents, float(live["spec"]["rr"]), days)
        h, lh = horizons[tag], live_horizons[tag]
        _log(
            f"  VAL {tag}: winner n={h['n']} E[R]={h['exp_r']} ${h['pnl']:,.0f} | "
            f"live n={lh['n']} E[R]={lh['exp_r']} ${lh['pnl']:,.0f}"
        )

    return {
        "exec_tf": exec_tf,
        "live": live,
        "winner": {
            "name": winner["name"],
            "note": winner["note"],
            "spec": winner["spec"],
            "windows": winner["windows"],
            "horizons": horizons,
        },
        "live_horizons": live_horizons,
        "top5": [
            {"name": c["name"], "windows": c["windows"], "note": c["note"]}
            for c in ranked[:5]
        ],
        "cells": cells,
        "span": {
            "bars": len(df_exec_full),
            "start": str(df_exec_full.index[0].date()),
            "end": str(df_exec_full.index[-1].date()),
        },
    }


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass

    _log("=== ETHUSD BB-MR vs EMA8/14 + 15m/1h sync · Dukascopy · RESEARCH ===")
    _log("Live SYMBOL_ENTRY_BRAIN NOT changed.")

    df_15 = _load_tf(SYMBOL, "15m", days=400)
    df_1h = _load_tf(SYMBOL, "1h", days=400)

    results = []
    for tf in EXEC_TFS:
        try:
            df_ex = _load_tf(SYMBOL, tf, days=400)
            results.append(run_exec_tf(tf, df_ex, df_15, df_1h))
        except Exception as exc:
            _log(f"FAIL {tf}: {exc}")
            results.append({"exec_tf": tf, "error": str(exc)})

    def best_key(r: dict) -> tuple:
        if r.get("error"):
            return (-1, -99, -99, -1e18)
        return _joint_key({"windows": r["winner"]["windows"]})

    best = max(results, key=best_key)
    _log("\n### BEST EXEC TF")
    if not best.get("error"):
        w = best["winner"]
        _log(
            f"BEST={best['exec_tf']} · {w['name']} · "
            f"6m E[R]={w['windows']['6m']['exp_r']} {w['windows']['6m']['verdict']} · "
            f"1y E[R]={w['windows']['1y']['exp_r']} {w['windows']['1y']['verdict']}"
        )
        _log(f"  note: {w['note']}")
        for tag, _ in VAL_HORIZONS:
            h = w["horizons"][tag]
            lh = best["live_horizons"][tag]
            _log(
                f"  {tag}: winner ${h['pnl']:,.0f} E[R]={h['exp_r']} n={h['n']} | "
                f"live-MR ${lh['pnl']:,.0f} E[R]={lh['exp_r']} n={lh['n']}"
            )

    # Head-to-head on 1m: live MR vs best trend name across tfs
    _log("\n### HEAD-TO-HEAD (per TF: LIVE_bb1.5 vs best EMA-with-HTF)")
    for r in results:
        if r.get("error"):
            continue
        live = r["live"]["windows"]
        ema_cells = [
            c
            for c in r["cells"]
            if c["spec"].get("family") == "ema" and c["spec"].get("htf") is not None
        ]
        if not ema_cells:
            continue
        best_ema = max(ema_cells, key=_joint_key)
        _log(
            f"{r['exec_tf']}: LIVE_MR 6m E[R]={live['6m']['exp_r']} {live['6m']['verdict']} | "
            f"best_trend={best_ema['name']} 6m E[R]={best_ema['windows']['6m']['exp_r']} "
            f"{best_ema['windows']['6m']['verdict']} / 1y E[R]={best_ema['windows']['1y']['exp_r']} "
            f"{best_ema['windows']['1y']['verdict']}"
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "symbol": SYMBOL,
        "risk_usd": RISK_USD,
        "method": {
            "mr": "BB 1.5 mean-revert (live)",
            "trend": "EMA 8/14 cross or pullback",
            "htf_sync": "15m + 1h must agree (candle colour OR EMA8>EMA14)",
            "train": "joint 6m + 1y",
            "validate": "1w / 2w / 1m",
        },
        "best": None
        if best.get("error")
        else {
            "exec_tf": best["exec_tf"],
            "name": best["winner"]["name"],
            "note": best["winner"]["note"],
            "spec": best["winner"]["spec"],
            "windows": best["winner"]["windows"],
            "horizons": best["winner"]["horizons"],
        },
        "results": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"\nWrote {OUT} in {payload['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
