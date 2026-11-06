#!/usr/bin/env python3
"""RESEARCH ONLY — GOLD PA process TF grid (1m–15m only).

Same process: liq sweep → dual-TF EMA bias → engulf entry.
All legs restricted to {1m, 3m, 5m, 15m}.

  liq_tf   ∈ {3m, 5m, 15m}   (1m liq skipped — noise + cost)
  entry_tf ∈ {1m, 3m, 5m, 15m} with entry ≤ liq
  bias_a/b ∈ TFs ≥ entry (two distinct; 15m-only → single bias)

Battery: 8 tests + MC. Rank SURVIVES → PnL.
Does NOT change live wiring.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD, SWING, _prep  # noqa: E402
from analysis.lab_agentic_1y_battery import _battery_from_trades  # noqa: E402
from analysis.lab_gold_pa_liq_engulf_bt import RRS, WINDOW_DAYS, _bias_series, _is_engulf, _resample_ohlc  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import _simulate_fast  # noqa: E402
from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_gold_pa_tf_grid_bt.json"
LOG = Path(ROOT) / "data" / "lab_gold_pa_tf_grid_bt.log"

TFS = ("1m", "3m", "5m", "15m")
LIQ_TFS = ("3m", "5m", "15m")
TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15}
SWEEP_WINDOW_MIN = 90
EQUAL_ATR = 0.25
LIQ_LOOKBACK = 80


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _load_frames() -> dict[str, pd.DataFrame]:
    clear_hist_cache()
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    frames: dict[str, pd.DataFrame] = {}
    for tf in TFS:
        if not has_local_hist("GOLD", tf):
            continue
        try:
            df = load_local_ohlcv(
                "GOLD", tf=tf, start=start.isoformat(), end=end.isoformat()
            )
        except Exception as exc:
            _log(f"  load GOLD {tf} FAIL: {exc}")
            continue
        if df is None or len(df) < 400:
            continue
        need = [c for c in ("open", "high", "low", "close") if c in df.columns]
        df = _prep(df[need].dropna())
        data_end = df.index[-1]
        df = df[df.index >= data_end - pd.Timedelta(days=WINDOW_DAYS)]
        if len(df) < 400:
            continue
        frames[tf] = df
        _log(
            f"  load GOLD {tf}: {len(df):,} bars "
            f"{df.index[0].date()}→{df.index[-1].date()}"
        )
    for tf, rule in (("3m", "3min"), ("5m", "5min"), ("15m", "15min")):
        if tf not in frames:
            src = frames.get("1m") or frames.get("5m")
            if src is None:
                raise RuntimeError("need GOLD 1m or 5m")
            frames[tf] = _resample_ohlc(src, rule)
            _log(f"  resample GOLD {tf}: {len(frames[tf]):,}")
    return frames


def _fast_sweeps(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Vectorized-leaning swing/equal liq sweep + reclaim (same rules as PA)."""
    n = len(df)
    if n < 60:
        return []
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float) if "atr" in df.columns else df["ATR"].to_numpy(dtype=float)
    idx = df.index

    is_sl = np.zeros(n, dtype=bool)
    is_sh = np.zeros(n, dtype=bool)
    for i in range(SWING, n - SWING):
        wlo = low[i - SWING : i + SWING + 1]
        whi = high[i - SWING : i + SWING + 1]
        if low[i] <= wlo.min():
            is_sl[i] = True
        if high[i] >= whi.max():
            is_sh[i] = True

    sl_i = np.flatnonzero(is_sl)
    sh_i = np.flatnonzero(is_sh)
    sweeps: list[dict[str, Any]] = []

    for ei in range(50, n - 1):
        a = float(atr[ei]) if atr[ei] == atr[ei] and atr[ei] > 0 else 1e-6
        # recent swing lows/highs
        lo_mask = (sl_i < ei - 2) & (sl_i >= ei - LIQ_LOOKBACK)
        hi_mask = (sh_i < ei - 2) & (sh_i >= ei - LIQ_LOOKBACK)
        rlo = low[sl_i[lo_mask]]
        rhi = high[sh_i[hi_mask]]
        if rlo.size == 0 and rhi.size == 0:
            continue
        tol = max(EQUAL_ATR * a, 1e-9)

        def _levels(arr: np.ndarray) -> list[float]:
            if arr.size == 0:
                return []
            s = np.sort(arr)
            clusters: list[list[float]] = [[float(s[0])]]
            for v in s[1:]:
                if abs(float(v) - clusters[-1][-1]) <= tol:
                    clusters[-1].append(float(v))
                else:
                    clusters.append([float(v)])
            return [float(np.mean(c)) for c in clusters]

        h, l, c, o = float(high[ei]), float(low[ei]), float(close[ei]), float(open_[ei])
        hit = False
        for liq in _levels(rlo):
            if l < liq - 0.02 * a and c > liq and c > o:
                sl = l - 0.08 * a
                if abs(c - sl) > 0.12 * a:
                    sweeps.append(
                        {"t": idx[ei], "side": "BUY", "liq": liq, "sl": sl}
                    )
                    hit = True
                    break
        if hit:
            continue
        for liq in _levels(rhi):
            if h > liq + 0.02 * a and c < liq and c < o:
                sl = h + 0.08 * a
                if abs(c - sl) > 0.12 * a:
                    sweeps.append(
                        {"t": idx[ei], "side": "SELL", "liq": liq, "sl": sl}
                    )
                    break
    return sweeps


def _pack(
    name: str, df: pd.DataFrame, entries: list[dict], rr: float, meta: dict
) -> dict[str, Any]:
    tr = _simulate_fast(df, entries, rr)
    bat = _battery_from_trades(tr, len(df))
    full = bat.get("full") or {}
    tests = bat.get("tests") or {}
    return {
        "name": name,
        **meta,
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
        "tests": {
            k: {
                "n": (tests.get(k) or {}).get("n"),
                "pnl": (tests.get(k) or {}).get("pnl"),
                "exp_r": (tests.get(k) or {}).get("exp_r"),
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
    }


def _combo_iter() -> list[dict[str, str]]:
    combos: list[dict[str, str]] = []
    for liq in LIQ_TFS:
        for entry in TFS:
            if TF_MIN[entry] > TF_MIN[liq]:
                continue
            bias_pool = [t for t in TFS if TF_MIN[t] >= TF_MIN[entry]]
            if len(bias_pool) >= 2:
                pairs = list(itertools.combinations(bias_pool, 2))
            elif bias_pool:
                pairs = [(bias_pool[0], bias_pool[0])]
            else:
                pairs = []
            for b1, b2 in pairs:
                if TF_MIN[b1] < TF_MIN[b2]:
                    b1, b2 = b2, b1
                combos.append(
                    {"liq_tf": liq, "bias_a": b1, "bias_b": b2, "entry_tf": entry}
                )
    return combos


def _collect_entries(
    df_e: pd.DataFrame,
    sweeps: list[dict],
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    entry_tf: str,
) -> list[dict[str, Any]]:
    if not sweeps:
        return []
    max_bars = max(3, int(SWEEP_WINDOW_MIN / TF_MIN[entry_tf]))
    b1 = _bias_series(df_a).reindex(df_e.index, method="ffill").fillna(0).astype(int)
    b2 = _bias_series(df_b).reindex(df_e.index, method="ffill").fillna(0).astype(int)

    # Map sweeps → entry index once
    sweep_pos = []
    for sw in sweeps:
        pos = int(df_e.index.searchsorted(sw["t"], side="left"))
        if pos < len(df_e):
            sweep_pos.append((pos, sw))
    if not sweep_pos:
        return []

    # Sort by pos for two-pointer
    sweep_pos.sort(key=lambda x: x[0])
    entries: list[dict[str, Any]] = []
    used: set[int] = set()
    # Index sweeps by start for sliding window — use deque over positions
    from collections import deque

    active: deque = deque()
    sp_i = 0
    n_sw = len(sweep_pos)
    for i in range(40, len(df_e) - 1):
        while sp_i < n_sw and sweep_pos[sp_i][0] <= i:
            active.append(sweep_pos[sp_i])
            sp_i += 1
        while active and active[0][0] < i - max_bars:
            active.popleft()
        if not active:
            continue
        bb1, bb2 = int(b1.iloc[i]), int(b2.iloc[i])
        if bb1 == 0 or bb1 != bb2:
            continue
        side = "BUY" if bb1 > 0 else "SELL"
        matched = None
        for sp, sw in reversed(active):
            if sp in used:
                continue
            if sw["side"] == side:
                matched = (sp, sw)
                break
        if matched is None:
            continue
        row, prev = df_e.iloc[i], df_e.iloc[i - 1]
        if not _is_engulf(row, prev, side):
            continue
        atr = float(row["atr"] or 0) or 1e-6
        sp, sw = matched
        c = float(row["close"])
        if side == "BUY":
            sl = min(float(sw["sl"]), float(row["low"]) - 0.08 * atr)
            if sl >= c or abs(c - sl) <= 0.12 * atr:
                continue
        else:
            sl = max(float(sw["sl"]), float(row["high"]) + 0.08 * atr)
            if sl <= c or abs(c - sl) <= 0.12 * atr:
                continue
        entries.append(
            {"ei": i, "side": side, "entry": c, "sl": sl, "entry_t": df_e.index[i]}
        )
        used.add(sp)
    return entries


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass
    _log("=== GOLD PA TF grid (1m–15m) · Dukascopy 1y ===")
    frames = _load_frames()

    sweeps_cache: dict[str, list] = {}
    for liq in LIQ_TFS:
        t1 = time.time()
        sweeps_cache[liq] = _fast_sweeps(frames[liq])
        _log(f"  sweeps {liq}: {len(sweeps_cache[liq])} ({time.time()-t1:.1f}s)")

    combos = _combo_iter()
    _log(f"Combos: {len(combos)} × {len(RRS)} RR = {len(combos)*len(RRS)} cells")

    cells: list[dict[str, Any]] = []
    for ci, cfg in enumerate(combos, 1):
        liq, ea, eb, ent = cfg["liq_tf"], cfg["bias_a"], cfg["bias_b"], cfg["entry_tf"]
        tag = f"liq={liq}|bias={ea}+{eb}|entry={ent}"
        t1 = time.time()
        entries = _collect_entries(
            frames[ent], sweeps_cache[liq], frames[ea], frames[eb], ent
        )
        _log(f"[{ci}/{len(combos)}] {tag}: entries={len(entries)} ({time.time()-t1:.1f}s)")
        meta = {"liq_tf": liq, "bias_a": ea, "bias_b": eb, "entry_tf": ent}
        for rr in RRS:
            cell = _pack(tag, frames[ent], entries, rr, meta)
            cells.append(cell)
            if cell["trades"]:
                _log(
                    f"    RR={rr}: n={cell['trades']} pnl=${cell['pnl']:,.0f} "
                    f"E[R]={cell['exp_r']} {cell['verdict']} "
                    f"WF={cell['wf_green']}/{cell['wf_n']}"
                )

    def _key(c: dict) -> tuple:
        return (
            1 if c.get("verdict") == "SURVIVES" else 0,
            float(c.get("pnl") or 0),
            float(c.get("exp_r") or 0),
            int(c.get("wf_green") or 0),
            float(c.get("mc_prob_profit") or 0),
        )

    ranked = sorted(cells, key=_key, reverse=True)
    best = ranked[0] if ranked else None
    survives = [c for c in ranked if c.get("verdict") == "SURVIVES"]
    top15 = ranked[:15]

    # Original reference: liq=15m bias=15m+5m entry=5m closest to prior (no 1h)
    ref_name = "liq=15m|bias=15m+5m|entry=5m"
    ref = [c for c in ranked if c["name"] == ref_name and c["rr"] == 4.0]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "symbol": "GOLD",
        "window_days": WINDOW_DAYS,
        "risk_usd": RISK_USD,
        "tf_universe": list(TFS),
        "liq_tfs": list(LIQ_TFS),
        "note": "1m used as entry only; liq on 3m/5m/15m",
        "combos": len(combos),
        "cells": len(cells),
        "survives_n": len(survives),
        "best": best,
        "top15": top15,
        "ref_15m5m_rr4": ref[0] if ref else None,
        "best_per_entry_tf": {
            etf: next((c for c in ranked if c.get("entry_tf") == etf), None)
            for etf in TFS
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    # slim top15 for JSON
    payload["top15"] = [
        {
            "name": c["name"],
            "liq_tf": c["liq_tf"],
            "bias_a": c["bias_a"],
            "bias_b": c["bias_b"],
            "entry_tf": c["entry_tf"],
            "rr": c["rr"],
            "trades": c["trades"],
            "pnl": c["pnl"],
            "exp_r": c["exp_r"],
            "win_rate": c.get("win_rate"),
            "verdict": c["verdict"],
            "wf_green": c["wf_green"],
            "mc_prob_profit": c["mc_prob_profit"],
            "tests": c["tests"],
        }
        for c in top15
    ]
    if best:
        payload["best"] = payload["top15"][0]

    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _log("\n### BEST")
    if best:
        _log(
            f"{best['name']} RR={best['rr']} → n={best['trades']} "
            f"PnL=${best['pnl']:,.0f} E[R]={best['exp_r']} {best['verdict']} "
            f"WF={best['wf_green']}/{best['wf_n']} MC={best['mc_prob_profit']}%"
        )
        for k, v in (best.get("tests") or {}).items():
            _log(f"  {k}: n={v.get('n')} pnl=${v.get('pnl')} E[R]={v.get('exp_r')}")
    _log("\nTOP 10:")
    for i, c in enumerate(top15[:10], 1):
        _log(
            f"  {i}. {c['name']} RR={c['rr']} pnl=${c['pnl']:,.0f} "
            f"E[R]={c['exp_r']} {c['verdict']} n={c['trades']}"
        )
    _log(
        f"\nWrote {OUT} ({payload['elapsed_s']}s) · "
        f"SURVIVES={len(survives)}/{len(cells)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
