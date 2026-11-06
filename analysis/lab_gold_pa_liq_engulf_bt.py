#!/usr/bin/env python3
"""RESEARCH ONLY — GOLD discretionary PA vs live don_100 (Dukascopy 1y).

User process (from chart / discretionary rules):
  1. Mark 15m liquidity (swing / equal highs-lows).
  2. Wait for that liquidity to be swept (wick beyond + reclaim close).
  3. Require 1h AND 25m both bullish or both bearish (EMA stack).
  4. Enter on 5m engulfing confirmation in that direction after the sweep.

Compare to wired live brain: don_100 @ 15m, RR=4.
Battery (same 8 tests as scalp/agentic hunts):
  1_full · 2_oos70 · 3–6_wf_fold_* · 7_torture_spread · 8_torture_skip
  + bootstrap MC → SURVIVES / FRAGILE / DEAD

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

from analysis._edge_survive_bt import RISK_USD, SWING, _prep, _swing_high, _swing_low  # noqa: E402
from analysis._rr_best_by_pair import _collect_breakout  # noqa: E402
from analysis.lab_agentic_1y_battery import _battery_from_trades  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import _simulate_fast  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_gold_pa_liq_engulf_bt.json"
LOG = Path(ROOT) / "data" / "lab_gold_pa_liq_engulf_bt.log"

WINDOW_DAYS = 365
RRS = (2.0, 2.4, 3.0, 4.0)
# How long after a 15m sweep we still allow a 5m engulf entry
SWEEP_MAX_5M_BARS = 18  # 90 minutes
EQUAL_ATR = 0.25
LIQ_LOOKBACK = 80


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _bias_series(df: pd.DataFrame) -> pd.Series:
    """+1 bull (ema20>ema50 & close>ema20), -1 bear, 0 chop."""
    c = df["close"]
    e20 = df["ema20"]
    e50 = df["ema50"]
    bull = (e20 > e50) & (c > e20)
    bear = (e20 < e50) & (c < e20)
    out = pd.Series(0, index=df.index, dtype=int)
    out = out.mask(bull, 1).mask(bear, -1)
    return out


def _resample_ohlc(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = (
        df5.resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    return _prep(ohlc)


def _equal_levels(levels: list[float], atr: float) -> list[float]:
    if not levels:
        return []
    tol = max(EQUAL_ATR * atr, 1e-9)
    levels = sorted(levels)
    clusters: list[list[float]] = [[levels[0]]]
    for v in levels[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    # Prefer equal-touch clusters (2+); keep singles as swing liq too
    out = [float(np.mean(c)) for c in clusters]
    return out


def _collect_15m_sweeps(df15: pd.DataFrame) -> list[dict[str, Any]]:
    """15m swing/equal liquidity swept + reclaim close."""
    sweeps: list[dict[str, Any]] = []
    lows: list[tuple[int, float]] = []
    highs: list[tuple[int, float]] = []
    for i in range(SWING, len(df15) - SWING):
        if _swing_low(df15, i):
            lows.append((i, float(df15.iloc[i]["low"])))
        if _swing_high(df15, i):
            highs.append((i, float(df15.iloc[i]["high"])))

    for ei in range(50, len(df15) - 1):
        row = df15.iloc[ei]
        atr = float(row["atr"] or 0) or 1e-6
        h, l, c, o = (
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["open"]),
        )
        rlo = [lv for si, lv in lows if 3 <= ei - si <= LIQ_LOOKBACK]
        rhi = [hv for si, hv in highs if 3 <= ei - si <= LIQ_LOOKBACK]
        lo_lvls = _equal_levels(rlo, atr)
        hi_lvls = _equal_levels(rhi, atr)

        # BUY: sweep lows (liquidity below), reclaim close above level, bullish close
        for liq in lo_lvls:
            if l < liq - 0.02 * atr and c > liq and c > o:
                sl = l - 0.08 * atr
                if abs(c - sl) > 0.12 * atr:
                    sweeps.append(
                        {
                            "t": df15.index[ei],
                            "side": "BUY",
                            "liq": liq,
                            "sl": sl,
                            "sweep_extreme": l,
                        }
                    )
                    break
        else:
            for liq in hi_lvls:
                if h > liq + 0.02 * atr and c < liq and c < o:
                    sl = h + 0.08 * atr
                    if abs(c - sl) > 0.12 * atr:
                        sweeps.append(
                            {
                                "t": df15.index[ei],
                                "side": "SELL",
                                "liq": liq,
                                "sl": sl,
                                "sweep_extreme": h,
                            }
                        )
                        break
    return sweeps


def _is_engulf(row: pd.Series, prev: pd.Series, side: str) -> bool:
    o, c = float(row["open"]), float(row["close"])
    po, pc = float(prev["open"]), float(prev["close"])
    if side == "BUY":
        return c > o and pc < po and c >= po and o <= pc
    return c < o and pc > po and c <= po and o >= pc


def _collect_pa_entries(
    df5: pd.DataFrame,
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    df25: pd.DataFrame,
) -> list[dict[str, Any]]:
    sweeps = _collect_15m_sweeps(df15)
    if not sweeps:
        return []

    bias_1h = _bias_series(df1h)
    bias_25 = _bias_series(df25)
    # asof join helpers
    b1h = bias_1h.reindex(df5.index, method="ffill").fillna(0).astype(int)
    b25 = bias_25.reindex(df5.index, method="ffill").fillna(0).astype(int)

    # map sweep times → index in 5m
    sweep_events: list[tuple[int, dict]] = []
    for sw in sweeps:
        # first 5m bar at/after sweep close
        pos = df5.index.searchsorted(sw["t"], side="left")
        if pos >= len(df5):
            continue
        sweep_events.append((int(pos), sw))
    if not sweep_events:
        return []

    entries: list[dict[str, Any]] = []
    used_sweep: set[int] = set()
    si = 0  # pointer into sweep_events
    for i in range(40, len(df5) - 1):
        # advance / collect active sweeps for this bar
        while si < len(sweep_events) and sweep_events[si][0] < i - SWEEP_MAX_5M_BARS:
            si += 1
        active = [
            (sp, sw)
            for sp, sw in sweep_events[si:]
            if sp <= i <= sp + SWEEP_MAX_5M_BARS and sp not in used_sweep
        ]
        if not active:
            continue

        row, prev = df5.iloc[i], df5.iloc[i - 1]
        atr = float(row["atr"] or 0) or 1e-6
        b1, b2 = int(b1h.iloc[i]), int(b25.iloc[i])
        if b1 == 0 or b1 != b2:
            continue  # need 1h + 25m same non-neutral bias

        side = "BUY" if b1 > 0 else "SELL"
        # only use sweeps matching HTF bias direction
        matched = [(sp, sw) for sp, sw in active if sw["side"] == side]
        if not matched:
            continue
        if not _is_engulf(row, prev, side):
            continue

        # nearest prior matching sweep
        sp, sw = matched[-1]
        c = float(row["close"])
        # SL: beyond 15m sweep wick, or 5m wick if wider
        if side == "BUY":
            sl = min(float(sw["sl"]), float(row["low"]) - 0.08 * atr)
            if sl >= c:
                continue
        else:
            sl = max(float(sw["sl"]), float(row["high"]) + 0.08 * atr)
            if sl <= c:
                continue
        if abs(c - sl) <= 0.12 * atr:
            continue

        entries.append(
            {
                "ei": i,
                "side": side,
                "entry": c,
                "sl": sl,
                "entry_t": df5.index[i],
                "liq": sw["liq"],
                "sweep_t": str(sw["t"])[:16],
            }
        )
        used_sweep.add(sp)
    return entries


def _pack_cell(
    name: str,
    df: pd.DataFrame,
    entries: list[dict],
    rr: float,
    *,
    note: str = "",
) -> dict[str, Any]:
    tr = _simulate_fast(df, entries, rr)
    bat = _battery_from_trades(tr, len(df))
    full = bat.get("full") or {}
    tests = bat.get("tests") or {}
    return {
        "name": name,
        "note": note,
        "rr": rr,
        "exec_tf": "see_note",
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
                "win_rate": (tests.get(k) or {}).get("win_rate"),
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
        "mc_prob_profit_raw": tests.get("mc_prob_profit"),
        "span": {
            "bars": len(df),
            "start": str(df.index[0].date()) if len(df) else None,
            "end": str(df.index[-1].date()) if len(df) else None,
        },
    }


def _load_gold_1y() -> dict[str, pd.DataFrame]:
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    clear_hist_cache()
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    # Prefer last bar in hist if wall clock is ahead of CSV
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    frames: dict[str, pd.DataFrame] = {}
    for tf in ("5m", "15m"):
        if not has_local_hist("GOLD", tf) and not has_local_hist("GOLD", "1m"):
            raise RuntimeError(f"no local GOLD {tf}")
        df = load_local_ohlcv(
            "GOLD", tf=tf, start=start.isoformat(), end=end.isoformat()
        )
        need = [c for c in ("open", "high", "low", "close") if c in df.columns]
        df = _prep(df[need].dropna())
        # align window to actual data end
        data_end = df.index[-1]
        data_start = data_end - pd.Timedelta(days=WINDOW_DAYS)
        df = df[df.index >= data_start]
        frames[tf] = df
        _log(
            f"  load GOLD {tf}: {len(df):,} bars {df.index[0].date()}→{df.index[-1].date()}"
        )

    df5 = frames["5m"]
    frames["1h"] = _resample_ohlc(df5, "1h")
    frames["25m"] = _resample_ohlc(df5, "25min")
    _log(
        f"  resample 1h: {len(frames['1h']):,} · 25m: {len(frames['25m']):,}"
    )
    return frames


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass
    _log("=== GOLD PA liq-sweep+engulf vs don_100 (Dukascopy 1y) ===")
    frames = _load_gold_1y()
    df5, df15 = frames["5m"], frames["15m"]
    df1h, df25 = frames["1h"], frames["25m"]

    _log("Collecting PA entries (15m sweep → 1h/25m bias → 5m engulf)…")
    pa_entries = _collect_pa_entries(df5, df15, df1h, df25)
    _log(f"  PA raw entries: {len(pa_entries)}")

    _log("Collecting don_100 entries on 15m…")
    don_entries = _collect_breakout(df15, lookback=100)
    _log(f"  don_100 raw entries: {len(don_entries)}")

    cells: list[dict[str, Any]] = []
    for rr in RRS:
        cell = _pack_cell(
            "pa_liq_1h25_engulf5",
            df5,
            pa_entries,
            rr,
            note="15m liq sweep + 1h&25m EMA bias + 5m engulf; exec on 5m",
        )
        cells.append(cell)
        _log(
            f"  PA RR={rr}: n={cell['trades']} pnl=${cell['pnl']:,.0f} "
            f"E[R]={cell['exp_r']} verdict={cell['verdict']} "
            f"WF={cell['wf_green']}/{cell['wf_n']} MC={cell['mc_prob_profit']}%"
        )

    for rr in RRS:
        cell = _pack_cell(
            "don_100",
            df15,
            don_entries,
            rr,
            note="live wired brain; exec on 15m",
        )
        cells.append(cell)
        _log(
            f"  don_100 RR={rr}: n={cell['trades']} pnl=${cell['pnl']:,.0f} "
            f"E[R]={cell['exp_r']} verdict={cell['verdict']} "
            f"WF={cell['wf_green']}/{cell['wf_n']} MC={cell['mc_prob_profit']}%"
        )

    # Head-to-head at wired RR=4
    pa4 = next(c for c in cells if c["name"] == "pa_liq_1h25_engulf5" and c["rr"] == 4.0)
    don4 = next(c for c in cells if c["name"] == "don_100" and c["rr"] == 4.0)
    best_pa = max(
        (c for c in cells if c["name"] == "pa_liq_1h25_engulf5"),
        key=lambda c: (c["verdict"] == "SURVIVES", c["pnl"]),
    )
    best_don = max(
        (c for c in cells if c["name"] == "don_100"),
        key=lambda c: (c["verdict"] == "SURVIVES", c["pnl"]),
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": "GOLD",
        "window_days": WINDOW_DAYS,
        "risk_usd": RISK_USD,
        "process": {
            "liquidity_tf": "15m",
            "bias_tfs": ["1h", "25m"],
            "bias_rule": "ema20>ema50 & close>ema20 (bull) / inverse (bear); both TFs same",
            "entry_tf": "5m",
            "entry_rule": "classic engulfing after 15m sweep reclaim, within 90m",
            "sl": "beyond 15m sweep wick (or 5m wick if wider)",
        },
        "head_to_head_rr4": {
            "pa": {
                "pnl": pa4["pnl"],
                "n": pa4["trades"],
                "exp_r": pa4["exp_r"],
                "verdict": pa4["verdict"],
                "tests": pa4["tests"],
                "mc_prob_profit": pa4["mc_prob_profit"],
            },
            "don_100": {
                "pnl": don4["pnl"],
                "n": don4["trades"],
                "exp_r": don4["exp_r"],
                "verdict": don4["verdict"],
                "tests": don4["tests"],
                "mc_prob_profit": don4["mc_prob_profit"],
            },
            "pnl_delta_pa_minus_don": round(float(pa4["pnl"]) - float(don4["pnl"]), 2),
        },
        "best_by_pnl": {"pa": best_pa, "don_100": best_don},
        "cells": cells,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _log("")
    _log("### HEAD-TO-HEAD @ RR=4 ($200/R)")
    _log(
        f"PA process:  n={pa4['trades']}  PnL=${pa4['pnl']:,.0f}  "
        f"E[R]={pa4['exp_r']}  {pa4['verdict']}  MC={pa4['mc_prob_profit']}%"
    )
    _log(
        f"don_100:     n={don4['trades']}  PnL=${don4['pnl']:,.0f}  "
        f"E[R]={don4['exp_r']}  {don4['verdict']}  MC={don4['mc_prob_profit']}%"
    )
    _log(f"Δ PnL (PA − don): ${payload['head_to_head_rr4']['pnl_delta_pa_minus_don']:,.0f}")
    _log("")
    _log("PA @ RR=4 — 8 tests:")
    for k, v in pa4["tests"].items():
        _log(f"  {k}: n={v['n']} pnl=${v['pnl']} E[R]={v['exp_r']}")
    _log("don_100 @ RR=4 — 8 tests:")
    for k, v in don4["tests"].items():
        _log(f"  {k}: n={v['n']} pnl=${v['pnl']} E[R]={v['exp_r']}")
    _log(f"\nWrote {OUT} ({payload['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
