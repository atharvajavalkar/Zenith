#!/usr/bin/env python3
"""RESEARCH ONLY — GOLD 15m Order Block + FVG, enter on 5m. No live tweaks.

Arms (RR = live GOLD SYMBOL_RR):
  1. ob_fvg_15_enter_5  — 15m OB+FVG zones; 5m tap+reclaim entry; SL/TP on 5m
  2. fvg_15_enter_5     — 15m FVG only → 5m entry
  3. ob_15_enter_5      — 15m OB only → 5m entry
  4. fib_1h_current     — current fib 1h proxy (same ~55d window)
  5. fib_1h_long        — current fib 1h (182d desk proxy)

Full pack: OOS / WF / MC / Lévy / Malliavin / BM / Itô / Martingale / HJM.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import (  # noqa: E402
    RISK_USD,
    _fetch_1h,
    _prep,
    _stats,
    monte_carlo,
    verdict,
)
from analysis._htf_trend_5m_exit_bt import _fetch_5m  # noqa: E402
from analysis._rr_best_by_pair import _collect_fib, simulate  # noqa: E402
from analysis.levy_malliavin import levy_malliavin_pack  # noqa: E402
from analysis.stochastic_pack import stochastic_pack  # noqa: E402
from config import symbol_rr  # noqa: E402
from indicators.prepare import prepare_df  # noqa: E402

OUT = "/tmp/gold_ob_fvg_15_enter5.json"
RR = float(symbol_rr("GOLD", default=4.0))
DISP_ATR = 1.15
FVG_LOOKBACK_BARS = 30
OB_MAX_AGE_BARS = 40  # 15m bars (~10h)
SL_PAD = 0.08


def _resample_ohlc(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    m = {"5m": "5min", "15m": "15min", "1h": "1h"}[rule]
    out = pd.DataFrame(
        {
            "open": df5["open"].resample(m).first(),
            "high": df5["high"].resample(m).max(),
            "low": df5["low"].resample(m).min(),
            "close": df5["close"].resample(m).last(),
        }
    ).dropna()
    return prepare_df(out)


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(hour=23, minute=59, tzinfo=timezone.utc)
    return df[(df.index >= start_dt) & (df.index <= end_dt)].copy()


def _build_zones(df15: pd.DataFrame) -> list[dict]:
    """Scan 15m for OB and FVG events; return active zone records keyed by bar index."""
    zones: list[dict] = []
    for i in range(5, len(df15)):
        row = df15.iloc[i]
        atr = float(row["ATR"] or 0) or 1e-6
        # --- FVG formed on this bar (3-candle gap) ---
        bull_lo = row.get("BULL_FVG_LOW")
        bull_hi = row.get("BULL_FVG_HIGH")
        bear_lo = row.get("BEAR_FVG_LOW")
        bear_hi = row.get("BEAR_FVG_HIGH")
        if pd.notna(bull_lo) and pd.notna(bull_hi):
            zones.append(
                {
                    "kind": "FVG",
                    "side": "BUY",
                    "i": i,
                    "t": df15.index[i],
                    "low": float(bull_lo),
                    "high": float(bull_hi),
                    "mid": 0.5 * (float(bull_lo) + float(bull_hi)),
                }
            )
        if pd.notna(bear_lo) and pd.notna(bear_hi):
            zones.append(
                {
                    "kind": "FVG",
                    "side": "SELL",
                    "i": i,
                    "t": df15.index[i],
                    "low": float(bear_lo),
                    "high": float(bear_hi),
                    "mid": 0.5 * (float(bear_lo) + float(bear_hi)),
                }
            )

        # --- Order Block: last opposing candle before displacement ---
        body = float(row["BODY"] or 0)
        bullish_disp = (
            float(row["close"]) > float(row["open"])
            and body >= DISP_ATR * atr
            and float(row["BODY_RATIO"] or 0) >= 0.55
        )
        bearish_disp = (
            float(row["close"]) < float(row["open"])
            and body >= DISP_ATR * atr
            and float(row["BODY_RATIO"] or 0) >= 0.55
        )
        if bullish_disp:
            # last bearish candle in prior 1–4 bars
            for k in range(1, 5):
                j = i - k
                if j < 1:
                    break
                prev = df15.iloc[j]
                if float(prev["close"]) < float(prev["open"]):
                    zones.append(
                        {
                            "kind": "OB",
                            "side": "BUY",
                            "i": i,
                            "t": df15.index[i],
                            "low": float(prev["low"]),
                            "high": float(prev["high"]),
                            "mid": 0.5 * (float(prev["low"]) + float(prev["high"])),
                            "ob_i": j,
                        }
                    )
                    break
        if bearish_disp:
            for k in range(1, 5):
                j = i - k
                if j < 1:
                    break
                prev = df15.iloc[j]
                if float(prev["close"]) > float(prev["open"]):
                    zones.append(
                        {
                            "kind": "OB",
                            "side": "SELL",
                            "i": i,
                            "t": df15.index[i],
                            "low": float(prev["low"]),
                            "high": float(prev["high"]),
                            "mid": 0.5 * (float(prev["low"]) + float(prev["high"])),
                            "ob_i": j,
                        }
                    )
                    break
    return zones


def _active_zones(
    zones: list[dict],
    pos15: int,
    *,
    kind: Optional[str] = None,
    require_both: bool = False,
) -> list[dict]:
    """Zones still 'fresh' at 15m index pos15 (not too old)."""
    lo = pos15 - OB_MAX_AGE_BARS
    cand = [z for z in zones if lo <= z["i"] <= pos15]
    if kind:
        cand = [z for z in cand if z["kind"] == kind]
    if not require_both:
        return cand
    # Need at least one OB and one FVG same side overlapping in age window
    out = []
    for side in ("BUY", "SELL"):
        obs = [z for z in cand if z["kind"] == "OB" and z["side"] == side]
        fvgs = [z for z in cand if z["kind"] == "FVG" and z["side"] == side]
        if not obs or not fvgs:
            continue
        # Prefer most recent of each; expose both as a combined zone (union / mid)
        ob, fvg = obs[-1], fvgs[-1]
        out.append(
            {
                "kind": "OB+FVG",
                "side": side,
                "i": max(ob["i"], fvg["i"]),
                "t": max(ob["t"], fvg["t"]),
                "low": min(ob["low"], fvg["low"]),
                "high": max(ob["high"], fvg["high"]),
                "mid": 0.5 * (ob["mid"] + fvg["mid"]),
                "ob": ob,
                "fvg": fvg,
            }
        )
    return out


def collect_entries_5m(
    df15: pd.DataFrame,
    df5: pd.DataFrame,
    zones: list[dict],
    *,
    mode: str,
) -> list[dict]:
    """mode: 'both' | 'fvg' | 'ob' — tap zone on 5m + reclaim close."""
    entries: list[dict] = []
    for ei in range(40, len(df5) - 1):
        ts = df5.index[ei]
        pos15 = int(df15.index.searchsorted(ts, side="right")) - 1
        if pos15 < 10:
            continue
        if mode == "both":
            active = _active_zones(zones, pos15, require_both=True)
        elif mode == "fvg":
            active = _active_zones(zones, pos15, kind="FVG")
        else:
            active = _active_zones(zones, pos15, kind="OB")
        if not active:
            continue

        row = df5.iloc[ei]
        h, l, c, o = (
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["open"]),
        )
        atr5 = float(row["ATR"] or 0) or max(h - l, 1e-6)

        signal = None
        chosen = None
        for z in reversed(active):  # prefer freshest
            # Tap into zone + close reclaim in direction
            tapped = l <= z["high"] and h >= z["low"]
            if not tapped:
                continue
            if z["side"] == "BUY" and c > z["mid"] and c > o and l <= z["high"]:
                signal = "BUY"
                entry = c
                sl = min(l, z["low"]) - SL_PAD * atr5
                chosen = z
                break
            if z["side"] == "SELL" and c < z["mid"] and c < o and h >= z["low"]:
                signal = "SELL"
                entry = c
                sl = max(h, z["high"]) + SL_PAD * atr5
                chosen = z
                break
        if not signal or chosen is None:
            continue
        risk = abs(entry - sl)
        if risk <= 0.12 * atr5 or risk > 3.0 * atr5:
            continue
        entries.append(
            {
                "ei": ei,
                "side": signal,
                "entry": float(entry),
                "sl": float(sl),
                "entry_t": ts,
                "htf_i": pos15,
                "zone_kind": chosen["kind"],
            }
        )
    return entries


def simulate_5m(df5: pd.DataFrame, entries: list[dict], rr: float) -> list[dict]:
    trades: list[dict] = []
    busy = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < busy:
            continue
        side = e["side"]
        entry = float(e["entry"])
        sl0 = float(e["sl"])
        risk = abs(entry - sl0)
        if risk <= 0:
            continue
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk
        exit_r = None
        exit_j = ei
        for j in range(ei + 1, len(df5)):
            row = df5.iloc[j]
            hh, ll = float(row["high"]), float(row["low"])
            hit_sl = (side == "BUY" and ll <= sl) or (side == "SELL" and hh >= sl)
            hit_tp = (side == "BUY" and hh >= tp) or (side == "SELL" and ll <= tp)
            if hit_sl or hit_tp:
                exit_r = -1.0 if hit_sl else float(rr)
                if hit_sl and hit_tp:
                    exit_r = -1.0
                exit_j = j
                break
        if exit_r is None:
            continue
        trades.append(
            {
                "i": int(e.get("htf_i", ei)),
                "entry_t": str(e["entry_t"])[:16],
                "side": side,
                "r": round(exit_r, 4),
                "pnl": round(exit_r * RISK_USD, 2),
            }
        )
        busy = exit_j + 2
    return trades


def _battery(full_tr, df_ito, n_fold, arm, label) -> dict:
    split = int(n_fold * 0.70)

    def slice_tr(start_i=0, end_i=None):
        end_i = end_i if end_i is not None else 10**9
        return [t for t in full_tr if start_i <= int(t["i"]) <= end_i]

    full = _stats(full_tr)
    oos = _stats(slice_tr(start_i=split))
    wf = []
    for k in range(4):
        a = int(n_fold * (0.40 + 0.15 * k))
        b = int(n_fold * (0.55 + 0.15 * k)) - 1
        if b <= a or a < 40:
            continue
        wf.append(
            {
                "fold": k + 1,
                "train": _stats(slice_tr(end_i=a - 1)),
                "oos": _stats(slice_tr(start_i=a, end_i=b)),
            }
        )
    torture = {
        "spread_0.15R": _stats(
            [
                {
                    **t,
                    "r": round(t["r"] - 0.15, 4),
                    "pnl": round((t["r"] - 0.15) * RISK_USD, 2),
                }
                for t in full_tr
            ]
        )
    }
    mc = monte_carlo(full_tr, paths=1200)
    lm = levy_malliavin_pack(full_tr)
    stoch = stochastic_pack(df_ito, full_tr)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
    return {
        "arm": arm,
        "label": label,
        "rr": RR,
        "full": full,
        "oos": oos,
        "wf_green": f"{wf_g}/{len(wf)}",
        "monte_carlo": mc,
        "levy": lm["levy"],
        "malliavin": lm["malliavin"],
        "brownian": stoch["brownian"],
        "ito": stoch["ito"],
        "martingale": stoch["martingale"],
        "hjm": stoch["hjm"],
        "verdict": v,
        "pnl_usd": full.get("pnl"),
        "oos_pnl_usd": oos.get("pnl"),
    }


def _run_arm(name, label, entries, trades, df_ito, n_fold, results):
    print(f"\n{name} entries={len(entries)} fills={len(trades)}", flush=True)
    row = _battery(trades, df_ito, n_fold, name, label)
    row["raw_entries"] = len(entries)
    row["fills"] = len(trades)
    f, o = row["full"], row["oos"]
    print(
        f"  n={f['n']} E[R]={f['exp_r']} pnl=${f['pnl']} | "
        f"OOS E[R]={o['exp_r']} pnl=${o['pnl']} | WF {row['wf_green']} | "
        f"MC={row['monte_carlo'].get('prob_profit')}% "
        f"Lévy={row['levy'].get('prob_profit')}% | {row['verdict']}",
        flush=True,
    )
    results.append(row)


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_5 = (end - timedelta(days=55)).isoformat()
    start_1h = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"GOLD OB+FVG 15m → enter 5m · {start_5}→{end_s} · RR=1:{RR:g} · "
        f"${RISK_USD}/R · NO LIVE TWEAK",
        flush=True,
    )

    raw5 = _fetch_5m("GOLD", start_5, end_s)
    clipped = _clip(raw5, start_5, end_s)
    # prepare_df needs volume-less OHLC; ensure columns
    for col in ("open", "high", "low", "close"):
        if col not in clipped.columns:
            raise RuntimeError(f"missing {col}")
    df5 = _resample_ohlc(clipped, "5m")
    df15 = _resample_ohlc(clipped, "15m")
    zones = _build_zones(df15)
    n_ob = sum(1 for z in zones if z["kind"] == "OB")
    n_fvg = sum(1 for z in zones if z["kind"] == "FVG")
    print(
        f"5m={len(df5)} 15m={len(df15)} zones OB={n_ob} FVG={n_fvg}",
        flush=True,
    )

    results = []

    e_both = collect_entries_5m(df15, df5, zones, mode="both")
    tr_both = simulate_5m(df5, e_both, RR)
    _run_arm(
        "ob_fvg_15_enter_5",
        "15m OB+FVG both required → 5m entry",
        e_both,
        tr_both,
        df5,
        len(df15),
        results,
    )

    e_fvg = collect_entries_5m(df15, df5, zones, mode="fvg")
    tr_fvg = simulate_5m(df5, e_fvg, RR)
    _run_arm(
        "fvg_15_enter_5",
        "15m FVG only → 5m entry",
        e_fvg,
        tr_fvg,
        df5,
        len(df15),
        results,
    )

    e_ob = collect_entries_5m(df15, df5, zones, mode="ob")
    tr_ob = simulate_5m(df5, e_ob, RR)
    _run_arm(
        "ob_15_enter_5",
        "15m OB only → 5m entry",
        e_ob,
        tr_ob,
        df5,
        len(df15),
        results,
    )

    raw1 = _fetch_1h("GOLD", start_5, end_s)
    df1 = _prep(_clip(raw1, start_5, end_s))
    e1 = _collect_fib(df1)
    tr1 = simulate(df1, e1, RR)
    _run_arm(
        "fib_1h_current",
        "current fib 1h (55d)",
        e1,
        tr1,
        df1,
        len(df1),
        results,
    )

    raw1L = _fetch_1h("GOLD", start_1h, end_s)
    df1L = _prep(_clip(raw1L, start_1h, end_s))
    e1L = _collect_fib(df1L)
    tr1L = simulate(df1L, e1L, RR)
    _run_arm(
        "fib_1h_long",
        "current fib 1h (182d)",
        e1L,
        tr1L,
        df1L,
        len(df1L),
        results,
    )

    payload = {
        "note": "RESEARCH ONLY — no live config changes",
        "window": {"start": start_5, "end": end_s},
        "rr": RR,
        "results": results,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n### GOLD: 15m OB+FVG → ENTER 5m vs CURRENT FIB\n", flush=True)
    print(
        "| Arm | Entries | n | Full E[R] | Full PnL | OOS PnL | WF | MC% | Lévy% | BM% | Itô | Mart | HJM | Mall | Verdict |",
        flush=True,
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
        flush=True,
    )
    for r in results:
        f = r["full"]
        print(
            f"| {r['arm']} | {r.get('raw_entries')} | {f['n']} | {f['exp_r']} | "
            f"${f['pnl']:+.0f} | ${r['oos']['pnl']:+.0f} | {r['wf_green']} | "
            f"{r['monte_carlo'].get('prob_profit')} | {r['levy'].get('prob_profit')} | "
            f"{r['brownian'].get('prob_profit')} | {r['ito'].get('verdict')} | "
            f"{r['martingale'].get('verdict')} | {r['hjm'].get('verdict')} | "
            f"{r['malliavin'].get('fragility')} | **{r['verdict']}** |",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
