#!/usr/bin/env python3
"""GOLD: session-overlap (fresh sweep→reentry) vs lab winner vs fib live.

Also screens AUDUSD for a better brain than current bb_1.5 and compares.

Research only — does not change live config.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis._edge_survive_bt import RISK_USD, _prep, _stats, monte_carlo, verdict  # noqa: E402
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_breakout,
    _collect_ema,
    _collect_fib,
    _collect_sma,
    simulate,
)
from analysis.strategy_lab import (  # noqa: E402
    _collect_cta,
    _collect_macd,
    _collect_mom_roc,
    _collect_rsi,
    _collect_supertrend,
    _collect_vol_scaled_mom,
)
from analysis.strategy_lab_hunt import _bb_std, _keltner_mult, _stoch_band  # noqa: E402
from analysis.strategy_lab_intraday_hunt import _load_intraday  # noqa: E402
from config import symbol_rr  # noqa: E402

OUT = Path(ROOT) / "data" / "gold_overlap_vs_lab.json"

# UTC minutes — Zenith overlap research baseline
EARLY = (7 * 60, 9 * 60)  # Tokyo–London
BRIDGE_END = 13 * 60
LATE = (13 * 60, 16 * 60)  # London–NY


def _collect_overlap_reentry(df: pd.DataFrame) -> list[dict]:
    """Fresh-boundary sweep + completed-bar re-entry → entries.

    SELL after fresh-high hunt + close back below early_high.
    BUY after fresh-low hunt + close back above early_low.
    SL just outside early extreme; TP via simulate RR.
    """
    if df is None or len(df) < 100:
        return []
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out["date"] = out.index.floor("D")
    out["minute"] = out.index.hour * 60 + out.index.minute
    atr = out["atr"] if "atr" in out.columns else out.get("ATR")
    entries: list[dict] = []

    for date, day in out.groupby("date"):
        try:
            dow = int(pd.Timestamp(date).dayofweek)
        except Exception:
            dow = 0
        if dow >= 5:
            continue
        early = day[(day["minute"] >= EARLY[0]) & (day["minute"] < EARLY[1])]
        bridge = day[(day["minute"] >= EARLY[1]) & (day["minute"] < BRIDGE_END)]
        later = day[(day["minute"] >= LATE[0]) & (day["minute"] < LATE[1])]
        # ~90% of expected 5m bars: early 24, bridge 48, late 36
        if len(early) < 20 or len(bridge) < 40 or len(later) < 30:
            continue

        early_high = float(early["high"].max())
        early_low = float(early["low"].min())
        if early_high <= early_low:
            continue

        fresh_high = float(bridge["high"].max()) < early_high
        fresh_low = float(bridge["low"].min()) > early_low

        # Fresh high hunt + re-entry → SELL
        if fresh_high:
            hunt_i = None
            for i, row in later.iterrows():
                if float(row["high"]) >= early_high:
                    hunt_i = i
                    break
            if hunt_i is not None:
                after = later.loc[later.index >= hunt_i]
                for i, row in after.iterrows():
                    if float(row["close"]) < early_high:
                        a = float(atr.loc[i]) if atr is not None else (early_high - early_low) * 0.2
                        if not a or a != a:
                            a = (early_high - early_low) * 0.25
                        pad = 0.15 * a
                        sl = early_high + pad
                        entry = float(row["close"])
                        if sl > entry:
                            # map timestamp to integer position in full df
                            pos = int(out.index.get_loc(i))
                            if isinstance(pos, slice):
                                pos = pos.start
                            entries.append(
                                {
                                    "ei": pos,
                                    "side": "SELL",
                                    "entry": entry,
                                    "sl": sl,
                                    "entry_t": i,
                                    "tag": "fresh_high_reentry",
                                }
                            )
                        break

        # Fresh low hunt + re-entry → BUY
        if fresh_low:
            hunt_i = None
            for i, row in later.iterrows():
                if float(row["low"]) <= early_low:
                    hunt_i = i
                    break
            if hunt_i is not None:
                after = later.loc[later.index >= hunt_i]
                for i, row in after.iterrows():
                    if float(row["close"]) > early_low:
                        a = float(atr.loc[i]) if atr is not None else (early_high - early_low) * 0.2
                        if not a or a != a:
                            a = (early_high - early_low) * 0.25
                        pad = 0.15 * a
                        sl = early_low - pad
                        entry = float(row["close"])
                        if sl < entry:
                            pos = int(out.index.get_loc(i))
                            if isinstance(pos, slice):
                                pos = pos.start
                            entries.append(
                                {
                                    "ei": pos,
                                    "side": "BUY",
                                    "entry": entry,
                                    "sl": sl,
                                    "entry_t": i,
                                    "tag": "fresh_low_reentry",
                                }
                            )
                        break

    entries.sort(key=lambda e: e["ei"])
    return entries


def _battery(df: pd.DataFrame, entries: list[dict], rr: float, label: str) -> dict[str, Any]:
    n = len(df)

    def gen(start_i=0, end_i=None, **kw):
        spread_r = float(kw.get("spread_r") or 0)
        skip_p = float(kw.get("skip_p") or 0)
        sl_widen = float(kw.get("sl_widen") or 1.0)
        rng = kw.get("rng") or random.Random(0)
        tr = simulate(df, entries, rr, start_i=start_i, end_i=end_i)
        if sl_widen != 1.0:
            out = []
            for t in tr:
                r = float(t["r"])
                if r < 0:
                    r = -1.0 * sl_widen
                out.append({**t, "r": round(r, 4), "pnl": round(r * RISK_USD, 2)})
            tr = out
        if spread_r > 0:
            tr = [
                {
                    **t,
                    "r": round(float(t["r"]) - spread_r, 4),
                    "pnl": round((float(t["r"]) - spread_r) * RISK_USD, 2),
                }
                for t in tr
            ]
        if skip_p > 0 and tr:
            tr = [t for t in tr if rng.random() >= skip_p]
        return tr

    full_tr = gen()
    full = _stats(full_tr)
    split = int(n * 0.70)
    oos = _stats(gen(start_i=split))
    wf = []
    for k in range(4):
        ts = int(n * (0.40 + 0.15 * k))
        te = int(n * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 80:
            continue
        wf.append(
            {
                "fold": k + 1,
                "train": _stats(gen(end_i=ts - 1)),
                "oos": _stats(gen(start_i=ts, end_i=te)),
            }
        )
    torture = {
        "spread_0.15R": _stats(gen(spread_r=0.15)),
        "skip_30pct": _stats(gen(skip_p=0.30, rng=random.Random(7))),
        "sl_widen_20pct": _stats(gen(sl_widen=1.20)),
    }
    mc = monte_carlo(full_tr)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
    print(
        f"  [{label}] n={full['n']} E[R]={full['exp_r']} OOS={oos['exp_r']} "
        f"WF={wf_g}/{len(wf)} MC={mc.get('prob_profit')}% → {v}",
        flush=True,
    )
    return {
        "label": label,
        "entries": len(entries),
        "full": full,
        "oos": oos,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": wf_g,
        "wf_n": len(wf),
        "score": round(
            (30 if v == "SURVIVES" else 10 if v == "FRAGILE" else -20)
            + float(oos["exp_r"]) * 25
            + float(mc.get("prob_profit") or 50) / 20
            + min(15.0, full["n"] / 10.0),
            3,
        ),
    }


AUD_CANDIDATES: list[tuple[str, Callable]] = [
    ("bb_1.5", lambda d: _bb_std(d, 1.5)),
    ("bb_2", lambda d: _bb_std(d, 2.0)),
    ("bb_2.5", lambda d: _bb_std(d, 2.5)),
    ("rsi", _collect_rsi),
    ("ema_tight", lambda d: _collect_ema(d, pull_atr=0.75, disp_atr=0.70)),
    ("ema_pull", lambda d: _collect_ema(d, pull_atr=1.0, disp_atr=0.55)),
    ("sma_cross", _collect_sma),
    ("don_20", lambda d: _collect_breakout(d, 20)),
    ("cta_10_40", lambda d: _collect_cta(d, 10, 40)),
    ("cta_12_26", lambda d: _collect_cta(d, 12, 26)),
    ("st_2", lambda d: _collect_supertrend(d, 2.0)),
    ("st_3", lambda d: _collect_supertrend(d, 3.0)),
    ("volmom", _collect_vol_scaled_mom),
    ("macd", _collect_macd),
    ("kelt_1.5", lambda d: _keltner_mult(d, 1.5)),
    ("stoch_20", lambda d: _stoch_band(d, 20, 80)),
    ("roc_20", lambda d: _collect_mom_roc(d, 20)),
]


def main() -> int:
    t0 = time.time()
    gold_rr = float(symbol_rr("GOLD", default=4.0))
    aud_rr = float(symbol_rr("AUDUSD", default=1.5))

    print("=== GOLD: overlap vs lab-winner(cta_10_40) vs prior-live(fib) ===\n", flush=True)
    g_frames = _load_intraday("GOLD")
    if "5m" not in g_frames:
        raise SystemExit("no GOLD 5m")
    g5 = g_frames["5m"]
    g15 = g_frames.get("15m")
    if g15 is None:
        # resample
        g15 = _prep(
            pd.DataFrame(
                {
                    "open": g5["open"].resample("15min").first(),
                    "high": g5["high"].resample("15min").max(),
                    "low": g5["low"].resample("15min").min(),
                    "close": g5["close"].resample("15min").last(),
                }
            ).dropna()
        )

    gold_rows = []
    # Overlap on 5m (native research TF)
    ov_entries = _collect_overlap_reentry(g5)
    gold_rows.append(
        {
            **_battery(g5, ov_entries, gold_rr, "overlap_fresh_reentry@5m"),
            "strategy": "overlap_fresh_reentry",
            "tf": "5m",
            "role": "overlap_package",
        }
    )
    # Lab winner / current live: cta_10_40 @ 15m
    cta_e = _collect_cta(g15, 10, 40)
    gold_rows.append(
        {
            **_battery(g15, cta_e, gold_rr, "cta_10_40@15m"),
            "strategy": "cta_10_40",
            "tf": "15m",
            "role": "lab_winner_and_current_live",
        }
    )
    # Prior live fib on 15m
    fib_e = _collect_fib(g15)
    gold_rows.append(
        {
            **_battery(g15, fib_e, gold_rr, "fib_liq@15m"),
            "strategy": "fib_liq",
            "tf": "15m",
            "role": "prior_live_brain",
        }
    )
    # Also fib on 5m for fairer TF
    fib5 = _collect_fib(g5)
    gold_rows.append(
        {
            **_battery(g5, fib5, gold_rr, "fib_liq@5m"),
            "strategy": "fib_liq",
            "tf": "5m",
            "role": "prior_live_on_5m",
        }
    )

    gold_rows.sort(key=lambda r: float(r.get("score") or -1e9), reverse=True)
    gold_best = gold_rows[0]

    print("\n=== AUDUSD: screen alternatives vs current bb_1.5 ===\n", flush=True)
    a_frames = _load_intraday("AUDUSD")
    aud_results = []
    for sid, fn in AUD_CANDIDATES:
        for tf in ("15m", "5m", "1m", "3m"):
            if tf not in a_frames:
                continue
            df = a_frames[tf]
            try:
                entries = fn(df)
            except Exception:
                continue
            if len(entries) < 4:
                continue
            row = _battery(df, entries, aud_rr, f"{sid}@{tf}")
            aud_results.append(
                {
                    **row,
                    "strategy": sid,
                    "tf": tf,
                    "role": "current_live" if sid == "bb_1.5" and tf == "15m" else "candidate",
                }
            )

    aud_results.sort(key=lambda r: float(r.get("score") or -1e9), reverse=True)
    current_aud = next(
        (r for r in aud_results if r["strategy"] == "bb_1.5" and r["tf"] == "15m"),
        None,
    )
    # Best alternative that is not bb_1.5@15m
    alt = next(
        (
            r
            for r in aud_results
            if not (r["strategy"] == "bb_1.5" and r["tf"] == "15m")
            and r.get("verdict") != "DEAD"
        ),
        aud_results[0] if aud_results else None,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "gold_overlap_vs_lab_and_aud_hunt",
        "risk_usd": RISK_USD,
        "gold": {
            "rr": gold_rr,
            "note": (
                "Current live GOLD brain IS lab winner cta_10_40@15m. "
                "fib_liq = prior live. overlap = Zenith session package as tradable re-entry rule."
            ),
            "results": gold_rows,
            "winner": {
                "strategy": gold_best.get("strategy"),
                "tf": gold_best.get("tf"),
                "role": gold_best.get("role"),
                "verdict": gold_best.get("verdict"),
                "score": gold_best.get("score"),
            },
        },
        "audusd": {
            "rr": aud_rr,
            "current_live": current_aud,
            "best_alternative": alt,
            "top10": aud_results[:10],
            "screened": len(aud_results),
        },
        "elapsed_s": round(time.time() - t0, 1),
        "note": "Research only — does not change SYMBOL_ENTRY_BRAIN",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### GOLD HEAD-TO-HEAD\n", flush=True)
    print(
        f"{'Role':28} {'Strategy':22} {'TF':5} {'Verdict':9} {'n':>5} "
        f"{'OOS E[R]':>8} {'MC%':>6} {'Score':>7}",
        flush=True,
    )
    print("-" * 100, flush=True)
    for r in gold_rows:
        mc = r.get("monte_carlo") or {}
        print(
            f"{r.get('role',''):28} {r.get('strategy',''):22} {r.get('tf',''):5} "
            f"{r['verdict']:9} {r['full']['n']:5} {r['oos']['exp_r']:8.3f} "
            f"{mc.get('prob_profit', '—')!s:>6} {r['score']:7.2f}",
            flush=True,
        )
    print(f"\nGOLD winner → {gold_best['strategy']}@{gold_best['tf']} ({gold_best['verdict']})", flush=True)

    print("\n### AUDUSD\n", flush=True)
    if current_aud:
        mc = current_aud.get("monte_carlo") or {}
        print(
            f"Current live bb_1.5@15m → {current_aud['verdict']} "
            f"OOS E[R]={current_aud['oos']['exp_r']} MC={mc.get('prob_profit')}%",
            flush=True,
        )
    if alt:
        mc = alt.get("monte_carlo") or {}
        print(
            f"Best alternative → {alt['strategy']}@{alt['tf']} {alt['verdict']} "
            f"OOS E[R]={alt['oos']['exp_r']} MC={mc.get('prob_profit')}% score={alt['score']}",
            flush=True,
        )
    print("Top 5:", flush=True)
    for r in aud_results[:5]:
        print(
            f"  {r['strategy']:12} @{r['tf']:3} {r['verdict']:9} "
            f"n={r['full']['n']:4} OOS={r['oos']['exp_r']:+.3f} score={r['score']:.1f}",
            flush=True,
        )
    print(f"\nWrote {OUT} in {payload['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
