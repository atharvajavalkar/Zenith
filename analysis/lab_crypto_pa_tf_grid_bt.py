#!/usr/bin/env python3
"""RESEARCH ONLY — ETHUSD / BTCUSD PA TF grid (same process as GOLD).

Process: liq sweep → dual-TF EMA bias → engulf entry
TFs: {1m, 3m, 5m, 15m} · liq ∈ {3m,5m,15m} · entry ≤ liq
Battery: 8 tests + MC over Dukascopy 1y
Also scores live wired BB baselines for comparison.

Does NOT change live SYMBOL_ENTRY_BRAIN.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD, _prep  # noqa: E402
from analysis.lab_gold_pa_liq_engulf_bt import RRS, WINDOW_DAYS, _resample_ohlc  # noqa: E402
from analysis.lab_gold_pa_tf_grid_bt import (  # noqa: E402
    LIQ_TFS,
    TFS,
    _collect_entries,
    _combo_iter,
    _fast_sweeps,
    _pack,
)
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402
from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_crypto_pa_tf_grid_bt.json"
LOG = Path(ROOT) / "data" / "lab_crypto_pa_tf_grid_bt.log"

SYMBOLS = ("ETHUSD", "BTCUSD")
# Live baselines (not changed)
WIRED = {
    "ETHUSD": {"brain": "bb_1.5", "tf": "1m", "rr": 4.0, "bb_mult": 1.5},
    "BTCUSD": {"brain": "bb_2.5", "tf": "1m", "rr": 2.4, "bb_mult": 2.5},
}
# GOLD's wired PA stack for cross-check
GOLD_STACK = {
    "liq_tf": "3m",
    "bias_a": "15m",
    "bias_b": "5m",
    "entry_tf": "1m",
    "rr": 2.0,
}


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _load_frames(sym: str) -> dict[str, pd.DataFrame]:
    clear_hist_cache()
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    frames: dict[str, pd.DataFrame] = {}
    for tf in TFS:
        if not has_local_hist(sym, tf):
            continue
        try:
            df = load_local_ohlcv(
                sym, tf=tf, start=start.isoformat(), end=end.isoformat()
            )
        except Exception as exc:
            _log(f"  load {sym} {tf} FAIL: {exc}")
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
            f"  load {sym} {tf}: {len(df):,} bars "
            f"{df.index[0].date()}→{df.index[-1].date()}"
        )
    for tf, rule in (("3m", "3min"), ("5m", "5min"), ("15m", "15min")):
        if tf not in frames:
            src = frames.get("1m") or frames.get("5m")
            if src is None:
                raise RuntimeError(f"{sym}: need 1m or 5m hist")
            frames[tf] = _resample_ohlc(src, rule)
            _log(f"  resample {sym} {tf}: {len(frames[tf]):,}")
    return frames


def _slim(c: dict) -> dict:
    return {
        "name": c["name"],
        "liq_tf": c.get("liq_tf"),
        "bias_a": c.get("bias_a"),
        "bias_b": c.get("bias_b"),
        "entry_tf": c.get("entry_tf"),
        "rr": c["rr"],
        "trades": c["trades"],
        "pnl": c["pnl"],
        "exp_r": c["exp_r"],
        "win_rate": c.get("win_rate"),
        "verdict": c["verdict"],
        "wf_green": c.get("wf_green"),
        "wf_n": c.get("wf_n"),
        "mc_prob_profit": c.get("mc_prob_profit"),
        "tests": c.get("tests"),
    }


def _rank_key(c: dict) -> tuple:
    return (
        1 if c.get("verdict") == "SURVIVES" else 0,
        float(c.get("pnl") or 0),
        float(c.get("exp_r") or 0),
        int(c.get("wf_green") or 0),
        float(c.get("mc_prob_profit") or 0),
    )


def _run_symbol(sym: str) -> dict[str, Any]:
    t0 = time.time()
    _log(f"\n{'='*60}\n===== {sym} PA TF GRID =====")
    frames = _load_frames(sym)

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

    ranked = sorted(cells, key=_rank_key, reverse=True)
    survives = [c for c in ranked if c.get("verdict") == "SURVIVES"]
    top15 = [_slim(c) for c in ranked[:15]]
    best = top15[0] if top15 else None

    # GOLD-stack cell
    gs = GOLD_STACK
    gold_cell = next(
        (
            c
            for c in ranked
            if c.get("liq_tf") == gs["liq_tf"]
            and c.get("bias_a") == gs["bias_a"]
            and c.get("bias_b") == gs["bias_b"]
            and c.get("entry_tf") == gs["entry_tf"]
            and abs(float(c["rr"]) - gs["rr"]) < 1e-9
        ),
        None,
    )

    # Wired BB baseline
    base = WIRED[sym]
    wired_cell = None
    if "1m" in frames:
        _log(f"  Collecting wired {base['brain']} @1m…")
        bb_entries = _bb_std(frames["1m"], float(base["bb_mult"]))
        _log(f"  BB entries: {len(bb_entries)}")
        wired_cell = _pack(
            f"wired_{base['brain']}",
            frames["1m"],
            bb_entries,
            float(base["rr"]),
            {
                "liq_tf": None,
                "bias_a": None,
                "bias_b": None,
                "entry_tf": "1m",
                "brain": base["brain"],
            },
        )
        _log(
            f"  WIRED {base['brain']} RR={base['rr']}: n={wired_cell['trades']} "
            f"pnl=${wired_cell['pnl']:,.0f} E[R]={wired_cell['exp_r']} "
            f"{wired_cell['verdict']}"
        )

    # All SURVIVES + every cell summary (compact) for deep report
    all_summary = [
        {
            "name": c["name"],
            "rr": c["rr"],
            "trades": c["trades"],
            "pnl": c["pnl"],
            "exp_r": c["exp_r"],
            "win_rate": c.get("win_rate"),
            "verdict": c["verdict"],
            "wf_green": c.get("wf_green"),
            "mc_prob_profit": c.get("mc_prob_profit"),
        }
        for c in ranked
    ]

    _log(f"\n### {sym} BEST")
    if best:
        _log(
            f"{best['name']} RR={best['rr']} → n={best['trades']} "
            f"PnL=${best['pnl']:,.0f} E[R]={best['exp_r']} {best['verdict']} "
            f"WF={best['wf_green']}/{best['wf_n']} MC={best['mc_prob_profit']}%"
        )
        for k, v in (best.get("tests") or {}).items():
            _log(f"  {k}: n={v.get('n')} pnl=${v.get('pnl')} E[R]={v.get('exp_r')}")

    return {
        "symbol": sym,
        "elapsed_s": round(time.time() - t0, 1),
        "combos": len(combos),
        "cells": len(cells),
        "survives_n": len(survives),
        "best": best,
        "top15": top15,
        "gold_stack_cell": _slim(gold_cell) if gold_cell else None,
        "wired_baseline": _slim(wired_cell) if wired_cell else None,
        "best_per_entry_tf": {
            etf: _slim(next(c for c in ranked if c.get("entry_tf") == etf))
            if any(c.get("entry_tf") == etf for c in ranked)
            else None
            for etf in TFS
        },
        "all_cells_summary": all_summary,
        "survives_full": [_slim(c) for c in survives[:40]],  # cap JSON size
    }


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass
    _log("=== CRYPTO PA TF grid (ETH+BTC) · Dukascopy 1y · RESEARCH ===")
    _log("Live map NOT changed.")

    results = []
    for sym in SYMBOLS:
        try:
            results.append(_run_symbol(sym))
        except Exception as exc:
            _log(f"FAIL {sym}: {exc}")
            import traceback

            traceback.print_exc()
            results.append({"symbol": sym, "error": str(exc)})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "live_wired": False,
        "window_days": WINDOW_DAYS,
        "risk_usd": RISK_USD,
        "process": "liq sweep → dual EMA bias → engulf (same as GOLD PA)",
        "tf_universe": list(TFS),
        "liq_tfs": list(LIQ_TFS),
        "rrs": list(RRS),
        "gold_stack_reference": GOLD_STACK,
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _log(f"\nWrote {OUT} ({payload['elapsed_s']}s)")

    # Final scoreboard
    _log("\n" + "=" * 60)
    _log("### SCOREBOARD")
    for r in results:
        if r.get("error"):
            _log(f"{r['symbol']}: ERROR {r['error']}")
            continue
        b, w, g = r.get("best"), r.get("wired_baseline"), r.get("gold_stack_cell")
        _log(f"\n{r['symbol']}  SURVIVES={r['survives_n']}/{r['cells']}")
        if b:
            _log(
                f"  BEST PA:  {b['name']} RR={b['rr']}  "
                f"PnL=${b['pnl']:,.0f}  {b['verdict']}  n={b['trades']}"
            )
        if g:
            _log(
                f"  GOLD stk: {g['name']} RR={g['rr']}  "
                f"PnL=${g['pnl']:,.0f}  {g['verdict']}  n={g['trades']}"
            )
        if w:
            _log(
                f"  WIRED BB: {w['name']} RR={w['rr']}  "
                f"PnL=${w['pnl']:,.0f}  {w['verdict']}  n={w['trades']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
