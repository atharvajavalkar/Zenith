#!/usr/bin/env python3
"""RESEARCH ONLY — USDCHF PA TF grid vs wired bb_2.5.

Same process / battery as GOLD/ETH/BTC PA grids:
  liq sweep → dual-TF EMA bias → engulf
  TFs {1m,3m,5m,15m} · liq ∈ {3m,5m,15m} · entry ≤ liq
  8 tests + MC · Dukascopy 1y

Baseline (live, not changed): bb_2.5 @ 1m, RR=4
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.lab_crypto_pa_tf_grid_bt import (  # noqa: E402
    GOLD_STACK,
    _load_frames,
    _log,
    _rank_key,
    _slim,
)
from analysis.lab_gold_pa_liq_engulf_bt import RRS, WINDOW_DAYS  # noqa: E402
from analysis.lab_gold_pa_tf_grid_bt import (  # noqa: E402
    LIQ_TFS,
    TFS,
    _collect_entries,
    _combo_iter,
    _fast_sweeps,
    _pack,
)
from analysis._edge_survive_bt import RISK_USD  # noqa: E402
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_usdchf_pa_tf_grid_bt.json"
LOG = Path(ROOT) / "data" / "lab_usdchf_pa_tf_grid_bt.log"

SYM = "USDCHF"
WIRED = {"brain": "bb_2.5", "tf": "1m", "rr": 4.0, "bb_mult": 2.5}


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass
    # patch _log to use this LOG
    import analysis.lab_crypto_pa_tf_grid_bt as crypto_mod

    def _log_local(msg: str) -> None:
        print(msg, flush=True)
        try:
            with LOG.open("a") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    global _log
    _log = _log_local  # type: ignore

    _log("=== USDCHF PA TF grid vs wired bb_2.5 · Dukascopy 1y · RESEARCH ===")
    _log("Live SYMBOL_ENTRY_BRAIN NOT changed.")

    frames = _load_frames(SYM)
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

    # Wired bb_2.5 @ RR=4 + RR sweep for fair compare
    wired_cells = []
    if "1m" in frames:
        _log(f"  Collecting wired {WIRED['brain']} @1m…")
        bb_entries = _bb_std(frames["1m"], float(WIRED["bb_mult"]))
        _log(f"  BB entries: {len(bb_entries)}")
        for rr in sorted(set(RRS) | {float(WIRED["rr"])}):
            wc = _pack(
                f"wired_{WIRED['brain']}",
                frames["1m"],
                bb_entries,
                float(rr),
                {
                    "liq_tf": None,
                    "bias_a": None,
                    "bias_b": None,
                    "entry_tf": "1m",
                    "brain": WIRED["brain"],
                },
            )
            wired_cells.append(wc)
            tag = "WIRED" if abs(rr - float(WIRED["rr"])) < 1e-9 else ""
            _log(
                f"  {WIRED['brain']} RR={rr} {tag}: n={wc['trades']} "
                f"pnl=${wc['pnl']:,.0f} E[R]={wc['exp_r']} {wc['verdict']} "
                f"WF={wc['wf_green']}/{wc['wf_n']}"
            )
    wired = next(
        (c for c in wired_cells if abs(float(c["rr"]) - float(WIRED["rr"])) < 1e-9),
        None,
    )

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

    _log("\n### BEST PA")
    if best:
        _log(
            f"{best['name']} RR={best['rr']} → n={best['trades']} "
            f"PnL=${best['pnl']:,.0f} E[R]={best['exp_r']} {best['verdict']} "
            f"WF={best['wf_green']}/{best['wf_n']} MC={best['mc_prob_profit']}%"
        )
        for k, v in (best.get("tests") or {}).items():
            _log(f"  {k}: n={v.get('n')} pnl=${v.get('pnl')} E[R]={v.get('exp_r')}")

    _log("\n### WIRED bb_2.5 @ RR=4")
    if wired:
        for k, v in (wired.get("tests") or {}).items():
            _log(f"  {k}: n={v.get('n')} pnl=${v.get('pnl')} E[R]={v.get('exp_r')}")

    delta = None
    if best and wired:
        delta = round(float(best["pnl"]) - float(wired["pnl"]), 2)
        _log(f"\nΔ BEST PA − wired: ${delta:,.0f}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "live_wired": False,
        "symbol": SYM,
        "window_days": WINDOW_DAYS,
        "risk_usd": RISK_USD,
        "process": "liq sweep → dual EMA bias → engulf (same as GOLD PA)",
        "tf_universe": list(TFS),
        "liq_tfs": list(LIQ_TFS),
        "rrs": list(RRS),
        "combos": len(combos),
        "cells": len(cells),
        "survives_n": len(survives),
        "best": best,
        "top15": top15,
        "gold_stack_cell": _slim(gold_cell) if gold_cell else None,
        "wired_baseline": _slim(wired) if wired else None,
        "wired_rr_sweep": [_slim(c) for c in wired_cells],
        "pnl_delta_best_pa_minus_wired": delta,
        "best_per_entry_tf": {
            etf: _slim(next(c for c in ranked if c.get("entry_tf") == etf))
            if any(c.get("entry_tf") == etf for c in ranked)
            else None
            for etf in TFS
        },
        "all_cells_summary": all_summary,
        "survives_full": [_slim(c) for c in survives[:40]],
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _log(f"\nWrote {OUT} ({payload['elapsed_s']}s) · SURVIVES={len(survives)}/{len(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
