#!/usr/bin/env python3
"""Dukascopy 1y: current wired brains ± live peak-DD equity protect.

Answers: is PEAK_DD_PROTECT better than running the current brains naked?
Desk-wide gate (same evaluate_equity_gates as live). LIVE MAP NOT MODIFIED.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import BOOK, RISK_USD  # noqa: E402
from analysis.equity_protect import evaluate_equity_gates  # noqa: E402
from analysis.lab_agentic_1y_battery import (  # noqa: E402
    _simulate_detailed,
    _stats,
    _wired_map,
)
from analysis.strategy_lab_duka_deep_hunt import _catalog  # noqa: E402
from analysis.strategy_lab_duka_scalp_1y_hunt import WINDOW_DAYS, _load_frames_1y  # noqa: E402
from config import (  # noqa: E402
    BANK_TARGET_PCT,
    HARVEST_PCT,
    PEAK_DD_PROTECT_PCT,
    START_CAPITAL_USD,
)

OUT = Path(ROOT) / "data" / "lab_peak_dd_1y_compare.json"


def _merge_chrono(per_sym: dict[str, list[dict]]) -> list[dict]:
    all_tr: list[dict] = []
    for sym, tr in per_sym.items():
        for t in tr:
            row = dict(t)
            row["symbol"] = sym
            all_tr.append(row)
    all_tr.sort(key=lambda t: (str(t.get("exit_t") or ""), str(t.get("entry_t") or ""), t.get("i", 0)))
    return all_tr


def filter_peak_dd_desk(
    trades: list[dict],
    *,
    start_capital: float,
    peak_dd_pct: float,
    harvest_pct: float,
    bank_target_pct: float,
) -> tuple[list[dict], dict[str, Any]]:
    """Apply live equity gates chronologically across the whole desk."""
    out: list[dict] = []
    realized = 0.0
    peak = 0.0
    blocked = 0
    first_block_at = None
    modes: dict[str, int] = {}

    for t in trades:
        gate = evaluate_equity_gates(
            start_capital=start_capital,
            realized_pnl=realized,
            peak_pnl=peak,
            day_pnl=0.0,
            trades_today=0,
            peak_dd_pct=peak_dd_pct,
            daily_loss_pct=0.0,  # match live test desk (daily off)
            max_trades_per_day=10_000,
            harvest_pct=harvest_pct,
            bank_target_pct=bank_target_pct,
        )
        modes[gate["mode"]] = modes.get(gate["mode"], 0) + 1
        if not gate["allow"]:
            blocked += 1
            if first_block_at is None:
                first_block_at = {
                    "at_trade_exit": t.get("exit_t"),
                    "symbol": t.get("symbol"),
                    "reason": gate["reason"],
                    "realized_pnl": realized,
                    "peak_pnl": peak,
                    "drawdown": gate["drawdown"],
                }
            continue
        out.append(t)
        realized += float(t["pnl"])
        peak = max(peak, realized)

    return out, {
        "taken": len(out),
        "blocked": blocked,
        "raw": len(trades),
        "final_realized": round(realized, 2),
        "final_peak": round(peak, 2),
        "first_block": first_block_at,
        "gate_mode_counts_on_candidates": modes,
    }


def main() -> int:
    end = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=1), tz="UTC")
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    catalog = {sid: (fam, lab, fn) for sid, fam, lab, fn in _catalog()}
    wired = _wired_map()
    cap = float(START_CAPITAL_USD or 10_000)
    peak_pct = float(PEAK_DD_PROTECT_PCT or 0.15)
    harvest = float(HARVEST_PCT or 0.50)
    bank = float(BANK_TARGET_PCT or 1.00)

    print(
        f"PEAK-DD 1y COMPARE · {start.date()}→{end.date()} · ${RISK_USD}/R · "
        f"cap=${cap:.0f} peak_dd={peak_pct:.0%} bank={bank:.0%} harvest={harvest:.0%}",
        flush=True,
    )
    print("LIVE BRAINS NOT MODIFIED\n", flush=True)

    raw_by_sym: dict[str, list[dict]] = {}
    meta: dict[str, Any] = {}

    for sym in BOOK:
        spec = wired.get(sym)
        if not spec:
            continue
        sid, tf, rr = spec["strategy"], spec["tf"], spec["rr"]
        if sid not in catalog:
            print(f"skip {sym}: unknown {sid}", flush=True)
            continue
        frames = _load_frames_1y(sym, start, end)
        df = frames.get(tf)
        if df is None or len(df) < 400:
            print(f"skip {sym}: no {tf}", flush=True)
            continue
        _, label, fn = catalog[sid]
        entries = fn(df)
        raw = _simulate_detailed(df, entries, rr)
        raw_by_sym[sym] = raw
        st = _stats(raw)
        meta[sym] = {
            "strategy": sid,
            "label": label,
            "tf": tf,
            "rr": rr,
            "raw": st,
        }
        print(f"  {sym} {sid}@{tf} RR={rr}: n={st['n']} pnl=${st['pnl']:,.0f}", flush=True)

    chrono = _merge_chrono(raw_by_sym)
    base_stats = _stats(chrono)

    protected, prot_meta = filter_peak_dd_desk(
        chrono,
        start_capital=cap,
        peak_dd_pct=peak_pct,
        harvest_pct=harvest,
        bank_target_pct=bank,
    )
    prot_stats = _stats(protected)

    # Also: protect disabled (peak_dd_pct=0 → limit 0 but gate requires peak>0 and dd>=limit;
    # with pct=0, peak_dd_limit=0 so any dd>=0 after a peak blocks — bad.
    # True "off" = skip filter entirely (= baseline).
    delta = round(float(prot_stats["pnl"]) - float(base_stats["pnl"]), 2)
    better = "peak_dd_protect" if prot_stats["pnl"] > base_stats["pnl"] else (
        "current_brains_no_protect" if prot_stats["pnl"] < base_stats["pnl"] else "tie"
    )

    # Per-pair under protect: which symbols still got fills after desk gate
    taken_by = {}
    for t in protected:
        s = t["symbol"]
        taken_by[s] = taken_by.get(s, 0) + 1
    raw_by = {s: len(v) for s, v in raw_by_sym.items()}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat(), "days": WINDOW_DAYS},
        "risk_usd": RISK_USD,
        "start_capital": cap,
        "peak_dd_protect_pct": peak_pct,
        "peak_dd_limit_usd": round(cap * peak_pct, 2),
        "bank_target_pct": bank,
        "harvest_pct": harvest,
        "wired": wired,
        "per_pair_raw": meta,
        "current_brains_no_protect": {
            "n": base_stats["n"],
            "pnl": base_stats["pnl"],
            "exp_r": base_stats["exp_r"],
            "win_rate": base_stats["win_rate"],
            "note": "Same as Stage A A0 wired map (no equity protect)",
        },
        "with_peak_dd_protect": {
            "n": prot_stats["n"],
            "pnl": prot_stats["pnl"],
            "exp_r": prot_stats["exp_r"],
            "win_rate": prot_stats["win_rate"],
            **prot_meta,
            "fills_per_symbol": taken_by,
            "raw_per_symbol": raw_by,
        },
        "delta_pnl_protect_minus_baseline": delta,
        "verdict": better,
        "recommendation": (
            "Keep peak-DD OFF for demo validation — protect cuts desk PnL vs current brains."
            if better == "current_brains_no_protect"
            else (
                "Peak-DD protect improves 1y desk PnL vs naked brains — keep it."
                if better == "peak_dd_protect"
                else "No material PnL difference."
            )
        ),
    }

    # Optional: quick GOLD-only — is another brain better than don_100 on same TF?
    # Skip heavy full catalog; just report wired GOLD under both regimes.
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n===== DESK =====", flush=True)
    print(
        f"Current brains (no protect): n={base_stats['n']} pnl=${base_stats['pnl']:,.0f} "
        f"E[R]={base_stats['exp_r']:+.3f}",
        flush=True,
    )
    print(
        f"With peak-DD {peak_pct:.0%} (${cap*peak_pct:.0f}): n={prot_stats['n']} "
        f"pnl=${prot_stats['pnl']:,.0f} E[R]={prot_stats['exp_r']:+.3f} "
        f"blocked={prot_meta['blocked']}",
        flush=True,
    )
    if prot_meta.get("first_block"):
        print(f"First block: {prot_meta['first_block']}", flush=True)
    print(f"Delta (protect - baseline): ${delta:,.0f}", flush=True)
    print(f"VERDICT: {better}", flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
