#!/usr/bin/env python3
"""RESEARCH ONLY — ETHUSD / BTCUSD: PA liq-engulf vs wired BB brains.

Same process as GOLD PA (no live wiring):
  15m liq sweep → 1h+25m EMA bias → 5m engulf

Baseline (live map, not changed):
  ETHUSD: bb_1.5 @ 1m, RR=4
  BTCUSD: bb_2.5 @ 1m, RR=2.4

Battery: 1_full · 2_oos70 · 3–6_wf · 7_spread · 8_skip + MC
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

from analysis._edge_survive_bt import RISK_USD, _prep  # noqa: E402
from analysis.lab_agentic_1y_battery import _battery_from_trades  # noqa: E402
from analysis.lab_gold_pa_liq_engulf_bt import (  # noqa: E402
    RRS,
    WINDOW_DAYS,
    _collect_pa_entries,
    _pack_cell,
    _resample_ohlc,
)
from analysis.strategy_lab_duka_deep_hunt import _simulate_fast  # noqa: E402
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_crypto_pa_liq_engulf_bt.json"
LOG = Path(ROOT) / "data" / "lab_crypto_pa_liq_engulf_bt.log"

# Live baselines (read-only compare)
BASELINES = {
    "ETHUSD": {"brain": "bb_1.5", "tf": "1m", "rr": 4.0, "bb_mult": 1.5},
    "BTCUSD": {"brain": "bb_2.5", "tf": "1m", "rr": 2.4, "bb_mult": 2.5},
}
SYMBOLS = ("ETHUSD", "BTCUSD")


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _load_sym_1y(sym: str) -> dict[str, Any]:
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    clear_hist_cache()
    end = pd_end = None
    import pandas as pd

    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    frames: dict[str, Any] = {}
    for tf in ("1m", "5m", "15m"):
        if not has_local_hist(sym, tf) and not (
            tf != "1m" and has_local_hist(sym, "1m")
        ):
            _log(f"  WARN no hist {sym} {tf}")
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
        data_start = data_end - pd.Timedelta(days=WINDOW_DAYS)
        df = df[df.index >= data_start]
        if len(df) < 400:
            continue
        frames[tf] = df
        _log(
            f"  load {sym} {tf}: {len(df):,} bars "
            f"{df.index[0].date()}→{df.index[-1].date()}"
        )

    if "5m" not in frames:
        raise RuntimeError(f"{sym}: need 5m hist for PA brain")
    df5 = frames["5m"]
    if "15m" not in frames:
        frames["15m"] = _resample_ohlc(df5, "15min")
        _log(f"  resample {sym} 15m: {len(frames['15m']):,}")
    frames["1h"] = _resample_ohlc(df5, "1h")
    frames["25m"] = _resample_ohlc(df5, "25min")
    _log(f"  resample {sym} 1h={len(frames['1h']):,} 25m={len(frames['25m']):,}")
    return frames


def _run_symbol(sym: str) -> dict[str, Any]:
    import pandas as pd  # noqa: F401 — used by load

    base = BASELINES[sym]
    _log(f"\n===== {sym} =====")
    frames = _load_sym_1y(sym)
    df5, df15 = frames["5m"], frames["15m"]
    df1h, df25 = frames["1h"], frames["25m"]
    df1m = frames.get("1m")

    _log(f"  Collecting PA entries…")
    pa_entries = _collect_pa_entries(df5, df15, df1h, df25)
    _log(f"  PA raw entries: {len(pa_entries)}")

    bb_entries: list = []
    bb_df = None
    if df1m is not None and len(df1m) >= 400:
        _log(f"  Collecting wired {base['brain']} on 1m…")
        bb_entries = _bb_std(df1m, float(base["bb_mult"]))
        bb_df = df1m
        _log(f"  BB raw entries: {len(bb_entries)}")
    else:
        _log("  WARN no 1m — BB baseline skipped")

    cells: list[dict[str, Any]] = []
    for rr in RRS:
        cell = _pack_cell(
            "pa_liq_engulf",
            df5,
            pa_entries,
            rr,
            note="15m sweep + 1h/25m bias + 5m engulf",
        )
        cells.append(cell)
        _log(
            f"  PA RR={rr}: n={cell['trades']} pnl=${cell['pnl']:,.0f} "
            f"E[R]={cell['exp_r']} {cell['verdict']} "
            f"WF={cell['wf_green']}/{cell['wf_n']} MC={cell['mc_prob_profit']}%"
        )

    if bb_df is not None:
        # wired RR + full RR sweep for fair compare
        rr_set = sorted(set(RRS) | {float(base["rr"])})
        for rr in rr_set:
            cell = _pack_cell(
                base["brain"],
                bb_df,
                bb_entries,
                rr,
                note=f"live wired baseline exec {base['tf']}",
            )
            cells.append(cell)
            tag = "WIRED" if abs(rr - float(base["rr"])) < 1e-9 else ""
            _log(
                f"  {base['brain']} RR={rr} {tag}: n={cell['trades']} "
                f"pnl=${cell['pnl']:,.0f} E[R]={cell['exp_r']} {cell['verdict']} "
                f"WF={cell['wf_green']}/{cell['wf_n']} MC={cell['mc_prob_profit']}%"
            )

    pa4 = next(c for c in cells if c["name"] == "pa_liq_engulf" and c["rr"] == 4.0)
    wired_rr = float(base["rr"])
    wired = next(
        (
            c
            for c in cells
            if c["name"] == base["brain"] and abs(c["rr"] - wired_rr) < 1e-9
        ),
        None,
    )
    # also PA at wired RR for BTC (2.4)
    pa_at_wired = next(
        (c for c in cells if c["name"] == "pa_liq_engulf" and abs(c["rr"] - wired_rr) < 1e-9),
        pa4,
    )
    best_pa = max(
        (c for c in cells if c["name"] == "pa_liq_engulf"),
        key=lambda c: (c["verdict"] == "SURVIVES", c["pnl"]),
    )

    h2h = {
        "pa_rr4": {
            "pnl": pa4["pnl"],
            "n": pa4["trades"],
            "exp_r": pa4["exp_r"],
            "verdict": pa4["verdict"],
            "tests": pa4["tests"],
            "mc_prob_profit": pa4["mc_prob_profit"],
            "wf_green": pa4["wf_green"],
        },
        "pa_at_wired_rr": {
            "rr": wired_rr,
            "pnl": pa_at_wired["pnl"],
            "n": pa_at_wired["trades"],
            "exp_r": pa_at_wired["exp_r"],
            "verdict": pa_at_wired["verdict"],
            "tests": pa_at_wired["tests"],
            "mc_prob_profit": pa_at_wired["mc_prob_profit"],
        },
        "wired": None
        if wired is None
        else {
            "brain": base["brain"],
            "tf": base["tf"],
            "rr": wired_rr,
            "pnl": wired["pnl"],
            "n": wired["trades"],
            "exp_r": wired["exp_r"],
            "verdict": wired["verdict"],
            "tests": wired["tests"],
            "mc_prob_profit": wired["mc_prob_profit"],
            "wf_green": wired["wf_green"],
        },
        "pnl_delta_pa4_minus_wired": None
        if wired is None
        else round(float(pa4["pnl"]) - float(wired["pnl"]), 2),
        "pnl_delta_pa_wiredrr_minus_wired": None
        if wired is None
        else round(float(pa_at_wired["pnl"]) - float(wired["pnl"]), 2),
    }

    _log(f"\n### {sym} HEAD-TO-HEAD")
    _log(
        f"PA @4: n={pa4['trades']} PnL=${pa4['pnl']:,.0f} "
        f"E[R]={pa4['exp_r']} {pa4['verdict']}"
    )
    if wired:
        _log(
            f"Wired {base['brain']}@{base['tf']} RR={wired_rr}: "
            f"n={wired['trades']} PnL=${wired['pnl']:,.0f} "
            f"E[R]={wired['exp_r']} {wired['verdict']}"
        )
        _log(f"Δ PA@4 − wired: ${h2h['pnl_delta_pa4_minus_wired']:,.0f}")
        _log(
            f"Δ PA@wiredRR − wired: ${h2h['pnl_delta_pa_wiredrr_minus_wired']:,.0f}"
        )

    return {
        "symbol": sym,
        "baseline": base,
        "head_to_head": h2h,
        "best_pa": best_pa,
        "cells": cells,
        "why_eth_note": (
            "ETH bb_1.5@1m prints huge paper PnL but ~23% WR / 13k trades — "
            "live feels like a loser from streak bleed, crypto trend regimes, "
            "and unmodeled XM spread/slip on 1m mean-reversion."
            if sym == "ETHUSD"
            else None
        ),
    }


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass
    _log("=== CRYPTO PA liq-engulf vs wired BB (Dukascopy 1y, RESEARCH) ===")
    _log("Live SYMBOL_ENTRY_BRAIN NOT changed.")

    results = []
    for sym in SYMBOLS:
        try:
            results.append(_run_symbol(sym))
        except Exception as exc:
            _log(f"FAIL {sym}: {exc}")
            results.append({"symbol": sym, "error": str(exc)})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "live_wired": False,
        "window_days": WINDOW_DAYS,
        "risk_usd": RISK_USD,
        "process": {
            "liquidity_tf": "15m",
            "bias_tfs": ["1h", "25m"],
            "entry_tf": "5m",
            "entry_rule": "engulf after 15m sweep reclaim, within 90m",
        },
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _log(f"\nWrote {OUT} ({payload['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
