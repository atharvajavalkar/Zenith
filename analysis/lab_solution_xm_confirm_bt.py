#!/usr/bin/env python3
"""Lock the pick on pre-holdout data, then score the holdout once.

The main hunt's sort included holdout PnL, so its 'best' cell is the one
that looked best on the period it was supposed to be tested on. This pass
ranks the coarse grid on data ending 2026-07-24 only, then reports whether
that locked cell is green afterward.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.lab_solution_xm_hunt_bt import (  # noqa: E402
    BOOK,
    COARSE_GAP,
    COARSE_RR,
    COARSE_SL,
    HOLDOUT_START,
    JOIN,
    TFS,
    _ctx,
    _eval_cell,
    _load,
    _loc,
    _log,
    _omni_sides,
    build_mega_catalog,
)
from analysis.lab_adaptive_xm_deep_hunt_bt import TF_YEARS  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_solution_xm_confirm.json"
# One timeframe per symbol: the live scan, not the holdout-max crown.
LOCK = {
    "BTCUSD": "5m",
    "UKOIL": "5m",
    "GOLD": "5m",
    "NAS100": "5m",
}


def _pre_key(row: dict) -> tuple:
    greens = sum(float(row[k] or 0) > 0 for k in ("pnl_30d", "pnl_90d", "pnl_180d", "pnl_1y"))
    passed = row.get("verdict_1y") in ("PASS", "SOFT") and float(row.get("oos_1y") or 0) > 0
    return (
        int(passed),
        greens,
        int(row.get("wf_1y") or 0),
        float(row.get("exp_r_1y") or 0),
        float(row.get("pnl_1y") or 0),
    )


def main() -> int:
    joins = json.loads(JOIN.read_text(encoding="utf-8")) if JOIN.exists() else {}
    report = {"holdout_start": str(HOLDOUT_START), "locked": {}}
    _log("=== LOCKED PRE-HOLDOUT CONFIRM ===")
    for sym in BOOK:
        tf = LOCK[sym]
        _log(f"\n######## {sym} @ {tf} ########")
        df = _load(sym, tf, float(TF_YEARS.get(tf, 5.0)))
        jmeta = ((joins.get(sym) or {}).get(tf) or {})
        cut_i = None
        if float(jmeta.get("gap_hours") or 0) > 72 and jmeta.get("join"):
            cut_i = _loc(df.index, pd.Timestamp(jmeta["join"]))
        ctx = _ctx(df, sym)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        modes = []
        for vid, _lab, fn in build_mega_catalog(lean=False, symbol=sym):
            try:
                modes.append((vid, fn(df, ctx)))
            except Exception:
                continue
        modes.extend(_omni_sides(df, ctx))
        rows = []
        for mode, sides in modes:
            for sl in COARSE_SL:
                for rr in COARSE_RR:
                    for gap in COARSE_GAP:
                        row = _eval_cell(
                            df, high, low, sides, sym, mode, float(sl), float(rr), int(gap), cut_i
                        )
                        if row:
                            rows.append(row)
        rows.sort(key=_pre_key, reverse=True)
        top = []
        for row in rows[:8]:
            top.append(
                {
                    "id": row["id"],
                    "pre": list(_pre_key(row)),
                    "pnl_1y": row["pnl_1y"],
                    "exp_r_1y": row["exp_r_1y"],
                    "oos_1y": row["oos_1y"],
                    "wf_1y": row["wf_1y"],
                    "verdict_1y": row["verdict_1y"],
                    "holdout_n": row["holdout_n"],
                    "holdout_pnl": row["holdout_pnl"],
                    "holdout_exp_r": row["holdout_exp_r"],
                    "solution": row["solution"],
                }
            )
            _log(
                f"  {row['id']} pre_exp={row['exp_r_1y']} 1y={row['pnl_1y']} "
                f"hold={row['holdout_pnl']} n={row['holdout_n']} sol={row['solution']}"
            )
        locked = top[0] if top else None
        report["locked"][sym] = {"tf": tf, "pick": locked, "top": top}
        if locked and locked["solution"]:
            _log(f"  LOCKED PASS {locked['id']}")
        else:
            _log(f"  LOCKED FAIL {None if not locked else locked['id']}")
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    n = sum(1 for s in report["locked"].values() if (s.get("pick") or {}).get("solution"))
    _log(f"\n=== LOCKED {n}/{len(BOOK)} ===")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
