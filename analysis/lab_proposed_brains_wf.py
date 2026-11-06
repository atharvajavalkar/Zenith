#!/usr/bin/env python3
"""Walk-forward windows for proposed XM-cost v4 absolute winners.

Windows: 7d · 15d · 30d · 90d · 180d (+ 1y / 3y reference)
Uses Dukascopy + XM session spreads. Research only.
"""
from __future__ import annotations

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

from analysis.lab_adaptive_minute_day_v3_bt import _calendar  # noqa: E402
from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    BOOK,
    TF,
    _classify_regime,
    _load,
    _sides_bimod,
    _sides_kalman,
    _sides_perm,
    _sides_snr,
    _static_sides_for_winner,
    _window_trades,
)
from analysis.lab_adaptive_regime_v2_bt import _family  # noqa: E402
from analysis.lab_adaptive_xm_cost_v4_bt import (  # noqa: E402
    RISK_USD,
    _catalog,
    _entries_xm,
    _sl_atr_for,
    _summarize,
    _trades_xm,
    _verdict,
)
from analysis.xm_session_costs import SPREAD_RISK_MULT  # noqa: E402
from config import symbol_rr  # noqa: E402

V4 = Path(ROOT) / "data" / "lab_adaptive_xm_cost_v4.json"
HUNT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"
OUT = Path(ROOT) / "data" / "lab_proposed_brains_wf.json"
LOG = Path(ROOT) / "data" / "lab_proposed_brains_wf.log"
PROGRESS = Path(ROOT) / "data" / "lab_proposed_brains_wf.progress.json"

WINDOWS = (
    ("7d", 7),
    ("15d", 15),
    ("30d", 30),
    ("90d", 90),
    ("180d", 180),
    ("1y", 365),
    ("3y", 365 * 3),
)
CAP_YEARS = 3.0


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    t0 = time.time()
    v4 = json.loads(V4.read_text(encoding="utf-8"))
    per_v4 = v4.get("per_pair") or {}
    hunt = json.loads(HUNT.read_text(encoding="utf-8")) if HUNT.exists() else {}
    per_hunt = hunt.get("per_pair") or {}
    formulas = {fid: fn for fid, _lab, fn in _catalog()}

    proposed = {
        s: (per_v4.get(s) or {}).get("absolute_winner")
        for s in BOOK
        if (per_v4.get(s) or {}).get("absolute_winner")
    }
    _log(
        f"=== Proposed brains WF · ${RISK_USD}/R · XM spreads · TF={TF} ===\n"
        f"map={proposed}\nwindows={[w for w,_ in WINDOWS]}\n"
    )

    desk = {w: 0.0 for w, _ in WINDOWS}
    per_pair: dict[str, Any] = {}

    for pi, sym in enumerate(BOOK):
        mode = proposed.get(sym)
        if not mode:
            continue
        wireable = bool((per_v4.get(sym) or {}).get("wireable"))
        _progress(
            {
                "status": "running",
                "pct": round(100 * pi / len(BOOK), 1),
                "pair": sym,
                "mode": mode,
            }
        )
        _log(f"\n## {sym} · {mode} · wireable={wireable}")
        try:
            df = _load(sym, TF, CAP_YEARS)
        except Exception as exc:
            per_pair[sym] = {"error": str(exc)}
            _log(f"  FAIL {exc}")
            continue

        w = (per_hunt.get(sym) or {}).get("winner") or {}
        wid = str(w.get("id") or "")
        fam = _family(wid)
        rr = float(symbol_rr(sym) or w.get("rr") or 2.4)
        sl = _sl_atr_for(sym, float(w.get("sl_atr") or 1.2))
        static, label, _ = _static_sides_for_winner(df, wid)
        regime = _classify_regime(df)
        cal = _calendar(df)
        ctx = {
            "sym": sym,
            "static": static,
            "regime": regime,
            "cal": cal,
            "v2_preferred_sides": static,
            "family": fam,
        }
        if mode not in formulas and mode != "static":
            # v2 modes
            from analysis.lab_adaptive_regime_v2_bt import _gated_sides, _switch_sides

            alt_trend = (
                _sides_bimod(df, 0.55)
                if sym in ("ETHUSD", "BTCUSD")
                else _sides_perm(df, 0.80)
                if sym == "GOLD"
                else _sides_snr(df, 0.40)
            )
            alt_fade = _sides_kalman(df, 2.2)
            if mode == "gated":
                sides = _gated_sides(static, regime, fam)
            elif mode == "switch":
                sides = _switch_sides(static, regime, fam, alt_trend, alt_fade)
            elif mode in ("v2_preferred", "static"):
                sides = static
            else:
                _log(f"  unknown mode {mode}")
                continue
        else:
            if mode == "static":
                sides = static
            else:
                sides = formulas[mode](df, ctx)

        entries = _entries_xm(df, sides, sym, sl_atr=sl)
        trades = _trades_xm(df, entries, rr, sym)
        wins = {}
        for wname, days in WINDOWS:
            tw = _window_trades(df, trades, days)
            end = df.index[-1]
            n_bars = int(
                np.sum(np.asarray(df.index >= end - pd.Timedelta(days=days)))
            )
            sm = _summarize(tw, n_bars)
            sm["verdict"] = _verdict(sm)
            wins[wname] = sm
            desk[wname] = round(desk[wname] + float(sm["pnl"]), 2)
            _log(
                f"  {wname:>5} ${sm['pnl']:>9,.0f} E[R]={sm['exp_r']:+.3f} "
                f"oos={sm['oos_exp_r']:+.3f} n={sm['n']} wf={sm['wf_green']}/4 "
                f"spR={sm['avg_spread_r']:.3f} {sm['verdict']}"
            )
        per_pair[sym] = {
            "symbol": sym,
            "brain": mode,
            "wireable": wireable,
            "rr": rr,
            "sl_atr": sl,
            "spread_mult": SPREAD_RISK_MULT.get(sym),
            "n_entries": len(entries),
            "n_trades_3y": len(trades),
            "windows": wins,
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "risk_usd": RISK_USD,
        "tf": TF,
        "proposed": proposed,
        "windows": [w for w, _ in WINDOWS],
        "elapsed_sec": round(time.time() - t0, 1),
        "desk": desk,
        "per_pair": per_pair,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress({"status": "done", "pct": 100, "desk": desk, "out": str(OUT)})
    _log("\n=== DESK ===")
    _log(" · ".join(f"{w} ${desk[w]:,.0f}" for w, _ in WINDOWS))
    _log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
