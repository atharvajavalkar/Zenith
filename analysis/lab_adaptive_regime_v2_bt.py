#!/usr/bin/env python3
"""RESEARCH ONLY — Adaptive v2: regime-gated static + smart switch.

v1 lesson: naive TREND/FADE routing overtrades and loses vs static on 1y/3y.
v2:
  · Start from each pair's invented static winner (RR/SL preserved)
  · FLAT → sit out
  · Regime MATCHES formula family → take static signal
  · Regime CONFLICTS (e.g. fade-regime but trend-formula) → switch to
    the other book OR sit out (configurable)
  · Compare: static | gated (sit-out only) | switch (gated+switch)

Does NOT change live brains.
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

from analysis._edge_survive_bt import RISK_USD, _prep, _stats  # noqa: E402
from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    TF,
    WINDOWS,
    _adaptive_sides,
    _classify_regime,
    _equity_curve,
    _load,
    _sides_bimod,
    _sides_kalman,
    _sides_perm,
    _sides_snr,
    _static_sides_for_winner,
    _summarize,
    _window_trades,
)
from analysis.lab_zenith_math_invention_deep_bt import (  # noqa: E402
    BOOK,
    _atr_sl_entries,
    _trades_with_i,
)

HUNT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"
OUT = Path(ROOT) / "data" / "lab_adaptive_regime_v2.json"
LOG = Path(ROOT) / "data" / "lab_adaptive_regime_v2.log"
CHARTS = Path(ROOT) / "data" / "lab_adaptive_regime_v2_charts.json"
PROGRESS = Path(ROOT) / "data" / "lab_adaptive_regime_v2.progress.json"
CAP_YEARS = 3.0


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _family(wid: str) -> str:
    w = str(wid or "")
    if "kalman" in w:
        return "fade"
    return "trend"


def _gated_sides(static: np.ndarray, regime: np.ndarray, family: str) -> np.ndarray:
    """Trade static only when regime matches family; else flat."""
    out = np.zeros(len(static), dtype=int)
    if family == "fade":
        m = regime == -1
    else:
        m = regime == 1
    out[m] = static[m]
    return out


def _switch_sides(
    static: np.ndarray,
    regime: np.ndarray,
    family: str,
    alt_trend: np.ndarray,
    alt_fade: np.ndarray,
) -> np.ndarray:
    """
    Match → static; conflict → alternate book; flat → 0.
    """
    out = np.zeros(len(static), dtype=int)
    if family == "fade":
        out[regime == -1] = static[regime == -1]
        out[regime == 1] = alt_trend[regime == 1]
    else:
        out[regime == 1] = static[regime == 1]
        out[regime == -1] = alt_fade[regime == -1]
    return out


def main() -> int:
    LOG.write_text("")
    t0 = time.time()
    hunt = json.loads(HUNT.read_text(encoding="utf-8")) if HUNT.exists() else {}
    per_hunt = hunt.get("per_pair") or {}

    _log(
        "=== ADAPTIVE v2 · GATED STATIC + SWITCH ===\n"
        f"TF={TF} {CAP_YEARS}y\n"
        "Modes: static | gated(sit-out) | switch(gated+alt)\n"
        "LIVE BRAINS NOT TOUCHED\n"
    )

    modes = ("static", "gated", "switch")
    desk = {m: {w[0]: 0.0 for w in WINDOWS} for m in modes}
    per_pair: dict[str, Any] = {}
    equity: dict[str, Any] = {}

    for pi, sym in enumerate(BOOK):
        _progress(
            {
                "status": "running",
                "pct": int(5 + 90 * pi / len(BOOK)),
                "message": f"v2 [{sym}]",
                "symbol": sym,
                "elapsed_sec": round(time.time() - t0, 1),
            }
        )
        _log(f"\n######## {sym} ########")
        df = _load(sym, TF, CAP_YEARS)
        w = (per_hunt.get(sym) or {}).get("winner") or {}
        wid = str(w.get("id") or "")
        fam = _family(wid)
        rr = float(w.get("rr") or 2.4)
        sl = float(w.get("sl_atr") or 1.2)

        static, label, _ = _static_sides_for_winner(df, wid)
        regime = _classify_regime(df)
        alt_trend = (
            _sides_bimod(df, 0.55)
            if sym in ("ETHUSD", "BTCUSD")
            else _sides_perm(df, 0.80)
            if sym == "GOLD"
            else _sides_snr(df, 0.40)
        )
        alt_fade = _sides_kalman(df, 2.2)

        sides = {
            "static": static,
            "gated": _gated_sides(static, regime, fam),
            "switch": _switch_sides(static, regime, fam, alt_trend, alt_fade),
        }

        trades = {}
        for mode, s in sides.items():
            ent = _atr_sl_entries(df, s, sl_atr=sl)
            trades[mode] = _trades_with_i(df, ent, rr)

        win_out = {}
        for wname, days in WINDOWS:
            end = df.index[-1]
            start = end - pd.Timedelta(days=days)
            n_bars = int(np.sum(np.asarray(df.index >= start)))
            block = {}
            for mode in modes:
                tw = _window_trades(df, trades[mode], days)
                sm = _summarize(tw, n_bars)
                block[mode] = sm
                desk[mode][wname] = round(desk[mode][wname] + float(sm["pnl"]), 2)
            win_out[wname] = block
            _log(
                f"  {wname:4} "
                + " | ".join(
                    f"{m} n={block[m]['n']:4} WR={block[m]['win_rate']:5} "
                    f"E[R]={block[m]['exp_r']:+.3f} ${block[m]['pnl']:>8,.0f}"
                    for m in modes
                )
            )

        equity[sym] = {
            m: _equity_curve(trades[m], df) for m in modes
        }
        per_pair[sym] = {
            "symbol": sym,
            "family": fam,
            "static_id": wid,
            "static_label": label,
            "rr": rr,
            "sl_atr": sl,
            "regime_pct": {
                "trend": round(100.0 * float(np.mean(regime == 1)), 2),
                "fade": round(100.0 * float(np.mean(regime == -1)), 2),
                "flat": round(100.0 * float(np.mean(regime == 0)), 2),
            },
            "windows": win_out,
            "best_mode_1y": max(
                modes, key=lambda m: float(win_out["1y"][m]["pnl"])
            ),
            "best_mode_7d": max(
                modes, key=lambda m: float(win_out["7d"][m]["pnl"])
            ),
        }

    charts = {
        "desk_pnl": [
            {
                "window": w,
                "static": desk["static"][w],
                "gated": desk["gated"][w],
                "switch": desk["switch"][w],
            }
            for w, _ in WINDOWS
        ],
        "pair_1y": [
            {
                "symbol": s,
                "static": per_pair[s]["windows"]["1y"]["static"]["pnl"],
                "gated": per_pair[s]["windows"]["1y"]["gated"]["pnl"],
                "switch": per_pair[s]["windows"]["1y"]["switch"]["pnl"],
                "best": per_pair[s]["best_mode_1y"],
            }
            for s in BOOK
            if s in per_pair
        ],
        "pair_7d": [
            {
                "symbol": s,
                "static": per_pair[s]["windows"]["7d"]["static"]["pnl"],
                "gated": per_pair[s]["windows"]["7d"]["gated"]["pnl"],
                "switch": per_pair[s]["windows"]["7d"]["switch"]["pnl"],
                "best": per_pair[s]["best_mode_7d"],
            }
            for s in BOOK
            if s in per_pair
        ],
        "equity": equity,
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "recommend_live": False,
        "elapsed_sec": round(time.time() - t0, 1),
        "version": "adaptive_v2_gated_switch",
        "tf": TF,
        "risk_usd": RISK_USD,
        "desk": desk,
        "per_pair": per_pair,
        "charts": charts,
        "note": "v2: gated sit-out + optional switch. Live brains unchanged.",
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    CHARTS.write_text(json.dumps(charts, indent=2, default=str), encoding="utf-8")
    _progress({"status": "done", "pct": 100, "elapsed_sec": round(time.time() - t0, 1)})
    _log("\nDESK 1y: " + ", ".join(f"{m}=${desk[m]['1y']:,.0f}" for m in modes))
    _log("DESK 7d: " + ", ".join(f"{m}=${desk[m]['7d']:,.0f}" for m in modes))
    _log(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
