#!/usr/bin/env python3
"""Validate recommended XM-cost desk (post fine-tune) across windows + WF."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.xm_cost_finetune_v2 import (  # noqa: E402
    _battery,
    _entries_spread_aware,
    _i0_for_days,
    _simulate,
)
from analysis.xm_cost_wired_battery import (  # noqa: E402
    RISK_USD,
    WINDOWS,
    _calendar,
    _load,
    _sides_for_brain,
    _summarize,
    _window_trades,
)
from analysis.xm_session_costs import SPREAD_RISK_MULT  # noqa: E402
from config import symbol_rr, symbol_scan_tfs  # noqa: E402

OUT = Path(ROOT) / "data" / "xm_cost_recommended_desk.json"
LOG = Path(ROOT) / "data" / "xm_cost_recommended_desk.log"

# Fine-tune LIVE+ / BEST survivors (halted pairs excluded)
REC = [
    {"symbol": "ETHUSD", "brain": "session_mix", "risk_mult": 8.0, "off": False},
    {"symbol": "GOLD", "brain": "entropy_tod", "risk_mult": 10.0, "off": True},
    {"symbol": "NAS100", "brain": "volmom", "risk_mult": 8.0, "off": False},
    {"symbol": "USDJPY", "brain": "circadian_kalman_z", "risk_mult": 10.0, "off": False},
    {"symbol": "UKOIL", "brain": "session_mix", "risk_mult": 1.0, "off": True},
]


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def main() -> int:
    t0 = time.time()
    LOG.write_text("", encoding="utf-8")
    _log(f"=== Recommended desk validation · ${RISK_USD}/R · XM spreads ===")
    desk = {w[0]: 0.0 for w in WINDOWS}
    rows = []
    for rec in REC:
        sym = rec["symbol"]
        brain = rec["brain"]
        rr = float(symbol_rr(sym) or 2)
        tf = (symbol_scan_tfs(sym) or ["5m"])[0]
        _log(f"\n{sym} {brain} tf={tf} RR={rr} mult={rec['risk_mult']} off={rec['off']}")
        df = _load(sym, tf)
        cal = _calendar(df)
        sides = _sides_for_brain(sym, brain, df, cal)
        entries = _entries_spread_aware(
            df, sides, sym, risk_mult=float(rec["risk_mult"])
        )
        trades = _simulate(
            df,
            entries,
            rr,
            sym,
            max_spread_r=None,
            off_sit=bool(rec["off"]),
        )
        wins = {
            w: _summarize(_window_trades(df, trades, d)) for w, d in WINDOWS
        }
        i0 = _i0_for_days(df, 365)
        bat = _battery(_window_trades(df, trades, 365), i0, len(df))
        for w, _ in WINDOWS:
            desk[w] = round(desk[w] + float(wins[w]["pnl"]), 2)
        row = {
            **rec,
            "tf": tf,
            "rr": rr,
            "spread_mult_table": SPREAD_RISK_MULT.get(sym),
            "windows": wins,
            "battery_1y": bat,
        }
        rows.append(row)
        _log(
            f"  1y ${wins['1y']['pnl']:,.0f} E[R]={wins['1y']['exp_r']:+.3f} "
            f"n={wins['1y']['n']} spR={wins['1y']['avg_spread_r']:.3f} "
            f"{bat['verdict']} wf={bat['wf_green']}/4 "
            f"oosE[R]={bat['oos'].get('exp_r')}"
        )
        for w, _ in WINDOWS:
            if w == "1y":
                continue
            _log(
                f"  {w:>4} ${wins[w]['pnl']:>10,.0f} E[R]={wins[w]['exp_r']:+.3f} "
                f"n={wins[w]['n']}"
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "risk_usd": RISK_USD,
        "halted": ["BTCUSD", "AUDUSD", "USDCHF"],
        "elapsed_sec": round(time.time() - t0, 1),
        "desk": desk,
        "pairs": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _log("\n=== DESK recommended ===")
    _log(" · ".join(f"{w} ${desk[w]:,.0f}" for w, _ in WINDOWS))
    _log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
