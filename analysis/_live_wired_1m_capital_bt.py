#!/usr/bin/env python3
"""1-month native-TF replay — capital / lot-size variations.

Compares flat $/R sizing (proxy for lot scale):
  $10k book → $200/R
  $20k book → $400/R  (2× lots)

Uses Dukascopy local hist + CURRENT live brains / scan TFs / RR.
READ-ONLY.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

import analysis._duka_brains_horizons_bt as duka  # noqa: E402
from analysis._edge_survive_bt import BOOK  # noqa: E402
from config import symbol_entry_brain, symbol_rr, symbol_scan_tfs  # noqa: E402

OUT = Path(ROOT) / "data" / "live_wired_1m_capital.json"
LOG = Path(ROOT) / "data" / "live_wired_1m_capital.log"
DAYS = int(os.getenv("DUKA_DAYS", "30"))
VARIANTS = (
    {"label": "10k_x1", "capital": 10_000.0, "risk_usd": 200.0},
    {"label": "20k_x2", "capital": 20_000.0, "risk_usd": 400.0},
)


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _run_variant(
    *,
    capital: float,
    risk_usd: float,
    label: str,
    end: datetime,
    brains: dict[str, str],
    scan_tfs: dict[str, str],
) -> dict[str, Any]:
    # Point pnl math at this risk ($/R); 2× risk ≈ 2× lots at same SL distance
    duka.RISK_USD = float(risk_usd)
    start_ts = pd.Timestamp(end - timedelta(days=DAYS))
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    rows: list[dict] = []
    _log(f"\n### {label} · capital=${capital:,.0f} · ${risk_usd:g}/R · {DAYS}d")
    for sym in BOOK:
        brain = brains[sym]
        rr = float(symbol_rr(sym, default=2.0))
        tf = scan_tfs.get(sym) or "5m"
        try:
            df_all = duka._load_tf(sym, tf)
            if df_all is None or len(df_all) < 80:
                raise RuntimeError("empty")
            df = df_all[df_all.index >= start_ts]
            if len(df) < 80:
                raise RuntimeError(f"only {len(df)} bars")
            entries = duka._collector(brain)(df)
            bat = duka._battery(df, entries, rr)
            rows.append(
                {
                    "symbol": sym,
                    "brain": brain,
                    "rr": rr,
                    "tf": tf,
                    "capital": capital,
                    "risk_usd": risk_usd,
                    **bat,
                }
            )
            f, o = bat["full"], bat["oos"]
            _log(
                f"  {sym:8} {brain:18} {tf:4} n={f['n']:4} "
                f"PnL=${f['pnl']:9,.0f} OOS=${o['pnl']:8,.0f} "
                f"E[R]={o['exp_r']:+.3f} WR={f.get('wr', 0):.1f}% {bat['verdict']}"
            )
        except Exception as exc:
            _log(f"  {sym:8} {brain:18} {tf:4} FAIL: {exc}")
            rows.append(
                {
                    "symbol": sym,
                    "brain": brain,
                    "tf": tf,
                    "capital": capital,
                    "risk_usd": risk_usd,
                    "error": str(exc),
                    "verdict": "DEAD",
                }
            )

    desk = duka._desk_fit(rows)
    _log(
        f"  → Desk {desk['survives']}S/{desk['fragile']}F/{desk['dead']}D · "
        f"Full ${desk['full_pnl']:,.0f} · OOS ${desk['oos_pnl']:,.0f} · "
        f"fit {desk['fit_score_1_to_10']}/10 · "
        f"{'FITS' if desk['fits'] else 'DOES NOT FIT'}"
    )
    return {
        "label": label,
        "capital": capital,
        "risk_usd": risk_usd,
        "days": DAYS,
        "start": start_ts.date().isoformat(),
        "end": end.date().isoformat(),
        "results": rows,
        "desk": desk,
    }


def main() -> int:
    t0 = time.time()
    if LOG.exists():
        LOG.write_text("")
    end = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=1)
    brains = {s: symbol_entry_brain(s) for s in BOOK}
    scan_tfs = {s: (symbol_scan_tfs(s)[0] if symbol_scan_tfs(s) else "5m") for s in BOOK}

    _log(
        f"LIVE-WIRED 1m capital variations · Dukascopy native TFs\n"
        f"Window: last {DAYS}d ending {end.date()}\n"
        f"Brains: {brains}\n"
        f"Scan TFs: {scan_tfs}\n"
        f"Variants: $10k→$200/R vs $20k→$400/R (2× lots)\n"
    )

    variants = [
        _run_variant(
            capital=v["capital"],
            risk_usd=v["risk_usd"],
            label=v["label"],
            end=end,
            brains=brains,
            scan_tfs=scan_tfs,
        )
        for v in VARIANTS
    ]

    # Side-by-side per pair
    by_sym: dict[str, dict[str, Any]] = {}
    for v in variants:
        for r in v["results"]:
            sym = r["symbol"]
            by_sym.setdefault(sym, {"symbol": sym, "brain": r.get("brain"), "tf": r.get("tf")})
            f = r.get("full") or {}
            by_sym[sym][v["label"]] = {
                "n": f.get("n"),
                "pnl": f.get("pnl"),
                "wr": f.get("wr"),
                "exp_r": f.get("exp_r"),
                "verdict": r.get("verdict"),
            }

    _log("\n" + "=" * 88)
    _log(f"{'Pair':8} {'Brain':12} {'TF':4} {'n':>4} {'$10k PnL':>10} {'$20k PnL':>10} {'Δ (2×)':>10}")
    _log("-" * 72)
    for sym in BOOK:
        row = by_sym.get(sym) or {}
        a = row.get("10k_x1") or {}
        b = row.get("20k_x2") or {}
        p10 = float(a.get("pnl") or 0)
        p20 = float(b.get("pnl") or 0)
        _log(
            f"{sym:8} {(row.get('brain') or '—'):12} {(row.get('tf') or '—'):4} "
            f"{int(a.get('n') or 0):4} ${p10:9,.0f} ${p20:9,.0f} ${p20 - p10:9,.0f}"
        )

    d10 = variants[0]["desk"]
    d20 = variants[1]["desk"]
    _log(
        f"\nDesk Full: $10k ${d10['full_pnl']:,.0f} → $20k ${d20['full_pnl']:,.0f} "
        f"(Δ ${d20['full_pnl'] - d10['full_pnl']:,.0f})"
    )
    _log(
        f"Desk OOS:  $10k ${d10['oos_pnl']:,.0f} → $20k ${d20['oos_pnl']:,.0f} "
        f"(Δ ${d20['oos_pnl'] - d10['oos_pnl']:,.0f})"
    )
    # Equity impact vs starting capital
    ret10 = 100.0 * d10["full_pnl"] / 10_000.0
    ret20 = 100.0 * d20["full_pnl"] / 20_000.0
    _log(f"Return on start capital: $10k {ret10:+.1f}% · $20k {ret20:+.1f}%")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live_wired_1m_capital_variations",
        "source": "dukascopy_local_hist",
        "days": DAYS,
        "brains": brains,
        "scan_tfs": scan_tfs,
        "variants": variants,
        "by_symbol": by_sym,
        "note": (
            "Flat $/R sizing as lot proxy: $200/R @ $10k vs $400/R @ $20k. "
            "Same entries/exits; PnL scales ~2×. Not live dynamic-lot path."
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _log(f"\nWrote {OUT} in {payload['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
