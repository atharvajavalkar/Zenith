#!/usr/bin/env python3
"""RESEARCH ONLY — last-week forward paper of invented math winners.

Uses each pair's winner from lab_zenith_math_invention_deep.json on the
last 7 calendar days of local Dukascopy hist (warmup before window).

Does NOT change live brains.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD, _prep, _stats  # noqa: E402
from analysis.lab_zenith_math_invention_deep_bt import (  # noqa: E402
    BOOK,
    _atr_sl_entries,
    _enrich,
    _trades_with_i,
)
from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv, local_span  # noqa: E402

HUNT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"
OUT = Path(ROOT) / "data" / "lab_math_invention_last_week.json"
LOG = Path(ROOT) / "data" / "lab_math_invention_last_week.log"


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _resolve_formula(wid: str) -> tuple[str, Callable]:
    """Map densified winner id → (family_label, sides_fn)."""
    from analysis.lab_zenith_math_invention_deep_bt import _catalog

    wid = str(wid or "")
    # strip densify suffixes _slX_DY
    base = re.sub(r"(_sl[0-9.]+(_D\d+)?)+$", "", wid)
    base = re.sub(r"_D\d+$", "", base)

    # exact catalog match first
    cat = {fid: (lab, fn) for fid, lab, fn in _catalog()}
    if base in cat:
        lab, fn = cat[base]
        return lab, lambda df, _fn=fn: _fn(df)[0]

    # parameterized rebuild from name
    m = re.match(r"bimod_([0-9.]+)", base)
    if m:
        thr = float(m.group(1))

        def fn(df, thr=thr):
            b = df["bimod"].to_numpy()
            ds = df["drift_sign"].to_numpy()
            return np.where((b >= thr) & (ds != 0), ds.astype(int), 0)

        return f"bimodality≥{thr}", fn

    m = re.match(r"perm_order_([0-9.]+)", base)
    if m:
        pmax = float(m.group(1))

        def fn(df, pmax=pmax):
            pe = df["perm_ent"].to_numpy()
            ds = df["drift_sign"].to_numpy()
            return np.where((pe <= pmax) & (ds != 0), ds.astype(int), 0)

        return f"perm_ent≤{pmax}", fn

    m = re.match(r"snr_drift_([0-9.]+)", base)
    if m:
        thr = float(m.group(1))

        def fn(df, thr=thr):
            snr = df["snr"].to_numpy()
            ds = df["drift_sign"].to_numpy()
            return np.where((snr >= thr) & (ds != 0), ds.astype(int), 0)

        return f"SNR≥{thr}", fn

    m = re.match(r"kalman_fade_([0-9.]+)", base)
    if m:
        z = float(m.group(1))

        def fn(df, z=z):
            kz = df["kalman_z"].to_numpy()
            sides = np.zeros(len(df), dtype=int)
            sides[kz <= -z] = 1
            sides[kz >= z] = -1
            return sides

        return f"kalman_fade|{z}", fn

    # fallback: first catalog prefix match
    for fid, lab, fn in _catalog():
        if base.startswith(fid.split("_")[0]):
            return lab, lambda df, _fn=fn: _fn(df)[0]

    raise RuntimeError(f"cannot resolve formula for {wid}")


def _load_week(sym: str, tf: str, days: int = 7, warm_days: int = 40) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    clear_hist_cache()
    if not has_local_hist(sym, tf):
        raise RuntimeError(f"no hist {sym} {tf}")
    sp = local_span(sym, tf)
    if not sp:
        raise RuntimeError(f"no span {sym} {tf}")
    end = sp[1]
    week_start = end - pd.Timedelta(days=days)
    load_start = week_start - pd.Timedelta(days=warm_days)
    raw = load_local_ohlcv(
        sym,
        tf=tf,
        start=load_start.isoformat(),
        end=end.isoformat(),
    )
    if raw is None or len(raw) < 100:
        raise RuntimeError(f"short hist {sym} {tf}")
    df = _prep(raw[["open", "high", "low", "close"]].dropna())
    df = _enrich(df)
    return df, week_start, end


def main() -> int:
    LOG.write_text("")
    if not HUNT.exists():
        _log(f"missing {HUNT}")
        return 1
    hunt = json.loads(HUNT.read_text(encoding="utf-8"))
    per = hunt.get("per_pair") or {}

    _log(
        "=== MATH INVENTION · LAST WEEK PAPER ===\n"
        f"Risk ${RISK_USD}/R · window=last 7d of local Dukascopy\n"
        "LIVE BRAINS NOT TOUCHED\n"
    )

    rows: dict[str, Any] = {}
    desk_pnl = 0.0
    desk_n = 0
    desk_wins = 0

    for sym in BOOK:
        row = per.get(sym) or {}
        w = row.get("winner") or {}
        if not w:
            rows[sym] = {"error": "no_winner"}
            _log(f"{sym}: no winner")
            continue
        wid = str(w.get("id") or "")
        tf = str(w.get("tf") or "15m")
        rr = float(w.get("rr") or 2.0)
        sl_atr = float(w.get("sl_atr") or 1.2)
        try:
            label, sides_fn = _resolve_formula(wid)
            df, week_start, week_end = _load_week(sym, tf)
            sides = sides_fn(df)
            entries = _atr_sl_entries(df, sides, sl_atr=sl_atr)
            # only entries inside week
            i0 = int(np.searchsorted(df.index.values, np.datetime64(week_start)))
            week_entries = [e for e in entries if int(e["ei"]) >= i0]
            trades = _trades_with_i(df, week_entries, rr)
            # keep trades whose entry bar is in week
            trades = [t for t in trades if int(t["i"]) >= i0]
            st = _stats(trades)
            n = int(st.get("n") or 0)
            pnl = float(st.get("pnl") or 0)
            wr = st.get("win_rate")
            er = st.get("exp_r")
            wins = sum(1 for t in trades if float(t["r"]) > 0)
            losses = sum(1 for t in trades if float(t["r"]) < 0)
            desk_pnl += pnl
            desk_n += n
            desk_wins += wins
            detail = {
                "symbol": sym,
                "winner_id": wid,
                "formula": label,
                "equation": w.get("equation"),
                "tf": tf,
                "rr": rr,
                "sl_atr": sl_atr,
                "window_start": str(week_start)[:19],
                "window_end": str(week_end)[:19],
                "n": n,
                "wins": wins,
                "losses": losses,
                "win_rate": wr,
                "exp_r": er,
                "pnl": pnl,
                "trades": [
                    {
                        "i": t["i"],
                        "t": str(df.index[int(t["i"])])[:19],
                        "side": t["side"],
                        "r": t["r"],
                        "pnl": t["pnl"],
                    }
                    for t in trades[:40]
                ],
            }
            rows[sym] = detail
            _log(
                f"{sym:8} {label:22} @{tf} RR={rr} sl={sl_atr} "
                f"n={n} W/L={wins}/{losses} WR={wr} E[R]={er} pnl=${pnl:,.0f} "
                f"[{str(week_start)[:10]}→{str(week_end)[:10]}]"
            )
        except Exception as exc:
            rows[sym] = {"symbol": sym, "error": str(exc), "winner_id": wid}
            _log(f"{sym}: FAIL {exc}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "recommend_live": False,
        "window": "last_7d_of_local_dukascopy",
        "risk_usd": RISK_USD,
        "desk": {
            "n": desk_n,
            "wins": desk_wins,
            "win_rate": round(100.0 * desk_wins / desk_n, 1) if desk_n else 0.0,
            "pnl": round(desk_pnl, 2),
            "exp_r": round(desk_pnl / RISK_USD / desk_n, 4) if desk_n else 0.0,
        },
        "per_pair": rows,
        "note": "Last week of available Dukascopy hist (may lag calendar today). Live brains unchanged.",
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _log(
        f"\nDESK n={desk_n} WR={payload['desk']['win_rate']}% "
        f"E[R]={payload['desk']['exp_r']} pnl=${desk_pnl:,.0f}\nWrote {OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
