#!/usr/bin/env python3
"""Build Strategy Lab 1y-hunt equity race curves from gated winners.

Replays each winner on Dukascopy local hist (same window as the scalp hunt),
writes data/strategy_lab_equity_1y.json for the Strategy Lab race chart.

Research / UI only — does not change live brains.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD  # noqa: E402
from analysis.lab_desk_recovery_scalp_hunt import _filter_entries_math  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import (  # noqa: E402
    _catalog,
    _simulate_fast,
)
from analysis.strategy_lab_duka_scalp_1y_hunt import WINDOW_DAYS, _load_frames_1y  # noqa: E402

LAB = Path(ROOT) / "data" / "strategy_lab.json"
HUNT = Path(ROOT) / "data" / "lab_duka_scalp_1y_hunt.json"
RECOVERY = Path(ROOT) / "data" / "lab_desk_recovery_scalp_hunt.json"
OUT = Path(ROOT) / "data" / "strategy_lab_equity_1y.json"
MAX_POINTS = 900


def _parse_sid(sid: str) -> tuple[str, Optional[float], Optional[float]]:
    """bb_2+hf0.45_e0.30 → base, hf_max, min_eff."""
    sid = str(sid or "").strip()
    if "+" not in sid:
        return sid, None, None
    base, tag = sid.split("+", 1)
    hf_max = None
    min_eff = None
    m_hf = re.search(r"hf([0-9]+(?:\.[0-9]+)?)", tag)
    if m_hf:
        hf_max = float(m_hf.group(1))
    m_e = re.search(r"(?:^|[+_])e([0-9]+(?:\.[0-9]+)?)", tag)
    if m_e:
        min_eff = float(m_e.group(1))
    elif re.match(r"e[0-9]", tag):
        min_eff = float(tag[1:].split("_")[0])
    return base, hf_max, min_eff


def _winners() -> dict[str, dict[str, Any]]:
    for path in (LAB, RECOVERY, HUNT):
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        w = raw.get("winners") or raw.get("survivors_book")
        if isinstance(w, dict) and w:
            # Prefer board winners when they look complete
            out_w = {str(k).upper(): dict(v) for k, v in w.items() if v}
            if out_w:
                return out_w
        # strategy_lab / recovery board shape
        pp = raw.get("per_pair") or {}
        out: dict[str, dict[str, Any]] = {}
        for sym, row in pp.items():
            best = (row or {}).get("best") or (row or {}).get("winner")
            if not best:
                continue
            # recovery winner nests stats under full/
            if best.get("full") and not best.get("full_pnl"):
                full = best.get("full") or {}
                best = {
                    **best,
                    "full_pnl": full.get("pnl"),
                    "n": full.get("n"),
                    "oos_exp_r": full.get("oos_exp_r") or full.get("exp_r"),
                }
            # winner nested under hunt may wrap full_span
            if "strategy" not in best and isinstance(best.get("full_span"), dict):
                best = {
                    "strategy": best.get("strategy"),
                    "tf": best.get("tf"),
                    "rr": best.get("rr"),
                    "label": best.get("label"),
                    "family": best.get("family"),
                    "full_pnl": (best.get("full_span") or {}).get("full", {}).get("pnl"),
                    "n": (best.get("full_span") or {}).get("full", {}).get("n"),
                }
            if best.get("strategy") and best.get("tf"):
                out[str(sym).upper()] = best
        if out:
            return out
    return {}


def _catalog_map() -> dict[str, Any]:
    return {sid: (family, label, fn) for sid, family, label, fn in _catalog()}


def _simulate_with_exit_t(df: pd.DataFrame, entries: list, rr: float) -> list[dict]:
    """Like _simulate_fast but includes exit timestamp for the race chart."""
    high = df["high"].to_numpy(dtype=float, copy=False)
    low = df["low"].to_numpy(dtype=float, copy=False)
    idx = df.index
    n = len(df)
    trades: list[dict] = []
    busy_until = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < busy_until or ei >= n - 1:
            continue
        side = e["side"]
        entry = float(e["entry"])
        sl0 = float(e["sl"])
        risk = abs(entry - sl0)
        if risk <= 0:
            continue
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk
        exit_r = None
        exit_i = ei
        if side == "BUY":
            for j in range(ei + 1, n):
                if low[j] <= sl or high[j] >= tp:
                    exit_r = -1.0 if low[j] <= sl else float(rr)
                    exit_i = j
                    break
        else:
            for j in range(ei + 1, n):
                if high[j] >= sl or low[j] <= tp:
                    exit_r = -1.0 if high[j] >= sl else float(rr)
                    exit_i = j
                    break
        if exit_r is None:
            continue
        ts = idx[exit_i]
        try:
            t_iso = pd.Timestamp(ts).tz_convert("UTC").isoformat()
        except Exception:
            t_iso = str(ts)
        trades.append(
            {
                "t": t_iso,
                "side": side,
                "r": round(float(exit_r), 4),
                "pnl": round(float(exit_r) * RISK_USD, 2),
            }
        )
        busy_until = exit_i + 2
    return trades


def _downsample_series(
    times: list[str],
    desk: list[float],
    by_symbol: dict[str, list[float]],
    max_points: int = MAX_POINTS,
) -> tuple[list[str], list[float], dict[str, list[float]]]:
    n = len(times)
    if n <= max_points:
        return times, desk, by_symbol
    step = max(1, n // max_points)
    keep = list(range(0, n, step))
    if keep[-1] != n - 1:
        keep.append(n - 1)
    return (
        [times[i] for i in keep],
        [desk[i] for i in keep],
        {s: [ser[i] for i in keep] for s, ser in by_symbol.items()},
    )


def build(*, force: bool = False) -> dict[str, Any]:
    if OUT.exists() and not force:
        try:
            cached = json.loads(OUT.read_text(encoding="utf-8"))
            if cached.get("n_trades", 0) > 0 and cached.get("by_symbol"):
                return cached
        except Exception:
            pass

    winners = _winners()
    if not winners:
        raise RuntimeError("No Strategy Lab winners found in strategy_lab.json / hunt file")

    cat = _catalog_map()
    end = pd.Timestamp.now(tz="UTC").normalize()
    # Align to hunt end if present
    try:
        hunt = json.loads(HUNT.read_text(encoding="utf-8")) if HUNT.exists() else {}
        if hunt.get("end"):
            end = pd.Timestamp(hunt["end"])
            if end.tzinfo is None:
                end = end.tz_localize("UTC")
    except Exception:
        pass
    start = end - pd.Timedelta(days=WINDOW_DAYS)

    prefer = [
        "ETHUSD",
        "BTCUSD",
        "GOLD",
        "NAS100",
        "USDJPY",
        "AUDUSD",
        "USDCHF",
        "UKOIL",
    ]
    ordered = [s for s in prefer if s in winners] + [
        s for s in winners if s not in prefer
    ]

    all_events: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    t0 = time.time()

    lab_mode = "duka_scalp_1y_hunt"
    try:
        lab_raw = json.loads(LAB.read_text(encoding="utf-8")) if LAB.exists() else {}
        lab_mode = str(lab_raw.get("mode") or lab_mode)
    except Exception:
        lab_raw = {}

    for sym in ordered:
        w = winners[sym]
        sid_raw = str(w.get("strategy") or "")
        base, hf_max, min_eff = _parse_sid(sid_raw)
        math_g = w.get("math") or {}
        if isinstance(math_g, dict):
            if math_g.get("hf_max") is not None:
                hf_max = math_g.get("hf_max")
            if math_g.get("min_eff") is not None:
                min_eff = math_g.get("min_eff")
        sid = str(w.get("strategy_base") or base or sid_raw)
        tf = str(w.get("tf") or "15m")
        rr = float(w.get("rr") or 2.0)
        if sid not in cat:
            print(f"  skip {sym}: unknown strategy {sid_raw} (base={sid})", flush=True)
            continue
        family, label, fn = cat[sid]
        tag = ""
        if hf_max is not None or min_eff is not None:
            tag = f" hf≤{hf_max}" if hf_max is not None else ""
            tag += f" eff≥{min_eff}" if min_eff is not None else ""
        print(f"  replay {sym} {sid_raw}@{tf} RR={rr}{tag} …", flush=True)
        frames = _load_frames_1y(sym, start, end)
        df = frames.get(tf)
        if df is None or len(df) < 400:
            print(f"  skip {sym}: no {tf} bars", flush=True)
            continue
        entries = fn(df)
        if hf_max is not None or min_eff is not None:
            entries = _filter_entries_math(
                df, entries, hf_max=hf_max, min_eff=min_eff
            )
        trades = _simulate_with_exit_t(df, entries, rr)
        # Sanity vs hunt full_pnl
        pnl_sum = round(sum(float(t["pnl"]) for t in trades), 2)
        expected = w.get("full_pnl")
        print(
            f"    → n={len(trades)} pnl=${pnl_sum:,.0f}"
            + (f" (hunt ${float(expected):,.0f})" if expected is not None else ""),
            flush=True,
        )
        meta[sym] = {
            "strategy": sid_raw,
            "strategy_base": sid,
            "label": w.get("label") or label,
            "family": family,
            "tf": tf,
            "rr": rr,
            "math": {"hf_max": hf_max, "min_eff": min_eff} if (hf_max is not None or min_eff is not None) else None,
            "n": len(trades),
            "full_pnl": pnl_sum,
            "hunt_full_pnl": expected,
            "bars": len(df),
            "start": str(df.index[0])[:19],
            "end": str(df.index[-1])[:19],
        }
        for t in trades:
            all_events.append({"t": t["t"], "symbol": sym, "pnl": t["pnl"], "side": t["side"]})

    all_events.sort(key=lambda e: e["t"])

    cum = {s: 0.0 for s in ordered if s in meta}
    live_syms = list(cum.keys())
    desk = 0.0
    times: list[str] = []
    desk_series: list[float] = []
    by_symbol: dict[str, list[float]] = {s: [] for s in live_syms}

    if all_events:
        times.append(all_events[0]["t"])
        desk_series.append(0.0)
        for s in live_syms:
            by_symbol[s].append(0.0)

    for e in all_events:
        sym = e["symbol"]
        if sym not in cum:
            continue
        cum[sym] = round(cum[sym] + float(e["pnl"]), 2)
        desk = round(desk + float(e["pnl"]), 2)
        times.append(e["t"])
        desk_series.append(desk)
        for s in live_syms:
            by_symbol[s].append(cum[s])

    times_s, desk_s, by_s = _downsample_series(times, desk_series, by_symbol)
    finals = {s: by_symbol[s][-1] if by_symbol[s] else 0.0 for s in live_syms}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "strategy_lab_winners",
        "mode": lab_mode,
        "risk_usd": RISK_USD,
        "window_days": WINDOW_DAYS,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "n_trades": len(all_events),
        "n_points": len(times_s),
        "symbols": live_syms,
        "times": times_s,
        "desk": desk_s,
        "by_symbol": by_s,
        "finals": finals,
        "desk_final": desk_s[-1] if desk_s else 0.0,
        "per_pair": meta,
        "note": (
            f"{lab_mode} winners · ${RISK_USD}/R · "
            "cumulative closed PnL race (math filters applied when tagged)"
        ),
    }
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(
        f"Wrote {OUT} · trades={payload['n_trades']} points={payload['n_points']} "
        f"desk=${payload['desk_final']:,.0f}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    force = "--force" in sys.argv
    build(force=force)
