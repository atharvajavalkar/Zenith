#!/usr/bin/env python3
"""Build Strategy Lab equity race for adaptive v3 absolute winners."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    BOOK,
    TF,
    _classify_regime,
    _equity_curve,
    _load,
    _sides_bimod,
    _sides_kalman,
    _sides_perm,
    _sides_snr,
    _static_sides_for_winner,
)
from analysis.lab_adaptive_regime_v2_bt import (  # noqa: E402
    _family,
    _gated_sides,
    _switch_sides,
)
from analysis.lab_adaptive_minute_day_v3_bt import (  # noqa: E402
    CAP_YEARS,
    _calendar,
    _catalog,
)
from analysis.lab_zenith_math_invention_deep_bt import (  # noqa: E402
    _atr_sl_entries,
    _trades_with_i,
)
from analysis._edge_survive_bt import RISK_USD  # noqa: E402

HUNT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"
V3 = Path(ROOT) / "data" / "lab_adaptive_minute_day_v3.json"


def build_equity() -> dict[str, Any]:
    hunt = json.loads(HUNT.read_text(encoding="utf-8")) if HUNT.exists() else {}
    per_hunt = hunt.get("per_pair") or {}
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    per = v3.get("per_pair") or {}
    formulas = {fid: fn for fid, _lab, fn in _catalog()}

    symbols = []
    series_map: dict[str, tuple[list, list]] = {}
    finals: dict[str, float] = {}
    per_pair_meta: dict[str, Any] = {}
    trades_total = 0

    for sym in BOOK:
        row = per.get(sym) or {}
        if row.get("error"):
            continue
        aw = row.get("absolute_winner") or "static"
        w = (per_hunt.get(sym) or {}).get("winner") or {}
        wid = str(w.get("id") or "")
        fam = _family(wid)
        rr = float(w.get("rr") or row.get("rr") or 2.4)
        sl = float(w.get("sl_atr") or row.get("sl_atr") or 1.2)

        df = _load(sym, TF, CAP_YEARS)
        static, _, _ = _static_sides_for_winner(df, wid)
        regime = _classify_regime(df)
        alt_trend = (
            _sides_bimod(df, 0.55)
            if sym in ("ETHUSD", "BTCUSD")
            else _sides_perm(df, 0.80)
            if sym == "GOLD"
            else _sides_snr(df, 0.40)
        )
        alt_fade = _sides_kalman(df, 2.2)
        v2 = {
            "static": static,
            "gated": _gated_sides(static, regime, fam),
            "switch": _switch_sides(static, regime, fam, alt_trend, alt_fade),
        }
        pref = row.get("v2_preferred") or "static"
        ctx = {
            "sym": sym,
            "static": static,
            "regime": regime,
            "cal": _calendar(df),
            "v2_preferred_sides": v2.get(pref, static),
            "family": fam,
        }
        if aw in formulas:
            sides = formulas[aw](df, ctx)
        elif aw in v2:
            sides = v2[aw]
        elif aw == "v2_preferred":
            sides = v2.get(pref, static)
        else:
            sides = static

        trades = _trades_with_i(df, _atr_sl_entries(df, sides, sl_atr=sl), rr)
        # 1y window for race
        end = df.index[-1]
        start = end - __import__("pandas").Timedelta(days=365)
        i0 = int(np.searchsorted(np.asarray(df.index), start))
        trades_1y = [t for t in trades if int(t["i"]) >= i0]
        eq = _equity_curve(trades_1y, df)
        if not eq.get("equity"):
            continue
        symbols.append(sym)
        series_map[sym] = (eq["times"], eq["equity"])
        finals[sym] = float(eq["final"])
        trades_total += len(trades_1y)
        per_pair_meta[sym] = {
            "strategy": f"adaptive_v3_{aw}",
            "mode": aw,
            "full_pnl": finals[sym],
            "tf": "5m",
            "rr": rr,
        }

    max_len = max((len(v[1]) for v in series_map.values()), default=0)
    n_pts = min(900, max_len) if max_len else 0
    times: list[str] = []
    desk_s: list[float] = []
    by_s: dict[str, list[float]] = {s: [] for s in symbols}
    for i in range(n_pts):
        frac = i / max(1, n_pts - 1)
        desk = 0.0
        t_pick = None
        for sym in symbols:
            ts, ser = series_map[sym]
            j = min(len(ser) - 1, int(frac * (len(ser) - 1)))
            by_s[sym].append(ser[j])
            desk += ser[j]
            if t_pick is None and ts:
                t_pick = ts[min(len(ts) - 1, j)]
        times.append(t_pick or f"p{i}")
        desk_s.append(round(desk, 2))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "adaptive_minute_day_v3_absolute",
        "mode": "adaptive_minute_day_v3",
        "risk_usd": RISK_USD,
        "window_days": 365,
        "n_trades": trades_total,
        "n_points": len(times),
        "symbols": symbols,
        "times": times,
        "desk": desk_s,
        "by_symbol": by_s,
        "finals": finals,
        "desk_final": desk_s[-1] if desk_s else 0.0,
        "per_pair": per_pair_meta,
        "note": "Adaptive v3 absolute-winner equity race (1y, index-aligned).",
    }


if __name__ == "__main__":
    out = Path(ROOT) / "data" / "strategy_lab_equity_1y.json"
    payload = build_equity()
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {out} desk_final={payload.get('desk_final')}", flush=True)
