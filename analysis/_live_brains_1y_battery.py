#!/usr/bin/env python3
"""Full battery for CURRENT live brains — target ≥1y history.

Primary: 1h ~365d (honest year coverage on free feeds).
Also: native scan TF (1m/3m/5m/15m) on max available depth.

Reports PnL / OOS / WF / torture / MC. Read-only.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from analysis._edge_survive_bt import (  # noqa: E402
    BOOK,
    RISK_USD,
    _fetch_1h,
    _prep,
    _stats,
    monte_carlo,
    verdict,
)
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_ema,
    _collect_fib,
    simulate,
)
from analysis.strategy_lab import (  # noqa: E402
    _clip,
    _collect_cta,
    _collect_rsi,
    _collect_supertrend,
)
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402
from analysis.strategy_lab_intraday_hunt import _load_intraday  # noqa: E402
from config import (  # noqa: E402
    SYMBOL_ENTRY_BRAIN,
    SYMBOL_SCAN_TFS,
    symbol_entry_brain,
    symbol_rr,
    symbol_scan_tfs,
)

OUT = Path(ROOT) / "data" / "live_brains_1y_battery.json"


def _collector(brain: str) -> Callable:
    b = brain.lower().strip()
    if b in ("fib_liq_sweep", "fib_liq"):
        return _collect_fib
    if b in ("ema_20_50_tight", "ema_tight"):
        return lambda d: _collect_ema(d, pull_atr=0.75, disp_atr=0.70)
    if b in ("ema_20_50_pull", "ema_pull"):
        return lambda d: _collect_ema(d, pull_atr=1.0, disp_atr=0.55)
    if b == "cta_10_40":
        return lambda d: _collect_cta(d, 10, 40)
    if b in ("rsi", "rsi_mr"):
        return _collect_rsi
    if b in ("bb_1.5", "bb_1_5"):
        return lambda d: _bb_std(d, 1.5)
    if b in ("bb_2", "bb_2.0"):
        return lambda d: _bb_std(d, 2.0)
    if b in ("st_3", "supertrend_3"):
        return lambda d: _collect_supertrend(d, 3.0)
    raise KeyError(brain)


def _battery(df, entries, rr: float, *, label: str) -> dict[str, Any]:
    n = len(df)

    def gen(start_i=0, end_i=None, **kw):
        spread_r = float(kw.get("spread_r") or 0)
        skip_p = float(kw.get("skip_p") or 0)
        sl_widen = float(kw.get("sl_widen") or 1.0)
        rng = kw.get("rng") or random.Random(0)
        tr = simulate(df, entries, rr, start_i=start_i, end_i=end_i)
        if sl_widen != 1.0:
            out = []
            for t in tr:
                r = float(t["r"])
                if r < 0:
                    r = -1.0 * sl_widen
                out.append({**t, "r": round(r, 4), "pnl": round(r * RISK_USD, 2)})
            tr = out
        if spread_r > 0:
            tr = [
                {
                    **t,
                    "r": round(float(t["r"]) - spread_r, 4),
                    "pnl": round((float(t["r"]) - spread_r) * RISK_USD, 2),
                }
                for t in tr
            ]
        if skip_p > 0 and tr:
            tr = [t for t in tr if rng.random() >= skip_p]
        return tr

    full_tr = gen()
    full = _stats(full_tr)
    split = int(n * 0.70)
    oos = _stats(gen(start_i=split))
    is_st = _stats(gen(end_i=split - 1))
    wf = []
    for k in range(4):
        ts = int(n * (0.40 + 0.15 * k))
        te = int(n * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 80:
            continue
        wf.append(
            {
                "fold": k + 1,
                "train": _stats(gen(end_i=ts - 1)),
                "oos": _stats(gen(start_i=ts, end_i=te)),
            }
        )
    torture = {
        "spread_0.15R": _stats(gen(spread_r=0.15)),
        "skip_30pct": _stats(gen(skip_p=0.30, rng=random.Random(7))),
        "sl_widen_20pct": _stats(gen(sl_widen=1.20)),
    }
    mc = monte_carlo(full_tr)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
    print(
        f"  {label}: n={full['n']} PnL=${full['pnl']:,.0f} E[R]={full['exp_r']} "
        f"| OOS n={oos['n']} PnL=${oos['pnl']:,.0f} E[R]={oos['exp_r']} "
        f"| WF {wf_g}/{len(wf)} | MC {mc.get('prob_profit')}% → {v}",
        flush=True,
    )
    return {
        "bars": n,
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "span_days": round((df.index[-1] - df.index[0]).total_seconds() / 86400.0, 1),
        "entries": len(entries),
        "full": full,
        "in_sample": is_st,
        "oos": oos,
        "walk_forward": [
            {
                "fold": f["fold"],
                "train_pnl": f["train"]["pnl"],
                "train_exp_r": f["train"]["exp_r"],
                "oos_n": f["oos"]["n"],
                "oos_pnl": f["oos"]["pnl"],
                "oos_exp_r": f["oos"]["exp_r"],
            }
            for f in wf
        ],
        "torture": {k: {"n": v["n"], "pnl": v["pnl"], "exp_r": v["exp_r"]} for k, v in torture.items()},
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": wf_g,
        "wf_n": len(wf),
    }


def main() -> int:
    t0 = time.time()
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_1y = (end - timedelta(days=365)).isoformat()
    end_s = end.isoformat()
    print(
        f"LIVE BRAINS · ≥1y battery · {start_1y}→{end_s} · ${RISK_USD}/R\n"
        f"Brains: {SYMBOL_ENTRY_BRAIN}\n",
        flush=True,
    )

    rows: list[dict] = []
    for sym in BOOK:
        brain = symbol_entry_brain(sym)
        rr = float(symbol_rr(sym, default=2.0))
        pref = (symbol_scan_tfs(sym) or ("15m",))[0]
        print(f"\n=== {sym} brain={brain} rr={rr:g} native_tf={pref} ===", flush=True)
        row: dict[str, Any] = {
            "symbol": sym,
            "brain": brain,
            "rr": rr,
            "native_tf": pref,
        }
        try:
            raw = _fetch_1h(sym, start_1y, end_s)
            df1h = _prep(_clip(raw, start_1y, end_s))
            start_dt = datetime.fromisoformat(start_1y).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            df1h = df1h[(df1h.index >= start_dt) & (df1h.index <= end_dt)]
            if len(df1h) < 200:
                raise RuntimeError(f"1h only {len(df1h)} bars")
            entries = _collector(brain)(df1h)
            row["year_1h"] = _battery(
                df1h, entries, rr, label=f"{sym}@1h~1y"
            )
        except Exception as exc:
            print(f"  1h FAIL: {exc}", flush=True)
            row["year_1h"] = {"error": str(exc), "verdict": "DEAD"}

        try:
            frames = _load_intraday(sym)
            tf = pref if pref in frames else next(iter(frames))
            df = frames[tf]
            entries = _collector(brain)(df)
            row["native"] = _battery(
                df, entries, rr, label=f"{sym}@{tf}_native"
            )
            row["native"]["tf"] = tf
        except Exception as exc:
            print(f"  native FAIL: {exc}", flush=True)
            row["native"] = {"error": str(exc), "verdict": "DEAD"}

        rows.append(row)

    # Desk scores from 1y 1h (honest year)
    y_ok = [r for r in rows if (r.get("year_1h") or {}).get("full")]
    desk_pnl = sum(float(r["year_1h"]["full"]["pnl"]) for r in y_ok)
    desk_oos = sum(float(r["year_1h"]["oos"]["pnl"]) for r in y_ok)
    s = sum(1 for r in rows if (r.get("year_1h") or {}).get("verdict") == "SURVIVES")
    f = sum(1 for r in rows if (r.get("year_1h") or {}).get("verdict") == "FRAGILE")
    d = sum(1 for r in rows if (r.get("year_1h") or {}).get("verdict") == "DEAD")

    n_ok = [r for r in rows if (r.get("native") or {}).get("full")]
    nat_pnl = sum(float(r["native"]["full"]["pnl"]) for r in n_ok)
    nat_oos = sum(float(r["native"]["oos"]["pnl"]) for r in n_ok)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live_brains_1y_full_battery",
        "window_1y": {"start": start_1y, "end": end_s, "tf": "1h"},
        "risk_usd": RISK_USD,
        "brains": dict(SYMBOL_ENTRY_BRAIN),
        "scan_tfs": {k: list(v) for k, v in SYMBOL_SCAN_TFS.items()},
        "results": rows,
        "desk_1y_1h": {
            "survives": s,
            "fragile": f,
            "dead": d,
            "full_pnl": round(desk_pnl, 2),
            "oos_pnl": round(desk_oos, 2),
            "pairs": len(y_ok),
        },
        "desk_native_max": {
            "full_pnl": round(nat_pnl, 2),
            "oos_pnl": round(nat_oos, 2),
            "pairs": len(n_ok),
            "note": "Max available 1m/3m/5m/15m depth (usually weeks–months, not full year)",
        },
        "elapsed_s": round(time.time() - t0, 1),
        "note": (
            "1h~1y = honest year coverage. Native TF = live scan TF on max feed depth. "
            "Read-only — no config changes."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n" + "=" * 100, flush=True)
    print("### 1-YEAR (1h) RESULTS — CURRENT LIVE BRAINS\n", flush=True)
    print(
        f"{'Pair':8} {'Brain':18} {'Days':>5} {'n':>5} {'Full PnL':>12} {'OOS PnL':>10} "
        f"{'OOS E[R]':>8} {'WF':>5} {'MC%':>6} {'Verdict':9}",
        flush=True,
    )
    print("-" * 110, flush=True)
    for r in rows:
        y = r.get("year_1h") or {}
        if y.get("error"):
            print(f"{r['symbol']:8} {r['brain']:18} FAIL {y['error'][:40]}", flush=True)
            continue
        mc = y.get("monte_carlo") or {}
        print(
            f"{r['symbol']:8} {r['brain']:18} {y.get('span_days', 0):5.0f} "
            f"{y['full']['n']:5} ${y['full']['pnl']:11,.0f} ${y['oos']['pnl']:9,.0f} "
            f"{y['oos']['exp_r']:8.3f} {y['wf_green']}/{y['wf_n']:>3} "
            f"{mc.get('prob_profit', '—')!s:>6} {y['verdict']:9}",
            flush=True,
        )
    print(
        f"\nDesk 1y/1h: {s}S/{f}F/{d}D · Full PnL ${desk_pnl:,.0f} · OOS PnL ${desk_oos:,.0f}",
        flush=True,
    )

    print("\n### NATIVE SCAN TF (max available depth)\n", flush=True)
    print(
        f"{'Pair':8} {'TF':5} {'Days':>5} {'n':>5} {'Full PnL':>12} {'OOS PnL':>10} "
        f"{'Verdict':9}",
        flush=True,
    )
    print("-" * 70, flush=True)
    for r in rows:
        n = r.get("native") or {}
        if n.get("error"):
            print(f"{r['symbol']:8} FAIL", flush=True)
            continue
        print(
            f"{r['symbol']:8} {n.get('tf', r.get('native_tf')):5} {n.get('span_days', 0):5.0f} "
            f"{n['full']['n']:5} ${n['full']['pnl']:11,.0f} ${n['oos']['pnl']:9,.0f} "
            f"{n['verdict']:9}",
            flush=True,
        )
    print(
        f"\nDesk native: Full PnL ${nat_pnl:,.0f} · OOS PnL ${nat_oos:,.0f}",
        flush=True,
    )
    print(f"\nWrote {OUT} in {payload['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
