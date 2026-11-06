#!/usr/bin/env python3
"""RESEARCH — OOS/WF/torture/MC for the CURRENT live SYMBOL_ENTRY_BRAIN map.

Intraday TFs only (1m/3m/5m/15m). Does not change config.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from analysis._edge_survive_bt import BOOK, RISK_USD, _prep, _stats, monte_carlo, verdict  # noqa: E402
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_ema,
    _collect_fib,
    simulate,
)
from analysis.strategy_lab import _collect_cta, _collect_rsi, _collect_supertrend  # noqa: E402
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402
from analysis.strategy_lab_intraday_hunt import _load_intraday  # noqa: E402
from config import SYMBOL_ENTRY_BRAIN, SYMBOL_SCAN_TFS, symbol_entry_brain, symbol_rr  # noqa: E402

OUT = Path(ROOT) / "data" / "live_wired_battery.json"


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


def _battery(symbol: str, df, *, brain: str, rr: float, tf: str) -> dict[str, Any]:
    n = len(df)
    entries = _collector(brain)(df)
    print(
        f"  {symbol}@{tf} brain={brain} rr={rr:g} bars={n} entries={len(entries)}",
        flush=True,
    )

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
    wf = []
    for k in range(4):
        test_start = int(n * (0.40 + 0.15 * k))
        test_end = int(n * (0.55 + 0.15 * k)) - 1
        if test_end <= test_start or test_start < 80:
            continue
        wf.append(
            {
                "fold": k + 1,
                "train": _stats(gen(end_i=test_start - 1)),
                "oos": _stats(gen(start_i=test_start, end_i=test_end)),
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
        f"    full n={full['n']} E[R]={full['exp_r']} | OOS E[R]={oos['exp_r']} "
        f"| WF {wf_g}/{len(wf)} | MC {mc.get('prob_profit')}% → {v}",
        flush=True,
    )
    return {
        "symbol": symbol,
        "brain": brain,
        "tf": tf,
        "rr": rr,
        "bars": n,
        "entries": len(entries),
        "full": full,
        "oos": oos,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": wf_g,
        "wf_n": len(wf),
    }


def _score(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    s = sum(1 for r in ok if r.get("verdict") == "SURVIVES")
    f = sum(1 for r in ok if r.get("verdict") == "FRAGILE")
    d = sum(1 for r in ok if r.get("verdict") == "DEAD")
    oos_er = [float(r["oos"]["exp_r"]) for r in ok if r.get("oos")]
    mc_p = [
        float(r["monte_carlo"]["prob_profit"])
        for r in ok
        if (r.get("monte_carlo") or {}).get("prob_profit") is not None
    ]
    score = 1.0 + 0.9 * s + 0.35 * f - 0.7 * d
    if oos_er:
        score += float(np.clip(np.mean(oos_er) * 4.0, -1.5, 2.0))
    if mc_p:
        score += float(np.clip((np.mean(mc_p) - 50.0) / 25.0, -1.0, 1.5))
    return {
        "survives": s,
        "fragile": f,
        "dead": d,
        "mean_oos_exp_r": round(float(np.mean(oos_er)), 4) if oos_er else None,
        "mean_mc_prob": round(float(np.mean(mc_p)), 1) if mc_p else None,
        "score_1_to_10": float(np.clip(round(score, 1), 1.0, 10.0)),
    }


def main() -> int:
    t0 = time.time()
    print(
        "LIVE-WIRED battery · current SYMBOL_ENTRY_BRAIN · intraday TFs · no config writes\n",
        flush=True,
    )
    print("brains:", SYMBOL_ENTRY_BRAIN, flush=True)
    print("scan:", SYMBOL_SCAN_TFS, flush=True)

    rows: list[dict] = []
    for sym in BOOK:
        brain = symbol_entry_brain(sym)
        rr = float(symbol_rr(sym, default=2.0))
        pref = (SYMBOL_SCAN_TFS.get(sym) or ("15m", "5m", "1m"))[0]
        try:
            frames = _load_intraday(sym)
            if not frames:
                raise RuntimeError("no frames")
            tf = pref if pref in frames else next(iter(frames))
            df = frames[tf]
            if len(df) < 200:
                raise RuntimeError(f"only {len(df)} bars on {tf}")
            rows.append(_battery(sym, df, brain=brain, rr=rr, tf=tf))
        except Exception as exc:
            print(f"  FAIL {sym}: {exc}", flush=True)
            rows.append(
                {"symbol": sym, "brain": brain, "error": str(exc), "verdict": "DEAD"}
            )

    desk = _score(rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live_wired_intraday_battery",
        "brains": dict(SYMBOL_ENTRY_BRAIN),
        "scan_tfs": {k: list(v) for k, v in SYMBOL_SCAN_TFS.items()},
        "results": rows,
        "desk": desk,
        "elapsed_s": round(time.time() - t0, 1),
        "note": "Read-only test of currently wired live brains",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### LIVE-WIRED DESK\n", flush=True)
    print(
        f"{'Pair':8} {'Brain':18} {'TF':5} {'Verdict':9} {'Full n':>7} "
        f"{'OOS E[R]':>8} {'WF':>5} {'MC%':>6}",
        flush=True,
    )
    print("-" * 80, flush=True)
    for r in rows:
        if r.get("error"):
            print(f"{r['symbol']:8} {(r.get('brain') or '—'):18} DEAD", flush=True)
            continue
        mc = r.get("monte_carlo") or {}
        print(
            f"{r['symbol']:8} {r['brain']:18} {r['tf']:5} {r['verdict']:9} "
            f"{r['full']['n']:7} {r['oos']['exp_r']:8.3f} "
            f"{r['wf_green']}/{r['wf_n']:>3} {mc.get('prob_profit', '—')!s:>6}",
            flush=True,
        )
    print(
        f"\nDesk {desk['survives']}S/{desk['fragile']}F/{desk['dead']}D "
        f"· {desk['score_1_to_10']}/10 · OOS E[R]={desk['mean_oos_exp_r']} "
        f"· MC={desk['mean_mc_prob']}%",
        flush=True,
    )
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
