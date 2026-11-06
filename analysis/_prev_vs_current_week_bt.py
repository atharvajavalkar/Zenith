#!/usr/bin/env python3
"""Compare PREVIOUS vs CURRENT live brains — ~1 week + 1y/1h.

READ-ONLY. Does not change .env / brains / config.
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
    _collect_breakout,
    _collect_ema,
    _collect_fib,
    _collect_sma,
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
from config import symbol_entry_brain, symbol_rr, symbol_scan_tfs  # noqa: E402

OUT = Path(ROOT) / "data" / "prev_vs_current_week.json"

# Brains BEFORE the intraday-lab rewire
PREVIOUS = {
    "GOLD": ("fib_liq_sweep", "15m"),
    "USDCHF": ("ema_20_50_tight", "5m"),
    "USDJPY": ("fib_liq_sweep", "15m"),
    "NAS100": ("sma_20_50_cross", "15m"),
    "UKOIL": ("breakout_20", "15m"),
    "AUDUSD": ("ema_20_50_pull", "15m"),
    "BTCUSD": ("ema_20_50_tight", "5m"),
    "ETHUSD": ("ema_20_50_tight", "5m"),
}


def _collector(brain: str) -> Callable:
    b = brain.lower().strip()
    if b in ("fib_liq_sweep", "fib_liq"):
        return _collect_fib
    if b in ("ema_20_50_tight", "ema_tight"):
        return lambda d: _collect_ema(d, pull_atr=0.75, disp_atr=0.70)
    if b in ("ema_20_50_pull", "ema_pull"):
        return lambda d: _collect_ema(d, pull_atr=1.0, disp_atr=0.55)
    if b == "sma_20_50_cross":
        return _collect_sma
    if b == "breakout_20":
        return lambda d: _collect_breakout(d, 20)
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


def _battery(df, entries, rr: float) -> dict[str, Any]:
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
    wf = []
    for k in range(4):
        ts = int(n * (0.40 + 0.15 * k))
        te = int(n * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 40:
            continue
        wf.append(
            {
                "fold": k + 1,
                "oos_pnl": _stats(gen(start_i=ts, end_i=te))["pnl"],
                "oos_exp_r": _stats(gen(start_i=ts, end_i=te))["exp_r"],
            }
        )
    torture = {
        "spread_0.15R": _stats(gen(spread_r=0.15)),
        "skip_30pct": _stats(gen(skip_p=0.30, rng=random.Random(7))),
        "sl_widen_20pct": _stats(gen(sl_widen=1.20)),
    }
    mc = monte_carlo(full_tr)
    v = verdict(full, oos, wf if wf else [], torture, mc)
    # fix wf for verdict — need full fold dicts
    wf_full = []
    for k in range(4):
        ts = int(n * (0.40 + 0.15 * k))
        te = int(n * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 40:
            continue
        wf_full.append(
            {
                "fold": k + 1,
                "train": _stats(gen(end_i=ts - 1)),
                "oos": _stats(gen(start_i=ts, end_i=te)),
            }
        )
    v = verdict(full, oos, wf_full, torture, mc)
    wf_g = sum(1 for f in wf_full if f["oos"]["exp_r"] > 0)
    return {
        "bars": n,
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "span_days": round((df.index[-1] - df.index[0]).total_seconds() / 86400.0, 2),
        "entries": len(entries),
        "full": full,
        "oos": oos,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": wf_g,
        "wf_n": len(wf_full),
        "torture_exp_r": {k: t["exp_r"] for k, t in torture.items()},
    }


def _run_setup(sym: str, brain: str, tf: str, frames: dict, df_1h, rr: float, week_start) -> dict:
    out: dict[str, Any] = {"brain": brain, "tf": tf}
    # Week on preferred TF (clip last 7d)
    try:
        base = frames.get(tf) or frames.get("5m") or next(iter(frames.values()))
        dfw = base[base.index >= week_start]
        if len(dfw) < 80:
            dfw = base.iloc[-min(len(base), 2000) :]
        entries = _collector(brain)(dfw)
        out["week"] = _battery(dfw, entries, rr)
        out["week"]["tf_used"] = tf if tf in frames else list(frames.keys())[0]
    except Exception as e:
        out["week"] = {"error": str(e), "verdict": "DEAD", "full": {"pnl": 0, "n": 0}, "oos": {"pnl": 0}}

    # 1y 1h
    try:
        entries = _collector(brain)(df_1h)
        out["year_1h"] = _battery(df_1h, entries, rr)
    except Exception as e:
        out["year_1h"] = {"error": str(e), "verdict": "DEAD", "full": {"pnl": 0, "n": 0}, "oos": {"pnl": 0}}
    return out


def main() -> int:
    t0 = time.time()
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_1y = (end - timedelta(days=365)).isoformat()
    end_s = end.isoformat()
    week_start = datetime.now(timezone.utc) - timedelta(days=7)

    print(
        f"PREV vs CURRENT · week(~7d) + 1y/1h · READ-ONLY · ${RISK_USD}/R\n"
        f"Window 1y: {start_1y}→{end_s}\n",
        flush=True,
    )

    comparisons = []
    for sym in BOOK:
        rr = float(symbol_rr(sym, default=2.0))
        cur_brain = symbol_entry_brain(sym)
        cur_tf = (symbol_scan_tfs(sym) or ("15m",))[0]
        prev_brain, prev_tf = PREVIOUS[sym]

        print(f"\n=== {sym} rr={rr:g} ===", flush=True)
        print(f"  PREV {prev_brain}@{prev_tf}  vs  CUR {cur_brain}@{cur_tf}", flush=True)

        try:
            frames = _load_intraday(sym)
        except Exception as e:
            print(f"  load fail: {e}", flush=True)
            frames = {}

        try:
            raw = _fetch_1h(sym, start_1y, end_s)
            df1h = _prep(_clip(raw, start_1y, end_s))
            start_dt = datetime.fromisoformat(start_1y).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            df1h = df1h[(df1h.index >= start_dt) & (df1h.index <= end_dt)]
        except Exception as e:
            print(f"  1h fail: {e}", flush=True)
            df1h = None

        if not frames or df1h is None or len(df1h) < 100:
            comparisons.append({"symbol": sym, "error": "no_data"})
            continue

        prev = _run_setup(sym, prev_brain, prev_tf, frames, df1h, rr, week_start)
        cur = _run_setup(sym, cur_brain, cur_tf, frames, df1h, rr, week_start)

        # Who wins week / year?
        def better(a, b, key_path=("week", "oos", "pnl")):
            def get(d, path):
                for p in path:
                    d = (d or {}).get(p) if isinstance(d, dict) else None
                return d

            ap, bp = get(a, key_path), get(b, key_path)
            if ap is None and bp is None:
                return "tie"
            if ap is None:
                return "current"
            if bp is None:
                return "previous"
            if float(bp) > float(ap) + 1:
                return "current"
            if float(ap) > float(bp) + 1:
                return "previous"
            # verdict tie-break
            rank = {"SURVIVES": 2, "FRAGILE": 1, "DEAD": 0}
            av = rank.get((a.get(key_path[0]) or {}).get("verdict"), 0)
            bv = rank.get((b.get(key_path[0]) or {}).get("verdict"), 0)
            if bv > av:
                return "current"
            if av > bv:
                return "previous"
            return "tie"

        week_winner = better(prev, cur, ("week", "oos", "pnl"))
        # also compare week full pnl
        pw = float((prev.get("week") or {}).get("full", {}).get("pnl") or 0)
        cw = float((cur.get("week") or {}).get("full", {}).get("pnl") or 0)
        if abs(cw - pw) > 1:
            week_full = "current" if cw > pw else "previous"
        else:
            week_full = week_winner

        year_winner = better(prev, cur, ("year_1h", "oos", "pnl"))
        py = float((prev.get("year_1h") or {}).get("full", {}).get("pnl") or 0)
        cy = float((cur.get("year_1h") or {}).get("full", {}).get("pnl") or 0)
        year_full = "current" if cy > py + 1 else "previous" if py > cy + 1 else "tie"

        same = prev_brain.replace("fib_liq_sweep", "fib_liq") == cur_brain.replace(
            "fib_liq_sweep", "fib_liq"
        ) and prev_brain == cur_brain

        comparisons.append(
            {
                "symbol": sym,
                "rr": rr,
                "previous": prev,
                "current": cur,
                "unchanged": prev_brain == cur_brain and prev_tf == cur_tf,
                "week_oos_winner": week_winner,
                "week_full_winner": week_full,
                "year_oos_winner": year_winner,
                "year_full_winner": year_full,
            }
        )
        pw_o = (prev.get("week") or {}).get("oos", {})
        cw_o = (cur.get("week") or {}).get("oos", {})
        print(
            f"  WEEK  PREV n={(prev.get('week') or {}).get('full',{}).get('n')} "
            f"PnL=${(prev.get('week') or {}).get('full',{}).get('pnl',0):,.0f} "
            f"OOS=${pw_o.get('pnl',0):,.0f} {(prev.get('week') or {}).get('verdict')}  |  "
            f"CUR n={(cur.get('week') or {}).get('full',{}).get('n')} "
            f"PnL=${(cur.get('week') or {}).get('full',{}).get('pnl',0):,.0f} "
            f"OOS=${cw_o.get('pnl',0):,.0f} {(cur.get('week') or {}).get('verdict')}  "
            f"→ {week_full}",
            flush=True,
        )
        print(
            f"  YEAR  PREV PnL=${(prev.get('year_1h') or {}).get('full',{}).get('pnl',0):,.0f} "
            f"OOS=${(prev.get('year_1h') or {}).get('oos',{}).get('pnl',0):,.0f} "
            f"{(prev.get('year_1h') or {}).get('verdict')}  |  "
            f"CUR PnL=${(cur.get('year_1h') or {}).get('full',{}).get('pnl',0):,.0f} "
            f"OOS=${(cur.get('year_1h') or {}).get('oos',{}).get('pnl',0):,.0f} "
            f"{(cur.get('year_1h') or {}).get('verdict')}  → {year_full}",
            flush=True,
        )

    # Desk totals
    def desk_sum(side: str, window: str, field: str) -> float:
        total = 0.0
        for c in comparisons:
            if c.get("error"):
                continue
            block = (c.get(side) or {}).get(window) or {}
            total += float((block.get(field) or {}).get("pnl") or 0)
        return total

    week_prev_full = desk_sum("previous", "week", "full")
    week_cur_full = desk_sum("current", "week", "full")
    week_prev_oos = desk_sum("previous", "week", "oos")
    week_cur_oos = desk_sum("current", "week", "oos")
    year_prev_full = desk_sum("previous", "year_1h", "full")
    year_cur_full = desk_sum("current", "year_1h", "full")
    year_prev_oos = desk_sum("previous", "year_1h", "oos")
    year_cur_oos = desk_sum("current", "year_1h", "oos")

    week_pair_wins = {
        "previous": sum(1 for c in comparisons if c.get("week_full_winner") == "previous"),
        "current": sum(1 for c in comparisons if c.get("week_full_winner") == "current"),
        "tie": sum(1 for c in comparisons if c.get("week_full_winner") == "tie"),
    }
    year_pair_wins = {
        "previous": sum(1 for c in comparisons if c.get("year_full_winner") == "previous"),
        "current": sum(1 for c in comparisons if c.get("year_full_winner") == "current"),
        "tie": sum(1 for c in comparisons if c.get("year_full_winner") == "tie"),
    }

    if year_cur_full > year_prev_full and week_cur_full >= week_prev_full:
        verdict = "current_better"
    elif year_prev_full > year_cur_full and week_prev_full >= week_cur_full:
        verdict = "previous_better"
    elif year_prev_full > year_cur_full:
        verdict = "previous_better_on_1y_mixed_week"
    elif week_cur_full > week_prev_full:
        verdict = "current_better_on_week_only"
    else:
        verdict = "mixed"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "prev_vs_current_readonly",
        "risk_usd": RISK_USD,
        "previous_brains": PREVIOUS,
        "current_brains": {s: symbol_entry_brain(s) for s in BOOK},
        "comparisons": comparisons,
        "desk": {
            "week": {
                "previous_full_pnl": round(week_prev_full, 2),
                "current_full_pnl": round(week_cur_full, 2),
                "previous_oos_pnl": round(week_prev_oos, 2),
                "current_oos_pnl": round(week_cur_oos, 2),
                "pair_wins": week_pair_wins,
            },
            "year_1h": {
                "previous_full_pnl": round(year_prev_full, 2),
                "current_full_pnl": round(year_cur_full, 2),
                "previous_oos_pnl": round(year_prev_oos, 2),
                "current_oos_pnl": round(year_cur_oos, 2),
                "pair_wins": year_pair_wins,
            },
            "overall_verdict": verdict,
        },
        "elapsed_s": round(time.time() - t0, 1),
        "note": "No live changes. Previous = pre-lab-rewire map. Current = wired intraday hunt map.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n" + "=" * 90, flush=True)
    print("### DESK SUMMARY (no changes made)\n", flush=True)
    print(
        f"WEEK  (~7d native TF):  PREV Full ${week_prev_full:,.0f} / OOS ${week_prev_oos:,.0f}  vs  "
        f"CUR Full ${week_cur_full:,.0f} / OOS ${week_cur_oos:,.0f}",
        flush=True,
    )
    print(f"      pair wins (full PnL): {week_pair_wins}", flush=True)
    print(
        f"YEAR  (1h ~365d):       PREV Full ${year_prev_full:,.0f} / OOS ${year_prev_oos:,.0f}  vs  "
        f"CUR Full ${year_cur_full:,.0f} / OOS ${year_cur_oos:,.0f}",
        flush=True,
    )
    print(f"      pair wins (full PnL): {year_pair_wins}", flush=True)
    print(f"\n→ Overall: {verdict}", flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
