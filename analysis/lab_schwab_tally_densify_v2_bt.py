#!/usr/bin/env python3
"""RESEARCH ONLY — Schwab tally densify v2: wide lookback permutations.

Expands v1 with combinatorial lookback sets (2/3/4/5-way) from a richer pool,
plus lot/hold/TP slices biased to what already showed hope (2× lot, multi-day
c2c; keep some TP=1.0/1.5).

Dukascopy daily. Nebula / live / .env untouched.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD, _stats, monte_carlo, verdict  # noqa: E402
from analysis.xm_session_costs import spread_price  # noqa: E402
from data.hist_local import clear_hist_cache, load_local_ohlcv  # noqa: E402

BOOK = [
    "GOLD",
    "BTCUSD",
    "UKOIL",
    "NAS100",
    "AUDUSD",
    "ETHUSD",
    "USDJPY",
    "USDCHF",
]

# Richer pool → all 2/3-sets + 4/5-sets from a focused core
POOL = (3, 5, 7, 10, 14, 20, 21, 28, 30, 42, 56, 60, 63, 84, 90)
POOL4 = (5, 7, 10, 14, 21, 28, 42, 56, 63, 84)
POOL5 = (7, 10, 14, 21, 42, 63, 84)


def _build_lookback_sets() -> list[tuple[int, ...]]:
    sets: set[tuple[int, ...]] = set()
    for k in (2, 3):
        for comb in itertools.combinations(POOL, k):
            sets.add(tuple(sorted(comb)))
    for comb in itertools.combinations(POOL4, 4):
        sets.add(tuple(sorted(comb)))
    for comb in itertools.combinations(POOL5, 5):
        sets.add(tuple(sorted(comb)))
    # keep v1 winners / baselines explicitly
    for extra in (
        (7, 14, 21, 42),
        (10, 21, 42, 84),
        (14, 42),
        (7, 21),
        (4, 8, 16, 32),
        (5, 20, 60),
        (7, 14, 28, 56),
        (10, 20, 40),
    ):
        sets.add(tuple(sorted(extra)))
    return sorted(sets, key=lambda t: (len(t), t))


LOOKBACK_SETS = _build_lookback_sets()

# Bias hard to v1 winners so lookback breadth fits in one run
TP_RS: tuple[Optional[float], ...] = (None, 1.0)  # c2c + mild TP
LOT_MULTS = (2.0, 2.5)
SIZE_MODES = ("strict", "linear")
HOLD_DAYS = (3, 4, 5)
SL_ATR = 1.0

# 1y primary + full confirm
HORIZONS = {"1y": 365, "full": None}

OUT = Path(ROOT) / "data" / "lab_schwab_tally_densify_v2.json"
LOG = Path(ROOT) / "data" / "lab_schwab_tally_densify_v2.log"
MD = Path(ROOT) / "data" / "lab_schwab_tally_densify_v2.md"
PROGRESS = Path(ROOT) / "data" / "lab_schwab_tally_densify_v2.progress.json"
PARTIAL = Path(ROOT) / "data" / "lab_schwab_tally_densify_v2.partial.json"


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _load_daily(symbol: str) -> Optional[pd.DataFrame]:
    for tf in ("15m", "5m", "1m"):
        df = load_local_ohlcv(symbol, tf=tf)
        if df is None or len(df) < 500:
            continue
        daily = pd.DataFrame(
            {
                "open": df["open"].resample("1D").first(),
                "high": df["high"].resample("1D").max(),
                "low": df["low"].resample("1D").min(),
                "close": df["close"].resample("1D").last(),
                "volume": df["volume"].resample("1D").sum(),
            }
        ).dropna()
        if len(daily) >= 120:
            return daily
    return None


def _atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return pd.Series(tr).rolling(n, min_periods=max(5, n // 2)).mean().to_numpy(dtype=float)


def _vote_cache(c: np.ndarray, max_lb: int) -> dict[int, np.ndarray]:
    """Precompute +1/-1/0 vote series for every lookback used."""
    out = {}
    for lb in range(1, max_lb + 1):
        prev = np.roll(c, lb).astype(float)
        prev[:lb] = np.nan
        v = np.where(c > prev, 1.0, np.where(c < prev, -1.0, 0.0))
        v[:lb] = np.nan
        out[lb] = v
    return out


def _tally_from_cache(votes: dict[int, np.ndarray], lookbacks: tuple[int, ...]) -> np.ndarray:
    stacked = np.vstack([votes[lb] for lb in lookbacks])
    return np.nansum(stacked, axis=0)


def _base_size(tally: float, n_votes: int, mode: str) -> float:
    if not np.isfinite(tally) or tally == 0:
        return 0.0
    a = abs(int(tally))
    if mode == "linear":
        return min(1.0, a / float(max(1, n_votes)))
    if a == n_votes:
        return 1.0
    if n_votes >= 4 and a == 2:
        return 0.5
    if n_votes == 5 and a in (1, 3):  # weak / mid — sit on 1, half on 3
        return 0.5 if a == 3 else 0.0
    if n_votes == 3 and a == 1:
        return 0.5
    return 0.0


def _simulate_fast(
    *,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    atr: np.ndarray,
    tally: np.ndarray,
    lookbacks: tuple[int, ...],
    size_mode: str,
    lot_mult: float,
    tp_r: Optional[float],
    hold_days: int,
    spread_r_default: float,
) -> list[dict]:
    n_votes = len(lookbacks)
    n = len(c)
    need = max(lookbacks) + 3
    trades: list[dict] = []
    i = need
    busy_until = -1
    while i < n - 1:
        if i < busy_until:
            i += 1
            continue
        a = float(atr[i])
        tv = float(tally[i])
        if not np.isfinite(a) or a <= 0 or not np.isfinite(tv):
            i += 1
            continue
        base = _base_size(tv, n_votes, size_mode)
        if base <= 0:
            i += 1
            continue
        side = 1 if tv > 0 else -1
        size = base * float(lot_mult)
        entry_i = i + 1
        if entry_i >= n:
            break
        entry = float(o[entry_i])
        sl_dist = SL_ATR * a
        if sl_dist <= 0:
            i += 1
            continue
        if side > 0:
            sl = entry - sl_dist
            tp = entry + (tp_r * a) if tp_r is not None else None
        else:
            sl = entry + sl_dist
            tp = entry - (tp_r * a) if tp_r is not None else None

        spread_r = spread_r_default
        max_j = min(n - 1, entry_i + hold_days - 1)
        exit_px = None
        exit_i = entry_i
        reason = "time"

        if tp_r is None:
            exit_i = max_j
            exit_px = float(c[exit_i])
            reason = "c2c"
        else:
            hit = False
            for j in range(entry_i, max_j + 1):
                if side > 0:
                    if float(l[j]) <= sl:
                        exit_px, exit_i, reason, hit = sl, j, "sl", True
                        break
                    if tp is not None and float(h[j]) >= tp:
                        exit_px, exit_i, reason, hit = tp, j, "tp", True
                        break
                else:
                    if float(h[j]) >= sl:
                        exit_px, exit_i, reason, hit = sl, j, "sl", True
                        break
                    if tp is not None and float(l[j]) <= tp:
                        exit_px, exit_i, reason, hit = tp, j, "tp", True
                        break
            if not hit:
                exit_i = max_j
                exit_px = float(c[exit_i])
                reason = "time"

        move = (float(exit_px) - entry) * side
        r = size * (move / sl_dist) - size * spread_r
        trades.append(
            {
                "i": i,
                "tally": int(tv),
                "size": round(size, 3),
                "side": side,
                "r": round(float(r), 4),
                "pnl": round(float(r) * RISK_USD, 2),
                "reason": reason,
            }
        )
        busy_until = exit_i + 1
        i = exit_i + 1
    return trades


def _battery(full_tr: list[dict], n_bars: int) -> dict[str, Any]:
    if not full_tr or n_bars < 80:
        empty = _stats([])
        return {
            "full": empty,
            "oos": empty,
            "monte_carlo": {},
            "verdict": "DEAD",
            "wf_green": 0,
            "wf_n": 0,
            "tests": {},
            "entries": 0,
        }

    def by_range(start_i: int = 0, end_i: Optional[int] = None) -> list[dict]:
        end_i = n_bars - 1 if end_i is None else end_i
        return [t for t in full_tr if start_i <= int(t["i"]) <= end_i]

    full = _stats(full_tr)
    # Fast reject: clearly dead — skip heavy WF/MC
    if full.get("n", 0) < 12 or float(full.get("exp_r") or 0) < -0.05:
        return {
            "full": full,
            "oos": _stats([]),
            "monte_carlo": {},
            "verdict": "DEAD",
            "wf_green": 0,
            "wf_n": 0,
            "tests": {"1_full": full},
            "entries": len(full_tr),
        }

    split = int(n_bars * 0.70)
    oos = _stats(by_range(start_i=split))
    wf = []
    wf_details = {}
    for k in range(4):
        ts = int(n_bars * (0.40 + 0.15 * k))
        te = int(n_bars * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 60:
            continue
        fold = _stats(by_range(start_i=ts, end_i=te))
        wf.append({"fold": k + 1, "oos": fold})
        wf_details[f"wf_fold_{k+1}"] = fold
    spread_tr = [
        {**t, "r": round(float(t["r"]) - 0.15, 4), "pnl": round((float(t["r"]) - 0.15) * RISK_USD, 2)}
        for t in full_tr
    ]
    rng = random.Random(7)
    skip_tr = [t for t in full_tr if rng.random() >= 0.30]
    torture = {"spread_0.15R": _stats(spread_tr), "skip_30pct": _stats(skip_tr)}
    mc = monte_carlo(full_tr, paths=800)
    v = verdict(full, oos, wf, torture, mc)
    tests = {
        "1_full": full,
        "2_oos70": oos,
        "3_wf_fold_1": wf_details.get("wf_fold_1") or {"n": 0, "exp_r": 0},
        "4_wf_fold_2": wf_details.get("wf_fold_2") or {"n": 0, "exp_r": 0},
        "5_wf_fold_3": wf_details.get("wf_fold_3") or {"n": 0, "exp_r": 0},
        "6_wf_fold_4": wf_details.get("wf_fold_4") or {"n": 0, "exp_r": 0},
        "7_torture_spread": torture["spread_0.15R"],
        "8_torture_skip": torture["skip_30pct"],
        "mc_prob_profit": mc.get("prob_profit"),
    }
    return {
        "full": full,
        "oos": oos,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": sum(1 for f in wf if f["oos"]["exp_r"] > 0),
        "wf_n": len(wf),
        "tests": tests,
        "entries": len(full_tr),
    }


def _rank_key(bat: dict) -> tuple:
    f = bat.get("full") or {}
    score = {"SURVIVES": 2, "FRAGILE": 1, "DEAD": 0}.get(bat.get("verdict"), 0)
    return (
        score,
        float(f.get("exp_r") or -9),
        int(bat.get("wf_green") or 0),
        float((bat.get("monte_carlo") or {}).get("prob_profit") or 0),
        float(f.get("pnl") or -1e9),
    )


def _cell_id(lb, mode, lot, tp, hold) -> str:
    tp_s = "c2c" if tp is None else f"tp{tp}"
    return f"lb{'-'.join(map(str, lb))}_{mode}_x{lot}_{tp_s}_h{hold}"


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    clear_hist_cache()

    def horizon_frame(daily: pd.DataFrame, days: Optional[int]) -> tuple[pd.DataFrame, int]:
        if days is None:
            return daily, 0
        end = daily.index[-1]
        cut = end - pd.Timedelta(days=int(days))
        warm = cut - pd.Timedelta(days=150)
        frame = daily[daily.index >= warm]
        cut_i = 0
        for i, ts in enumerate(frame.index):
            if ts >= cut:
                cut_i = i
                break
        return frame, cut_i

    exec_cells = list(itertools.product(SIZE_MODES, LOT_MULTS, TP_RS, HOLD_DAYS))
    total_est = len(BOOK) * len(LOOKBACK_SETS) * len(exec_cells) * len(HORIZONS)
    _log("=== Schwab tally DENSIFY v2 · wide lookback permutations · Nebula untouched ===")
    _log(
        f"lookback_sets={len(LOOKBACK_SETS)} · exec_cells={len(exec_cells)} · "
        f"cells/pair≈{len(LOOKBACK_SETS)*len(exec_cells)} · total≈{total_est} · "
        f"$/{RISK_USD}/R · SL={SL_ATR}ATR"
    )
    _log(f"pool={POOL} · started {datetime.now(timezone.utc).isoformat()}")
    t0 = time.time()

    results: dict[str, Any] = {
        "meta": {
            "version": 2,
            "gist": "combinatorial lookbacks 2/3/4/5-way + lot/hold/TP densify",
            "n_lookback_sets": len(LOOKBACK_SETS),
            "lookback_pool": list(POOL),
            "tp_rs": list(TP_RS),
            "lot_mults": list(LOT_MULTS),
            "size_modes": list(SIZE_MODES),
            "hold_days": list(HOLD_DAYS),
            "horizons": list(HORIZONS),
            "nebula": "untouched",
            "started": datetime.now(timezone.utc).isoformat(),
        },
        "symbols": {},
        "crowns_1y": [],
    }

    done = 0
    crowns: list[dict] = []
    max_lb = max(max(lb) for lb in LOOKBACK_SETS)

    for sym in BOOK:
        daily = _load_daily(sym)
        if daily is None:
            _log(f"{sym}: SKIP")
            results["symbols"][sym] = {"error": "no_daily"}
            continue
        _log(f"\n--- {sym} daily={len(daily)} {str(daily.index[0])[:10]}→{str(daily.index[-1])[:10]} ---")
        try:
            sp = float(spread_price(sym, hour_utc=12.0))
        except Exception:
            sp = 0.0
        # rough default spread in R using median ATR later per bar; use 0.05 floor
        spread_r_default = 0.05

        sym_out: dict[str, Any] = {"daily_bars": len(daily), "horizons": {}}

        for hname, days in HORIZONS.items():
            frame, cut_i = horizon_frame(daily, days)
            o = frame["open"].to_numpy(dtype=float)
            h = frame["high"].to_numpy(dtype=float)
            l = frame["low"].to_numpy(dtype=float)
            c = frame["close"].to_numpy(dtype=float)
            atr = _atr_arr(frame)
            votes = _vote_cache(c, max_lb)
            n_bars = len(frame)
            # refine spread_r from median atr
            med_a = float(np.nanmedian(atr[np.isfinite(atr)])) if np.isfinite(atr).any() else 1.0
            if med_a > 0 and sp > 0:
                spread_r_default = max(0.02, sp / med_a)

            best = None
            best_bat = None
            top5: list[dict] = []
            survives_n = 0
            t_sym = time.time()

            for lb in LOOKBACK_SETS:
                tally = _tally_from_cache(votes, lb)
                for mode, lot, tp, hold in exec_cells:
                    trades = _simulate_fast(
                        o=o,
                        h=h,
                        l=l,
                        c=c,
                        atr=atr,
                        tally=tally,
                        lookbacks=lb,
                        size_mode=mode,
                        lot_mult=lot,
                        tp_r=tp,
                        hold_days=hold,
                        spread_r_default=spread_r_default,
                    )
                    if cut_i > 0:
                        trades = [t for t in trades if int(t["i"]) >= cut_i]
                    bat = _battery(trades, n_bars)
                    done += 1
                    row = {
                        "id": _cell_id(lb, mode, lot, tp, hold),
                        "lookbacks": list(lb),
                        "size_mode": mode,
                        "lot_mult": lot,
                        "tp_r": tp,
                        "hold_days": hold,
                        "verdict": bat["verdict"],
                        "full": bat["full"],
                        "oos": bat["oos"],
                        "wf_green": bat["wf_green"],
                        "wf_n": bat["wf_n"],
                        "mc": bat["monte_carlo"],
                        "tests": bat["tests"],
                        "entries": bat["entries"],
                    }
                    if bat["verdict"] == "SURVIVES":
                        survives_n += 1
                    if best is None or _rank_key(bat) > _rank_key(best_bat or {}):
                        best, best_bat = row, bat
                    top5.append(row)
                    top5.sort(
                        key=lambda r: _rank_key(
                            {
                                "verdict": r["verdict"],
                                "full": r["full"],
                                "wf_green": r["wf_green"],
                                "monte_carlo": r["mc"],
                            }
                        ),
                        reverse=True,
                    )
                    top5 = top5[:8]

            assert best is not None
            f = best["full"]
            _log(
                f"  {hname:4} best={best['id'][:70]} → {best['verdict']} "
                f"n={f['n']} E[R]={f['exp_r']:+.3f} pnl=${f['pnl']:+.0f} "
                f"OOS={best['oos']['exp_r']:+.3f} WF={best['wf_green']}/{best['wf_n']} "
                f"MC={best['mc'].get('prob_profit')}% · SURVIVES={survives_n} "
                f"({time.time()-t_sym:.0f}s)"
            )
            # slim tests in stored best/top to keep JSON smaller
            def slim(r: dict) -> dict:
                out = {k: r[k] for k in r if k != "tests"}
                out["mc_prob"] = (r.get("mc") or {}).get("prob_profit")
                out["tests_exp_r"] = {
                    k: (r.get("tests") or {}).get(k, {}).get("exp_r")
                    if isinstance((r.get("tests") or {}).get(k), dict)
                    else (r.get("tests") or {}).get(k)
                    for k in (
                        "1_full",
                        "2_oos70",
                        "3_wf_fold_1",
                        "4_wf_fold_2",
                        "5_wf_fold_3",
                        "6_wf_fold_4",
                        "7_torture_spread",
                        "8_torture_skip",
                        "mc_prob_profit",
                    )
                }
                return out

            sym_out["horizons"][hname] = {
                "best": slim(best),
                "top8": [slim(x) for x in top5],
                "survives_cells": survives_n,
                "cells_tested": len(LOOKBACK_SETS) * len(exec_cells),
            }
            if hname == "1y":
                for r in top5:
                    if r["verdict"] == "SURVIVES":
                        crowns.append({"symbol": sym, **slim(r)})

            PROGRESS.write_text(
                json.dumps(
                    {
                        "done": done,
                        "total_est": total_est,
                        "symbol": sym,
                        "horizon": hname,
                        "elapsed_s": round(time.time() - t0, 1),
                        "pct": round(100.0 * done / max(1, total_est), 2),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        results["symbols"][sym] = sym_out
        PARTIAL.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    results["crowns_1y"] = sorted(
        crowns,
        key=lambda r: (
            {"SURVIVES": 2}.get(r["verdict"], 0),
            float((r.get("full") or {}).get("exp_r") or -9),
        ),
        reverse=True,
    )[:60]

    best_rows = []
    survives_pairs = 0
    fragile_pairs = 0
    for s in BOOK:
        so = results["symbols"].get(s) or {}
        b = ((so.get("horizons") or {}).get("1y") or {}).get("best") or {}
        if not b:
            continue
        if b.get("verdict") == "SURVIVES":
            survives_pairs += 1
        elif b.get("verdict") == "FRAGILE":
            fragile_pairs += 1
        best_rows.append(
            {
                "symbol": s,
                "verdict": b.get("verdict"),
                "exp_r": (b.get("full") or {}).get("exp_r"),
                "pnl": (b.get("full") or {}).get("pnl"),
                "lookbacks": b.get("lookbacks"),
                "lot": b.get("lot_mult"),
                "tp_r": b.get("tp_r"),
                "hold": b.get("hold_days"),
                "mode": b.get("size_mode"),
                "wf": f"{b.get('wf_green')}/{b.get('wf_n')}",
                "mc": b.get("mc_prob"),
                "id": b.get("id"),
            }
        )

    # vs v1: did we find better GOLD/BTC?
    v1_path = Path(ROOT) / "data" / "lab_schwab_tally_densify.json"
    vs_v1 = {}
    if v1_path.exists():
        try:
            v1 = json.loads(v1_path.read_text())
            for s in BOOK:
                v1b = (
                    ((v1.get("symbols") or {}).get(s) or {})
                    .get("horizons", {})
                    .get("1y", {})
                    .get("best")
                    or {}
                )
                v2b = next((x for x in best_rows if x["symbol"] == s), {})
                vs_v1[s] = {
                    "v1_exp_r": (v1b.get("full") or {}).get("exp_r"),
                    "v2_exp_r": v2b.get("exp_r"),
                    "delta": round(
                        float(v2b.get("exp_r") or 0) - float((v1b.get("full") or {}).get("exp_r") or 0),
                        4,
                    ),
                    "v2_lb": v2b.get("lookbacks"),
                }
        except Exception:
            vs_v1 = {}

    if survives_pairs >= 5:
        rec, note = "RESEARCH_MORE", f"{survives_pairs}/8 SURVIVES on 1y after wide permute — keep going (WF lock next)."
    elif survives_pairs >= 2:
        rec, note = "NARROW", f"{survives_pairs}/8 SURVIVES — sleeve candidates only."
    else:
        rec, note = "SKIP", "Wide permute did not hold — hope faded."

    results["summary"] = {
        "pairs_best_1y_SURVIVES": survives_pairs,
        "pairs_best_1y_FRAGILE": fragile_pairs,
        "recommendation": rec,
        "note": note,
        "best_per_pair_1y": best_rows,
        "vs_v1_exp_r": vs_v1,
        "n_lookback_sets": len(LOOKBACK_SETS),
        "elapsed_s": round(time.time() - t0, 1),
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    results["meta"]["finished"] = results["summary"]["finished"]

    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_md(results)
    _log(f"\n=== SUMMARY === {json.dumps(results['summary'], default=str)[:1200]}")
    _log(f"wrote {OUT}")
    _log(f"wrote {MD}")


def _write_md(results: dict[str, Any]) -> None:
    lines = [
        "# Schwab tally densify v2 — wide lookback permutations",
        "",
        f"- **{results['summary']['recommendation']}**: {results['summary']['note']}",
        f"- Lookback sets: **{results['summary']['n_lookback_sets']}** (2/3/4/5-way combos)",
        f"- Nebula untouched · Dukascopy · ${RISK_USD}/R",
        "",
        "## Best per pair (1y)",
        "",
        "| Pair | Verdict | E[R] | PnL$ | Lot | Hold | TP | Mode | Lookbacks | WF | MC |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for b in results["summary"]["best_per_pair_1y"]:
        lines.append(
            f"| {b['symbol']} | {b['verdict']} | {b['exp_r']} | {b['pnl']} | {b['lot']} | "
            f"{b['hold']} | {b['tp_r']} | {b['mode']} | `{b['lookbacks']}` | {b['wf']} | {b['mc']} |"
        )
    lines += ["", "## vs v1 (1y E[R] delta)", ""]
    for s, d in (results["summary"].get("vs_v1_exp_r") or {}).items():
        lines.append(
            f"- **{s}**: v1={d.get('v1_exp_r')} → v2={d.get('v2_exp_r')} (Δ {d.get('delta')}) lb=`{d.get('v2_lb')}`"
        )
    lines += ["", "## Top crowns", ""]
    for c in (results.get("crowns_1y") or [])[:25]:
        f = c.get("full") or {}
        lines.append(
            f"- **{c['symbol']}** E[R]={f.get('exp_r')} pnl=${f.get('pnl')} "
            f"lb=`{c.get('lookbacks')}` lot={c.get('lot_mult')} hold={c.get('hold_days')} tp={c.get('tp_r')}"
        )
    MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
