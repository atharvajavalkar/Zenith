#!/usr/bin/env python3
"""RESEARCH ONLY — Schwab tally densify: lookback permutations × size × small-TP.

Gist: multi-horizon close votes, then try
  • different lookback combinations (permute/combine)
  • larger lot multipliers
  • smaller take-profits (ATR multiples) with ATR stop
  • short max-hold (1–3 days)

Dukascopy daily (resampled). Nebula / live / .env untouched.
Primary score: 1y battery; confirm on 3y + full for crowns.
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

# Curated lookback sets (trading days) — permute/combine family
LOOKBACK_SETS: list[tuple[int, ...]] = [
    (7, 14, 21, 42),  # baseline
    (5, 10, 20, 40),
    (3, 7, 14, 28),
    (10, 20, 40, 60),
    (7, 14, 28, 56),
    (5, 15, 30, 60),
    (14, 21, 42, 63),
    (7, 21, 42, 84),
    (10, 21, 42, 84),
    (3, 5, 10, 20),
    (5, 10, 15, 30),
    (7, 14, 21),
    (14, 28, 56),
    (21, 42, 63),
    (5, 10, 20),
    (7, 14, 42),
    (10, 20, 40),
    (14, 42),
    (7, 21),
    (5, 20, 60),
    (3, 14, 42),
    (10, 30, 60),
    (7, 14, 30, 60),
    (4, 8, 16, 32),
]

TP_RS = (0.4, 0.6, 0.8, 1.0, None)  # None = close-to-close overnight (no fixed TP)
LOT_MULTS = (1.0, 1.5, 2.0)
SIZE_MODES = ("strict", "linear")  # strict: ±max→1 ±half→0.5; linear: |t|/n
HOLD_DAYS = (1, 2, 3)
SL_ATR = 1.0

HORIZONS = {"1y": 365, "3y": 1095, "full": None}

OUT = Path(ROOT) / "data" / "lab_schwab_tally_densify.json"
LOG = Path(ROOT) / "data" / "lab_schwab_tally_densify.log"
MD = Path(ROOT) / "data" / "lab_schwab_tally_densify.md"
PROGRESS = Path(ROOT) / "data" / "lab_schwab_tally_densify.progress.json"


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
    s = pd.Series(tr).rolling(n, min_periods=max(5, n // 2)).mean().to_numpy(dtype=float)
    return s


def _tally(c: np.ndarray, lookbacks: tuple[int, ...]) -> np.ndarray:
    votes = []
    for lb in lookbacks:
        prev = np.roll(c, lb)
        prev[:lb] = np.nan
        v = np.where(c > prev, 1, np.where(c < prev, -1, 0)).astype(float)
        v[:lb] = np.nan
        votes.append(v)
    return np.nansum(np.vstack(votes), axis=0)


def _base_size(tally: float, n_votes: int, mode: str) -> float:
    if not np.isfinite(tally) or tally == 0:
        return 0.0
    a = abs(int(tally))
    if mode == "linear":
        return min(1.0, a / float(max(1, n_votes)))
    # strict: unanimous → full; half-strength → half; else sit
    if a == n_votes:
        return 1.0
    if n_votes >= 4 and a == 2:
        return 0.5
    if n_votes == 3 and a == 1:
        return 0.5
    if n_votes == 2 and a == 2:
        return 1.0
    return 0.0


def _simulate(
    df: pd.DataFrame,
    symbol: str,
    *,
    lookbacks: tuple[int, ...],
    size_mode: str,
    lot_mult: float,
    tp_r: Optional[float],
    hold_days: int,
    start_i: int = 0,
) -> list[dict]:
    """Signal at close i → enter next open i+1; TP/SL in ATR·R; else exit by hold."""
    c = df["close"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    atr = _atr_arr(df)
    tally = _tally(c, lookbacks)
    n_votes = len(lookbacks)
    n = len(df)
    need = max(lookbacks) + 3
    trades: list[dict] = []
    i = max(need, start_i)
    busy_until = -1

    while i < n - 1:
        if i < busy_until:
            i += 1
            continue
        a = float(atr[i])
        if not np.isfinite(a) or a <= 0 or not np.isfinite(tally[i]):
            i += 1
            continue
        base = _base_size(float(tally[i]), n_votes, size_mode)
        if base <= 0:
            i += 1
            continue
        side = 1 if tally[i] > 0 else -1
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

        # spread cost in R at entry
        try:
            hour = float(df.index[i].hour) if hasattr(df.index[i], "hour") else 12.0
            sp = float(spread_price(symbol, hour_utc=hour))
            spread_r = sp / sl_dist
        except Exception:
            spread_r = 0.05

        exit_px = None
        exit_i = entry_i
        reason = "time"
        max_j = min(n - 1, entry_i + hold_days - 1)

        if tp_r is None:
            # overnight / hold close-to-close for hold_days
            exit_i = max_j
            exit_px = float(c[exit_i])
            reason = "c2c"
        else:
            hit = False
            for j in range(entry_i, max_j + 1):
                if side > 0:
                    # SL first if both (conservative)
                    if float(l[j]) <= sl:
                        exit_px = sl
                        exit_i = j
                        reason = "sl"
                        hit = True
                        break
                    if tp is not None and float(h[j]) >= tp:
                        exit_px = tp
                        exit_i = j
                        reason = "tp"
                        hit = True
                        break
                else:
                    if float(h[j]) >= sl:
                        exit_px = sl
                        exit_i = j
                        reason = "sl"
                        hit = True
                        break
                    if tp is not None and float(l[j]) <= tp:
                        exit_px = tp
                        exit_i = j
                        reason = "tp"
                        hit = True
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
                "t": str(df.index[i])[:10],
                "tally": int(tally[i]),
                "size": round(size, 3),
                "side": side,
                "r": round(float(r), 4),
                "pnl": round(float(r) * RISK_USD, 2),
                "reason": reason,
                "tp_r": tp_r,
                "hold": hold_days,
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
            "torture": {},
            "tests": {},
            "entries": 0,
        }

    def by_range(start_i: int = 0, end_i: Optional[int] = None) -> list[dict]:
        end_i = n_bars - 1 if end_i is None else end_i
        return [t for t in full_tr if start_i <= int(t["i"]) <= end_i]

    full = _stats(full_tr)
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
    mc = monte_carlo(full_tr)
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
        "torture": {k: {"n": x["n"], "pnl": x["pnl"], "exp_r": x["exp_r"]} for k, x in torture.items()},
        "tests": tests,
        "entries": len(full_tr),
    }


def _horizon_frame(daily: pd.DataFrame, days: Optional[int]) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    if days is None:
        return daily, None
    end = daily.index[-1]
    cut = end - pd.Timedelta(days=int(days))
    warm = cut - pd.Timedelta(days=120)
    return daily[daily.index >= warm], pd.Timestamp(cut).tz_convert("UTC") if cut.tzinfo else pd.Timestamp(cut, tz="UTC")


def _filter_trades(trades: list[dict], cut: Optional[pd.Timestamp]) -> list[dict]:
    if cut is None:
        return trades
    out = []
    for t in trades:
        tt = pd.Timestamp(t["t"])
        if tt.tzinfo is None:
            tt = tt.tz_localize("UTC")
        if tt >= cut:
            out.append(t)
    return out


def _cell_id(lb, mode, lot, tp, hold) -> str:
    tp_s = "c2c" if tp is None else f"tp{tp}"
    return f"lb{'-'.join(map(str, lb))}_{mode}_x{lot}_{tp_s}_h{hold}"


def _rank_key(bat: dict) -> tuple:
    f = bat.get("full") or {}
    v = bat.get("verdict")
    score = {"SURVIVES": 2, "FRAGILE": 1, "DEAD": 0}.get(v, 0)
    return (
        score,
        float(f.get("exp_r") or -9),
        int(bat.get("wf_green") or 0),
        float((bat.get("monte_carlo") or {}).get("prob_profit") or 0),
        float(f.get("pnl") or -1e9),
    )


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    clear_hist_cache()
    cells = list(
        itertools.product(LOOKBACK_SETS, SIZE_MODES, LOT_MULTS, TP_RS, HOLD_DAYS)
    )
    # c2c only needs hold variation once meaningfully — still OK
    total = len(BOOK) * len(cells) * len(HORIZONS)
    _log("=== Schwab tally DENSIFY · Dukascopy · Nebula untouched ===")
    _log(
        f"lookback_sets={len(LOOKBACK_SETS)} · cells/pair={len(cells)} · "
        f"horizons={list(HORIZONS)} · total≈{total} · $/{RISK_USD}/R · SL={SL_ATR}ATR"
    )
    _log(f"started {datetime.now(timezone.utc).isoformat()}")
    t0 = time.time()

    results: dict[str, Any] = {
        "meta": {
            "gist": "permute lookbacks + bigger lot + smaller TP",
            "lookback_sets": [list(x) for x in LOOKBACK_SETS],
            "tp_rs": list(TP_RS),
            "lot_mults": list(LOT_MULTS),
            "size_modes": list(SIZE_MODES),
            "hold_days": list(HOLD_DAYS),
            "sl_atr": SL_ATR,
            "risk_usd": RISK_USD,
            "nebula": "untouched",
            "started": datetime.now(timezone.utc).isoformat(),
        },
        "symbols": {},
        "crowns_1y": [],
    }

    done = 0
    crowns: list[dict] = []

    for sym in BOOK:
        daily = _load_daily(sym)
        if daily is None:
            _log(f"{sym}: SKIP no daily")
            results["symbols"][sym] = {"error": "no_daily"}
            continue
        _log(f"\n--- {sym} daily={len(daily)} {str(daily.index[0])[:10]}→{str(daily.index[-1])[:10]} ---")
        sym_best: dict[str, Any] = {}
        sym_crowns = []

        for hname, days in HORIZONS.items():
            frame, cut = _horizon_frame(daily, days)
            n_bars = len(frame)
            best = None
            best_bat = None
            top5: list[dict] = []
            survives_n = 0

            for lb, mode, lot, tp, hold in cells:
                # skip absurd: c2c with hold>1 still ok; tp with hold
                trades = _simulate(
                    frame,
                    sym,
                    lookbacks=lb,
                    size_mode=mode,
                    lot_mult=lot,
                    tp_r=tp,
                    hold_days=hold,
                )
                trades = _filter_trades(trades, cut)
                bat = _battery(trades, n_bars)
                done += 1
                cid = _cell_id(lb, mode, lot, tp, hold)
                row = {
                    "id": cid,
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
                top5.sort(key=lambda r: _rank_key({"verdict": r["verdict"], "full": r["full"], "wf_green": r["wf_green"], "monte_carlo": r["mc"]}), reverse=True)
                top5 = top5[:5]

            assert best is not None
            f = best["full"]
            _log(
                f"  {hname:4} best={best['id'][:60]}… → {best['verdict']} "
                f"n={f['n']} E[R]={f['exp_r']:+.3f} pnl=${f['pnl']:+.0f} "
                f"OOS={best['oos']['exp_r']:+.3f} WF={best['wf_green']}/{best['wf_n']} "
                f"MC={best['mc'].get('prob_profit')}% · SURVIVES_cells={survives_n}"
            )
            sym_best[hname] = {
                "best": best,
                "top5": top5,
                "survives_cells": survives_n,
                "cells_tested": len(cells),
            }
            if hname == "1y":
                for r in top5:
                    if r["verdict"] == "SURVIVES":
                        crowns.append({"symbol": sym, **{k: r[k] for k in r if k != "tests"}})
                        sym_crowns.append(r)

            if done % 500 == 0:
                PROGRESS.write_text(
                    json.dumps(
                        {"done": done, "total_est": total, "elapsed_s": round(time.time() - t0, 1)},
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        results["symbols"][sym] = {
            "daily_bars": len(daily),
            "horizons": sym_best,
            "crowns_1y": [
                {k: c[k] for k in ("id", "lookbacks", "size_mode", "lot_mult", "tp_r", "hold_days", "verdict", "full", "oos", "wf_green", "mc")}
                for c in sym_crowns[:10]
            ],
        }

    # Book summary
    results["crowns_1y"] = sorted(
        crowns,
        key=lambda r: (
            {"SURVIVES": 2, "FRAGILE": 1}.get(r["verdict"], 0),
            float((r.get("full") or {}).get("exp_r") or -9),
        ),
        reverse=True,
    )[:40]

    survives_pairs = sum(
        1
        for s, so in results["symbols"].items()
        if ((so.get("horizons") or {}).get("1y") or {}).get("best", {}).get("verdict") == "SURVIVES"
    )
    fragile_pairs = sum(
        1
        for s, so in results["symbols"].items()
        if ((so.get("horizons") or {}).get("1y") or {}).get("best", {}).get("verdict") == "FRAGILE"
    )
    # Did small-TP + big-lot help vs baseline?
    baseline_beats = []
    for s, so in results["symbols"].items():
        b = ((so.get("horizons") or {}).get("1y") or {}).get("best") or {}
        if not b:
            continue
        baseline_beats.append(
            {
                "symbol": s,
                "best_id": b.get("id"),
                "verdict": b.get("verdict"),
                "exp_r": (b.get("full") or {}).get("exp_r"),
                "tp_r": b.get("tp_r"),
                "lot": b.get("lot_mult"),
                "lookbacks": b.get("lookbacks"),
                "mode": b.get("size_mode"),
            }
        )

    rec = "SKIP"
    note = "Densify did not lift the book — still not a desk candidate."
    if survives_pairs >= 3:
        rec = "RESEARCH_MORE"
        note = f"{survives_pairs}/8 pairs have a SURVIVES densify crown on 1y — worth GOLD/crypto sleeve work."
    elif survives_pairs >= 1:
        rec = "NARROW"
        note = f"Only {survives_pairs} pair(s) SURVIVES after densify — sleeve-only, not book-wide."

    results["summary"] = {
        "pairs_best_1y_SURVIVES": survives_pairs,
        "pairs_best_1y_FRAGILE": fragile_pairs,
        "crowns_1y_n": len(results["crowns_1y"]),
        "recommendation": rec,
        "note": note,
        "best_per_pair_1y": baseline_beats,
        "elapsed_s": round(time.time() - t0, 1),
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    results["meta"]["finished"] = results["summary"]["finished"]

    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_md(results)
    _log(f"\n=== SUMMARY === {results['summary']}")
    _log(f"wrote {OUT}")
    _log(f"wrote {MD}")


def _write_md(results: dict[str, Any]) -> None:
    lines = [
        "# Schwab tally densify — lookbacks × lot × small-TP",
        "",
        f"- **{results['summary']['recommendation']}**: {results['summary']['note']}",
        f"- Grid: {len(LOOKBACK_SETS)} lookback sets × modes × lots × TP × holds",
        f"- Nebula/live untouched · Dukascopy daily · ${RISK_USD}/R · SL={SL_ATR}ATR",
        "",
        "## Best cell per pair (1y)",
        "",
        "| Pair | Verdict | E[R] | PnL$ | TP | Lot | Mode | Lookbacks | WF | MC% |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in results["summary"]["best_per_pair_1y"]:
        so = results["symbols"].get(row["symbol"], {})
        best = ((so.get("horizons") or {}).get("1y") or {}).get("best") or {}
        f = best.get("full") or {}
        lines.append(
            f"| {row['symbol']} | {best.get('verdict')} | {f.get('exp_r')} | {f.get('pnl')} | "
            f"{best.get('tp_r')} | {best.get('lot_mult')} | {best.get('size_mode')} | "
            f"`{best.get('lookbacks')}` | {best.get('wf_green')}/{best.get('wf_n')} | "
            f"{(best.get('mc') or {}).get('prob_profit')} |"
        )
    lines += ["", "## Top crowns (1y SURVIVES)", ""]
    for c in results.get("crowns_1y") or []:
        if c.get("verdict") != "SURVIVES":
            continue
        f = c.get("full") or {}
        lines.append(
            f"- **{c['symbol']}** `{c['id']}` E[R]={f.get('exp_r')} pnl=${f.get('pnl')} "
            f"tp={c.get('tp_r')} lot={c.get('lot_mult')} lb={c.get('lookbacks')}"
        )
    lines += ["", "## 3y / full bests", ""]
    for sym in BOOK:
        so = results["symbols"].get(sym) or {}
        for hn in ("3y", "full"):
            b = ((so.get("horizons") or {}).get(hn) or {}).get("best") or {}
            if not b:
                continue
            f = b.get("full") or {}
            lines.append(
                f"- {sym} **{hn}**: {b.get('verdict')} E[R]={f.get('exp_r')} "
                f"tp={b.get('tp_r')} lot={b.get('lot_mult')} lb={b.get('lookbacks')}"
            )
    MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
