#!/usr/bin/env python3
"""CURRENT live brains × Dukascopy multi-year intraday horizons.

Uses local data/hist/ (M1→resampled). Runs each pair on its SYMBOL_SCAN_TFS
native TF, plus a desk-wide 5m and 15m stress for apples-to-apples.

Horizons: 1,2,3,5,7,10y (hist depth ≈10y — no honest 15y yet).
$200/R. READ-ONLY — no live brain changes.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis._edge_survive_bt import (  # noqa: E402
    BOOK,
    RISK_USD,
    _prep,
    _stats,
    monte_carlo,
    verdict,
)
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_breakout,
    _collect_ema,
    _collect_sma,
)
from analysis.strategy_lab import (  # noqa: E402
    _collect_cta,
    _collect_mom_roc,
    _collect_rsi,
    _collect_supertrend,
)
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402
from config import (  # noqa: E402
    SYMBOL_ENTRY_BRAIN,
    symbol_entry_brain,
    symbol_rr,
    symbol_scan_tfs,
)

OUT = Path(ROOT) / "data" / "duka_brains_horizons.json"
LOG = Path(ROOT) / "data" / "duka_brains_horizons.log"
# Optional override: DUKA_BT_OUT / DUKA_BT_LOG
if os.getenv("DUKA_BT_OUT"):
    OUT = Path(os.environ["DUKA_BT_OUT"])
if os.getenv("DUKA_BT_LOG"):
    LOG = Path(os.environ["DUKA_BT_LOG"])
HORIZONS_Y = [1, 2, 3, 5, 7, 10]
if os.getenv("DUKA_HORIZONS"):
    HORIZONS_Y = [int(x.strip()) for x in os.environ["DUKA_HORIZONS"].split(",") if x.strip()]
# 15m first (honest multi-year, tractable). 5m second.
COMMON_TFS = ("15m", "5m")
if os.getenv("DUKA_COMMON_TFS"):
    COMMON_TFS = tuple(
        x.strip() for x in os.environ["DUKA_COMMON_TFS"].split(",") if x.strip()
    )
# DUKA_NATIVE=1 → also (or only) run each pair on its live SYMBOL_SCAN_TFS
NATIVE_ONLY = os.getenv("DUKA_NATIVE_ONLY", "").strip() in ("1", "true", "yes")
RUN_NATIVE = NATIVE_ONLY or os.getenv("DUKA_NATIVE", "").strip() in ("1", "true", "yes")


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _simulate_fast(
    df: pd.DataFrame,
    entries: list,
    rr: float,
    *,
    start_i: int = 0,
    end_i: Optional[int] = None,
) -> list[dict]:
    """Same semantics as _rr_best_by_pair.simulate, numpy path (multi-year safe)."""
    end_i = end_i if end_i is not None else len(df) - 1
    high = df["high"].to_numpy(dtype=float, copy=False)
    low = df["low"].to_numpy(dtype=float, copy=False)
    n = len(df)
    trades: list[dict] = []
    busy_until = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < start_i or ei > end_i or ei < busy_until:
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
        j_end = min(end_i, n - 1)
        if side == "BUY":
            for j in range(ei + 1, j_end + 1):
                hit_sl = low[j] <= sl
                hit_tp = high[j] >= tp
                if hit_sl or hit_tp:
                    exit_r = -1.0 if hit_sl else float(rr)
                    if hit_sl and hit_tp:
                        exit_r = -1.0
                    exit_i = j
                    break
        else:
            for j in range(ei + 1, j_end + 1):
                hit_sl = high[j] >= sl
                hit_tp = low[j] <= tp
                if hit_sl or hit_tp:
                    exit_r = -1.0 if hit_sl else float(rr)
                    if hit_sl and hit_tp:
                        exit_r = -1.0
                    exit_i = j
                    break
        if exit_r is None:
            continue
        trades.append(
            {
                "i": ei,
                "entry_t": str(e["entry_t"])[:16],
                "side": side,
                "r": round(exit_r, 4),
                "pnl": round(exit_r * RISK_USD, 2),
            }
        )
        busy_until = exit_i + 2
    return trades


def _collect_fib_fast(df: pd.DataFrame, fibs=(0.50, 0.618)) -> list[dict]:
    """Faster fib collector for multi-year frames (numpy + stride on huge DFS)."""
    from analysis._edge_survive_bt import SWING, _fib_zone, _impulse

    n = len(df)
    # Stride impulse cache — _impulse is expensive; multi-year needs coarse step
    step = 1
    if n > 250_000:
        step = 8
    elif n > 120_000:
        step = 5
    elif n > 60_000:
        step = 3
    elif n > 30_000:
        step = 2

    cache: list = [None] * n
    for i in range(40, n, step):
        atr = float(df.iloc[i]["atr"] or 1)
        cache[i] = _impulse(df, i, atr)
    # forward-fill cache for skipped bars
    last = None
    for i in range(40, n):
        if cache[i] is not None:
            last = cache[i]
        elif last is not None:
            cache[i] = last

    entries: list[dict] = []
    for ei in range(40, n - 1):
        ts = df.index[ei]
        row = df.iloc[ei]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        atr = float(row["atr"] or 0) or max(h - l, 1e-6)
        imp = cache[ei]
        if not imp or (ts - imp["end_t"]).total_seconds() > 48 * 3600:
            continue
        zlo, zhi = _fib_zone(imp, atr, fibs)
        win = df.iloc[max(0, ei - 20) : ei]
        signal = None
        if imp["side"] == "BUY":
            liq = float(win["low"].min()) if len(win) else l
            if l < min(liq, zlo) and c > zlo and l <= zhi and c > float(row["open"]):
                signal = "BUY"
                entry, sl = c, min(l, liq) - 0.08 * atr
        else:
            liq = float(win["high"].max()) if len(win) else h
            if h > max(liq, zhi) and c < zhi and h >= zlo and c < float(row["open"]):
                signal = "SELL"
                entry, sl = c, max(h, liq) + 0.08 * atr
        if not signal:
            continue
        risk = abs(entry - sl)
        if risk <= 0.12 * atr:
            continue
        entries.append(
            {
                "ei": ei,
                "side": signal,
                "entry": float(entry),
                "sl": float(sl),
                "entry_t": df.index[ei],
            }
        )
    return entries


def _collector(brain: str) -> Callable:
    b = brain.lower().strip()
    if b in ("fib_liq_sweep", "fib_liq"):
        return _collect_fib_fast
    if b in ("ema_20_50_tight", "ema_tight"):
        # ema collect is also O(n²) via growing slices — stride on huge DFS
        def _ema_tight(d: pd.DataFrame):
            if len(d) > 250_000:
                # evaluate on 15m-equivalent density: every 3rd 5m bar
                sub = d.iloc[::3].copy()
                ents = _collect_ema(sub, pull_atr=0.75, disp_atr=0.70)
                # map ei back roughly
                for e in ents:
                    e["ei"] = int(e["ei"]) * 3
                return [e for e in ents if e["ei"] < len(d) - 1]
            return _collect_ema(d, pull_atr=0.75, disp_atr=0.70)

        return _ema_tight

    if b in ("ema_20_50_pull", "ema_pull"):
        def _ema_pull(d: pd.DataFrame):
            if len(d) > 250_000:
                sub = d.iloc[::3].copy()
                ents = _collect_ema(sub, pull_atr=1.0, disp_atr=0.55)
                for e in ents:
                    e["ei"] = int(e["ei"]) * 3
                return [e for e in ents if e["ei"] < len(d) - 1]
            return _collect_ema(d, pull_atr=1.0, disp_atr=0.55)

        return _ema_pull
    if b == "sma_20_50_cross":
        return _collect_sma
    if b == "cta_10_40":
        return lambda d: _collect_cta(d, 10, 40)
    if b in ("cta_50_200", "cta_50/200"):
        return lambda d: _collect_cta(d, 50, 200)
    if b in ("don_100", "donchian_100"):
        return lambda d: _collect_breakout(d, 100)
    if b in ("roc_120", "roc120"):
        return lambda d: _collect_mom_roc(d, 120)
    if b in ("rsi", "rsi_mr"):
        return _collect_rsi
    if b in ("bb_1.5", "bb_1_5"):
        return lambda d: _bb_std(d, 1.5)
    if b in ("bb_2", "bb_2.0"):
        return lambda d: _bb_std(d, 2.0)
    if b in ("bb_2.5", "bb_2_5", "bb25"):
        return lambda d: _bb_std(d, 2.5)
    if b in ("st_3", "supertrend_3"):
        return lambda d: _collect_supertrend(d, 3.0)
    raise KeyError(brain)


def _battery(df: pd.DataFrame, entries: list, rr: float) -> dict[str, Any]:
    n = len(df)
    if n < 200 or not entries:
        return {
            "bars": n,
            "entries": len(entries) if entries else 0,
            "full": {"n": 0, "pnl": 0.0, "exp_r": 0.0, "win_rate": 0.0},
            "oos": {"n": 0, "pnl": 0.0, "exp_r": 0.0, "win_rate": 0.0},
            "monte_carlo": {},
            "verdict": "DEAD",
            "wf_green": 0,
            "wf_n": 0,
            "note": "too_few_bars_or_entries",
        }

    def gen(start_i=0, end_i=None, **kw):
        spread_r = float(kw.get("spread_r") or 0)
        skip_p = float(kw.get("skip_p") or 0)
        rng = kw.get("rng") or random.Random(0)
        tr = _simulate_fast(df, entries, rr, start_i=start_i, end_i=end_i)
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
    }
    mc = monte_carlo(full_tr)
    v = verdict(full, oos, wf, torture, mc)
    return {
        "bars": n,
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "entries": len(entries),
        "full": full,
        "oos": oos,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": sum(1 for f in wf if f["oos"]["exp_r"] > 0),
        "wf_n": len(wf),
        "torture": {k: {"n": v2["n"], "pnl": v2["pnl"], "exp_r": v2["exp_r"]} for k, v2 in torture.items()},
    }


def _load_tf(sym: str, tf: str) -> Optional[pd.DataFrame]:
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    if not has_local_hist(sym, tf):
        return None
    # Avoid holding every TF of every symbol forever
    clear_hist_cache()
    df = load_local_ohlcv(sym, tf=tf)
    if df is None or df.empty:
        return None
    need = [c for c in ("open", "high", "low", "close") if c in df.columns]
    return _prep(df[need].dropna())


def _desk_fit(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("full")]
    s = sum(1 for r in rows if r.get("verdict") == "SURVIVES")
    f = sum(1 for r in rows if r.get("verdict") == "FRAGILE")
    d = sum(1 for r in rows if r.get("verdict") == "DEAD")
    full_pnl = sum(float(r["full"]["pnl"]) for r in ok)
    oos_pnl = sum(float(r["oos"]["pnl"]) for r in ok)
    mean_oos_er = float(np.mean([r["oos"]["exp_r"] for r in ok])) if ok else None
    fit = s * 1.0 + f * 0.35 - d * 0.5
    if oos_pnl > 0:
        fit += 1.0
    if full_pnl > 0:
        fit += 0.5
    fit_score = float(np.clip(round(1 + fit, 1), 1.0, 10.0))
    return {
        "survives": s,
        "fragile": f,
        "dead": d,
        "full_pnl": round(full_pnl, 2),
        "oos_pnl": round(oos_pnl, 2),
        "mean_oos_exp_r": round(mean_oos_er, 4) if mean_oos_er is not None else None,
        "fit_score_1_to_10": fit_score,
        "fits": fit_score >= 5.0 and oos_pnl > 0 and s >= 3,
    }


def _run_horizon(
    frames: dict[str, pd.DataFrame],
    brains: dict[str, str],
    years: int,
    end: datetime,
    tf_label: str,
) -> dict[str, Any]:
    start = end - timedelta(days=int(years * 365.25))
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    rows: list[dict] = []
    for sym in BOOK:
        brain = brains[sym]
        rr = float(symbol_rr(sym, default=2.0))
        df_all = frames.get(sym)
        if df_all is None or len(df_all) < 200:
            rows.append({"symbol": sym, "brain": brain, "error": "no_data", "verdict": "DEAD"})
            continue
        df = df_all[df_all.index >= start_ts]
        if len(df) < 200:
            rows.append(
                {
                    "symbol": sym,
                    "brain": brain,
                    "error": f"only {len(df)} bars in window",
                    "verdict": "DEAD",
                }
            )
            continue
        try:
            entries = _collector(brain)(df)
            bat = _battery(df, entries, rr)
        except Exception as exc:
            rows.append({"symbol": sym, "brain": brain, "error": str(exc), "verdict": "DEAD"})
            continue
        rows.append({"symbol": sym, "brain": brain, "rr": rr, "tf": tf_label, **bat})
        _log(
            f"  {sym:8} {brain:18} {tf_label:4} n={bat['full']['n']:5} "
            f"PnL=${bat['full']['pnl']:12,.0f} OOS=${bat['oos']['pnl']:11,.0f} "
            f"E[R]={bat['oos']['exp_r']:+.3f} {bat['verdict']}"
        )
    desk = _desk_fit(rows)
    _log(
        f"  → Desk {desk['survives']}S/{desk['fragile']}F/{desk['dead']}D · "
        f"Full ${desk['full_pnl']:,.0f} · OOS ${desk['oos_pnl']:,.0f} · "
        f"fit {desk['fit_score_1_to_10']}/10 · "
        f"{'FITS' if desk['fits'] else 'DOES NOT FIT'}"
    )
    return {
        "years": years,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "tf": tf_label,
        "results": rows,
        "desk": desk,
    }


def main() -> int:
    t0 = time.time()
    lock = Path(ROOT) / "data" / "duka_brains_horizons.lock"
    if lock.exists():
        try:
            old = int(lock.read_text().strip() or "0")
            if old and Path(f"/proc/{old}").exists():
                print(f"another duka battery running (pid {old}) — exit", flush=True)
                return 2
        except Exception:
            pass
    lock.write_text(str(os.getpid()))
    if LOG.exists():
        LOG.write_text("")
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
        days=1
    )
    brains = {s: symbol_entry_brain(s) for s in BOOK}
    scan_tfs: dict[str, str] = {}
    for s in BOOK:
        tfs = symbol_scan_tfs(s)
        scan_tfs[s] = tfs[0] if tfs else "5m"

    _log(
        f"DUKA BRAINS horizons · Dukascopy 5m/15m · ${RISK_USD}/R\n"
        f"Brains: {brains}\n"
        f"Live scan TFs (ref): {scan_tfs}\n"
        f"Horizons: {HORIZONS_Y}y (hist ≈10y; no honest 15y yet)\n"
    )

    # Desk-wide common TFs (one symbol at a time to keep RAM flat)
    common: dict[str, dict[str, Any]] = {}
    tfs_to_run = () if NATIVE_ONLY else COMMON_TFS
    for tf in tfs_to_run:
        by_h: dict[str, Any] = {
            str(y): {
                "years": y,
                "start": (end - timedelta(days=int(y * 365.25))).date().isoformat(),
                "end": end.date().isoformat(),
                "tf": tf,
                "results": [],
                "desk": {},
            }
            for y in HORIZONS_Y
        }
        _log(f"\n### TF {tf}")
        for sym in BOOK:
            brain = brains[sym]
            rr = float(symbol_rr(sym, default=2.0))
            try:
                df_all = _load_tf(sym, tf)
                if df_all is None or len(df_all) < 200:
                    raise RuntimeError("empty")
                _log(
                    f"  load {sym} {tf}: {len(df_all):,} bars "
                    f"{df_all.index[0].date()}→{df_all.index[-1].date()}"
                )
            except Exception as exc:
                _log(f"  load {sym} {tf} FAIL: {exc}")
                for years in HORIZONS_Y:
                    by_h[str(years)]["results"].append(
                        {
                            "symbol": sym,
                            "brain": brain,
                            "tf": tf,
                            "error": str(exc),
                            "verdict": "DEAD",
                        }
                    )
                continue

            for years in HORIZONS_Y:
                start_ts = pd.Timestamp(end - timedelta(days=int(years * 365.25)))
                if start_ts.tzinfo is None:
                    start_ts = start_ts.tz_localize("UTC")
                else:
                    start_ts = start_ts.tz_convert("UTC")
                df = df_all[df_all.index >= start_ts]
                if len(df) < 200:
                    by_h[str(years)]["results"].append(
                        {
                            "symbol": sym,
                            "brain": brain,
                            "tf": tf,
                            "error": f"only {len(df)} bars",
                            "verdict": "DEAD",
                        }
                    )
                    continue
                try:
                    entries = _collector(brain)(df)
                    bat = _battery(df, entries, rr)
                except Exception as exc:
                    by_h[str(years)]["results"].append(
                        {
                            "symbol": sym,
                            "brain": brain,
                            "tf": tf,
                            "error": str(exc),
                            "verdict": "DEAD",
                        }
                    )
                    continue
                by_h[str(years)]["results"].append(
                    {"symbol": sym, "brain": brain, "rr": rr, "tf": tf, **bat}
                )
                _log(
                    f"  {years}y {sym:8} {brain:18} {tf:4} n={bat['full']['n']:5} "
                    f"PnL=${bat['full']['pnl']:12,.0f} OOS=${bat['oos']['pnl']:11,.0f} "
                    f"E[R]={bat['oos']['exp_r']:+.3f} {bat['verdict']}"
                )
            del df_all

        for years in HORIZONS_Y:
            desk = _desk_fit(by_h[str(years)]["results"])
            by_h[str(years)]["desk"] = desk
            _log(
                f"\n===== {years}y {tf} desk → "
                f"{desk['survives']}S/{desk['fragile']}F/{desk['dead']}D · "
                f"Full ${desk['full_pnl']:,.0f} · OOS ${desk['oos_pnl']:,.0f} · "
                f"fit {desk['fit_score_1_to_10']}/10 · "
                f"{'FITS' if desk['fits'] else 'DOES NOT FIT'}"
            )
        common[tf] = by_h

    # Per-pair live scan TF (matches wired SYMBOL_SCAN_TFS)
    native: dict[str, Any] = {}
    if RUN_NATIVE:
        _log("\n### NATIVE live scan TFs (per-pair)")
        for years in HORIZONS_Y:
            rows: list[dict] = []
            start_ts = pd.Timestamp(end - timedelta(days=int(years * 365.25)))
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            _log(f"\n--- {years}y native ---")
            for sym in BOOK:
                brain = brains[sym]
                rr = float(symbol_rr(sym, default=2.0))
                tf = scan_tfs.get(sym) or "5m"
                try:
                    df_all = _load_tf(sym, tf)
                    if df_all is None or len(df_all) < 200:
                        raise RuntimeError("empty")
                    df = df_all[df_all.index >= start_ts]
                    if len(df) < 200:
                        raise RuntimeError(f"only {len(df)} bars in window")
                    entries = _collector(brain)(df)
                    bat = _battery(df, entries, rr)
                    rows.append({"symbol": sym, "brain": brain, "rr": rr, "tf": tf, **bat})
                    _log(
                        f"  {sym:8} {brain:18} {tf:4} n={bat['full']['n']:5} "
                        f"PnL=${bat['full']['pnl']:12,.0f} OOS=${bat['oos']['pnl']:11,.0f} "
                        f"E[R]={bat['oos']['exp_r']:+.3f} {bat['verdict']}"
                    )
                except Exception as exc:
                    _log(f"  {sym:8} {brain:18} {tf:4} FAIL: {exc}")
                    rows.append(
                        {
                            "symbol": sym,
                            "brain": brain,
                            "tf": tf,
                            "error": str(exc),
                            "verdict": "DEAD",
                        }
                    )
            desk = _desk_fit(rows)
            native[str(years)] = {
                "years": years,
                "start": start_ts.date().isoformat(),
                "end": end.date().isoformat(),
                "mode": "native_scan_tfs",
                "results": rows,
                "desk": desk,
            }
            _log(
                f"  → Native {years}y desk {desk['survives']}S/{desk['fragile']}F/{desk['dead']}D · "
                f"Full ${desk['full_pnl']:,.0f} · OOS ${desk['oos_pnl']:,.0f} · "
                f"fit {desk['fit_score_1_to_10']}/10 · "
                f"{'FITS' if desk['fits'] else 'DOES NOT FIT'}"
            )

    _log("\n" + "=" * 95)
    fits_5 = []
    for tf in tfs_to_run:
        _log(f"\n### DUKA FIT BY HORIZON — {tf}\n")
        _log(f"{'Years':>6} {'S/F/D':>8} {'Full PnL':>14} {'OOS PnL':>14} {'Fit':>5} {'Fits?':>8}")
        _log("-" * 64)
        for years in HORIZONS_Y:
            desk = common[tf][str(years)]["desk"]
            tag = "YES" if desk["fits"] else "NO"
            if tf == "5m":
                fits_5.append(desk["fits"])
            _log(
                f"{years:6} {desk['survives']}/{desk['fragile']}/{desk['dead']} "
                f"${desk['full_pnl']:13,.0f} ${desk['oos_pnl']:13,.0f} "
                f"{desk['fit_score_1_to_10']:5.1f} {tag:>8}"
            )

    if RUN_NATIVE and native:
        _log("\n### DUKA FIT BY HORIZON — native scan TFs\n")
        _log(f"{'Years':>6} {'S/F/D':>8} {'Full PnL':>14} {'OOS PnL':>14} {'Fit':>5} {'Fits?':>8}")
        _log("-" * 64)
        for years in HORIZONS_Y:
            desk = native[str(years)]["desk"]
            tag = "YES" if desk["fits"] else "NO"
            _log(
                f"{years:6} {desk['survives']}/{desk['fragile']}/{desk['dead']} "
                f"${desk['full_pnl']:13,.0f} ${desk['oos_pnl']:13,.0f} "
                f"{desk['fit_score_1_to_10']:5.1f} {tag:>8}"
            )

    n_fit = sum(1 for x in fits_5 if x)
    h10_5 = common.get("5m", {}).get("10", {}).get("desk", {})
    native_2y = native.get("2", {}).get("desk", {}) if native else {}
    if RUN_NATIVE and native and not fits_5:
        n_fit = sum(1 for y in HORIZONS_Y if native[str(y)]["desk"]["fits"])
        overall = (
            "fits_most_horizons"
            if n_fit >= max(1, len(HORIZONS_Y) - 1)
            else "fits_some"
            if n_fit >= 1
            else "does_not_fit_multi_year"
        )
    else:
        overall = (
            "fits_most_horizons"
            if n_fit >= 4
            else "fits_some"
            if n_fit >= 2
            else "does_not_fit_multi_year"
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "duka_brains_horizons_readonly",
        "source": "dukascopy_local_hist",
        "brains": brains,
        "scan_tfs": scan_tfs,
        "risk_usd": RISK_USD,
        "horizons_y": HORIZONS_Y,
        "common_tf_horizons": common,
        "native_tf_horizons": native,
        "summary": {
            "tf5m_horizons_that_fit": [
                y
                for y in HORIZONS_Y
                if "5m" in common and common["5m"][str(y)]["desk"]["fits"]
            ],
            "tf5m_fit_count": n_fit if "5m" in common else None,
            "overall_5m": overall,
            "desk_5m_10y": h10_5,
            "desk_15m_10y": common.get("15m", {}).get("10", {}).get("desk", {}),
            "desk_native_2y": native_2y,
            "note": (
                "Dukascopy local hist (~10y). CURRENT live brains. "
                "Native = per-pair SYMBOL_SCAN_TFS. "
                f"${RISK_USD}/R flat (not live dynamic lots)."
            ),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    if "5m" in common:
        _log(f"\n→ Overall 5m: {overall} ({n_fit}/{len(HORIZONS_Y)} horizons fit)")
    if native_2y:
        _log(
            f"→ Headline native/2y desk: Full ${native_2y.get('full_pnl', 0):,.0f} · "
            f"OOS ${native_2y.get('oos_pnl', 0):,.0f} · "
            f"{native_2y.get('survives')}S/{native_2y.get('fragile')}F/{native_2y.get('dead')}D · "
            f"fit {native_2y.get('fit_score_1_to_10')}/10"
        )
    if h10_5:
        _log(
            f"→ Headline 5m/10y desk: Full ${h10_5.get('full_pnl', 0):,.0f} · "
            f"OOS ${h10_5.get('oos_pnl', 0):,.0f} · "
            f"{h10_5.get('survives')}S/{h10_5.get('fragile')}F/{h10_5.get('dead')}D · "
            f"fit {h10_5.get('fit_score_1_to_10')}/10"
        )
    _log(f"Wrote {OUT} in {payload['elapsed_s']}s")
    try:
        lock.unlink(missing_ok=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        try:
            (Path(ROOT) / "data" / "duka_brains_horizons.lock").unlink(missing_ok=True)
        except Exception:
            pass

