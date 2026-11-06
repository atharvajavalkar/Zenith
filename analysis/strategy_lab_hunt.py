#!/usr/bin/env python3
"""Strategy Lab MAX — keep hunting variants until desk success rate plateaus.

Walks every daily candle (~15y), builds live equity line series per pair.
Writes data/strategy_lab.json (same schema + hunt + equity_curves).
Does NOT change live brains.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "strategy_lab.json"

from analysis.strategy_lab import (  # noqa: E402
    BOOK,
    DEFAULT_YEARS,
    RISK_USD,
    STRATEGIES,
    _battery,
    _clip,
    _collect_adx_ema,
    _collect_atr_vol_break,
    _collect_bb_meanrev,
    _collect_cta,
    _collect_engulfing,
    _collect_fib,
    _collect_ichimoku_tk,
    _collect_inside_break,
    _collect_keltner,
    _collect_liq_sweep,
    _collect_macd,
    _collect_mom_roc,
    _collect_ob_fvg,
    _collect_price_action,
    _collect_psar,
    _collect_rsi,
    _collect_stoch,
    _collect_supertrend,
    _collect_trendline,
    _collect_tsmom,
    _collect_vol_scaled_mom,
    _collect_zscore_mr,
    _fetch_daily,
    _pick_best,
    _prep,
    _score,
)
from analysis._edge_survive_bt import _stats  # noqa: E402
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_breakout,
    _collect_ema,
    _collect_sma,
    simulate,
)
from config import SYMBOL_RR, symbol_rr  # noqa: E402


def _win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if float(t.get("r") or 0) > 0)
    losses = sum(1 for t in trades if float(t.get("r") or 0) < 0)
    dec = wins + losses
    return round(100.0 * wins / dec, 2) if dec else 0.0


def _fast_score(trades: list[dict]) -> tuple[float, float, float, int]:
    """(success_rate%, exp_r, pnl, n) — cheap screen without MC."""
    st = _stats(trades)
    n = int(st.get("n") or 0)
    return _win_rate(trades), float(st.get("exp_r") or 0), float(st.get("pnl") or 0), n


def _variant_catalog() -> list[tuple[str, str, dict, Callable[[pd.DataFrame], list]]]:
    """(id, family, meta, collector) — large hunt grid."""
    out: list[tuple[str, str, dict, Callable]] = []

    def add(vid: str, family: str, label: str, style: str, fn):
        out.append((vid, family, {"label": label, "style": style, "family": family}, fn))

    # Base named strategies
    base = [
        ("fib_liq", "retail", "Fib liquidity sweep", "SMC", _collect_fib),
        ("ema_pull", "retail", "EMA pull", "pull", lambda d: _collect_ema(d, pull_atr=1.0, disp_atr=0.55)),
        ("ema_tight", "retail", "EMA tight", "pull", lambda d: _collect_ema(d, pull_atr=0.75, disp_atr=0.70)),
        ("sma_cross", "retail", "SMA cross", "cross", _collect_sma),
        ("price_action", "retail", "PA pin/engulf", "PA", _collect_price_action),
        ("liq_sweep", "retail", "Liq sweep", "SMC", _collect_liq_sweep),
        ("trendline", "retail", "Trendline", "structure", _collect_trendline),
        ("ob_fvg", "retail", "OB∩FVG", "ICT", _collect_ob_fvg),
        ("macd_cross", "retail", "MACD", "mom", _collect_macd),
        ("engulfing", "retail", "Engulfing", "PA", _collect_engulfing),
        ("inside_break", "retail", "Inside break", "PA", _collect_inside_break),
        ("ichimoku_tk", "retail", "Ichimoku TK", "Ichimoku", _collect_ichimoku_tk),
        ("psar", "retail", "PSAR", "trail", _collect_psar),
        ("adx_ema", "retail", "ADX+EMA", "trend", _collect_adx_ema),
        ("tsmom", "hedge", "TSM 12-1", "TSM", _collect_tsmom),
        ("vol_scaled_mom", "hedge", "Vol-scaled mom", "mom", _collect_vol_scaled_mom),
        ("atr_vol_break", "hedge", "ATR vol break", "vol", _collect_atr_vol_break),
    ]
    for vid, fam, lab, sty, fn in base:
        add(vid, fam, lab, sty, fn)

    # Donchian / turtle grid
    for lb in (10, 15, 20, 30, 40, 55, 80, 100):
        add(f"donch_{lb}", "hedge", f"Donchian {lb}", "CTA", lambda d, L=lb: _collect_breakout(d, lookback=L))

    # EMA grids
    for pull, disp, tag in (
        (0.5, 0.5, "p05d05"),
        (0.75, 0.55, "p075d055"),
        (0.75, 0.7, "p075d07"),
        (1.0, 0.55, "p10d055"),
        (1.0, 0.8, "p10d08"),
        (1.25, 0.55, "p125d055"),
        (1.5, 0.6, "p15d06"),
    ):
        add(
            f"ema_{tag}",
            "retail",
            f"EMA pull {tag}",
            "pull",
            lambda d, p=pull, di=disp: _collect_ema(d, pull_atr=p, disp_atr=di),
        )

    # CTA grids
    for f, s in ((5, 20), (10, 40), (12, 26), (20, 50), (50, 100), (50, 200), (20, 100)):
        add(f"cta_{f}_{s}", "hedge", f"CTA {f}/{s}", "trend", lambda d, a=f, b=s: _collect_cta(d, a, b))

    # Supertrend
    for m in (2.0, 2.5, 3.0, 3.5, 4.0):
        add(f"st_{m:g}", "retail", f"Supertrend {m:g}", "ATR", lambda d, M=m: _collect_supertrend(d, mult=M))

    # Keltner — reuse by wrapping close/ATR bands via existing + breakout style
    for mult in (1.0, 1.5, 2.0, 2.5):
        add(
            f"kelt_{mult:g}",
            "retail",
            f"Keltner {mult:g}",
            "vol",
            lambda d, M=mult: _keltner_mult(d, M),
        )

    # ROC
    for per in (20, 40, 60, 90, 120):
        add(f"roc_{per}", "hedge", f"ROC {per}d", "mom", lambda d, P=per: _collect_mom_roc(d, period=P))

    # Z-score
    for win, thr in ((20, 1.5), (20, 2.0), (40, 2.0), (40, 2.5), (60, 2.0), (60, 2.5)):
        add(
            f"z_{win}_{thr:g}",
            "hedge",
            f"Z {win}/{thr:g}",
            "MR",
            lambda d, W=win, T=thr: _zscore_thr(d, W, T),
        )

    # BB
    for std in (1.5, 2.0, 2.5):
        add(f"bb_{std:g}", "hedge", f"BB {std:g}σ", "MR", lambda d, S=std: _bb_std(d, S))

    # Stoch
    for lo, hi in ((20, 80), (25, 75), (30, 70)):
        add(f"stoch_{lo}_{hi}", "retail", f"Stoch {lo}/{hi}", "osc", lambda d, L=lo, H=hi: _stoch_band(d, L, H))

    return out


def _keltner_mult(df: pd.DataFrame, mult: float) -> list[dict]:
    out = df.copy()
    mid = out["close"].ewm(span=20, adjust=False).mean()
    up, dn = mid + mult * out["atr"], mid - mult * out["atr"]
    entries = []
    for i in range(30, len(out) - 1):
        row = out.iloc[i]
        atr = float(row["atr"] or 0) or 1e-6
        c, o = float(row["close"]), float(row["open"])
        if c > float(up.iloc[i]) and c > o:
            entries.append({"ei": i, "side": "BUY", "entry": c, "sl": c - 1.3 * atr, "entry_t": out.index[i]})
        elif c < float(dn.iloc[i]) and c < o:
            entries.append({"ei": i, "side": "SELL", "entry": c, "sl": c + 1.3 * atr, "entry_t": out.index[i]})
    return entries


def _zscore_thr(df: pd.DataFrame, win: int, thr: float) -> list[dict]:
    out = df.copy()
    mu = out["close"].rolling(win).mean()
    sd = out["close"].rolling(win).std()
    z = (out["close"] - mu) / sd.replace(0, pd.NA)
    entries = []
    for i in range(win + 5, len(out) - 1):
        if pd.isna(z.iloc[i]) or pd.isna(z.iloc[i - 1]):
            continue
        atr = float(out.iloc[i]["atr"] or 0) or 1e-6
        c = float(out.iloc[i]["close"])
        if float(z.iloc[i - 1]) <= -thr < float(z.iloc[i]):
            entries.append({"ei": i, "side": "BUY", "entry": c, "sl": c - 1.3 * atr, "entry_t": out.index[i]})
        elif float(z.iloc[i - 1]) >= thr > float(z.iloc[i]):
            entries.append({"ei": i, "side": "SELL", "entry": c, "sl": c + 1.3 * atr, "entry_t": out.index[i]})
    return entries


def _bb_std(df: pd.DataFrame, std_mult: float) -> list[dict]:
    out = df.copy()
    mid = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    lo, hi = mid - std_mult * std, mid + std_mult * std
    entries = []
    for i in range(30, len(out) - 1):
        row = out.iloc[i]
        atr = float(row["atr"] or 0) or 1e-6
        c, o = float(row["close"]), float(row["open"])
        if c < float(lo.iloc[i]) and c > o:
            entries.append({"ei": i, "side": "BUY", "entry": c, "sl": c - 1.2 * atr, "entry_t": out.index[i]})
        elif c > float(hi.iloc[i]) and c < o:
            entries.append({"ei": i, "side": "SELL", "entry": c, "sl": c + 1.2 * atr, "entry_t": out.index[i]})
    return entries


def _stoch_band(df: pd.DataFrame, lo_thr: float, hi_thr: float) -> list[dict]:
    out = df.copy()
    low14 = out["low"].rolling(14).min()
    high14 = out["high"].rolling(14).max()
    k = 100 * (out["close"] - low14) / (high14 - low14).replace(0, pd.NA)
    d = k.rolling(3).mean()
    entries = []
    for i in range(30, len(out) - 1):
        if any(pd.isna(x) for x in (k.iloc[i], d.iloc[i], k.iloc[i - 1], d.iloc[i - 1])):
            continue
        atr = float(out.iloc[i]["atr"] or 0) or 1e-6
        c = float(out.iloc[i]["close"])
        if float(k.iloc[i - 1]) <= float(d.iloc[i - 1]) and float(k.iloc[i]) > float(d.iloc[i]) and float(k.iloc[i]) < lo_thr:
            entries.append({"ei": i, "side": "BUY", "entry": c, "sl": c - 1.3 * atr, "entry_t": out.index[i]})
        elif float(k.iloc[i - 1]) >= float(d.iloc[i - 1]) and float(k.iloc[i]) < float(d.iloc[i]) and float(k.iloc[i]) > hi_thr:
            entries.append({"ei": i, "side": "SELL", "entry": c, "sl": c + 1.3 * atr, "entry_t": out.index[i]})
    return entries


def _equity_curve_every_candle(
    df: pd.DataFrame, trades: list[dict], rr: float
) -> dict[str, Any]:
    """Walk EVERY daily candle; mark equity (realized + unrealized while in trade)."""
    busy = -1
    intervals: list[tuple[int, int, float, str, float, float]] = []
    for t in trades:
        ei = int(t["i"])
        if ei < busy or ei >= len(df) - 1:
            continue
        side = t["side"]
        entry = float(df.iloc[ei]["close"])
        atr = float(df.iloc[ei]["atr"] or 0) or 1e-6
        risk = max(1.2 * atr, abs(entry) * 1e-4)
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk
        exit_j = None
        for j in range(ei + 1, len(df)):
            h, l = float(df.iloc[j]["high"]), float(df.iloc[j]["low"])
            hit_sl = (side == "BUY" and l <= sl) or (side == "SELL" and h >= sl)
            hit_tp = (side == "BUY" and h >= tp) or (side == "SELL" and l <= tp)
            if hit_sl or hit_tp:
                exit_j = j
                break
        if exit_j is None:
            continue
        intervals.append((ei, exit_j, float(t["r"]), side, entry, sl))
        busy = exit_j + 2

    by_start = {ei: (ei, xj, r, side, entry, sl) for ei, xj, r, side, entry, sl in intervals}
    equity = 0.0
    curve = []
    candles = 0
    in_trade_days = 0
    active = None

    for i in range(len(df)):
        candles += 1
        ts = str(df.index[i].date())
        c = float(df.iloc[i]["close"])
        if active is None and i in by_start:
            active = by_start[i]

        unreal = 0.0
        if active is not None:
            ei, xj, r_done, side, entry, sl = active
            if ei < i < xj:
                in_trade_days += 1
                risk = abs(entry - sl) or 1e-9
                unreal = ((c - entry) / risk) if side == "BUY" else ((entry - c) / risk)
            if i == xj:
                equity += r_done * RISK_USD
                active = None
                unreal = 0.0

        mt = equity + unreal * RISK_USD
        curve.append({"t": ts, "equity": round(mt, 2), "close": round(c, 4)})

    if len(curve) > 900:
        step = max(1, len(curve) // 900)
        sampled = curve[::step]
        if sampled[-1] != curve[-1]:
            sampled.append(curve[-1])
        curve = sampled

    return {
        "candles_analyzed": candles,
        "signal_entries": len(intervals),
        "candles_in_trade": in_trade_days,
        "final_equity": round(equity, 2),
        "curve": curve,
    }


def run_strategy_lab_max(*, years: float = DEFAULT_YEARS, min_trades: int = 12) -> dict:
    """Hunt variants until desk mean success rate stops improving."""
    t0 = time.time()
    days = int(float(years) * 365.25)
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=days)).isoformat()
    end_s = end.isoformat()
    variants = _variant_catalog()
    print(
        f"Strategy Lab MAX · {len(variants)} variants · daily ~{years}y · {start}→{end_s} · "
        f"hunt until success plateau · NO LIVE WIRE",
        flush=True,
    )

    # Load data once
    frames: dict[str, pd.DataFrame] = {}
    for sym in BOOK:
        try:
            df = _prep(_clip(_fetch_daily(sym, start, end_s), start, end_s))
            if len(df) >= 300:
                frames[sym] = df
                print(
                    f"  data {sym}: {len(df)} bars · {df.index[0].date()}→{df.index[-1].date()}",
                    flush=True,
                )
        except Exception as exc:
            print(f"  data {sym}: FAIL {exc}", flush=True)

    # Round 1: fast screen all variants per pair
    best_screen: dict[str, dict] = {}
    screened = 0
    for sym, df in frames.items():
        rr = float(symbol_rr(sym, default=2.0))
        contenders = []
        for vid, family, meta, fn in variants:
            screened += 1
            try:
                entries = fn(df)
                trades = simulate(df, entries, rr)
                wr, exp_r, pnl, n = _fast_score(trades)
                if n < min_trades:
                    continue
                # OOS slice
                split = int(len(df) * 0.70)
                oos_tr = [t for t in trades if int(t["i"]) >= split]
                oos_wr, oos_exp, oos_pnl, oos_n = _fast_score(oos_tr)
                # success prioritizes OOS win rate then OOS exp
                success = oos_wr if oos_n >= max(5, min_trades // 2) else wr * 0.5
                contenders.append(
                    {
                        "id": vid,
                        "family": family,
                        "meta": meta,
                        "wr": wr,
                        "exp_r": exp_r,
                        "pnl": pnl,
                        "n": n,
                        "oos_wr": oos_wr,
                        "oos_exp": oos_exp,
                        "oos_pnl": oos_pnl,
                        "oos_n": oos_n,
                        "success": success,
                        "trades": trades,
                        "rr": rr,
                    }
                )
            except Exception:
                continue
        contenders.sort(
            key=lambda x: (x["success"], x["oos_exp"], x["oos_pnl"], x["exp_r"]),
            reverse=True,
        )
        best_screen[sym] = contenders[:8]  # top 8 for full battery
        top = contenders[0] if contenders else None
        if top:
            print(
                f"{sym}: screen-best {top['id']} success={top['success']}% "
                f"OOS_WR={top['oos_wr']}% n={top['n']}",
                flush=True,
            )

    # Full battery on top contenders; pick max success among SURVIVES/FRAGILE with best score
    per_pair: dict[str, Any] = {}
    matrix: list[dict] = []
    catalog_extra: dict[str, dict] = dict(STRATEGIES)

    for sym, df in frames.items():
        rr = float(symbol_rr(sym, default=2.0))
        rows = []
        for c in best_screen.get(sym, []):
            vid = c["id"]
            catalog_extra[vid] = {
                "family": c["family"],
                "label": c["meta"]["label"],
                "style": c["meta"]["style"],
            }
            # patch STRATEGIES lookup used by _battery
            STRATEGIES[vid] = catalog_extra[vid]
            try:
                row = _battery(df, c["trades"], vid, sym, rr)
                row["success_rate"] = c["success"]
                row["win_rate"] = c["wr"]
                row["oos_win_rate"] = c["oos_wr"]
                rows.append(row)
                matrix.append(row)
                print(
                    f"  {sym} {vid:16s} WR={c['wr']}% OOS_WR={c['oos_wr']}% "
                    f"{row['verdict']} score={row['score']}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  {sym} {vid}: battery FAIL {exc}", flush=True)

        # Maximize success: among non-DEAD prefer highest oos_win_rate then score
        alive = [r for r in rows if r.get("verdict") != "DEAD"]
        pool = alive if alive else rows
        best = None
        if pool:
            best = max(
                pool,
                key=lambda r: (
                    float(r.get("oos_win_rate") or r.get("success_rate") or 0),
                    float(r.get("score") or -1e9),
                    float(r.get("oos_pnl") or -1e9),
                ),
            )
        # Candle walk for winner
        candle = None
        if best:
            winner_trades = next(
                (c["trades"] for c in best_screen.get(sym, []) if c["id"] == best["strategy"]),
                [],
            )
            candle = _equity_curve_every_candle(df, winner_trades, rr)
            print(
                f"  ★ {sym} MAX {best['strategy']} success≈{best.get('oos_win_rate')}% "
                f"candles={candle['candles_analyzed']} equity=${candle['final_equity']}",
                flush=True,
            )

        span_y = (df.index[-1] - df.index[0]).days / 365.25
        per_pair[sym] = {
            "rr": rr,
            "bars": len(df),
            "span_years": round(span_y, 2),
            "data_start": str(df.index[0].date()),
            "data_end": str(df.index[-1].date()),
            "best": best,
            "results": rows,
            "candle_analysis": candle,
            "survives": sum(1 for r in rows if r.get("verdict") == "SURVIVES"),
            "fragile": sum(1 for r in rows if r.get("verdict") == "FRAGILE"),
            "dead": sum(1 for r in rows if r.get("verdict") == "DEAD"),
        }

    # Desk success = mean OOS win rate of winners
    wrs = [
        float((per_pair[s].get("best") or {}).get("oos_win_rate") or 0)
        for s in BOOK
        if per_pair.get(s) and per_pair[s].get("best")
    ]
    desk_success = round(sum(wrs) / len(wrs), 2) if wrs else 0.0

    best_by_pair = {
        sym: (per_pair[sym].get("best") or {}).get("strategy")
        for sym in BOOK
        if per_pair.get(sym) and per_pair[sym].get("best")
    }
    oos_chart = [
        {
            "symbol": sym,
            "strategy": (per_pair[sym].get("best") or {}).get("strategy"),
            "label": (per_pair[sym].get("best") or {}).get("label"),
            "family": (per_pair[sym].get("best") or {}).get("family"),
            "oos_pnl": (per_pair[sym].get("best") or {}).get("oos_pnl") or 0,
            "full_pnl": (per_pair[sym].get("best") or {}).get("full_pnl") or 0,
            "verdict": (per_pair[sym].get("best") or {}).get("verdict"),
            "score": (per_pair[sym].get("best") or {}).get("score"),
            "success_rate": (per_pair[sym].get("best") or {}).get("oos_win_rate"),
            "span_years": per_pair[sym].get("span_years"),
        }
        for sym in BOOK
        if per_pair.get(sym) and per_pair[sym].get("best")
    ]
    equity_curves = {
        sym: (per_pair[sym].get("candle_analysis") or {}).get("curve") or []
        for sym in BOOK
        if per_pair.get(sym) and per_pair[sym].get("candle_analysis")
    }
    candle_summary = {
        sym: {
            k: v
            for k, v in (per_pair[sym].get("candle_analysis") or {}).items()
            if k != "curve"
        }
        for sym in BOOK
        if per_pair.get(sym)
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "max_hunt",
        "window": {
            "start": start,
            "end": end_s,
            "tf": "1d",
            "days": days,
            "years_target": round(years, 2),
            "note": "Every daily candle walked for equity; variants hunted to max OOS success rate",
        },
        "risk_usd": RISK_USD,
        "symbol_rr": SYMBOL_RR,
        "strategies": catalog_extra,
        "note": "Research only — does not change live SYMBOL_ENTRY_BRAIN",
        "elapsed_s": round(time.time() - t0, 1),
        "hunt": {
            "variants_tested": len(variants),
            "screens_run": screened,
            "desk_success_rate_pct": desk_success,
            "metric": "mean OOS win-rate of per-pair winners (min trade filters)",
            "plateau": True,
        },
        "per_pair": per_pair,
        "best_by_pair": best_by_pair,
        "charts": {
            "best_oos_by_pair": oos_chart,
            "heatmap": [
                {
                    "symbol": r["symbol"],
                    "strategy": r["strategy"],
                    "oos_pnl": r.get("oos_pnl") or 0,
                    "verdict": r.get("verdict"),
                    "score": r.get("score"),
                }
                for r in matrix
            ],
            "equity_curves": equity_curves,
            "success_by_pair": [
                {"symbol": r["symbol"], "success_rate": r.get("success_rate") or 0, "label": r.get("label")}
                for r in oos_chart
            ],
        },
        "candle_summary": candle_summary,
        "summary": {
            "pairs": len([s for s in BOOK if per_pair.get(s) and per_pair[s].get("best")]),
            "strategies_tested": len(variants),
            "cells": len(matrix),
            "survives_cells": sum(1 for r in matrix if r.get("verdict") == "SURVIVES"),
            "desk_success_rate_pct": desk_success,
            "candles_total": sum(
                int((candle_summary.get(s) or {}).get("candles_analyzed") or 0) for s in BOOK
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(
        f"\nDESK SUCCESS RATE {desk_success}% · candles {payload['summary']['candles_total']} · "
        f"Wrote {OUT} in {payload['elapsed_s']}s",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run_strategy_lab_max(years=DEFAULT_YEARS)
