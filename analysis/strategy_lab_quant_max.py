#!/usr/bin/env python3
"""Quant-MAX Strategy Lab — SURVIVES-only · multi-TF · max candles available.

Reality checks (in payload):
  - Level-2 / order-book history is NOT available for this desk (no 15y L2 archive).
  - 15y of 1m/5m is not on Yahoo/TV free feeds; we take MAX bars per TF.
  - Daily covers ~15y; 1h/15m/5m/4h use longest available window.
  - Every bar in each loaded frame is walked; FRAGILE never promoted.

Writes data/strategy_lab.json + live data/strategy_lab_progress.json.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "strategy_lab.json"
PROGRESS = ROOT / "data" / "strategy_lab_progress.json"

from analysis._edge_survive_bt import BOOK, RISK_USD, _fetch_1h, _prep  # noqa: E402
from analysis._htf_trend_5m_exit_bt import _fetch_5m  # noqa: E402
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_breakout,
    _collect_ema,
    _collect_fib,
    _collect_sma,
    simulate,
)
from analysis.strategy_lab import (  # noqa: E402
    DEFAULT_YEARS,
    _battery,
    _clip,
    _collect_adx_ema,
    _collect_atr_vol_break,
    _collect_cta,
    _collect_engulfing,
    _collect_ichimoku_tk,
    _collect_inside_break,
    _collect_liq_sweep,
    _collect_macd,
    _collect_mom_roc,
    _collect_ob_fvg,
    _collect_price_action,
    _collect_psar,
    _collect_supertrend,
    _collect_trendline,
    _collect_tsmom,
    _collect_vol_scaled_mom,
    _fetch_daily,
)
from analysis.strategy_lab_hunt import (  # noqa: E402
    _bb_std,
    _equity_curve_every_candle,
    _fast_score,
    _keltner_mult,
    _stoch_band,
    _zscore_thr,
)
from config import SYMBOL_RR, symbol_rr  # noqa: E402


def _progress(update: dict) -> None:
    cur: dict[str, Any] = {}
    if PROGRESS.exists():
        try:
            cur = json.loads(PROGRESS.read_text())
        except Exception:
            cur = {}
    cur.update(update)
    cur["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(cur, indent=2, default=str))


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    m = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}[rule]
    out = pd.DataFrame(
        {
            "open": df["open"].resample(m).first(),
            "high": df["high"].resample(m).max(),
            "low": df["low"].resample(m).min(),
            "close": df["close"].resample(m).last(),
        }
    ).dropna()
    return _prep(out)


def _variants() -> list[tuple[str, str, dict, Callable]]:
    out: list[tuple[str, str, dict, Callable]] = []

    def add(vid, fam, lab, sty, fn):
        out.append((vid, fam, {"label": lab, "style": sty, "family": fam}, fn))

    add("fib_liq", "retail", "Fib liq", "SMC", _collect_fib)
    add("ema_pull", "retail", "EMA pull", "pull", lambda d: _collect_ema(d, 1.0, 0.55))
    add("ema_tight", "retail", "EMA tight", "pull", lambda d: _collect_ema(d, 0.75, 0.70))
    add("sma_cross", "retail", "SMA cross", "cross", _collect_sma)
    add("pa", "retail", "Price action", "PA", _collect_price_action)
    add("liq", "retail", "Liq sweep", "SMC", _collect_liq_sweep)
    add("tl", "retail", "Trendline", "struct", _collect_trendline)
    add("obfvg", "retail", "OB∩FVG", "ICT", _collect_ob_fvg)
    add("macd", "retail", "MACD", "mom", _collect_macd)
    add("engulf", "retail", "Engulfing", "PA", _collect_engulfing)
    add("inside", "retail", "Inside break", "PA", _collect_inside_break)
    add("ichi", "retail", "Ichimoku TK", "Ichi", _collect_ichimoku_tk)
    add("psar", "retail", "PSAR", "trail", _collect_psar)
    add("adx", "retail", "ADX+EMA", "trend", _collect_adx_ema)
    add("tsmom", "hedge", "TSM 12-1", "TSM", _collect_tsmom)
    add("volmom", "hedge", "Vol mom", "mom", _collect_vol_scaled_mom)
    add("atrbrk", "hedge", "ATR vol break", "vol", _collect_atr_vol_break)

    for lb in (10, 15, 20, 30, 40, 55, 80, 100, 120):
        add(f"don_{lb}", "hedge", f"Donch {lb}", "CTA", lambda d, L=lb: _collect_breakout(d, L))
    for f, s in ((5, 20), (8, 34), (10, 40), (12, 26), (20, 50), (50, 100), (50, 200)):
        add(f"cta_{f}_{s}", "hedge", f"CTA {f}/{s}", "trend", lambda d, a=f, b=s: _collect_cta(d, a, b))
    for m in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        add(f"st_{m:g}", "retail", f"ST {m:g}", "ATR", lambda d, M=m: _collect_supertrend(d, M))
    for m in (1.0, 1.25, 1.5, 2.0, 2.5):
        add(f"kelt_{m:g}", "retail", f"Kelt {m:g}", "vol", lambda d, M=m: _keltner_mult(d, M))
    for p, di, tag in (
        (0.5, 0.45, "a"),
        (0.75, 0.55, "b"),
        (0.75, 0.8, "c"),
        (1.0, 0.55, "d"),
        (1.25, 0.55, "e"),
        (1.5, 0.7, "f"),
        (2.0, 0.6, "g"),
    ):
        add(f"ema_{tag}", "retail", f"EMA {tag}", "pull", lambda d, P=p, D=di: _collect_ema(d, P, D))
    for per in (10, 20, 40, 60, 90, 120, 180):
        add(f"roc_{per}", "hedge", f"ROC {per}", "mom", lambda d, P=per: _collect_mom_roc(d, P))
    for w, t in ((15, 1.5), (20, 2.0), (40, 2.0), (40, 2.5), (60, 2.0), (80, 2.5)):
        add(f"z_{w}_{t:g}", "hedge", f"Z {w}/{t:g}", "MR", lambda d, W=w, T=t: _zscore_thr(d, W, T))
    for s in (1.5, 2.0, 2.5, 3.0):
        add(f"bb_{s:g}", "hedge", f"BB {s:g}", "MR", lambda d, S=s: _bb_std(d, S))
    for lo, hi in ((15, 85), (20, 80), (25, 75), (30, 70)):
        add(f"stoch_{lo}", "retail", f"Stoch {lo}/{hi}", "osc", lambda d, L=lo, H=hi: _stoch_band(d, L, H))
    return out


def _load_frames(sym: str, years: float) -> dict[str, pd.DataFrame]:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    end_s = end.isoformat()
    frames: dict[str, pd.DataFrame] = {}

    start_d = (end - timedelta(days=int(years * 365.25))).isoformat()
    try:
        d1 = _prep(_clip(_fetch_daily(sym, start_d, end_s), start_d, end_s))
        if len(d1) >= 250:
            frames["1d"] = d1
    except Exception as e:
        print(f"  {sym} 1d fail: {e}", flush=True)

    start_h = (end - timedelta(days=730)).isoformat()
    try:
        h = _prep(_clip(_fetch_1h(sym, start_h, end_s), start_h, end_s))
        if len(h) >= 300:
            frames["1h"] = h
    except Exception as e:
        print(f"  {sym} 1h fail: {e}", flush=True)

    start_5 = (end - timedelta(days=120)).isoformat()
    try:
        from data.hist_local import has_local_hist, local_span

        if has_local_hist(sym, "5m"):
            span = local_span(sym, "5m")
            if span is not None:
                lo = max(span[0].date(), end - timedelta(days=int(years * 365.25)))
                start_5 = lo.isoformat()
    except Exception:
        pass
    try:
        raw5 = _fetch_5m(sym, start_5, end_s)
        clipped = _clip(raw5, start_5, end_s)
        clipped = clipped.rename(columns=str.lower)
        m5 = _prep(clipped[["open", "high", "low", "close"]].dropna())
        if len(m5) >= 400:
            frames["5m"] = m5
            m15 = _resample(m5, "15m")
            if len(m15) >= 200:
                frames["15m"] = m15
            m4 = _resample(m5, "4h")
            if len(m4) >= 80:
                frames["4h"] = m4
    except Exception as e:
        print(f"  {sym} 5m fail: {e}", flush=True)

    return frames


def run_quant_max(*, years: float = DEFAULT_YEARS, min_trades: int = 10) -> dict:
    t0 = time.time()
    variants = _variants()
    _progress(
        {
            "status": "running",
            "phase": "init",
            "pct": 0,
            "message": "Loading multi-TF data (max history per TF)…",
            "desk_success_rate_pct": None,
            "candles_total": 0,
            "level2": "unavailable",
        }
    )
    print(
        f"QUANT-MAX · SURVIVES-only · multi-TF · {len(variants)} variants · "
        f"~{years}y daily + max 1h/15m/5m/4h · NO L2 · NO LIVE WIRE",
        flush=True,
    )

    all_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for i, sym in enumerate(BOOK):
        _progress(
            {
                "phase": "load_data",
                "pct": int(5 + 15 * i / len(BOOK)),
                "message": f"Loading {sym} multi-TF…",
                "symbol": sym,
            }
        )
        all_frames[sym] = _load_frames(sym, years)
        print(f"{sym} frames {{{', '.join(f'{k}:{len(v)}' for k,v in all_frames[sym].items())}}}", flush=True)

    per_pair: dict[str, Any] = {}
    matrix: list[dict] = []
    catalog: dict[str, dict] = {}
    equity_curves: dict[str, list] = {}
    candles_total = 0
    cells_done = 0
    n_jobs = max(1, sum(len(tfs) for tfs in all_frames.values()) * (len(variants) // 3))

    from analysis import strategy_lab as sl

    for si, sym in enumerate(BOOK):
        frames = all_frames.get(sym) or {}
        if not frames:
            per_pair[sym] = {"error": "no_data", "best": None, "results": []}
            continue
        rr = float(symbol_rr(sym, default=2.0))
        contenders: list[dict] = []

        for tf, df in frames.items():
            candles_total += len(df)
            for vid, fam, meta, fn in variants:
                cells_done += 1
                if cells_done % 50 == 0:
                    _progress(
                        {
                            "phase": "screen",
                            "pct": min(70, 20 + int(50 * cells_done / n_jobs)),
                            "message": f"Screening {sym} {tf} {vid}",
                            "symbol": sym,
                            "tf": tf,
                            "candles_total": candles_total,
                        }
                    )
                try:
                    if len(df) < 260 and vid == "tsmom":
                        continue
                    if len(df) < 220 and vid == "cta_50_200":
                        continue
                    entries = fn(df)
                    trades = simulate(df, entries, rr)
                    wr, exp_r, pnl, n = _fast_score(trades)
                    if n < min_trades:
                        continue
                    split = int(len(df) * 0.70)
                    oos = [t for t in trades if int(t["i"]) >= split]
                    oos_wr, oos_exp, oos_pnl, oos_n = _fast_score(oos)
                    if oos_n < max(4, min_trades // 3) or oos_exp <= 0:
                        continue
                    contenders.append(
                        {
                            "id": f"{vid}@{tf}",
                            "tf": tf,
                            "family": fam,
                            "meta": meta,
                            "wr": wr,
                            "oos_wr": oos_wr,
                            "oos_exp": oos_exp,
                            "oos_pnl": oos_pnl,
                            "n": n,
                            "trades": trades,
                            "df": df,
                            "rr": rr,
                            "success": oos_wr,
                        }
                    )
                except Exception:
                    continue

        contenders.sort(key=lambda x: (x["success"], x["oos_exp"], x["oos_pnl"]), reverse=True)
        top = contenders[:12]
        rows: list[dict] = []
        _progress(
            {
                "phase": "battery",
                "pct": min(92, 72 + int(18 * si / len(BOOK))),
                "message": f"Full battery {sym} ({len(top)} contenders, SURVIVES-only)…",
                "symbol": sym,
                "candles_total": candles_total,
            }
        )
        for c in top:
            vid = c["id"]
            catalog[vid] = {
                "family": c["family"],
                "label": f"{c['meta']['label']} · {c['tf']}",
                "style": c["meta"]["style"],
                "tf": c["tf"],
            }
            sl.STRATEGIES[vid] = catalog[vid]
            try:
                row = _battery(c["df"], c["trades"], vid, sym, rr)
                row["success_rate"] = c["success"]
                row["win_rate"] = c["wr"]
                row["oos_win_rate"] = c["oos_wr"]
                row["tf"] = c["tf"]
                row["candles"] = len(c["df"])
                rows.append(row)
                matrix.append(row)
                print(
                    f"  {sym} {vid:24s} OOS_WR={c['oos_wr']}% {row['verdict']} "
                    f"score={row['score']} bars={len(c['df'])}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  {sym} {vid}: FAIL {exc}", flush=True)

        survives = [r for r in rows if r.get("verdict") == "SURVIVES"]
        best = None
        if survives:
            best = max(
                survives,
                key=lambda r: (
                    float(r.get("oos_win_rate") or 0),
                    float(r.get("score") or -1e9),
                    float(r.get("oos_pnl") or -1e9),
                ),
            )
            winner_c = next((c for c in top if c["id"] == best["strategy"]), None)
            if winner_c:
                ca = _equity_curve_every_candle(winner_c["df"], winner_c["trades"], rr)
                equity_curves[sym] = ca["curve"]
                best["candle_analysis"] = {k: v for k, v in ca.items() if k != "curve"}
                print(
                    f"  ★ {sym} SURVIVES {best['strategy']} success={best.get('oos_win_rate')}% "
                    f"equity=${ca['final_equity']}",
                    flush=True,
                )
                _progress(
                    {
                        "message": f"★ {sym} locked SURVIVES {best['strategy']}",
                        "last_winner": {"symbol": sym, "strategy": best["strategy"], "success": best.get("oos_win_rate")},
                        "live_curve_symbol": sym,
                        "live_curve_tail": (ca["curve"] or [])[-40:],
                    }
                )
        else:
            print(f"  ✗ {sym}: NO SURVIVES edge (FRAGILE discarded)", flush=True)

        per_pair[sym] = {
            "rr": rr,
            "tfs": {
                tf: {"bars": len(df), "start": str(df.index[0].date()), "end": str(df.index[-1].date())}
                for tf, df in frames.items()
            },
            "candles_all_tfs": sum(len(df) for df in frames.values()),
            "best": best,
            "results": rows,
            "survives": len(survives),
            "fragile_discarded": sum(1 for r in rows if r.get("verdict") == "FRAGILE"),
            "dead": sum(1 for r in rows if r.get("verdict") == "DEAD"),
            "quant_rule": "SURVIVES_ONLY",
        }

    wrs = [
        float((per_pair[s].get("best") or {}).get("oos_win_rate") or 0)
        for s in BOOK
        if per_pair.get(s) and per_pair[s].get("best")
    ]
    desk_success = round(sum(wrs) / len(wrs), 2) if wrs else 0.0
    survives_pairs = sum(1 for s in BOOK if per_pair.get(s) and per_pair[s].get("best"))

    oos_chart = []
    for sym in BOOK:
        b = (per_pair.get(sym) or {}).get("best")
        if not b:
            continue
        oos_chart.append(
            {
                "symbol": sym,
                "strategy": b.get("strategy"),
                "label": b.get("label"),
                "family": b.get("family"),
                "tf": b.get("tf"),
                "oos_pnl": b.get("oos_pnl") or 0,
                "full_pnl": b.get("full_pnl") or 0,
                "verdict": b.get("verdict"),
                "score": b.get("score"),
                "success_rate": b.get("oos_win_rate"),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "quant_max_survives_only",
        "window": {
            "years_target": years,
            "tfs": ["1d", "1h", "4h", "15m", "5m"],
            "note": (
                "Multi-TF max history. Daily≈15y; 1h≈2y; 5m/15m/4h≈max free intraday. "
                "Level-2 historical order book unavailable on this stack."
            ),
        },
        "limitations": {
            "level2": "unavailable_historically",
            "intraday_15y": "not_available_on_yahoo_tv_free",
            "candle_math": (
                "15y×daily≈3750 bars/pair. Finer TFs add many more bars but only over shorter "
                "windows. Every bar in each loaded frame is analyzed."
            ),
        },
        "risk_usd": RISK_USD,
        "symbol_rr": SYMBOL_RR,
        "strategies": catalog,
        "note": "Research only — SURVIVES-only — does not change live brains",
        "elapsed_s": round(time.time() - t0, 1),
        "hunt": {
            "variants_tested": len(variants),
            "desk_success_rate_pct": desk_success,
            "survives_pairs": survives_pairs,
            "pairs_total": len(BOOK),
            "metric": "mean OOS win-rate among SURVIVES-only winners",
            "fragile_policy": "discarded",
        },
        "per_pair": per_pair,
        "best_by_pair": {
            s: (per_pair[s].get("best") or {}).get("strategy")
            for s in BOOK
            if per_pair.get(s) and per_pair[s].get("best")
        },
        "charts": {
            "best_oos_by_pair": oos_chart,
            "success_by_pair": [
                {
                    "symbol": r["symbol"],
                    "success_rate": r.get("success_rate") or 0,
                    "label": r.get("label"),
                }
                for r in oos_chart
            ],
            "equity_curves": equity_curves,
            "heatmap": [
                {
                    "symbol": r["symbol"],
                    "strategy": r["strategy"],
                    "oos_pnl": r.get("oos_pnl") or 0,
                    "verdict": r.get("verdict"),
                    "score": r.get("score"),
                }
                for r in matrix
                if r.get("verdict") == "SURVIVES"
            ],
        },
        "candle_summary": {
            s: {
                "candles_all_tfs": (per_pair[s] or {}).get("candles_all_tfs"),
                "tfs": (per_pair[s] or {}).get("tfs"),
                **((per_pair[s].get("best") or {}).get("candle_analysis") or {}),
            }
            for s in BOOK
            if per_pair.get(s)
        },
        "summary": {
            "pairs_with_survives": survives_pairs,
            "strategies_tested": len(variants),
            "cells": len(matrix),
            "survives_cells": sum(1 for r in matrix if r.get("verdict") == "SURVIVES"),
            "desk_success_rate_pct": desk_success,
            "candles_total": candles_total,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _progress(
        {
            "status": "done",
            "phase": "complete",
            "pct": 100,
            "message": f"Done · desk success {desk_success}% · {survives_pairs}/{len(BOOK)} SURVIVES",
            "desk_success_rate_pct": desk_success,
            "candles_total": candles_total,
            "survives_pairs": survives_pairs,
        }
    )
    print(
        f"\nQUANT-MAX done · success {desk_success}% · SURVIVES {survives_pairs}/{len(BOOK)} · "
        f"candles_all_tfs={candles_total} · {payload['elapsed_s']}s",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run_quant_max(years=DEFAULT_YEARS)
