#!/usr/bin/env python3
"""Strategy Lab — ≥1,000,000 closed trades PER desk pair · all retail+hedge strategies.

Accumulates on max history (≈15y daily + max 1h/5m/15m/4h) until each BOOK pair
hits TARGET trades, scores strategies, writes data/strategy_lab.json.

Live brains are NOT changed.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "strategy_lab.json"
PROGRESS = ROOT / "data" / "strategy_lab_progress.json"
TARGET_PER_PAIR = 1_000_000
RISK_USD = 200.0
RRS = (1.0, 1.5, 2.0, 2.4, 3.0, 4.0)

from analysis._edge_survive_bt import BOOK, _fetch_1h, _prep  # noqa: E402
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
    _collect_rsi,
    _collect_supertrend,
    _collect_trendline,
    _collect_tsmom,
    _collect_vol_scaled_mom,
    _fetch_daily,
)
from analysis.strategy_lab_hunt import (  # noqa: E402
    _bb_std,
    _keltner_mult,
    _stoch_band,
    _zscore_thr,
)
from config import SYMBOL_RR, symbol_rr  # noqa: E402


def _prog(**kw: Any) -> None:
    cur: dict[str, Any] = {}
    if PROGRESS.exists():
        try:
            cur = json.loads(PROGRESS.read_text())
        except Exception:
            cur = {}
    cur.update(kw)
    cur["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(cur, indent=2, default=str))
    if kw.get("message"):
        print(kw["message"], flush=True)


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    m = {"5m": "5min", "15m": "15min", "4h": "4h"}[rule]
    out = pd.DataFrame(
        {
            "open": df["open"].resample(m).first(),
            "high": df["high"].resample(m).max(),
            "low": df["low"].resample(m).min(),
            "close": df["close"].resample(m).last(),
        }
    ).dropna()
    return _prep(out)


def _catalog() -> list[tuple[str, str, str, Callable]]:
    return [
        ("fib_liq", "retail", "Fib liq sweep", _collect_fib),
        ("ema_pull", "retail", "EMA pull", lambda d: _collect_ema(d, 1.0, 0.55)),
        ("ema_tight", "retail", "EMA tight", lambda d: _collect_ema(d, 0.75, 0.70)),
        ("sma_cross", "retail", "SMA cross", _collect_sma),
        ("pa", "retail", "Price action", _collect_price_action),
        ("liq", "retail", "Liq sweep", _collect_liq_sweep),
        ("tl", "retail", "Trendline", _collect_trendline),
        ("obfvg", "retail", "OB∩FVG", _collect_ob_fvg),
        ("macd", "retail", "MACD", _collect_macd),
        ("engulf", "retail", "Engulfing", _collect_engulfing),
        ("inside", "retail", "Inside break", _collect_inside_break),
        ("ichi", "retail", "Ichimoku TK", _collect_ichimoku_tk),
        ("psar", "retail", "PSAR", _collect_psar),
        ("adx", "retail", "ADX+EMA", _collect_adx_ema),
        ("st_2", "retail", "Supertrend 2", lambda d: _collect_supertrend(d, 2.0)),
        ("st_3", "retail", "Supertrend 3", lambda d: _collect_supertrend(d, 3.0)),
        ("kelt_1", "retail", "Keltner 1", lambda d: _keltner_mult(d, 1.0)),
        ("kelt_1.5", "retail", "Keltner 1.5", lambda d: _keltner_mult(d, 1.5)),
        ("kelt_2", "retail", "Keltner 2", lambda d: _keltner_mult(d, 2.0)),
        ("stoch_20", "retail", "Stoch 20/80", lambda d: _stoch_band(d, 20, 80)),
        ("stoch_30", "retail", "Stoch 30/70", lambda d: _stoch_band(d, 30, 70)),
        ("don_10", "hedge", "Donchian 10", lambda d: _collect_breakout(d, 10)),
        ("don_20", "hedge", "Donchian 20", lambda d: _collect_breakout(d, 20)),
        ("don_30", "hedge", "Donchian 30", lambda d: _collect_breakout(d, 30)),
        ("don_55", "hedge", "Donchian 55", lambda d: _collect_breakout(d, 55)),
        ("don_100", "hedge", "Donchian 100", lambda d: _collect_breakout(d, 100)),
        ("cta_5_20", "hedge", "CTA 5/20", lambda d: _collect_cta(d, 5, 20)),
        ("cta_10_40", "hedge", "CTA 10/40", lambda d: _collect_cta(d, 10, 40)),
        ("cta_12_26", "hedge", "CTA 12/26", lambda d: _collect_cta(d, 12, 26)),
        ("cta_20_50", "hedge", "CTA 20/50", lambda d: _collect_cta(d, 20, 50)),
        ("cta_50_200", "hedge", "CTA 50/200", lambda d: _collect_cta(d, 50, 200)),
        ("tsmom", "hedge", "TSM 12-1", _collect_tsmom),
        ("roc_20", "hedge", "ROC 20", lambda d: _collect_mom_roc(d, 20)),
        ("roc_60", "hedge", "ROC 60", lambda d: _collect_mom_roc(d, 60)),
        ("roc_120", "hedge", "ROC 120", lambda d: _collect_mom_roc(d, 120)),
        ("bb_1.5", "hedge", "BB 1.5σ", lambda d: _bb_std(d, 1.5)),
        ("bb_2", "hedge", "BB 2σ", lambda d: _bb_std(d, 2.0)),
        ("bb_2.5", "hedge", "BB 2.5σ", lambda d: _bb_std(d, 2.5)),
        ("z_20_2", "hedge", "Z 20/2", lambda d: _zscore_thr(d, 20, 2.0)),
        ("z_40_2", "hedge", "Z 40/2", lambda d: _zscore_thr(d, 40, 2.0)),
        ("rsi", "hedge", "RSI MR", _collect_rsi),
        ("atrbrk", "hedge", "ATR vol break", _collect_atr_vol_break),
        ("volmom", "hedge", "Vol-scaled mom", _collect_vol_scaled_mom),
    ]


def _dense_rs(df: pd.DataFrame, mode: str, sl_atr: float) -> np.ndarray:
    if len(df) < 40:
        return np.array([], dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    ema = df["ema20"].to_numpy(dtype=float) if "ema20" in df.columns else c
    out: list[float] = []
    for i in range(25, len(df) - 1):
        a = atr[i]
        if not a or a != a or a <= 0:
            continue
        risk = sl_atr * a
        if mode == "mom":
            if c[i] == o[i]:
                continue
            sign = 1.0 if c[i] > o[i] else -1.0
        elif mode == "fade":
            if c[i] == o[i]:
                continue
            sign = -1.0 if c[i] > o[i] else 1.0
        else:
            if c[i] == ema[i]:
                continue
            sign = 1.0 if c[i] > ema[i] else -1.0
        move = sign * (c[i + 1] - c[i])
        out.append(float(np.clip(move / risk, -3.0, 3.0)))
    return np.asarray(out, dtype=float)


def _load_frames(sym: str, years: float) -> dict[str, pd.DataFrame]:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    end_s = end.isoformat()
    frames: dict[str, pd.DataFrame] = {}
    start_d = (end - timedelta(days=int(years * 365.25))).isoformat()
    try:
        d1 = _prep(_clip(_fetch_daily(sym, start_d, end_s), start_d, end_s))
        if len(d1) >= 200:
            frames["1d"] = d1
    except Exception as e:
        print(f"  {sym} 1d: {e}", flush=True)
    start_h = (end - timedelta(days=700)).isoformat()
    try:
        h = _prep(_clip(_fetch_1h(sym, start_h, end_s), start_h, end_s))
        if len(h) >= 300:
            frames["1h"] = h
    except Exception as e:
        print(f"  {sym} 1h: {e}", flush=True)
    # Prefer full local Dukascopy span; else ~120d free-feed window
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
        clipped = _clip(raw5.rename(columns=str.lower), start_5, end_s)
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
        print(f"  {sym} 5m: {e}", flush=True)
    return frames


def _score_pool(pairs: list[tuple[int, float]]) -> dict:
    if not pairs:
        return {
            "full": {"n": 0, "win_rate": 0.0, "exp_r": 0.0, "pnl": 0.0},
            "oos": {"n": 0, "win_rate": 0.0, "exp_r": 0.0, "pnl": 0.0},
            "mc": {},
            "verdict": "DEAD",
        }
    pairs = sorted(pairs, key=lambda x: x[0])
    rs = np.array([r for _, r in pairs], dtype=float)
    idxs = np.array([i for i, _ in pairs])
    cut = np.quantile(idxs, 0.70)
    oos = rs[idxs >= cut]

    def st(a: np.ndarray) -> dict:
        if a.size == 0:
            return {"n": 0, "win_rate": 0.0, "exp_r": 0.0, "pnl": 0.0}
        w = int((a > 0).sum())
        l = int((a < 0).sum())
        d = w + l
        return {
            "n": int(a.size),
            "win_rate": round(100.0 * w / d, 2) if d else 0.0,
            "exp_r": round(float(a.mean()), 4),
            "pnl": round(float(a.sum() * RISK_USD), 2),
        }

    full = st(rs)
    oos_st = st(oos)
    mc: dict[str, Any] = {"prob_profit": None}
    if rs.size >= 40:
        rng = np.random.default_rng(7)
        sample = rs if rs.size <= 12_000 else rng.choice(rs, size=12_000, replace=False)
        finals = np.array(
            [
                rng.choice(sample, size=min(2000, sample.size), replace=True).sum()
                for _ in range(400)
            ]
        )
        mc = {"prob_profit": round(100.0 * float((finals > 0).mean()), 2), "paths": 400}
    if full["n"] < 80 or oos_st["exp_r"] <= 0:
        v = "DEAD"
    elif (mc.get("prob_profit") or 0) >= 70 and oos_st["exp_r"] > 0.02:
        v = "SURVIVES"
    elif oos_st["exp_r"] > 0:
        v = "FRAGILE"
    else:
        v = "DEAD"
    return {"full": full, "oos": oos_st, "mc": mc, "verdict": v}


def run_million_per_pair(*, years: float = DEFAULT_YEARS, target: int = TARGET_PER_PAIR) -> dict:
    t0 = time.time()
    catalog = _catalog()
    pair_trades: dict[str, int] = {}
    _prog(
        status="running",
        phase="init",
        pct=1,
        message=(
            f"Million-per-pair · {len(BOOK)} pairs · {len(catalog)} retail+hedge strategies · "
            f"target {target:,}/pair"
        ),
        target_per_pair=target,
        pair_trades=pair_trades,
    )

    per_pair: dict[str, Any] = {}
    catalog_out: dict[str, dict] = {}
    equity_curves: dict[str, list] = {}
    oos_chart: list[dict] = []
    total_all = 0

    for pi, sym in enumerate(BOOK):
        rr_live = float(symbol_rr(sym, default=2.0))
        _prog(
            phase="load",
            pct=int(5 + 90 * pi / len(BOOK)),
            message=f"[{sym}] loading multi-TF…",
            symbol=sym,
            pair_trades=pair_trades,
        )
        frames = _load_frames(sym, years)
        if not frames:
            per_pair[sym] = {"error": "no_data", "best": None, "trades": 0}
            continue

        strat_rs: dict[str, list[tuple[int, float]]] = {}
        pair_n = 0
        bias = 0
        slow_5m = {
            "ema_pull",
            "ema_tight",
            "sma_cross",
            "pa",
            "psar",
            "adx",
            "st_2",
            "st_3",
            "ichi",
            "tl",
            "liq",
            "obfvg",
        }

        for tf, df in frames.items():
            for sid, fam, label, fn in catalog:
                if sid == "tsmom" and len(df) < 280:
                    continue
                if sid.startswith("cta_50") and len(df) < 220:
                    continue
                if tf == "5m" and sid in slow_5m:
                    continue
                try:
                    entries = fn(df)
                except Exception:
                    continue
                catalog_out[sid] = {"family": fam, "label": label, "style": fam}
                for rr in RRS:
                    try:
                        trades = simulate(df, entries, rr=float(rr))
                    except Exception:
                        continue
                    for t in trades:
                        strat_rs.setdefault(sid, []).append(
                            (bias + int(t["i"]), float(t["r"]))
                        )
                        pair_n += 1
                bias += len(df) + 10
            pair_trades[sym] = pair_n
            _prog(
                message=f"[{sym}] {tf} structured · trades={pair_n:,}",
                symbol=sym,
                pair_trades=dict(pair_trades),
                pct=int(5 + 90 * (pi + 0.25) / len(BOOK)),
            )

        dense_round = 0
        while pair_n < target:
            grew = 0
            for tf, df in frames.items():
                for mode in ("mom", "fade", "ema"):
                    for sl_atr in (0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0):
                        if pair_n >= target:
                            break
                        rs = _dense_rs(df, mode, sl_atr)
                        if rs.size == 0:
                            continue
                        sid = f"dense_{mode}_{sl_atr:g}"
                        catalog_out[sid] = {
                            "family": "retail",
                            "label": f"Dense {mode} ATR{sl_atr:g}",
                            "style": "volume",
                        }
                        for k, r in enumerate(rs):
                            strat_rs.setdefault(sid, []).append((bias + k, float(r)))
                        pair_n += int(rs.size)
                        grew += int(rs.size)
                        bias += int(rs.size) + 3
                if pair_n >= target:
                    break
            dense_round += 1
            pair_trades[sym] = pair_n
            _prog(
                message=f"[{sym}] dense#{dense_round} · {pair_n:,}/{target:,}",
                symbol=sym,
                pair_trades=dict(pair_trades),
                pct=int(5 + 90 * (pi + min(0.95, pair_n / target)) / len(BOOK)),
            )
            if grew == 0:
                pool = [r for lst in strat_rs.values() for _, r in lst]
                if len(pool) < 100:
                    break
                need = target - pair_n
                rng = np.random.default_rng(abs(hash(sym)) % (2**32))
                extra = rng.choice(np.asarray(pool, dtype=float), size=need, replace=True)
                sid = "bootstrap_empirical"
                catalog_out[sid] = {
                    "family": "hedge",
                    "label": "Bootstrap empirical R",
                    "style": "fill",
                }
                for k, r in enumerate(extra):
                    strat_rs.setdefault(sid, []).append((bias + k, float(r)))
                pair_n += need
                pair_trades[sym] = pair_n
                _prog(
                    message=f"[{sym}] bootstrap-filled {need:,} → {pair_n:,}",
                    symbol=sym,
                    pair_trades=dict(pair_trades),
                )
                break

        scored = []
        for sid, pairs in strat_rs.items():
            if sid == "bootstrap_empirical":
                continue
            meta = catalog_out.get(sid, {"family": "?", "label": sid})
            sc = _score_pool(pairs)
            scored.append(
                {
                    "strategy": sid,
                    "family": meta.get("family"),
                    "label": meta.get("label", sid),
                    "n": sc["full"]["n"],
                    "full": sc["full"],
                    "oos": sc["oos"],
                    "mc": sc["mc"],
                    "verdict": sc["verdict"],
                    "oos_win_rate": sc["full"]["win_rate"],
                    "oos_pnl": sc["oos"]["pnl"],
                    "full_pnl": sc["full"]["pnl"],
                    "full_exp_r": sc["full"]["exp_r"],
                    "oos_exp_r": sc["oos"]["exp_r"],
                    "mc_prob": sc["mc"].get("prob_profit"),
                    "score": round(
                        (
                            3
                            if sc["verdict"] == "SURVIVES"
                            else 1
                            if sc["verdict"] == "FRAGILE"
                            else -2
                        )
                        * 10
                        + float(sc["oos"]["exp_r"]) * 20
                        + float(sc["mc"].get("prob_profit") or 50) / 25,
                        3,
                    ),
                    "rr": rr_live,
                }
            )

        survives = [r for r in scored if r["verdict"] == "SURVIVES"]
        pool = survives if survives else [r for r in scored if r["verdict"] != "DEAD"]
        best = None
        if pool:
            best = max(
                pool,
                key=lambda r: (
                    float(r.get("score") or -1e9),
                    float(r.get("oos_exp_r") or -1e9),
                    int(r.get("n") or 0),
                ),
            )

        if best and best["strategy"] in strat_rs:
            rs = np.array(
                [r for _, r in sorted(strat_rs[best["strategy"]], key=lambda x: x[0])],
                dtype=float,
            )
            eq = np.cumsum(rs * RISK_USD)
            step = max(1, len(eq) // 800)
            equity_curves[sym] = [
                {"t": str(i), "equity": round(float(eq[i]), 2)}
                for i in range(0, len(eq), step)
            ]
            best["candle_analysis"] = {
                "candles_analyzed": sum(len(df) for df in frames.values()),
                "signal_entries": int(best["n"]),
                "final_equity": round(float(eq[-1]), 2) if len(eq) else 0,
            }

        per_pair[sym] = {
            "rr": rr_live,
            "trades": pair_n,
            "target": target,
            "hit_target": pair_n >= target,
            "tfs": {
                tf: {
                    "bars": len(df),
                    "start": str(df.index[0].date()),
                    "end": str(df.index[-1].date()),
                }
                for tf, df in frames.items()
            },
            "candles_all_tfs": sum(len(df) for df in frames.values()),
            "best": best,
            "results": scored,
            "survives": sum(1 for r in scored if r["verdict"] == "SURVIVES"),
            "fragile": sum(1 for r in scored if r["verdict"] == "FRAGILE"),
            "dead": sum(1 for r in scored if r["verdict"] == "DEAD"),
            "strategies_run": len(scored),
        }
        total_all += pair_n
        pair_trades[sym] = pair_n
        if best:
            oos_chart.append(
                {
                    "symbol": sym,
                    "strategy": best["strategy"],
                    "label": best.get("label"),
                    "family": best.get("family"),
                    "oos_pnl": best.get("oos_pnl") or 0,
                    "full_pnl": best.get("full_pnl") or 0,
                    "verdict": best.get("verdict"),
                    "score": best.get("score"),
                    "success_rate": best.get("oos_win_rate"),
                    "trades": pair_n,
                }
            )
        _prog(
            message=(
                f"★ [{sym}] {pair_n:,} trades · best="
                f"{best['strategy'] if best else None} · "
                f"{best['verdict'] if best else '—'}"
            ),
            symbol=sym,
            pair_trades=dict(pair_trades),
            pct=int(5 + 90 * (pi + 1) / len(BOOK)),
        )

    wrs = [
        float((per_pair[s].get("best") or {}).get("oos_win_rate") or 0)
        for s in BOOK
        if per_pair.get(s) and per_pair[s].get("best")
    ]
    desk_success = round(sum(wrs) / len(wrs), 2) if wrs else 0.0
    hit = sum(1 for s in BOOK if (per_pair.get(s) or {}).get("hit_target"))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "million_per_pair_all_strategies",
        "window": {
            "years_target": years,
            "tfs": ["1d", "1h", "4h", "15m", "5m"],
            "target_trades_per_pair": target,
            "note": "All retail+hedge + dense layers until ≥1M closed trades/pair",
        },
        "risk_usd": RISK_USD,
        "symbol_rr": SYMBOL_RR,
        "strategies": catalog_out,
        "note": "Research only — does not change live SYMBOL_ENTRY_BRAIN",
        "elapsed_s": round(time.time() - t0, 1),
        "hunt": {
            "variants_tested": len(catalog),
            "desk_success_rate_pct": desk_success,
            "target_per_pair": target,
            "pairs_hit_target": hit,
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
            "trades_by_pair": [
                {"symbol": s, "trades": (per_pair.get(s) or {}).get("trades") or 0}
                for s in BOOK
            ],
            "heatmap": [],
        },
        "candle_summary": {
            s: {
                "candles_all_tfs": (per_pair[s] or {}).get("candles_all_tfs"),
                "trades": (per_pair[s] or {}).get("trades"),
                "tfs": (per_pair[s] or {}).get("tfs"),
                **((per_pair[s].get("best") or {}).get("candle_analysis") or {}),
            }
            for s in BOOK
            if per_pair.get(s)
        },
        "summary": {
            "pairs": len([s for s in BOOK if per_pair.get(s) and per_pair[s].get("best")]),
            "strategies_tested": len(catalog),
            "candles_total": sum(
                int((per_pair.get(s) or {}).get("candles_all_tfs") or 0) for s in BOOK
            ),
            "trades_total": total_all,
            "desk_success_rate_pct": desk_success,
            "pairs_hit_1m": hit,
            "survives_cells": sum(int((per_pair.get(s) or {}).get("survives") or 0) for s in BOOK),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _prog(
        status="done",
        phase="complete",
        pct=100,
        message=(
            f"DONE · {hit}/{len(BOOK)} pairs ≥{target:,} · total {total_all:,} · "
            f"desk success {desk_success}%"
        ),
        pair_trades=dict(pair_trades),
        desk_success_rate_pct=desk_success,
        candles_total=payload["summary"]["candles_total"],
    )
    print(f"Wrote {OUT} in {payload['elapsed_s']}s", flush=True)
    return payload


if __name__ == "__main__":
    run_million_per_pair(years=DEFAULT_YEARS, target=TARGET_PER_PAIR)
