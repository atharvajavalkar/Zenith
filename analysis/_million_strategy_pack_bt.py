#!/usr/bin/env python3
"""RESEARCH — Run ALL Strategy Lab strategies on the million-trade-scale universe.

Replays the expanded Yahoo universe (same book as _million_trade_bt) through the
full strategy catalog. Aggregates closed trades toward ≥1M, then scores each
strategy (full / OOS / MC). Does NOT change live brains.

Writes data/million_strategy_pack.json
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "million_strategy_pack.json"
PROGRESS = ROOT / "data" / "million_strategy_progress.json"
TARGET = 1_000_000
RISK_USD = 200.0

from analysis._million_trade_bt import UNIVERSE, _atr, _fetch_1h, _fetch_daily  # noqa: E402
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_breakout,
    _collect_ema,
    _collect_fib,
    _collect_sma,
    simulate,
)
from analysis.strategy_lab import (  # noqa: E402
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
    _prep,
)
from analysis.strategy_lab_hunt import (  # noqa: E402
    _bb_std,
    _keltner_mult,
    _stoch_band,
    _zscore_thr,
)


def _prog(msg: str, **kw: Any) -> None:
    payload = {"message": msg, "updated_at": datetime.now(timezone.utc).isoformat(), **kw}
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(payload, indent=2, default=str))
    print(msg, flush=True)


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    a = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    b = datetime.fromisoformat(end).replace(hour=23, minute=59, tzinfo=timezone.utc)
    return df[(df.index >= a) & (df.index <= b)].copy()


def _catalog() -> list[tuple[str, str, Callable[[pd.DataFrame], list]]]:
    """(id, family, collector) — full pack."""
    rows: list[tuple[str, str, Callable]] = [
        ("fib_liq", "retail", _collect_fib),
        ("ema_pull", "retail", lambda d: _collect_ema(d, pull_atr=1.0, disp_atr=0.55)),
        ("ema_tight", "retail", lambda d: _collect_ema(d, pull_atr=0.75, disp_atr=0.70)),
        ("sma_cross", "retail", _collect_sma),
        ("donch_20", "hedge", lambda d: _collect_breakout(d, 20)),
        ("donch_55", "hedge", lambda d: _collect_breakout(d, 55)),
        ("donch_100", "hedge", lambda d: _collect_breakout(d, 100)),
        ("price_action", "retail", _collect_price_action),
        ("liq_sweep", "retail", _collect_liq_sweep),
        ("trendline", "retail", _collect_trendline),
        ("ob_fvg", "retail", _collect_ob_fvg),
        ("macd", "retail", _collect_macd),
        ("engulfing", "retail", _collect_engulfing),
        ("inside_break", "retail", _collect_inside_break),
        ("supertrend_3", "retail", lambda d: _collect_supertrend(d, 3.0)),
        ("supertrend_2", "retail", lambda d: _collect_supertrend(d, 2.0)),
        ("ichimoku_tk", "retail", _collect_ichimoku_tk),
        ("kelt_1.5", "retail", lambda d: _keltner_mult(d, 1.5)),
        ("kelt_2", "retail", lambda d: _keltner_mult(d, 2.0)),
        ("adx_ema", "retail", _collect_adx_ema),
        ("psar", "retail", _collect_psar),
        ("stoch_30", "retail", lambda d: _stoch_band(d, 30, 70)),
        ("cta_10_40", "hedge", lambda d: _collect_cta(d, 10, 40)),
        ("cta_12_26", "hedge", lambda d: _collect_cta(d, 12, 26)),
        ("cta_50_200", "hedge", lambda d: _collect_cta(d, 50, 200)),
        ("turtle_55", "hedge", lambda d: _collect_breakout(d, 55)),
        ("tsmom", "hedge", _collect_tsmom),
        ("roc_60", "hedge", lambda d: _collect_mom_roc(d, 60)),
        ("roc_120", "hedge", lambda d: _collect_mom_roc(d, 120)),
        ("bb_2", "hedge", lambda d: _bb_std(d, 2.0)),
        ("z_40_2", "hedge", lambda d: _zscore_thr(d, 40, 2.0)),
        ("rsi", "hedge", _collect_rsi),
        ("atr_vol_break", "hedge", _collect_atr_vol_break),
        ("vol_scaled_mom", "hedge", _collect_vol_scaled_mom),
    ]
    return rows


def _score_rs(rs: np.ndarray) -> dict:
    n = int(rs.size)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "exp_r": 0.0, "pnl": 0.0}
    wins = int((rs > 0).sum())
    losses = int((rs < 0).sum())
    dec = wins + losses
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / dec, 2) if dec else 0.0,
        "exp_r": round(float(rs.mean()), 4),
        "pnl": round(float(rs.sum() * RISK_USD), 2),
    }


def _mc(rs: np.ndarray, paths: int = 800) -> dict:
    if rs.size < 30:
        return {"prob_profit": None}
    rng = np.random.default_rng(7)
    sample = rs if rs.size <= 20_000 else rng.choice(rs, size=20_000, replace=False)
    finals = np.empty(paths)
    for p in range(paths):
        finals[p] = rng.choice(sample, size=sample.size, replace=True).sum()
    return {
        "prob_profit": round(100.0 * float((finals > 0).mean()), 2),
        "sample_n": int(sample.size),
        "paths": paths,
    }


def _verdict(full: dict, oos: dict, mc: dict) -> str:
    if (full.get("n") or 0) < 40:
        return "DEAD"
    if (oos.get("exp_r") or 0) <= 0:
        return "DEAD"
    if (mc.get("prob_profit") or 0) >= 70 and (oos.get("exp_r") or 0) > 0.02:
        return "SURVIVES"
    if (oos.get("exp_r") or 0) > 0:
        return "FRAGILE"
    return "DEAD"


def main() -> int:
    t0 = time.time()
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_d = (end - timedelta(days=int(15 * 365.25))).isoformat()
    start_h = (end - timedelta(days=720)).isoformat()
    end_s = end.isoformat()
    end_fetch = (end + timedelta(days=1)).isoformat()
    catalog = _catalog()

    _prog(
        f"All strategies × million-scale universe · {len(catalog)} strats · {len(UNIVERSE)} symbols · target {TARGET:,}",
        status="running",
        trades=0,
        pct=0,
    )

    # Preload daily frames
    daily: dict[str, pd.DataFrame] = {}
    hourly: dict[str, pd.DataFrame] = {}
    for i, (name, ticker) in enumerate(UNIVERSE):
        df = _fetch_daily(ticker, start_d, end_fetch)
        if df is not None and len(df) >= 120:
            daily[name] = _prep(_clip(df, start_d, end_s))
        _prog(f"load daily {name} ({i+1}/{len(UNIVERSE)})", pct=int(10 * (i + 1) / len(UNIVERSE)))

    # Per-strategy R pools: list of (bar_index_proxy, r)
    strat_rs: dict[str, list[tuple[int, float]]] = {sid: [] for sid, _, _ in catalog}
    strat_meta = {sid: fam for sid, fam, _ in catalog}
    total = 0
    jobs = 0
    total_jobs = len(catalog) * max(1, len(daily))

    # Daily pass — all strategies
    for name, df in daily.items():
        for sid, fam, fn in catalog:
            jobs += 1
            try:
                entries = fn(df)
                # Fixed RR=2 for apples-to-apples across strategies at scale
                trades = simulate(df, entries, rr=2.0)
                for t in trades:
                    strat_rs[sid].append((int(t["i"]), float(t["r"])))
                    total += 1
            except Exception:
                continue
            if jobs % 25 == 0:
                _prog(
                    f"daily screen {name} · {sid} · trades={total:,}",
                    status="running",
                    trades=total,
                    pct=min(55, 10 + int(45 * jobs / total_jobs)),
                )
        if total >= TARGET:
            break

    # Hourly pass if under target
    if total < TARGET:
        _prog(f"daily yielded {total:,} — loading 1h for volume…", trades=total, pct=56)
        for name, ticker in UNIVERSE:
            if total >= TARGET:
                break
            df = _fetch_1h(ticker, start_h, end_fetch)
            if df is None or len(df) < 200:
                continue
            df = _prep(_clip(df, start_h, end_s))
            hourly[name] = df
            for sid, fam, fn in catalog:
                if total >= TARGET:
                    break
                # Skip slow/heavy on short 1h if needed
                if sid == "tsmom" and len(df) < 280:
                    continue
                try:
                    entries = fn(df)
                    trades = simulate(df, entries, rr=2.0)
                    for t in trades:
                        # offset index so OOS split still chronological-ish
                        strat_rs[sid].append((1_000_000 + int(t["i"]), float(t["r"])))
                        total += 1
                except Exception:
                    continue
            _prog(
                f"1h {name} · trades={total:,}",
                status="running",
                trades=total,
                pct=min(90, 56 + int(34 * total / TARGET)),
            )

    # Score each strategy
    _prog(f"scoring {len(catalog)} strategies on {total:,} trades…", trades=total, pct=92)
    results = []
    for sid, fam, _ in catalog:
        pairs = strat_rs[sid]
        if not pairs:
            results.append(
                {
                    "strategy": sid,
                    "family": fam,
                    "n": 0,
                    "verdict": "DEAD",
                    "full": _score_rs(np.array([])),
                    "oos": _score_rs(np.array([])),
                    "mc": {},
                }
            )
            continue
        pairs.sort(key=lambda x: x[0])
        rs = np.array([r for _, r in pairs], dtype=float)
        idxs = np.array([i for i, _ in pairs])
        cut = np.quantile(idxs, 0.70)
        oos = rs[idxs >= cut]
        full = _score_rs(rs)
        oos_st = _score_rs(oos)
        mc = _mc(rs)
        v = _verdict(full, oos_st, mc)
        results.append(
            {
                "strategy": sid,
                "family": fam,
                "n": int(rs.size),
                "full": full,
                "oos": oos_st,
                "mc": mc,
                "verdict": v,
                "score": round(
                    (3 if v == "SURVIVES" else 1 if v == "FRAGILE" else -2) * 10
                    + float(oos_st.get("exp_r") or 0) * 20
                    + float(mc.get("prob_profit") or 50) / 25
                    + min(5, (full.get("n") or 0) / 5000),
                    3,
                ),
            }
        )

    results.sort(
        key=lambda r: (
            {"SURVIVES": 2, "FRAGILE": 1, "DEAD": 0}[r["verdict"]],
            float(r.get("score") or -1e9),
            float((r.get("oos") or {}).get("exp_r") or -1e9),
        ),
        reverse=True,
    )

    survives = [r for r in results if r["verdict"] == "SURVIVES"]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "achieved_total_trades": total,
        "hit_target": total >= TARGET,
        "symbols_daily": len(daily),
        "symbols_1h": len(hourly),
        "strategies_tested": len(catalog),
        "risk_usd": RISK_USD,
        "rr_fixed": 2.0,
        "note": "All Strategy Lab strategies on million-scale universe. Live brains unchanged.",
        "elapsed_s": round(time.time() - t0, 1),
        "summary": {
            "survives": len(survives),
            "fragile": sum(1 for r in results if r["verdict"] == "FRAGILE"),
            "dead": sum(1 for r in results if r["verdict"] == "DEAD"),
            "top": [r["strategy"] for r in results[:10]],
        },
        "results": results,
        "leaderboard": [
            {
                "rank": i + 1,
                "strategy": r["strategy"],
                "family": r["family"],
                "n": r["n"],
                "oos_exp_r": (r.get("oos") or {}).get("exp_r"),
                "oos_pnl": (r.get("oos") or {}).get("pnl"),
                "win_rate": (r.get("full") or {}).get("win_rate"),
                "mc": (r.get("mc") or {}).get("prob_profit"),
                "verdict": r["verdict"],
                "score": r.get("score"),
            }
            for i, r in enumerate(results[:25])
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _prog(
        f"DONE · {total:,} trades across {len(catalog)} strategies · "
        f"S={payload['summary']['survives']} F={payload['summary']['fragile']} "
        f"D={payload['summary']['dead']} · {payload['elapsed_s']}s",
        status="done",
        trades=total,
        pct=100,
    )
    print("\n### TOP 15\n", flush=True)
    for row in payload["leaderboard"][:15]:
        print(
            f"{row['rank']:2d}. {row['strategy']:16s} n={row['n']:7d} "
            f"OOS_E[R]={row['oos_exp_r']} MC={row['mc']} {row['verdict']}",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
