#!/usr/bin/env python3
"""RESEARCH — ALL strategies on million-scale universe (fast path).

Uses O(n) / light collectors + RR∈{1.5,2,3} stacking to reach ≥1M closed trades,
then full/OOS/MC scoreboard per strategy. Live brains unchanged.
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
OUT = ROOT / "data" / "million_strategy_pack.json"
PROGRESS = ROOT / "data" / "million_strategy_progress.json"
TARGET = 1_000_000
RISK_USD = 200.0
RRS = (1.5, 2.0, 3.0)

from analysis._million_trade_bt import UNIVERSE, _fetch_1h, _fetch_daily  # noqa: E402
from analysis._rr_best_by_pair import simulate  # noqa: E402
from analysis.strategy_lab import (  # noqa: E402
    _collect_atr_vol_break,
    _collect_cta,
    _collect_engulfing,
    _collect_fib,
    _collect_inside_break,
    _collect_macd,
    _collect_mom_roc,
    _collect_rsi,
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
from analysis._rr_best_by_pair import _collect_breakout, _collect_sma  # noqa: E402


def _prog(msg: str, **kw: Any) -> None:
    payload = {"message": msg, "updated_at": datetime.now(timezone.utc).isoformat(), **kw}
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(payload, indent=2, default=str))
    print(msg, flush=True)


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    a = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    b = datetime.fromisoformat(end).replace(hour=23, minute=59, tzinfo=timezone.utc)
    return df[(df.index >= a) & (df.index <= b)].copy()


def _catalog() -> list[tuple[str, str, Callable]]:
    return [
        ("fib_liq", "retail", _collect_fib),
        ("sma_cross", "retail", _collect_sma),
        ("donch_20", "hedge", lambda d: _collect_breakout(d, 20)),
        ("donch_55", "hedge", lambda d: _collect_breakout(d, 55)),
        ("donch_100", "hedge", lambda d: _collect_breakout(d, 100)),
        ("macd", "retail", _collect_macd),
        ("engulfing", "retail", _collect_engulfing),
        ("inside_break", "retail", _collect_inside_break),
        ("kelt_1.5", "retail", lambda d: _keltner_mult(d, 1.5)),
        ("kelt_2", "retail", lambda d: _keltner_mult(d, 2.0)),
        ("stoch_30", "retail", lambda d: _stoch_band(d, 30, 70)),
        ("cta_10_40", "hedge", lambda d: _collect_cta(d, 10, 40)),
        ("cta_12_26", "hedge", lambda d: _collect_cta(d, 12, 26)),
        ("cta_50_200", "hedge", lambda d: _collect_cta(d, 50, 200)),
        ("tsmom", "hedge", _collect_tsmom),
        ("roc_60", "hedge", lambda d: _collect_mom_roc(d, 60)),
        ("roc_120", "hedge", lambda d: _collect_mom_roc(d, 120)),
        ("bb_2", "hedge", lambda d: _bb_std(d, 2.0)),
        ("z_40_2", "hedge", lambda d: _zscore_thr(d, 40, 2.0)),
        ("rsi", "hedge", _collect_rsi),
        ("atr_vol_break", "hedge", _collect_atr_vol_break),
        ("vol_scaled_mom", "hedge", _collect_vol_scaled_mom),
        # denser vector-ish extras for volume
        ("donch_10", "hedge", lambda d: _collect_breakout(d, 10)),
        ("donch_30", "hedge", lambda d: _collect_breakout(d, 30)),
        ("cta_5_20", "hedge", lambda d: _collect_cta(d, 5, 20)),
        ("cta_20_50", "hedge", lambda d: _collect_cta(d, 20, 50)),
        ("roc_20", "hedge", lambda d: _collect_mom_roc(d, 20)),
        ("roc_90", "hedge", lambda d: _collect_mom_roc(d, 90)),
        ("bb_1.5", "hedge", lambda d: _bb_std(d, 1.5)),
        ("bb_2.5", "hedge", lambda d: _bb_std(d, 2.5)),
        ("z_20_2", "hedge", lambda d: _zscore_thr(d, 20, 2.0)),
        ("z_60_2", "hedge", lambda d: _zscore_thr(d, 60, 2.0)),
        ("kelt_1", "retail", lambda d: _keltner_mult(d, 1.0)),
        ("stoch_20", "retail", lambda d: _stoch_band(d, 20, 80)),
    ]


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


def _mc(rs: np.ndarray, paths: int = 600) -> dict:
    if rs.size < 30:
        return {"prob_profit": None}
    rng = np.random.default_rng(7)
    sample = rs if rs.size <= 15_000 else rng.choice(rs, size=15_000, replace=False)
    finals = np.array(
        [rng.choice(sample, size=sample.size, replace=True).sum() for _ in range(paths)]
    )
    return {"prob_profit": round(100.0 * float((finals > 0).mean()), 2), "paths": paths}


def _verdict(full: dict, oos: dict, mc: dict) -> str:
    if (full.get("n") or 0) < 80:
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
        f"FAST all-strategies pack · {len(catalog)} strats · RR={list(RRS)} · target {TARGET:,}",
        status="running",
        trades=0,
        pct=0,
    )

    daily: dict[str, pd.DataFrame] = {}
    for i, (name, ticker) in enumerate(UNIVERSE):
        df = _fetch_daily(ticker, start_d, end_fetch)
        if df is not None and len(df) >= 120:
            daily[name] = _prep(_clip(df, start_d, end_s))
        if (i + 1) % 10 == 0:
            _prog(f"loaded daily {i+1}/{len(UNIVERSE)}", pct=int(8 * (i + 1) / len(UNIVERSE)))

    strat_rs: dict[str, list[tuple[int, float]]] = {sid: [] for sid, _, _ in catalog}
    total = 0

    # Daily × all strategies × RR stack
    for ni, (name, df) in enumerate(daily.items()):
        for sid, fam, fn in catalog:
            try:
                entries = fn(df)
            except Exception:
                continue
            for rr in RRS:
                try:
                    trades = simulate(df, entries, rr=rr)
                except Exception:
                    continue
                for t in trades:
                    # encode rr in index channel lightly
                    strat_rs[sid].append((int(t["i"]) + int(rr * 100000), float(t["r"])))
                    total += 1
        _prog(
            f"daily {name} ({ni+1}/{len(daily)}) · trades={total:,}",
            status="running",
            trades=total,
            pct=min(50, 8 + int(42 * (ni + 1) / max(1, len(daily)))),
        )
        if total >= TARGET:
            break

    # 1h liquid subset if needed
    liquid = [
        "GOLD", "USDCHF", "USDJPY", "AUDUSD", "NAS100", "UKOIL", "BTCUSD", "ETHUSD",
        "EURUSD", "GBPUSD", "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "SPX", "WTI",
        "AAPL", "MSFT", "NVDA", "GLD", "TLT", "SOL", "XRP", "BNB",
    ]
    ticker_map = {n: t for n, t in UNIVERSE}
    if total < TARGET:
        for ni, name in enumerate(liquid):
            if total >= TARGET:
                break
            ticker = ticker_map.get(name)
            if not ticker:
                continue
            raw = _fetch_1h(ticker, start_h, end_fetch)
            if raw is None or len(raw) < 200:
                continue
            df = _prep(_clip(raw, start_h, end_s))
            for sid, fam, fn in catalog:
                if sid == "tsmom" and len(df) < 280:
                    continue
                try:
                    entries = fn(df)
                except Exception:
                    continue
                for rr in RRS:
                    try:
                        trades = simulate(df, entries, rr=rr)
                    except Exception:
                        continue
                    for t in trades:
                        strat_rs[sid].append((2_000_000 + int(t["i"]), float(t["r"])))
                        total += 1
            _prog(
                f"1h {name} · trades={total:,}",
                status="running",
                trades=total,
                pct=min(90, 50 + int(40 * total / TARGET)),
            )

    _prog(f"scoring {len(catalog)} strategies on {total:,} trades…", trades=total, pct=93)
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
                    "score": -99,
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
                    + min(5, (full.get("n") or 0) / 10000),
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

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "achieved_total_trades": total,
        "hit_target": total >= TARGET,
        "strategies_tested": len(catalog),
        "symbols_daily": len(daily),
        "rrs": list(RRS),
        "risk_usd": RISK_USD,
        "note": "All strategies on million-scale universe (fast collectors + RR stack). Live unchanged.",
        "elapsed_s": round(time.time() - t0, 1),
        "summary": {
            "survives": sum(1 for r in results if r["verdict"] == "SURVIVES"),
            "fragile": sum(1 for r in results if r["verdict"] == "FRAGILE"),
            "dead": sum(1 for r in results if r["verdict"] == "DEAD"),
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
            for i, r in enumerate(results)
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _prog(
        f"DONE · {total:,} trades · {len(catalog)} strategies · "
        f"S={payload['summary']['survives']} F={payload['summary']['fragile']} "
        f"D={payload['summary']['dead']} · {payload['elapsed_s']}s",
        status="done",
        trades=total,
        pct=100,
    )
    print("\n### LEADERBOARD\n", flush=True)
    for row in payload["leaderboard"][:20]:
        print(
            f"{row['rank']:2d}. {row['strategy']:16s} n={row['n']:8d} "
            f"OOS_E[R]={row['oos_exp_r']} WR={row['win_rate']} MC={row['mc']} {row['verdict']}",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
