#!/usr/bin/env python3
"""RESEARCH ONLY — Schwab-style daily multi-lookback close tally.

Rule (daily closes, Dukascopy local hist):
  Compare today's close vs closes 7d / 14d / 21d / 42d ago (trading bars):
    up → +1, down → -1, flat → 0
  Tally = sum of four votes ∈ roughly {-4…+4}
  Size / side (strict as specified):
    +4 → long 100%   |  +2 → long 50%
     0 → sit out
    -2 → short 50%   |  -4 → short 100%
    ±1 / ±3 → sit (unspecified by rule)

Execution: signal at day-i close → hold day i→i+1 close-to-close.
Risk unit: 1R = ATR(14) at signal bar. $200/R. XM session spread on flips.

Horizons: 6m, 1y, 2y, 3y, 5y, full(~10y).
Battery: full, OOS70, WF×4, torture spread, torture skip + MC (8 tests).

Does NOT touch Nebula / live brains / .env.
"""
from __future__ import annotations

import json
import random
import sys
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
from data.hist_local import clear_hist_cache, load_local_ohlcv, local_span  # noqa: E402

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

# trading-day lookbacks ≈ 1w / 2w / 1m / 2m
LOOKBACKS = (7, 14, 21, 42)
HORIZONS_DAYS = {
    "6m": 183,
    "1y": 365,
    "2y": 730,
    "3y": 1095,
    "5y": 1825,
    "full": None,
}

OUT = Path(ROOT) / "data" / "lab_schwab_daily_tally.json"
LOG = Path(ROOT) / "data" / "lab_schwab_daily_tally.log"
MD = Path(ROOT) / "data" / "lab_schwab_daily_tally.md"


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _load_daily(symbol: str) -> Optional[pd.DataFrame]:
    """Resample Dukascopy 15m (fallback 5m/1m) → daily UTC."""
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
        if len(daily) >= 80:
            return daily
    return None


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(5, n // 2)).mean()


def _size_for_tally(tally: int) -> float:
    a = abs(int(tally))
    if a == 4:
        return 1.0
    if a == 2:
        return 0.5
    return 0.0  # 0, ±1, ±3 sit


def _build_trades(df: pd.DataFrame, symbol: str) -> list[dict]:
    """Close-to-close daily holds from Schwab tally signals."""
    c = df["close"].astype(float)
    atr = _atr(df)
    votes = []
    for lb in LOOKBACKS:
        prev = c.shift(lb)
        v = np.where(c > prev, 1, np.where(c < prev, -1, 0))
        votes.append(v)
    tally = np.sum(np.vstack(votes), axis=0).astype(int)

    trades: list[dict] = []
    prev_side = 0
    for i in range(len(df) - 1):
        if i < max(LOOKBACKS) + 2:
            continue
        a = float(atr.iloc[i])
        if not np.isfinite(a) or a <= 0:
            continue
        t = int(tally[i])
        size = _size_for_tally(t)
        side = 0 if size <= 0 else (1 if t > 0 else -1)
        # hold next bar close-to-close
        move = float(c.iloc[i + 1] - c.iloc[i])
        r = size * side * (move / a) if side != 0 else 0.0
        # XM spread on flip / new entry (price → R)
        if side != 0 and side != prev_side:
            try:
                hour = float(df.index[i].hour) if hasattr(df.index[i], "hour") else 12.0
                sp = float(spread_price(symbol, hour_utc=hour))
                r -= size * (sp / a)
            except Exception:
                r -= 0.05 * size
        if side == 0:
            prev_side = 0
            continue
        pnl = r * RISK_USD
        trades.append(
            {
                "i": i,
                "t": str(df.index[i])[:10],
                "tally": t,
                "size": size,
                "side": side,
                "r": round(float(r), 4),
                "pnl": round(float(pnl), 2),
                "atr": round(a, 6),
            }
        )
        prev_side = side
    return trades


def _battery_from_trades(full_tr: list[dict], n_bars: int) -> dict[str, Any]:
    """Standard desk 8-test battery + MC."""
    if not full_tr or n_bars < 80:
        empty = _stats([])
        return {
            "bars": n_bars,
            "entries": 0,
            "full": empty,
            "oos": empty,
            "monte_carlo": {},
            "verdict": "DEAD",
            "wf_green": 0,
            "wf_n": 0,
            "torture": {},
            "tests": {},
        }

    def by_range(start_i: int = 0, end_i: Optional[int] = None) -> list[dict]:
        end_i = n_bars - 1 if end_i is None else end_i
        return [t for t in full_tr if start_i <= int(t["i"]) <= end_i]

    full = _stats(full_tr)
    split = int(n_bars * 0.70)
    oos = _stats(by_range(start_i=split))

    wf = []
    wf_details: dict[str, Any] = {}
    for k in range(4):
        ts = int(n_bars * (0.40 + 0.15 * k))
        te = int(n_bars * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 60:
            continue
        fold_oos = _stats(by_range(start_i=ts, end_i=te))
        wf.append({"fold": k + 1, "oos": fold_oos})
        wf_details[f"wf_fold_{k+1}"] = fold_oos

    spread_tr = [
        {
            **t,
            "r": round(float(t["r"]) - 0.15, 4),
            "pnl": round((float(t["r"]) - 0.15) * RISK_USD, 2),
        }
        for t in full_tr
    ]
    rng = random.Random(7)
    skip_tr = [t for t in full_tr if rng.random() >= 0.30]
    torture = {
        "spread_0.15R": _stats(spread_tr),
        "skip_30pct": _stats(skip_tr),
    }
    mc = monte_carlo(full_tr)
    v = verdict(full, oos, wf, torture, mc)
    tests = {
        "1_full": full,
        "2_oos70": oos,
        "3_wf_fold_1": wf_details.get("wf_fold_1") or {"n": 0, "exp_r": 0, "pnl": 0},
        "4_wf_fold_2": wf_details.get("wf_fold_2") or {"n": 0, "exp_r": 0, "pnl": 0},
        "5_wf_fold_3": wf_details.get("wf_fold_3") or {"n": 0, "exp_r": 0, "pnl": 0},
        "6_wf_fold_4": wf_details.get("wf_fold_4") or {"n": 0, "exp_r": 0, "pnl": 0},
        "7_torture_spread": torture["spread_0.15R"],
        "8_torture_skip": torture["skip_30pct"],
        "mc_prob_profit": mc.get("prob_profit"),
    }
    return {
        "bars": n_bars,
        "entries": len(full_tr),
        "full": full,
        "oos": oos,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": sum(1 for f in wf if f["oos"]["exp_r"] > 0),
        "wf_n": len(wf),
        "torture": {
            k: {"n": x["n"], "pnl": x["pnl"], "exp_r": x["exp_r"]} for k, x in torture.items()
        },
        "tests": tests,
    }


def _trades_for_horizon(daily: pd.DataFrame, symbol: str, days: Optional[int]) -> tuple[list[dict], int]:
    """Build trades with 90d warmup; keep signals inside the horizon window."""
    if days is None:
        trades = _build_trades(daily, symbol)
        return trades, len(daily)
    end = daily.index[-1]
    cut = end - pd.Timedelta(days=int(days))
    warm = cut - pd.Timedelta(days=90)
    core = daily[daily.index >= warm]
    raw = _build_trades(core, symbol)
    cut_ts = pd.Timestamp(cut)
    if cut_ts.tzinfo is None:
        cut_ts = cut_ts.tz_localize("UTC")
    else:
        cut_ts = cut_ts.tz_convert("UTC")
    trades: list[dict] = []
    for t in raw:
        tt = pd.Timestamp(t["t"])
        if tt.tzinfo is None:
            tt = tt.tz_localize("UTC")
        else:
            tt = tt.tz_convert("UTC")
        if tt >= cut_ts:
            trades.append(t)
    return trades, len(core)


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    clear_hist_cache()
    _log("=== Schwab daily tally · Dukascopy · RESEARCH (Nebula untouched) ===")
    _log(
        f"lookbacks={LOOKBACKS} · size: ±4→100% ±2→50% else sit · "
        f"$/{RISK_USD}/R · horizons={list(HORIZONS_DAYS)} · book={BOOK}"
    )
    _log(f"started {datetime.now(timezone.utc).isoformat()}")

    results: dict[str, Any] = {
        "meta": {
            "rule": "daily close tally 7/14/21/42 · ±4 full ±2 half · 0/odds sit",
            "source": "dukascopy_local_hist_resampled_1D",
            "risk_usd": RISK_USD,
            "lookbacks": list(LOOKBACKS),
            "nebula": "untouched",
            "live": "untouched",
            "started": datetime.now(timezone.utc).isoformat(),
        },
        "symbols": {},
    }

    for sym in BOOK:
        span = local_span(sym, "15m")
        _log(f"\n--- {sym} hist={span} ---")
        daily = _load_daily(sym)
        if daily is None or len(daily) < 120:
            _log(f"  SKIP no daily bars")
            results["symbols"][sym] = {"error": "no_daily"}
            continue
        _log(
            f"  daily bars={len(daily)} {str(daily.index[0])[:10]}→{str(daily.index[-1])[:10]}"
        )
        sym_out: dict[str, Any] = {"daily_bars": len(daily), "horizons": {}}
        for hname, days in HORIZONS_DAYS.items():
            trades, n_bars = _trades_for_horizon(daily, sym, days)
            bat = _battery_from_trades(trades, n_bars)
            tallies = {}
            for t in trades:
                k = str(t["tally"])
                tallies[k] = tallies.get(k, 0) + 1
            bat["tally_hist"] = tallies
            bat["horizon"] = hname
            bat["n_trades"] = len(trades)
            sym_out["horizons"][hname] = bat
            f = bat["full"]
            _log(
                f"  {hname:5} n={f['n']:4} WR={f['wr']:5}% E[R]={f['exp_r']:+.3f} "
                f"pnl=${f['pnl']:+.0f} DD=${f['max_dd']:.0f} "
                f"OOS_E[R]={bat['oos']['exp_r']:+.3f} WF={bat['wf_green']}/{bat['wf_n']} "
                f"MC={bat['monte_carlo'].get('prob_profit')}% → {bat['verdict']}"
            )
        results["symbols"][sym] = sym_out

    results["meta"]["finished"] = datetime.now(timezone.utc).isoformat()

    # Worth-more research score
    survives = 0
    fragile = 0
    dead = 0
    pos_1y = 0
    for sym, so in results["symbols"].items():
        if "error" in so:
            dead += 1
            continue
        h1 = (so.get("horizons") or {}).get("1y") or {}
        v = h1.get("verdict")
        if v == "SURVIVES":
            survives += 1
        elif v == "FRAGILE":
            fragile += 1
        else:
            dead += 1
        if (h1.get("full") or {}).get("exp_r", 0) > 0:
            pos_1y += 1

    recommendation = "SKIP"
    note = "Mostly DEAD / negative E[R] on 1y — not worth deeper research."
    if survives >= 3:
        recommendation = "RESEARCH_MORE"
        note = "≥3 pairs SURVIVES on 1y — worth densifying lookbacks/sizing."
    elif survives + fragile >= 5 and pos_1y >= 5:
        recommendation = "SOFT_INTEREST"
        note = "Several FRAGILE/positive 1y — optional light densify, not priority."
    elif pos_1y >= 4:
        recommendation = "WEAK"
        note = "Some positive 1y E[R] but battery weak — low priority."

    results["summary"] = {
        "verdict_1y_counts": {
            "SURVIVES": survives,
            "FRAGILE": fragile,
            "DEAD": dead,
        },
        "pairs_positive_exp_r_1y": pos_1y,
        "recommendation": recommendation,
        "note": note,
    }
    _log(f"\n=== SUMMARY 1y === {results['summary']}")

    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_md(results)
    _log(f"wrote {OUT}")
    _log(f"wrote {MD}")


def _write_md(results: dict[str, Any]) -> None:
    lines = [
        "# Schwab daily tally — Dukascopy research",
        "",
        f"- Rule: closes vs 7/14/21/42d · **±4 full / ±2 half / else sit**",
        f"- Risk: ${RISK_USD}/R · XM spread on flips · Nebula/live untouched",
        f"- Recommendation: **{results['summary']['recommendation']}** — {results['summary']['note']}",
        "",
        "## 1y battery (primary)",
        "",
        "| Pair | n | WR% | E[R] | PnL$ | OOS E[R] | WF | MC% | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sym in BOOK:
        so = results["symbols"].get(sym) or {}
        h = (so.get("horizons") or {}).get("1y") or {}
        f = h.get("full") or {}
        if not f:
            lines.append(f"| {sym} | — | — | — | — | — | — | — | ERR |")
            continue
        lines.append(
            f"| {sym} | {f.get('n')} | {f.get('wr')} | {f.get('exp_r')} | "
            f"{f.get('pnl')} | {(h.get('oos') or {}).get('exp_r')} | "
            f"{h.get('wf_green')}/{h.get('wf_n')} | "
            f"{(h.get('monte_carlo') or {}).get('prob_profit')} | {h.get('verdict')} |"
        )
    lines += ["", "## All horizons (E[R] / verdict)", ""]
    for sym in BOOK:
        so = results["symbols"].get(sym) or {}
        lines.append(f"### {sym}")
        for hname in HORIZONS_DAYS:
            h = (so.get("horizons") or {}).get(hname) or {}
            f = h.get("full") or {}
            if not f:
                continue
            lines.append(
                f"- **{hname}**: n={f.get('n')} E[R]={f.get('exp_r')} "
                f"pnl=${f.get('pnl')} → {h.get('verdict')}"
            )
        lines.append("")
    MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
