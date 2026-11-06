#!/usr/bin/env python3
"""RESEARCH ONLY — GOLD Fib + liq-sweep multi-timeframe grid (last ~1 month).

Impulse TF × Entry TF combinations. Walk candles forward chronologically.
No live wiring.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = Path("/tmp/gold_fib_tf_grid_050_618.json")
RISK_USD = 200.0
SWING = 3
# Shallow pocket: 50%–61.8% retracement (user test)
FIBS = (0.50, 0.618)
ZONE_ATR_MULT = 0.40
SESS = (7, 16)  # UTC
# Impulse TF → entry TF candidates
COMBOS = [
    ("15m", "5m"),
    ("15m", "15m"),
    ("1h", "5m"),
    ("1h", "15m"),
    ("1h", "1h"),
    ("4h", "5m"),
    ("4h", "15m"),
    ("4h", "1h"),
    ("4h", "4h"),
]


def _fetch_5m(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    warm = (datetime.fromisoformat(start) - timedelta(days=14)).strftime("%Y-%m-%d")
    end_p = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.Ticker("GC=F").history(start=warm, end=end_p, interval="5m", auto_adjust=True)
    if df is None or df.empty:
        from data.historical import fetch_symbol_ohlcv

        df = fetch_symbol_ohlcv("GOLD", warm, end_p, interval="5m")
    if df is None or df.empty:
        raise RuntimeError("no GOLD/GC 5m")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _resample(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    m = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}[rule]
    o = df5["open"].resample(m).first()
    h = df5["high"].resample(m).max()
    l = df5["low"].resample(m).min()
    c = df5["close"].resample(m).last()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    return out


def _is_swing_low(df: pd.DataFrame, i: int, n: int = SWING) -> bool:
    if i < n or i + n >= len(df):
        return False
    lo = float(df.iloc[i]["low"])
    return all(lo <= float(df.iloc[i + k]["low"]) for k in range(-n, n + 1) if k != 0)


def _is_swing_high(df: pd.DataFrame, i: int, n: int = SWING) -> bool:
    if i < n or i + n >= len(df):
        return False
    hi = float(df.iloc[i]["high"])
    return all(hi >= float(df.iloc[i + k]["high"]) for k in range(-n, n + 1) if k != 0)


def _impulse_at(df: pd.DataFrame, i: int) -> Optional[dict[str, Any]]:
    atr = float(df.iloc[i]["atr"] or 0) or 1.0
    look = range(max(SWING, i - 60), i - SWING)
    lows = [(j, float(df.iloc[j]["low"])) for j in look if _is_swing_low(df, j)]
    highs = [(j, float(df.iloc[j]["high"])) for j in look if _is_swing_high(df, j)]
    if not lows or not highs:
        return None
    cands = []
    for lj, lv in reversed(lows):
        later = [(hj, hv) for hj, hv in highs if hj > lj]
        if not later:
            continue
        hj, hv = max(later, key=lambda x: x[1])
        if hv - lv >= 1.15 * atr:
            cands.append({"side": "BUY", "a": lv, "b": hv, "bi": hj, "end_t": df.index[hj]})
            break
    for hj, hv in reversed(highs):
        later = [(lj, lv) for lj, lv in lows if lj > hj]
        if not later:
            continue
        lj, lv = min(later, key=lambda x: x[1])
        if hv - lv >= 1.15 * atr:
            cands.append({"side": "SELL", "a": hv, "b": lv, "bi": lj, "end_t": df.index[lj]})
            break
    if not cands:
        return None
    return max(cands, key=lambda x: x["bi"])


def _fib_zone(imp: dict[str, Any], atr: float) -> tuple[float, float, list[float]]:
    a, b = float(imp["a"]), float(imp["b"])
    if imp["side"] == "BUY":
        levels = [b - r * (b - a) for r in FIBS]
    else:
        levels = [b + r * (a - b) for r in FIBS]
    pad = ZONE_ATR_MULT * atr
    return min(levels) - pad, max(levels) + pad, levels


def run_combo(
    df_imp: pd.DataFrame,
    df_ent: pd.DataFrame,
    start: str,
    end: str,
    *,
    impulse_tf: str,
    entry_tf: str,
    session_filter: bool,
) -> dict[str, Any]:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    trades: list[dict] = []
    open_t = None
    cooldown = df_ent.index[0]
    # Precompute impulse on each completed impulse bar, map to entry timeline
    # At each entry bar, use last completed impulse bar (impulse index <= entry time)
    imp_cache: list[Optional[dict]] = [None] * len(df_imp)
    for i in range(40, len(df_imp)):
        imp_cache[i] = _impulse_at(df_imp, i)

    ent_times = df_ent.index
    # pointer into impulse df
    for ei in range(40, len(df_ent)):
        ts = ent_times[ei]
        row = df_ent.iloc[ei]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        atr = float(row["atr"] or 0) or max(h - l, 1e-6)

        if open_t is not None:
            side = open_t["side"]
            hit_sl = (side == "BUY" and l <= open_t["sl"]) or (
                side == "SELL" and h >= open_t["sl"]
            )
            hit_tp = (side == "BUY" and h >= open_t["tp"]) or (
                side == "SELL" and l <= open_t["tp"]
            )
            if hit_sl or hit_tp:
                exit_px = open_t["sl"] if hit_sl else open_t["tp"]
                risk = abs(open_t["entry"] - open_t["sl"])
                r_mult = (
                    (exit_px - open_t["entry"]) / risk
                    if side == "BUY"
                    else (open_t["entry"] - exit_px) / risk
                )
                if start_dt <= open_t["entry_t"] <= end_dt:
                    trades.append(
                        {
                            "entry_t": str(open_t["entry_t"])[:16],
                            "side": side,
                            "pnl": round(float(r_mult) * RISK_USD, 2),
                            "reason": "sl" if hit_sl else "tp_2r",
                        }
                    )
                open_t = None
                cooldown = ts + timedelta(minutes=30 if entry_tf in ("5m", "15m") else 120)
            continue

        if ts < start_dt or ts > end_dt or ts < cooldown:
            continue
        if session_filter and not (SESS[0] <= ts.hour < SESS[1]):
            continue

        # last completed impulse bar at or before ts
        # find rightmost impulse index with index time <= ts
        # binary-ish via searchsorted
        pos = df_imp.index.searchsorted(ts, side="right") - 1
        if pos < 40:
            continue
        imp = imp_cache[pos]
        if not imp:
            continue
        # Impulse must be "fresh" — ended within last N entry bars
        if ts - imp["end_t"] > timedelta(hours=48 if impulse_tf == "4h" else 24):
            continue

        zlo, zhi, _ = _fib_zone(imp, atr)
        win = df_ent.iloc[max(0, ei - 20) : ei]
        if imp["side"] == "BUY":
            liq = float(win["low"].min()) if len(win) else l
            swept = l < min(liq, zlo)
            reclaim = c > zlo and l <= zhi
            if swept and reclaim and c > float(row["open"]):
                entry, sl = c, min(l, liq) - 0.08 * atr
                risk = entry - sl
                if risk > 0.12 * atr:
                    open_t = {
                        "side": "BUY",
                        "entry": entry,
                        "sl": sl,
                        "tp": entry + 2.0 * risk,
                        "entry_t": ts,
                    }
        else:
            liq = float(win["high"].max()) if len(win) else h
            swept = h > max(liq, zhi)
            reclaim = c < zhi and h >= zlo
            if swept and reclaim and c < float(row["open"]):
                entry, sl = c, max(h, liq) + 0.08 * atr
                risk = sl - entry
                if risk > 0.12 * atr:
                    open_t = {
                        "side": "SELL",
                        "entry": entry,
                        "sl": sl,
                        "tp": entry - 2.0 * risk,
                        "entry_t": ts,
                    }

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    dec = wins + losses
    pnl = round(sum(t["pnl"] for t in trades), 2)
    return {
        "impulse_tf": impulse_tf,
        "entry_tf": entry_tf,
        "session_filter": session_filter,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / dec, 1) if dec else 0.0,
        "pnl": pnl,
        "avg_pnl": round(pnl / len(trades), 2) if trades else 0.0,
        "expectancy_r": round((pnl / RISK_USD) / len(trades), 3) if trades else 0.0,
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=29)).isoformat()
    end_s = end.isoformat()
    print(f"GOLD Fib TF grid {start} → {end_s}", flush=True)
    df5 = _fetch_5m(start, end_s)
    frames = {
        "5m": _resample(df5, "5m"),
        "15m": _resample(df5, "15m"),
        "1h": _resample(df5, "1h"),
        "4h": _resample(df5, "4h"),
    }
    for k, v in frames.items():
        print(f"  {k}: {len(v)} bars", flush=True)

    rows: list[dict] = []
    for imp_tf, ent_tf in COMBOS:
        for sess in (True, False):
            r = run_combo(
                frames[imp_tf],
                frames[ent_tf],
                start,
                end_s,
                impulse_tf=imp_tf,
                entry_tf=ent_tf,
                session_filter=sess,
            )
            rows.append(r)
            tag = "LN/NY" if sess else "24h"
            print(
                f"  {imp_tf}/{ent_tf} [{tag}] n={r['trades']} wr={r['win_rate']}% pnl=${r['pnl']}",
                flush=True,
            )

    # Rank by PnL then WR (min 5 trades for "best")
    ranked = sorted(rows, key=lambda x: (x["pnl"], x["win_rate"], x["trades"]), reverse=True)
    eligible = [r for r in ranked if r["trades"] >= 5]

    payload = {
        "symbol": "GC=F / GOLD",
        "window": {"start": start, "end": end_s},
        "risk_usd": RISK_USD,
        "rules": "Fib 50–61.8 of impulse + sweep reclaim; SL beyond wick; TP 2R",
        "results": rows,
        "best_overall": ranked[:5],
        "best_min5_trades": eligible[:5],
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print("\n### GOLD Fib+Sweep TF grid — 0.50–0.618 — last 1 month ($200/R)\n", flush=True)
    print(
        "| Impulse TF | Entry TF | Session | Trades | WR | PnL | Avg/trade | E[R] |",
        flush=True,
    )
    print("|---|---|---|---:|---:|---:|---:|---:|", flush=True)
    for r in sorted(rows, key=lambda x: (-x["pnl"], -x["win_rate"])):
        sess = "LN/NY" if r["session_filter"] else "24h"
        print(
            f"| {r['impulse_tf']} | {r['entry_tf']} | {sess} | {r['trades']} | "
            f"{r['win_rate']}% | ${r['pnl']} | ${r['avg_pnl']} | {r['expectancy_r']} |",
            flush=True,
        )

    print("\n**Top (min 5 trades)**", flush=True)
    for i, r in enumerate(eligible[:5], 1):
        sess = "LN/NY" if r["session_filter"] else "24h"
        print(
            f"{i}. {r['impulse_tf']} impulse → {r['entry_tf']} entry [{sess}] · "
            f"n={r['trades']} WR={r['win_rate']}% PnL=${r['pnl']}",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
