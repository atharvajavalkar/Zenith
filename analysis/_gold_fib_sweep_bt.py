#!/usr/bin/env python3
"""RESEARCH ONLY — GOLD Fib retracement + liquidity sweep (no live wiring).

Idea (common prop/ICT hybrid):
  1. Find impulse swing (recent swing low→high for bull, high→low for bear).
  2. Mark Fib 0.618 / 0.705 / 0.786 of that impulse.
  3. Wait for a liquidity sweep of a nearby equal-low/high or the Fib zone extreme.
  4. Enter on reclaim (close back through Fib); SL beyond sweep; TP 2R (or Fib 0).

Windows: 14d · 1m. Risk $200/R. Session 07–16 UTC.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = Path("/tmp/gold_fib_sweep_bt.json")
RISK_USD = 200.0
SWING = 5
FIBS = (0.618, 0.705, 0.786)
ZONE_ATR = 0.35
SESS_START, SESS_END = 7, 16


def _fetch(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    warm = (datetime.fromisoformat(start) - timedelta(days=10)).strftime("%Y-%m-%d")
    end_p = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.Ticker("GC=F").history(start=warm, end=end_p, interval="5m", auto_adjust=True)
    if df is None or df.empty:
        from data.historical import fetch_symbol_ohlcv

        df = fetch_symbol_ohlcv("GOLD", warm, end_p, interval="5m")
    if df is None or df.empty:
        raise RuntimeError("no data")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


def _swing_low(df: pd.DataFrame, i: int) -> bool:
    if i < SWING or i + SWING >= len(df):
        return False
    lo = float(df.iloc[i]["low"])
    return all(lo <= float(df.iloc[i + k]["low"]) for k in range(-SWING, SWING + 1) if k)

def _swing_high(df: pd.DataFrame, i: int) -> bool:
    if i < SWING or i + SWING >= len(df):
        return False
    hi = float(df.iloc[i]["high"])
    return all(hi >= float(df.iloc[i + k]["high"]) for k in range(-SWING, SWING + 1) if k)


def _last_impulse(df: pd.DataFrame, i: int) -> Optional[dict[str, Any]]:
    """Most recent clean impulse ending before i."""
    look = range(max(SWING, i - 80), i - SWING)
    lows = [(j, float(df.iloc[j]["low"])) for j in look if _swing_low(df, j)]
    highs = [(j, float(df.iloc[j]["high"])) for j in look if _swing_high(df, j)]
    if len(lows) < 1 or len(highs) < 1:
        return None
    # Bull impulse: swing low then later swing high
    bull = None
    for lj, lv in reversed(lows):
        later_h = [(hj, hv) for hj, hv in highs if hj > lj]
        if not later_h:
            continue
        hj, hv = max(later_h, key=lambda x: x[1])
        if hv - lv >= 1.2 * float(df.iloc[i]["atr"] or 1):
            bull = {"side": "BUY", "a": lv, "b": hv, "ai": lj, "bi": hj}
            break
    bear = None
    for hj, hv in reversed(highs):
        later_l = [(lj, lv) for lj, lv in lows if lj > hj]
        if not later_l:
            continue
        lj, lv = min(later_l, key=lambda x: x[1])
        if hv - lv >= 1.2 * float(df.iloc[i]["atr"] or 1):
            bear = {"side": "SELL", "a": hv, "b": lv, "ai": hj, "bi": lj}
            break
    # Prefer most recent impulse end
    cands = [x for x in (bull, bear) if x]
    if not cands:
        return None
    return max(cands, key=lambda x: x["bi"])


def _fib_levels(imp: dict[str, Any]) -> list[float]:
    a, b = float(imp["a"]), float(imp["b"])
    if imp["side"] == "BUY":
        # retrace down from high b toward low a
        return [b - r * (b - a) for r in FIBS]
    return [b + r * (a - b) for r in FIBS]  # retrace up from low b toward high a


def backtest(df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    trades: list[dict] = []
    open_t = None
    cooldown = df.index[0]

    for i in range(90, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
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
                            "pnl": round(r_mult * RISK_USD, 2),
                            "reason": "sl" if hit_sl else "tp_2r",
                            "fib": open_t.get("fib"),
                        }
                    )
                open_t = None
                cooldown = ts + timedelta(minutes=40)
            continue

        if ts < start_dt or ts > end_dt or ts < cooldown:
            continue
        if not (SESS_START <= ts.hour < SESS_END):
            continue

        imp = _last_impulse(df, i)
        if not imp:
            continue
        levels = _fib_levels(imp)
        zone_lo, zone_hi = min(levels) - ZONE_ATR * atr, max(levels) + ZONE_ATR * atr

        # Prior equal extreme in last 24 bars near fib zone (liquidity)
        window = df.iloc[max(0, i - 24) : i]
        if imp["side"] == "BUY":
            liq = float(window["low"].min())
            # Sweep below fib zone / equal low, reclaim into zone
            swept = l < liq or l < zone_lo
            reclaim = c > zone_lo and c <= zone_hi + atr
            in_zone_touch = l <= zone_hi and h >= zone_lo
            if swept and reclaim and in_zone_touch and c > float(row["open"]):
                entry = c
                sl = min(l, liq) - 0.1 * atr
                risk = entry - sl
                if risk > 0.15 * atr:
                    open_t = {
                        "side": "BUY",
                        "entry": entry,
                        "sl": sl,
                        "tp": entry + 2.0 * risk,
                        "entry_t": ts,
                        "fib": round(min(levels), 2),
                    }
        else:
            liq = float(window["high"].max())
            swept = h > liq or h > zone_hi
            reclaim = c < zone_hi and c >= zone_lo - atr
            in_zone_touch = l <= zone_hi and h >= zone_lo
            if swept and reclaim and in_zone_touch and c < float(row["open"]):
                entry = c
                sl = max(h, liq) + 0.1 * atr
                risk = sl - entry
                if risk > 0.15 * atr:
                    open_t = {
                        "side": "SELL",
                        "entry": entry,
                        "sl": sl,
                        "tp": entry - 2.0 * risk,
                        "entry_t": ts,
                        "fib": round(max(levels), 2),
                    }

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    dec = wins + losses
    pnl = round(sum(t["pnl"] for t in trades), 2)
    return {
        "start": start,
        "end": end,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / dec, 1) if dec else 0.0,
        "pnl": pnl,
        "avg_pnl": round(pnl / len(trades), 2) if trades else 0.0,
        "trade_list": trades[:10],
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    windows = [
        ("14d", (end - timedelta(days=13)).isoformat(), end.isoformat()),
        ("1m", (end - timedelta(days=29)).isoformat(), end.isoformat()),
    ]
    df = _fetch(windows[1][1], windows[1][2])
    print(f"Fib+Sweep GOLD bars={len(df)}", flush=True)
    results = {}
    print("\n| Strategy | Window | Trades | Win rate | PnL | Avg/trade |", flush=True)
    print("|---|---|---:|---:|---:|---:|", flush=True)
    for label, a, b in windows:
        r = backtest(df, a, b)
        results[label] = r
        print(
            f"| Fib + Liq Sweep | {label} | {r['trades']} | {r['win_rate']}% | "
            f"${r['pnl']} | ${r['avg_pnl']} |",
            flush=True,
        )
    OUT.write_text(
        json.dumps(
            {
                "strategy": "Fib 61.8/70.5/78.6 retrace + liquidity sweep reclaim",
                "risk_usd": RISK_USD,
                "windows": results,
            },
            indent=2,
        )
    )
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
