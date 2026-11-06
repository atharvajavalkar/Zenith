#!/usr/bin/env python3
"""RESEARCH ONLY — Inna Rosputnia 18-day MA commodity breakout on GOLD.

No backend / live algo changes. Public rules (MoneyShow / managed-accounts-ir):
  Long:  two consecutive daily bars with lows > SMA18 of closes,
         neither an inside day → buy stop at highest high of those two bars.
  Short: two consecutive daily bars with highs < SMA18,
         neither an inside day → sell stop at lowest low of those two bars.
  Exit:  opposite 18MA signal (her stated stop), plus research 2R target.

Daily GC=F via yfinance. Windows: 14d + 1m (plus warmup for SMA18).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = Path("/tmp/inna_rosputnia_gold_bt.json")
RISK_USD = 200.0


def _fetch_daily(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    start_dt = datetime.fromisoformat(start) - timedelta(days=90)
    end_dt = datetime.fromisoformat(end) + timedelta(days=1)
    df = yf.Ticker("GC=F").history(
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise RuntimeError("no GC=F daily")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df["sma18"] = df["close"].rolling(18).mean()
    return df


def _is_inside(cur: pd.Series, prev: pd.Series) -> bool:
    return float(cur["high"]) < float(prev["high"]) and float(cur["low"]) > float(prev["low"])


def backtest(df: pd.DataFrame, start: str, end: str) -> dict:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    trades: list[dict] = []
    pending = None  # frozen setup until fill or invalidate
    open_t = None

    for i in range(20, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        a = df.iloc[i - 2]  # older of the two signal bars
        b = df.iloc[i - 1]  # newer of the two signal bars
        if pd.isna(float(row["sma18"])) or pd.isna(float(a["sma18"])) or pd.isna(
            float(b["sma18"])
        ):
            continue

        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])

        # --- manage open ---
        if open_t is not None:
            side = open_t["side"]
            hit_sl = (side == "BUY" and l <= open_t["sl"]) or (
                side == "SELL" and h >= open_t["sl"]
            )
            hit_tp = (side == "BUY" and h >= open_t["tp"]) or (
                side == "SELL" and l <= open_t["tp"]
            )
            # Opposite qualified setup forms on bars a,b
            long_opp = (
                float(a["low"]) > float(a["sma18"])
                and float(b["low"]) > float(b["sma18"])
                and not _is_inside(b, a)
            )
            short_opp = (
                float(a["high"]) < float(a["sma18"])
                and float(b["high"]) < float(b["sma18"])
                and not _is_inside(b, a)
            )
            opp = (side == "BUY" and short_opp) or (side == "SELL" and long_opp)

            exit_px = None
            reason = None
            if hit_sl:
                exit_px, reason = open_t["sl"], "sl"
            elif hit_tp:
                exit_px, reason = open_t["tp"], "tp"
            elif opp:
                exit_px, reason = c, "opposite_signal"

            if exit_px is not None:
                if start_dt <= open_t["entry_t"] <= end_dt:
                    risk = abs(open_t["entry"] - open_t["sl"])
                    if side == "BUY":
                        r_mult = (exit_px - open_t["entry"]) / risk if risk else 0.0
                    else:
                        r_mult = (open_t["entry"] - exit_px) / risk if risk else 0.0
                    trades.append(
                        {
                            "entry_t": str(open_t["entry_t"])[:10],
                            "exit_t": str(ts)[:10],
                            "side": side,
                            "entry": round(open_t["entry"], 2),
                            "exit": round(float(exit_px), 2),
                            "pnl": round(r_mult * RISK_USD, 2),
                            "reason": reason,
                        }
                    )
                open_t = None
                pending = None
                continue

        if open_t is not None:
            continue

        # --- try fill frozen pending (buy/sell stop) ---
        if pending is not None:
            # Invalidate if price closes back through SMA18 against setup
            if pending["side"] == "BUY" and c < float(row["sma18"]):
                pending = None
            elif pending["side"] == "SELL" and c > float(row["sma18"]):
                pending = None
            else:
                if pending["side"] == "BUY" and h >= pending["trigger"]:
                    entry = pending["trigger"]
                    sl = pending["sl"]
                    risk = entry - sl
                    if risk > 0:
                        open_t = {
                            "side": "BUY",
                            "entry": entry,
                            "sl": sl,
                            "tp": entry + 2.0 * risk,
                            "entry_t": ts,
                        }
                    pending = None
                    continue
                if pending["side"] == "SELL" and l <= pending["trigger"]:
                    entry = pending["trigger"]
                    sl = pending["sl"]
                    risk = sl - entry
                    if risk > 0:
                        open_t = {
                            "side": "SELL",
                            "entry": entry,
                            "sl": sl,
                            "tp": entry - 2.0 * risk,
                            "entry_t": ts,
                        }
                    pending = None
                    continue

        # --- form new pending only when flat and no pending ---
        if pending is not None:
            continue

        # Inside-day check: neither of the two signal bars may be inside vs prior bar
        # Rosputnia: "Neither of these days can be an inside day"
        b_inside = _is_inside(b, a)
        a_inside = _is_inside(a, df.iloc[i - 3])

        long_setup = (
            float(a["low"]) > float(a["sma18"])
            and float(b["low"]) > float(b["sma18"])
            and not a_inside
            and not b_inside
        )
        short_setup = (
            float(a["high"]) < float(a["sma18"])
            and float(b["high"]) < float(b["sma18"])
            and not a_inside
            and not b_inside
        )

        if long_setup:
            trigger = max(float(a["high"]), float(b["high"]))
            sl = min(float(a["low"]), float(b["low"]))
            # Freeze setup; fill on break (same bar allowed if breaks after setup forms)
            pending = {
                "side": "BUY",
                "trigger": trigger,
                "sl": sl,
                "formed_i": i,
            }
            if h >= trigger and (trigger - sl) > 0:
                open_t = {
                    "side": "BUY",
                    "entry": trigger,
                    "sl": sl,
                    "tp": trigger + 2.0 * (trigger - sl),
                    "entry_t": ts,
                }
                pending = None
        elif short_setup:
            trigger = min(float(a["low"]), float(b["low"]))
            sl = max(float(a["high"]), float(b["high"]))
            pending = {
                "side": "SELL",
                "trigger": trigger,
                "sl": sl,
                "formed_i": i,
            }
            if l <= trigger and (sl - trigger) > 0:
                open_t = {
                    "side": "SELL",
                    "entry": trigger,
                    "sl": sl,
                    "tp": trigger - 2.0 * (sl - trigger),
                    "entry_t": ts,
                }
                pending = None

    if open_t is not None and start_dt <= open_t["entry_t"] <= end_dt:
        c = float(df.iloc[-1]["close"])
        risk = abs(open_t["entry"] - open_t["sl"])
        if open_t["side"] == "BUY":
            r_mult = (c - open_t["entry"]) / risk if risk else 0.0
        else:
            r_mult = (open_t["entry"] - c) / risk if risk else 0.0
        trades.append(
            {
                "entry_t": str(open_t["entry_t"])[:10],
                "exit_t": str(df.index[-1])[:10],
                "side": open_t["side"],
                "entry": round(open_t["entry"], 2),
                "exit": round(c, 2),
                "pnl": round(r_mult * RISK_USD, 2),
                "reason": "eod",
            }
        )

    in_win = [t for t in trades if start <= t["entry_t"] <= end]
    wins = sum(1 for t in in_win if t["pnl"] > 0)
    losses = sum(1 for t in in_win if t["pnl"] < 0)
    pnl = round(sum(t["pnl"] for t in in_win), 2)
    decisive = wins + losses
    return {
        "start": start,
        "end": end,
        "trades": len(in_win),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / decisive, 1) if decisive else 0.0,
        "pnl": pnl,
        "trade_list": in_win,
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    windows = [
        ("14d", (end - timedelta(days=13)).isoformat(), end.isoformat()),
        ("1m", (end - timedelta(days=29)).isoformat(), end.isoformat()),
        # Extra context window — still research-only
        ("3m", (end - timedelta(days=89)).isoformat(), end.isoformat()),
    ]
    df = _fetch_daily(windows[2][1], windows[2][2])
    print(f"Inna Rosputnia 18MA GOLD daily bars={len(df)}", flush=True)

    results = {}
    for label, a, b in windows:
        results[label] = backtest(df, a, b)
        r = results[label]
        print(
            f"[{label}] n={r['trades']} wr={r['win_rate']}% pnl=${r['pnl']} "
            f"trades={r['trade_list']}",
            flush=True,
        )

    payload = {
        "strategy": "Inna Rosputnia — 18-day MA qualified breakout (commodities)",
        "rules": {
            "long": "2 daily bars lows > SMA18, neither inside → buy stop at 2-bar high; SL at 2-bar low; exit opposite signal or 2R",
            "short": "2 daily bars highs < SMA18, neither inside → sell stop at 2-bar low; SL at 2-bar high; exit opposite signal or 2R",
            "note": "Mechanical public rules only. Accumulation/OBV + fundamental filter NOT coded (her discretionary overlay).",
        },
        "symbol": "GOLD (GC=F daily)",
        "risk_usd": RISK_USD,
        "windows": results,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
