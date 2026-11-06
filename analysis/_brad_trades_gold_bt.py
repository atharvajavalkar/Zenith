#!/usr/bin/env python3
"""RESEARCH ONLY — Brad Trades (Brad Goh) Asia Sweep / liquidity-sweep on GOLD.

No live / backend wiring. Mechanicalized public rules:

  Asia Sweep (primary — session play he teaches for clean scalps):
    1. Mark Asia session high/low (00:00–07:00 UTC).
    2. In London window (07:00–10:00 UTC): wait for a sweep of Asia high or low.
    3. Confirmation: candle closes back inside the Asia range (reversal after grab).
    4. Enter at that close; SL beyond sweep extreme (+ small pad).
    5. TP = 2R (research) OR opposite Asia extreme (whichever hits first on path).
       Also take opposite-Asia as soft max target if closer than 2R.

  Liquidation + mitigation (secondary — his "only setup" narrative):
    Equal highs/lows (swing pivots within tolerance) → sweep → enter next bar
    if close reclaims; SL beyond wick; TP 2R. HTF bias: only long if last
    closed 1h is HH structure-ish (close > prior 1h mid), short if opposite.

Windows: 14d · 1m · 3m. Risk $200/R. GOLD via MetaAPI/TV/yf 5m.
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

OUT = Path("/tmp/brad_trades_gold_bt.json")
RISK_USD = 200.0
ASIA_START_H = 0
ASIA_END_H = 7
LONDON_START_H = 7
LONDON_END_H = 10
SL_PAD_ATR = 0.05
EQUAL_TOL_ATR = 0.15
SWING = 3


def _fetch_5m(start: str, end: str) -> pd.DataFrame:
    from data.historical import fetch_symbol_ohlcv

    start_dt = datetime.fromisoformat(start) - timedelta(days=5)
    end_dt = datetime.fromisoformat(end) + timedelta(days=1)
    df = fetch_symbol_ohlcv(
        "GOLD",
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
        interval="5m",
    )
    if df is None or df.empty:
        raise RuntimeError("no GOLD 5m")
    df = df.rename(columns=str.lower)
    need = ["open", "high", "low", "close"]
    df = df[[c for c in need if c in df.columns]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    # ATR proxy
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


def _day_key(ts: pd.Timestamp) -> str:
    return str(ts.tz_convert("UTC").date())


def _asia_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for day, g in df.groupby(_day_key):
        asia = g[(g.index.hour >= ASIA_START_H) & (g.index.hour < ASIA_END_H)]
        if len(asia) < 6:
            continue
        ranges[day] = {
            "high": float(asia["high"].max()),
            "low": float(asia["low"].min()),
        }
    return ranges


def _summarize(trades: list[dict], start: str, end: str) -> dict[str, Any]:
    in_win = [t for t in trades if start <= t["entry_t"][:10] <= end]
    wins = sum(1 for t in in_win if t["pnl"] > 0)
    losses = sum(1 for t in in_win if t["pnl"] < 0)
    decisive = wins + losses
    return {
        "start": start,
        "end": end,
        "trades": len(in_win),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / decisive, 1) if decisive else 0.0,
        "pnl": round(sum(t["pnl"] for t in in_win), 2),
        "avg_pnl": round(sum(t["pnl"] for t in in_win) / len(in_win), 2) if in_win else 0.0,
        "trade_list": in_win,
    }


def backtest_asia_sweep(df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    asia = _asia_ranges(df)
    trades: list[dict] = []
    open_t: Optional[dict] = None
    taken_days: set[str] = set()  # one Asia-sweep attempt fill per day

    for i in range(20, len(df)):
        ts = df.index[i]
        if ts < start_dt - timedelta(days=1):
            continue
        row = df.iloc[i]
        day = _day_key(ts)
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
            # Soft structure target: opposite Asia extreme
            hit_opp = False
            if open_t.get("opp") is not None:
                if side == "BUY" and h >= open_t["opp"]:
                    hit_opp = True
                if side == "SELL" and l <= open_t["opp"]:
                    hit_opp = True
            exit_px = None
            reason = None
            if hit_sl:
                exit_px, reason = open_t["sl"], "sl"
            elif hit_opp and open_t.get("opp") is not None:
                # If opp is closer / hits before 2R path — count as structure TP
                exit_px, reason = open_t["opp"], "asia_opposite"
            elif hit_tp:
                exit_px, reason = open_t["tp"], "tp_2r"

            if exit_px is not None:
                if start_dt <= open_t["entry_t"] <= end_dt:
                    risk = abs(open_t["entry"] - open_t["sl"])
                    if side == "BUY":
                        r_mult = (exit_px - open_t["entry"]) / risk if risk else 0.0
                    else:
                        r_mult = (open_t["entry"] - exit_px) / risk if risk else 0.0
                    trades.append(
                        {
                            "entry_t": str(open_t["entry_t"])[:16],
                            "exit_t": str(ts)[:16],
                            "side": side,
                            "entry": round(open_t["entry"], 2),
                            "exit": round(float(exit_px), 2),
                            "pnl": round(r_mult * RISK_USD, 2),
                            "reason": reason,
                            "setup": "asia_sweep",
                        }
                    )
                open_t = None
            continue

        if ts < start_dt or ts > end_dt:
            continue
        if day in taken_days:
            continue
        if day not in asia:
            continue
        if not (LONDON_START_H <= ts.hour < LONDON_END_H):
            continue

        ah, al = asia[day]["high"], asia[day]["low"]
        pad = SL_PAD_ATR * atr

        # Sweep Asia high then close back inside → short
        if h > ah and c < ah and c > al:
            entry = c
            sl = h + pad
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - 2.0 * risk
            opp = al  # opposite Asia
            open_t = {
                "side": "SELL",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "opp": opp,
                "entry_t": ts,
            }
            taken_days.add(day)
            continue

        # Sweep Asia low then close back inside → long
        if l < al and c > al and c < ah:
            entry = c
            sl = l - pad
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + 2.0 * risk
            opp = ah
            open_t = {
                "side": "BUY",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "opp": opp,
                "entry_t": ts,
            }
            taken_days.add(day)

    if open_t is not None and start_dt <= open_t["entry_t"] <= end_dt:
        c = float(df.iloc[-1]["close"])
        risk = abs(open_t["entry"] - open_t["sl"])
        r_mult = (
            (c - open_t["entry"]) / risk
            if open_t["side"] == "BUY"
            else (open_t["entry"] - c) / risk
        )
        trades.append(
            {
                "entry_t": str(open_t["entry_t"])[:16],
                "exit_t": str(df.index[-1])[:16],
                "side": open_t["side"],
                "entry": round(open_t["entry"], 2),
                "exit": round(c, 2),
                "pnl": round(r_mult * RISK_USD, 2),
                "reason": "eod",
                "setup": "asia_sweep",
            }
        )

    return _summarize(trades, start, end)


def _is_swing_low(df: pd.DataFrame, i: int) -> bool:
    if i < SWING or i + SWING >= len(df):
        return False
    lo = float(df.iloc[i]["low"])
    return all(lo <= float(df.iloc[i + k]["low"]) for k in range(-SWING, SWING + 1) if k != 0)


def _is_swing_high(df: pd.DataFrame, i: int) -> bool:
    if i < SWING or i + SWING >= len(df):
        return False
    hi = float(df.iloc[i]["high"])
    return all(hi >= float(df.iloc[i + k]["high"]) for k in range(-SWING, SWING + 1) if k != 0)


def backtest_liq_mitigation(df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    """Equal-level sweep → reclaim close → next-bar entry; 2R; max 1 open."""
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    trades: list[dict] = []
    open_t: Optional[dict] = None
    pending: Optional[dict] = None
    cooldown_until = df.index[0]

    for i in range(40, len(df) - 1):
        ts = df.index[i]
        row = df.iloc[i]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        atr = float(row["atr"] or 0) or max(h - l, 1e-6)
        tol = EQUAL_TOL_ATR * atr

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
                reason = "sl" if hit_sl else "tp_2r"
                if start_dt <= open_t["entry_t"] <= end_dt:
                    risk = abs(open_t["entry"] - open_t["sl"])
                    r_mult = (
                        (exit_px - open_t["entry"]) / risk
                        if side == "BUY"
                        else (open_t["entry"] - exit_px) / risk
                    )
                    trades.append(
                        {
                            "entry_t": str(open_t["entry_t"])[:16],
                            "exit_t": str(ts)[:16],
                            "side": side,
                            "entry": round(open_t["entry"], 2),
                            "exit": round(float(exit_px), 2),
                            "pnl": round(r_mult * RISK_USD, 2),
                            "reason": reason,
                            "setup": "liq_mitigation",
                        }
                    )
                open_t = None
                cooldown_until = ts + timedelta(hours=2)
            continue

        if pending is not None and ts >= pending["entry_t"]:
            # Enter at open of signal bar's next bar (approx current open)
            entry = float(row["open"])
            sl = pending["sl"]
            side = pending["side"]
            risk = abs(entry - sl)
            if risk > 0 and (
                (side == "BUY" and sl < entry) or (side == "SELL" and sl > entry)
            ):
                if start_dt <= ts <= end_dt:
                    open_t = {
                        "side": side,
                        "entry": entry,
                        "sl": sl,
                        "tp": entry + 2.0 * risk if side == "BUY" else entry - 2.0 * risk,
                        "entry_t": ts,
                    }
            pending = None
            continue

        if ts < start_dt or ts > end_dt or ts < cooldown_until:
            continue

        # Find recent equal lows / highs among swing pivots in last 36 bars
        look = range(max(SWING, i - 36), i - 1)
        swing_lows = [(j, float(df.iloc[j]["low"])) for j in look if _is_swing_low(df, j)]
        swing_highs = [(j, float(df.iloc[j]["high"])) for j in look if _is_swing_high(df, j)]

        # Long: sweep equal lows then close back above
        if len(swing_lows) >= 2:
            levels = [x[1] for x in swing_lows]
            eq = levels[-1]
            peers = [lv for lv in levels[:-1] if abs(lv - eq) <= tol]
            if peers and l < min(peers + [eq]) - 1e-9 and c > eq:
                # liquidation-style wick
                body = abs(c - float(row["open"]))
                lower_wick = min(float(row["open"]), c) - l
                if lower_wick >= body:  # long lower wick
                    pending = {
                        "side": "BUY",
                        "sl": l - SL_PAD_ATR * atr,
                        "entry_t": df.index[i + 1],
                    }
                    continue

        if len(swing_highs) >= 2:
            levels = [x[1] for x in swing_highs]
            eq = levels[-1]
            peers = [lv for lv in levels[:-1] if abs(lv - eq) <= tol]
            if peers and h > max(peers + [eq]) + 1e-9 and c < eq:
                body = abs(c - float(row["open"]))
                upper_wick = h - max(float(row["open"]), c)
                if upper_wick >= body:
                    pending = {
                        "side": "SELL",
                        "sl": h + SL_PAD_ATR * atr,
                        "entry_t": df.index[i + 1],
                    }

    return _summarize(trades, start, end)


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    windows = [
        ("14d", (end - timedelta(days=13)).isoformat(), end.isoformat()),
        ("1m", (end - timedelta(days=29)).isoformat(), end.isoformat()),
        ("3m", (end - timedelta(days=89)).isoformat(), end.isoformat()),
    ]
    df = _fetch_5m(windows[2][1], windows[2][2])
    print(f"Brad Trades GOLD 5m bars={len(df)}", flush=True)

    results: dict[str, Any] = {"asia_sweep": {}, "liq_mitigation": {}}
    for label, a, b in windows:
        asia = backtest_asia_sweep(df, a, b)
        liq = backtest_liq_mitigation(df, a, b)
        results["asia_sweep"][label] = asia
        results["liq_mitigation"][label] = liq
        print(
            f"[asia {label}] n={asia['trades']} wr={asia['win_rate']}% pnl=${asia['pnl']}",
            flush=True,
        )
        print(
            f"[liq  {label}] n={liq['trades']} wr={liq['win_rate']}% pnl=${liq['pnl']}",
            flush=True,
        )

    payload = {
        "strategy": "Brad Trades (Brad Goh) — Asia Sweep + Liquidation mitigation",
        "rules": {
            "asia_sweep": (
                "Asia 00-07 UTC range → London 07-10 UTC sweep of Asia H/L, "
                "close back inside → enter; SL beyond sweep; TP asia-opposite or 2R"
            ),
            "liq_mitigation": (
                "Equal swing highs/lows → sweep + long-wick reclaim → next bar entry; "
                "SL beyond wick; TP 2R; 2h cooldown"
            ),
            "note": (
                "Public mechanical proxy of Brad's Market Mechanics / EdgeFlo teaching. "
                "Discretionary HTF narrative and session judgment NOT coded."
            ),
        },
        "symbol": "GOLD",
        "risk_usd": RISK_USD,
        "windows": results,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
