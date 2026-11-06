#!/usr/bin/env python3
"""RESEARCH ONLY — mastery grid: which setups deserve more study.

Compares last ~1 month ($200/R):
  GOLD fib variants (shallow/deep × TF × H1 filter)
  BTC / ETH: fib 15m/15m 50-618 vs EMA20/50 5m

No live wiring.
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

OUT = Path("/tmp/mastery_grid_bt.json")
RISK_USD = 200.0
SWING = 3


def _fetch_5m(symbol: str, start: str, end: str) -> pd.DataFrame:
    warm = (datetime.fromisoformat(start) - timedelta(days=14)).strftime("%Y-%m-%d")
    end_p = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    yf_map = {
        "GOLD": "GC=F",
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
    }
    import yfinance as yf

    ticker = yf_map.get(symbol, symbol)
    df = yf.Ticker(ticker).history(start=warm, end=end_p, interval="5m", auto_adjust=True)
    if df is None or df.empty:
        from data.historical import fetch_symbol_ohlcv

        df = fetch_symbol_ohlcv(symbol, warm, end_p, interval="5m")
    if df is None or df.empty:
        raise RuntimeError(f"no data {symbol}")
    df = df.rename(columns=str.lower)
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    df = df[cols].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _resample(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    m = {"5m": "5min", "15m": "15min", "1h": "1h"}[rule]
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
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    return out


def _swing_low(df, i, n=SWING):
    if i < n or i + n >= len(df):
        return False
    lo = float(df.iloc[i]["low"])
    return all(lo <= float(df.iloc[i + k]["low"]) for k in range(-n, n + 1) if k != 0)


def _swing_high(df, i, n=SWING):
    if i < n or i + n >= len(df):
        return False
    hi = float(df.iloc[i]["high"])
    return all(hi >= float(df.iloc[i + k]["high"]) for k in range(-n, n + 1) if k != 0)


def _impulse(df, i, atr):
    look = range(max(SWING, i - 60), i - SWING)
    lows = [(j, float(df.iloc[j]["low"])) for j in look if _swing_low(df, j)]
    highs = [(j, float(df.iloc[j]["high"])) for j in look if _swing_high(df, j)]
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
    return max(cands, key=lambda x: x["bi"]) if cands else None


def _fib_zone(imp, atr, fibs):
    a, b = float(imp["a"]), float(imp["b"])
    if imp["side"] == "BUY":
        levels = [b - r * (b - a) for r in fibs]
    else:
        levels = [b + r * (a - b) for r in fibs]
    pad = 0.40 * atr
    return min(levels) - pad, max(levels) + pad


def _h1_bias(df_ent: pd.DataFrame, ts) -> int:
    """+1 bull, -1 bear from 1h EMAs built off entry series."""
    h1 = df_ent["close"].resample("1h").last().dropna()
    if len(h1) < 55:
        return 0
    e20 = h1.ewm(span=20, adjust=False).mean()
    e50 = h1.ewm(span=50, adjust=False).mean()
    past = h1.index[h1.index <= ts]
    if past.empty:
        return 0
    t = past[-1]
    return 1 if float(e20.loc[t]) > float(e50.loc[t]) else -1


def run_fib(
    df_imp: pd.DataFrame,
    df_ent: pd.DataFrame,
    start: str,
    end: str,
    *,
    fibs: tuple[float, ...],
    h1_filter: bool,
    max_age_h: float = 24.0,
) -> dict[str, Any]:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    trades = []
    open_t = None
    cooldown = df_ent.index[0]
    cache = [None] * len(df_imp)
    for i in range(40, len(df_imp)):
        atr = float(df_imp.iloc[i]["atr"] or 1)
        cache[i] = _impulse(df_imp, i, atr)

    for ei in range(40, len(df_ent)):
        ts = df_ent.index[ei]
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
                    trades.append(round(float(r_mult) * RISK_USD, 2))
                open_t = None
                cooldown = ts + timedelta(minutes=30)
            continue

        if ts < start_dt or ts > end_dt or ts < cooldown:
            continue
        pos = df_imp.index.searchsorted(ts, side="right") - 1
        if pos < 40:
            continue
        imp = cache[pos]
        if not imp or (ts - imp["end_t"]).total_seconds() > max_age_h * 3600:
            continue
        if h1_filter:
            bias = _h1_bias(df_ent, ts)
            if imp["side"] == "BUY" and bias < 0:
                continue
            if imp["side"] == "SELL" and bias > 0:
                continue

        zlo, zhi = _fib_zone(imp, atr, fibs)
        win = df_ent.iloc[max(0, ei - 20) : ei]
        if imp["side"] == "BUY":
            liq = float(win["low"].min()) if len(win) else l
            if l < min(liq, zlo) and c > zlo and l <= zhi and c > float(row["open"]):
                entry, sl = c, min(l, liq) - 0.08 * atr
                risk = entry - sl
                if risk > 0.12 * atr:
                    open_t = {
                        "side": "BUY",
                        "entry": entry,
                        "sl": sl,
                        "tp": entry + 2 * risk,
                        "entry_t": ts,
                    }
        else:
            liq = float(win["high"].max()) if len(win) else h
            if h > max(liq, zhi) and c < zhi and h >= zlo and c < float(row["open"]):
                entry, sl = c, max(h, liq) + 0.08 * atr
                risk = sl - entry
                if risk > 0.12 * atr:
                    open_t = {
                        "side": "SELL",
                        "entry": entry,
                        "sl": sl,
                        "tp": entry - 2 * risk,
                        "entry_t": ts,
                    }

    wins = sum(1 for p in trades if p > 0)
    losses = sum(1 for p in trades if p < 0)
    dec = wins + losses
    pnl = round(sum(trades), 2)
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / dec, 1) if dec else 0.0,
        "pnl": pnl,
        "avg_pnl": round(pnl / len(trades), 2) if trades else 0.0,
        "exp_r": round((pnl / RISK_USD) / len(trades), 3) if trades else 0.0,
    }


def run_ema(df5: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    from strategy.ema_pull_brain import evaluate_ema_2050_pull

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    work = df5.rename(columns={"atr": "ATR", "ema20": "EMA20", "ema50": "EMA50"})
    trades = []
    open_t = None
    cooldown = None
    for i in range(60, len(work)):
        ts = work.index[i]
        row = work.iloc[i]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
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
                    trades.append(round(float(r_mult) * RISK_USD, 2))
                open_t = None
                cooldown = ts + timedelta(minutes=30)
            continue
        if ts < start_dt or ts > end_dt:
            continue
        if cooldown is not None and ts < cooldown:
            continue
        hit = evaluate_ema_2050_pull(work.iloc[: i + 1])
        if not hit:
            continue
        entry, sl = float(hit["entry"]), float(hit["sl"])
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        tp = entry + 2 * risk if hit["signal"] == "BUY" else entry - 2 * risk
        open_t = {
            "side": hit["signal"],
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "entry_t": ts,
        }

    wins = sum(1 for p in trades if p > 0)
    losses = sum(1 for p in trades if p < 0)
    dec = wins + losses
    pnl = round(sum(trades), 2)
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / dec, 1) if dec else 0.0,
        "pnl": pnl,
        "avg_pnl": round(pnl / len(trades), 2) if trades else 0.0,
        "exp_r": round((pnl / RISK_USD) / len(trades), 3) if trades else 0.0,
    }


def _score(r: dict) -> float:
    """Rank: PnL primary, need enough trades, prefer positive E[R]."""
    if r["trades"] < 5:
        return r["pnl"] - 500  # penalize tiny samples
    return r["pnl"] + 50 * r["exp_r"] * r["trades"]


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=29)).isoformat()
    end_s = end.isoformat()
    print(f"Mastery grid {start} → {end_s}", flush=True)

    rows = []

    # --- GOLD fib variants ---
    g5 = _fetch_5m("GOLD", start, end_s)
    frames = {tf: _resample(g5, tf) for tf in ("5m", "15m", "1h")}
    gold_specs = [
        ("GOLD", "fib50-618", "15m", "15m", (0.50, 0.618), False),
        ("GOLD", "fib50-618+H1", "15m", "15m", (0.50, 0.618), True),
        ("GOLD", "fib618-786", "15m", "15m", (0.618, 0.705, 0.786), False),
        ("GOLD", "fib618-786", "15m", "5m", (0.618, 0.705, 0.786), False),
        ("GOLD", "fib618-786+H1", "15m", "5m", (0.618, 0.705, 0.786), True),
        ("GOLD", "fib50-618", "1h", "15m", (0.50, 0.618), False),
        ("GOLD", "fib50-618+H1", "1h", "15m", (0.50, 0.618), True),
        ("GOLD", "ema20/50", "5m", "5m", None, False),
    ]
    for sym, name, imp, ent, fibs, h1 in gold_specs:
        if fibs is None:
            r = run_ema(frames["5m"], start, end_s)
        else:
            r = run_fib(
                frames[imp], frames[ent], start, end_s, fibs=fibs, h1_filter=h1
            )
        row = {
            "symbol": sym,
            "setup": name,
            "impulse": imp,
            "entry": ent,
            "h1_filter": h1,
            **r,
        }
        rows.append(row)
        print(
            f"  {sym} {name} {imp}→{ent} h1={h1} n={r['trades']} wr={r['win_rate']}% pnl=${r['pnl']}",
            flush=True,
        )

    # --- Crypto ---
    for sym in ("BTCUSD", "ETHUSD"):
        try:
            c5 = _fetch_5m(sym, start, end_s)
        except Exception as exc:
            print(f"  skip {sym}: {exc}", flush=True)
            continue
        cf = {tf: _resample(c5, tf) for tf in ("5m", "15m")}
        for name, imp, ent, fibs, h1 in [
            ("fib50-618", "15m", "15m", (0.50, 0.618), False),
            ("fib50-618+H1", "15m", "15m", (0.50, 0.618), True),
            ("ema20/50", "5m", "5m", None, False),
        ]:
            if fibs is None:
                r = run_ema(cf["5m"], start, end_s)
            else:
                r = run_fib(cf[imp], cf[ent], start, end_s, fibs=fibs, h1_filter=h1)
            rows.append(
                {
                    "symbol": sym,
                    "setup": name,
                    "impulse": imp,
                    "entry": ent,
                    "h1_filter": h1,
                    **r,
                }
            )
            print(
                f"  {sym} {name} {imp}→{ent} n={r['trades']} wr={r['win_rate']}% pnl=${r['pnl']}",
                flush=True,
            )

    ranked = sorted(rows, key=_score, reverse=True)
    payload = {
        "window": {"start": start, "end": end_s},
        "risk_usd": RISK_USD,
        "results": rows,
        "master_next": ranked[:8],
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print("\n### Mastery grid — last 1 month ($200/R)\n", flush=True)
    print(
        "| Rank | Symbol | Setup | TF | Trades | WR | PnL | E[R] |",
        flush=True,
    )
    print("|---:|---|---|---|---:|---:|---:|---:|", flush=True)
    for i, r in enumerate(ranked, 1):
        tf = f"{r['impulse']}→{r['entry']}"
        print(
            f"| {i} | {r['symbol']} | {r['setup']} | {tf} | {r['trades']} | "
            f"{r['win_rate']}% | ${r['pnl']} | {r['exp_r']} |",
            flush=True,
        )

    print("\n**Study / master next (top)**", flush=True)
    for i, r in enumerate(ranked[:5], 1):
        print(
            f"{i}. {r['symbol']} · {r['setup']} · {r['impulse']}→{r['entry']} · "
            f"n={r['trades']} WR={r['win_rate']}% PnL=${r['pnl']}",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
