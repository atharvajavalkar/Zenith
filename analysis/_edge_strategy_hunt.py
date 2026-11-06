#!/usr/bin/env python3
"""RESEARCH ONLY — strategy hunt for FRAGILE/DEAD sleeves.

Runs the same edge battery (full / OOS 30% / walk-forward 4 / torture / MC)
across alternate setups. No live wiring.

Targets: USDCHF, USDJPY, NAS100, UKOIL, BTCUSD, ETHUSD
(GOLD+AUDUSD already SURVIVES — skipped)
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = Path("/tmp/edge_strategy_hunt.json")
RISK_USD = 200.0
MC_PATHS = 1500
SWING = 3

TARGETS = ("USDCHF", "USDJPY", "NAS100", "UKOIL", "BTCUSD", "ETHUSD")
YF = {
    "USDCHF": "USDCHF=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "NAS100": "NQ=F",
    "UKOIL": "BZ=F",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}


def _fetch_1h(symbol: str, start: str, end: str) -> pd.DataFrame:
    warm = (datetime.fromisoformat(start) - timedelta(days=20)).strftime("%Y-%m-%d")
    end_p = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    import yfinance as yf

    df = None
    for ticker in (YF.get(symbol, symbol), f"{symbol}=X", symbol):
        try:
            df = yf.Ticker(ticker).history(
                start=warm, end=end_p, interval="1h", auto_adjust=True
            )
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        raise RuntimeError(f"no 1h {symbol}")
    df = df.rename(columns=str.lower)
    df = df[[c for c in ("open", "high", "low", "close") if c in df.columns]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    out["ATR"] = out["atr"]
    out["EMA8"] = out["close"].ewm(span=8, adjust=False).mean()
    out["EMA9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["EMA20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["EMA21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["EMA50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["SMA20"] = out["close"].rolling(20).mean()
    out["SMA50"] = out["close"].rolling(50).mean()
    out["ema20"] = out["EMA20"]
    out["ema50"] = out["EMA50"]
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


def _sim_open(df, open_t, ei, start_i, spread_r, trades):
    row = df.iloc[ei]
    h, l = float(row["high"]), float(row["low"])
    side = open_t["side"]
    hit_sl = (side == "BUY" and l <= open_t["sl"]) or (side == "SELL" and h >= open_t["sl"])
    hit_tp = (side == "BUY" and h >= open_t["tp"]) or (side == "SELL" and l <= open_t["tp"])
    if not (hit_sl or hit_tp):
        return open_t, False
    exit_px = open_t["sl"] if hit_sl else open_t["tp"]
    risk = abs(open_t["entry"] - open_t["sl"])
    r_mult = (
        (exit_px - open_t["entry"]) / risk
        if side == "BUY"
        else (open_t["entry"] - exit_px) / risk
    )
    r_mult = float(r_mult) - spread_r
    if open_t["ei"] >= start_i:
        trades.append(
            {
                "i": open_t["ei"],
                "entry_t": str(open_t["entry_t"])[:16],
                "side": side,
                "r": round(r_mult, 4),
                "pnl": round(r_mult * RISK_USD, 2),
            }
        )
    return None, True


def run_ema_generic(
    df,
    *,
    fast_col: str,
    slow_col: str,
    pull_atr: float,
    disp_atr: float,
    start_i=0,
    end_i=None,
    late_bars=0,
    sl_widen=1.0,
    spread_r=0.0,
    skip_p=0.0,
    rng=None,
    rr=2.0,
):
    """EMA pullback using named columns (8/21, 9/21, 20/50…)."""
    rng = rng or random.Random(0)
    end_i = end_i if end_i is not None else len(df) - 1
    trades = []
    open_t = None
    cooldown = -1
    swing = 10
    for i in range(max(60, start_i), min(end_i, len(df) - 1) + 1):
        row = df.iloc[i]
        h, l, c, o = float(row["high"]), float(row["low"]), float(row["close"]), float(row["open"])
        atr = float(row["ATR"] or 0) or max(h - l, 1e-6)
        if open_t is not None:
            open_t, closed = _sim_open(df, open_t, i, start_i, spread_r, trades)
            if closed:
                cooldown = i + 2
            continue
        if i < cooldown:
            continue
        if skip_p > 0 and rng.random() < skip_p:
            continue
        ema_f = float(row[fast_col])
        ema_s = float(row[slow_col])
        mid = 0.5 * (ema_f + ema_s)
        body = abs(c - o)
        rng_ = max(h - l, 1e-12)
        bull, bear = ema_f > ema_s, ema_f < ema_s
        disp_buy = c > o and body >= disp_atr * atr and (c - l) / rng_ >= 0.75
        disp_sell = c < o and body >= disp_atr * atr and (h - c) / rng_ >= 0.75
        near = abs(c - mid) <= pull_atr * atr
        tail = df.iloc[-(swing + 1) :] if False else df.iloc[max(0, i - swing) : i + 1]
        signal = None
        sl = None
        if bull and disp_buy and near and c > ema_f:
            signal = "BUY"
            sl = float(tail["low"].min()) - 0.10 * atr
        elif bear and disp_sell and near and c < ema_f:
            signal = "SELL"
            sl = float(tail["high"].max()) + 0.10 * atr
        if not signal or sl is None:
            continue
        ei_e = min(i + late_bars, len(df) - 2)
        entry = float(df.iloc[ei_e]["open"]) if late_bars else c
        risk0 = abs(entry - sl)
        if risk0 <= 0:
            continue
        risk = risk0 * sl_widen
        if signal == "BUY":
            sl_e, tp = entry - risk, entry + rr * risk
        else:
            sl_e, tp = entry + risk, entry - rr * risk
        open_t = {
            "side": signal,
            "entry": entry,
            "sl": sl_e,
            "tp": tp,
            "entry_t": df.index[ei_e],
            "ei": ei_e,
        }
    return trades


def run_sma_cross(
    df,
    *,
    start_i=0,
    end_i=None,
    late_bars=0,
    sl_widen=1.0,
    spread_r=0.0,
    skip_p=0.0,
    rng=None,
    rr=2.0,
):
    rng = rng or random.Random(0)
    end_i = end_i if end_i is not None else len(df) - 1
    trades = []
    open_t = None
    cooldown = -1
    for i in range(max(55, start_i), min(end_i, len(df) - 1) + 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        atr = float(row["ATR"] or 0) or max(h - l, 1e-6)
        if open_t is not None:
            open_t, closed = _sim_open(df, open_t, i, start_i, spread_r, trades)
            if closed:
                cooldown = i + 3
            continue
        if i < cooldown:
            continue
        if skip_p > 0 and rng.random() < skip_p:
            continue
        f0, s0 = float(prev["SMA20"]), float(prev["SMA50"])
        f1, s1 = float(row["SMA20"]), float(row["SMA50"])
        signal = None
        if f0 <= s0 and f1 > s1:
            signal = "BUY"
            sl = c - 1.0 * atr
        elif f0 >= s0 and f1 < s1:
            signal = "SELL"
            sl = c + 1.0 * atr
        if not signal:
            continue
        ei_e = min(i + late_bars, len(df) - 2)
        entry = float(df.iloc[ei_e]["open"]) if late_bars else c
        risk = abs(entry - sl) * sl_widen
        if risk <= 0:
            continue
        if signal == "BUY":
            sl_e, tp = entry - risk, entry + rr * risk
        else:
            sl_e, tp = entry + risk, entry - rr * risk
        open_t = {
            "side": signal,
            "entry": entry,
            "sl": sl_e,
            "tp": tp,
            "entry_t": df.index[ei_e],
            "ei": ei_e,
        }
    return trades


def run_fib(
    df,
    *,
    fibs=(0.50, 0.618),
    start_i=0,
    end_i=None,
    late_bars=0,
    sl_widen=1.0,
    spread_r=0.0,
    skip_p=0.0,
    rng=None,
    rr=2.0,
):
    rng = rng or random.Random(0)
    end_i = end_i if end_i is not None else len(df) - 1
    trades = []
    open_t = None
    cooldown = -1
    cache = [None] * len(df)
    for i in range(40, len(df)):
        atr = float(df.iloc[i]["atr"] or 1)
        cache[i] = _impulse(df, i, atr)
    for ei in range(max(40, start_i), min(end_i, len(df) - 1) + 1):
        ts = df.index[ei]
        row = df.iloc[ei]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        atr = float(row["atr"] or 0) or max(h - l, 1e-6)
        if open_t is not None:
            open_t, closed = _sim_open(df, open_t, ei, start_i, spread_r, trades)
            if closed:
                cooldown = ei + 2
            continue
        if ei < cooldown:
            continue
        if skip_p > 0 and rng.random() < skip_p:
            continue
        imp = cache[ei]
        if not imp or (ts - imp["end_t"]).total_seconds() > 48 * 3600:
            continue
        zlo, zhi = _fib_zone(imp, atr, fibs)
        win = df.iloc[max(0, ei - 20) : ei]
        signal = None
        if imp["side"] == "BUY":
            liq = float(win["low"].min()) if len(win) else l
            if l < min(liq, zlo) and c > zlo and l <= zhi and c > float(row["open"]):
                signal = "BUY"
                entry, sl = c, min(l, liq) - 0.08 * atr
        else:
            liq = float(win["high"].max()) if len(win) else h
            if h > max(liq, zhi) and c < zhi and h >= zlo and c < float(row["open"]):
                signal = "SELL"
                entry, sl = c, max(h, liq) + 0.08 * atr
        if not signal:
            continue
        ei_e = min(ei + late_bars, len(df) - 2)
        if late_bars:
            entry = float(df.iloc[ei_e]["open"])
        risk = abs(entry - sl) * sl_widen
        if risk <= 0.12 * atr:
            continue
        if signal == "BUY":
            sl_e, tp = entry - risk, entry + rr * risk
        else:
            sl_e, tp = entry + risk, entry - rr * risk
        open_t = {
            "side": signal,
            "entry": entry,
            "sl": sl_e,
            "tp": tp,
            "entry_t": df.index[ei_e],
            "ei": ei_e,
        }
    return trades


def run_breakout(
    df,
    *,
    lookback=20,
    start_i=0,
    end_i=None,
    late_bars=0,
    sl_widen=1.0,
    spread_r=0.0,
    skip_p=0.0,
    rng=None,
    rr=2.0,
):
    """Donchian-style breakout of lookback high/low + ATR SL."""
    rng = rng or random.Random(0)
    end_i = end_i if end_i is not None else len(df) - 1
    trades = []
    open_t = None
    cooldown = -1
    for i in range(max(lookback + 5, start_i), min(end_i, len(df) - 1) + 1):
        row = df.iloc[i]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        atr = float(row["ATR"] or 0) or max(h - l, 1e-6)
        if open_t is not None:
            open_t, closed = _sim_open(df, open_t, i, start_i, spread_r, trades)
            if closed:
                cooldown = i + 3
            continue
        if i < cooldown:
            continue
        if skip_p > 0 and rng.random() < skip_p:
            continue
        window = df.iloc[i - lookback : i]
        hh, ll = float(window["high"].max()), float(window["low"].min())
        signal = None
        if c > hh and float(row["close"]) > float(row["open"]):
            signal = "BUY"
            sl = c - 1.2 * atr
        elif c < ll and float(row["close"]) < float(row["open"]):
            signal = "SELL"
            sl = c + 1.2 * atr
        if not signal:
            continue
        ei_e = min(i + late_bars, len(df) - 2)
        entry = float(df.iloc[ei_e]["open"]) if late_bars else c
        risk = abs(entry - sl) * sl_widen
        if risk <= 0:
            continue
        if signal == "BUY":
            sl_e, tp = entry - risk, entry + rr * risk
        else:
            sl_e, tp = entry + risk, entry - rr * risk
        open_t = {
            "side": signal,
            "entry": entry,
            "sl": sl_e,
            "tp": tp,
            "entry_t": df.index[ei_e],
            "ei": ei_e,
        }
    return trades


# Setup registry
SETUPS: list[tuple[str, Callable]] = [
    (
        "ema_20_50_live",
        lambda df, **kw: run_ema_generic(
            df, fast_col="EMA20", slow_col="EMA50", pull_atr=1.0, disp_atr=0.55, **kw
        ),
    ),
    (
        "ema_20_50_tight",
        lambda df, **kw: run_ema_generic(
            df, fast_col="EMA20", slow_col="EMA50", pull_atr=0.75, disp_atr=0.70, **kw
        ),
    ),
    (
        "ema_8_21",
        lambda df, **kw: run_ema_generic(
            df, fast_col="EMA8", slow_col="EMA21", pull_atr=1.0, disp_atr=0.55, **kw
        ),
    ),
    (
        "ema_9_21",
        lambda df, **kw: run_ema_generic(
            df, fast_col="EMA9", slow_col="EMA21", pull_atr=1.0, disp_atr=0.55, **kw
        ),
    ),
    (
        "ema_8_21_strict",
        lambda df, **kw: run_ema_generic(
            df, fast_col="EMA8", slow_col="EMA21", pull_atr=0.70, disp_atr=0.75, **kw
        ),
    ),
    ("sma_20_50_cross", lambda df, **kw: run_sma_cross(df, **kw)),
    ("fib_50_618", lambda df, **kw: run_fib(df, fibs=(0.50, 0.618), **kw)),
    ("fib_618_786", lambda df, **kw: run_fib(df, fibs=(0.618, 0.786), **kw)),
    ("breakout_20", lambda df, **kw: run_breakout(df, lookback=20, **kw)),
    ("breakout_40", lambda df, **kw: run_breakout(df, lookback=40, **kw)),
]


def _stats(trades: list[dict]) -> dict[str, Any]:
    if not trades:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "pnl": 0.0,
            "exp_r": 0.0,
            "max_dd": 0.0,
            "pf": 0.0,
        }
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    pnl = sum(t["pnl"] for t in trades)
    gp = sum(t["pnl"] for t in wins) or 0.0
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    eq = peak = dd = 0.0
    for t in trades:
        eq += t["pnl"]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    rs = [t["r"] for t in trades]
    return {
        "n": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": round(100.0 * len(wins) / max(1, len(wins) + len(losses)), 1),
        "pnl": round(pnl, 2),
        "exp_r": round(float(np.mean(rs)), 3),
        "max_dd": round(dd, 2),
        "pf": round(gp / gl, 2),
    }


def monte_carlo(trades, paths=MC_PATHS, seed=42):
    if len(trades) < 5:
        return {"paths": 0, "note": "too_few"}
    rs = np.array([t["r"] for t in trades], dtype=float)
    rng = np.random.default_rng(seed)
    finals, max_dds, ruin = [], [], 0
    for _ in range(paths):
        sample = rng.choice(rs, size=len(rs), replace=True)
        eq = np.cumsum(sample)
        peak = np.maximum.accumulate(eq)
        finals.append(float(eq[-1]))
        max_dds.append(float((eq - peak).min()))
        if eq.min() < -10.0:
            ruin += 1
    fa = np.array(finals)
    return {
        "paths": paths,
        "median_R": round(float(np.median(fa)), 3),
        "p5_R": round(float(np.percentile(fa, 5)), 3),
        "prob_profit": round(100.0 * float(np.mean(fa > 0)), 1),
        "prob_ruin_10R": round(100.0 * ruin / paths, 1),
    }


def verdict(full, oos, wf, torture, mc) -> str:
    score = 0
    if full.get("n", 0) >= 8 and full.get("exp_r", 0) > 0.05:
        score += 2
    elif full.get("exp_r", 0) > 0:
        score += 1
    if oos.get("n", 0) >= 4 and oos.get("exp_r", 0) > 0:
        score += 2
    elif oos.get("exp_r", 0) > -0.05:
        score += 1
    if wf:
        green = sum(1 for f in wf if f.get("oos", {}).get("exp_r", 0) > 0)
        if green >= max(1, len(wf) // 2 + 1):
            score += 2
        elif green >= 1:
            score += 1
    if torture:
        ok = sum(
            1
            for v in torture.values()
            if isinstance(v, dict) and v.get("exp_r", -9) > -0.05
        )
        if ok >= max(2, len(torture) // 2):
            score += 2
        elif ok >= 1:
            score += 1
    if mc.get("paths"):
        if mc.get("prob_profit", 0) >= 60 and mc.get("p5_R", -99) > -2:
            score += 2
        elif mc.get("prob_profit", 0) >= 50:
            score += 1
    if score >= 8:
        return "SURVIVES"
    if score >= 4:
        return "FRAGILE"
    return "DEAD"


def battery(df, gen_fn) -> dict:
    n = len(df)

    def gen(start_i=0, end_i=None, **kw):
        return gen_fn(df, start_i=start_i, end_i=end_i, **kw)

    full_tr = gen()
    full = _stats(full_tr)
    split = int(n * 0.70)
    oos = _stats(gen(start_i=split))
    wf = []
    for k in range(4):
        ts = int(n * (0.40 + 0.15 * k))
        te = int(n * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 80:
            continue
        wf.append(
            {
                "fold": k + 1,
                "oos": _stats(gen(start_i=ts, end_i=te)),
            }
        )
    torture = {}
    for name, kw in [
        ("spread_0.15R", {"spread_r": 0.15}),
        ("late_1bar", {"late_bars": 1}),
        ("skip_30pct", {"skip_p": 0.30}),
        ("sl_widen_20pct", {"sl_widen": 1.20}),
    ]:
        torture[name] = _stats(gen(rng=random.Random(hash(name) % 9999), **kw))
    mc = monte_carlo(full_tr)
    return {
        "full": full,
        "oos": oos,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "verdict": verdict(full, oos, wf, torture, mc),
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"Strategy hunt (FRAGILE/DEAD) {start}→{end_s}  $200/R · 1h · no live changes",
        flush=True,
    )

    by_symbol: dict[str, Any] = {}
    leaders: list[dict] = []

    for sym in TARGETS:
        print(f"\n######## {sym} ########", flush=True)
        try:
            raw = _fetch_1h(sym, start, end_s)
            df = _prep(raw)
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if len(df) < 200:
                print(f"  SKIP short {len(df)}", flush=True)
                continue
        except Exception as exc:
            print(f"  FAIL fetch: {exc}", flush=True)
            continue

        rows = []
        for name, fn in SETUPS:
            try:
                res = battery(df, fn)
                row = {"setup": name, **res}
                rows.append(row)
                print(
                    f"  {name:20} n={res['full']['n']:3} E[R]={res['full']['exp_r']:+.3f} "
                    f"OOS={res['oos']['exp_r']:+.3f} MC%={res['monte_carlo'].get('prob_profit', 0):.0f} "
                    f"→ {res['verdict']}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  {name}: FAIL {exc}", flush=True)

        # Rank: SURVIVES first, then by OOS E[R], MC profit, full E[R]
        rank_key = {"SURVIVES": 2, "FRAGILE": 1, "DEAD": 0}
        ranked = sorted(
            rows,
            key=lambda r: (
                rank_key.get(r["verdict"], 0),
                r["oos"]["exp_r"],
                r["monte_carlo"].get("prob_profit", 0),
                r["full"]["exp_r"],
            ),
            reverse=True,
        )
        by_symbol[sym] = {"ranked": ranked, "best": ranked[0] if ranked else None}
        if ranked:
            b = ranked[0]
            leaders.append(
                {
                    "symbol": sym,
                    "best_setup": b["setup"],
                    "verdict": b["verdict"],
                    "full_exp_r": b["full"]["exp_r"],
                    "oos_exp_r": b["oos"]["exp_r"],
                    "mc_prob_profit": b["monte_carlo"].get("prob_profit"),
                    "beats_live": b["setup"] != "ema_20_50_live"
                    and rank_key.get(b["verdict"], 0)
                    > rank_key.get(
                        next(
                            (
                                x["verdict"]
                                for x in ranked
                                if x["setup"] == "ema_20_50_live"
                            ),
                            "DEAD",
                        ),
                        0,
                    ),
                }
            )

    payload = {
        "window": {"start": start, "end": end_s, "tf": "1h_proxy"},
        "risk_usd": RISK_USD,
        "note": "RESEARCH only — no live changes",
        "by_symbol": {
            s: {
                "best": v["best"]["setup"] if v.get("best") else None,
                "best_verdict": v["best"]["verdict"] if v.get("best") else None,
                "ranked": [
                    {
                        "setup": r["setup"],
                        "verdict": r["verdict"],
                        "full": r["full"],
                        "oos": r["oos"],
                        "mc": r["monte_carlo"],
                        "wf_green": sum(
                            1 for f in r["walk_forward"] if f["oos"]["exp_r"] > 0
                        ),
                        "wf_n": len(r["walk_forward"]),
                    }
                    for r in v["ranked"]
                ],
            }
            for s, v in by_symbol.items()
        },
        "leaders": leaders,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### BEST SETUP PER SYMBOL (research)\n", flush=True)
    print(
        "| Symbol | Best setup | Verdict | Full E[R] | OOS E[R] | MC P(profit) | vs live EMA |",
        flush=True,
    )
    print("|---|---|---|---:|---:|---:|---|", flush=True)
    rk = {"SURVIVES": 2, "FRAGILE": 1, "DEAD": 0}
    for L in leaders:
        live = next(
            (
                r
                for r in by_symbol[L["symbol"]]["ranked"]
                if r["setup"] == "ema_20_50_live"
            ),
            None,
        )
        vs = "same"
        if live and L["best_setup"] != "ema_20_50_live":
            if rk.get(L["verdict"], 0) > rk.get(live["verdict"], 0):
                vs = "UPGRADE"
            elif L["oos_exp_r"] > live["oos"]["exp_r"] + 0.05:
                vs = "oos+"
            else:
                vs = "alt"
        print(
            f"| {L['symbol']} | {L['best_setup']} | **{L['verdict']}** | "
            f"{L['full_exp_r']} | {L['oos_exp_r']} | {L['mc_prob_profit']}% | {vs} |",
            flush=True,
        )

    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
