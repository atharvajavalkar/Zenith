#!/usr/bin/env python3
"""RESEARCH ONLY — Edge survival battery for the live Zenith book.

For every symbol (GOLD fib · book EMA):
  1. Full-sample paper ($200/R, forced 2R)
  2. Out-of-sample (last 30% held out)
  3. Walk-forward (4 anchored folds)
  4. Torture (spread↑, late entry, skip 30%, wider SL, param jitter)
  5. Monte Carlo (2000 bootstrap equity paths on trade R sequence)

Data: yfinance 1h ~6m (5m cannot cover OOS/WF honestly).
Live brains proxied on 1h — same logic, coarser bars.
No live wiring.
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = Path("/tmp/edge_survive_bt.json")
RISK_USD = 200.0
MC_PATHS = 2000
SWING = 3

BOOK = ("GOLD", "USDCHF", "USDJPY", "NAS100", "UKOIL", "AUDUSD", "BTCUSD", "ETHUSD")
# Post edge-hunt upgrades (same map as live SYMBOL_ENTRY_BRAIN)
BRAINS = {
    "GOLD": "cta_10_40",
    "USDCHF": "rsi",
    "USDJPY": "bb_1.5",
    "NAS100": "ema_20_50_tight",
    "UKOIL": "st_3",
    "AUDUSD": "bb_1.5",
    "BTCUSD": "fib_liq",
    "ETHUSD": "bb_2",
}
YF = {
    "GOLD": "GC=F",
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
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["EMA20"] = out["ema20"]
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["EMA50"] = out["ema50"]
    out["SMA20"] = out["close"].rolling(20).mean()
    out["SMA50"] = out["close"].rolling(50).mean()
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


def _fib_zone(imp, atr, fibs=(0.50, 0.618)):
    a, b = float(imp["a"]), float(imp["b"])
    if imp["side"] == "BUY":
        levels = [b - r * (b - a) for r in fibs]
    else:
        levels = [b + r * (a - b) for r in fibs]
    pad = 0.40 * atr
    return min(levels) - pad, max(levels) + pad


def run_fib_trades(
    df: pd.DataFrame,
    *,
    start_i: int = 0,
    end_i: Optional[int] = None,
    fibs=(0.50, 0.618),
    late_bars: int = 0,
    sl_widen: float = 1.0,
    spread_r: float = 0.0,
    skip_p: float = 0.0,
    rr: float = 2.0,
    rng: Optional[random.Random] = None,
    **_kw,
) -> list[dict]:
    rng = rng or random.Random(0)
    end_i = end_i if end_i is not None else len(df) - 1
    trades: list[dict] = []
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
                open_t = None
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
        # late entry torture
        ei_e = min(ei + late_bars, len(df) - 2)
        if late_bars:
            entry = float(df.iloc[ei_e]["open"])
        risk = abs(entry - sl) * sl_widen
        if risk <= 0.12 * atr:
            continue
        rr_m = float(rr) if rr and rr > 0 else 2.0
        if signal == "BUY":
            sl_e = entry - risk
            tp = entry + rr_m * risk
        else:
            sl_e = entry + risk
            tp = entry - rr_m * risk
        open_t = {
            "side": signal,
            "entry": entry,
            "sl": sl_e,
            "tp": tp,
            "entry_t": df.index[ei_e],
            "ei": ei_e,
        }
    return trades


def run_ema_trades(
    df: pd.DataFrame,
    *,
    start_i: int = 0,
    end_i: Optional[int] = None,
    pull_atr: float = 1.0,
    disp_atr: float = 0.55,
    late_bars: int = 0,
    sl_widen: float = 1.0,
    spread_r: float = 0.0,
    skip_p: float = 0.0,
    rr: float = 2.0,
    rng: Optional[random.Random] = None,
    **_kw,
) -> list[dict]:
    from strategy.ema_pull_brain import evaluate_ema_2050_pull

    rng = rng or random.Random(0)
    end_i = end_i if end_i is not None else len(df) - 1
    trades: list[dict] = []
    open_t = None
    cooldown = -1
    for i in range(max(60, start_i), min(end_i, len(df) - 1) + 1):
        row = df.iloc[i]
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
                open_t = None
                cooldown = i + 2
            continue
        if i < cooldown:
            continue
        if skip_p > 0 and rng.random() < skip_p:
            continue
        hit = evaluate_ema_2050_pull(
            df.iloc[: i + 1], pull_atr=pull_atr, disp_atr=disp_atr
        )
        if not hit:
            continue
        ei_e = min(i + late_bars, len(df) - 2)
        entry = float(df.iloc[ei_e]["open"]) if late_bars else float(hit["entry"])
        sl = float(hit["sl"])
        risk0 = abs(entry - sl)
        if risk0 <= 0:
            continue
        risk = risk0 * sl_widen
        side = hit["signal"]
        rr_m = float(rr) if rr and rr > 0 else 2.0
        if side == "BUY":
            sl_e = entry - risk
            tp = entry + rr_m * risk
        else:
            sl_e = entry + risk
            tp = entry - rr_m * risk
        open_t = {
            "side": side,
            "entry": entry,
            "sl": sl_e,
            "tp": tp,
            "entry_t": df.index[ei_e],
            "ei": ei_e,
        }
    return trades


def run_sma_trades(
    df: pd.DataFrame,
    *,
    start_i: int = 0,
    end_i: Optional[int] = None,
    late_bars: int = 0,
    sl_widen: float = 1.0,
    spread_r: float = 0.0,
    skip_p: float = 0.0,
    rr: float = 2.0,
    rng: Optional[random.Random] = None,
    **_kw,
) -> list[dict]:
    from strategy.sma_cross_brain import evaluate_sma_2050_cross

    rng = rng or random.Random(0)
    end_i = end_i if end_i is not None else len(df) - 1
    trades: list[dict] = []
    open_t = None
    cooldown = -1
    for i in range(max(60, start_i), min(end_i, len(df) - 1) + 1):
        row = df.iloc[i]
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
                open_t = None
                cooldown = i + 2
            continue
        if i < cooldown:
            continue
        if skip_p > 0 and rng.random() < skip_p:
            continue
        hit = evaluate_sma_2050_cross(df.iloc[: i + 1])
        if not hit:
            continue
        ei_e = min(i + late_bars, len(df) - 2)
        entry = float(df.iloc[ei_e]["open"]) if late_bars else float(hit["entry"])
        sl = float(hit["sl"])
        risk0 = abs(entry - sl)
        if risk0 <= 0:
            continue
        risk = risk0 * sl_widen
        side = hit["signal"]
        rr_m = float(rr) if rr and rr > 0 else 2.0
        if side == "BUY":
            sl_e, tp = entry - risk, entry + rr_m * risk
        else:
            sl_e, tp = entry + risk, entry - rr_m * risk
        open_t = {
            "side": side,
            "entry": entry,
            "sl": sl_e,
            "tp": tp,
            "entry_t": df.index[ei_e],
            "ei": ei_e,
        }
    return trades


def run_breakout_trades(
    df: pd.DataFrame,
    *,
    start_i: int = 0,
    end_i: Optional[int] = None,
    lookback: int = 20,
    late_bars: int = 0,
    sl_widen: float = 1.0,
    spread_r: float = 0.0,
    skip_p: float = 0.0,
    rr: float = 2.0,
    rng: Optional[random.Random] = None,
    **_kw,
) -> list[dict]:
    from strategy.breakout_brain import evaluate_breakout

    rng = rng or random.Random(0)
    end_i = end_i if end_i is not None else len(df) - 1
    trades: list[dict] = []
    open_t = None
    cooldown = -1
    for i in range(max(lookback + 10, start_i), min(end_i, len(df) - 1) + 1):
        row = df.iloc[i]
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
                open_t = None
                cooldown = i + 2
            continue
        if i < cooldown:
            continue
        if skip_p > 0 and rng.random() < skip_p:
            continue
        hit = evaluate_breakout(df.iloc[: i + 1], lookback=lookback)
        if not hit:
            continue
        ei_e = min(i + late_bars, len(df) - 2)
        entry = float(df.iloc[ei_e]["open"]) if late_bars else float(hit["entry"])
        sl = float(hit["sl"])
        risk0 = abs(entry - sl)
        if risk0 <= 0:
            continue
        risk = risk0 * sl_widen
        side = hit["signal"]
        rr_m = float(rr) if rr and rr > 0 else 2.0
        if side == "BUY":
            sl_e, tp = entry - risk, entry + rr_m * risk
        else:
            sl_e, tp = entry + risk, entry - rr_m * risk
        open_t = {
            "side": side,
            "entry": entry,
            "sl": sl_e,
            "tp": tp,
            "entry_t": df.index[ei_e],
            "ei": ei_e,
        }
    return trades


def _stats(trades: list[dict]) -> dict[str, Any]:
    if not trades:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "pnl": 0.0,
            "avg_r": 0.0,
            "exp_r": 0.0,
            "max_dd": 0.0,
            "pf": 0.0,
        }
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    pnl = sum(t["pnl"] for t in trades)
    gp = sum(t["pnl"] for t in wins) or 0.0
    gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
    eq = 0.0
    peak = 0.0
    dd = 0.0
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
        "avg_r": round(float(np.mean(rs)), 3),
        "exp_r": round(float(np.mean(rs)), 3),
        "max_dd": round(dd, 2),
        "pf": round(gp / gl, 2),
    }


def monte_carlo(trades: list[dict], paths: int = MC_PATHS, seed: int = 42) -> dict:
    if len(trades) < 5:
        return {"paths": 0, "note": "too_few_trades"}
    rs = np.array([t["r"] for t in trades], dtype=float)
    rng = np.random.default_rng(seed)
    finals = []
    max_dds = []
    ruin = 0  # equity < -10R cumulative
    for _ in range(paths):
        sample = rng.choice(rs, size=len(rs), replace=True)
        eq = np.cumsum(sample)
        peak = np.maximum.accumulate(eq)
        dd = float((eq - peak).min())
        finals.append(float(eq[-1]))
        max_dds.append(dd)
        if eq.min() < -10.0:
            ruin += 1
    finals_a = np.array(finals)
    return {
        "paths": paths,
        "median_R": round(float(np.median(finals_a)), 3),
        "p5_R": round(float(np.percentile(finals_a, 5)), 3),
        "p95_R": round(float(np.percentile(finals_a, 95)), 3),
        "mean_R": round(float(np.mean(finals_a)), 3),
        "prob_profit": round(100.0 * float(np.mean(finals_a > 0)), 1),
        "prob_ruin_10R": round(100.0 * ruin / paths, 1),
        "median_maxDD_R": round(float(np.median(max_dds)), 3),
        "p5_maxDD_R": round(float(np.percentile(max_dds, 5)), 3),
    }


def verdict(full: dict, oos: dict, wf: list, torture: dict, mc: dict) -> str:
    """SURVIVES / FRAGILE / DEAD."""
    score = 0
    # Full edge
    if full.get("n", 0) >= 8 and full.get("exp_r", 0) > 0.05:
        score += 2
    elif full.get("exp_r", 0) > 0:
        score += 1
    # OOS
    if oos.get("n", 0) >= 4 and oos.get("exp_r", 0) > 0:
        score += 2
    elif oos.get("exp_r", 0) > -0.05:
        score += 1
    # Walk-forward: majority of folds green E[R]
    if wf:
        green = sum(1 for f in wf if f.get("oos", {}).get("exp_r", 0) > 0)
        if green >= max(1, len(wf) // 2 + 1):
            score += 2
        elif green >= 1:
            score += 1
    # Torture: majority of stresses still E[R] > -0.05
    if torture:
        ok = sum(1 for v in torture.values() if isinstance(v, dict) and v.get("exp_r", -9) > -0.05)
        if ok >= max(2, len(torture) // 2):
            score += 2
        elif ok >= 1:
            score += 1
    # MC
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


def battery_for(symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    brain = BRAINS.get(symbol.upper(), "ema_20_50_pull")
    n = len(df)
    print(f"  {symbol}: bars={n} brain={brain}", flush=True)

    def gen(start_i=0, end_i=None, **kw):
        if brain == "fib_liq":
            return run_fib_trades(df, start_i=start_i, end_i=end_i, **kw)
        if brain == "sma_20_50_cross":
            return run_sma_trades(df, start_i=start_i, end_i=end_i, **kw)
        if brain == "breakout_20":
            return run_breakout_trades(df, start_i=start_i, end_i=end_i, **kw)
        pull = 0.75 if brain == "ema_20_50_tight" else 1.0
        disp = 0.70 if brain == "ema_20_50_tight" else 0.55
        # Allow torture to override pull/disp
        pull = float(kw.pop("pull_atr", pull))
        disp = float(kw.pop("disp_atr", disp))
        return run_ema_trades(
            df, start_i=start_i, end_i=end_i, pull_atr=pull, disp_atr=disp, **kw
        )

    # 1) Full sample
    full_tr = gen()
    full = _stats(full_tr)
    print(f"    full n={full['n']} wr={full['wr']}% E[R]={full['exp_r']} pnl=${full['pnl']}", flush=True)

    # 2) OOS — last 30%
    split = int(n * 0.70)
    is_tr = gen(end_i=split - 1)
    oos_tr = gen(start_i=split)
    is_st = _stats(is_tr)
    oos_st = _stats(oos_tr)
    print(
        f"    OOS  IS n={is_st['n']} E[R]={is_st['exp_r']} | "
        f"OOS n={oos_st['n']} E[R]={oos_st['exp_r']} pnl=${oos_st['pnl']}",
        flush=True,
    )

    # 3) Walk-forward — 4 anchored folds (train grows, test next 15%)
    wf = []
    for k in range(4):
        test_start = int(n * (0.40 + 0.15 * k))
        test_end = int(n * (0.55 + 0.15 * k)) - 1
        if test_end <= test_start or test_start < 80:
            continue
        train_tr = gen(end_i=test_start - 1)
        test_tr = gen(start_i=test_start, end_i=test_end)
        fold = {
            "fold": k + 1,
            "train_end_i": test_start - 1,
            "test_i": [test_start, test_end],
            "train": _stats(train_tr),
            "oos": _stats(test_tr),
        }
        wf.append(fold)
        print(
            f"    WF{k+1} train E[R]={fold['train']['exp_r']} "
            f"→ test n={fold['oos']['n']} E[R]={fold['oos']['exp_r']} "
            f"pnl=${fold['oos']['pnl']}",
            flush=True,
        )

    # 4) Torture
    torture = {}
    specs = [
        ("spread_0.15R", {"spread_r": 0.15}),
        ("late_1bar", {"late_bars": 1}),
        ("skip_30pct", {"skip_p": 0.30}),
        ("sl_widen_20pct", {"sl_widen": 1.20}),
    ]
    if brain == "fib_liq":
        specs += [("fib_deep", {"fibs": (0.618, 0.786)})]
    elif brain.startswith("ema_"):
        specs += [
            ("pull_tight", {"pull_atr": 0.75, "disp_atr": 0.55}),
            ("disp_strict", {"pull_atr": 1.0, "disp_atr": 0.75}),
        ]
    elif brain == "breakout_20":
        specs += [("breakout_40", {"lookback": 40})]
    for name, kw in specs:
        rng = random.Random(hash(name) % 10_000)
        tr = gen(rng=rng, **kw)
        st = _stats(tr)
        torture[name] = st
        print(
            f"    TORT {name}: n={st['n']} E[R]={st['exp_r']} pnl=${st['pnl']}",
            flush=True,
        )

    # 5) Monte Carlo on full-sample R sequence
    mc = monte_carlo(full_tr)
    if mc.get("paths"):
        print(
            f"    MC medianR={mc['median_R']} p5={mc['p5_R']} "
            f"P(profit)={mc['prob_profit']}% P(ruin10R)={mc['prob_ruin_10R']}%",
            flush=True,
        )

    v = verdict(full, oos_st, wf, torture, mc)
    print(f"    → VERDICT {v}", flush=True)
    return {
        "symbol": symbol,
        "brain": brain,
        "bars": n,
        "full": full,
        "in_sample": is_st,
        "oos": oos_st,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "verdict": v,
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"Edge survival battery {start} → {end_s}  "
        f"($200/R · 1h proxy · UPGRADED brains)",
        flush=True,
    )

    results = []
    for sym in BOOK:
        try:
            raw = _fetch_1h(sym, start, end_s)
            df = _prep(raw)
            # Restrict to window
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if len(df) < 200:
                print(f"  SKIP {sym}: only {len(df)} bars", flush=True)
                continue
            results.append(battery_for(sym, df))
        except Exception as exc:
            print(f"  FAIL {sym}: {exc}", flush=True)
            results.append({"symbol": sym, "error": str(exc), "verdict": "DEAD"})

    # Desk summary
    survives = [r for r in results if r.get("verdict") == "SURVIVES"]
    fragile = [r for r in results if r.get("verdict") == "FRAGILE"]
    dead = [r for r in results if r.get("verdict") == "DEAD"]

    payload = {
        "window": {"start": start, "end": end_s, "tf": "1h_proxy"},
        "risk_usd": RISK_USD,
        "mc_paths": MC_PATHS,
        "results": results,
        "summary": {
            "survives": [r["symbol"] for r in survives],
            "fragile": [r["symbol"] for r in fragile],
            "dead": [r["symbol"] for r in dead],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### EDGE SURVIVAL SCORECARD\n", flush=True)
    print("| Symbol | Brain | Full E[R] | OOS E[R] | WF green | MC P(profit) | Verdict |", flush=True)
    print("|---|---|---:|---:|---:|---:|---|", flush=True)
    for r in results:
        if r.get("error"):
            print(f"| {r['symbol']} | — | — | — | — | — | DEAD |", flush=True)
            continue
        wf_g = sum(1 for f in r.get("walk_forward") or [] if f["oos"]["exp_r"] > 0)
        wf_n = len(r.get("walk_forward") or [])
        mc = r.get("monte_carlo") or {}
        print(
            f"| {r['symbol']} | {r['brain']} | {r['full']['exp_r']} | "
            f"{r['oos']['exp_r']} | {wf_g}/{wf_n} | "
            f"{mc.get('prob_profit', '—')}% | **{r['verdict']}** |",
            flush=True,
        )
    print(
        f"\nSURVIVES: {payload['summary']['survives']}\n"
        f"FRAGILE:  {payload['summary']['fragile']}\n"
        f"DEAD:     {payload['summary']['dead']}",
        flush=True,
    )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
