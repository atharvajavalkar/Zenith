#!/usr/bin/env python3
"""Strategy Lab — expanded HF + retail bake-off on ~15y daily data.

Writes data/strategy_lab.json. Does NOT change live brains.

Note: 1h cannot cover 15y on Yahoo; lab uses daily bars (honest long history).
Crypto shorter listing → max available history.
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
OUT = ROOT / "data" / "strategy_lab.json"
DEFAULT_YEARS = 15
DEFAULT_DAYS = int(DEFAULT_YEARS * 365.25)

from analysis._edge_survive_bt import (  # noqa: E402
    BOOK,
    RISK_USD,
    SWING,
    YF,
    _prep,
    _stats,
    _swing_high,
    _swing_low,
    monte_carlo,
    verdict,
)
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_breakout,
    _collect_ema,
    _collect_fib,
    _collect_sma,
    simulate,
)
from analysis.levy_malliavin import levy_malliavin_pack  # noqa: E402
from analysis.stochastic_pack import stochastic_pack  # noqa: E402
from config import SYMBOL_RR, symbol_rr  # noqa: E402

# ── catalog ─────────────────────────────────────────────────────────────────

STRATEGIES: dict[str, dict[str, Any]] = {
    # Retail / discretionary proxies
    "fib_liq": {"family": "retail", "label": "Fib liquidity sweep", "style": "SMC / fib"},
    "ema_pull": {"family": "retail", "label": "EMA 20/50 pullback", "style": "trend pull"},
    "ema_tight": {"family": "retail", "label": "EMA tight pull", "style": "scalp pull"},
    "sma_cross": {"family": "retail", "label": "SMA 20/50 cross", "style": "classic cross"},
    "donchian_20": {"family": "retail", "label": "Donchian 20 breakout", "style": "breakout"},
    "price_action": {"family": "retail", "label": "Pin / engulf PA", "style": "price action"},
    "liq_sweep": {"family": "retail", "label": "Swing liquidity sweep", "style": "SMC"},
    "trendline": {"family": "retail", "label": "Trendline break", "style": "structure"},
    "ob_fvg": {"family": "retail", "label": "OB ∩ FVG", "style": "ICT / SMC"},
    "macd_cross": {"family": "retail", "label": "MACD signal cross", "style": "momentum"},
    "stoch_cross": {"family": "retail", "label": "Stochastic cross", "style": "oscillator"},
    "engulfing": {"family": "retail", "label": "Engulfing candle", "style": "price action"},
    "inside_break": {"family": "retail", "label": "Inside-bar breakout", "style": "compression"},
    "supertrend": {"family": "retail", "label": "Supertrend flip", "style": "ATR trail"},
    "ichimoku_tk": {"family": "retail", "label": "Ichimoku TK cross", "style": "Ichimoku"},
    "keltner_break": {"family": "retail", "label": "Keltner channel break", "style": "volatility"},
    "adx_ema": {"family": "retail", "label": "ADX + EMA trend", "style": "trend filter"},
    "psar": {"family": "retail", "label": "Parabolic SAR flip", "style": "trail stop"},
    # Hedge / systematic proxies
    "cta_dual_ma": {"family": "hedge", "label": "CTA dual MA (50/200)", "style": "trend follow"},
    "cta_fast": {"family": "hedge", "label": "CTA fast MA (10/40)", "style": "trend follow"},
    "turtle_55": {"family": "hedge", "label": "Turtle Donchian 55", "style": "CTA classic"},
    "turtle_20": {"family": "hedge", "label": "Turtle Donchian 20", "style": "CTA"},
    "tsmom": {"family": "hedge", "label": "Time-series momentum 12-1", "style": "AQR-style TSM"},
    "mom_roc": {"family": "hedge", "label": "Momentum ROC (60d)", "style": "momentum"},
    "bb_meanrev": {"family": "hedge", "label": "Bollinger mean reversion", "style": "stat arb proxy"},
    "zscore_mr": {"family": "hedge", "label": "Z-score mean reversion", "style": "stat arb"},
    "rsi_reversion": {"family": "hedge", "label": "RSI mean reversion", "style": "mean rev"},
    "atr_vol_break": {"family": "hedge", "label": "ATR volatility breakout", "style": "vol breakout"},
    "vol_scaled_mom": {"family": "hedge", "label": "Vol-scaled momentum", "style": "risk parity mom"},
    "channel_100": {"family": "hedge", "label": "100-day channel break", "style": "long CTA"},
}


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(hour=23, minute=59, tzinfo=timezone.utc)
    return df[(df.index >= start_dt) & (df.index <= end_dt)].copy()


def _fetch_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    warm = (datetime.fromisoformat(start) - timedelta(days=400)).strftime("%Y-%m-%d")
    end_p = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    import yfinance as yf

    df = None
    last_err = None
    for ticker in (YF.get(symbol, symbol), f"{symbol}=X", symbol):
        try:
            df = yf.Ticker(ticker).history(
                start=warm, end=end_p, interval="1d", auto_adjust=True
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_err = e
            df = None
    if df is None or df.empty:
        raise RuntimeError(f"no daily {symbol}: {last_err}")
    df = df.rename(columns=str.lower)
    df = df[[c for c in ("open", "high", "low", "close") if c in df.columns]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _entry(i, side, entry, sl, idx) -> dict:
    return {"ei": i, "side": side, "entry": float(entry), "sl": float(sl), "entry_t": idx}


# ── collectors (existing + new) ─────────────────────────────────────────────

def _collect_price_action(df: pd.DataFrame) -> list[dict]:
    entries: list[dict] = []
    for i in range(60, len(df) - 1):
        row, prev = df.iloc[i], df.iloc[i - 1]
        atr = float(row["atr"] or 0) or 1e-6
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        po, pc = float(prev["open"]), float(prev["close"])
        body, rng = abs(c - o), max(h - l, 1e-12)
        upper, lower = h - max(c, o), min(c, o) - l
        bull_pin = lower >= 2.0 * body and lower >= 0.55 * rng and c > o
        bear_pin = upper >= 2.0 * body and upper >= 0.55 * rng and c < o
        bull_eng = c > o and pc < po and c >= po and o <= pc and body >= 0.35 * atr
        bear_eng = c < o and pc > po and c <= po and o >= pc and body >= 0.35 * atr
        win = df.iloc[max(0, i - 12) : i + 1]
        near_lo = l <= float(win["low"].min()) + 0.15 * atr
        near_hi = h >= float(win["high"].max()) - 0.15 * atr
        e20, e50 = float(row["ema20"]), float(row["ema50"])
        if e20 > e50 and near_lo and (bull_pin or bull_eng):
            if abs(c - (l - 0.08 * atr)) > 0.12 * atr:
                entries.append(_entry(i, "BUY", c, l - 0.08 * atr, df.index[i]))
        elif e20 < e50 and near_hi and (bear_pin or bear_eng):
            if abs(c - (h + 0.08 * atr)) > 0.12 * atr:
                entries.append(_entry(i, "SELL", c, h + 0.08 * atr, df.index[i]))
    return entries


def _collect_liq_sweep(df: pd.DataFrame) -> list[dict]:
    entries: list[dict] = []
    lows, highs = [], []
    for i in range(SWING, len(df) - SWING):
        if _swing_low(df, i):
            lows.append((i, float(df.iloc[i]["low"])))
        if _swing_high(df, i):
            highs.append((i, float(df.iloc[i]["high"])))
    for ei in range(40, len(df) - 1):
        row = df.iloc[ei]
        atr = float(row["atr"] or 0) or 1e-6
        h, l, c, o = float(row["high"]), float(row["low"]), float(row["close"]), float(row["open"])
        rlo = [lv for si, lv in lows if 5 <= ei - si <= 80]
        rhi = [lv for si, lv in highs if 5 <= ei - si <= 80]
        if rlo:
            liq = min(rlo)
            if l < liq - 0.02 * atr and c > liq and c > o and abs(c - (l - 0.08 * atr)) > 0.12 * atr:
                entries.append(_entry(ei, "BUY", c, l - 0.08 * atr, df.index[ei]))
                continue
        if rhi:
            liq = max(rhi)
            if h > liq + 0.02 * atr and c < liq and c < o and abs(c - (h + 0.08 * atr)) > 0.12 * atr:
                entries.append(_entry(ei, "SELL", c, h + 0.08 * atr, df.index[ei]))
    return entries


def _collect_trendline(df: pd.DataFrame) -> list[dict]:
    entries: list[dict] = []
    lows, highs = [], []
    for i in range(SWING, len(df) - SWING):
        if _swing_low(df, i):
            lows.append((i, float(df.iloc[i]["low"])))
        if _swing_high(df, i):
            highs.append((i, float(df.iloc[i]["high"])))
    for ei in range(80, len(df) - 1):
        row = df.iloc[ei]
        atr = float(row["atr"] or 0) or 1e-6
        c, o, h, l = float(row["close"]), float(row["open"]), float(row["high"]), float(row["low"])
        lo_piv = [(si, lv) for si, lv in lows if si < ei - 2]
        hi_piv = [(si, lv) for si, lv in highs if si < ei - 2]
        signal = entry = sl = None
        if len(lo_piv) >= 2:
            (i1, y1), (i2, y2) = lo_piv[-2], lo_piv[-1]
            if i2 > i1 and y2 > y1:
                slope = (y2 - y1) / (i2 - i1)
                line = y2 + slope * (ei - i2)
                prev_c = float(df.iloc[ei - 1]["close"])
                prev_line = y2 + slope * (ei - 1 - i2)
                if prev_c <= prev_line + 0.05 * atr and c > line + 0.05 * atr and c > o:
                    signal, entry, sl = "BUY", c, min(l, line) - 0.08 * atr
        if signal is None and len(hi_piv) >= 2:
            (i1, y1), (i2, y2) = hi_piv[-2], hi_piv[-1]
            if i2 > i1 and y2 < y1:
                slope = (y2 - y1) / (i2 - i1)
                line = y2 + slope * (ei - i2)
                prev_c = float(df.iloc[ei - 1]["close"])
                prev_line = y2 + slope * (ei - 1 - i2)
                if prev_c >= prev_line - 0.05 * atr and c < line - 0.05 * atr and c < o:
                    signal, entry, sl = "SELL", c, max(h, line) + 0.08 * atr
        if not signal:
            continue
        risk = abs(entry - sl)
        if 0.15 * atr < risk <= 2.5 * atr:
            entries.append(_entry(ei, signal, entry, sl, df.index[ei]))
    return entries


def _collect_ob_fvg(df: pd.DataFrame) -> list[dict]:
    bull_fvg, bear_fvg, obs = [], [], []
    for i in range(5, len(df)):
        row = df.iloc[i]
        atr = float(row["atr"] or 0) or 1e-6
        if float(row["low"]) > float(df.iloc[i - 2]["high"]):
            bull_fvg.append(
                {"i": i, "low": float(df.iloc[i - 2]["high"]), "high": float(row["low"]), "side": "BUY"}
            )
        if float(row["high"]) < float(df.iloc[i - 2]["low"]):
            bear_fvg.append(
                {"i": i, "low": float(row["high"]), "high": float(df.iloc[i - 2]["low"]), "side": "SELL"}
            )
        body = abs(float(row["close"]) - float(row["open"]))
        if float(row["close"]) > float(row["open"]) and body >= 1.15 * atr:
            for k in range(1, 5):
                j = i - k
                if j < 1:
                    break
                prev = df.iloc[j]
                if float(prev["close"]) < float(prev["open"]):
                    obs.append(
                        {"i": i, "low": float(prev["low"]), "high": float(prev["high"]), "side": "BUY"}
                    )
                    break
        if float(row["close"]) < float(row["open"]) and body >= 1.15 * atr:
            for k in range(1, 5):
                j = i - k
                if j < 1:
                    break
                prev = df.iloc[j]
                if float(prev["close"]) > float(prev["open"]):
                    obs.append(
                        {"i": i, "low": float(prev["low"]), "high": float(prev["high"]), "side": "SELL"}
                    )
                    break
    entries: list[dict] = []
    for ei in range(60, len(df) - 1):
        row = df.iloc[ei]
        atr = float(row["atr"] or 0) or 1e-6
        h, l, c, o = float(row["high"]), float(row["low"]), float(row["close"]), float(row["open"])
        for side, fvgs in (("BUY", bull_fvg), ("SELL", bear_fvg)):
            recent_f = [z for z in fvgs if 0 <= ei - z["i"] <= 40]
            recent_o = [z for z in obs if z["side"] == side and 0 <= ei - z["i"] <= 40]
            if not recent_f or not recent_o:
                continue
            zone = None
            for f in reversed(recent_f[-4:]):
                for ob in reversed(recent_o[-4:]):
                    lo, hi = max(f["low"], ob["low"]), min(f["high"], ob["high"])
                    if hi > lo:
                        zone = {"low": lo, "high": hi, "mid": 0.5 * (lo + hi)}
                        break
                if zone:
                    break
            if not zone or not (l <= zone["high"] and h >= zone["low"]):
                continue
            if side == "BUY" and c > zone["mid"] and c > o:
                entry, sl = c, min(l, zone["low"]) - 0.08 * atr
            elif side == "SELL" and c < zone["mid"] and c < o:
                entry, sl = c, max(h, zone["high"]) + 0.08 * atr
            else:
                continue
            if abs(entry - sl) > 0.12 * atr:
                entries.append(_entry(ei, side, entry, sl, df.index[ei]))
                break
    return entries


def _collect_cta(df: pd.DataFrame, fast: int, slow: int) -> list[dict]:
    out = df.copy()
    out["f"] = out["close"].ewm(span=fast, adjust=False).mean()
    out["s"] = out["close"].ewm(span=slow, adjust=False).mean()
    entries: list[dict] = []
    for i in range(slow + 10, len(out) - 1):
        row, prev = out.iloc[i], out.iloc[i - 1]
        atr = float(row["atr"] or 0) or 1e-6
        c = float(row["close"])
        if float(prev["f"]) <= float(prev["s"]) and float(row["f"]) > float(row["s"]):
            entries.append(_entry(i, "BUY", c, c - 1.5 * atr, out.index[i]))
        elif float(prev["f"]) >= float(prev["s"]) and float(row["f"]) < float(row["s"]):
            entries.append(_entry(i, "SELL", c, c + 1.5 * atr, out.index[i]))
    return entries


def _collect_mom_roc(df: pd.DataFrame, period: int = 60) -> list[dict]:
    out = df.copy()
    out["roc"] = out["close"].pct_change(period)
    entries: list[dict] = []
    for i in range(period + 20, len(out) - 1):
        row, prev = out.iloc[i], out.iloc[i - 1]
        atr = float(row["atr"] or 0) or 1e-6
        roc, proc = float(row["roc"] or 0), float(prev["roc"] or 0)
        c = float(row["close"])
        if proc <= 0 < roc and abs(roc) >= 0.02:
            entries.append(_entry(i, "BUY", c, c - 1.4 * atr, out.index[i]))
        elif proc >= 0 > roc and abs(roc) >= 0.02:
            entries.append(_entry(i, "SELL", c, c + 1.4 * atr, out.index[i]))
    return entries


def _collect_bb_meanrev(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    mid = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["bb_lo"], out["bb_hi"] = mid - 2 * std, mid + 2 * std
    entries: list[dict] = []
    for i in range(30, len(out) - 1):
        row = out.iloc[i]
        atr = float(row["atr"] or 0) or 1e-6
        c, o = float(row["close"]), float(row["open"])
        lo, hi = float(row["bb_lo"] or 0), float(row["bb_hi"] or 0)
        if c < lo and c > o:
            entries.append(_entry(i, "BUY", c, c - 1.2 * atr, out.index[i]))
        elif c > hi and c < o:
            entries.append(_entry(i, "SELL", c, c + 1.2 * atr, out.index[i]))
    return entries


def _collect_rsi(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    out["rsi"] = 100 - (100 / (1 + rs))
    entries: list[dict] = []
    for i in range(30, len(out) - 1):
        row, prev = out.iloc[i], out.iloc[i - 1]
        if pd.isna(row["rsi"]) or pd.isna(prev["rsi"]):
            continue
        atr = float(row["atr"] or 0) or 1e-6
        rsi, prsi = float(row["rsi"]), float(prev["rsi"])
        c = float(row["close"])
        if prsi < 30 <= rsi:
            entries.append(_entry(i, "BUY", c, c - 1.3 * atr, out.index[i]))
        elif prsi > 70 >= rsi:
            entries.append(_entry(i, "SELL", c, c + 1.3 * atr, out.index[i]))
    return entries


def _collect_atr_vol_break(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    out["atr_ma"] = out["atr"].rolling(20).mean()
    entries: list[dict] = []
    for i in range(40, len(out) - 1):
        row = out.iloc[i]
        atr = float(row["atr"] or 0) or 1e-6
        atr_ma = float(row["atr_ma"] or atr)
        if atr < 1.25 * atr_ma:
            continue
        win = out.iloc[i - 10 : i]
        hh, ll = float(win["high"].max()), float(win["low"].min())
        c, o = float(row["close"]), float(row["open"])
        if c > hh and c > o:
            entries.append(_entry(i, "BUY", c, c - 1.4 * atr, out.index[i]))
        elif c < ll and c < o:
            entries.append(_entry(i, "SELL", c, c + 1.4 * atr, out.index[i]))
    return entries


def _collect_macd(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"], out["sig"] = macd, signal
    entries: list[dict] = []
    for i in range(40, len(out) - 1):
        row, prev = out.iloc[i], out.iloc[i - 1]
        atr = float(row["atr"] or 0) or 1e-6
        c = float(row["close"])
        if float(prev["macd"]) <= float(prev["sig"]) and float(row["macd"]) > float(row["sig"]):
            entries.append(_entry(i, "BUY", c, c - 1.4 * atr, out.index[i]))
        elif float(prev["macd"]) >= float(prev["sig"]) and float(row["macd"]) < float(row["sig"]):
            entries.append(_entry(i, "SELL", c, c + 1.4 * atr, out.index[i]))
    return entries


def _collect_stoch(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    low14 = out["low"].rolling(14).min()
    high14 = out["high"].rolling(14).max()
    k = 100 * (out["close"] - low14) / (high14 - low14).replace(0, np.nan)
    d = k.rolling(3).mean()
    out["k"], out["d"] = k, d
    entries: list[dict] = []
    for i in range(30, len(out) - 1):
        row, prev = out.iloc[i], out.iloc[i - 1]
        if pd.isna(row["k"]) or pd.isna(row["d"]) or pd.isna(prev["k"]) or pd.isna(prev["d"]):
            continue
        atr = float(row["atr"] or 0) or 1e-6
        c = float(row["close"])
        if float(prev["k"]) <= float(prev["d"]) and float(row["k"]) > float(row["d"]) and float(row["k"]) < 30:
            entries.append(_entry(i, "BUY", c, c - 1.3 * atr, out.index[i]))
        elif float(prev["k"]) >= float(prev["d"]) and float(row["k"]) < float(row["d"]) and float(row["k"]) > 70:
            entries.append(_entry(i, "SELL", c, c + 1.3 * atr, out.index[i]))
    return entries


def _collect_engulfing(df: pd.DataFrame) -> list[dict]:
    entries: list[dict] = []
    for i in range(30, len(df) - 1):
        row, prev = df.iloc[i], df.iloc[i - 1]
        atr = float(row["atr"] or 0) or 1e-6
        o, c = float(row["open"]), float(row["close"])
        po, pc = float(prev["open"]), float(prev["close"])
        e20, e50 = float(row["ema20"]), float(row["ema50"])
        if c > o and pc < po and c >= po and o <= pc and e20 > e50:
            entries.append(_entry(i, "BUY", c, float(row["low"]) - 0.1 * atr, df.index[i]))
        elif c < o and pc > po and c <= po and o >= pc and e20 < e50:
            entries.append(_entry(i, "SELL", c, float(row["high"]) + 0.1 * atr, df.index[i]))
    return entries


def _collect_inside_break(df: pd.DataFrame) -> list[dict]:
    entries: list[dict] = []
    for i in range(30, len(df) - 1):
        prev, row = df.iloc[i - 1], df.iloc[i]
        # prev is inside bar vs i-2
        p2 = df.iloc[i - 2]
        atr = float(row["atr"] or 0) or 1e-6
        inside = float(prev["high"]) <= float(p2["high"]) and float(prev["low"]) >= float(p2["low"])
        if not inside:
            continue
        c = float(row["close"])
        if c > float(prev["high"]):
            entries.append(_entry(i, "BUY", c, float(prev["low"]) - 0.1 * atr, df.index[i]))
        elif c < float(prev["low"]):
            entries.append(_entry(i, "SELL", c, float(prev["high"]) + 0.1 * atr, df.index[i]))
    return entries


def _collect_supertrend(df: pd.DataFrame, mult: float = 3.0) -> list[dict]:
    out = df.copy()
    hl2 = (out["high"] + out["low"]) / 2
    atr = out["atr"]
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    st = pd.Series(index=out.index, dtype=float)
    direction = pd.Series(1, index=out.index)
    for i in range(1, len(out)):
        if float(out.iloc[i]["close"]) > float(upper.iloc[i - 1]):
            direction.iloc[i] = 1
        elif float(out.iloc[i]["close"]) < float(lower.iloc[i - 1]):
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
        st.iloc[i] = float(lower.iloc[i]) if direction.iloc[i] == 1 else float(upper.iloc[i])
    entries: list[dict] = []
    for i in range(20, len(out) - 1):
        if direction.iloc[i] == direction.iloc[i - 1]:
            continue
        atr_i = float(out.iloc[i]["atr"] or 0) or 1e-6
        c = float(out.iloc[i]["close"])
        if direction.iloc[i] == 1:
            entries.append(_entry(i, "BUY", c, c - 1.5 * atr_i, out.index[i]))
        else:
            entries.append(_entry(i, "SELL", c, c + 1.5 * atr_i, out.index[i]))
    return entries


def _collect_ichimoku_tk(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    high9 = out["high"].rolling(9).max()
    low9 = out["low"].rolling(9).min()
    high26 = out["high"].rolling(26).max()
    low26 = out["low"].rolling(26).min()
    tenkan = (high9 + low9) / 2
    kijun = (high26 + low26) / 2
    entries: list[dict] = []
    for i in range(30, len(out) - 1):
        if pd.isna(tenkan.iloc[i]) or pd.isna(kijun.iloc[i]):
            continue
        atr = float(out.iloc[i]["atr"] or 0) or 1e-6
        c = float(out.iloc[i]["close"])
        if float(tenkan.iloc[i - 1]) <= float(kijun.iloc[i - 1]) and float(tenkan.iloc[i]) > float(
            kijun.iloc[i]
        ):
            entries.append(_entry(i, "BUY", c, c - 1.4 * atr, out.index[i]))
        elif float(tenkan.iloc[i - 1]) >= float(kijun.iloc[i - 1]) and float(tenkan.iloc[i]) < float(
            kijun.iloc[i]
        ):
            entries.append(_entry(i, "SELL", c, c + 1.4 * atr, out.index[i]))
    return entries


def _collect_keltner(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    mid = out["close"].ewm(span=20, adjust=False).mean()
    up, dn = mid + 1.5 * out["atr"], mid - 1.5 * out["atr"]
    entries: list[dict] = []
    for i in range(30, len(out) - 1):
        row = out.iloc[i]
        atr = float(row["atr"] or 0) or 1e-6
        c, o = float(row["close"]), float(row["open"])
        if c > float(up.iloc[i]) and c > o:
            entries.append(_entry(i, "BUY", c, c - 1.3 * atr, out.index[i]))
        elif c < float(dn.iloc[i]) and c < o:
            entries.append(_entry(i, "SELL", c, c + 1.3 * atr, out.index[i]))
    return entries


def _collect_adx_ema(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    up = out["high"].diff()
    down = -out["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=out.index).rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=out.index).rolling(14).mean() / atr14.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.rolling(14).mean()
    entries: list[dict] = []
    for i in range(50, len(out) - 1):
        if pd.isna(adx.iloc[i]) or float(adx.iloc[i]) < 25:
            continue
        row, prev = out.iloc[i], out.iloc[i - 1]
        atr = float(row["atr"] or 0) or 1e-6
        c = float(row["close"])
        e20, e50 = float(row["ema20"]), float(row["ema50"])
        pe20, pe50 = float(prev["ema20"]), float(prev["ema50"])
        if pe20 <= pe50 and e20 > e50:
            entries.append(_entry(i, "BUY", c, c - 1.5 * atr, out.index[i]))
        elif pe20 >= pe50 and e20 < e50:
            entries.append(_entry(i, "SELL", c, c + 1.5 * atr, out.index[i]))
    return entries


def _collect_psar(df: pd.DataFrame) -> list[dict]:
    """Simplified parabolic SAR flip."""
    af_step, af_max = 0.02, 0.2
    entries: list[dict] = []
    if len(df) < 40:
        return entries
    bull = True
    af = af_step
    ep = float(df.iloc[0]["high"])
    sar = float(df.iloc[0]["low"])
    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        atr = float(row["atr"] or 0) or 1e-6
        prev_sar = sar
        sar = sar + af * (ep - sar)
        flipped = False
        if bull:
            sar = min(sar, float(df.iloc[i - 1]["low"]), float(df.iloc[max(0, i - 2)]["low"]))
            if l < sar:
                bull, flipped, sar, ep, af = False, True, ep, l, af_step
            else:
                if h > ep:
                    ep, af = h, min(af + af_step, af_max)
        else:
            sar = max(sar, float(df.iloc[i - 1]["high"]), float(df.iloc[max(0, i - 2)]["high"]))
            if h > sar:
                bull, flipped, sar, ep, af = True, True, ep, h, af_step
            else:
                if l < ep:
                    ep, af = l, min(af + af_step, af_max)
        if flipped and i >= 30:
            if bull:
                entries.append(_entry(i, "BUY", c, c - 1.4 * atr, df.index[i]))
            else:
                entries.append(_entry(i, "SELL", c, c + 1.4 * atr, df.index[i]))
        _ = prev_sar
    return entries


def _collect_tsmom(df: pd.DataFrame) -> list[dict]:
    """12-1 month time-series momentum (skip most recent month)."""
    out = df.copy()
    # ~252 trading days/year → 12m≈252, 1m≈21
    out["ret_12_1"] = out["close"].shift(21) / out["close"].shift(252) - 1.0
    entries: list[dict] = []
    last_side = None
    for i in range(260, len(out) - 1):
        r = out.iloc[i]["ret_12_1"]
        if pd.isna(r):
            continue
        atr = float(out.iloc[i]["atr"] or 0) or 1e-6
        c = float(out.iloc[i]["close"])
        side = "BUY" if float(r) > 0 else "SELL"
        if side == last_side:
            continue
        last_side = side
        sl = c - 2.0 * atr if side == "BUY" else c + 2.0 * atr
        entries.append(_entry(i, side, c, sl, out.index[i]))
    return entries


def _collect_zscore_mr(df: pd.DataFrame, win: int = 40) -> list[dict]:
    out = df.copy()
    mu = out["close"].rolling(win).mean()
    sd = out["close"].rolling(win).std()
    z = (out["close"] - mu) / sd.replace(0, np.nan)
    entries: list[dict] = []
    for i in range(win + 5, len(out) - 1):
        if pd.isna(z.iloc[i]) or pd.isna(z.iloc[i - 1]):
            continue
        atr = float(out.iloc[i]["atr"] or 0) or 1e-6
        c = float(out.iloc[i]["close"])
        if float(z.iloc[i - 1]) <= -2.0 < float(z.iloc[i]):
            entries.append(_entry(i, "BUY", c, c - 1.3 * atr, out.index[i]))
        elif float(z.iloc[i - 1]) >= 2.0 > float(z.iloc[i]):
            entries.append(_entry(i, "SELL", c, c + 1.3 * atr, out.index[i]))
    return entries


def _collect_vol_scaled_mom(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    ret = out["close"].pct_change(20)
    vol = out["close"].pct_change().rolling(20).std()
    score = ret / vol.replace(0, np.nan)
    entries: list[dict] = []
    for i in range(40, len(out) - 1):
        if pd.isna(score.iloc[i]) or pd.isna(score.iloc[i - 1]):
            continue
        atr = float(out.iloc[i]["atr"] or 0) or 1e-6
        c = float(out.iloc[i]["close"])
        if float(score.iloc[i - 1]) <= 0 < float(score.iloc[i]) and float(score.iloc[i]) > 0.5:
            entries.append(_entry(i, "BUY", c, c - 1.5 * atr, out.index[i]))
        elif float(score.iloc[i - 1]) >= 0 > float(score.iloc[i]) and float(score.iloc[i]) < -0.5:
            entries.append(_entry(i, "SELL", c, c + 1.5 * atr, out.index[i]))
    return entries


COLLECTORS: dict[str, Callable[[pd.DataFrame], list[dict]]] = {
    "fib_liq": _collect_fib,
    "ema_pull": lambda d: _collect_ema(d, pull_atr=1.0, disp_atr=0.55),
    "ema_tight": lambda d: _collect_ema(d, pull_atr=0.75, disp_atr=0.70),
    "sma_cross": _collect_sma,
    "donchian_20": lambda d: _collect_breakout(d, lookback=20),
    "price_action": _collect_price_action,
    "liq_sweep": _collect_liq_sweep,
    "trendline": _collect_trendline,
    "ob_fvg": _collect_ob_fvg,
    "macd_cross": _collect_macd,
    "stoch_cross": _collect_stoch,
    "engulfing": _collect_engulfing,
    "inside_break": _collect_inside_break,
    "supertrend": _collect_supertrend,
    "ichimoku_tk": _collect_ichimoku_tk,
    "keltner_break": _collect_keltner,
    "adx_ema": _collect_adx_ema,
    "psar": _collect_psar,
    "cta_dual_ma": lambda d: _collect_cta(d, 50, 200),
    "cta_fast": lambda d: _collect_cta(d, 10, 40),
    "turtle_55": lambda d: _collect_breakout(d, lookback=55),
    "turtle_20": lambda d: _collect_breakout(d, lookback=20),
    "tsmom": _collect_tsmom,
    "mom_roc": lambda d: _collect_mom_roc(d, 60),
    "bb_meanrev": _collect_bb_meanrev,
    "zscore_mr": _collect_zscore_mr,
    "rsi_reversion": _collect_rsi,
    "atr_vol_break": _collect_atr_vol_break,
    "vol_scaled_mom": _collect_vol_scaled_mom,
    "channel_100": lambda d: _collect_breakout(d, lookback=100),
}


def _battery(df: pd.DataFrame, trades: list[dict], strategy: str, symbol: str, rr: float) -> dict:
    n = len(df)
    split = int(n * 0.70)

    def slice_tr(start_i=0, end_i=None):
        end_i = end_i if end_i is not None else 10**9
        return [t for t in trades if start_i <= int(t["i"]) <= end_i]

    full = _stats(trades)
    oos = _stats(slice_tr(start_i=split))
    wf = []
    for k in range(4):
        a = int(n * (0.40 + 0.15 * k))
        b = int(n * (0.55 + 0.15 * k)) - 1
        if b <= a or a < 200:
            continue
        wf.append({"fold": k + 1, "oos": _stats(slice_tr(start_i=a, end_i=b))})
    torture = {
        "spread_0.15R": _stats(
            [
                {
                    **t,
                    "r": round(t["r"] - 0.15, 4),
                    "pnl": round((t["r"] - 0.15) * RISK_USD, 2),
                }
                for t in trades
            ]
        )
    }
    mc = monte_carlo(trades, paths=600)
    lm = levy_malliavin_pack(trades)
    stoch = stochastic_pack(df, trades)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
    meta = STRATEGIES[strategy]
    score = _score(v, oos, full, mc)
    return {
        "symbol": symbol,
        "strategy": strategy,
        "family": meta["family"],
        "label": meta["label"],
        "style": meta["style"],
        "rr": rr,
        "n": full.get("n") or 0,
        "full_exp_r": full.get("exp_r"),
        "full_pnl": full.get("pnl"),
        "oos_exp_r": oos.get("exp_r"),
        "oos_pnl": oos.get("pnl"),
        "wf_green": f"{wf_g}/{len(wf)}",
        "wf_green_n": wf_g,
        "wf_folds": len(wf),
        "mc_prob": mc.get("prob_profit"),
        "levy_prob": lm["levy"].get("prob_profit"),
        "bm_prob": stoch["brownian"].get("prob_profit"),
        "ito": stoch["ito"].get("verdict"),
        "martingale": stoch["martingale"].get("verdict"),
        "hjm": stoch["hjm"].get("verdict"),
        "malliavin": lm["malliavin"].get("fragility"),
        "verdict": v,
        "score": score,
    }


def _score(verdict_s: str, oos: dict, full: dict, mc: dict) -> float:
    w = {"SURVIVES": 3.0, "FRAGILE": 1.0, "DEAD": -2.0}.get(verdict_s, 0.0)
    oos_pnl = float(oos.get("pnl") or 0)
    exp = float(oos.get("exp_r") or 0)
    mc_p = float(mc.get("prob_profit") or 50) / 100.0
    return round(w * 10 + exp * 5 + (oos_pnl / 2000.0) * 3 + mc_p * 4, 3)


def _pick_best(rows: list[dict]) -> Optional[dict]:
    if not rows:
        return None
    return max(
        rows,
        key=lambda r: (
            float(r.get("score") or -1e9),
            float(r.get("oos_pnl") or -1e9),
            float(r.get("full_exp_r") or -1e9),
        ),
    )


def run_strategy_lab(
    *,
    days: int = DEFAULT_DAYS,
    years: Optional[float] = None,
    strategies: Optional[list[str]] = None,
) -> dict:
    """Run full bake-off on daily data (~15y default)."""
    t0 = time.time()
    if years is not None:
        days = int(float(years) * 365.25)
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=days)).isoformat()
    end_s = end.isoformat()
    keys = list(strategies) if strategies else list(STRATEGIES.keys())
    keys = [k for k in keys if k in COLLECTORS]

    print(
        f"Strategy Lab · DAILY · target {days}d (~{days/365.25:.1f}y) · {start}→{end_s} · "
        f"{len(BOOK)} pairs · {len(keys)} strategies · ${RISK_USD}/R · NO LIVE WIRE",
        flush=True,
    )

    matrix: list[dict] = []
    per_pair: dict[str, Any] = {}

    for sym in BOOK:
        rr = float(symbol_rr(sym, default=2.0))
        try:
            raw = _fetch_daily(sym, start, end_s)
            df = _prep(_clip(raw, start, end_s))
        except Exception as exc:
            print(f"{sym}: data fail {exc}", flush=True)
            per_pair[sym] = {"error": str(exc), "best": None, "results": []}
            continue
        if len(df) < 300:
            per_pair[sym] = {"error": f"thin:{len(df)}", "best": None, "results": []}
            continue

        span_y = (df.index[-1] - df.index[0]).days / 365.25
        print(
            f"\n=== {sym} bars={len(df)} span≈{span_y:.1f}y "
            f"{str(df.index[0].date())}→{str(df.index[-1].date())} RR=1:{rr:g} ===",
            flush=True,
        )
        rows = []
        for sk in keys:
            try:
                entries = COLLECTORS[sk](df)
                trades = simulate(df, entries, rr)
                row = _battery(df, trades, sk, sym, rr)
                rows.append(row)
                matrix.append(row)
                print(
                    f"  {sk:16s} n={row['n']:4d} E[R]={row['full_exp_r']} "
                    f"OOS=${row['oos_pnl']} MC={row['mc_prob']}% {row['verdict']} score={row['score']}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  {sk}: FAIL {exc}", flush=True)
                rows.append(
                    {
                        "symbol": sym,
                        "strategy": sk,
                        "family": STRATEGIES[sk]["family"],
                        "label": STRATEGIES[sk]["label"],
                        "verdict": "DEAD",
                        "score": -99,
                        "error": str(exc),
                        "n": 0,
                        "oos_pnl": 0,
                        "full_pnl": 0,
                    }
                )

        best = _pick_best([r for r in rows if not r.get("error")])
        per_pair[sym] = {
            "rr": rr,
            "bars": len(df),
            "span_years": round(span_y, 2),
            "data_start": str(df.index[0].date()),
            "data_end": str(df.index[-1].date()),
            "best": best,
            "results": rows,
            "survives": sum(1 for r in rows if r.get("verdict") == "SURVIVES"),
            "fragile": sum(1 for r in rows if r.get("verdict") == "FRAGILE"),
            "dead": sum(1 for r in rows if r.get("verdict") == "DEAD"),
        }
        if best:
            print(
                f"  ★ BEST {best['strategy']} ({best['label']}) · {best['verdict']} · "
                f"OOS ${best['oos_pnl']} · score {best['score']}",
                flush=True,
            )

    best_by_pair = {
        sym: (per_pair[sym].get("best") or {}).get("strategy")
        for sym in BOOK
        if per_pair.get(sym) and per_pair[sym].get("best")
    }
    oos_chart = [
        {
            "symbol": sym,
            "strategy": (per_pair[sym].get("best") or {}).get("strategy"),
            "label": (per_pair[sym].get("best") or {}).get("label"),
            "family": (per_pair[sym].get("best") or {}).get("family"),
            "oos_pnl": (per_pair[sym].get("best") or {}).get("oos_pnl") or 0,
            "full_pnl": (per_pair[sym].get("best") or {}).get("full_pnl") or 0,
            "verdict": (per_pair[sym].get("best") or {}).get("verdict"),
            "score": (per_pair[sym].get("best") or {}).get("score"),
            "span_years": per_pair[sym].get("span_years"),
        }
        for sym in BOOK
        if per_pair.get(sym) and per_pair[sym].get("best")
    ]
    heatmap = [
        {
            "symbol": r["symbol"],
            "strategy": r["strategy"],
            "oos_pnl": r.get("oos_pnl") or 0,
            "verdict": r.get("verdict"),
            "score": r.get("score"),
        }
        for r in matrix
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {
            "start": start,
            "end": end_s,
            "tf": "1d",
            "days": days,
            "years_target": round(days / 365.25, 2),
            "note": "Daily bars — 1h cannot cover 15y on Yahoo; crypto may be shorter listing",
        },
        "risk_usd": RISK_USD,
        "symbol_rr": SYMBOL_RR,
        "strategies": STRATEGIES,
        "note": "Research only — does not change live SYMBOL_ENTRY_BRAIN",
        "elapsed_s": round(time.time() - t0, 1),
        "per_pair": per_pair,
        "best_by_pair": best_by_pair,
        "charts": {"best_oos_by_pair": oos_chart, "heatmap": heatmap},
        "summary": {
            "pairs": len([s for s in BOOK if per_pair.get(s) and per_pair[s].get("best")]),
            "strategies_tested": len(keys),
            "cells": len(matrix),
            "survives_cells": sum(1 for r in matrix if r.get("verdict") == "SURVIVES"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {OUT} in {payload['elapsed_s']}s", flush=True)
    return payload


def latest_strategy_lab() -> Optional[dict]:
    if not OUT.exists():
        return None
    try:
        return json.loads(OUT.read_text())
    except Exception:
        return None


if __name__ == "__main__":
    run_strategy_lab(years=DEFAULT_YEARS)
