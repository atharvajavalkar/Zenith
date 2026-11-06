#!/usr/bin/env python3
"""RESEARCH ONLY — GOLD sleeve A/B compare (no live wiring).

A) Macro Gate + EMA20/50 pull (H1 bias)
   - Long only if DXY falling OR US10Y falling (daily vs prior)
   - Short only if DXY rising OR US10Y rising
   - Entry: EMA20/50 pull + displacement on 5m (same brain as live)
   - Exit: 2R or SL; London/NY hours only (07–16 UTC)

B) GC Flow Proxy (volume/delta stand-in for Bookmap)
   - Use GC=F 5m volume (COMEX proxy); signed delta = vol * sign(close-open)
   - Setups:
     1) Sweep + delta divergence: new swing high but 20-bar cum-delta lower → SELL
        (mirror for lows → BUY)
     2) Absorption: vol ≥ 2×SMA20, range ≤ 0.55·ATR, close in extreme 30% of bar
        → fade (close near high after absorb sell → SELL; near low → BUY)
   - Exit: 2R / SL; same session window

Windows: 14d · 1m. Risk $200/R. Prints markdown table.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = Path("/tmp/gold_macro_flow_bt.json")
RISK_USD = 200.0
SESS_START, SESS_END = 7, 16  # UTC


def _yf_hist(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(
        start=start, end=end, interval=interval, auto_adjust=True
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[cols].dropna(how="any")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _fetch_gold_5m(start: str, end: str) -> pd.DataFrame:
    """Prefer GC=F (has real volume); fallback GOLD via desk fetcher."""
    warm = (datetime.fromisoformat(start) - timedelta(days=10)).strftime("%Y-%m-%d")
    end_p = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    gc = _yf_hist("GC=F", warm, end_p, "5m")
    if not gc.empty and "volume" in gc.columns:
        gc["source"] = "GC=F"
        return gc
    from data.historical import fetch_symbol_ohlcv

    df = fetch_symbol_ohlcv("GOLD", warm, end_p, interval="5m")
    if df is None or df.empty:
        raise RuntimeError("no GOLD/GC 5m")
    df = df.rename(columns=str.lower)
    if "volume" not in df.columns:
        df["volume"] = (df["high"] - df["low"]).abs() * 1000.0  # weak proxy
    df["source"] = "GOLD"
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df[["open", "high", "low", "close", "volume", "source"]]


def _fetch_macro(start: str, end: str) -> pd.DataFrame:
    warm = (datetime.fromisoformat(start) - timedelta(days=40)).strftime("%Y-%m-%d")
    end_p = (datetime.fromisoformat(end) + timedelta(days=2)).strftime("%Y-%m-%d")
    dxy = _yf_hist("DX-Y.NYB", warm, end_p, "1d")
    if dxy.empty:
        dxy = _yf_hist("DX=F", warm, end_p, "1d")
    tnx = _yf_hist("^TNX", warm, end_p, "1d")
    out = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    if not dxy.empty:
        out["dxy"] = dxy["close"]
    if not tnx.empty:
        out["tnx"] = tnx["close"]
    out = out.sort_index().ffill()
    out["dxy_down"] = out["dxy"].diff() < 0 if "dxy" in out else False
    out["dxy_up"] = out["dxy"].diff() > 0 if "dxy" in out else False
    out["tnx_down"] = out["tnx"].diff() < 0 if "tnx" in out else False
    out["tnx_up"] = out["tnx"].diff() > 0 if "tnx" in out else False
    # Gold-friendly: DXY or yields falling → allow long; rising → allow short
    out["allow_long"] = out.get("dxy_down", False) | out.get("tnx_down", False)
    out["allow_short"] = out.get("dxy_up", False) | out.get("tnx_up", False)
    # If macro missing, allow both (don't starve sample)
    if "dxy" not in out.columns and "tnx" not in out.columns:
        out["allow_long"] = True
        out["allow_short"] = True
    return out


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - x["close"].shift()).abs(),
            (x["low"] - x["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    # 1h bias from resampled closes
    h1 = x["close"].resample("1h").last().dropna()
    h1_ema20 = h1.ewm(span=20, adjust=False).mean()
    h1_ema50 = h1.ewm(span=50, adjust=False).mean()
    bias = pd.Series(np.where(h1_ema20 > h1_ema50, 1, -1), index=h1.index)
    x["h1_bias"] = bias.reindex(x.index, method="ffill")
    # Flow proxies
    sign = np.sign(x["close"] - x["open"]).replace(0, 0.0)
    x["delta"] = x["volume"].astype(float) * sign
    x["cum_delta_20"] = x["delta"].rolling(20).sum()
    x["vol_sma20"] = x["volume"].rolling(20).mean()
    x["swing_hi"] = x["high"].rolling(12).max().shift(1)
    x["swing_lo"] = x["low"].rolling(12).min().shift(1)
    return x


def _in_session(ts: pd.Timestamp) -> bool:
    return SESS_START <= ts.hour < SESS_END


def _macro_ok(macro: pd.DataFrame, ts: pd.Timestamp, side: str) -> bool:
    if macro is None or macro.empty:
        return True
    day = ts.tz_convert("UTC").normalize()
    # use prior completed daily bar
    past = macro[macro.index <= day]
    if past.empty:
        return True
    row = past.iloc[-1]
    if side == "BUY":
        return bool(row.get("allow_long", True))
    return bool(row.get("allow_short", True))


def _simulate(
    df: pd.DataFrame,
    start: str,
    end: str,
    signals: list[tuple[pd.Timestamp, str, float, float]],
) -> dict[str, Any]:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    sig_i = 0
    open_t: Optional[dict] = None
    trades: list[dict] = []
    n_sig = 0

    for i in range(len(df)):
        ts = df.index[i]
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
                if start_dt <= open_t["entry_t"] <= end_dt:
                    trades.append(
                        {
                            "entry_t": str(open_t["entry_t"])[:16],
                            "exit_t": str(ts)[:16],
                            "side": side,
                            "pnl": round(float(r_mult) * RISK_USD, 2),
                            "reason": "sl" if hit_sl else "tp_2r",
                        }
                    )
                open_t = None
            continue

        if ts < start_dt or ts > end_dt:
            continue

        while sig_i < len(signals) and signals[sig_i][0] < ts:
            sig_i += 1
        if sig_i >= len(signals):
            continue
        st, side, entry, sl = signals[sig_i]
        if st != ts:
            continue
        n_sig += 1
        risk = abs(entry - sl)
        if risk <= 0:
            sig_i += 1
            continue
        tp = entry + 2.0 * risk if side == "BUY" else entry - 2.0 * risk
        open_t = {
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "entry_t": ts,
        }
        sig_i += 1

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
                "pnl": round(float(r_mult) * RISK_USD, 2),
                "reason": "eod",
            }
        )

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    decisive = wins + losses
    pnl = round(sum(t["pnl"] for t in trades), 2)
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / decisive, 1) if decisive else 0.0,
        "pnl": pnl,
        "avg_pnl": round(pnl / len(trades), 2) if trades else 0.0,
        "signals": n_sig,
        "trade_list": trades[:12],
    }


def signals_macro_ema(
    df: pd.DataFrame, macro: pd.DataFrame, start: str, end: str
) -> list[tuple[pd.Timestamp, str, float, float]]:
    from strategy.ema_pull_brain import evaluate_ema_2050_pull

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    out: list[tuple[pd.Timestamp, str, float, float]] = []
    cooldown = None
    # Build a prepare-like frame for brain
    work = df.rename(columns={"atr": "ATR", "ema20": "EMA20", "ema50": "EMA50"})
    for i in range(60, len(work)):
        ts = work.index[i]
        if ts < start_dt or ts > end_dt:
            continue
        if not _in_session(ts):
            continue
        if cooldown is not None and ts < cooldown:
            continue
        if float(work.iloc[i]["h1_bias"] or 0) == 0:
            continue
        pack = evaluate_ema_2050_pull(work.iloc[: i + 1])
        if not pack:
            continue
        side = pack["signal"]
        # H1 align
        if side == "BUY" and float(work.iloc[i]["h1_bias"]) < 0:
            continue
        if side == "SELL" and float(work.iloc[i]["h1_bias"]) > 0:
            continue
        if not _macro_ok(macro, ts, side):
            continue
        out.append((ts, side, float(pack["entry"]), float(pack["sl"])))
        cooldown = ts + timedelta(minutes=30)
    return out


def signals_gc_flow(
    df: pd.DataFrame, start: str, end: str
) -> list[tuple[pd.Timestamp, str, float, float]]:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    out: list[tuple[pd.Timestamp, str, float, float]] = []
    cooldown = None
    for i in range(40, len(df)):
        ts = df.index[i]
        if ts < start_dt or ts > end_dt:
            continue
        if not _in_session(ts):
            continue
        if cooldown is not None and ts < cooldown:
            continue
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        h, l, c, o = (
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["open"]),
        )
        atr = float(row["atr"] or 0) or max(h - l, 1e-6)
        vol = float(row["volume"] or 0)
        vsma = float(row["vol_sma20"] or 0) or 1.0
        rng = max(h - l, 1e-9)
        shi, slo = float(row["swing_hi"] or h), float(row["swing_lo"] or l)
        cd = float(row["cum_delta_20"] or 0)
        cd_prev = float(prev["cum_delta_20"] or 0)

        side = None
        sl = None

        # 1) Sweep + delta divergence
        if h > shi and c < shi and cd < cd_prev:
            side, sl = "SELL", h + 0.1 * atr
        elif l < slo and c > slo and cd > cd_prev:
            side, sl = "BUY", l - 0.1 * atr

        # 2) Absorption fade (if no sweep signal)
        if side is None and vol >= 2.0 * vsma and rng <= 0.55 * atr:
            pos = (c - l) / rng
            if pos >= 0.70 and c < o:  # absorbed buys near highs, closed weak
                side, sl = "SELL", h + 0.1 * atr
            elif pos <= 0.30 and c > o:  # absorbed sells near lows, closed strong
                side, sl = "BUY", l - 0.1 * atr

        if side is None or sl is None:
            continue
        if (side == "BUY" and sl >= c) or (side == "SELL" and sl <= c):
            continue
        out.append((ts, side, c, float(sl)))
        cooldown = ts + timedelta(minutes=45)
    return out


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    windows = [
        ("14d", (end - timedelta(days=13)).isoformat(), end.isoformat()),
        ("1m", (end - timedelta(days=29)).isoformat(), end.isoformat()),
    ]
    print("Fetching GC/GOLD 5m + DXY/TNX…", flush=True)
    raw = _fetch_gold_5m(windows[1][1], windows[1][2])
    macro = _fetch_macro(windows[1][1], windows[1][2])
    df = _prep(raw)
    src = str(raw["source"].iloc[-1]) if "source" in raw.columns else "?"
    print(f"bars={len(df)} source={src} macro_days={len(macro)}", flush=True)

    results: dict[str, Any] = {"A_macro_ema": {}, "B_gc_flow": {}}
    rows = []
    for label, a, b in windows:
        sig_a = signals_macro_ema(df, macro, a, b)
        sig_b = signals_gc_flow(df, a, b)
        ra = _simulate(df, a, b, sig_a)
        rb = _simulate(df, a, b, sig_b)
        results["A_macro_ema"][label] = {**ra, "start": a, "end": b}
        results["B_gc_flow"][label] = {**rb, "start": a, "end": b}
        rows.append(("A Macro+EMA", label, ra))
        rows.append(("B GC Flow", label, rb))
        print(
            f"A {label}: n={ra['trades']} wr={ra['win_rate']}% pnl=${ra['pnl']}",
            flush=True,
        )
        print(
            f"B {label}: n={rb['trades']} wr={rb['win_rate']}% pnl=${rb['pnl']}",
            flush=True,
        )

    payload = {
        "strategies": {
            "A": "Macro (DXY/TNX) gate + H1 bias + EMA20/50 pull · 2R · London/NY",
            "B": "GC volume delta divergence + absorption fade · 2R · London/NY",
        },
        "symbol": src,
        "risk_usd": RISK_USD,
        "note": (
            "B uses GC=F volume as COMEX proxy (not XM Bookmap). "
            "True L2/Bookmap not available on MetaAPI XM CFD."
        ),
        "windows": results,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print("\n### GOLD sleeve research ($200/R)\n", flush=True)
    print("| Strategy | Window | Trades | Win rate | PnL | Avg/trade |", flush=True)
    print("|---|---|---:|---:|---:|---:|", flush=True)
    for name, label, r in rows:
        print(
            f"| {name} | {label} | {r['trades']} | {r['win_rate']}% | "
            f"${r['pnl']} | ${r['avg_pnl']} |",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
