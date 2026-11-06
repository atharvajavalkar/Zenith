#!/usr/bin/env python3
"""RESEARCH ONLY — do not wire into live.

Compare current Zenith brain vs alternate EMA / Fib setups on the live book
for last 14d and 1m. Fixed $200 risk, partials+trail exits where staged TPs
exist; else fixed TP/SL.

Writes /tmp/setup_compare.json
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Live composite book
BOOK = ("USDCHF", "USDJPY", "NAS100", "UKOIL", "AUDUSD", "BTCUSD")
RISK_USD = 200.0
MC_FLOOR = 42.0
OUT = "/tmp/setup_compare.json"


def _windows(end):
    return [
        ("14d", (end - timedelta(days=13)).isoformat(), end.isoformat()),
        ("1m", (end - timedelta(days=29)).isoformat(), end.isoformat()),
    ]


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _lot_for_risk(symbol: str, entry: float, sl: float) -> float:
    from strategy.position_sizing import leg_pnl

    p1 = abs(float(leg_pnl("BUY", float(entry), float(sl), 1.0, symbol=symbol)))
    if p1 < 1e-12:
        return 0.01
    return max(0.01, round(RISK_USD / p1, 2))


def _prepare_live(df_raw: pd.DataFrame):
    from analysis.backtest_engine import _resample_ohlcv
    from indicators.prepare import prepare_df

    return (
        prepare_df(df_raw.copy()),
        prepare_df(_resample_ohlcv(df_raw, "15min").copy()),
        prepare_df(_resample_ohlcv(df_raw, "1h").copy()),
        prepare_df(_resample_ohlcv(df_raw, "4h").copy()),
    )


def fetch_prepared(symbol: str, start: str, end: str):
    from analysis.backtest_engine import warmup_days_for_backtest
    from data.historical import fetch_symbol_ohlcv

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    warmup = warmup_days_for_backtest()
    ws = (start_dt - timedelta(days=warmup)).strftime("%Y-%m-%d")
    fe = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    df_raw = fetch_symbol_ohlcv(symbol, ws, fe, interval="5m")
    if df_raw is None or df_raw.empty:
        raise RuntimeError("no OHLCV")
    full_5m, full_15m, full_1h, full_4h = _prepare_live(df_raw)
    return {
        "raw": df_raw,
        "full_5m": full_5m,
        "full_15m": full_15m,
        "full_1h": full_1h,
        "full_4h": full_4h,
        "start_dt": start_dt,
        "end_dt": end_dt,
    }


# ── alternate signal generators (research) ───────────────────────────────────

def _staged_tps(signal: str, entry: float, sl: float, rr: float = 2.0):
    risk = abs(entry - sl)
    if risk < 1e-12:
        return None
    if signal == "BUY":
        tp1 = entry + risk
        tp = entry + risk * rr
    else:
        tp1 = entry - risk
        tp = entry - risk * rr
    # Label 1:2 so live simulator partial fracs work; TP distance still uses rr.
    return {"TP": tp, "TP1": tp1, "TP2": tp, "TP3": None, "RR": "1:2"}


def gen_ema_pull(
    df: pd.DataFrame,
    *,
    fast: int,
    slow: int,
    pull_atr: float = 0.85,
    disp_atr: float = 0.65,
    swing: int = 10,
    rr: float = 2.0,
) -> list[Optional[dict]]:
    """Trend EMA pullback + displacement on 5m."""
    out: list[Optional[dict]] = [None] * len(df)
    if len(df) < slow + swing + 5:
        return out
    close = df["close"]
    high, low, open_ = df["high"], df["low"], df["open"]
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    atr = _atr(df)
    mid = (ema_f + ema_s) / 2.0

    for i in range(slow + swing, len(df)):
        a = float(atr.iloc[i])
        if not a or math.isnan(a):
            continue
        c = float(close.iloc[i])
        o = float(open_.iloc[i])
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        ef, es, m = float(ema_f.iloc[i]), float(ema_s.iloc[i]), float(mid.iloc[i])
        bull = ef > es
        bear = ef < es
        body = abs(c - o)
        rng = max(h - l, 1e-12)
        disp_buy = c > o and body >= disp_atr * a and (c - l) / rng >= 0.75
        disp_sell = c < o and body >= disp_atr * a and (h - c) / rng >= 0.75
        near_mid = abs(c - m) <= pull_atr * a
        if bull and disp_buy and near_mid and c > ef:
            sl = float(low.iloc[i - swing : i + 1].min()) - 0.1 * a
            if sl >= c:
                continue
            tps = _staged_tps("BUY", c, sl, rr)
            if not tps:
                continue
            out[i] = {"SIGNAL": "BUY", "ENTRY": c, "SL": sl, **tps, "SUCCESS_PERCENT": 50.0}
        elif bear and disp_sell and near_mid and c < ef:
            sl = float(high.iloc[i - swing : i + 1].max()) + 0.1 * a
            if sl <= c:
                continue
            tps = _staged_tps("SELL", c, sl, rr)
            if not tps:
                continue
            out[i] = {"SIGNAL": "SELL", "ENTRY": c, "SL": sl, **tps, "SUCCESS_PERCENT": 50.0}
    return out


def gen_fib_retrace(
    df: pd.DataFrame,
    *,
    lookback: int = 40,
    levels: tuple[float, ...] = (0.382, 0.5, 0.618),
    tol_atr: float = 0.35,
    ema_fast: int = 9,
    ema_slow: int = 21,
    rr: float = 2.0,
) -> list[Optional[dict]]:
    """Enter at Fib retracement of last swing in EMA trend direction."""
    out: list[Optional[dict]] = [None] * len(df)
    if len(df) < lookback + ema_slow + 5:
        return out
    close = df["close"]
    high, low = df["high"], df["low"]
    ema_f = _ema(close, ema_fast)
    ema_s = _ema(close, ema_slow)
    atr = _atr(df)

    for i in range(lookback + ema_slow, len(df)):
        a = float(atr.iloc[i])
        if not a or math.isnan(a):
            continue
        window = df.iloc[i - lookback : i]
        swing_hi = float(window["high"].max())
        swing_lo = float(window["low"].min())
        span = swing_hi - swing_lo
        if span < 1.5 * a:
            continue
        c = float(close.iloc[i])
        ef, es = float(ema_f.iloc[i]), float(ema_s.iloc[i])
        bull = ef > es
        bear = ef < es
        hit = False
        level_px = None
        for fib in levels:
            if bull:
                px = swing_hi - span * fib
                if abs(c - px) <= tol_atr * a and c > es:
                    hit, level_px = True, px
                    break
            elif bear:
                px = swing_lo + span * fib
                if abs(c - px) <= tol_atr * a and c < es:
                    hit, level_px = True, px
                    break
        if not hit:
            continue
        if bull:
            sl = swing_lo - 0.15 * a
            if sl >= c:
                continue
            tps = _staged_tps("BUY", c, sl, rr)
            if tps:
                out[i] = {"SIGNAL": "BUY", "ENTRY": c, "SL": sl, **tps, "SUCCESS_PERCENT": 50.0}
        else:
            sl = swing_hi + 0.15 * a
            if sl <= c:
                continue
            tps = _staged_tps("SELL", c, sl, rr)
            if tps:
                out[i] = {"SIGNAL": "SELL", "ENTRY": c, "SL": sl, **tps, "SUCCESS_PERCENT": 50.0}
    return out


def gen_sma_cross(
    df: pd.DataFrame,
    *,
    fast: int = 20,
    slow: int = 50,
    rr: float = 2.0,
    atr_sl: float = 1.2,
) -> list[Optional[dict]]:
    """Classic SMA cross entries, ATR stop."""
    out: list[Optional[dict]] = [None] * len(df)
    if len(df) < slow + 3:
        return out
    close = df["close"]
    sma_f = close.rolling(fast).mean()
    sma_s = close.rolling(slow).mean()
    atr = _atr(df)
    for i in range(slow + 1, len(df)):
        a = float(atr.iloc[i])
        if not a or math.isnan(a):
            continue
        f0, f1 = float(sma_f.iloc[i - 1]), float(sma_f.iloc[i])
        s0, s1 = float(sma_s.iloc[i - 1]), float(sma_s.iloc[i])
        c = float(close.iloc[i])
        if f0 <= s0 and f1 > s1:
            sl = c - atr_sl * a
            tps = _staged_tps("BUY", c, sl, rr)
            if tps:
                out[i] = {"SIGNAL": "BUY", "ENTRY": c, "SL": sl, **tps, "SUCCESS_PERCENT": 50.0}
        elif f0 >= s0 and f1 < s1:
            sl = c + atr_sl * a
            tps = _staged_tps("SELL", c, sl, rr)
            if tps:
                out[i] = {"SIGNAL": "SELL", "ENTRY": c, "SL": sl, **tps, "SUCCESS_PERCENT": 50.0}
    return out


def cache_zenith_signals(pack: dict, symbol: str) -> list[Optional[dict]]:
    from analysis.backtest_engine import signal_slices_ready
    from analysis.session import set_backtest_bar_time
    from strategy.signal import _compute_signal

    full_5m = pack["full_5m"]
    in_range = full_5m.index[
        (full_5m.index >= pack["start_dt"]) & (full_5m.index <= pack["end_dt"])
    ]
    signals: list[Optional[dict]] = []
    for step, bar_time in enumerate(in_range):
        bar_pos = full_5m.index.get_loc(bar_time)
        p5m = full_5m.iloc[: bar_pos + 1]
        p15m = pack["full_15m"][pack["full_15m"].index <= bar_time]
        p1h = pack["full_1h"][pack["full_1h"].index <= bar_time]
        p4h = pack["full_4h"][pack["full_4h"].index <= bar_time]
        if not signal_slices_ready(p5m, p15m, p1h, p4h):
            signals.append(None)
            continue
        set_backtest_bar_time(bar_time.to_pydatetime())
        try:
            r = _compute_signal(
                df_1m=p5m, df_3m=p5m, df_5m=p5m, df_15m=p15m, df_1h=p1h, df_4h=p4h, symbol=symbol
            )
            if r.get("SIGNAL") in ("BUY", "SELL"):
                signals.append(
                    {
                        "SIGNAL": r.get("SIGNAL"),
                        "ENTRY": r.get("ENTRY"),
                        "SL": r.get("SL"),
                        "TP": r.get("TP"),
                        "TP1": r.get("TP1"),
                        "TP2": r.get("TP2"),
                        "TP3": r.get("TP3"),
                        "RR": r.get("RR"),
                        "SUCCESS_PERCENT": r.get("SUCCESS_PERCENT"),
                        "MARKOV_MC_PROB": r.get("MARKOV_MC_PROB"),
                    }
                )
            else:
                signals.append(None)
        except Exception:
            signals.append(None)
        finally:
            set_backtest_bar_time(None)
        if step and step % 500 == 0:
            print(f"      zenith {symbol}: {step}/{len(in_range)}", flush=True)
    return signals


def bars_meta(pack: dict) -> list[dict]:
    full_5m = pack["full_5m"]
    raw = pack["raw"].reindex(full_5m.index).ffill()
    in_range = full_5m.index[
        (full_5m.index >= pack["start_dt"]) & (full_5m.index <= pack["end_dt"])
    ]
    bars = []
    for t in in_range:
        row = raw.loc[t] if t in raw.index else full_5m.loc[t]
        bars.append(
            {"t": str(t), "h": float(row["high"]), "l": float(row["low"]), "c": float(row["close"])}
        )
    return bars


def slice_range(bars, signals, start: str, end: str):
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    b2, s2 = [], []
    for b, s in zip(bars, signals):
        t = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        if start_dt <= t <= end_dt:
            b2.append(b)
            s2.append(s)
    return b2, s2


def replay(symbol: str, bars, signals, *, use_mc_floor: bool) -> dict:
    from config import ENTRY_COOLDOWN_BARS
    from strategy.trade_simulator import (
        apply_rr_mode,
        close_at_market,
        new_open_trade,
        stats_from_trades,
        update_trade_on_bar,
    )

    if not bars:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "pnl": 0.0}

    open_trade = None
    cooldown = -1
    trades = []
    for step, meta in enumerate(bars):
        h, l, c = meta["h"], meta["l"], meta["c"]
        if open_trade is not None:
            lot = float(open_trade.get("lot_size") or 0.01)
            if update_trade_on_bar(open_trade, h, l, c, lot, mgmt_mode="partials_trail"):
                open_trade["exit_time"] = meta["t"][:16]
                open_trade["status"] = "closed"
                trades.append(open_trade)
                open_trade = None
                cooldown = step + int(ENTRY_COOLDOWN_BARS or 3)
            continue
        if step < cooldown:
            continue
        raw = signals[step]
        if not raw:
            continue
        result = apply_rr_mode(dict(raw), "auto" if use_mc_floor else "auto")
        # For alts RR already set; apply_rr_mode auto leaves as-is
        if use_mc_floor:
            result = apply_rr_mode(dict(raw), "auto")
        else:
            result = dict(raw)
        if result is None or result.get("SIGNAL") not in ("BUY", "SELL"):
            continue
        mc = float(result.get("SUCCESS_PERCENT") or result.get("MARKOV_MC_PROB") or 50)
        if use_mc_floor and mc < MC_FLOOR:
            continue
        entry, sl, tp = result.get("ENTRY"), result.get("SL"), result.get("TP")
        if entry is None or sl is None:
            continue
        lot = _lot_for_risk(symbol, float(entry), float(sl))
        open_trade = new_open_trade(
            signal=result["SIGNAL"],
            entry=float(entry),
            sl=float(sl),
            tp=float(tp) if tp else None,
            tp1=result.get("TP1"),
            tp2=result.get("TP2"),
            tp3=result.get("TP3"),
            rr_type=result.get("RR"),
            lot_size=lot,
            mc_prob=mc,
            entry_time=meta["t"][:16],
            symbol=symbol,
            SYMBOL=symbol,
            mgmt_mode="partials_trail",
        )
    if open_trade is not None:
        close_at_market(open_trade, bars[-1]["c"], bars[-1]["t"][:16])
        trades.append(open_trade)
    st = stats_from_trades(trades)
    return {
        "trades": st["closed"],
        "wins": st["wins"],
        "losses": st["losses"],
        "win_rate": float(st["win_rate"]),
        "pnl": float(st["realized_pnl"]),
    }


SETUP_SPECS: list[tuple[str, str, Callable]] = [
    ("current_zenith", "Live Zenith (structure/physics/MC)", None),  # special
    (
        "ema_8_21_pull",
        "EMA8/21 pullback + displacement · 2R",
        lambda df: gen_ema_pull(df, fast=8, slow=21, rr=2.0),
    ),
    (
        "ema_9_21_pull",
        "EMA9/21 pullback + displacement · 2R (closer to live EMAs)",
        lambda df: gen_ema_pull(df, fast=9, slow=21, rr=2.0),
    ),
    (
        "ema_20_50_pull",
        "EMA20/50 slower pullback · 2R",
        lambda df: gen_ema_pull(df, fast=20, slow=50, pull_atr=1.0, disp_atr=0.55, rr=2.0),
    ),
    (
        "ema_8_21_15r",
        "EMA8/21 pullback · 1.5R (Hyper-like target)",
        lambda df: gen_ema_pull(df, fast=8, slow=21, rr=1.5),
    ),
    (
        "fib_382_618",
        "Fib 38.2/50/61.8 of swing + EMA9/21 filter · 2R",
        lambda df: gen_fib_retrace(df, levels=(0.382, 0.5, 0.618), rr=2.0),
    ),
    (
        "fib_618_only",
        "Fib 61.8 deep pullback + EMA · 2R",
        lambda df: gen_fib_retrace(df, levels=(0.618,), tol_atr=0.45, rr=2.0),
    ),
    (
        "sma_20_50_cross",
        "SMA20/50 cross · ATR SL · 2R",
        lambda df: gen_sma_cross(df, fast=20, slow=50, rr=2.0),
    ),
]


def main() -> int:
    from data import historical as hist

    hist._YF_SYMBOLS.update({"UKOIL": ("BZ=F",), "AUDUSD": ("AUDUSD=X",)})

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    windows = _windows(end)
    win_1m = next(w for w in windows if w[0] == "1m")
    print("Setup compare RESEARCH (no live changes)", windows, flush=True)

    # setup -> window -> aggregate
    agg: dict[str, dict[str, dict]] = {}
    detail: list[dict] = []

    for sym in BOOK:
        print(f"\n######## {sym} ########", flush=True)
        try:
            pack = fetch_prepared(sym, win_1m[1], win_1m[2])
            # Restrict pack end/start to 1m for zenith cache covering both windows
            bars = bars_meta(pack)
            df_5m = pack["full_5m"]
            in_idx = df_5m.index[
                (df_5m.index >= pack["start_dt"]) & (df_5m.index <= pack["end_dt"])
            ]
            df_slice = df_5m.loc[in_idx]
            print(f"  bars={len(bars)} t_fetch ok", flush=True)
        except Exception as exc:
            print(f"  FAIL fetch: {exc}", flush=True)
            traceback.print_exc()
            continue

        # Zenith signals (expensive, once)
        print(f"  caching current_zenith…", flush=True)
        t0 = time.time()
        zenith_sigs = cache_zenith_signals(pack, sym)
        print(f"  zenith done in {time.time()-t0:.0f}s", flush=True)

        alt_sigs: dict[str, list] = {}
        for name, _desc, fn in SETUP_SPECS:
            if fn is None:
                continue
            raw_sigs = fn(df_slice)
            # Align length to bars (df_slice == in_range)
            alt_sigs[name] = list(raw_sigs)

        for name, desc, fn in SETUP_SPECS:
            sigs = zenith_sigs if fn is None else alt_sigs[name]
            use_mc = fn is None
            for label, start, end_s in windows:
                b2, s2 = slice_range(bars, sigs, start, end_s)
                st = replay(sym, b2, s2, use_mc_floor=use_mc)
                row = {
                    "symbol": sym,
                    "setup": name,
                    "desc": desc,
                    "window": label,
                    "trades": st["trades"],
                    "wins": st["wins"],
                    "losses": st["losses"],
                    "win_rate": st["win_rate"],
                    "pnl": st["pnl"],
                }
                detail.append(row)
                bucket = agg.setdefault(name, {}).setdefault(
                    label,
                    {"setup": name, "desc": desc, "window": label, "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
                )
                bucket["trades"] += st["trades"]
                bucket["wins"] += st["wins"]
                bucket["losses"] += st["losses"]
                bucket["pnl"] += st["pnl"]
                print(
                    f"  [{name:16}] {label}: n={st['trades']:3} "
                    f"wr={100*st['win_rate']:.0f}% pnl=${st['pnl']:.0f}",
                    flush=True,
                )

    # finalize WR
    for setup in agg.values():
        for bucket in setup.values():
            d = bucket["wins"] + bucket["losses"]
            bucket["win_rate"] = round(100.0 * bucket["wins"] / d, 1) if d else 0.0
            bucket["pnl"] = round(bucket["pnl"], 2)

    # rank by 1m pnl, also show 14d
    rank_1m = sorted(
        [agg[s]["1m"] for s in agg if "1m" in agg[s]],
        key=lambda x: x["pnl"],
        reverse=True,
    )
    rank_14d = sorted(
        [agg[s]["14d"] for s in agg if "14d" in agg[s]],
        key=lambda x: x["pnl"],
        reverse=True,
    )

    baseline_1m = next((r for r in rank_1m if r["setup"] == "current_zenith"), None)
    baseline_14 = next((r for r in rank_14d if r["setup"] == "current_zenith"), None)

    payload = {
        "note": "RESEARCH ONLY — live algo untouched. Fixed $200 risk/trade.",
        "book": list(BOOK),
        "windows": {l: {"start": a, "end": b} for l, a, b in windows},
        "rank_14d": rank_14d,
        "rank_1m": rank_1m,
        "baseline_14d": baseline_14,
        "baseline_1m": baseline_1m,
        "detail": detail,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n### 14d BOOK TOTAL (all symbols)\n", flush=True)
    print("| rank | setup | trades | WR% | PnL | vs Zenith |", flush=True)
    print("|---:|---|---:|---:|---:|---:|", flush=True)
    base_p = float(baseline_14["pnl"]) if baseline_14 else 0.0
    for i, r in enumerate(rank_14d, 1):
        delta = r["pnl"] - base_p
        print(
            f"| {i} | {r['setup']} | {r['trades']} | {r['win_rate']} | "
            f"{r['pnl']} | {delta:+.0f} |",
            flush=True,
        )

    print("\n### 1m BOOK TOTAL\n", flush=True)
    print("| rank | setup | trades | WR% | PnL | vs Zenith |", flush=True)
    print("|---:|---|---:|---:|---:|---:|", flush=True)
    base_p = float(baseline_1m["pnl"]) if baseline_1m else 0.0
    for i, r in enumerate(rank_1m, 1):
        delta = r["pnl"] - base_p
        print(
            f"| {i} | {r['setup']} | {r['trades']} | {r['win_rate']} | "
            f"{r['pnl']} | {delta:+.0f} |",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
