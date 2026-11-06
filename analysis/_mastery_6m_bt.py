#!/usr/bin/env python3
"""RESEARCH ONLY — 6-month mastery backtest for live Zenith brains.

Live brains (proxy TF — yfinance/TV 5m only ~25–60d; 6m needs 1h):
  GOLD: fib_liq 0.50–0.618 · 1h→1h · 2R   (live: 15m→15m)
  Book: EMA20/50 · 1h · 2R                 (live: 5m)
  Alts: GOLD deep fib; BTC/ETH fib; GOLD ema

Also prints monthly PnL + efficiency notes.
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

OUT = Path("/tmp/mastery_6m_bt.json")
RISK_USD = 200.0
SWING = 3

BOOK = ("USDCHF", "USDJPY", "NAS100", "UKOIL", "AUDUSD", "BTCUSD", "ETHUSD")
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
    """Full ~6m coverage via yfinance 1h (5m cannot cover 6m)."""
    warm = (datetime.fromisoformat(start) - timedelta(days=20)).strftime("%Y-%m-%d")
    end_p = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    import yfinance as yf

    last_err = None
    for ticker in (YF.get(symbol, symbol), f"{symbol}=X", symbol):
        try:
            df = yf.Ticker(ticker).history(
                start=warm, end=end_p, interval="1h", auto_adjust=True
            )
            if df is not None and not df.empty:
                break
        except Exception as exc:
            last_err = exc
            df = None
    if df is None or df.empty:
        raise RuntimeError(f"no 1h data {symbol}: {last_err}")
    df = df.rename(columns=str.lower)
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    df = df[cols].dropna()
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
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    return out


def _resample(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    m = {"5m": "5min", "15m": "15min", "1h": "1h"}[rule]
    o = df5["open"].resample(m).first()
    h = df5["high"].resample(m).max()
    l = df5["low"].resample(m).min()
    c = df5["close"].resample(m).last()
    return _prep(pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna())


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


def _stats(trades: list[dict], monthly: bool = False) -> dict[str, Any]:
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    dec = wins + losses
    pnl = round(sum(t["pnl"] for t in trades), 2)
    out = {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / dec, 1) if dec else 0.0,
        "pnl": pnl,
        "avg_pnl": round(pnl / len(trades), 2) if trades else 0.0,
        "exp_r": round((pnl / RISK_USD) / len(trades), 3) if trades else 0.0,
        "max_dd": _max_dd(trades),
    }
    if monthly:
        by = {}
        for t in trades:
            m = t["entry_t"][:7]
            by.setdefault(m, []).append(t["pnl"])
        out["monthly"] = {
            m: {
                "n": len(v),
                "pnl": round(sum(v), 2),
                "wr": round(100.0 * sum(1 for x in v if x > 0) / max(1, sum(1 for x in v if x != 0)), 1),
            }
            for m, v in sorted(by.items())
        }
    return out


def _max_dd(trades: list[dict]) -> float:
    eq = 0.0
    peak = 0.0
    dd = 0.0
    for t in trades:
        eq += t["pnl"]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return round(dd, 2)


def run_fib(
    df_imp: pd.DataFrame,
    df_ent: pd.DataFrame,
    start: str,
    end: str,
    fibs: tuple[float, ...],
) -> list[dict]:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    trades: list[dict] = []
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
                    trades.append(
                        {
                            "entry_t": str(open_t["entry_t"])[:16],
                            "side": side,
                            "pnl": round(float(r_mult) * RISK_USD, 2),
                        }
                    )
                open_t = None
                cooldown = ts + timedelta(minutes=30)
            continue

        if ts < start_dt or ts > end_dt or ts < cooldown:
            continue
        pos = df_imp.index.searchsorted(ts, side="right") - 1
        if pos < 40:
            continue
        imp = cache[pos]
        if not imp or (ts - imp["end_t"]).total_seconds() > 24 * 3600:
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
    return trades


def run_ema(df: pd.DataFrame, start: str, end: str) -> list[dict]:
    from strategy.ema_pull_brain import evaluate_ema_2050_pull

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    work = df.rename(columns={"atr": "ATR", "ema20": "EMA20", "ema50": "EMA50"})
    trades: list[dict] = []
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
                    trades.append(
                        {
                            "entry_t": str(open_t["entry_t"])[:16],
                            "side": side,
                            "pnl": round(float(r_mult) * RISK_USD, 2),
                        }
                    )
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
    return trades


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"6m mastery {start} → {end_s}  ($200/R)  [1h proxy of live 15m/5m brains]",
        flush=True,
    )

    rows = []
    details: dict[str, Any] = {}
    monthly_book: dict[str, dict] = {}

    g_raw = _fetch_1h("GOLD", start, end_s)
    span_days = (g_raw.index[-1] - g_raw.index[0]).days if len(g_raw) else 0
    med = (
        (g_raw.index[1:] - g_raw.index[:-1]).median().total_seconds() / 60
        if len(g_raw) > 2
        else 60
    )
    print(f"GOLD bars={len(g_raw)} span≈{span_days}d medianΔ={med:.0f}m", flush=True)

    g1h = _prep(g_raw)

    # Live GOLD brain (1h proxy)
    t = run_fib(g1h, g1h, start, end_s, (0.50, 0.618))
    st = _stats(t, monthly=True)
    rows.append({"symbol": "GOLD", "setup": "fib50-618 1h LIVE-proxy", **st})
    details["GOLD_fib_live"] = st
    print(f"  GOLD fib live n={st['trades']} wr={st['win_rate']}% pnl=${st['pnl']}", flush=True)

    # Alt GOLD deep
    t2 = run_fib(g1h, g1h, start, end_s, (0.618, 0.705, 0.786))
    st2 = _stats(t2, monthly=True)
    rows.append({"symbol": "GOLD", "setup": "fib618-786 1h ALT", **st2})
    print(f"  GOLD fib deep n={st2['trades']} wr={st2['win_rate']}% pnl=${st2['pnl']}", flush=True)

    # GOLD ema (control — should lag)
    te = run_ema(g1h, start, end_s)
    se = _stats(te)
    rows.append({"symbol": "GOLD", "setup": "ema20/50 1h", **se})
    print(f"  GOLD ema n={se['trades']} wr={se['win_rate']}% pnl=${se['pnl']}", flush=True)

    for sym in BOOK:
        try:
            raw = _fetch_1h(sym, start, end_s)
            df = _prep(raw)
            tr = run_ema(df, start, end_s)
            s = _stats(tr, monthly=True)
            rows.append({"symbol": sym, "setup": "ema20/50 1h LIVE-proxy", **s})
            monthly_book[sym] = s.get("monthly") or {}
            print(
                f"  {sym} ema n={s['trades']} wr={s['win_rate']}% pnl=${s['pnl']} dd=${s['max_dd']}",
                flush=True,
            )
            if sym in ("BTCUSD", "ETHUSD"):
                tf = run_fib(df, df, start, end_s, (0.50, 0.618))
                sf = _stats(tf)
                rows.append({"symbol": sym, "setup": "fib50-618 1h ALT", **sf})
                print(
                    f"  {sym} fib n={sf['trades']} wr={sf['win_rate']}% pnl=${sf['pnl']}",
                    flush=True,
                )
        except Exception as exc:
            print(f"  SKIP {sym}: {exc}", flush=True)
            rows.append(
                {
                    "symbol": sym,
                    "setup": "ema20/50 1h LIVE-proxy",
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0,
                    "pnl": 0.0,
                    "avg_pnl": 0.0,
                    "exp_r": 0.0,
                    "max_dd": 0.0,
                    "error": str(exc),
                }
            )

    live_gold = next(r for r in rows if "fib50-618 1h LIVE-proxy" in r["setup"])
    live_book = [r for r in rows if r["setup"] == "ema20/50 1h LIVE-proxy"]
    desk = {
        "gold_pnl": live_gold["pnl"],
        "book_ema_pnl": round(sum(r["pnl"] for r in live_book), 2),
        "desk_total": round(live_gold["pnl"] + sum(r["pnl"] for r in live_book), 2),
        "data_note": (
            f"True 6m via yfinance 1h (span≈{span_days}d). "
            "Proxy for live GOLD 15m fib + book 5m EMA — same logic, coarser bars."
        ),
    }

    # Efficiency score: E[R] * sqrt(trades) / (1 + |max_dd|/RISK) — simple rank aid
    for r in rows:
        n = max(r.get("trades", 0), 1)
        dd = abs(r.get("max_dd", 0) or 0)
        r["eff"] = round(
            (r.get("exp_r", 0) or 0) * (n**0.5) / (1.0 + dd / RISK_USD), 3
        )

    ranked = sorted(
        rows, key=lambda r: (r.get("eff", 0), r.get("pnl", 0)), reverse=True
    )
    payload = {
        "window": {"start": start, "end": end_s, "label": "6m-1h-proxy"},
        "risk_usd": RISK_USD,
        "desk": desk,
        "results": rows,
        "ranked": ranked,
        "gold_monthly": details.get("GOLD_fib_live", {}).get("monthly"),
        "book_monthly": monthly_book,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print("\n### 6m mastery results — 1h proxy ($200/R)\n", flush=True)
    print("| Rank | Symbol | Setup | Trades | WR | PnL | E[R] | MaxDD | Eff |", flush=True)
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|", flush=True)
    for i, r in enumerate(ranked, 1):
        print(
            f"| {i} | {r['symbol']} | {r['setup']} | {r['trades']} | {r['win_rate']}% | "
            f"${r['pnl']} | {r['exp_r']} | ${r.get('max_dd', 0)} | {r.get('eff', 0)} |",
            flush=True,
        )
    print(
        f"\n**Desk (live-proxy brains):** GOLD fib ${desk['gold_pnl']} + book EMA "
        f"${desk['book_ema_pnl']} = **${desk['desk_total']}**",
        flush=True,
    )
    print(f"Note: {desk['data_note']}", flush=True)
    if payload.get("gold_monthly"):
        print("\n### GOLD fib monthly\n", flush=True)
        print("| Month | N | WR | PnL |", flush=True)
        print("|---|---:|---:|---:|", flush=True)
        for m, v in payload["gold_monthly"].items():
            print(f"| {m} | {v['n']} | {v['wr']}% | ${v['pnl']} |", flush=True)
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
