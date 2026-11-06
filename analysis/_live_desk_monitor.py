#!/usr/bin/env python3
"""RESEARCH ONLY — live-desk health check while RL trains.

Universe + brains match .env:
  GOLD → fib_liq 0.50–0.618 (15m)
  Rest → ema_20_50_pull (5m)

Windows: 14d / 1m / 2m · $200/R · partials+trail (live-style)
Also prints forced-2R sleeve totals for comparison.
No live wiring.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BOOK = ("GOLD", "USDCHF", "USDJPY", "NAS100", "UKOIL", "AUDUSD", "BTCUSD", "ETHUSD")
RISK_USD = 200.0
OUT = Path("/tmp/live_desk_monitor.json")


def _windows(end):
    return [
        ("14d", (end - timedelta(days=13)).isoformat(), end.isoformat()),
        ("1m", (end - timedelta(days=29)).isoformat(), end.isoformat()),
        ("2m", (end - timedelta(days=59)).isoformat(), end.isoformat()),
    ]


def _lot_for_risk(symbol: str, entry: float, sl: float) -> float:
    from strategy.position_sizing import leg_pnl

    p1 = abs(float(leg_pnl("BUY", float(entry), float(sl), 1.0, symbol=symbol)))
    if p1 < 1e-12:
        return 0.01
    return max(0.01, round(RISK_USD / p1, 2))


def _prepare(df_raw):
    from analysis.backtest_engine import _resample_ohlcv
    from indicators.prepare import prepare_df

    return (
        prepare_df(df_raw.copy()),
        prepare_df(_resample_ohlcv(df_raw, "15min").copy()),
        prepare_df(_resample_ohlcv(df_raw, "1h").copy()),
        prepare_df(_resample_ohlcv(df_raw, "4h").copy()),
    )


def fetch_pack(symbol: str, start: str, end: str) -> dict:
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
    f5, f15, f1h, f4h = _prepare(df_raw)
    return {
        "raw": df_raw,
        "full_5m": f5,
        "full_15m": f15,
        "full_1h": f1h,
        "full_4h": f4h,
        "start_dt": start_dt,
        "end_dt": end_dt,
    }


def cache_live_signals(pack: dict, symbol: str) -> tuple[list[dict], list[Optional[dict]]]:
    """Bar-by-bar live brains (direct evaluators — same as .env ENTRY_BRAIN path)."""
    from strategy.ema_pull_brain import evaluate_ema_2050_pull
    from strategy.fib_liq_brain import evaluate_fib_liq_sweep

    df5 = pack["full_5m"]
    df15 = pack["full_15m"]
    start_dt, end_dt = pack["start_dt"], pack["end_dt"]
    idx = df5.index[(df5.index >= start_dt) & (df5.index <= end_dt)]
    is_gold = str(symbol).upper() in ("GOLD", "XAUUSD", "XAU")
    bars: list[dict] = []
    sigs: list[Optional[dict]] = []
    for i, t in enumerate(idx):
        row = df5.loc[t]
        bars.append(
            {
                "t": str(t),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
            }
        )
        hit = None
        try:
            if is_gold:
                p15 = df15[df15.index <= t]
                if len(p15) >= 80:
                    hit = evaluate_fib_liq_sweep(p15)
            else:
                p5 = df5.loc[:t]
                if len(p5) >= 80:
                    hit = evaluate_ema_2050_pull(p5)
        except Exception:
            hit = None
        if not hit:
            sigs.append(None)
            continue
        entry = float(hit["entry"])
        sl = float(hit["sl"])
        risk = abs(entry - sl)
        side = hit["signal"]
        if risk <= 0:
            sigs.append(None)
            continue
        tp = entry + 2 * risk if side == "BUY" else entry - 2 * risk
        # Staged TPs so partials_trail can work like live 2R sleeve
        tp1 = entry + 1.0 * risk if side == "BUY" else entry - 1.0 * risk
        tp2 = entry + 1.5 * risk if side == "BUY" else entry - 1.5 * risk
        sigs.append(
            {
                "SIGNAL": side,
                "ENTRY": entry,
                "SL": sl,
                "TP": tp,
                "TP1": tp1,
                "TP2": tp2,
                "TP3": tp,
                "RR": "1:2",
                "SUCCESS_PERCENT": 55.0,
                "BRAIN": "fib_liq" if is_gold else "ema_20_50",
            }
        )
        if i and i % 1500 == 0:
            print(f"    {symbol}: {i}/{len(idx)}", flush=True)
    return bars, sigs


def slice_range(bars, sigs, start: str, end: str):
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    b2, s2 = [], []
    for b, s in zip(bars, sigs):
        t = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        if start_dt <= t <= end_dt:
            b2.append(b)
            s2.append(s)
    return b2, s2


def replay(symbol: str, bars, signals, *, mgmt: str, force_2r: bool) -> dict:
    from config import ENTRY_COOLDOWN_BARS
    from strategy.trade_simulator import (
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
            if update_trade_on_bar(open_trade, h, l, c, lot, mgmt_mode=mgmt):
                open_trade["exit_time"] = meta["t"][:16]
                open_trade["status"] = "closed"
                trades.append(open_trade)
                open_trade = None
                cooldown = step + int(ENTRY_COOLDOWN_BARS or 3)
            continue
        if step < cooldown:
            continue
        raw = signals[step]
        if not raw or raw.get("SIGNAL") not in ("BUY", "SELL"):
            continue
        entry, sl = raw.get("ENTRY"), raw.get("SL")
        if entry is None or sl is None:
            continue
        entry, sl = float(entry), float(sl)
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        side = raw["SIGNAL"]
        if force_2r:
            tp = entry + 2 * risk if side == "BUY" else entry - 2 * risk
            tp1 = tp2 = tp3 = None
            rr = "1:2"
        else:
            tp = raw.get("TP")
            tp1, tp2, tp3 = raw.get("TP1"), raw.get("TP2"), raw.get("TP3")
            rr = raw.get("RR") or "auto"
        lot = _lot_for_risk(symbol, entry, sl)
        open_trade = new_open_trade(
            signal=side,
            entry=entry,
            sl=sl,
            tp=float(tp) if tp else None,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            rr_type=rr,
            lot_size=lot,
            mc_prob=float(raw.get("SUCCESS_PERCENT") or 50),
            entry_time=meta["t"][:16],
            symbol=symbol,
            SYMBOL=symbol,
            mgmt_mode=mgmt,
        )
    if open_trade is not None:
        close_at_market(open_trade, bars[-1]["c"], bars[-1]["t"][:16])
        trades.append(open_trade)
    st = stats_from_trades(trades)
    return {
        "trades": st["closed"],
        "wins": st["wins"],
        "losses": st["losses"],
        "win_rate": round(100.0 * float(st["win_rate"]), 1),
        "pnl": round(float(st["realized_pnl"]), 2),
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    windows = _windows(end)
    win_2m = next(w for w in windows if w[0] == "2m")
    print(f"Live desk monitor RESEARCH {windows}", flush=True)
    print("Brains: GOLD=fib_liq · book=ema_20_50 · exits=partials_trail + 2R", flush=True)

    detail: list[dict] = []
    # Cache once over 2m, slice windows
    for sym in BOOK:
        print(f"\n######## {sym} ########", flush=True)
        try:
            t0 = time.time()
            pack = fetch_pack(sym, win_2m[1], win_2m[2])
            print(f"  fetch ok in {time.time()-t0:.0f}s bars5m={len(pack['full_5m'])}", flush=True)
            t0 = time.time()
            bars, sigs = cache_live_signals(pack, sym)
            n_hit = sum(1 for s in sigs if s)
            print(
                f"  signals cached in {time.time()-t0:.0f}s  bars={len(bars)} hits={n_hit}",
                flush=True,
            )
        except Exception as exc:
            print(f"  FAIL: {exc}", flush=True)
            traceback.print_exc()
            continue

        for label, start, end_s in windows:
            b2, s2 = slice_range(bars, sigs, start, end_s)
            live = replay(sym, b2, s2, mgmt="partials_trail", force_2r=False)
            fixed = replay(sym, b2, s2, mgmt="fixed", force_2r=True)
            for mode, st in (("live_exits", live), ("forced_2r", fixed)):
                row = {
                    "symbol": sym,
                    "window": label,
                    "mode": mode,
                    **st,
                    "brain": "fib_liq" if sym == "GOLD" else "ema_20_50",
                }
                detail.append(row)
            print(
                f"  {label}: live n={live['trades']} wr={live['win_rate']}% "
                f"pnl=${live['pnl']} | 2R n={fixed['trades']} wr={fixed['win_rate']}% "
                f"pnl=${fixed['pnl']}",
                flush=True,
            )

    # Aggregates
    book_agg: dict[str, dict[str, Any]] = {}
    for row in detail:
        if row["mode"] != "live_exits":
            continue
        key = row["window"]
        bucket = book_agg.setdefault(
            key, {"window": key, "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "by_symbol": {}}
        )
        bucket["trades"] += row["trades"]
        bucket["wins"] += row["wins"]
        bucket["losses"] += row["losses"]
        bucket["pnl"] += row["pnl"]
        bucket["by_symbol"][row["symbol"]] = {
            "trades": row["trades"],
            "win_rate": row["win_rate"],
            "pnl": row["pnl"],
            "brain": row["brain"],
        }
    for b in book_agg.values():
        dec = b["wins"] + b["losses"]
        b["win_rate"] = round(100.0 * b["wins"] / dec, 1) if dec else 0.0
        b["pnl"] = round(b["pnl"], 2)

    ranked = {}
    for w, bucket in book_agg.items():
        ranked[w] = sorted(
            (
                {"symbol": s, **v}
                for s, v in bucket["by_symbol"].items()
            ),
            key=lambda x: x["pnl"],
            reverse=True,
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "windows": windows,
        "risk_usd": RISK_USD,
        "note": "Live brains via _compute_signal; RL untouched",
        "desk_live_exits": book_agg,
        "ranked_live_exits": ranked,
        "detail": detail,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print("\n### Desk totals (live exits, $200/R)\n", flush=True)
    print("| Window | Trades | WR | PnL |", flush=True)
    print("|---|---:|---:|---:|", flush=True)
    for w in ("14d", "1m", "2m"):
        b = book_agg.get(w)
        if not b:
            continue
        print(f"| {w} | {b['trades']} | {b['win_rate']}% | ${b['pnl']} |", flush=True)

    for w in ("14d", "1m", "2m"):
        rows = ranked.get(w) or []
        if not rows:
            continue
        print(f"\n### Ranked {w} (live exits)\n", flush=True)
        print("| # | Symbol | Brain | Trades | WR | PnL |", flush=True)
        print("|---:|---|---|---:|---:|---:|", flush=True)
        for i, r in enumerate(rows, 1):
            print(
                f"| {i} | {r['symbol']} | {r['brain']} | {r['trades']} | "
                f"{r['win_rate']}% | ${r['pnl']} |",
                flush=True,
            )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
