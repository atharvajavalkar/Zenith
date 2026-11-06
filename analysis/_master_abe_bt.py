#!/usr/bin/env python3
"""RESEARCH — efficiency check for desk mastery A/B/C/E vs baseline.

Same live brains (GOLD fib / book EMA), 14d + 1m windows, $200/R paper.
  baseline: take all signals, no corr, no ETH cooldown
  abce:     corr groups + risk-on cap + ETH cooldown 12 bars
            + report alloc-weighted desk PnL (B)

Does not change live wiring.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BOOK = ("GOLD", "USDCHF", "USDJPY", "NAS100", "UKOIL", "AUDUSD", "BTCUSD", "ETHUSD")
RISK_USD = 200.0
OUT = Path("/tmp/master_abe_bt.json")

# B — proposed / live alloc after patch
ALLOC = {
    "GOLD": 0.10,
    "USDCHF": 0.20,
    "USDJPY": 0.12,
    "NAS100": 0.05,
    "UKOIL": 0.18,
    "AUDUSD": 0.20,
    "BTCUSD": 0.05,
    "ETHUSD": 0.05,
}
OLD_ALLOC = {
    "GOLD": 0.10,
    "USDCHF": 0.18,
    "USDJPY": 0.16,
    "NAS100": 0.14,
    "UKOIL": 0.10,
    "AUDUSD": 0.09,
    "BTCUSD": 0.12,
    "ETHUSD": 0.11,
}


def _windows(end):
    return [
        ("14d", (end - timedelta(days=13)).isoformat(), end.isoformat()),
        ("1m", (end - timedelta(days=29)).isoformat(), end.isoformat()),
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
        "full_5m": f5,
        "full_15m": f15,
        "full_1h": f1h,
        "full_4h": f4h,
        "start_dt": start_dt,
        "end_dt": end_dt,
    }


def cache_signals(pack: dict, symbol: str):
    from strategy.ema_pull_brain import evaluate_ema_2050_pull
    from strategy.fib_liq_brain import evaluate_fib_liq_sweep

    df5 = pack["full_5m"]
    df15 = pack["full_15m"]
    start_dt, end_dt = pack["start_dt"], pack["end_dt"]
    idx = df5.index[(df5.index >= start_dt) & (df5.index <= end_dt)]
    is_gold = str(symbol).upper() in ("GOLD", "XAUUSD", "XAU")
    bars, sigs = [], []
    for t in idx:
        row = df5.loc[t]
        bars.append(
            {"t": str(t), "h": float(row["high"]), "l": float(row["low"]), "c": float(row["close"])}
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
        entry, sl = float(hit["entry"]), float(hit["sl"])
        risk = abs(entry - sl)
        if risk <= 0:
            sigs.append(None)
            continue
        side = hit["signal"]
        tp = entry + 2 * risk if side == "BUY" else entry - 2 * risk
        sigs.append(
            {
                "SIGNAL": side,
                "ENTRY": entry,
                "SL": sl,
                "TP": tp,
                "TP1": entry + risk if side == "BUY" else entry - risk,
                "TP2": entry + 1.5 * risk if side == "BUY" else entry - 1.5 * risk,
                "TP3": tp,
                "RR": "1:2",
                "SUCCESS_PERCENT": 55.0 if symbol != "ETHUSD" else 48.0,  # ETH often weaker MC
                "SYMBOL": symbol,
            }
        )
    return bars, sigs


def slice_range(bars, sigs, start, end):
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


def replay_book(
    book_bars: dict[str, list],
    book_sigs: dict[str, list],
    *,
    mode: str,
    eth_cooldown: int = 12,
    eth_mc_floor: float = 44.0,
) -> dict[str, Any]:
    """Bar-aligned multi-symbol replay with optional A/B/C/E filters."""
    from analysis.correlation_groups import can_open_symbol
    from strategy.trade_simulator import (
        close_at_market,
        new_open_trade,
        stats_from_trades,
        update_trade_on_bar,
    )

    # Union of timestamps
    all_ts = sorted({b["t"] for bars in book_bars.values() for b in bars})
    # Index maps
    idx = {
        sym: {b["t"]: i for i, b in enumerate(bars)}
        for sym, bars in book_bars.items()
    }
    open_trades: dict[str, dict] = {}  # sym -> trade
    eth_last_i = -10_000
    trades: list[dict] = []
    skipped = {"corr": 0, "risk_on": 0, "eth_cd": 0, "eth_mc": 0}

    for ti, ts in enumerate(all_ts):
        # Manage opens
        for sym in list(open_trades.keys()):
            if ts not in idx.get(sym, {}):
                continue
            i = idx[sym][ts]
            meta = book_bars[sym][i]
            ot = open_trades[sym]
            lot = float(ot.get("lot_size") or 0.01)
            if update_trade_on_bar(
                ot, meta["h"], meta["l"], meta["c"], lot, mgmt_mode="partials_trail"
            ):
                ot["exit_time"] = meta["t"][:16]
                ot["status"] = "closed"
                trades.append(ot)
                del open_trades[sym]

        open_syms = list(open_trades.keys())
        # Entries
        for sym in BOOK:
            if sym in open_trades or ts not in idx.get(sym, {}):
                continue
            i = idx[sym][ts]
            raw = book_sigs[sym][i]
            if not raw or raw.get("SIGNAL") not in ("BUY", "SELL"):
                continue

            if mode == "abce":
                ok, reason = can_open_symbol(sym, open_syms)
                if not ok:
                    if "Risk-on" in reason:
                        skipped["risk_on"] += 1
                    else:
                        skipped["corr"] += 1
                    continue
                if sym == "ETHUSD":
                    if i - eth_last_i < eth_cooldown:
                        skipped["eth_cd"] += 1
                        continue
                    mc = float(raw.get("SUCCESS_PERCENT") or 0)
                    if mc < eth_mc_floor:
                        skipped["eth_mc"] += 1
                        continue

            entry, sl = float(raw["ENTRY"]), float(raw["SL"])
            lot = _lot_for_risk(sym, entry, sl)
            ot = new_open_trade(
                signal=raw["SIGNAL"],
                entry=entry,
                sl=sl,
                tp=float(raw["TP"]),
                tp1=raw.get("TP1"),
                tp2=raw.get("TP2"),
                tp3=raw.get("TP3"),
                rr_type=raw.get("RR") or "1:2",
                lot_size=lot,
                mc_prob=float(raw.get("SUCCESS_PERCENT") or 50),
                entry_time=ts[:16],
                symbol=sym,
                SYMBOL=sym,
                mgmt_mode="partials_trail",
            )
            open_trades[sym] = ot
            open_syms.append(sym)
            if sym == "ETHUSD":
                eth_last_i = i

    # Flatten close remaining
    for sym, ot in open_trades.items():
        bars = book_bars[sym]
        close_at_market(ot, bars[-1]["c"], bars[-1]["t"][:16])
        trades.append(ot)

    st = stats_from_trades(trades)
    by_sym: dict[str, dict] = {}
    for t in trades:
        s = str(t.get("symbol") or t.get("SYMBOL") or "?")
        b = by_sym.setdefault(s, {"n": 0, "pnl": 0.0, "wins": 0, "losses": 0})
        pnl = float(t.get("realized_pnl") or t.get("pnl") or 0)
        b["n"] += 1
        b["pnl"] += pnl
        if pnl > 0:
            b["wins"] += 1
        elif pnl < 0:
            b["losses"] += 1

    def weighted(alloc: dict) -> float:
        # Scale each symbol's PnL by alloc/equal so higher alloc = more desk weight
        equal = 1.0 / max(len(BOOK), 1)
        total = 0.0
        for s, b in by_sym.items():
            w = float(alloc.get(s, equal)) / equal
            total += b["pnl"] * w
        return round(total, 2)

    return {
        "mode": mode,
        "trades": st["closed"],
        "wins": st["wins"],
        "losses": st["losses"],
        "win_rate": round(100.0 * float(st["win_rate"]), 1),
        "pnl_equal": round(float(st["realized_pnl"]), 2),
        "pnl_old_alloc": weighted(OLD_ALLOC),
        "pnl_new_alloc": weighted(ALLOC),
        "by_symbol": {
            s: {
                "n": v["n"],
                "pnl": round(v["pnl"], 2),
                "wr": round(100.0 * v["wins"] / max(1, v["wins"] + v["losses"]), 1),
            }
            for s, v in sorted(by_sym.items(), key=lambda x: -x[1]["pnl"])
        },
        "skipped": skipped if mode == "abce" else {},
    }


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    windows = _windows(end)
    win_1m = next(w for w in windows if w[0] == "1m")
    print(f"Master A/B/C/E efficiency {windows}", flush=True)

    caches: dict[str, tuple] = {}
    for sym in BOOK:
        print(f"  fetch+signal {sym}…", flush=True)
        t0 = time.time()
        try:
            pack = fetch_pack(sym, win_1m[1], win_1m[2])
            bars, sigs = cache_signals(pack, sym)
            caches[sym] = (bars, sigs)
            print(
                f"    ok {time.time()-t0:.0f}s bars={len(bars)} hits={sum(1 for s in sigs if s)}",
                flush=True,
            )
        except Exception as exc:
            print(f"    FAIL {sym}: {exc}", flush=True)

    results = []
    for label, start, end_s in windows:
        book_bars, book_sigs = {}, {}
        for sym, (bars, sigs) in caches.items():
            b2, s2 = slice_range(bars, sigs, start, end_s)
            book_bars[sym] = b2
            book_sigs[sym] = s2
        base = replay_book(book_bars, book_sigs, mode="baseline")
        abce = replay_book(book_bars, book_sigs, mode="abce")
        row = {"window": label, "baseline": base, "abce": abce}
        results.append(row)
        print(f"\n### {label}", flush=True)
        print(
            f"  baseline: n={base['trades']} wr={base['win_rate']}% "
            f"pnl_eq=${base['pnl_equal']} oldAlloc=${base['pnl_old_alloc']} "
            f"newAlloc=${base['pnl_new_alloc']}",
            flush=True,
        )
        print(
            f"  abce:     n={abce['trades']} wr={abce['win_rate']}% "
            f"pnl_eq=${abce['pnl_equal']} oldAlloc=${abce['pnl_old_alloc']} "
            f"newAlloc=${abce['pnl_new_alloc']} skip={abce['skipped']}",
            flush=True,
        )
        print(f"  abce by_sym: {abce['by_symbol']}", flush=True)

    # Live lot cap check (A)
    from strategy.dynamic_lots import sizing_snapshot

    lot_check = {}
    for s in BOOK:
        snap = sizing_snapshot(s)
        lot_check[s] = {
            "lot": snap.get("lot_size"),
            "exp_loss": snap.get("expected_loss_usd"),
            "exp_profit": snap.get("expected_profit_usd"),
            "cap": snap.get("risk_usd_cap"),
            "alloc": snap.get("allocation_mult"),
        }

    payload = {
        "results": results,
        "lot_check": lot_check,
        "alloc_new": ALLOC,
        "alloc_old": OLD_ALLOC,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n### Lot risk cap check (A+B live sizing)", flush=True)
    for s, v in lot_check.items():
        ok = (v.get("exp_loss") or 0) <= (v.get("cap") or 200) + 1
        print(
            f"  {s:8} lot={v['lot']} loss=${v['exp_loss']} cap=${v['cap']} "
            f"alloc×={v['alloc']} {'OK' if ok else 'OVER'}",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
