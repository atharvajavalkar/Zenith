#!/usr/bin/env python3
"""XM multi-window scan — 14d / 1m / 2m — composite top 5.

Caches signals once over the longest window, then slices + replays.
Fixed $200 risk/trade, auto + partials + trail.
Writes /tmp/xm_multi_scan.json
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCAN = [
    ("EURUSD", "fx"),
    ("GBPUSD", "fx"),
    ("USDJPY", "fx"),
    ("USDCHF", "fx"),
    ("AUDUSD", "fx"),
    ("EURJPY", "fx"),
    ("GOLD", "commodity"),
    ("SILVER", "commodity"),
    ("USOIL", "commodity"),
    ("UKOIL", "commodity"),
    ("BTCUSD", "crypto"),
    ("ETHUSD", "crypto"),
    ("US30", "index"),
    ("US500", "index"),
    ("NAS100", "index"),
]

RISK_USD = 200.0
MC_FLOOR = 42.0
OUT = Path("/tmp/xm_multi_scan.json")


def _windows(end) -> list[tuple[str, str, str]]:
    return [
        ("14d", (end - timedelta(days=13)).isoformat(), end.isoformat()),
        ("1m", (end - timedelta(days=29)).isoformat(), end.isoformat()),
        ("2m", (end - timedelta(days=59)).isoformat(), end.isoformat()),
    ]


def _lot_for_risk(symbol: str, entry: float, sl: float, risk: float = RISK_USD) -> float:
    from strategy.position_sizing import leg_pnl

    p1 = abs(float(leg_pnl("BUY", float(entry), float(sl), 1.0, symbol=symbol)))
    if p1 < 1e-12:
        return 0.01
    return max(0.01, round(risk / p1, 2))


def _prepare(df_raw):
    from analysis.backtest_engine import _resample_ohlcv
    from indicators.prepare import prepare_df

    return (
        prepare_df(df_raw.copy()),
        prepare_df(_resample_ohlcv(df_raw, "15min").copy()),
        prepare_df(_resample_ohlcv(df_raw, "1h").copy()),
        prepare_df(_resample_ohlcv(df_raw, "4h").copy()),
    )


def cache_signals_2m(symbol: str, start: str, end: str) -> dict[str, Any]:
    from analysis.backtest_engine import signal_slices_ready, warmup_days_for_backtest
    from analysis.session import set_backtest_bar_time
    from data.historical import fetch_symbol_ohlcv
    from strategy.signal import _compute_signal

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    warmup = warmup_days_for_backtest()
    ws = (start_dt - timedelta(days=warmup)).strftime("%Y-%m-%d")
    fe = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    t0 = time.time()
    df_raw = fetch_symbol_ohlcv(symbol, ws, fe, interval="5m")
    if df_raw is None or df_raw.empty:
        raise RuntimeError("no OHLCV")

    full_5m, full_15m, full_1h, full_4h = _prepare(df_raw)
    in_range = full_5m.index[(full_5m.index >= start_dt) & (full_5m.index <= end_dt)]
    raw_5m = df_raw.reindex(full_5m.index).ffill()

    signals: list[Optional[dict]] = []
    bars: list[dict] = []
    for step, bar_time in enumerate(in_range):
        bar_pos = full_5m.index.get_loc(bar_time)
        p5m = full_5m.iloc[: bar_pos + 1]
        p15m = full_15m[full_15m.index <= bar_time]
        p1h = full_1h[full_1h.index <= bar_time]
        p4h = full_4h[full_4h.index <= bar_time]
        row = raw_5m.loc[bar_time] if bar_time in raw_5m.index else full_5m.loc[bar_time]
        bars.append(
            {
                "t": str(bar_time),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
            }
        )
        if not signal_slices_ready(p5m, p15m, p1h, p4h):
            signals.append(None)
            continue
        set_backtest_bar_time(bar_time.to_pydatetime())
        try:
            r = _compute_signal(
                df_1m=p5m,
                df_3m=p5m,
                df_5m=p5m,
                df_15m=p15m,
                df_1h=p1h,
                df_4h=p4h,
                symbol=symbol,
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
            print(f"    {symbol}: {step}/{len(in_range)}", flush=True)

    return {
        "symbol": symbol,
        "bars": bars,
        "signals": signals,
        "sec": round(time.time() - t0, 1),
        "n_bars": len(bars),
    }


def slice_cache(cache: dict, start: str, end: str) -> dict:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    bars, signals = [], []
    for b, s in zip(cache["bars"], cache["signals"]):
        t = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        if start_dt <= t <= end_dt:
            bars.append(b)
            signals.append(s)
    return {"symbol": cache["symbol"], "bars": bars, "signals": signals}


def replay(cache: dict) -> dict[str, Any]:
    from config import ENTRY_COOLDOWN_BARS
    from strategy.trade_simulator import (
        apply_rr_mode,
        close_at_market,
        new_open_trade,
        stats_from_trades,
        update_trade_on_bar,
    )

    bars, signals = cache["bars"], cache["signals"]
    if not bars:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "pnl": 0.0}

    open_trade = None
    cooldown = -1
    trades: list[dict] = []

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
        result = apply_rr_mode(dict(raw), "auto")
        if result is None or result.get("SIGNAL") not in ("BUY", "SELL"):
            continue
        mc = float(result.get("SUCCESS_PERCENT") or result.get("MARKOV_MC_PROB") or 0)
        if mc < MC_FLOOR:
            continue
        entry, sl, tp = result.get("ENTRY"), result.get("SL"), result.get("TP")
        if entry is None or sl is None:
            continue
        lot = _lot_for_risk(cache["symbol"], float(entry), float(sl))
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
            symbol=cache["symbol"],
            SYMBOL=cache["symbol"],
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
        "win_rate": st["win_rate"],
        "pnl": float(st["realized_pnl"]),
    }


def main() -> int:
    from data import historical as hist

    hist._YF_SYMBOLS.update(
        {
            "UKOIL": ("BZ=F",),
            "AUDUSD": ("AUDUSD=X",),
            "EURJPY": ("EURJPY=X",),
        }
    )

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    windows = _windows(end)
    win_2m = next(w for w in windows if w[0] == "2m")
    print(
        "XM multi-window scan (one 2m cache → slice)",
        [f"{l}:{a}→{b}" for l, a, b in windows],
        flush=True,
    )

    by_window: dict[str, list[dict]] = {l: [] for l, _, _ in windows}

    for sym, cls in SCAN:
        print(f"\n######## {sym} ({cls}) — cache 2m ########", flush=True)
        try:
            cache = cache_signals_2m(sym, win_2m[1], win_2m[2])
            print(f"  cached bars={cache['n_bars']} t={cache['sec']}s", flush=True)
        except Exception as exc:
            print(f"  FAIL cache: {exc}", flush=True)
            traceback.print_exc()
            for label, _, _ in windows:
                by_window[label].append(
                    {
                        "symbol": sym,
                        "class": cls,
                        "window": label,
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "win_rate": 0,
                        "pnl": 0.0,
                        "error": str(exc),
                    }
                )
            continue

        for label, start, end_s in windows:
            sub = slice_cache(cache, start, end_s)
            st = replay(sub)
            row = {
                "symbol": sym,
                "class": cls,
                "window": label,
                "start": start,
                "end": end_s,
                "bars": len(sub["bars"]),
                "trades": st["trades"],
                "wins": st["wins"],
                "losses": st["losses"],
                "win_rate": st["win_rate"],
                "pnl": st["pnl"],
            }
            by_window[label].append(row)
            print(
                f"  [{label}] n={st['trades']} wr={100*float(st['win_rate']):.0f}% "
                f"pnl=${st['pnl']:.0f} bars={len(sub['bars'])}",
                flush=True,
            )

    class_map = {s: c for s, c in SCAN}
    composite = []
    for sym, _ in SCAN:
        pnls, trades, wrs = {}, {}, {}
        for label, rows in by_window.items():
            hit = next((r for r in rows if r["symbol"] == sym), None)
            pnls[label] = float(hit["pnl"]) if hit and not hit.get("error") else 0.0
            trades[label] = int(hit["trades"]) if hit else 0
            wrs[label] = float(hit["win_rate"]) if hit else 0.0
        green = sum(1 for v in pnls.values() if v > 0)
        total = sum(pnls.values())
        score = total + green * 500.0
        composite.append(
            {
                "symbol": sym,
                "class": class_map[sym],
                "pnl_14d": round(pnls.get("14d", 0), 2),
                "pnl_1m": round(pnls.get("1m", 0), 2),
                "pnl_2m": round(pnls.get("2m", 0), 2),
                "trades_14d": trades.get("14d", 0),
                "trades_1m": trades.get("1m", 0),
                "trades_2m": trades.get("2m", 0),
                "wr_14d": wrs.get("14d", 0),
                "wr_1m": wrs.get("1m", 0),
                "wr_2m": wrs.get("2m", 0),
                "green_windows": green,
                "pnl_sum": round(total, 2),
                "score": round(score, 2),
            }
        )

    ranked = sorted(composite, key=lambda r: (r["score"], r["pnl_sum"]), reverse=True)
    top5 = ranked[:5]

    window_tops = {}
    for label, rows in by_window.items():
        ok = [r for r in rows if not r.get("error")]
        window_tops[label] = sorted(ok, key=lambda r: r["pnl"], reverse=True)[:5]

    payload = {
        "risk_usd": RISK_USD,
        "mode": "auto + partials + trail",
        "windows": {l: {"start": a, "end": b} for l, a, b in windows},
        "note": (
            "Composite = sum(PnL 14d+1m+2m) + 500×(# green windows). "
            "Fixed $200 risk/trade. Indices = XM stock CFDs."
        ),
        "top5": top5,
        "all": ranked,
        "window_tops": {
            k: [
                {
                    "symbol": r["symbol"],
                    "class": r["class"],
                    "trades": r["trades"],
                    "win_rate": r["win_rate"],
                    "pnl": r["pnl"],
                }
                for r in v
            ]
            for k, v in window_tops.items()
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print("\n### PER-WINDOW TOP 5\n", flush=True)
    for label in ("14d", "1m", "2m"):
        print(f"— {label} —", flush=True)
        for i, r in enumerate(window_tops[label], 1):
            print(
                f"  {i}. {r['symbol']:8} n={r['trades']:3} "
                f"wr={100*float(r['win_rate']):5.1f}% pnl=${r['pnl']:.0f}",
                flush=True,
            )

    print("\n### COMPOSITE TOP 5\n", flush=True)
    print("| # | symbol | class | 14d | 1m | 2m | green | sum |", flush=True)
    print("|---:|---|---|---:|---:|---:|---:|---:|", flush=True)
    for i, r in enumerate(top5, 1):
        print(
            f"| {i} | {r['symbol']} | {r['class']} | "
            f"{r['pnl_14d']:.0f} | {r['pnl_1m']:.0f} | {r['pnl_2m']:.0f} | "
            f"{r['green_windows']}/3 | {r['pnl_sum']:.0f} |",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
