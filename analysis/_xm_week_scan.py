#!/usr/bin/env python3
"""Last-week XM-relevant scan — rank symbols by live-style PnL.

Universe = FX + metals/energy + crypto + index CFDs (XM "stocks").
Uses fixed $200 risk/trade so oil/index lots don't dominate the ranking.
Also reports desk-lot PnL for reference.

Writes /tmp/xm_week_scan.json
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

# XM-tradable set we can actually backtest (indices = stock CFDs on XM)
SCAN = [
    # FX
    ("EURUSD", "fx"),
    ("GBPUSD", "fx"),
    ("USDJPY", "fx"),
    ("USDCHF", "fx"),
    ("AUDUSD", "fx"),
    ("EURJPY", "fx"),
    # commodities
    ("GOLD", "commodity"),
    ("SILVER", "commodity"),
    ("USOIL", "commodity"),
    ("UKOIL", "commodity"),
    # crypto
    ("BTCUSD", "crypto"),
    ("ETHUSD", "crypto"),
    # index / "stocks" CFDs on XM
    ("US30", "index"),
    ("US500", "index"),
    ("NAS100", "index"),
]

RISK_USD = 200.0
CAPITAL = 9928.0
MC_FLOOR = 42.0
OUT = Path("/tmp/xm_week_scan.json")


def _week_range() -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    weekday = today.weekday()
    last_fri = today - timedelta(days=(weekday - 4) % 7 or 7)
    last_mon = last_fri - timedelta(days=4)
    return last_mon.isoformat(), last_fri.isoformat()


def _lot_for_risk(symbol: str, entry: float, sl: float, risk: float = RISK_USD) -> float:
    from strategy.position_sizing import leg_pnl

    p1 = abs(float(leg_pnl("BUY", float(entry), float(sl), 1.0, symbol=symbol)))
    if p1 < 1e-12:
        return 0.01
    return max(0.01, round(risk / p1, 2))


def _desk_lot(symbol: str) -> float:
    from strategy.dynamic_lots import lot_for_profit_target

    return float(lot_for_profit_target(symbol, capital=CAPITAL).get("lot_size") or 0.01)


def _prepare(df_raw):
    from analysis.backtest_engine import _resample_ohlcv
    from indicators.prepare import prepare_df

    return (
        prepare_df(df_raw.copy()),
        prepare_df(_resample_ohlcv(df_raw, "15min").copy()),
        prepare_df(_resample_ohlcv(df_raw, "1h").copy()),
        prepare_df(_resample_ohlcv(df_raw, "4h").copy()),
    )


def cache_signals(symbol: str, start: str, end: str) -> dict[str, Any]:
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
            {"t": str(bar_time), "h": float(row["high"]), "l": float(row["low"]), "c": float(row["close"])}
        )
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
                        "BRAIN": r.get("BRAIN"),
                    }
                )
            else:
                signals.append(None)
        except Exception:
            signals.append(None)
        finally:
            set_backtest_bar_time(None)
        if step and step % 250 == 0:
            print(f"  {symbol}: {step}/{len(in_range)}", flush=True)

    return {
        "symbol": symbol,
        "bars": bars,
        "signals": signals,
        "desk_lot": _desk_lot(symbol),
        "sec": round(time.time() - t0, 1),
        "n_bars": len(bars),
    }


def replay(cache: dict, *, fixed_risk: bool) -> dict[str, Any]:
    from config import ENTRY_COOLDOWN_BARS
    from strategy.trade_simulator import (
        apply_rr_mode,
        close_at_market,
        new_open_trade,
        stats_from_trades,
        update_trade_on_bar,
    )

    bars, signals = cache["bars"], cache["signals"]
    desk = float(cache["desk_lot"])
    open_trade = None
    cooldown = -1
    trades: list[dict] = []

    for step, meta in enumerate(bars):
        h, l, c = meta["h"], meta["l"], meta["c"]
        if open_trade is not None:
            lot = float(open_trade.get("lot_size") or desk)
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
        lot = (
            _lot_for_risk(cache["symbol"], float(entry), float(sl))
            if fixed_risk
            else desk
        )
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
        "pnl": st["realized_pnl"],
        "avg_lot": round(
            sum(float(t.get("lot_size") or 0) for t in trades) / len(trades), 4
        )
        if trades
        else None,
    }


def main() -> int:
    # ensure UKOIL / AUDUSD / EURJPY in yfinance map
    from data import historical as hist

    hist._YF_SYMBOLS.update(
        {
            "UKOIL": ("BZ=F",),
            "AUDUSD": ("AUDUSD=X",),
            "EURJPY": ("EURJPY=X",),
            "GER40": ("^GDAXI",),
        }
    )

    start, end = _week_range()
    print(f"XM week scan {start}→{end}  fixed risk ${RISK_USD:.0f}/trade", flush=True)

    rows = []
    for sym, cls in SCAN:
        print(f"\n=== {sym} ({cls}) ===", flush=True)
        try:
            cache = cache_signals(sym, start, end)
            print(f"  bars={cache['n_bars']} desk_lot={cache['desk_lot']} t={cache['sec']}s", flush=True)
            risk = replay(cache, fixed_risk=True)
            desk = replay(cache, fixed_risk=False)
            row = {
                "symbol": sym,
                "class": cls,
                "trades": risk["trades"],
                "wins": risk["wins"],
                "losses": risk["losses"],
                "win_rate": risk["win_rate"],
                "pnl_risk200": risk["pnl"],
                "avg_lot_risk200": risk["avg_lot"],
                "pnl_desk": desk["pnl"],
                "desk_lot": cache["desk_lot"],
            }
            rows.append(row)
            print(
                f"  risk$200: n={risk['trades']} wr={100*float(risk['win_rate']):.0f}% "
                f"pnl=${risk['pnl']:.0f} | desk: ${desk['pnl']:.0f}",
                flush=True,
            )
        except Exception as exc:
            print(f"  FAIL: {exc}", flush=True)
            traceback.print_exc()
            rows.append(
                {
                    "symbol": sym,
                    "class": cls,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0,
                    "pnl_risk200": 0,
                    "pnl_desk": 0,
                    "error": str(exc),
                }
            )

    ranked = sorted(
        [r for r in rows if not r.get("error")],
        key=lambda r: (float(r["pnl_risk200"]), float(r.get("win_rate") or 0)),
        reverse=True,
    )
    top5 = ranked[:5]

    payload = {
        "range": {"start": start, "end": end},
        "risk_usd": RISK_USD,
        "mode": "auto + partials + trail",
        "note": (
            "XM indices (US30/US500/NAS100) stand in for stock CFDs — "
            "single-name stocks aren't in this desk's MetaAPI map. "
            "Ranked by fixed $200 risk/trade PnL."
        ),
        "top5": top5,
        "all": ranked,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print("\n### TOP 5 (fixed $200 risk)\n", flush=True)
    print("| rank | symbol | class | trades | WR | PnL@$200R | desk PnL |", flush=True)
    print("|---:|---|---|---:|---:|---:|---:|", flush=True)
    for i, r in enumerate(top5, 1):
        print(
            f"| {i} | {r['symbol']} | {r['class']} | {r['trades']} | "
            f"{100*float(r['win_rate']):.0f}% | {r['pnl_risk200']} | {r['pnl_desk']} |",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
