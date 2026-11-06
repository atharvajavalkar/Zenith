#!/usr/bin/env python3
"""Backtest GOLD fib_liq_sweep brain (15m) + feed closes into instrument RL.

Live brain parity: Fib 0.50–0.618 + liq sweep · 2R · partials/trail optional.
Windows: 14d + 1m. Risk $200/R.
RL: chronological on_trade_closed for GOLD only (does not rebuild other symbols).
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

OUT = Path("/tmp/gold_fib_rl_bt.json")
RISK_USD = 200.0


def _fetch_15m(start: str, end: str):
    from data.historical import fetch_symbol_ohlcv
    from indicators.prepare import prepare_df
    import pandas as pd

    warm = (datetime.fromisoformat(start) - timedelta(days=20)).strftime("%Y-%m-%d")
    end_p = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = fetch_symbol_ohlcv("GOLD", warm, end_p, interval="5m")
    if raw is None or raw.empty:
        import yfinance as yf

        df = yf.Ticker("GC=F").history(start=warm, end=end_p, interval="5m", auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError("no GOLD data")
        df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        raw = df
    else:
        raw = raw.rename(columns=str.lower)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")

    o = raw["open"].resample("15min").first()
    h = raw["high"].resample("15min").max()
    l = raw["low"].resample("15min").min()
    c = raw["close"].resample("15min").last()
    df15 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    return prepare_df(df15.copy())


def _simulate(df, start: str, end: str) -> dict[str, Any]:
    from strategy.fib_liq_brain import evaluate_fib_liq_sweep
    from analysis.instrument_rl import get_active_arm, overlays_for, on_trade_closed

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    trades: list[dict] = []
    open_t: Optional[dict] = None
    cooldown_i = -1
    signals = 0
    skipped_rl = 0

    for i in range(80, len(df)):
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
                # RL risk_scale already applied at entry via lot/risk — PnL in $200R units
                pnl = round(float(r_mult) * RISK_USD * float(open_t.get("risk_scale") or 1.0), 2)
                if start_dt <= open_t["entry_t"] <= end_dt:
                    closed = {
                        "status": "closed",
                        "signal": side,
                        "symbol": "GOLD",
                        "pnl": pnl,
                        "entry": open_t["entry"],
                        "sl": open_t["sl"],
                        "tp": open_t["tp"],
                        "exit": float(exit_px),
                        "lot_size": 0.1,
                        "created_at": str(open_t["entry_t"]),
                        "log_time": str(ts),
                        "features": {
                            "rl_arm": open_t.get("rl_arm"),
                            "brain": "fib_liq",
                            "exec_tf": "15m",
                            "risk_usd": RISK_USD * float(open_t.get("risk_scale") or 1.0),
                        },
                        "oanda_transaction_id": f"fib-bt-{len(trades)}",
                    }
                    # Feed live GOLD RL (online update — other symbols untouched)
                    rl_snap = on_trade_closed(closed)
                    trades.append(
                        {
                            "entry_t": str(open_t["entry_t"])[:16],
                            "exit_t": str(ts)[:16],
                            "side": side,
                            "pnl": pnl,
                            "r": round(float(r_mult), 3),
                            "reason": "sl" if hit_sl else "tp_2r",
                            "rl_arm": open_t.get("rl_arm"),
                            "rl_after": rl_snap.get("active_arm"),
                            "rl_msg": rl_snap.get("message"),
                        }
                    )
                open_t = None
                cooldown_i = i + 2
            continue

        if ts < start_dt or ts > end_dt or i < cooldown_i:
            continue

        hit = evaluate_fib_liq_sweep(df.iloc[: i + 1])
        if not hit:
            continue
        signals += 1

        ov = overlays_for("GOLD")
        arm = get_active_arm("GOLD")
        risk_scale = float(ov.get("risk_scale") or 1.0)
        # Tight arm after losses: still take trade but scaled risk (research + live policy)
        if risk_scale < 0.5:
            skipped_rl += 1
            continue

        entry = float(hit["entry"])
        sl = float(hit["sl"])
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        # Optional: widen prefer-RR via arm — fib brain stays 2R (forced research)
        tp = entry + 2.0 * risk if hit["signal"] == "BUY" else entry - 2.0 * risk
        open_t = {
            "side": hit["signal"],
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "entry_t": ts,
            "rl_arm": arm,
            "risk_scale": risk_scale,
        }

    if open_t is not None and start_dt <= open_t["entry_t"] <= end_dt:
        c = float(df.iloc[-1]["close"])
        risk = abs(open_t["entry"] - open_t["sl"])
        r_mult = (
            (c - open_t["entry"]) / risk
            if open_t["side"] == "BUY"
            else (open_t["entry"] - c) / risk
        )
        pnl = round(float(r_mult) * RISK_USD * float(open_t.get("risk_scale") or 1.0), 2)
        closed = {
            "status": "closed",
            "signal": open_t["side"],
            "symbol": "GOLD",
            "pnl": pnl,
            "entry": open_t["entry"],
            "sl": open_t["sl"],
            "lot_size": 0.1,
            "features": {
                "rl_arm": open_t.get("rl_arm"),
                "brain": "fib_liq",
                "risk_usd": RISK_USD * float(open_t.get("risk_scale") or 1.0),
            },
            "oanda_transaction_id": f"fib-bt-eod-{len(trades)}",
            "created_at": str(open_t["entry_t"]),
            "log_time": str(df.index[-1]),
        }
        rl_snap = on_trade_closed(closed)
        trades.append(
            {
                "entry_t": str(open_t["entry_t"])[:16],
                "exit_t": str(df.index[-1])[:16],
                "side": open_t["side"],
                "pnl": pnl,
                "r": round(float(r_mult), 3),
                "reason": "eod",
                "rl_arm": open_t.get("rl_arm"),
                "rl_after": rl_snap.get("active_arm"),
                "rl_msg": rl_snap.get("message"),
            }
        )

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    dec = wins + losses
    pnl = round(sum(t["pnl"] for t in trades), 2)
    return {
        "start": start,
        "end": end,
        "signals": signals,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / dec, 1) if dec else 0.0,
        "pnl": pnl,
        "avg_pnl": round(pnl / len(trades), 2) if trades else 0.0,
        "skipped_rl": skipped_rl,
        "trade_list": trades,
    }


def main() -> int:
    from analysis.instrument_rl import get_state, overlays_for, get_active_arm

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    windows = [
        ("14d", (end - timedelta(days=13)).isoformat(), end.isoformat()),
        ("1m", (end - timedelta(days=29)).isoformat(), end.isoformat()),
    ]
    print("GOLD fib_liq + RL backtest", windows, flush=True)
    t0 = time.time()
    df = _fetch_15m(windows[1][1], windows[1][2])
    print(f"15m bars={len(df)}", flush=True)

    rl_before = {
        "arm": get_active_arm("GOLD"),
        "overlays": overlays_for("GOLD"),
        "row": (get_state().get("instruments") or {}).get("GOLD"),
    }

    results = {}
    full = _simulate(df, windows[1][1], windows[1][2])
    all_trades = list(full["trade_list"])
    results["1m"] = {k: v for k, v in full.items() if k != "trade_list"}
    results["1m"]["trade_list"] = all_trades[:12]

    start_14 = windows[0][1]
    t14 = [t for t in all_trades if t["entry_t"][:10] >= start_14]
    w14 = sum(1 for t in t14 if t["pnl"] > 0)
    l14 = sum(1 for t in t14 if t["pnl"] < 0)
    d14 = w14 + l14
    p14 = round(sum(t["pnl"] for t in t14), 2)
    results["14d"] = {
        "start": windows[0][1],
        "end": windows[0][2],
        "trades": len(t14),
        "wins": w14,
        "losses": l14,
        "win_rate": round(100.0 * w14 / d14, 1) if d14 else 0.0,
        "pnl": p14,
        "avg_pnl": round(p14 / len(t14), 2) if t14 else 0.0,
        "trade_list": t14[:12],
    }

    gold_rl = (get_state().get("instruments") or {}).get("GOLD") or {}
    rl_after = {
        "arm": get_active_arm("GOLD"),
        "overlays": overlays_for("GOLD"),
        "n_trades": gold_rl.get("n_trades"),
        "wins": gold_rl.get("wins"),
        "losses": gold_rl.get("losses"),
        "win_rate": gold_rl.get("win_rate"),
        "expectancy_r": gold_rl.get("expectancy_r"),
        "message": gold_rl.get("message"),
        "policy": gold_rl.get("policy"),
        "arms": gold_rl.get("arms"),
    }

    payload = {
        "brain": "fib_liq_sweep",
        "fibs": [0.50, 0.618],
        "exec_tf": "15m",
        "risk_usd": RISK_USD,
        "sec": round(time.time() - t0, 1),
        "windows": results,
        "rl_before": {
            "arm": rl_before["arm"],
            "overlays": {
                k: rl_before["overlays"].get(k)
                for k in ("arm", "aggression", "risk_scale", "mc_floor_delta", "rr_prefer")
            },
        },
        "rl_after": rl_after,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### GOLD Fib+Liq Sweep backtest ($200/R)\n", flush=True)
    print("| Window | Trades | WR | PnL | Avg/trade |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    for label in ("14d", "1m"):
        r = results[label]
        print(
            f"| {label} | {r['trades']} | {r['win_rate']}% | ${r['pnl']} | ${r['avg_pnl']} |",
            flush=True,
        )

    print("\n### GOLD RL after feeding backtest closes\n", flush=True)
    print("| Field | Before | After |", flush=True)
    print("|---|---|---|", flush=True)
    print(
        f"| Arm | {rl_before['arm']} | {rl_after['arm']} |",
        flush=True,
    )
    print(
        f"| Aggression | {rl_before['overlays'].get('aggression')} | "
        f"{(rl_after.get('overlays') or {}).get('aggression')} |",
        flush=True,
    )
    print(
        f"| Risk scale | {rl_before['overlays'].get('risk_scale')} | "
        f"{(rl_after.get('overlays') or {}).get('risk_scale')} |",
        flush=True,
    )
    print(
        f"| MC floor Δ | {rl_before['overlays'].get('mc_floor_delta')} | "
        f"{(rl_after.get('overlays') or {}).get('mc_floor_delta')} |",
        flush=True,
    )
    print(
        f"| N / WR / E[R] | — | {rl_after.get('n_trades')} / "
        f"{rl_after.get('win_rate')} / {rl_after.get('expectancy_r')} |",
        flush=True,
    )
    print(f"| Message | — | {rl_after.get('message')} |", flush=True)
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
