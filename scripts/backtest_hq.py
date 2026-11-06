"""High-quantity / tiny-reward Hyper research backtest (NOT integrated).

Tests the user's idea: drop the time-stop, punch a trade nearly every candle
in the trend direction with a very small TP/SL, and try to win on QUANTITY.

Honesty constraints:
  • Real 1m gold history is capped at ~1000 bars (~18h) via the live feed, so
    we also test the same concept on ~30d of real 5m to get a bigger sample.
  • The make-or-break factor for tiny-TP/high-frequency is SPREAD, so every
    trade is charged a realistic round-trip gold spread. We print results
    WITHOUT spread (fantasy) and WITH spread (reality) side by side.

Strategy under test ("trade every candle"):
  • Trend filter: EMA(fast) vs EMA(slow) -> BUY if fast>=slow else SELL.
  • Enter at each candle open in the trend direction (one position at a time;
    tiny stops usually resolve within 1-2 candles, so this ~= every candle).
  • TP = tp_atr * ATR, SL = sl_atr * ATR. No time stop (runs to TP or SL).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, "/app")

from data.historical import fetch_symbol_ohlcv  # noqa: E402
from data.market_data import fetch_ohlcv  # noqa: E402
from indicators.technical import atr as atr_fn, ema as ema_fn  # noqa: E402

# Gold: 1.0 lot = 100 oz -> $1.00 price move = $100 per lot.
USD_PER_DOLLAR_PER_LOT = 100.0
# Realistic XM gold spreads (USD). We sweep a few.
SPREADS = (0.0, 0.15, 0.25, 0.35)


@dataclass
class Params:
    ema_fast: int = 8
    ema_slow: int = 14
    tp_atr: float = 0.20
    sl_atr: float = 0.20
    max_hold: int = 40      # generous cap only so runaway trades still close
    lot: float = 1.0


@dataclass
class Result:
    r: list = field(default_factory=list)          # realized R (pre-spread)
    price_pnl: list = field(default_factory=list)   # realized price move (pre-spread)
    risk: list = field(default_factory=list)        # $risk distance per trade

    def stats(self, spread: float) -> dict[str, Any]:
        if not self.r:
            return {"trades": 0}
        price = np.array(self.price_pnl) - spread   # charge round-trip spread
        risk = np.array(self.risk)
        r_net = price / risk
        usd = price * USD_PER_DOLLAR_PER_LOT        # per 1.0 lot
        wins = int((price > 0).sum())
        losses = int((price < 0).sum())
        dec = wins + losses
        gw = price[price > 0].sum()
        gl = -price[price < 0].sum()
        pf = (gw / gl) if gl > 1e-9 else (999.0 if gw > 0 else 0.0)
        eq = np.cumsum(usd)
        dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
        return {
            "trades": len(price),
            "win_rate": round(100.0 * wins / dec, 1) if dec else 0.0,
            "expectancy_R": round(float(r_net.mean()), 4),
            "net_$_per_lot": round(float(usd.sum()), 0),
            "avg_$_per_trade": round(float(usd.mean()), 2),
            "profit_factor": round(float(pf), 2),
            "max_dd_$": round(dd, 0),
        }


def run(df: pd.DataFrame, p: Params) -> Result:
    close = df["close"].astype(float)
    ef = ema_fn(close, p.ema_fast).to_numpy()
    es = ema_fn(close, p.ema_slow).to_numpy()
    atr = atr_fn(df, 14).bfill().fillna(0.0).to_numpy()
    o = df["open"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    n = len(df)
    res = Result()
    i = p.ema_slow + 1
    while i < n - 1:
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue
        direction = "BUY" if ef[i] >= es[i] else "SELL"
        entry = o[i + 1]
        tp_d = p.tp_atr * a
        sl_d = p.sl_atr * a
        if direction == "BUY":
            sl, tp = entry - sl_d, entry + tp_d
        else:
            sl, tp = entry + sl_d, entry - tp_d
        risk = abs(entry - sl)
        exit_px = None
        for j in range(i + 1, min(i + 1 + p.max_hold, n)):
            if direction == "BUY":
                if lo[j] <= sl:
                    exit_px = sl
                    break
                if hi[j] >= tp:
                    exit_px = tp
                    break
            else:
                if hi[j] >= sl:
                    exit_px = sl
                    break
                if lo[j] <= tp:
                    exit_px = tp
                    break
        if exit_px is None:
            j = min(i + p.max_hold, n - 1)
            exit_px = cl[j]
        move = (exit_px - entry) if direction == "BUY" else (entry - exit_px)
        res.r.append(move / risk)
        res.price_pnl.append(move)
        res.risk.append(risk)
        i = j + 1  # next trade after this one closes (serialized, ~every candle)
    return res


def study(label: str, df: pd.DataFrame, grid: list[Params]) -> None:
    print(f"\n{'='*78}\n{label}  |  bars={len(df)}  "
          f"({df.index[0]:%Y-%m-%d %H:%M} -> {df.index[-1]:%Y-%m-%d %H:%M})\n{'='*78}")
    for p in grid:
        res = run(df, p)
        head = f"  TP={p.tp_atr}xATR SL={p.sl_atr}xATR "
        print(head)
        for sp in SPREADS:
            st = res.stats(sp)
            tag = "no-spread" if sp == 0 else f"spread ${sp:.2f}"
            print(f"      {tag:>13}: {st}")


def main() -> None:
    grid = [
        Params(tp_atr=0.15, sl_atr=0.15),
        Params(tp_atr=0.20, sl_atr=0.20),
        Params(tp_atr=0.30, sl_atr=0.30),
        Params(tp_atr=0.50, sl_atr=0.30),  # slightly bigger reward
    ]

    # Real 1m (small ~18h sample)
    try:
        df1 = fetch_ohlcv("1m", 1000, symbol="GOLD")
        df1 = df1[["open", "high", "low", "close", "volume"]].astype(float)
        study("GOLD 1m (real, ~18h)", df1, grid)
    except Exception as e:
        print("1m load failed:", e)

    # Real 5m (~30d) for a bigger sample of the SAME concept
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=150)
        df5 = fetch_symbol_ohlcv("GOLD", start.isoformat(), end.isoformat())
        df5 = df5[["open", "high", "low", "close", "volume"]].astype(float)
        study("GOLD 5m (real, ~30d)", df5, grid)
    except Exception as e:
        print("5m load failed:", e)


if __name__ == "__main__":
    main()
