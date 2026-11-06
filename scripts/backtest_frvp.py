"""FRVP value-area backtest for the Hyper research question.

Tests the user's TradingView setup (Fixed Range Volume Profile: 50 rows,
70% value area, plotting VAH/VAL/POC) as a mechanical strategy on REAL data.

Because Zenith has no historical sub-minute data, we cannot faithfully replay
the 10s Hyper engine. Instead we validate the *volume-profile value-area edge*
on the finest real history we have:

  • 1h gold  — faithful to the screenshot (resampled from real 5m; coarsening
               is lossless), and
  • 5m gold  — faster cadence, a proxy for "does this logic survive at scalp
               speed" (the only thing that could justify feeding it to Hyper).

Two entry models, matching standard VP practice:
  • reversion : fade VAH -> POC (SELL) and VAL -> POC (BUY) inside balance
  • breakout  : buy acceptance above VAH, sell acceptance below VAL

We RL/grid-tune parameters on a train split and report OUT-OF-SAMPLE results.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, "/app")

from data.historical import fetch_symbol_ohlcv, resample_ohlcv  # noqa: E402
from indicators.technical import atr as atr_fn  # noqa: E402


def fast_profile(win: pd.DataFrame, n_bins: int, va_pct: float) -> Optional[dict]:
    """Vectorized FRVP: hlc3-weighted-by-volume histogram, VA grown from POC.

    Faithful enough to the production profile for research (POC/VAH/VAL), but
    ~100x faster because it avoids per-row Python iteration.
    """
    lo = float(win["low"].min())
    hi = float(win["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    tp = (win["high"].to_numpy() + win["low"].to_numpy() + win["close"].to_numpy()) / 3.0
    vol = win["volume"].to_numpy(dtype=float)
    if not np.isfinite(vol).any() or vol.sum() <= 0:
        vol = (win["high"].to_numpy() - win["low"].to_numpy()).clip(min=1e-6)
    edges = np.linspace(lo, hi, n_bins + 1)
    hist, _ = np.histogram(tp, bins=edges, weights=vol)
    if hist.sum() <= 0:
        return None
    poc_idx = int(np.argmax(hist))
    poc = float(0.5 * (edges[poc_idx] + edges[poc_idx + 1]))
    target = float(hist.sum()) * float(va_pct)
    left = right = poc_idx
    covered = float(hist[poc_idx])
    while covered < target and (left > 0 or right < n_bins - 1):
        lv = hist[left - 1] if left > 0 else -1.0
        rv = hist[right + 1] if right < n_bins - 1 else -1.0
        if rv >= lv and right < n_bins - 1:
            right += 1
            covered += float(hist[right])
        elif left > 0:
            left -= 1
            covered += float(hist[left])
        else:
            break
    return {"available": True, "poc": poc, "val": float(edges[left]), "vah": float(edges[right + 1])}


# ----------------------------------------------------------------------------- data
def load(symbol: str, days: int) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    df = fetch_symbol_ohlcv(symbol, start.isoformat(), end.isoformat())
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ----------------------------------------------------------------------------- params
@dataclass
class Params:
    mode: str = "reversion"        # reversion | breakout
    window: int = 48               # profile lookback (bars)
    n_bins: int = 50               # == FRVP "Row Size 50"
    va_pct: float = 0.70           # == FRVP "Value Area 70"
    trig_atr: float = 0.25         # how close to VAH/VAL to arm (ATR)
    sl_atr: float = 1.0            # stop beyond the extreme (ATR)
    tp_mode: str = "poc"           # poc | rr
    rr: float = 1.5                # TP as R multiple when tp_mode == rr
    max_hold: int = 24             # bars before time-stop
    rebuild_every: int = 1         # rebuild profile every N bars (speed)


# ----------------------------------------------------------------------------- engine
@dataclass
class Result:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    r_list: list = field(default_factory=list)

    def add(self, r: float) -> None:
        self.trades += 1
        self.r_list.append(r)
        if r > 0:
            self.wins += 1
        elif r < 0:
            self.losses += 1

    def stats(self) -> dict[str, Any]:
        if not self.r_list:
            return {"trades": 0}
        r = np.array(self.r_list, dtype=float)
        gross_win = r[r > 0].sum()
        gross_loss = -r[r < 0].sum()
        pf = (gross_win / gross_loss) if gross_loss > 1e-9 else (999.0 if gross_win > 0 else 0.0)
        eq = np.cumsum(r)
        peak = np.maximum.accumulate(eq)
        dd = float((peak - eq).max()) if len(eq) else 0.0
        decisive = self.wins + self.losses
        return {
            "trades": int(self.trades),
            "win_rate": round(100.0 * self.wins / decisive, 1) if decisive else 0.0,
            "expectancy_R": round(float(r.mean()), 4),
            "total_R": round(float(r.sum()), 2),
            "profit_factor": round(float(pf), 2),
            "avg_win_R": round(float(r[r > 0].mean()), 3) if (r > 0).any() else 0.0,
            "avg_loss_R": round(float(r[r < 0].mean()), 3) if (r < 0).any() else 0.0,
            "max_dd_R": round(dd, 2),
        }


def _simulate(df: pd.DataFrame, atr: pd.Series, i_entry: int, direction: str,
              entry: float, sl: float, tp: float, max_hold: int) -> Optional[float]:
    """Walk forward from i_entry; return realized R (SL-first on ambiguous bars)."""
    risk = abs(entry - sl)
    if risk <= 1e-9:
        return None
    n = len(df)
    for j in range(i_entry, min(i_entry + max_hold, n)):
        hi = float(df["high"].iloc[j])
        lo = float(df["low"].iloc[j])
        if direction == "BUY":
            if lo <= sl:
                return -1.0
            if hi >= tp:
                return (tp - entry) / risk
        else:
            if hi >= sl:
                return -1.0
            if lo <= tp:
                return (entry - tp) / risk
    # time stop at close
    last = float(df["close"].iloc[min(i_entry + max_hold, n) - 1])
    return ((last - entry) if direction == "BUY" else (entry - last)) / risk


def run(df: pd.DataFrame, p: Params) -> Result:
    atr = atr_fn(df, 14).bfill().fillna(0.0)
    res = Result()
    n = len(df)
    in_pos_until = -1
    prof: Optional[dict] = None

    for i in range(p.window, n - 1):
        if i <= in_pos_until:
            continue
        a = float(atr.iloc[i])
        if a <= 0:
            continue
        if prof is None or (i % p.rebuild_every == 0):
            win = df.iloc[i - p.window: i]
            prof = fast_profile(win, p.n_bins, p.va_pct)
        if not prof or not prof.get("available"):
            continue
        poc, vah, val = prof["poc"], prof["vah"], prof["val"]
        close = float(df["close"].iloc[i])
        entry = float(df["open"].iloc[i + 1])  # next-bar open (no lookahead)
        direction = None

        if p.mode == "reversion":
            # fade the extreme back toward POC while price is still in/at value edge
            if close >= vah - p.trig_atr * a and close <= vah + p.trig_atr * a:
                direction, ref = "SELL", vah
            elif close <= val + p.trig_atr * a and close >= val - p.trig_atr * a:
                direction, ref = "BUY", val
            if direction is None:
                continue
            if direction == "BUY":
                sl = ref - p.sl_atr * a
                tp = poc if p.tp_mode == "poc" else entry + p.rr * (entry - sl)
                if tp <= entry:
                    continue
            else:
                sl = ref + p.sl_atr * a
                tp = poc if p.tp_mode == "poc" else entry - p.rr * (sl - entry)
                if tp >= entry:
                    continue

        else:  # breakout / acceptance
            if close > vah + p.trig_atr * a:
                direction = "BUY"
                sl = vah - p.sl_atr * a
                tp = entry + p.rr * (entry - sl)
            elif close < val - p.trig_atr * a:
                direction = "SELL"
                sl = val + p.sl_atr * a
                tp = entry - p.rr * (sl - entry)
            else:
                continue

        r = _simulate(df, atr, i + 1, direction, entry, sl, tp, p.max_hold)
        if r is None:
            continue
        res.add(r)
        in_pos_until = i + 1  # block re-entry same bar; simple serialization

    return res


# ----------------------------------------------------------------------------- tuning
def grid() -> list[Params]:
    out: list[Params] = []
    for mode in ("reversion", "breakout"):
        for window in (36, 48, 72):
            for trig in (0.1, 0.25, 0.5):
                for sl in (0.75, 1.0, 1.5):
                    if mode == "reversion":
                        for tp_mode, rr in (("poc", 0.0), ("rr", 1.5), ("rr", 2.0)):
                            out.append(Params(mode=mode, window=window, trig_atr=trig,
                                              sl_atr=sl, tp_mode=tp_mode, rr=rr))
                    else:
                        for rr in (1.5, 2.0, 3.0):
                            out.append(Params(mode=mode, window=window, trig_atr=trig,
                                              sl_atr=sl, tp_mode="rr", rr=rr))
    return out


def split(df: pd.DataFrame, frac: float = 0.7):
    k = int(len(df) * frac)
    return df.iloc[:k], df.iloc[k:]


def score(st: dict) -> float:
    # rank configs: expectancy weighted, require a minimum sample
    if st.get("trades", 0) < 25:
        return -999.0
    return st["expectancy_R"] * min(st["trades"], 300) ** 0.5


def study(symbol: str, tf_label: str, df: pd.DataFrame) -> None:
    train, test = split(df)
    print(f"\n{'='*72}\n{symbol} {tf_label}  |  bars={len(df)}  "
          f"({df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d})  "
          f"train={len(train)} test={len(test)}\n{'='*72}")

    best_p, best_s, best_train = None, -1e9, None
    for p in grid():
        st = run(train, p).stats()
        s = score(st)
        if s > best_s:
            best_s, best_p, best_train = s, p, st

    if best_p is None:
        print("  no viable config")
        return

    oos = run(test, best_p).stats()
    full = run(df, best_p).stats()
    print(f"  BEST cfg: mode={best_p.mode} window={best_p.window} trig={best_p.trig_atr} "
          f"sl={best_p.sl_atr} tp={best_p.tp_mode}{'' if best_p.tp_mode=='poc' else f'/{best_p.rr}R'} "
          f"hold={best_p.max_hold}")
    print(f"  TRAIN : {best_train}")
    print(f"  TEST  : {oos}   <-- out-of-sample (the honest number)")
    print(f"  FULL  : {full}")


def main() -> None:
    # 5m gold, ~150d (as much as TV reliably serves), then resample to 1h.
    df5 = load("GOLD", days=150)
    print(f"loaded {len(df5)} 5m bars")
    df1h = resample_ohlcv(df5, "1h")

    study("GOLD", "1h (screenshot setup)", df1h)
    study("GOLD", "5m (scalp-cadence proxy)", df5)


if __name__ == "__main__":
    main()
