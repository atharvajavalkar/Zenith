"""Backtest + walk-forward RL for Hyper LIQUIDATION MAGNET (failed auction).

Replays the live idea on real 1m history (5m value from resampled 1m):
  • 5m value break (z-score + bps)
  • 1m reclaim
  • tape imbalance + acceleration
  • cost gate vs spread proxy
  • TP at POC, SL beyond failed extreme

Train split: grid-search / RL-style param pick by expectancy.
Test split: report OOS only (the number that matters).

Research only — does not auto-integrate.
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

from data.historical import fetch_symbol_ohlcv  # noqa: E402

SPREAD_BPS = {
    "GOLD": 1.8,
    "SILVER": 3.5,
    "EURUSD": 1.0,
    "GBPUSD": 1.2,
    "USDJPY": 1.0,
    "USOIL": 2.5,
    "US30": 1.5,
    "BTCUSD": 2.0,
}

SYMBOLS = ["GOLD", "SILVER", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD"]


def fast_profile(win: pd.DataFrame, n_bins: int = 36, va_pct: float = 0.70) -> Optional[dict]:
    lo = float(win["low"].min())
    hi = float(win["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    tp = (win["high"].to_numpy() + win["low"].to_numpy() + win["close"].to_numpy()) / 3.0
    vol = win["volume"].to_numpy(dtype=float)
    if not np.isfinite(vol).any() or float(vol.sum()) <= 0:
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
    return {"poc": poc, "val": float(edges[left]), "vah": float(edges[right + 1])}


def load_bars(symbol: str, days: int) -> pd.DataFrame:
    """Finest history available (~5m). Strategy TFs scale to this bar size."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    df = fetch_symbol_ohlcv(symbol, start.isoformat(), end.isoformat())
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def to_htf(df: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    """Higher-TF value frame from 5m (15m ≈ live 5m value on a slower clock)."""
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()


def imbalance(close: np.ndarray, open_: np.ndarray, vol: np.ndarray, i: int, lb: int) -> Optional[float]:
    a = max(0, i - lb + 1)
    c = close[a : i + 1]
    o = open_[a : i + 1]
    v = vol[a : i + 1]
    if len(c) < max(4, lb // 2):
        return None
    buy = float(v[c >= o].sum())
    sell = float(v[c < o].sum())
    tot = buy + sell
    if tot <= 0:
        # range proxy
        return None
    return (buy - sell) / tot


@dataclass
class Params:
    value_bars: int = 48          # HTF (15m) value lookback
    break_lookback: int = 6
    reject_bars: int = 2          # entry TF (5m) reclaim window
    imb_lookback: int = 10
    min_imbalance: float = 0.12
    min_break_bps: float = 2.0
    min_break_z: float = 0.85
    cost_mult: float = 2.5
    min_range_bps: float = 1.2
    imb_accel: float = 0.04
    min_rr: float = 1.20
    swing_pad_bps: float = 1.5
    max_hold: int = 18            # 5m bars (~90m)


@dataclass
class Result:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    r_list: list = field(default_factory=list)
    symbols: dict = field(default_factory=dict)

    def add(self, symbol: str, r: float) -> None:
        self.trades += 1
        self.r_list.append(r)
        if r > 0:
            self.wins += 1
        elif r < 0:
            self.losses += 1
        bucket = self.symbols.setdefault(symbol, {"n": 0, "wins": 0, "r": []})
        bucket["n"] += 1
        bucket["r"].append(r)
        if r > 0:
            bucket["wins"] += 1

    def stats(self) -> dict[str, Any]:
        if not self.r_list:
            return {"trades": 0, "win_rate": 0.0, "expectancy_R": 0.0, "total_R": 0.0}
        r = np.array(self.r_list, dtype=float)
        gross_win = float(r[r > 0].sum()) if (r > 0).any() else 0.0
        gross_loss = float(-r[r < 0].sum()) if (r < 0).any() else 0.0
        pf = (gross_win / gross_loss) if gross_loss > 1e-9 else (999.0 if gross_win > 0 else 0.0)
        eq = np.cumsum(r)
        peak = np.maximum.accumulate(eq)
        dd = float((peak - eq).max()) if len(eq) else 0.0
        decisive = self.wins + self.losses
        per = {}
        for sym, b in self.symbols.items():
            rr = np.array(b["r"], dtype=float)
            per[sym] = {
                "n": b["n"],
                "wr": round(100.0 * b["wins"] / max(1, b["n"]), 1),
                "E[R]": round(float(rr.mean()), 3),
                "sum_R": round(float(rr.sum()), 2),
            }
        return {
            "trades": int(self.trades),
            "win_rate": round(100.0 * self.wins / decisive, 1) if decisive else 0.0,
            "expectancy_R": round(float(r.mean()), 4),
            "total_R": round(float(r.sum()), 2),
            "profit_factor": round(float(pf), 2),
            "avg_win_R": round(float(r[r > 0].mean()), 3) if (r > 0).any() else 0.0,
            "avg_loss_R": round(float(r[r < 0].mean()), 3) if (r < 0).any() else 0.0,
            "max_dd_R": round(dd, 2),
            "per_symbol": per,
        }


def _simulate(df: pd.DataFrame, i_entry: int, direction: str, entry: float, sl: float, tp: float, max_hold: int) -> Optional[float]:
    risk = abs(entry - sl)
    if risk <= 1e-12:
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
    last = float(df["close"].iloc[min(i_entry + max_hold, n) - 1])
    return ((last - entry) if direction == "BUY" else (entry - last)) / risk


def run_symbol(symbol: str, df: pd.DataFrame, p: Params, i0: int, i1: int) -> Result:
    """df = 5m entry frame; value built on 15m resample (proxy for live 1m/5m)."""
    res = Result()
    if df is None or len(df) < 400:
        return res
    df_htf = to_htf(df, "15min")
    if len(df_htf) < p.value_bars + 10:
        return res

    # Align entry bars to HTF without tz headaches
    htf_times = df_htf.index
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    idx = df.index

    spread_bps = float(SPREAD_BPS.get(symbol, 2.0))
    in_pos_until = -1
    cooldown = 4  # 5m bars

    start = max(i0, 80)
    end = min(i1, len(df) - 2)
    for i in range(start, end):
        if i <= in_pos_until:
            continue
        px = float(c[i])
        slc = slice(max(0, i - 19), i + 1)
        med_range = float(np.median(h[slc] - l[slc]))
        med_bps = med_range / max(abs(px), 1e-9) * 1e4
        if med_bps < p.min_range_bps:
            continue

        ts = idx[i]
        # last HTF bar with timestamp <= current entry bar
        pos = int(htf_times.searchsorted(ts, side="right") - 1)
        if pos < p.value_bars:
            continue
        win_htf = df_htf.iloc[pos - p.value_bars + 1 : pos + 1]
        prof = fast_profile(win_htf)
        if not prof:
            continue
        vah, val, poc = float(prof["vah"]), float(prof["val"]), float(prof["poc"])
        if vah <= val:
            continue

        lb = p.break_lookback
        w = df_htf.iloc[max(0, pos - lb + 1) : pos + 1]
        hi5 = float(w["high"].max())
        lo5 = float(w["low"].min())
        rng = (df_htf["high"] - df_htf["low"]).iloc[max(0, pos - 29) : pos + 1].astype(float)
        sigma = float(rng.std()) if len(rng) >= 8 else float(rng.mean() or abs(px) * 0.0004)
        sigma = max(sigma, abs(px) * 1e-6)
        up_ext = max(0.0, hi5 - vah)
        dn_ext = max(0.0, val - lo5)
        up_z, dn_z = up_ext / sigma, dn_ext / sigma
        up_bps = up_ext / max(abs(px), 1e-9) * 1e4
        dn_bps = dn_ext / max(abs(px), 1e-9) * 1e4
        up_break = up_z >= p.min_break_z and up_bps >= p.min_break_bps
        dn_break = dn_z >= p.min_break_z and dn_bps >= p.min_break_bps
        if up_break and dn_break:
            if up_z >= dn_z:
                dn_break = False
            else:
                up_break = False
        if not up_break and not dn_break:
            continue

        n_rej = p.reject_bars
        a = max(0, i - n_rej + 1)
        h_win = float(h[a : i + 1].max())
        l_win = float(l[a : i + 1].min())
        c_last = float(c[i])
        o_last = float(o[i])
        signal = None
        failed = None
        if up_break:
            pierced = h_win > vah
            reclaimed = c_last < vah
            bear = c_last <= o_last or (i > 0 and c_last < float(c[i - 1]))
            if pierced and reclaimed and bear:
                signal, failed = "SELL", h_win
        elif dn_break:
            pierced = l_win < val
            reclaimed = c_last > val
            bull = c_last >= o_last or (i > 0 and c_last > float(c[i - 1]))
            if pierced and reclaimed and bull:
                signal, failed = "BUY", l_win
        if signal is None:
            continue

        imb = imbalance(c, o, v, i, p.imb_lookback)
        if imb is None:
            continue
        half = max(3, p.imb_lookback // 2)
        imb_fast = imbalance(c, o, v, i, half)
        imb_prev = imbalance(c, o, v, max(0, i - half), p.imb_lookback) if i > half + p.imb_lookback else None
        if signal == "SELL":
            if imb > -p.min_imbalance:
                continue
            if imb_prev is not None and imb_fast is not None and float(imb_fast) > float(imb_prev) - p.imb_accel:
                continue
        else:
            if imb < p.min_imbalance:
                continue
            if imb_prev is not None and imb_fast is not None and float(imb_fast) < float(imb_prev) + p.imb_accel:
                continue

        pad = abs(px) * p.swing_pad_bps / 1e4
        if signal == "SELL":
            sl = float(failed) + pad
            tp = float(poc)
            risk = sl - px
            reward = px - tp
            if risk <= 0 or tp >= px:
                continue
            if reward / risk < p.min_rr:
                tp = min(tp, px - p.min_rr * risk, val)
                reward = px - tp
                if reward / risk < p.min_rr * 0.85:
                    continue
        else:
            sl = float(failed) - pad
            tp = float(poc)
            risk = px - sl
            reward = tp - px
            if risk <= 0 or tp <= px:
                continue
            if reward / risk < p.min_rr:
                tp = max(tp, px + p.min_rr * risk, vah)
                reward = tp - px
                if reward / risk < p.min_rr * 0.85:
                    continue

        spread = abs(px) * spread_bps / 1e4
        if reward < p.cost_mult * spread:
            continue

        entry = float(o[i + 1])
        r = _simulate(df, i + 1, signal, entry, sl, tp, p.max_hold)
        if r is None:
            continue
        cost_r = spread / max(abs(entry - sl), 1e-12)
        res.add(symbol, float(r - cost_r))
        in_pos_until = i + 1 + cooldown

    return res


def merge(results: list[Result]) -> Result:
    out = Result()
    for r in results:
        for sym, b in r.symbols.items():
            for rr in b["r"]:
                out.add(sym, rr)
    return out


def grid() -> list[Params]:
    combos = []
    for min_break_z, min_imbalance, cost_mult, min_rr in itertools.product(
        [0.7, 0.85, 1.0],
        [0.10, 0.14],
        [2.0, 2.5],
        [1.1, 1.25],
    ):
        combos.append(
            Params(
                min_break_z=min_break_z,
                min_imbalance=min_imbalance,
                cost_mult=cost_mult,
                min_rr=min_rr,
            )
        )
    return combos


def main() -> None:
    days = 45
    print(f"=== FAILED AUCTION backtest · {days}d · 5m entry / 15m value · symbols={SYMBOLS} ===")
    print("(Proxy for live 1m/5m — same rules, coarser clock. Costs deducted.)\n")
    data: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        try:
            df = load_bars(sym, days)
            data[sym] = df
            print(f"loaded {sym}: {len(df)} bars  {df.index[0]} → {df.index[-1]}")
        except Exception as exc:
            print(f"SKIP {sym}: {exc}")

    if not data:
        print("No data — abort")
        return

    clock = data["GOLD"] if "GOLD" in data else next(iter(data.values()))
    print(f"\nTrain ~70%  |  Test ~30%  (per-symbol split)  clock={len(clock)} bars\n")

    best: Optional[Params] = None
    best_score = -999.0
    best_train_stats: dict = {}
    for p in grid():
        parts = []
        for sym, df in data.items():
            if len(df) < 400:
                continue
            s = int(len(df) * 0.70)
            parts.append(run_symbol(sym, df, p, 0, s))
        train = merge(parts)
        st = train.stats()
        if st["trades"] < 12:
            score = -999.0
        else:
            score = float(st["expectancy_R"]) * np.log1p(st["trades"]) - 0.15 * float(st["max_dd_R"])
        if score > best_score:
            best_score = score
            best = p
            best_train_stats = st

    assert best is not None
    print("TRAIN best params (RL/grid):")
    print(
        f"  z={best.min_break_z} imb={best.min_imbalance} cost×={best.cost_mult} "
        f"rr={best.min_rr}"
    )
    print(f"  {best_train_stats}\n")

    parts = []
    for sym, df in data.items():
        s = int(len(df) * 0.70)
        parts.append(run_symbol(sym, df, best, s, len(df)))
    test = merge(parts)
    ts = test.stats()
    print("OUT-OF-SAMPLE (test):")
    print(f"  trades={ts.get('trades')}  WR={ts.get('win_rate')}%  "
          f"E[R]={ts.get('expectancy_R')}  total_R={ts.get('total_R')}  "
          f"PF={ts.get('profit_factor')}  maxDD_R={ts.get('max_dd_R')}")
    print(f"  avg_win={ts.get('avg_win_R')}  avg_loss={ts.get('avg_loss_R')}")
    print("  per symbol:")
    for sym, row in (ts.get("per_symbol") or {}).items():
        print(f"    {sym}: {row}")

    print("\n=== Live Hyper RL state (account — may include older Hyper strategies) ===")
    try:
        from strategy.hyper_rl import snapshot

        snap = snapshot()
        rows = snap.get("symbols") or []
        if not rows:
            print("  no closed Hyper RL samples yet")
        for row in rows:
            print(
                f"  {row.get('symbol')}: n={row.get('n')} WR={round(100*float(row.get('win_rate') or 0),1)}% "
                f"E[R]={row.get('expectancy_r')} streak={row.get('streak')} lot×={row.get('lot_scale')}"
            )
    except Exception as exc:
        print(f"  (unavailable: {exc})")

    print("\n=== VERDICT ===")
    er = float(ts.get("expectancy_R") or 0)
    n = int(ts.get("trades") or 0)
    pf = float(ts.get("profit_factor") or 0)
    if n < 20:
        print(f"INCONCLUSIVE — only {n} OOS trades (need ≥20).")
    elif er > 0.05 and pf >= 1.15:
        print(f"EDGE LIKELY — OOS E[R]={er:.3f}, PF={pf:.2f}, n={n}. Still demo-size only.")
    elif er > 0:
        print(f"WEAK / FRAGILE — OOS E[R]={er:.3f}, PF={pf:.2f}, n={n}. Not scale-ready.")
    else:
        print(f"NO EDGE OOS — E[R]={er:.3f}, PF={pf:.2f}, n={n}. Do not trust live punches yet.")


if __name__ == "__main__":
    main()
