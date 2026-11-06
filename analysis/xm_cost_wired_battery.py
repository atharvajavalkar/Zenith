#!/usr/bin/env python3
"""XM-cost honesty battery for LIVE-wired desk.

Runs current .env brains × live SL floors × measured XM session spreads across:
  · calendar windows: 7d / 30d / 90d / 180d / 1y / 3y
  · walk-forward ×4 on 1y and 3y trade sets
  · variants: mid-only | XM cost | XM + Off sit-out | XM + spread gate

Does NOT change live brains. Writes JSON + progress for monitoring.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import _prep, monte_carlo  # noqa: E402
from analysis.lab_zenith_math_invention_deep_bt import (  # noqa: E402
    _hurst_rs,
    _perm_entropy,
)
from analysis.xm_session_costs import session_name, spread_price  # noqa: E402
from config import symbol_entry_brain, symbol_rr, symbol_scan_tfs  # noqa: E402
from data.hist_local import (  # noqa: E402
    clear_hist_cache,
    has_local_hist,
    load_local_ohlcv,
    local_span,
)
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL, _min_sl_px  # noqa: E402
from strategy.adaptive_v3_runtime import NIGHT_SITOUT  # noqa: E402

OUT = Path(ROOT) / "data" / "xm_cost_wired_battery.json"
LOG = Path(ROOT) / "data" / "xm_cost_wired_battery.log"
PROGRESS = Path(ROOT) / "data" / "xm_cost_wired_battery.progress.json"

# Match live MAX_RISK_USD_PER_TRADE
RISK_USD = 100.0
CAP_YEARS = 3.0

WINDOWS = (
    ("7d", 7),
    ("30d", 30),
    ("90d", 90),
    ("180d", 180),
    ("1y", 365),
    ("3y", 365 * 3),
)

BOOK = [
    "ETHUSD",
    "BTCUSD",
    "GOLD",
    "NAS100",
    "USDJPY",
    "AUDUSD",
    "USDCHF",
    "UKOIL",
]

# Fine-tune variants tested after base
VARIANTS = (
    "mid",  # no spread (old paper)
    "xm",  # XM session spread
    "xm_off_sitout",  # XM + skip Off session for FX/oil
    "xm_spread_gate",  # XM + skip if risk < 3× spread
    "xm_off_gate",  # XM + Off sitout + spread gate
)


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _progress(payload: dict) -> None:
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _enrich_light(df: pd.DataFrame) -> pd.DataFrame:
    """Only features needed by live adaptive v3 brains (+ atr from _prep)."""
    out = df.copy()
    c = out["close"].astype(float)
    atr = out["atr"].replace(0, np.nan)
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rn = ret.to_numpy(dtype=float)
    mu = ret.rolling(40).mean()
    sig = ret.rolling(40).std().replace(0, np.nan)
    out["snr"] = (mu.abs() / sig).fillna(0)
    out["drift_sign"] = np.sign(mu).fillna(0)
    ema = c.ewm(span=20, adjust=False).mean()
    resid = (c - ema) / atr
    out["kalman_z"] = (
        (resid - resid.rolling(60).mean()) / resid.rolling(60).std().replace(0, np.nan)
    ).fillna(0)
    out["hurst"] = _hurst_rs(rn, 128, 24)
    out["perm_ent"] = _perm_entropy(rn, 48, 24, 3)
    return out


def _calendar(df: pd.DataFrame) -> dict[str, np.ndarray]:
    idx = pd.DatetimeIndex(df.index)
    hour = idx.hour.to_numpy(dtype=float)
    minute = idx.minute.to_numpy(dtype=float)
    dow = idx.dayofweek.to_numpy(dtype=float)
    mod = hour * 60.0 + minute
    tod = mod / 1440.0
    ang = 2.0 * math.pi * tod
    # Match live adaptive_v3_brains session ids: 0 asia 1 london 2 ny 3 overlap
    sess = np.zeros(len(df), dtype=int)
    sess[(hour >= 7) & (hour < 12)] = 1
    sess[(hour >= 12) & (hour < 16)] = 3
    sess[(hour >= 16) & (hour < 21)] = 2
    return {
        "hour": hour,
        "minute": minute,
        "dow": dow,
        "tod": tod,
        "sin_tod": np.sin(ang),
        "cos_tod": np.cos(ang),
        "sin_week": np.sin(2.0 * math.pi * (dow + tod) / 7.0),
        "cos_week": np.cos(2.0 * math.pi * (dow + tod) / 7.0),
        "sess": sess,
    }


def _sides_session_mix(df: pd.DataFrame, cal: dict) -> np.ndarray:
    """Live thresholds (stricter than research eq_session_mix)."""
    snr = df["snr"].to_numpy(dtype=float)
    ds = df["drift_sign"].to_numpy(dtype=float)
    H = df["hurst"].to_numpy(dtype=float)
    kz = df["kalman_z"].to_numpy(dtype=float)
    st = ds * (0.55 * snr + 0.45 * np.clip((H - 0.5) * 4.0, -1.0, 1.0))
    sf = -np.tanh(kz / 2.0)
    alpha = np.array([0.35, 0.75, 0.55, 0.45], dtype=float)[cal["sess"]]
    score = alpha * st + (1.0 - alpha) * sf
    thr = np.where(cal["sess"] == 3, 0.42, 0.32)
    out = np.zeros(len(df), dtype=int)
    out[score >= thr] = 1
    out[score <= -thr] = -1
    return out


def _sides_circadian(df: pd.DataFrame, cal: dict) -> np.ndarray:
    zthr = 2.2 + 0.5 * cal["sin_tod"] + 0.25 * (cal["dow"] / 7.0)
    kz = df["kalman_z"].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=int)
    out[kz <= -zthr] = 1
    out[kz >= zthr] = -1
    return out


def _sides_entropy(df: pd.DataFrame, cal: dict) -> np.ndarray:
    pmax = 0.72 + 0.08 * cal["cos_tod"] + 0.03 * cal["sin_week"]
    pe = df["perm_ent"].to_numpy(dtype=float)
    ds = df["drift_sign"].to_numpy(dtype=float)
    snr = df["snr"].to_numpy(dtype=float)
    ok = (pe <= pmax) & (ds != 0) & (snr >= 0.22)
    out = np.zeros(len(df), dtype=int)
    out[ok] = ds[ok].astype(int)
    return out


def _sides_volmom(df: pd.DataFrame) -> np.ndarray:
    ret = df["close"].pct_change(20)
    vol = df["close"].pct_change().rolling(20).std().replace(0, np.nan)
    score = (ret / vol).fillna(0).to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=int)
    for i in range(40, len(df) - 1):
        a, b = score[i - 1], score[i]
        if a <= 0 < b and b > 0.5:
            out[i] = 1
        elif a >= 0 > b and b < -0.5:
            out[i] = -1
    return out


def _apply_night_sitout(sym: str, sides: np.ndarray, cal: dict) -> np.ndarray:
    if sym not in NIGHT_SITOUT:
        return sides
    out = sides.copy()
    night = (cal["hour"] < 5) | (cal["hour"] >= 22)
    out[night] = 0
    return out


def _sides_for_brain(sym: str, brain: str, df: pd.DataFrame, cal: dict) -> np.ndarray:
    b = str(brain or "").lower().strip()
    if b in ("session_mix", "adaptive_session_mix"):
        sides = _sides_session_mix(df, cal)
    elif b in ("circadian_kalman_z", "circadian_kalman", "adaptive_circadian_kalman_z"):
        sides = _sides_circadian(df, cal)
    elif b in ("entropy_tod", "adaptive_entropy_tod"):
        sides = _sides_entropy(df, cal)
    elif b in ("volmom", "vol_scaled_mom", "vol_mom"):
        sides = _sides_volmom(df)
    else:
        raise ValueError(f"unsupported brain {brain} for {sym}")
    return _apply_night_sitout(sym, sides, cal)


def _entries_live_sl(
    df: pd.DataFrame,
    sides: np.ndarray,
    sym: str,
    *,
    min_gap: int = 3,
) -> list[dict]:
    atr = df["atr"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    sl_atr = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))
    floor = float(_min_sl_px(sym))
    n = len(df)
    out: list[dict] = []
    busy = -1
    for i in range(40, n - 2):
        if i < busy:
            continue
        s = int(sides[i])
        if s == 0:
            continue
        a = float(atr[i] or 0)
        if a <= 0 or not np.isfinite(a):
            continue
        entry = float(c[i])
        risk = max(sl_atr * a, floor)
        if risk <= 0:
            continue
        if s > 0:
            sl = entry - risk
            side = "BUY"
        else:
            sl = entry + risk
            side = "SELL"
        out.append(
            {
                "ei": i,
                "side": side,
                "entry": entry,
                "sl": sl,
                "risk": risk,
            }
        )
        busy = i + min_gap
    return out


def _simulate(
    df: pd.DataFrame,
    entries: list[dict],
    rr: float,
    sym: str,
    *,
    variant: str,
) -> list[dict]:
    """Barrier sim with optional XM session spread cost + gates."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    hours = pd.DatetimeIndex(df.index).hour.to_numpy(dtype=float)
    n = len(df)
    use_xm = variant.startswith("xm")
    off_sit = variant in ("xm_off_sitout", "xm_off_gate")
    spread_gate = variant in ("xm_spread_gate", "xm_off_gate")
    trades: list[dict] = []
    busy = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < busy or ei >= n - 1:
            continue
        hour = float(hours[ei])
        sess = session_name(hour)
        if off_sit and sess == "Off" and sym in (
            "AUDUSD",
            "USDCHF",
            "USDJPY",
            "UKOIL",
            "GOLD",
        ):
            continue
        side = e["side"]
        entry = float(e["entry"])
        risk = float(e["risk"])
        if risk <= 0:
            continue
        sp = spread_price(sym, hour_utc=hour) if use_xm else 0.0
        if spread_gate and sp > 0 and risk < 3.0 * sp:
            continue
        spread_r = (sp / risk) if use_xm else 0.0
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk
        exit_r = None
        exit_i = ei
        if side == "BUY":
            for j in range(ei + 1, n):
                if low[j] <= sl or high[j] >= tp:
                    exit_r = -1.0 if low[j] <= sl else float(rr)
                    exit_i = j
                    break
        else:
            for j in range(ei + 1, n):
                if high[j] >= sl or low[j] <= tp:
                    exit_r = -1.0 if high[j] >= sl else float(rr)
                    exit_i = j
                    break
        if exit_r is None:
            continue
        r_net = float(exit_r) - float(spread_r)
        trades.append(
            {
                "i": ei,
                "exit_i": exit_i,
                "side": side,
                "session": sess,
                "hour": hour,
                "r_gross": round(float(exit_r), 4),
                "spread_r": round(float(spread_r), 4),
                "r": round(r_net, 4),
                "pnl": round(r_net * RISK_USD, 2),
                "risk_px": round(risk, 8),
                "spread_px": round(sp, 8),
            }
        )
        busy = exit_i + 2
    return trades


def _window_trades(df: pd.DataFrame, trades: list[dict], days: int) -> list[dict]:
    end = df.index[-1]
    start = end - pd.Timedelta(days=days)
    i0 = int(np.searchsorted(np.asarray(df.index), start, side="left"))
    return [t for t in trades if int(t["i"]) >= i0]


def _wf_folds(trades: list[dict], i0: int, i1: int) -> list[dict]:
    """Anchored WF folds inside [i0, i1) using absolute trade indices."""
    n = max(i1 - i0, 1)
    folds = []
    for k in range(4):
        a = i0 + int(n * (0.40 + 0.15 * k))
        b = i0 + int(n * (0.55 + 0.15 * k)) - 1
        if b <= a:
            continue
        sub = [t for t in trades if a <= int(t["i"]) <= b]
        folds.append({"fold": k + 1, "a": a, "b": b, **_summarize(sub)})
    return folds


def _summarize(trades: list[dict]) -> dict:
    if not trades:
        return {
            "n": 0,
            "exp_r": 0.0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "avg_spread_r": 0.0,
            "by_session": {},
        }
    n = len(trades)
    rs = [float(t["r"]) for t in trades]
    pnl = sum(float(t["pnl"]) for t in trades)
    exp_r = sum(rs) / n
    wr = sum(1 for r in rs if r > 0) / n
    avg_sp = sum(float(t.get("spread_r") or 0) for t in trades) / n
    by_sess: dict[str, dict] = {}
    for sess in ("Asia", "London", "Overlap", "NY", "Off"):
        sub = [t for t in trades if t.get("session") == sess]
        if not sub:
            continue
        sr = [float(t["r"]) for t in sub]
        by_sess[sess] = {
            "n": len(sub),
            "exp_r": round(sum(sr) / len(sr), 4),
            "pnl": round(sum(float(t["pnl"]) for t in sub), 2),
            "avg_spread_r": round(
                sum(float(t.get("spread_r") or 0) for t in sub) / len(sub), 4
            ),
        }
    return {
        "n": n,
        "exp_r": round(exp_r, 4),
        "win_rate": round(wr, 4),
        "pnl": round(pnl, 2),
        "avg_spread_r": round(avg_sp, 4),
        "by_session": by_sess,
    }


def _battery(trades: list[dict], i0: int, i1: int) -> dict:
    """OOS70 + WF4 + MC on a trade list (already XM-costed)."""
    if not trades:
        return {"verdict": "NO_TRADES", "wf_green": 0, "oos": {}, "wf": [], "mc": {}}
    n = max(i1 - i0, 1)
    split = i0 + int(n * 0.70)
    oos = _summarize([t for t in trades if int(t["i"]) >= split])
    wf = _wf_folds(trades, i0, i1)
    wf_green = sum(
        1 for f in wf if float(f.get("exp_r") or 0) > 0 and int(f.get("n") or 0) >= 5
    )
    full = _summarize(trades)
    mc = {}
    try:
        mc_raw = monte_carlo(
            [{"r": float(t["r"]), "pnl": float(t["pnl"])} for t in trades]
        )
        mc = {
            "prob_profit": (mc_raw or {}).get("prob_profit"),
            "p05_pnl": (mc_raw or {}).get("p05_pnl"),
            "p50_pnl": (mc_raw or {}).get("p50_pnl"),
        }
    except Exception:
        mc = {}
    er = float(full.get("exp_r") or 0)
    oos_er = float(oos.get("exp_r") or 0)
    ntr = int(full.get("n") or 0)
    ok = er > 0 and oos_er > 0 and wf_green >= 3 and ntr >= 20
    soft = er > 0 and wf_green >= 2 and ntr >= 12
    verd = "PASS" if ok else ("SOFT" if soft else "FAIL")
    return {
        "verdict": verd,
        "wf_green": wf_green,
        "full": full,
        "oos": oos,
        "wf": wf,
        "mc": mc,
    }


def _load(sym: str, tf: str) -> pd.DataFrame:
    clear_hist_cache()
    if not has_local_hist(sym, tf):
        raise RuntimeError(f"no hist {sym} {tf}")
    sp = local_span(sym, tf)
    assert sp
    end = sp[1]
    start = max(sp[0], end - pd.Timedelta(days=int(CAP_YEARS * 365.25) + 40))
    raw = load_local_ohlcv(sym, tf=tf, start=start.isoformat(), end=end.isoformat())
    if raw is None or len(raw) < 800:
        raise RuntimeError(f"short {sym} {tf}")
    base = _prep(raw[["open", "high", "low", "close"]].dropna())
    return _enrich_light(base)


def main() -> int:
    t0 = time.time()
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    _log(
        f"=== XM-cost wired battery · risk=${RISK_USD}/R · cap={CAP_YEARS}y · "
        f"variants={list(VARIANTS)} ==="
    )
    _progress({"status": "running", "pct": 0, "pair": None})

    desk: dict[str, dict[str, float]] = {
        v: {w[0]: 0.0 for w in WINDOWS} for v in VARIANTS
    }
    per_pair: dict[str, Any] = {}
    n_pairs = len(BOOK)

    for pi, sym in enumerate(BOOK):
        brain = symbol_entry_brain(sym)
        rr = float(symbol_rr(sym) or 2)
        tfs = symbol_scan_tfs(sym) or ["5m"]
        tf = tfs[0] if tfs else "5m"
        _log(f"\n[{pi+1}/{n_pairs}] {sym} brain={brain} tf={tf} RR={rr}")
        _progress(
            {
                "status": "running",
                "pct": round(100 * pi / n_pairs, 1),
                "pair": sym,
                "brain": brain,
                "elapsed_sec": round(time.time() - t0, 1),
            }
        )
        try:
            df = _load(sym, tf)
        except Exception as exc:
            _log(f"  LOAD FAIL: {exc}")
            per_pair[sym] = {"error": str(exc)}
            continue
        cal = _calendar(df)
        try:
            sides = _sides_for_brain(sym, brain, df, cal)
        except Exception as exc:
            _log(f"  SIDES FAIL: {exc}")
            per_pair[sym] = {"error": str(exc)}
            continue
        entries = _entries_live_sl(df, sides, sym)
        _log(
            f"  bars={len(df)} sides_nonzero={int((sides != 0).sum())} "
            f"entries={len(entries)} span={df.index[0]}→{df.index[-1]}"
        )

        pair_out: dict[str, Any] = {
            "symbol": sym,
            "brain": brain,
            "tf": tf,
            "rr": rr,
            "sl_atr": float(SL_ATR_BY_SYMBOL.get(sym, 2.0)),
            "sl_floor": float(_min_sl_px(sym)),
            "n_bars": len(df),
            "n_entries": len(entries),
            "variants": {},
        }

        for variant in VARIANTS:
            trades = _simulate(df, entries, rr, sym, variant=variant)
            wins = {}
            for wname, days in WINDOWS:
                tw = _window_trades(df, trades, days)
                sm = _summarize(tw)
                wins[wname] = sm
                desk[variant][wname] = round(
                    desk[variant][wname] + float(sm["pnl"]), 2
                )
            # batteries on 1y and 3y trade slices
            t_1y = _window_trades(df, trades, 365)
            t_3y = _window_trades(df, trades, 365 * 3)
            end = df.index[-1]
            i0_1y = int(
                np.searchsorted(
                    np.asarray(df.index), end - pd.Timedelta(days=365), side="left"
                )
            )
            i0_3y = int(
                np.searchsorted(
                    np.asarray(df.index),
                    end - pd.Timedelta(days=365 * 3),
                    side="left",
                )
            )
            i1 = len(df)
            bat_1y = _battery(t_1y, i0_1y, i1)
            bat_3y = _battery(t_3y, i0_3y, i1)
            pair_out["variants"][variant] = {
                "windows": wins,
                "battery_1y": bat_1y,
                "battery_3y": bat_3y,
            }
            _log(
                f"  {variant:14s} 1y ${wins['1y']['pnl']:>10,.0f} "
                f"E[R]={wins['1y']['exp_r']:+.3f} n={wins['1y']['n']} "
                f"avgSpR={wins['1y']['avg_spread_r']:.3f} "
                f"bat={bat_1y['verdict']} wf={bat_1y['wf_green']}/4"
            )

        # pick best surviving fine-tune on 1y under XM family
        xm_family = [v for v in VARIANTS if v.startswith("xm")]
        ranked = sorted(
            xm_family,
            key=lambda v: (
                1 if pair_out["variants"][v]["battery_1y"]["verdict"] == "PASS" else 0,
                1 if pair_out["variants"][v]["battery_1y"]["verdict"] == "SOFT" else 0,
                float(pair_out["variants"][v]["windows"]["1y"]["pnl"]),
                float(pair_out["variants"][v]["windows"]["1y"]["exp_r"]),
            ),
            reverse=True,
        )
        pair_out["best_xm_variant"] = ranked[0]
        pair_out["best_1y"] = pair_out["variants"][ranked[0]]["windows"]["1y"]
        pair_out["mid_vs_xm_1y"] = {
            "mid_pnl": pair_out["variants"]["mid"]["windows"]["1y"]["pnl"],
            "xm_pnl": pair_out["variants"]["xm"]["windows"]["1y"]["pnl"],
            "gap": round(
                float(pair_out["variants"]["mid"]["windows"]["1y"]["pnl"])
                - float(pair_out["variants"]["xm"]["windows"]["1y"]["pnl"]),
                2,
            ),
        }
        per_pair[sym] = pair_out

    # Desk recommendation: per-pair best XM variant
    desk_best = {w[0]: 0.0 for w in WINDOWS}
    recs = []
    for sym in BOOK:
        row = per_pair.get(sym) or {}
        if row.get("error"):
            continue
        bv = row["best_xm_variant"]
        for wname, _ in WINDOWS:
            desk_best[wname] = round(
                desk_best[wname]
                + float(row["variants"][bv]["windows"][wname]["pnl"]),
                2,
            )
        recs.append(
            {
                "symbol": sym,
                "brain": row["brain"],
                "variant": bv,
                "verdict_1y": row["variants"][bv]["battery_1y"]["verdict"],
                "wf_green_1y": row["variants"][bv]["battery_1y"]["wf_green"],
                "pnl_1y": row["variants"][bv]["windows"]["1y"]["pnl"],
                "exp_r_1y": row["variants"][bv]["windows"]["1y"]["exp_r"],
                "mid_gap_1y": row["mid_vs_xm_1y"]["gap"],
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "risk_usd": RISK_USD,
        "cap_years": CAP_YEARS,
        "spreads": "analysis/xm_session_costs.py (MetaAPI Jun–Aug 2026)",
        "sl": "live SL_ATR_BY_SYMBOL + pip floors",
        "brains": "config.symbol_entry_brain (.env)",
        "variants": list(VARIANTS),
        "windows": [w[0] for w in WINDOWS],
        "elapsed_sec": round(time.time() - t0, 1),
        "desk": desk,
        "desk_best_xm": desk_best,
        "recommendations": recs,
        "per_pair": per_pair,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress(
        {
            "status": "done",
            "pct": 100,
            "elapsed_sec": round(time.time() - t0, 1),
            "desk_xm_1y": desk["xm"]["1y"],
            "desk_best_1y": desk_best["1y"],
            "out": str(OUT),
        }
    )
    _log("\n=== DESK PnL by variant ($100/R) ===")
    for v in VARIANTS:
        parts = " · ".join(f"{w} ${desk[v][w]:,.0f}" for w, _ in WINDOWS)
        _log(f"  {v:14s} {parts}")
    _log(
        f"\nBEST XM mix 1y ${desk_best['1y']:,.0f} · 3y ${desk_best['3y']:,.0f} "
        f"· wrote {OUT}"
    )
    for r in recs:
        _log(
            f"  REC {r['symbol']:<8} {r['variant']:<14} "
            f"{r['verdict_1y']:<5} wf={r['wf_green_1y']}/4 "
            f"1y ${r['pnl_1y']:,.0f} E[R]={r['exp_r_1y']:+.3f} "
            f"mid−xm gap ${r['mid_gap_1y']:,.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
