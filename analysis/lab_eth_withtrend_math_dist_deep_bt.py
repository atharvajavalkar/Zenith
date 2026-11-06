#!/usr/bin/env python3
"""RESEARCH ONLY — ETHUSD with-trend math/probability deep hunt (≥1h grid).

Design (retail with-trend — NOT institutional fade):
  • Bias ONLY: 15m + 1h EMA8/14 must agree
  • Entry: 1m / 3m / 5m
  • Features: Fourier spectrum, return z/kurtosis, efficiency, trend strength
  • Prefer TRENDING regimes (low HF power) + pullbacks with the bias
  • Full 8-test battery on 1m / 3m / 6m / 1y windows

Firm rule to recommend live: SURVIVES on BOTH 6m and 1y, OOS E[R]>0,
WF green ≥2, MC prob_profit ≥55%, min trades on 1y ≥80.

Does NOT auto-wire live.
"""
from __future__ import annotations

import json
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

from analysis._edge_survive_bt import RISK_USD, _prep  # noqa: E402
from analysis.lab_agentic_1y_battery import _battery_from_trades  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import _simulate_fast  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_eth_withtrend_math_dist_deep_bt.json"
LOG = Path(ROOT) / "data" / "lab_eth_withtrend_math_dist_deep_bt.log"
PROGRESS = Path(ROOT) / "data" / "lab_eth_withtrend_math_dist_deep_bt.progress.json"

SYMBOL = "ETHUSD"
ENTRY_TFS = ("1m", "3m", "5m", "15m")
WINDOWS = (("1m", 30), ("3m", 91), ("6m", 183), ("1y", 365))
# Firm evaluation windows
FIRM_WINDOWS = ("6m", "1y")
# User asked ≥1h deep battery — densify around leaders until this floor
MIN_RUNTIME_SEC = 3600


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _progress(payload: dict) -> None:
    try:
        PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_tf(sym: str, tf: str, days: int = 400) -> pd.DataFrame:
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    clear_hist_cache()
    if not has_local_hist(sym, tf):
        raise RuntimeError(f"no hist {sym} {tf}")
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    raw = load_local_ohlcv(
        sym,
        tf=tf,
        start=(end - pd.Timedelta(days=days + 40)).isoformat(),
        end=end.isoformat(),
    )
    if raw is None or len(raw) < 500:
        raise RuntimeError(f"{sym} {tf} too short")
    df = _prep(raw[["open", "high", "low", "close"]].dropna())
    data_end = df.index[-1]
    return df[df.index >= data_end - pd.Timedelta(days=days)].copy()


def _rolling_fourier_hf(ret: np.ndarray, win: int = 64, stride: int = 8) -> np.ndarray:
    n = len(ret)
    out = np.full(n, np.nan, dtype=float)
    for i in range(win, n, stride):
        seg = ret[i - win : i]
        if not np.isfinite(seg).all():
            continue
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg)) ** 2
        tot = float(spec.sum())
        if tot <= 0:
            continue
        cut = max(2, len(spec) // 2)
        out[i] = float(spec[cut:].sum() / tot)
    return pd.Series(out).ffill().fillna(0.5).to_numpy(dtype=float)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    o = out["open"].astype(float)
    rng = (h - l).replace(0, np.nan)
    out["eff"] = ((c - o).abs() / rng).clip(0, 1).fillna(0)
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ret"] = ret
    out["ret_z"] = (
        (ret - ret.rolling(60).mean()) / ret.rolling(60).std().replace(0, np.nan)
    ).fillna(0.0)
    out["kurt60"] = ret.rolling(60).kurt().fillna(0.0)
    out["vol60"] = ret.rolling(60).std().fillna(0.0)
    out["fourier_hf"] = _rolling_fourier_hf(ret.to_numpy(), 64, 8)
    # low HF ⇒ more persistent / trend-like spectrum
    out["fourier_lf"] = 1.0 - out["fourier_hf"]
    ema8 = c.ewm(span=8, adjust=False).mean()
    ema14 = c.ewm(span=14, adjust=False).mean()
    atr = out["atr"].replace(0, np.nan)
    out["ema8"] = ema8
    out["ema14"] = ema14
    out["trend_str"] = ((ema8 - ema14).abs() / atr).fillna(0.0)
    out["ema_slope"] = (ema8 - ema8.shift(5)) / atr.replace(0, np.nan)
    out["ema_slope"] = out["ema_slope"].fillna(0.0)
    # soft skew of returns (distribution lean)
    out["skew60"] = ret.rolling(60).skew().fillna(0.0)
    return out


def _htf_bias_series(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    e8 = c.ewm(span=8, adjust=False).mean()
    e14 = c.ewm(span=14, adjust=False).mean()
    s = pd.Series(0, index=df.index, dtype=int)
    return s.mask(e8 > e14, 1).mask(e8 < e14, -1)


def _build_sync(exec_df: pd.DataFrame, df_15: pd.DataFrame, df_1h: pd.DataFrame) -> np.ndarray:
    a15 = _htf_bias_series(df_15).shift(1).reindex(exec_df.index, method="ffill").fillna(0)
    a1h = _htf_bias_series(df_1h).shift(1).reindex(exec_df.index, method="ffill").fillna(0)
    a15 = a15.astype(int).to_numpy()
    a1h = a1h.astype(int).to_numpy()
    out = np.zeros(len(exec_df), dtype=int)
    m = (a15 == a1h) & (a15 != 0)
    out[m] = a15[m]
    return out


def _gate_ok(
    i: int,
    *,
    side: str,
    fh: np.ndarray,
    rz: np.ndarray,
    kurt: np.ndarray,
    eff: np.ndarray,
    tstr: np.ndarray,
    slope: np.ndarray,
    skew: np.ndarray,
    fourier_hf_max: Optional[float],
    fourier_lf_min: Optional[float],
    abs_z_pull_min: Optional[float],
    abs_z_pull_max: Optional[float],
    kurt_max: Optional[float],
    min_eff: Optional[float],
    min_trend_str: Optional[float],
    require_slope_align: bool,
    skew_align: bool,
) -> bool:
    if fourier_hf_max is not None and fh[i] > fourier_hf_max:
        return False
    if fourier_lf_min is not None and (1.0 - fh[i]) < fourier_lf_min:
        return False
    z = float(rz[i])
    az = abs(z)
    # pullback z: BUY wants mild negative z (dip), SELL mild positive z (rip)
    if abs_z_pull_min is not None or abs_z_pull_max is not None:
        if side == "BUY":
            if z >= 0:
                return False
            depth = -z
        else:
            if z <= 0:
                return False
            depth = z
        if abs_z_pull_min is not None and depth < abs_z_pull_min:
            return False
        if abs_z_pull_max is not None and depth > abs_z_pull_max:
            return False
    if kurt_max is not None and abs(float(kurt[i])) > kurt_max:
        return False
    if min_eff is not None and float(eff[i]) < min_eff:
        return False
    if min_trend_str is not None and float(tstr[i]) < min_trend_str:
        return False
    if require_slope_align:
        slp = float(slope[i])
        if side == "BUY" and slp <= 0:
            return False
        if side == "SELL" and slp >= 0:
            return False
    if skew_align:
        sk = float(skew[i])
        # distribution lean with trade
        if side == "BUY" and sk < -0.5:
            return False
        if side == "SELL" and sk > 0.5:
            return False
    return True


def _collect_entries(
    df: pd.DataFrame,
    sync: np.ndarray,
    spec: dict[str, Any],
) -> list[dict]:
    fam = spec["family"]
    atr = df["atr"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    fh = df["fourier_hf"].to_numpy(dtype=float)
    rz = df["ret_z"].to_numpy(dtype=float)
    kurt = df["kurt60"].to_numpy(dtype=float)
    eff = df["eff"].to_numpy(dtype=float)
    tstr = df["trend_str"].to_numpy(dtype=float)
    slope = df["ema_slope"].to_numpy(dtype=float)
    skew = df["skew60"].to_numpy(dtype=float)
    ema8 = df["ema8"].to_numpy(dtype=float)
    ema14 = df["ema14"].to_numpy(dtype=float)

    bb_mult = float(spec.get("bb_mult") or 1.5)
    sl_atr = float(spec["sl_atr"])
    min_risk = float(spec.get("min_risk_px") or 0)
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    lo = (mid - bb_mult * std).to_numpy(dtype=float)
    hi = (mid + bb_mult * std).to_numpy(dtype=float)

    gate_kw = dict(
        fh=fh,
        rz=rz,
        kurt=kurt,
        eff=eff,
        tstr=tstr,
        slope=slope,
        skew=skew,
        fourier_hf_max=spec.get("fourier_hf_max"),
        fourier_lf_min=spec.get("fourier_lf_min"),
        abs_z_pull_min=spec.get("abs_z_pull_min"),
        abs_z_pull_max=spec.get("abs_z_pull_max"),
        kurt_max=spec.get("kurt_max"),
        min_eff=spec.get("min_eff"),
        min_trend_str=spec.get("min_trend_str"),
        require_slope_align=bool(spec.get("require_slope_align")),
        skew_align=bool(spec.get("skew_align")),
    )

    entries: list[dict] = []
    for i in range(80, len(df) - 1):
        bias = int(sync[i])
        if bias == 0:
            continue
        a = float(atr[i] or 0) or 1e-6
        c, o = float(close[i]), float(open_[i])
        side = None
        sl = None

        if fam == "bb_pull":
            if bias == 1 and np.isfinite(lo[i]) and c < lo[i] and c > o:
                side, sl = "BUY", c - sl_atr * a
            elif bias == -1 and np.isfinite(hi[i]) and c > hi[i] and c < o:
                side, sl = "SELL", c + sl_atr * a
        elif fam == "ema_pull":
            if bias == 1 and ema8[i] > ema14[i]:
                if low[i] <= ema8[i] + 0.08 * a and c > ema8[i] and c > o:
                    side, sl = "BUY", c - sl_atr * a
            elif bias == -1 and ema8[i] < ema14[i]:
                if high[i] >= ema8[i] - 0.08 * a and c < ema8[i] and c < o:
                    side, sl = "SELL", c + sl_atr * a
        elif fam == "ema_cont":
            # continuation: close beyond prior 5-bar extreme with ribbon
            if bias == 1 and ema8[i] > ema14[i] and c > float(np.max(high[i - 5 : i])) and c > o:
                side, sl = "BUY", c - sl_atr * a
            elif bias == -1 and ema8[i] < ema14[i] and c < float(np.min(low[i - 5 : i])) and c < o:
                side, sl = "SELL", c + sl_atr * a
        elif fam == "don_break":
            # Donchian break with HTF bias (trend run)
            look = int(spec.get("don_n") or 20)
            if i < look + 1:
                continue
            if bias == 1 and c > float(np.max(high[i - look : i])) and c > o:
                side, sl = "BUY", c - sl_atr * a
            elif bias == -1 and c < float(np.min(low[i - look : i])) and c < o:
                side, sl = "SELL", c + sl_atr * a
        else:
            continue

        if side is None or sl is None:
            continue
        risk = abs(c - sl)
        if risk < min_risk:
            continue
        if not _gate_ok(i, side=side, **gate_kw):
            continue
        entries.append(
            {
                "ei": i,
                "side": side,
                "entry": c,
                "sl": float(sl),
                "entry_t": df.index[i],
                "risk_px": risk,
            }
        )
    return entries


def _entry_key(spec: dict) -> str:
    """Everything that affects entries (not RR)."""
    keys = (
        "family",
        "bb_mult",
        "don_n",
        "sl_atr",
        "min_risk_px",
        "fourier_hf_max",
        "fourier_lf_min",
        "abs_z_pull_min",
        "abs_z_pull_max",
        "kurt_max",
        "min_eff",
        "min_trend_str",
        "require_slope_align",
        "skew_align",
    )
    return json.dumps({k: spec.get(k) for k in keys}, sort_keys=True)


def _mk_spec(name: str, family: str, *, note: str = "", **kw: Any) -> dict[str, Any]:
    s: dict[str, Any] = {
        "name": name[:95],
        "family": family,
        "bb_mult": kw.get("bb_mult", 1.5),
        "don_n": kw.get("don_n", 20),
        "sl_atr": kw.get("sl_atr", 2.0),
        "rr": kw.get("rr", 2.4),
        "min_risk_px": kw.get("min_risk_px", 6.0),
        "fourier_hf_max": kw.get("fourier_hf_max"),
        "fourier_lf_min": kw.get("fourier_lf_min"),
        "abs_z_pull_min": kw.get("abs_z_pull_min"),
        "abs_z_pull_max": kw.get("abs_z_pull_max"),
        "kurt_max": kw.get("kurt_max"),
        "min_eff": kw.get("min_eff"),
        "min_trend_str": kw.get("min_trend_str"),
        "require_slope_align": bool(kw.get("require_slope_align", False)),
        "skew_align": bool(kw.get("skew_align", False)),
        "note": note or "with-trend math/dist ablation",
    }
    return s


def _candidate_specs() -> list[dict[str, Any]]:
    """Large ablation grid: single/dual gates (not full stacks that kill n).

    Prior pass: stacked Fourier+z+kurt+eff+tstr → almost all n=0.
    This pass: BASE + 1-feature + 2-feature combos × families × SL/RR.
    """
    specs: list[dict[str, Any]] = [
        _mk_spec("BASE_bb_wt_no_math", "bb_pull", bb_mult=1.5, sl_atr=2.0, rr=4.0, note="BB with-trend bare"),
        _mk_spec("BASE_bb25_wt_no_math", "bb_pull", bb_mult=2.5, sl_atr=2.0, rr=4.0, note="BB2.5 with-trend bare"),
        _mk_spec("BASE_ema_pull_wt", "ema_pull", sl_atr=1.5, rr=2.0, note="EMA pull bare"),
        _mk_spec("BASE_ema_cont_wt", "ema_cont", sl_atr=2.0, rr=2.4, note="EMA cont bare"),
        _mk_spec("BASE_don20_wt", "don_break", don_n=20, sl_atr=2.0, rr=2.4, note="Donchian20 bare"),
        _mk_spec("BASE_don40_wt", "don_break", don_n=40, sl_atr=2.5, rr=3.0, note="Donchian40 bare"),
    ]

    families = (
        ("bb_pull", {"bb_mult": 1.5}),
        ("bb_pull", {"bb_mult": 2.0}),
        ("bb_pull", {"bb_mult": 2.5}),
        ("ema_pull", {}),
        ("ema_cont", {}),
        ("don_break", {"don_n": 20}),
        ("don_break", {"don_n": 40}),
    )
    sls = (1.5, 2.0, 2.5, 3.0)
    rrs = (2.0, 2.4, 3.0, 4.0)
    risks = (4.0, 6.0)

    # Math ablations — None means gate OFF (critical: avoid stacked n=0)
    gate_sets: list[dict[str, Any]] = [
        {},  # bare (also covered by BASE)
        {"fourier_hf_max": 0.55},
        {"fourier_hf_max": 0.50},
        {"fourier_hf_max": 0.45},
        {"fourier_hf_max": 0.40},
        {"fourier_lf_min": 0.50},
        {"fourier_lf_min": 0.55},
        {"fourier_lf_min": 0.60},
        {"abs_z_pull_min": 0.3, "abs_z_pull_max": 2.5},
        {"abs_z_pull_min": 0.5, "abs_z_pull_max": 2.0},
        {"abs_z_pull_min": 0.7, "abs_z_pull_max": 2.5},
        {"abs_z_pull_min": 1.0, "abs_z_pull_max": 3.0},
        {"kurt_max": 8.0},
        {"kurt_max": 6.0},
        {"kurt_max": 4.0},
        {"min_eff": 0.20},
        {"min_eff": 0.30},
        {"min_eff": 0.40},
        {"min_trend_str": 0.10},
        {"min_trend_str": 0.20},
        {"min_trend_str": 0.35},
        {"require_slope_align": True},
        {"skew_align": True},
        # dual soft
        {"fourier_hf_max": 0.50, "min_trend_str": 0.15},
        {"fourier_hf_max": 0.45, "min_eff": 0.25},
        {"fourier_hf_max": 0.45, "require_slope_align": True},
        {"fourier_lf_min": 0.55, "min_trend_str": 0.20},
        {"fourier_lf_min": 0.55, "min_eff": 0.30},
        {"abs_z_pull_min": 0.5, "abs_z_pull_max": 2.5, "fourier_hf_max": 0.50},
        {"abs_z_pull_min": 0.5, "abs_z_pull_max": 2.5, "min_trend_str": 0.15},
        {"kurt_max": 6.0, "fourier_hf_max": 0.50},
        {"kurt_max": 6.0, "min_eff": 0.25},
        {"min_eff": 0.30, "min_trend_str": 0.20, "require_slope_align": True},
        {"fourier_hf_max": 0.45, "kurt_max": 6.0, "skew_align": True},
        # triple soft (still sparse)
        {"fourier_hf_max": 0.50, "abs_z_pull_min": 0.4, "abs_z_pull_max": 2.5, "min_trend_str": 0.12},
        {"fourier_lf_min": 0.52, "min_eff": 0.25, "require_slope_align": True},
        {"fourier_hf_max": 0.48, "kurt_max": 7.0, "min_eff": 0.22, "skew_align": True},
    ]

    # Pull families use z-pull; continuation/donchian skip z-pull in most ablations
    pull_fams = {"bb_pull", "ema_pull"}

    for fam, fkw in families:
        for gates in gate_sets:
            g = dict(gates)
            # Don't force pullback-z onto breakout/continuation (kills n)
            if fam not in pull_fams:
                g.pop("abs_z_pull_min", None)
                g.pop("abs_z_pull_max", None)
            for sl in sls:
                for rr in rrs:
                    for mr in risks:
                        # subsample: keep RR×SL density but skip some risk×gate combos
                        if mr == 4.0 and rr not in (2.0, 2.4, 4.0):
                            continue
                        if sl == 3.0 and rr == 2.0 and "fourier" not in str(g) and "min_trend" not in str(g):
                            # keep some bare high-SL
                            if g and list(g.keys()) not in (["kurt_max"], ["min_eff"], ["require_slope_align"]):
                                pass
                        tag = "_".join(f"{k}{g[k]}" for k in sorted(g)) if g else "bare"
                        name = f"{fam}_sl{sl:g}_rr{rr:g}_r{mr:g}_{tag}"
                        if fkw.get("bb_mult"):
                            name = f"bb{fkw['bb_mult']:g}_" + name
                        if fkw.get("don_n"):
                            name = f"don{fkw['don_n']}_" + name
                        specs.append(
                            _mk_spec(
                                name,
                                fam,
                                note="ablation with-trend math/dist",
                                bb_mult=fkw.get("bb_mult", 1.5),
                                don_n=fkw.get("don_n", 20),
                                sl_atr=sl,
                                rr=rr,
                                min_risk_px=mr,
                                **g,
                            )
                        )

    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for s in specs:
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        uniq.append(s)
    return uniq


def _densify_specs(seed: list[dict[str, Any]], round_i: int) -> list[dict[str, Any]]:
    """Neighborhood expand around leaders until runtime floor."""
    out: list[dict[str, Any]] = []
    hf_extra = (0.42, 0.47, 0.52, 0.58)
    tstr_extra = (0.08, 0.18, 0.28, 0.40)
    eff_extra = (0.18, 0.28, 0.38)
    z_bands = ((0.2, 2.0), (0.4, 2.2), (0.6, 2.8), (0.8, 3.2))
    for base in seed[:40]:
        fam = base["family"]
        for hf in hf_extra:
            for t in tstr_extra:
                for e in eff_extra:
                    for rr in (2.0, 2.4, 3.0, 4.0):
                        for sl in (1.5, 2.0, 2.5):
                            g: dict[str, Any] = {
                                "fourier_hf_max": hf,
                                "min_trend_str": t,
                                "min_eff": e,
                                "require_slope_align": True,
                                "sl_atr": sl,
                                "rr": rr,
                                "min_risk_px": float(base.get("min_risk_px") or 6.0),
                                "bb_mult": float(base.get("bb_mult") or 1.5),
                                "don_n": int(base.get("don_n") or 20),
                            }
                            if fam in ("bb_pull", "ema_pull"):
                                z0, z1 = z_bands[(round_i + int(hf * 10)) % len(z_bands)]
                                g["abs_z_pull_min"] = z0
                                g["abs_z_pull_max"] = z1
                            name = (
                                f"D{round_i}_{fam}_hf{hf:g}_t{t:g}_e{e:g}"
                                f"_sl{sl:g}_rr{rr:g}"
                            )
                            out.append(_mk_spec(name, fam, note=f"densify r{round_i}", **g))
    seen: set[str] = set()
    uniq = []
    for s in out:
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        uniq.append(s)
    return uniq


def _rank_key(cell: dict) -> tuple:
    return (
        1 if cell.get("verdict") == "SURVIVES" else 0,
        float(cell.get("exp_r") or -99),
        float(cell.get("oos_exp_r") or -99),
        int(cell.get("wf_green") or 0),
        float(cell.get("mc_prob_profit") or 0),
        float(cell.get("pnl") or -1e18),
    )


def _slim_tests(tests: dict) -> dict:
    out = {}
    for k, v in (tests or {}).items():
        if k == "mc_prob_profit":
            out[k] = v
            continue
        if isinstance(v, dict):
            out[k] = {
                "n": v.get("n"),
                "pnl": v.get("pnl"),
                "exp_r": v.get("exp_r"),
                "win_rate": v.get("win_rate"),
            }
    return out


def _is_firm_pair(a: dict, b: dict) -> bool:
    """a=6m cell, b=1y cell."""
    if a.get("verdict") != "SURVIVES" or b.get("verdict") != "SURVIVES":
        return False
    if float(a.get("oos_exp_r") or -1) <= 0 or float(b.get("oos_exp_r") or -1) <= 0:
        return False
    if int(a.get("wf_green") or 0) < 2 or int(b.get("wf_green") or 0) < 2:
        return False
    mc_a = a.get("mc_prob_profit")
    mc_b = b.get("mc_prob_profit")
    if mc_a is not None and float(mc_a) < 55:
        return False
    if mc_b is not None and float(mc_b) < 55:
        return False
    if int(b.get("trades") or 0) < 80:
        return False
    return True


def _eval_specs(
    specs: list[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    df_15: pd.DataFrame,
    df_1h: pd.DataFrame,
    *,
    t0: float,
    done0: int,
    total_est: int,
    all_cells: list[dict],
    seen_names: set[str],
) -> int:
    """Evaluate specs across TF×windows; append to all_cells. Returns new done count."""
    done = done0
    # skip already-evaluated names (densify rounds)
    specs = [s for s in specs if s["name"] not in seen_names]
    for s in specs:
        seen_names.add(s["name"])

    for entry_tf in ENTRY_TFS:
        df_full = frames[entry_tf]
        for wtag, wdays in WINDOWS:
            end = df_full.index[-1]
            ex = df_full[df_full.index >= end - pd.Timedelta(days=wdays)].copy()
            d15 = df_15[df_15.index >= end - pd.Timedelta(days=wdays + 10)].copy()
            d1h = df_1h[df_1h.index >= end - pd.Timedelta(days=wdays + 10)].copy()
            if len(ex) < 250:
                _log(f"SKIP {entry_tf}/{wtag}: bars={len(ex)}")
                done += len(specs)
                continue
            sync = _build_sync(ex, d15, d1h)
            sync_rate = float((sync != 0).mean())
            _log(
                f"\n===== {entry_tf} / {wtag} bars={len(ex):,} sync={sync_rate:.1%} "
                f"elapsed={time.time()-t0:.0f}s specs={len(specs)} ====="
            )
            entry_cache: dict[str, list] = {}
            for spec in specs:
                ek = _entry_key(spec)
                if ek not in entry_cache:
                    entry_cache[ek] = _collect_entries(ex, sync, spec)
                ents = entry_cache[ek]
                tr = _simulate_fast(ex, ents, float(spec["rr"]))
                bat = _battery_from_trades(tr, len(ex))
                full = bat.get("full") or {}
                cell = {
                    "name": spec["name"],
                    "note": spec.get("note", ""),
                    "entry_tf": entry_tf,
                    "window": wtag,
                    "spec": {k: v for k, v in spec.items() if k != "note"},
                    "entries_raw": len(ents),
                    "trades": full.get("n", 0),
                    "pnl": full.get("pnl", 0.0),
                    "exp_r": full.get("exp_r", 0.0),
                    "win_rate": full.get("win_rate", 0.0),
                    "verdict": bat.get("verdict"),
                    "wf_green": bat.get("wf_green"),
                    "wf_n": bat.get("wf_n"),
                    "oos_exp_r": ((bat.get("oos") or {}).get("exp_r")),
                    "mc_prob_profit": (bat.get("monte_carlo") or {}).get("prob_profit"),
                    "tests": _slim_tests(bat.get("tests") or {}),
                }
                all_cells.append(cell)
                done += 1
                if done % 50 == 0 or cell["verdict"] == "SURVIVES":
                    _log(
                        f"  [{done}/{total_est}] {entry_tf}/{wtag} {cell['name'][:55]} "
                        f"n={cell['trades']} E[R]={cell['exp_r']} {cell['verdict']} "
                        f"OOS={cell['oos_exp_r']} WF={cell['wf_green']}/{cell['wf_n']} "
                        f"t={time.time()-t0:.0f}s"
                    )
                if done % 80 == 0:
                    _progress(
                        {
                            "done": done,
                            "total": total_est,
                            "elapsed_sec": round(time.time() - t0, 1),
                            "pct": round(100.0 * done / max(total_est, 1), 1),
                        }
                    )
    return done


def _select_firm(all_cells: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for c in all_cells:
        if c["window"] not in FIRM_WINDOWS:
            continue
        k = (c["entry_tf"], c["name"])
        by_key.setdefault(k, {})[c["window"]] = c

    firm = []
    for (tf, name), wins in by_key.items():
        if "6m" not in wins or "1y" not in wins:
            continue
        a, b = wins["6m"], wins["1y"]
        if not _is_firm_pair(a, b):
            continue
        firm.append(
            {
                "entry_tf": tf,
                "name": name,
                "spec": a["spec"],
                "note": a["note"],
                "6m": {
                    "exp_r": a["exp_r"],
                    "oos_exp_r": a["oos_exp_r"],
                    "pnl": a["pnl"],
                    "trades": a["trades"],
                    "wf_green": a["wf_green"],
                    "mc": a["mc_prob_profit"],
                    "tests": a["tests"],
                },
                "1y": {
                    "exp_r": b["exp_r"],
                    "oos_exp_r": b["oos_exp_r"],
                    "pnl": b["pnl"],
                    "trades": b["trades"],
                    "wf_green": b["wf_green"],
                    "mc": b["mc_prob_profit"],
                    "tests": b["tests"],
                },
                "score": float(a["exp_r"])
                + float(b["exp_r"])
                + 0.5 * float(a.get("oos_exp_r") or 0)
                + 0.5 * float(b.get("oos_exp_r") or 0),
            }
        )
    return sorted(firm, key=lambda x: x["score"], reverse=True)


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass

    specs = _candidate_specs()
    n_keys = len({_entry_key(s) for s in specs})
    _log(
        f"=== ETH with-trend MATH/DIST deep hunt · {len(specs)} specs · "
        f"{n_keys} entry-keys · TF={ENTRY_TFS} · windows={[w for w,_ in WINDOWS]} ==="
    )
    _log("Bias=15m+1h sync ONLY · ablation gates · 8-test battery · RESEARCH")
    _log(
        f"Promise check: no strategy never bleeds — firm = survive hard tests. "
        f"min_runtime={MIN_RUNTIME_SEC}s"
    )

    # 15m frame used both as entry TF and HTF bias source
    df_15 = _enrich(_load_tf(SYMBOL, "15m", 400))
    df_1h = _enrich(_load_tf(SYMBOL, "1h", 400))
    frames = {tf: _enrich(_load_tf(SYMBOL, tf, 400)) for tf in ENTRY_TFS if tf != "15m"}
    frames["15m"] = df_15
    for tf, df in frames.items():
        _log(f"ready {tf}: {len(df):,} bars {df.index[0].date()}→{df.index[-1].date()}")

    all_cells: list[dict] = []
    seen_names: set[str] = set()
    total_est = len(ENTRY_TFS) * len(WINDOWS) * len(specs)
    done = _eval_specs(
        specs,
        frames,
        df_15,
        df_1h,
        t0=t0,
        done0=0,
        total_est=total_est,
        all_cells=all_cells,
        seen_names=seen_names,
    )

    # Densify until ≥1h or firm winner found
    densify_round = 0
    while time.time() - t0 < MIN_RUNTIME_SEC:
        firm_probe = _select_firm(all_cells)
        if firm_probe:
            _log(
                f"\n*** firm winner found at t={time.time()-t0:.0f}s — "
                f"continuing densify until {MIN_RUNTIME_SEC}s floor ***"
            )
        # leaders: best 1y with trades
        leaders = [
            c
            for c in all_cells
            if c["window"] == "1y" and int(c.get("trades") or 0) >= 20
        ]
        leaders = sorted(leaders, key=_rank_key, reverse=True)[:30]
        if not leaders:
            leaders = sorted(all_cells, key=_rank_key, reverse=True)[:30]
        seed = [c["spec"] | {"family": c["spec"]["family"], "name": c["name"]} for c in leaders]
        # restore family from cell.spec
        densify_round += 1
        extra = _densify_specs(seed, densify_round)
        # cap chunk so we don't OOM JSON; loop until time
        extra = extra[:1200]
        _log(
            f"\n##### DENSIFY round {densify_round}: +{len(extra)} specs "
            f"(elapsed={time.time()-t0:.0f}s / {MIN_RUNTIME_SEC}s) #####"
        )
        total_est = done + len(ENTRY_TFS) * len(WINDOWS) * len(extra)
        done = _eval_specs(
            extra,
            frames,
            df_15,
            df_1h,
            t0=t0,
            done0=done,
            total_est=total_est,
            all_cells=all_cells,
            seen_names=seen_names,
        )
        if densify_round >= 12:
            _log("densify round cap hit")
            break

    # Rank firm pairs
    _log("\n########## FIRM SELECTION ##########")
    firm = _select_firm(all_cells)

    # Window winners (any)
    winners_by_window = {}
    for wtag, _ in WINDOWS:
        cells = [c for c in all_cells if c["window"] == wtag]
        if not cells:
            continue
        best = max(cells, key=_rank_key)
        winners_by_window[wtag] = {
            "name": best["name"],
            "entry_tf": best["entry_tf"],
            "verdict": best["verdict"],
            "exp_r": best["exp_r"],
            "oos_exp_r": best["oos_exp_r"],
            "pnl": best["pnl"],
            "trades": best["trades"],
            "tests": best["tests"],
        }
        _log(
            f"WINDOW {wtag}: {best['entry_tf']} {best['name'][:50]} "
            f"{best['verdict']} E[R]={best['exp_r']} n={best['trades']}"
        )

    if firm:
        _log(f"\nFIRM COUNT={len(firm)}")
        for f in firm[:10]:
            _log(
                f"  {f['entry_tf']} {f['name'][:55]} score={f['score']:.3f} "
                f"6m E[R]={f['6m']['exp_r']} 1y E[R]={f['1y']['exp_r']} "
                f"n1y={f['1y']['trades']}"
            )
        clear = firm[0]
        _log(
            f"\n>>> CLEAR FIRM WINNER: entry={clear['entry_tf']} · {clear['name']}\n"
            f"    {clear['note']}\n"
            f"    spec={clear['spec']}"
        )
    else:
        clear = None
        _log("\n>>> NO FIRM WINNER under hard gates")
        yt = [
            c
            for c in all_cells
            if c["window"] == "1y"
            and float(c.get("exp_r") or -1) > 0
            and int(c.get("trades") or 0) >= 40
        ]
        if yt:
            best = max(yt, key=_rank_key)
            _log(
                f"    soft 1y best: {best['entry_tf']} {best['name'][:60]} "
                f"{best['verdict']} E[R]={best['exp_r']} n={best['trades']}"
            )

    elapsed = round(time.time() - t0, 1)
    # slim cells for disk if huge — keep survivors + top by score
    cells_out = all_cells
    if len(all_cells) > 8000:
        keep = {id(c) for c in all_cells if c.get("verdict") == "SURVIVES"}
        ranked = sorted(all_cells, key=_rank_key, reverse=True)[:4000]
        keep |= {id(c) for c in ranked}
        cells_out = [c for c in all_cells if id(c) in keep]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "symbol": SYMBOL,
        "risk_usd": RISK_USD,
        "elapsed_sec": elapsed,
        "n_specs_evaluated": len(seen_names),
        "n_cells": len(all_cells),
        "n_cells_saved": len(cells_out),
        "densify_rounds": densify_round,
        "method": {
            "bias": "15m+1h EMA8/14 sync",
            "math": "ablation Fourier HF/LF, ret_z pullback, kurtosis, skew, eff, trend_str, slope + densify",
            "battery": "8 tests + MC",
            "firm_rule": "SURVIVES 6m+1y, OOS E[R]>0, WF≥2, MC≥55%, n_1y≥80",
            "min_runtime_sec": MIN_RUNTIME_SEC,
            "disclaimer": "No strategy never bleeds; firm = hard-test survivor only",
        },
        "winners_by_window": winners_by_window,
        "firm_candidates": firm[:20],
        "clear_winner": clear,
        "recommend_live": bool(clear),
        "cells": cells_out,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _progress({"done": done, "total": done, "elapsed_sec": elapsed, "pct": 100.0})
    _log(f"\nWrote {OUT} in {elapsed}s ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
