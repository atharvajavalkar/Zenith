#!/usr/bin/env python3
"""RESEARCH ONLY — Zenith deep MATH INVENTION hunt (5–10h).

Invent weird/crazy quantitative entry formulas (NOT TV/retail/HF clones),
score them on Dukascopy decade hist with multi-horizon walk-forward.

Universe: GOLD USDCHF USDJPY NAS100 UKOIL AUDUSD BTCUSD ETHUSD
TFs: 5m · 15m (decade-scale); densify leaders on 3m when time allows.

Batteries per formula:
  • Full-span SURVIVES-style battery (OOS / 4-fold WF / torture / MC)
  • Multi-period WF: 1y · 3y · 5y · full available span
  • Prefer high win-rate + E[R]>0 + MC + multi-horizon green

DOES NOT change live brains / .env. Report equations only.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD, _prep, _stats  # noqa: E402
from analysis.lab_agentic_1y_battery import _battery_from_trades  # noqa: E402
from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv, local_span  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"
EQ = Path(ROOT) / "data" / "lab_zenith_math_invention_equations.md"
LOG = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.log"
PROGRESS = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.progress.json"
LOCK = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.lock"
OBSIDIAN = Path(ROOT) / "data" / "ZenithRecovery" / "MathInvention"

BOOK = (
    "ETHUSD",
    "BTCUSD",
    "GOLD",
    "NAS100",
    "USDJPY",
    "AUDUSD",
    "USDCHF",
    "UKOIL",
)
ENTRY_TFS = ("5m", "15m")
DENSE_TFS = ("3m", "5m", "15m")
RRS = (1.5, 2.0, 2.4, 3.0, 4.0)

# Multi-horizon evaluation windows (days); "full" = all available
PERIODS = (
    ("1y", 365),
    ("3y", 365 * 3),
    ("5y", 365 * 5),
    ("full", None),
)

MIN_RUNTIME_SEC = int(os.getenv("MATH_INVENTION_MIN_SEC", str(5 * 3600)))
MAX_RUNTIME_SEC = int(os.getenv("MATH_INVENTION_MAX_SEC", str(10 * 3600)))
# Cap bars for speed on decade 5m (~1M) — use last N years for feature build,
# but still score multi-period on available span up to CAP_YEARS
CAP_YEARS = float(os.getenv("MATH_INVENTION_CAP_YEARS", "8"))


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _progress(payload: dict) -> None:
    try:
        PROGRESS.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def _obs(rel: str, body: str) -> None:
    path = OBSIDIAN / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ── Feature engineering (deep / weird) ──────────────────────────────────────


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


def _spectral_entropy(ret: np.ndarray, win: int = 64, stride: int = 8) -> np.ndarray:
    n = len(ret)
    out = np.full(n, np.nan, dtype=float)
    log2 = math.log(2.0)
    for i in range(win, n, stride):
        seg = ret[i - win : i]
        if not np.isfinite(seg).all():
            continue
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg)) ** 2
        s = float(spec.sum())
        if s <= 0:
            continue
        p = spec / s
        p = p[p > 0]
        ent = float(-(p * np.log(p) / log2).sum())
        out[i] = ent / max(1.0, math.log(len(spec) or 2) / log2)
    return pd.Series(out).ffill().fillna(0.5).to_numpy(dtype=float)


def _hurst_rs(ret: np.ndarray, win: int = 128, stride: int = 16) -> np.ndarray:
    """Rough R/S Hurst on rolling window (vector-ish, strided)."""
    n = len(ret)
    out = np.full(n, np.nan, dtype=float)
    for i in range(win, n, stride):
        x = ret[i - win : i]
        if not np.isfinite(x).all() or np.std(x) < 1e-12:
            continue
        y = np.cumsum(x - x.mean())
        r = float(y.max() - y.min())
        s = float(np.std(x))
        if s <= 0 or r <= 0:
            continue
        out[i] = math.log(r / s) / math.log(win)
    return pd.Series(out).ffill().fillna(0.5).to_numpy(dtype=float)


def _higuchi_fd(close: np.ndarray, win: int = 64, stride: int = 16, kmax: int = 6) -> np.ndarray:
    """Higuchi fractal dimension of price path (rolling, strided)."""
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    for i in range(win, n, stride):
        x = close[i - win : i].astype(float)
        if not np.isfinite(x).all():
            continue
        lk = []
        ks = []
        for k in range(1, kmax + 1):
            lengths = []
            for m in range(k):
                idx = np.arange(m, win, k)
                if len(idx) < 2:
                    continue
                seg = x[idx]
                L = np.sum(np.abs(np.diff(seg))) * (win - 1) / (k * k * (len(idx) - 1))
                lengths.append(L)
            if lengths:
                lk.append(math.log(max(np.mean(lengths), 1e-12)))
                ks.append(math.log(1.0 / k))
        if len(ks) >= 2:
            # slope of log L vs log(1/k) ≈ FD
            coef = np.polyfit(ks, lk, 1)
            out[i] = float(coef[0])
    return pd.Series(out).ffill().fillna(1.5).to_numpy(dtype=float)


def _ou_halflife(close: np.ndarray, win: int = 64, stride: int = 16) -> np.ndarray:
    """Ornstein–Uhlenbeck half-life estimate on log-price vs its mean."""
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    for i in range(win + 1, n, stride):
        y = np.log(np.clip(close[i - win : i], 1e-12, None))
        x = y[:-1] - y[:-1].mean()
        dy = np.diff(y)
        if len(x) < 10 or np.std(x) < 1e-12:
            continue
        # dy = a + b x + e  → kappa ≈ -b
        b = float(np.cov(x, dy)[0, 1] / (np.var(x) + 1e-18))
        if b >= -1e-8:
            out[i] = 999.0
            continue
        kappa = -b
        out[i] = math.log(2.0) / kappa
    return pd.Series(out).ffill().fillna(999.0).to_numpy(dtype=float)


def _allan_var_ratio(ret: np.ndarray, short: int = 8, long: int = 32, stride: int = 8) -> np.ndarray:
    n = len(ret)
    out = np.full(n, np.nan, dtype=float)
    win = long * 4
    for i in range(win, n, stride):
        x = ret[i - win : i]
        if not np.isfinite(x).all():
            continue

        def avar(tau: int) -> float:
            m = len(x) // tau
            if m < 3:
                return np.nan
            chunk = x[: m * tau].reshape(m, tau).sum(axis=1)
            d = np.diff(chunk)
            return float(0.5 * np.mean(d * d))

        a_s, a_l = avar(short), avar(long)
        if not np.isfinite(a_s) or not np.isfinite(a_l) or a_l <= 0:
            continue
        out[i] = a_s / a_l
    return pd.Series(out).ffill().fillna(1.0).to_numpy(dtype=float)


def _perm_entropy(ret: np.ndarray, win: int = 64, stride: int = 16, order: int = 3) -> np.ndarray:
    n = len(ret)
    out = np.full(n, np.nan, dtype=float)
    from itertools import permutations

    patterns = list(permutations(range(order)))
    pat_idx = {p: i for i, p in enumerate(patterns)}
    npat = len(patterns)
    for i in range(win, n, stride):
        x = ret[i - win : i]
        if not np.isfinite(x).all():
            continue
        counts = np.zeros(npat, dtype=float)
        for j in range(len(x) - order + 1):
            w = x[j : j + order]
            ranks = tuple(np.argsort(np.argsort(w)))
            counts[pat_idx[ranks]] += 1
        s = counts.sum()
        if s <= 0:
            continue
        p = counts / s
        p = p[p > 0]
        ent = float(-(p * np.log(p)).sum() / math.log(npat))
        out[i] = ent
    return pd.Series(out).ffill().fillna(0.5).to_numpy(dtype=float)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    o = out["open"].astype(float)
    atr = out["atr"].replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ret"] = ret
    out["eff"] = ((c - o).abs() / rng).clip(0, 1).fillna(0)
    out["ret_z"] = ((ret - ret.rolling(60).mean()) / ret.rolling(60).std().replace(0, np.nan)).fillna(0)
    out["kurt60"] = ret.rolling(60).kurt().fillna(0)
    out["skew60"] = ret.rolling(60).skew().fillna(0)
    out["vol60"] = ret.rolling(60).std().fillna(0)
    out["vol_ratio"] = (ret.rolling(20).std() / ret.rolling(60).std().replace(0, np.nan)).fillna(1)
    cn = c.to_numpy(dtype=float)
    rn = ret.to_numpy(dtype=float)
    out["fourier_hf"] = _rolling_fourier_hf(rn, 64, 12)
    out["fourier_lf"] = 1.0 - out["fourier_hf"]
    out["spec_ent"] = _spectral_entropy(rn, 64, 12)
    out["hurst"] = _hurst_rs(rn, 128, 24)
    out["higuchi"] = _higuchi_fd(cn, 64, 24, 5)
    out["ou_hl"] = _ou_halflife(cn, 64, 24)
    out["allan_r"] = _allan_var_ratio(rn, 8, 32, 16)
    out["perm_ent"] = _perm_entropy(rn, 48, 24, 3)
    # Drift-diffusion SNR
    mu = ret.rolling(40).mean()
    sig = ret.rolling(40).std().replace(0, np.nan)
    out["snr"] = (mu.abs() / sig).fillna(0)
    out["drift_sign"] = np.sign(mu).fillna(0)
    # Path torsion (angle of consecutive return vectors)
    r1 = ret
    r2 = ret.shift(1)
    denom = (r1.abs() + r2.abs()).replace(0, np.nan)
    out["torsion"] = ((r1 - r2).abs() / denom).fillna(0)
    # Zero-crossing rate of detrended close
    det = c - c.rolling(40).mean()
    zc = ((det * det.shift(1)) < 0).astype(float)
    out["zcr"] = zc.rolling(40).mean().fillna(0)
    # Spectral centroid proxy
    out["spec_cent"] = out["fourier_hf"]  # already HF share; densify uses both
    # Kalman-ish innov z: residual of close vs EMA
    ema = c.ewm(span=20, adjust=False).mean()
    resid = (c - ema) / atr
    out["kalman_z"] = (
        (resid - resid.rolling(60).mean()) / resid.rolling(60).std().replace(0, np.nan)
    ).fillna(0)
    # Bimodality coefficient on rolling returns
    # γ = (skew^2 + 1) / (kurt + 3*(n-1)^2/((n-2)(n-3)))
    sk = out["skew60"]
    ku = out["kurt60"]
    out["bimod"] = ((sk ** 2 + 1) / (ku + 3).replace(0, np.nan)).fillna(0).clip(0, 5)
    # Cusum of signed SNR
    out["cusum"] = (out["snr"] * out["drift_sign"]).cumsum()
    out["cusum_slope"] = out["cusum"].diff(10) / 10.0
    # Efficiency × LF coherence
    out["eff_lf"] = out["eff"] * out["fourier_lf"]
    # Levy-ish tail index proxy: 1 / mean(|z|^α) rough
    az = out["ret_z"].abs().clip(0, 8)
    out["tail_proxy"] = (1.0 / az.rolling(80).mean().replace(0, np.nan)).fillna(1)
    return out


def _atr_sl_entries(
    df: pd.DataFrame,
    sides: np.ndarray,
    *,
    sl_atr: float = 1.2,
    min_gap: int = 3,
) -> list[dict]:
    """sides: 1 BUY, -1 SELL, 0 flat."""
    atr = df["atr"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
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
        if s > 0:
            sl = entry - sl_atr * a
            side = "BUY"
        else:
            sl = entry + sl_atr * a
            side = "SELL"
        out.append({"ei": i, "side": side, "entry": entry, "sl": sl})
        busy = i + min_gap
    return out


# ── Invented formula catalog ────────────────────────────────────────────────


FormulaFn = Callable[[pd.DataFrame], tuple[np.ndarray, str]]


def _catalog() -> list[tuple[str, str, FormulaFn]]:
    """id, latex/equation blurb, fn→(sides, equation)."""

    def snr_drift(df: pd.DataFrame, thr: float = 0.35) -> tuple[np.ndarray, str]:
        snr = df["snr"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((snr >= thr) & (ds != 0), ds.astype(int), 0)
        eq = (
            rf"SNR_t=\frac{{|\mu_{{40}}(r)|}}{{\sigma_{{40}}(r)}};\ "
            rf"side=\mathrm{{sign}}(\mu)\cdot\mathbf{{1}}[SNR\ge {thr}]"
        )
        return sides, eq

    def hurst_trend(df: pd.DataFrame, hmin: float = 0.55) -> tuple[np.ndarray, str]:
        H = df["hurst"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((H >= hmin) & (ds != 0), ds.astype(int), 0)
        eq = rf"H_{{RS}}\ge {hmin}\ \wedge\ side=\mathrm{{sign}}(\mu_{{40}})"
        return sides, eq

    def hurst_mr(df: pd.DataFrame, hmax: float = 0.45) -> tuple[np.ndarray, str]:
        H = df["hurst"].to_numpy()
        kz = df["kalman_z"].to_numpy()
        sides = np.zeros(len(df), dtype=int)
        sides[(H <= hmax) & (kz <= -1.0)] = 1
        sides[(H <= hmax) & (kz >= 1.0)] = -1
        eq = (
            rf"H_{{RS}}\le {hmax};\ BUY: z_{{kal}}\le -1;\ SELL: z_{{kal}}\ge 1"
        )
        return sides, eq

    def ou_snap(df: pd.DataFrame, hl_lo: float = 5, hl_hi: float = 40) -> tuple[np.ndarray, str]:
        hl = df["ou_hl"].to_numpy()
        kz = df["kalman_z"].to_numpy()
        sides = np.zeros(len(df), dtype=int)
        m = (hl >= hl_lo) & (hl <= hl_hi)
        sides[m & (kz <= -1.2)] = 1
        sides[m & (kz >= 1.2)] = -1
        eq = (
            rf"t_{{1/2}}^{{OU}}\in[{hl_lo},{hl_hi}];\ "
            rf"fade\ z_{{kal}}"
        )
        return sides, eq

    def spec_ent_break(df: pd.DataFrame, emax: float = 0.55) -> tuple[np.ndarray, str]:
        se = df["spec_ent"].to_numpy()
        lf = df["fourier_lf"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((se <= emax) & (lf >= 0.55) & (ds != 0), ds.astype(int), 0)
        eq = (
            rf"S_{{spec}}\le {emax}\ \wedge\ LF\ge 0.55\ \wedge\ "
            rf"side=\mathrm{{sign}}(\mu)"
        )
        return sides, eq

    def higuchi_smooth_trend(df: pd.DataFrame, fd_max: float = 1.45) -> tuple[np.ndarray, str]:
        fd = df["higuchi"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((fd <= fd_max) & (ds != 0), ds.astype(int), 0)
        eq = rf"D_{{Higuchi}}\le {fd_max}\ \wedge\ side=\mathrm{{sign}}(\mu)"
        return sides, eq

    def allan_persist(df: pd.DataFrame, rmax: float = 0.85) -> tuple[np.ndarray, str]:
        ar = df["allan_r"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((ar <= rmax) & (ds != 0), ds.astype(int), 0)
        eq = rf"\mathcal{{A}}_{{8}}/\mathcal{{A}}_{{32}}\le {rmax}\ \wedge\ trend"
        return sides, eq

    def perm_order(df: pd.DataFrame, pmax: float = 0.85) -> tuple[np.ndarray, str]:
        pe = df["perm_ent"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((pe <= pmax) & (ds != 0), ds.astype(int), 0)
        eq = rf"H_{{perm}}\le {pmax}\ \wedge\ side=\mathrm{{sign}}(\mu)"
        return sides, eq

    def eff_lf_thrust(df: pd.DataFrame, thr: float = 0.28) -> tuple[np.ndarray, str]:
        x = df["eff_lf"].to_numpy()
        ds = np.sign(df["close"].diff(3).to_numpy())
        sides = np.where((x >= thr) & (ds != 0), ds.astype(int), 0)
        eq = rf"eff\cdot LF\ge {thr};\ side=\mathrm{{sign}}(\Delta_3 close)"
        return sides, eq

    def kalman_fade(df: pd.DataFrame, z: float = 1.8) -> tuple[np.ndarray, str]:
        kz = df["kalman_z"].to_numpy()
        sides = np.zeros(len(df), dtype=int)
        sides[kz <= -z] = 1
        sides[kz >= z] = -1
        eq = rf"z_{{kal}}=\frac{{c-EMA_{{20}}}}{{ATR}};\ fade\ |z|\ge {z}"
        return sides, eq

    def torsion_quiet_trend(df: pd.DataFrame, tmax: float = 0.35) -> tuple[np.ndarray, str]:
        t = df["torsion"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((t <= tmax) & (ds != 0), ds.astype(int), 0)
        eq = rf"\tau_{{path}}\le {tmax}\ \wedge\ trend"
        return sides, eq

    def zcr_compress(df: pd.DataFrame, zmax: float = 0.25) -> tuple[np.ndarray, str]:
        zcr = df["zcr"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((zcr <= zmax) & (ds != 0), ds.astype(int), 0)
        eq = rf"ZCR_{{40}}\le {zmax}\ \wedge\ trend"
        return sides, eq

    def bimod_break(df: pd.DataFrame, bmin: float = 0.55) -> tuple[np.ndarray, str]:
        b = df["bimod"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((b >= bmin) & (ds != 0), ds.astype(int), 0)
        eq = rf"\beta=\frac{{skew^2+1}}{{kurt+3}}\ge {bmin}\ \wedge\ trend"
        return sides, eq

    def cusum_fire(df: pd.DataFrame, thr: float = 0.02) -> tuple[np.ndarray, str]:
        cs = df["cusum_slope"].to_numpy()
        sides = np.zeros(len(df), dtype=int)
        sides[cs >= thr] = 1
        sides[cs <= -thr] = -1
        eq = rf"slope_{{10}}(\sum SNR\cdot sign(\mu))\ \mathrm{{fire}}\ @ {thr}"
        return sides, eq

    def tail_mid_snr(df: pd.DataFrame) -> tuple[np.ndarray, str]:
        tp = df["tail_proxy"].to_numpy()
        snr = df["snr"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((tp >= 0.8) & (tp <= 2.5) & (snr >= 0.3) & (ds != 0), ds.astype(int), 0)
        eq = r"tailProxy\in[0.8,2.5]\ \wedge\ SNR\ge 0.3\ \wedge\ trend"
        return sides, eq

    def hf_spike_fade(df: pd.DataFrame, hfmin: float = 0.62) -> tuple[np.ndarray, str]:
        hf = df["fourier_hf"].to_numpy()
        kz = df["kalman_z"].to_numpy()
        sides = np.zeros(len(df), dtype=int)
        m = hf >= hfmin
        sides[m & (kz <= -1.0)] = 1
        sides[m & (kz >= 1.0)] = -1
        eq = rf"HF\ge {hfmin}\ \Rightarrow\ fade\ z_{{kal}}"
        return sides, eq

    def lf_thrust_eff(df: pd.DataFrame) -> tuple[np.ndarray, str]:
        lf = df["fourier_lf"].to_numpy()
        eff = df["eff"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        sides = np.where((lf >= 0.60) & (eff >= 0.35) & (ds != 0), ds.astype(int), 0)
        eq = r"LF\ge 0.60\ \wedge\ eff\ge 0.35\ \wedge\ trend"
        return sides, eq

    def vol_compress_release(df: pd.DataFrame) -> tuple[np.ndarray, str]:
        vr = df["vol_ratio"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        # compress then expand: vol_ratio rising through 1 from below
        vr_prev = np.roll(vr, 3)
        fire = (vr_prev < 0.85) & (vr >= 1.05)
        sides = np.where(fire & (ds != 0), ds.astype(int), 0)
        eq = r"\sigma_{20}/\sigma_{60}:\ <0.85\to\ge 1.05\ \wedge\ trend"
        return sides, eq

    def skew_align_snr(df: pd.DataFrame) -> tuple[np.ndarray, str]:
        sk = df["skew60"].to_numpy()
        snr = df["snr"].to_numpy()
        sides = np.zeros(len(df), dtype=int)
        sides[(snr >= 0.28) & (sk > 0.15)] = 1
        sides[(snr >= 0.28) & (sk < -0.15)] = -1
        eq = r"SNR\ge 0.28\ \wedge\ side=\mathrm{{sign}}(skew_{60})"
        return sides, eq

    def fractal_mr(df: pd.DataFrame) -> tuple[np.ndarray, str]:
        fd = df["higuchi"].to_numpy()
        kz = df["kalman_z"].to_numpy()
        sides = np.zeros(len(df), dtype=int)
        m = fd >= 1.65
        sides[m & (kz <= -1.3)] = 1
        sides[m & (kz >= 1.3)] = -1
        eq = r"D_{Higuchi}\ge 1.65\ \Rightarrow\ fade\ |z_{kal}|\ge 1.3"
        return sides, eq

    # Parameterized variants for densify
    formulas: list[tuple[str, str, FormulaFn]] = [
        ("snr_drift_0.30", "Drift-diffusion SNR trend", lambda d: snr_drift(d, 0.30)),
        ("snr_drift_0.40", "Drift-diffusion SNR trend", lambda d: snr_drift(d, 0.40)),
        ("snr_drift_0.55", "Drift-diffusion SNR trend", lambda d: snr_drift(d, 0.55)),
        ("hurst_trend_0.52", "Hurst persistence trend", lambda d: hurst_trend(d, 0.52)),
        ("hurst_trend_0.58", "Hurst persistence trend", lambda d: hurst_trend(d, 0.58)),
        ("hurst_mr_0.45", "Hurst anti-persist mean-revert", lambda d: hurst_mr(d, 0.45)),
        ("hurst_mr_0.40", "Hurst anti-persist mean-revert", lambda d: hurst_mr(d, 0.40)),
        ("ou_snap_5_40", "OU half-life snapback", lambda d: ou_snap(d, 5, 40)),
        ("ou_snap_8_25", "OU half-life snapback", lambda d: ou_snap(d, 8, 25)),
        ("spec_ent_0.50", "Spectral entropy collapse trend", lambda d: spec_ent_break(d, 0.50)),
        ("spec_ent_0.60", "Spectral entropy collapse trend", lambda d: spec_ent_break(d, 0.60)),
        ("higuchi_trend_1.40", "Low fractal-D smooth trend", lambda d: higuchi_smooth_trend(d, 1.40)),
        ("higuchi_trend_1.50", "Low fractal-D smooth trend", lambda d: higuchi_smooth_trend(d, 1.50)),
        ("allan_persist_0.80", "Allan variance persistence", lambda d: allan_persist(d, 0.80)),
        ("allan_persist_0.95", "Allan variance persistence", lambda d: allan_persist(d, 0.95)),
        ("perm_order_0.80", "Permutation-entropy order", lambda d: perm_order(d, 0.80)),
        ("perm_order_0.90", "Permutation-entropy order", lambda d: perm_order(d, 0.90)),
        ("eff_lf_0.25", "Efficiency×LF thrust", lambda d: eff_lf_thrust(d, 0.25)),
        ("eff_lf_0.32", "Efficiency×LF thrust", lambda d: eff_lf_thrust(d, 0.32)),
        ("kalman_fade_1.6", "Kalman innov fade", lambda d: kalman_fade(d, 1.6)),
        ("kalman_fade_2.0", "Kalman innov fade", lambda d: kalman_fade(d, 2.0)),
        ("kalman_fade_2.4", "Kalman innov fade", lambda d: kalman_fade(d, 2.4)),
        ("torsion_0.30", "Path-torsion quiet trend", lambda d: torsion_quiet_trend(d, 0.30)),
        ("torsion_0.40", "Path-torsion quiet trend", lambda d: torsion_quiet_trend(d, 0.40)),
        ("zcr_0.22", "Zero-cross compress trend", lambda d: zcr_compress(d, 0.22)),
        ("zcr_0.30", "Zero-cross compress trend", lambda d: zcr_compress(d, 0.30)),
        ("bimod_0.55", "Bimodality breakout", lambda d: bimod_break(d, 0.55)),
        ("bimod_0.70", "Bimodality breakout", lambda d: bimod_break(d, 0.70)),
        ("cusum_0.015", "CUSUM SNR fire", lambda d: cusum_fire(d, 0.015)),
        ("cusum_0.025", "CUSUM SNR fire", lambda d: cusum_fire(d, 0.025)),
        ("tail_mid_snr", "Mid-tail + SNR trend", tail_mid_snr),
        ("hf_spike_fade_0.60", "HF spike fade", lambda d: hf_spike_fade(d, 0.60)),
        ("hf_spike_fade_0.70", "HF spike fade", lambda d: hf_spike_fade(d, 0.70)),
        ("lf_thrust_eff", "LF thrust + efficiency", lf_thrust_eff),
        ("vol_compress_release", "Vol compress→release", vol_compress_release),
        ("skew_align_snr", "Skew-aligned SNR", skew_align_snr),
        ("fractal_mr", "High fractal-D mean-revert", fractal_mr),
    ]
    return formulas


def _load_enriched(sym: str, tf: str, years: float) -> Optional[pd.DataFrame]:
    clear_hist_cache()
    if not has_local_hist(sym, tf):
        return None
    sp = local_span(sym, tf)
    if not sp:
        return None
    end = sp[1]
    start = max(sp[0], end - pd.Timedelta(days=int(years * 365.25) + 40))
    raw = load_local_ohlcv(
        sym,
        tf=tf,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    if raw is None or len(raw) < 800:
        return None
    df = _prep(raw[["open", "high", "low", "close"]].dropna())
    df = _enrich(df)
    return df


def _slice_days(df: pd.DataFrame, days: Optional[int]) -> pd.DataFrame:
    if days is None:
        return df
    end = df.index[-1]
    return df[df.index >= end - pd.Timedelta(days=days)].copy()


def _trades_with_i(df: pd.DataFrame, entries: list, rr: float) -> list[dict]:
    """Simulate entries and attach bar index i for multi-period slicing."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    trades: list[dict] = []
    busy = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < busy or ei >= n - 1:
            continue
        side = e["side"]
        entry = float(e["entry"])
        sl0 = float(e["sl"])
        risk = abs(entry - sl0)
        if risk <= 0:
            continue
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
        trades.append(
            {
                "i": ei,
                "exit_i": exit_i,
                "side": side,
                "r": round(float(exit_r), 4),
                "pnl": round(float(exit_r) * RISK_USD, 2),
            }
        )
        busy = exit_i + 2
    return trades


def _period_stats(df: pd.DataFrame, trades: list[dict], days: Optional[int]) -> dict:
    if days is None:
        sub = trades
        n_bars = len(df)
    else:
        end = df.index[-1]
        start = end - pd.Timedelta(days=days)
        idxs = np.where(np.asarray(df.index >= start))[0]
        if len(idxs) < 200:
            return {"n": 0, "exp_r": 0.0, "win_rate": 0.0, "pnl": 0.0}
        i0 = int(idxs[0])
        sub = [t for t in trades if int(t["i"]) >= i0]
        n_bars = len(df) - i0
    st = _stats(sub)
    # mini WF on period
    bat = _battery_from_trades(sub, max(n_bars, 250))
    return {
        "n": st.get("n"),
        "exp_r": st.get("exp_r"),
        "win_rate": st.get("win_rate"),
        "pnl": st.get("pnl"),
        "verdict": bat.get("verdict"),
        "wf_green": bat.get("wf_green"),
        "mc": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "oos_exp_r": (bat.get("oos") or {}).get("exp_r"),
    }


def _score_cell(periods: dict[str, dict], bat: dict) -> float:
    """Prefer multi-horizon green + win rate + E[R] + MC."""
    s = 0.0
    full = bat.get("full") or {}
    oos = bat.get("oos") or {}
    mc = (bat.get("monte_carlo") or {}).get("prob_profit") or 0
    wr = float(full.get("win_rate") or 0)
    er = float(full.get("exp_r") or 0)
    oer = float(oos.get("exp_r") or 0)
    n = int(full.get("n") or 0)
    s += 4.0 * er
    s += 3.0 * oer
    s += 0.04 * wr  # win rate in %
    s += 0.03 * float(mc)
    s += 0.4 * float(bat.get("wf_green") or 0)
    s += min(2.5, n / 200.0)
    # multi-horizon bonuses
    green = 0
    for name, p in periods.items():
        if float(p.get("exp_r") or 0) > 0 and int(p.get("wf_green") or 0) >= 2:
            green += 1
            s += 1.2
        if float(p.get("exp_r") or 0) > 0:
            s += 0.4
    if green >= 3:
        s += 2.0
    if bat.get("verdict") == "SURVIVES":
        s += 2.5
    return round(s, 3)


def _firm_ok(periods: dict, bat: dict) -> bool:
    if bat.get("verdict") != "SURVIVES":
        return False
    if int(bat.get("wf_green") or 0) < 2:
        return False
    if float((bat.get("oos") or {}).get("exp_r") or 0) <= 0:
        return False
    # need 1y and 3y positive E[R] at minimum
    for key in ("1y", "3y"):
        p = periods.get(key) or {}
        if float(p.get("exp_r") or 0) <= 0:
            return False
        if int(p.get("n") or 0) < 25:
            return False
    return True


def _run_formula(
    df: pd.DataFrame,
    fid: str,
    label: str,
    fn: FormulaFn,
    rr: float,
    sl_atr: float = 1.2,
) -> Optional[dict]:
    sides, equation = fn(df)
    entries = _atr_sl_entries(df, sides, sl_atr=sl_atr)
    if len(entries) < 40:
        return None
    trades = _trades_with_i(df, entries, rr)
    if len(trades) < 30:
        return None
    bat = _battery_from_trades(trades, len(df))
    periods = {}
    for pname, days in PERIODS:
        periods[pname] = _period_stats(df, trades, days)
    cell = {
        "id": fid,
        "label": label,
        "equation": equation,
        "tf": None,
        "rr": rr,
        "sl_atr": sl_atr,
        "n": (bat.get("full") or {}).get("n"),
        "win_rate": (bat.get("full") or {}).get("win_rate"),
        "exp_r": (bat.get("full") or {}).get("exp_r"),
        "oos_exp_r": (bat.get("oos") or {}).get("exp_r"),
        "wf_green": bat.get("wf_green"),
        "mc": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "verdict": bat.get("verdict"),
        "periods": periods,
        "deep_gate": _firm_ok(periods, bat),
        "score": _score_cell(periods, bat),
    }
    return cell


def main() -> int:
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip() or "0")
            if pid and pid != os.getpid() and Path(f"/proc/{pid}").exists():
                _log(f"another math invention hunt running (pid {pid})")
                return 2
        except Exception:
            pass
    LOCK.write_text(str(os.getpid()))
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass

    catalog = _catalog()
    _log(
        "=== ZENITH MATH INVENTION DEEP HUNT ===\n"
        f"Book={BOOK}\nTFs={ENTRY_TFS} densify+=3m\n"
        f"Formulas={len(catalog)} · RR={list(RRS)}\n"
        f"Periods={ [p[0] for p in PERIODS] }\n"
        f"cap_years={CAP_YEARS} min={MIN_RUNTIME_SEC}s max={MAX_RUNTIME_SEC}s\n"
        "LIVE BRAINS NOT TOUCHED\n"
        "Goal: invent math edges — cannot promise win-all-trades\n"
    )
    _obs(
        "00_Math_Invention_Board.md",
        f"# Zenith Math Invention Board\n\nStarted {datetime.now(timezone.utc).isoformat()}\n"
        f"Formulas in catalog: {len(catalog)}\nResearch only — no live wire.\n",
    )

    depth = {}
    for sym in BOOK:
        depth[sym] = {}
        for tf in ("1m", "5m", "15m"):
            sp = local_span(sym, tf) if has_local_hist(sym, tf) else None
            if sp:
                depth[sym][tf] = {
                    "start": str(sp[0])[:19],
                    "end": str(sp[1])[:19],
                    "years": round((sp[1] - sp[0]).days / 365.25, 2),
                }
    _log("Hist depth: " + json.dumps(depth))

    per_pair: dict[str, Any] = {}
    n_sym = len(BOOK)

    for pi, sym in enumerate(BOOK):
        if time.time() - t0 > MAX_RUNTIME_SEC:
            _log("MAX_RUNTIME — stop")
            break
        _log(f"\n######## {sym} ########")
        scored: list[dict] = []
        for tf in ENTRY_TFS:
            if time.time() - t0 > MAX_RUNTIME_SEC:
                break
            _log(f"  load {sym} {tf} ({CAP_YEARS}y) …")
            _progress(
                {
                    "status": "running",
                    "pct": int(5 + 70 * pi / n_sym),
                    "message": f"[{sym}] enrich {tf}",
                    "symbol": sym,
                    "elapsed_sec": round(time.time() - t0, 1),
                }
            )
            try:
                df = _load_enriched(sym, tf, CAP_YEARS)
            except Exception as exc:
                _log(f"  FAIL load {sym} {tf}: {exc}")
                continue
            if df is None:
                _log(f"  skip {sym} {tf}: no hist")
                continue
            _log(f"  enriched {len(df):,} bars {str(df.index[0])[:10]}→{str(df.index[-1])[:10]}")
            for fid, label, fn in catalog:
                if time.time() - t0 > MAX_RUNTIME_SEC:
                    break
                for rr in RRS:
                    try:
                        cell = _run_formula(df, fid, label, fn, rr)
                    except Exception as exc:
                        _log(f"  skip {fid}@{tf} RR={rr}: {exc}")
                        continue
                    if not cell:
                        continue
                    cell["tf"] = tf
                    cell["symbol"] = sym
                    scored.append(cell)
                    if cell["deep_gate"]:
                        _log(
                            f"  ✓ GATE {fid:22} {tf} RR={rr} n={cell['n']} "
                            f"WR={cell['win_rate']} E[R]={cell['exp_r']} "
                            f"1y={cell['periods'].get('1y',{}).get('exp_r')} "
                            f"3y={cell['periods'].get('3y',{}).get('exp_r')} "
                            f"score={cell['score']}"
                        )
                _progress(
                    {
                        "status": "running",
                        "pct": int(5 + 70 * (pi + 0.5) / n_sym),
                        "message": f"[{sym}] {fid}@{tf} scored={len(scored)}",
                        "symbol": sym,
                        "elapsed_sec": round(time.time() - t0, 1),
                    }
                )

        scored.sort(key=lambda r: float(r.get("score") or -1e9), reverse=True)
        gated = [r for r in scored if r.get("deep_gate")]
        best = (gated or scored or [None])[0]
        per_pair[sym] = {
            "symbol": sym,
            "hist_depth": depth.get(sym),
            "n_scored": len(scored),
            "n_gated": len(gated),
            "winner": best,
            "top10": scored[:10],
            "gated": gated[:15],
        }
        if best:
            _obs(
                f"{sym}.md",
                f"""# {sym} — invented math formula

- **Winner:** `{best.get('id')}` · {best.get('label')}
- **Equation:** ${best.get('equation')}$
- **TF / RR:** {best.get('tf')} / {best.get('rr')}
- **n / WR / E[R]:** {best.get('n')} / {best.get('win_rate')} / {best.get('exp_r')}
- **OOS E[R] / WF / MC:** {best.get('oos_exp_r')} / {best.get('wf_green')} / {best.get('mc')}
- **Deep gate:** {best.get('deep_gate')} · score={best.get('score')}
- **Periods:** `{json.dumps(best.get('periods'), default=str)}`

## Top 5
"""
                + "\n".join(
                    f"{i}. `{c['id']}` @{c['tf']} RR={c['rr']} n={c['n']} "
                    f"WR={c['win_rate']} E[R]={c['exp_r']} gate={c['deep_gate']} "
                    f"eq=`{c['equation']}`"
                    for i, c in enumerate(scored[:5], 1)
                )
                + f"\n\n_Updated {datetime.now(timezone.utc).isoformat()}_\n",
            )
        # checkpoint
        OUT.write_text(
            json.dumps(
                {
                    "status": "running",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "hist_depth": depth,
                    "per_pair": per_pair,
                    "research_only": True,
                    "recommend_live": False,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    # Densify: SL ATR grid + formula param tweaks on leaders until min runtime
    densify = 0
    while time.time() - t0 < MIN_RUNTIME_SEC and time.time() - t0 < MAX_RUNTIME_SEC:
        densify += 1
        _log(f"\n##### DENSIFY {densify} elapsed={time.time()-t0:.0f}s #####")
        for sym in BOOK:
            if time.time() - t0 >= min(MIN_RUNTIME_SEC, MAX_RUNTIME_SEC):
                break
            row = per_pair.get(sym) or {}
            w = row.get("winner")
            if not w:
                continue
            tf = w.get("tf") or "15m"
            # try densify TFs
            for dtf in DENSE_TFS:
                if time.time() - t0 >= min(MIN_RUNTIME_SEC, MAX_RUNTIME_SEC):
                    break
                try:
                    df = _load_enriched(sym, dtf, min(CAP_YEARS, 5.0 if dtf == "3m" else CAP_YEARS))
                except Exception:
                    continue
                if df is None:
                    continue
                # find formula
                fn = None
                label = w.get("label") or ""
                for fid, lab, f in catalog:
                    if fid == w.get("id") or fid.split("_")[0] == str(w.get("id") or "").split("_")[0]:
                        # prefer exact id match
                        if fid == w.get("id"):
                            fn = f
                            label = lab
                            break
                        if fn is None:
                            fn = f
                            label = lab
                if fn is None:
                    # try family prefix
                    pref = str(w.get("id") or "").rsplit("_", 1)[0]
                    for fid, lab, f in catalog:
                        if fid.startswith(pref[:8]):
                            fn = f
                            label = lab
                            break
                if fn is None:
                    continue
                improved = False
                for sl in (1.0, 1.2, 1.5, 1.8):
                    for rr in (2.0, 2.4, 3.0, 4.0):
                        try:
                            cell = _run_formula(df, str(w.get("id")), label, fn, rr, sl_atr=sl)
                        except Exception:
                            continue
                        if not cell:
                            continue
                        cell["tf"] = dtf
                        cell["symbol"] = sym
                        cell["id"] = f"{w.get('id')}_sl{sl:g}_D{densify}"
                        if cell["score"] > float(w.get("score") or -1e9):
                            row["winner"] = cell
                            w = cell
                            improved = True
                            _log(
                                f"  ↑ {sym} → {cell['id']} @{dtf} RR={rr} sl={sl} "
                                f"n={cell['n']} WR={cell['win_rate']} "
                                f"E[R]={cell['exp_r']} score={cell['score']}"
                            )
                if improved:
                    per_pair[sym] = row
                _progress(
                    {
                        "status": "running",
                        "pct": min(99, 75 + densify),
                        "message": f"densify {densify} [{sym} @{dtf}]",
                        "symbol": sym,
                        "elapsed_sec": round(time.time() - t0, 1),
                    }
                )
        if densify >= 40:
            break

    # Equations markdown
    lines = [
        "# Zenith Invented Math Equations",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Elapsed: {(time.time()-t0)/3600:.2f}h · densify rounds: {densify}",
        "Research only — **live brains NOT changed**.",
        "",
        "## Honesty",
        "No formula wins *all* trades. Below are multi-horizon firm survivors "
        "optimized for WR + E[R] + MC + WF across 1y/3y/5y/full.",
        "",
        "## Dukascopy hist depth",
        "```json",
        json.dumps(depth, indent=2),
        "```",
        "",
        "## Per-pair winners",
        "",
    ]
    firm = []
    for sym in BOOK:
        row = per_pair.get(sym) or {}
        w = row.get("winner") or {}
        if not w:
            lines.append(f"### {sym}\n_No survivor_\n")
            continue
        lines += [
            f"### {sym}",
            f"- **ID:** `{w.get('id')}`",
            f"- **Equation:** $${w.get('equation')}$$",
            f"- **TF / RR / SL·ATR:** {w.get('tf')} / {w.get('rr')} / {w.get('sl_atr')}",
            f"- **n / WR% / E[R]:** {w.get('n')} / {w.get('win_rate')} / {w.get('exp_r')}",
            f"- **OOS E[R] / WF / MC%:** {w.get('oos_exp_r')} / {w.get('wf_green')} / {w.get('mc')}",
            f"- **Deep gate:** {w.get('deep_gate')} · score={w.get('score')}",
            f"- **1y E[R]:** {(w.get('periods') or {}).get('1y', {}).get('exp_r')}",
            f"- **3y E[R]:** {(w.get('periods') or {}).get('3y', {}).get('exp_r')}",
            f"- **5y E[R]:** {(w.get('periods') or {}).get('5y', {}).get('exp_r')}",
            f"- **full E[R]:** {(w.get('periods') or {}).get('full', {}).get('exp_r')}",
            "",
        ]
        if w.get("deep_gate"):
            firm.append(sym)

    lines += ["## Firm multi-horizon candidates", ""]
    if firm:
        for s in firm:
            lines.append(f"- **{s}**: `{(per_pair[s].get('winner') or {}).get('id')}`")
    else:
        lines.append("- _None cleared full firm gate — see top scores anyway_")
    lines += [
        "",
        "## Catalog (all invented families)",
        "",
    ]
    seen = set()
    for fid, label, fn in catalog:
        fam = fid.rsplit("_", 1)[0]
        if fam in seen:
            continue
        seen.add(fam)
        # get equation from a probe if needed — store label only
        lines.append(f"- `{fam}*`: {label}")

    EQ.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _obs("00_Math_Invention_Board.md", "\n".join(lines) + "\n")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "recommend_live": False,
        "elapsed_sec": round(time.time() - t0, 1),
        "densify_rounds": densify,
        "min_runtime_sec": MIN_RUNTIME_SEC,
        "max_runtime_sec": MAX_RUNTIME_SEC,
        "cap_years": CAP_YEARS,
        "hist_depth": depth,
        "catalog_size": len(catalog),
        "periods": [p[0] for p in PERIODS],
        "per_pair": per_pair,
        "firm_candidates": [
            {"symbol": s, "winner": (per_pair.get(s) or {}).get("winner")} for s in firm
        ],
        "equations_path": str(EQ),
        "note": "Invented math formulas — do NOT wire until user confirms",
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress(
        {
            "status": "done",
            "pct": 100,
            "message": f"done {(time.time()-t0)/3600:.2f}h firm={len(firm)}",
            "elapsed_sec": round(time.time() - t0, 1),
        }
    )
    _log(f"\nWrote {OUT}\nEquations {EQ}\nFirm: {firm or 'none'}")
    try:
        LOCK.unlink(missing_ok=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
