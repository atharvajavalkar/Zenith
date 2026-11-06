"""AUDUSD + ETHUSD math/econ invention catalog (research-only).

Pure SideFn formulas from OHLCV proxies — no autopsy soft overlays.
Includes static equations and dynamic/adaptive routers (regime/session/bayes
inside the equation).

Economics (price-implied, no external feeds required):
  AUD — commodity FX / risk-on: session microstructure, Off cost, carry-ish
        overnight drift, terms-of-trade vol-compress release.
  ETH — crypto microstructure: jump fade, vol clustering, funding-proxy skew,
        spread-aware thrust (ATR vs XM spread economics).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from analysis.xm_session_costs import spread_price

SideFn = Callable[[pd.DataFrame, dict[str, Any]], np.ndarray]


def _sign_thr(score: np.ndarray, thr: float) -> np.ndarray:
    out = np.zeros(len(score), dtype=int)
    out[score >= thr] = 1
    out[score <= -thr] = -1
    return out


def _hour(ctx: dict) -> np.ndarray:
    return np.asarray(ctx["cal"]["hour"], dtype=float)


# ── AUDUSD economics ─────────────────────────────────────────────────────────


def mk_aud_carry_proxy(thr: float = 0.00012) -> SideFn:
    """Overnight carry proxy: Asia open drift persistence into London."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        c = df["close"].to_numpy(dtype=float)
        hour = _hour(ctx)
        atr = df["atr"].to_numpy(dtype=float)
        med = float(np.nanmedian(atr[atr > 0])) if np.any(atr > 0) else 1e-4
        nbar = 80 if med < 0.01 else 40
        prev = np.roll(c, nbar)
        overnight = (c - prev) / np.maximum(prev, 1e-12)
        overnight[:nbar] = 0.0
        ds = df["drift_sign"].to_numpy(dtype=float)
        snr = df["snr"].to_numpy(dtype=float)
        liquid = (hour >= 7) & (hour < 21)
        score = overnight / max(thr, 1e-8) + 0.35 * ds * snr
        out = _sign_thr(score, 0.85)
        out[~liquid] = 0
        return out

    return fn


def mk_aud_risk_on(snr_min: float = 0.32, bimod_min: float = 0.45) -> SideFn:
    """Risk-on commodity FX: trend when SNR+bimod, flat Off."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        hour = _hour(ctx)
        snr = df["snr"].to_numpy(dtype=float)
        bim = df["bimod"].to_numpy(dtype=float)
        ds = df["drift_sign"].to_numpy(dtype=float)
        liquid = (hour >= 7) & (hour < 21)
        m = liquid & (snr >= snr_min) & (bim >= bimod_min) & (ds != 0)
        out = np.zeros(len(df), dtype=int)
        out[m] = ds[m].astype(int)
        return out

    return fn


def mk_aud_tot_compress(vr_lo: float = 0.82, vr_hi: float = 1.08) -> SideFn:
    """Terms-of-trade proxy: vol compress→release with drift."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        hour = _hour(ctx)
        vr = df["vol_ratio"].to_numpy(dtype=float)
        vr_prev = np.roll(vr, 4)
        ds = df["drift_sign"].to_numpy(dtype=float)
        fire = (vr_prev < vr_lo) & (vr >= vr_hi)
        liquid = (hour >= 7) & (hour < 21)
        out = np.zeros(len(df), dtype=int)
        m = fire & liquid & (ds != 0)
        out[m] = ds[m].astype(int)
        return out

    return fn


def mk_aud_session_ou(hl_lo: float = 6.0, hl_hi: float = 35.0, z: float = 1.25) -> SideFn:
    """OU snapback only in London/NY — Off excluded (XM Off cost)."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        hour = _hour(ctx)
        hl = df["ou_hl"].to_numpy(dtype=float)
        kz = df["kalman_z"].to_numpy(dtype=float)
        sess = (hour >= 8) & (hour < 21)
        m = sess & (hl >= hl_lo) & (hl <= hl_hi)
        out = np.zeros(len(df), dtype=int)
        out[m & (kz <= -z)] = 1
        out[m & (kz >= z)] = -1
        return out

    return fn


def mk_aud_asia_fade_london_trend(z: float = 1.6, snr_min: float = 0.30) -> SideFn:
    """Adaptive session book: Asia fade extremes; London/NY SNR trend."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        hour = _hour(ctx)
        kz = df["kalman_z"].to_numpy(dtype=float)
        snr = df["snr"].to_numpy(dtype=float)
        ds = df["drift_sign"].to_numpy(dtype=float)
        out = np.zeros(len(df), dtype=int)
        asia = hour < 8
        lon_ny = (hour >= 8) & (hour < 21)
        out[asia & (kz <= -z)] = 1
        out[asia & (kz >= z)] = -1
        m = lon_ny & (snr >= snr_min) & (ds != 0)
        out[m] = ds[m].astype(int)
        out[hour >= 21] = 0
        return out

    return fn


# ── ETHUSD economics ─────────────────────────────────────────────────────────


def mk_eth_jump_fade(jump_z: float = 2.8, fade_z: float = 1.1) -> SideFn:
    """Jump-diffusion: after |ret_z| spike, fade Kalman residual."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        rz = df["ret_z"].to_numpy(dtype=float)
        kz = df["kalman_z"].to_numpy(dtype=float)
        j = np.maximum.reduce(
            [np.abs(rz), np.abs(np.roll(rz, 1)), np.abs(np.roll(rz, 2))]
        )
        j[:2] = 0.0
        jumped = j >= jump_z
        out = np.zeros(len(df), dtype=int)
        out[jumped & (kz <= -fade_z)] = 1
        out[jumped & (kz >= fade_z)] = -1
        return out

    return fn


def mk_eth_vol_cluster(vv_min: float = 1.25, snr_min: float = 0.28) -> SideFn:
    """Vol-of-vol clustering: trend when short/long σ elevated."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        vr = df["vol_ratio"].to_numpy(dtype=float)
        snr = df["snr"].to_numpy(dtype=float)
        ds = df["drift_sign"].to_numpy(dtype=float)
        m = (vr >= vv_min) & (snr >= snr_min) & (ds != 0)
        out = np.zeros(len(df), dtype=int)
        out[m] = ds[m].astype(int)
        return out

    return fn


def mk_eth_funding_proxy(look: int = 120, thr: float = 0.012) -> SideFn:
    """Funding proxy: cumulative return skew — fade stretched flow."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        ret = df["ret"].to_numpy(dtype=float)
        s = (
            pd.Series(ret)
            .rolling(look, min_periods=max(20, look // 2))
            .sum()
            .fillna(0)
            .to_numpy()
        )
        kz = df["kalman_z"].to_numpy(dtype=float)
        out = np.zeros(len(df), dtype=int)
        out[(s >= thr) & (kz >= 0.8)] = -1
        out[(s <= -thr) & (kz <= -0.8)] = 1
        return out

    return fn


def mk_eth_spread_thrust(spread_mult: float = 12.0, snr_min: float = 0.35) -> SideFn:
    """Thrust only when ATR >> XM spread (ETH ~$4.8 economics)."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sym = str(ctx.get("sym") or "ETHUSD")
        atr = df["atr"].to_numpy(dtype=float)
        hour = _hour(ctx)
        # vectorized-ish: unique hours then map (faster than per-bar python loop)
        sp = np.empty(len(hour), dtype=float)
        cache: dict[int, float] = {}
        for i, h in enumerate(hour):
            key = int(h) % 24
            if key not in cache:
                cache[key] = float(spread_price(sym, hour_utc=float(key)))
            sp[i] = cache[key]
        ok = atr >= (spread_mult * np.maximum(sp, 1e-9))
        snr = df["snr"].to_numpy(dtype=float)
        ds = df["drift_sign"].to_numpy(dtype=float)
        lf = df["fourier_lf"].to_numpy(dtype=float)
        m = ok & (snr >= snr_min) & (lf >= 0.55) & (ds != 0)
        out = np.zeros(len(df), dtype=int)
        out[m] = ds[m].astype(int)
        return out

    return fn


def mk_eth_hurst_adaptive(h_hi: float = 0.55, h_lo: float = 0.45, z: float = 1.5) -> SideFn:
    """Dynamic: Hurst high → trend; Hurst low → Kalman fade."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        H = df["hurst"].to_numpy(dtype=float)
        ds = df["drift_sign"].to_numpy(dtype=float)
        kz = df["kalman_z"].to_numpy(dtype=float)
        snr = df["snr"].to_numpy(dtype=float)
        out = np.zeros(len(df), dtype=int)
        trend = (H >= h_hi) & (snr >= 0.28) & (ds != 0)
        out[trend] = ds[trend].astype(int)
        fade = H <= h_lo
        out[fade & (kz <= -z)] = 1
        out[fade & (kz >= z)] = -1
        return out

    return fn


# ── Adaptive routers (dynamic, inside equation — not soft gates) ─────────────


def mk_dyn_hurst_switch(trend_fn: SideFn, fade_fn: SideFn, h_cut: float = 0.50) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        H = df["hurst"].to_numpy(dtype=float)
        a = trend_fn(df, ctx)
        b = fade_fn(df, ctx)
        return np.asarray(np.where(H >= h_cut, a, b), dtype=int)

    return fn


def mk_dyn_bayes_pair(
    a_fn: SideFn, b_fn: SideFn, look: int = 400, prior: float = 0.52
) -> SideFn:
    """Online rolling hit-rate pick between two child equations."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        ret = df["ret"].to_numpy(dtype=float)
        nxt = np.roll(ret, -1)
        nxt[-1] = 0.0
        sa = a_fn(df, ctx).astype(float)
        sb = b_fn(df, ctx).astype(float)
        ha = (sa * nxt) > 0
        hb = (sb * nxt) > 0
        ra = (
            pd.Series(ha.astype(float))
            .shift(1)
            .rolling(look, min_periods=max(40, look // 3))
            .mean()
            .fillna(prior)
            .to_numpy()
        )
        rb = (
            pd.Series(hb.astype(float))
            .shift(1)
            .rolling(look, min_periods=max(40, look // 3))
            .mean()
            .fillna(prior)
            .to_numpy()
        )
        pick_a = ra >= rb
        out = np.where(pick_a, sa, sb)
        post = np.where(pick_a, ra, rb)
        out = np.where(post >= prior, out, 0)
        return np.asarray(out, dtype=int)

    return fn


def mk_dyn_vol_regime(trend_fn: SideFn, fade_fn: SideFn, vr_cut: float = 1.05) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        vr = df["vol_ratio"].to_numpy(dtype=float)
        a = trend_fn(df, ctx)
        b = fade_fn(df, ctx)
        return np.asarray(np.where(vr >= vr_cut, a, b), dtype=int)

    return fn


def build_aud_eth_catalog(
    *, lean: bool = False, symbol: Optional[str] = None
) -> list[tuple[str, str, SideFn]]:
    """Return (id, label, SideFn). Filter by symbol when provided."""
    aud: list[tuple[str, str, SideFn]] = [
        ("aud_carry_0.00012", "AUD overnight carry proxy", mk_aud_carry_proxy(0.00012)),
        ("aud_carry_0.00018", "AUD overnight carry proxy", mk_aud_carry_proxy(0.00018)),
        ("aud_riskon_0.32", "AUD risk-on SNR×bimod", mk_aud_risk_on(0.32, 0.45)),
        ("aud_riskon_0.40", "AUD risk-on SNR×bimod", mk_aud_risk_on(0.40, 0.50)),
        ("aud_tot_compress", "AUD terms-of-trade compress", mk_aud_tot_compress()),
        ("aud_session_ou", "AUD London/NY OU snap", mk_aud_session_ou()),
        ("aud_session_ou_tight", "AUD London/NY OU snap", mk_aud_session_ou(8, 28, 1.45)),
        ("aud_asia_fade_lon_trend", "AUD Asia fade / Lon trend", mk_aud_asia_fade_london_trend()),
        ("aud_asia_fade_lon_z2", "AUD Asia fade / Lon trend", mk_aud_asia_fade_london_trend(2.0, 0.35)),
    ]
    eth: list[tuple[str, str, SideFn]] = [
        ("eth_jump_fade", "ETH jump-diffusion fade", mk_eth_jump_fade()),
        ("eth_jump_fade_soft", "ETH jump-diffusion fade", mk_eth_jump_fade(2.4, 1.0)),
        ("eth_vol_cluster", "ETH vol-cluster trend", mk_eth_vol_cluster()),
        ("eth_vol_cluster_hi", "ETH vol-cluster trend", mk_eth_vol_cluster(1.40, 0.32)),
        ("eth_funding_proxy", "ETH funding-proxy fade", mk_eth_funding_proxy()),
        ("eth_funding_proxy_fast", "ETH funding-proxy fade", mk_eth_funding_proxy(80, 0.008)),
        ("eth_spread_thrust", "ETH spread-aware thrust", mk_eth_spread_thrust(12.0, 0.35)),
        ("eth_spread_thrust_16", "ETH spread-aware thrust", mk_eth_spread_thrust(16.0, 0.40)),
        ("eth_hurst_adaptive", "ETH Hurst adaptive trend/fade", mk_eth_hurst_adaptive()),
        ("eth_hurst_adaptive_wide", "ETH Hurst adaptive", mk_eth_hurst_adaptive(0.52, 0.48, 1.3)),
    ]

    dyn_aud: list[tuple[str, str, SideFn]] = [
        (
            "dyn_aud_hurst_risk_ou",
            "AUD adaptive Hurst: risk-on ↔ OU",
            mk_dyn_hurst_switch(mk_aud_risk_on(0.32), mk_aud_session_ou(), 0.52),
        ),
        (
            "dyn_aud_vol_carry_tot",
            "AUD adaptive vol: carry ↔ ToT",
            mk_dyn_vol_regime(mk_aud_carry_proxy(), mk_aud_tot_compress(), 1.05),
        ),
        (
            "dyn_aud_bayes_risk_session",
            "AUD bayes: risk-on vs Asia/Lon",
            mk_dyn_bayes_pair(
                mk_aud_risk_on(0.32), mk_aud_asia_fade_london_trend(), 400, 0.52
            ),
        ),
    ]
    dyn_eth: list[tuple[str, str, SideFn]] = [
        (
            "dyn_eth_hurst_jump_cluster",
            "ETH adaptive Hurst: jump ↔ cluster",
            mk_dyn_hurst_switch(mk_eth_vol_cluster(), mk_eth_jump_fade(), 0.52),
        ),
        (
            "dyn_eth_vol_thrust_fund",
            "ETH adaptive vol: thrust ↔ funding",
            mk_dyn_vol_regime(mk_eth_spread_thrust(), mk_eth_funding_proxy(), 1.10),
        ),
        (
            "dyn_eth_bayes_hurst_spread",
            "ETH bayes: hurst-adapt vs spread-thrust",
            mk_dyn_bayes_pair(
                mk_eth_hurst_adaptive(), mk_eth_spread_thrust(), 450, 0.53
            ),
        ),
    ]

    if lean:
        aud = aud[:5]
        eth = eth[:5]
        dyn_aud = dyn_aud[:2]
        dyn_eth = dyn_eth[:2]

    sym = (symbol or "").upper()
    if sym == "AUDUSD":
        return aud + dyn_aud
    if sym == "ETHUSD":
        return eth + dyn_eth
    return aud + eth + dyn_aud + dyn_eth


def equations_md() -> str:
    return """# AUD / ETH math–econ invention equations

## AUDUSD
- Carry proxy: overnight Δclose + SNR·sign(μ); flat Off
- Risk-on: 1[SNR∧bimod]·sign(μ); liquid hours
- ToT compress: σ20/σ60 compress→release ∧ trend
- Session OU: t½^OU band ∧ fade z_kal; London–NY
- Asia fade / Lon trend: session-adaptive book

## ETHUSD
- Jump fade: |ret_z| spike → fade z_kal
- Vol cluster: σ-ratio ∧ SNR → trend
- Funding proxy: rolling Σr stretch → fade
- Spread thrust: ATR ≥ m·XM_spread ∧ LF∧SNR
- Hurst adaptive: H high trend / H low fade

## Dynamic
- Hurst switch / vol-regime switch / rolling Bayes between child eqs
"""
