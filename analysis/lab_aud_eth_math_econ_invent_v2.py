"""AUD/ETH invent catalog v2 — denser math/econ + proven adapters + new algos.

Research-only. No soft overlays. Builds on v1 with many more permutations.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from analysis.lab_adaptive_xm_deep_hunt_bt import (
    mk_circadian,
    mk_entropy,
    mk_fourier_trend,
    mk_ou_fade,
    mk_rolling_edge_gate,
    mk_session_mix,
    mk_snr,
    mk_volmom,
)
from analysis.lab_adaptive_xm_rest5_hunt_bt import (
    mk_dual_agree,
    mk_hurst_gate,
    mk_london_ny_only,
    mk_off_flat,
)
from analysis.lab_aud_eth_math_econ_invent import (
    SideFn,
    build_aud_eth_catalog as build_v1,
    mk_aud_asia_fade_london_trend,
    mk_aud_carry_proxy,
    mk_aud_risk_on,
    mk_aud_session_ou,
    mk_aud_tot_compress,
    mk_dyn_bayes_pair,
    mk_dyn_hurst_switch,
    mk_dyn_vol_regime,
    mk_eth_funding_proxy,
    mk_eth_hurst_adaptive,
    mk_eth_jump_fade,
    mk_eth_spread_thrust,
    mk_eth_vol_cluster,
)
from analysis.lab_zenith_math_invention_deep_bt import _catalog as _zenith_catalog
from analysis.xm_session_costs import spread_price


def _hour(ctx: dict) -> np.ndarray:
    return np.asarray(ctx["cal"]["hour"], dtype=float)


def _wrap_zenith(fid: str, fn) -> SideFn:
    def side_fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sides, _eq = fn(df)
        return np.asarray(sides, dtype=int)

    side_fn.__name__ = f"zenith_{fid}"
    return side_fn


# ── New algorithms ───────────────────────────────────────────────────────────


def mk_ema_pullback(fast: int = 20, slow: int = 50, z: float = 0.8) -> SideFn:
    """Classic trend pullback: EMA stack + fade to EMA then resume."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        c = df["close"].astype(float)
        atr = df["atr"].replace(0, np.nan)
        ema_f = c.ewm(span=fast, adjust=False).mean()
        ema_s = c.ewm(span=slow, adjust=False).mean()
        z_ema = ((c - ema_f) / atr).fillna(0).to_numpy()
        up = (ema_f > ema_s).to_numpy()
        dn = ~up
        hour = _hour(ctx)
        liquid = (hour >= 7) & (hour < 21)
        out = np.zeros(len(df), dtype=int)
        out[liquid & up & (z_ema <= -z) & (z_ema >= -2.5)] = 1
        out[liquid & dn & (z_ema >= z) & (z_ema <= 2.5)] = -1
        return out

    return fn


def mk_breakout_asia_range(look: int = 48, buf: float = 0.15) -> SideFn:
    """London break of Asia range (session econ microstructure)."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        atr = df["atr"].to_numpy(dtype=float)
        hour = _hour(ctx)
        # rolling Asia high/low proxy: prior look bars ending at London open
        hh = pd.Series(h).rolling(look, min_periods=look // 2).max().shift(1).to_numpy()
        ll = pd.Series(l).rolling(look, min_periods=look // 2).min().shift(1).to_numpy()
        out = np.zeros(len(df), dtype=int)
        lon = (hour >= 8) & (hour < 16)
        buf_px = buf * np.maximum(atr, 1e-12)
        out[lon & (c > hh + buf_px)] = 1
        out[lon & (c < ll - buf_px)] = -1
        return out

    return fn


def mk_bb_mr(win: int = 40, k: float = 2.0) -> SideFn:
    """Bollinger mean-revert with liquid-hours filter."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        c = df["close"].astype(float)
        m = c.rolling(win).mean()
        s = c.rolling(win).std().replace(0, np.nan)
        z = ((c - m) / s).fillna(0).to_numpy()
        hour = _hour(ctx)
        liquid = (hour >= 7) & (hour < 21)
        out = np.zeros(len(df), dtype=int)
        out[liquid & (z <= -k)] = 1
        out[liquid & (z >= k)] = -1
        return out

    return fn


def mk_cusum_clock(thr: float = 0.015) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        cs = df["cusum_slope"].to_numpy(dtype=float)
        hour = _hour(ctx)
        out = np.zeros(len(df), dtype=int)
        liquid = (hour >= 7) & (hour < 21)
        out[liquid & (cs >= thr)] = 1
        out[liquid & (cs <= -thr)] = -1
        return out

    return fn


def mk_rsi_mr(period: int = 14, lo: float = 30.0, hi: float = 70.0) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        c = df["close"].astype(float)
        d = c.diff()
        gain = d.clip(lower=0).rolling(period).mean()
        loss = (-d.clip(upper=0)).rolling(period).mean().replace(0, np.nan)
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).fillna(50).to_numpy()
        hour = _hour(ctx)
        liquid = (hour >= 7) & (hour < 21)
        out = np.zeros(len(df), dtype=int)
        out[liquid & (rsi <= lo)] = 1
        out[liquid & (rsi >= hi)] = -1
        return out

    return fn


def mk_donchian_trend(look: int = 55) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        h = df["high"].rolling(look).max().shift(1)
        l = df["low"].rolling(look).min().shift(1)
        c = df["close"]
        hour = _hour(ctx)
        liquid = (hour >= 7) & (hour < 21)
        out = np.zeros(len(df), dtype=int)
        out[liquid & (c > h).fillna(False).to_numpy()] = 1
        out[liquid & (c < l).fillna(False).to_numpy()] = -1
        return out

    return fn


def mk_eth_atr_expand(mult: float = 1.2, snr_min: float = 0.30) -> SideFn:
    """ETH: ATR expanding vs median → trend (vol regime econ)."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        atr = df["atr"].to_numpy(dtype=float)
        med = pd.Series(atr).rolling(100, min_periods=40).median().to_numpy()
        ds = df["drift_sign"].to_numpy(dtype=float)
        snr = df["snr"].to_numpy(dtype=float)
        m = (atr >= mult * np.maximum(med, 1e-12)) & (snr >= snr_min) & (ds != 0)
        out = np.zeros(len(df), dtype=int)
        out[m] = ds[m].astype(int)
        return out

    return fn


def mk_aud_london_open_impulse(bars: int = 6, z: float = 0.9) -> SideFn:
    """AUD: first N London bars impulse continuation / fade extremes."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        hour = _hour(ctx)
        # minutes within hour rough via index — use hour==8 first bars via rolling open
        c = df["close"].to_numpy(dtype=float)
        o = df["open"].to_numpy(dtype=float)
        atr = df["atr"].to_numpy(dtype=float)
        move = (c - o) / np.maximum(atr, 1e-12)
        # mark London open window hour 8
        lon0 = (hour >= 8) & (hour < 9)
        # extend impulse for next bars via rolling max abs move at open
        impulse = pd.Series(np.where(lon0, move, np.nan)).ffill(limit=bars).fillna(0).to_numpy()
        snr = df["snr"].to_numpy(dtype=float)
        out = np.zeros(len(df), dtype=int)
        cont = (hour >= 8) & (hour < 12) & (np.abs(impulse) >= z) & (snr >= 0.25)
        out[cont & (impulse > 0)] = 1
        out[cont & (impulse < 0)] = -1
        return out

    return fn


def mk_triple_agree(a: SideFn, b: SideFn, c: SideFn) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sa, sb, sc = a(df, ctx), b(df, ctx), c(df, ctx)
        out = np.zeros(len(df), dtype=int)
        m = (sa != 0) & (sa == sb) & (sa == sc)
        out[m] = sa[m]
        return out

    return fn


def build_aud_eth_catalog_v2(
    *, lean: bool = False, symbol: Optional[str] = None, densify: int = 0
) -> list[tuple[str, str, SideFn]]:
    """Large permutation catalog. densify=0..2 adds more param variants."""
    out: list[tuple[str, str, SideFn]] = []

    # v1 pair-specific
    out.extend(build_v1(lean=False, symbol=symbol))

    # Proven adapters (Rest-5 / deep) — need static in ctx for some
    sm_grid = [(0.25, 0.40), (0.28, 0.38), (0.35, 0.45), (0.40, 0.50)]
    if densify >= 1:
        sm_grid += [(0.22, 0.35), (0.30, 0.42), (0.32, 0.48)]
    for tb, to in sm_grid:
        base = mk_session_mix(tb, to)
        out.append((f"session_mix_t{tb:.2f}_o{to:.2f}", "session_mix", base))
        out.append((f"offflat_sm_t{tb:.2f}", "offflat session", mk_off_flat(base)))
        if densify >= 1:
            out.append((f"lonny_sm_t{tb:.2f}", "london-ny session", mk_london_ny_only(base)))

    for thr, lb in ((0.35, 20), (0.5, 20), (0.5, 30), (0.65, 20), (0.4, 15)):
        vm = mk_volmom(thr, lb)
        out.append((f"volmom_t{thr}_l{lb}", "volmom", vm))
        out.append((f"offflat_volmom_t{thr}", "offflat volmom", mk_off_flat(vm)))
        if densify >= 1:
            out.append(
                (f"hurst_volmom_t{thr}", "hurst×volmom", mk_hurst_gate(vm, mode="trend", thr=0.52))
            )

    for z0, zs in ((1.8, 0.6), (2.0, 0.6), (2.4, 0.5), (2.8, 0.4)):
        out.append((f"circadian_z{z0}_s{zs}", "circadian", mk_circadian(z0, zs)))
        out.append((f"offflat_cir_z{z0}", "offflat circadian", mk_off_flat(mk_circadian(z0, zs))))

    for p0, sn in ((0.68, 0.2), (0.72, 0.22), (0.80, 0.15)):
        out.append((f"entropy_p{p0}_s{sn}", "entropy", mk_entropy(p0, sn)))

    for thr in (0.30, 0.40, 0.50):
        out.append((f"snr_{thr}", "snr drift", mk_snr(thr)))

    for z in (1.4, 1.8, 2.2):
        out.append((f"ou_fade_{z}", "OU fade", mk_ou_fade(z)))

    for lf, sn in ((0.55, 0.30), (0.60, 0.35), (0.65, 0.40)):
        out.append((f"fourier_lf{lf}_s{sn}", "fourier trend", mk_fourier_trend(lf, sn)))

    # Dual agrees
    out.append(
        (
            "agree_sm_volmom",
            "dual agree",
            mk_dual_agree(mk_session_mix(0.28, 0.38), mk_volmom(0.5, 20)),
        )
    )
    out.append(
        (
            "agree_snr_ou",
            "dual agree",
            mk_dual_agree(mk_snr(0.40), mk_ou_fade(1.8)),
        )
    )
    out.append(
        (
            "agree_aud_risk_sm",
            "dual agree",
            mk_dual_agree(mk_aud_risk_on(0.32), mk_session_mix(0.35, 0.45)),
        )
    )
    out.append(
        (
            "agree_eth_cluster_sm",
            "dual agree",
            mk_dual_agree(mk_eth_vol_cluster(), mk_session_mix(0.28, 0.38)),
        )
    )

    # New algos
    for f, s, z in ((12, 36, 0.7), (20, 50, 0.8), (20, 50, 1.1), (8, 34, 0.6)):
        out.append((f"ema_pb_{f}_{s}_z{z}", "EMA pullback", mk_ema_pullback(f, s, z)))
    for look, buf in ((36, 0.1), (48, 0.15), (64, 0.2), (96, 0.12)):
        out.append((f"asia_brk_l{look}_b{buf}", "Asia range break", mk_breakout_asia_range(look, buf)))
    for win, k in ((30, 1.8), (40, 2.0), (50, 2.2), (40, 2.5)):
        out.append((f"bb_mr_w{win}_k{k}", "BB mean-revert", mk_bb_mr(win, k)))
    for thr in (0.01, 0.015, 0.02, 0.025):
        out.append((f"cusum_{thr}", "CUSUM clock", mk_cusum_clock(thr)))
    for lo, hi in ((25, 75), (30, 70), (35, 65)):
        out.append((f"rsi_mr_{int(lo)}_{int(hi)}", "RSI MR", mk_rsi_mr(14, lo, hi)))
    for look in (40, 55, 80):
        out.append((f"donchian_{look}", "Donchian trend", mk_donchian_trend(look)))

    # ETH/AUD specialized densify
    for m, s in ((1.15, 0.28), (1.25, 0.30), (1.40, 0.35)):
        out.append((f"eth_atr_exp_{m}_{s}", "ETH ATR expand", mk_eth_atr_expand(m, s)))
    for bars, z in ((4, 0.8), (6, 0.9), (8, 1.1)):
        out.append((f"aud_lon_impulse_b{bars}_z{z}", "AUD London impulse", mk_aud_london_open_impulse(bars, z)))

    # Edge-gated winners
    out.append(
        (
            "edge_volmom",
            "edge×volmom",
            mk_rolling_edge_gate(mk_volmom(0.5, 20), 400, 0.0),
        )
    )
    out.append(
        (
            "edge_session",
            "edge×session",
            mk_rolling_edge_gate(mk_session_mix(0.35, 0.45), 500, 0.0),
        )
    )
    out.append(
        (
            "edge_eth_thrust",
            "edge×ETH thrust",
            mk_rolling_edge_gate(mk_eth_spread_thrust(12.0), 450, 0.0),
        )
    )

    # Adaptive composites densify
    out.append(
        (
            "dyn_sm_volmom_hurst",
            "adaptive Hurst sm↔volmom",
            mk_dyn_hurst_switch(mk_session_mix(0.35, 0.45), mk_volmom(0.5), 0.52),
        )
    )
    out.append(
        (
            "dyn_bayes_sm_volmom",
            "bayes session vs volmom",
            mk_dyn_bayes_pair(mk_session_mix(0.28, 0.38), mk_volmom(0.5), 400, 0.52),
        )
    )
    out.append(
        (
            "dyn_vol_ema_bb",
            "vol regime EMA↔BB",
            mk_dyn_vol_regime(mk_ema_pullback(20, 50, 0.8), mk_bb_mr(40, 2.0), 1.05),
        )
    )
    out.append(
        (
            "triple_sm_snr_ou",
            "triple agree",
            mk_triple_agree(mk_session_mix(0.28, 0.38), mk_snr(0.40), mk_ou_fade(1.8)),
        )
    )

    # Zenith math subset
    z_keep = {
        "snr_drift_0.40",
        "snr_drift_0.55",
        "hurst_trend_0.55",
        "hurst_trend_0.58",
        "hurst_mr_0.45",
        "ou_snap_5_40",
        "ou_snap_8_25",
        "kalman_fade_1.6",
        "kalman_fade_2.0",
        "kalman_fade_2.4",
        "cusum_0.015",
        "vol_compress_release",
        "skew_align_snr",
        "eff_lf_0.25",
        "bimod_0.55",
        "allan_persist_0.95",
    }
    if densify >= 1:
        z_keep |= {
            "snr_drift_0.30",
            "spec_ent_0.50",
            "higuchi_trend_1.45",
            "hf_spike_fade_0.62",
            "perm_order_0.80",
            "fractal_mr",
        }
    for fid, lab, fn in _zenith_catalog():
        if fid in z_keep:
            out.append((f"math_{fid}", f"zenith:{lab}", _wrap_zenith(fid, fn)))

    # More AUD/ETH param permutations
    if densify >= 1:
        for thr in (0.00008, 0.00015, 0.00022):
            out.append((f"aud_carry_{thr}", "AUD carry densify", mk_aud_carry_proxy(thr)))
        for snr, bim in ((0.28, 0.40), (0.36, 0.48), (0.45, 0.55)):
            out.append((f"aud_riskon_{snr}_{bim}", "AUD risk densify", mk_aud_risk_on(snr, bim)))
        for jz, fz in ((2.2, 1.0), (3.0, 1.2), (3.5, 1.4)):
            out.append((f"eth_jump_{jz}_{fz}", "ETH jump densify", mk_eth_jump_fade(jz, fz)))
        for sm in (10.0, 14.0, 18.0):
            out.append((f"eth_spread_m{sm}", "ETH spread densify", mk_eth_spread_thrust(sm, 0.32)))

    if densify >= 2:
        for tb, to in ((0.20, 0.32), (0.38, 0.52), (0.45, 0.55)):
            out.append((f"session_mix_d2_t{tb}_o{to}", "session densify2", mk_session_mix(tb, to)))
        for thr in (0.25, 0.55, 0.75):
            out.append((f"volmom_d2_t{thr}", "volmom densify2", mk_volmom(thr, 25)))
        out.append(
            (
                "dyn_bayes_aud_eth_style",
                "bayes asia-lon vs risk",
                mk_dyn_bayes_pair(
                    mk_aud_asia_fade_london_trend(1.8, 0.32),
                    mk_aud_session_ou(6, 30, 1.3),
                    350,
                    0.51,
                ),
            )
        )
        out.append(
            (
                "dyn_eth_jump_fund",
                "bayes jump vs funding",
                mk_dyn_bayes_pair(mk_eth_jump_fade(), mk_eth_funding_proxy(100, 0.01), 400, 0.52),
            )
        )

    # Filter by symbol family preference (still allow shared adapters)
    sym = (symbol or "").upper()
    if lean:
        # keep ~half: prioritize shared proven + a few pair-specific
        prefer_prefix = ()
        if sym == "AUDUSD":
            prefer_prefix = ("aud_", "dyn_aud", "session_", "volmom", "offflat", "agree_", "ema_", "asia_", "math_")
        elif sym == "ETHUSD":
            prefer_prefix = ("eth_", "dyn_eth", "session_", "volmom", "offflat", "agree_", "math_", "fourier", "edge_")
        else:
            prefer_prefix = ("session_", "volmom", "aud_", "eth_", "dyn_", "math_")
        filtered = []
        for vid, lab, fn in out:
            if any(vid.startswith(p) or p.rstrip("_") in vid for p in prefer_prefix):
                filtered.append((vid, lab, fn))
            if len(filtered) >= 28:
                break
        # dedupe by id
        seen = set()
        lean_out = []
        for vid, lab, fn in filtered:
            if vid in seen:
                continue
            seen.add(vid)
            lean_out.append((vid, lab, fn))
        return lean_out

    # dedupe
    seen: set[str] = set()
    uniq: list[tuple[str, str, SideFn]] = []
    for vid, lab, fn in out:
        if vid in seen:
            continue
        seen.add(vid)
        # symbol filter: drop opposite-pair-only prefixes
        if sym == "AUDUSD" and vid.startswith("eth_"):
            continue
        if sym == "ETHUSD" and vid.startswith("aud_"):
            continue
        uniq.append((vid, lab, fn))
    return uniq
