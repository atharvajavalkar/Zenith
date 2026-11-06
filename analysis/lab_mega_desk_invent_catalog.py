"""Full-desk mega invent catalog — 400–600+ math/econ/adaptive strategies.

Research-only. Covers GOLD BTCUSD UKOIL NAS100 AUDUSD ETHUSD USDJPY USDCHF.
Dense param grids of session/volmom/OU/Zenith/EMA/BB/RSI/Donchian/CUSUM/
pair-econ + adaptive Bayes/Hurst/vol routers. No soft overlays.
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
from analysis.lab_aud_eth_math_econ_invent_v2 import (
    mk_aud_london_open_impulse,
    mk_bb_mr,
    mk_breakout_asia_range,
    mk_cusum_clock,
    mk_donchian_trend,
    mk_ema_pullback,
    mk_eth_atr_expand,
    mk_rsi_mr,
    mk_triple_agree,
)
from analysis.lab_zenith_math_invention_deep_bt import _catalog as _zenith_catalog

BOOK = (
    "GOLD",
    "BTCUSD",
    "UKOIL",
    "NAS100",
    "AUDUSD",
    "ETHUSD",
    "USDJPY",
    "USDCHF",
)


def _wrap_zenith(fid: str, fn) -> SideFn:
    def side_fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sides, _eq = fn(df)
        return np.asarray(sides, dtype=int)

    return side_fn


def _hour(ctx: dict) -> np.ndarray:
    return np.asarray(ctx["cal"]["hour"], dtype=float)


def mk_macd_cross(fast: int = 12, slow: int = 26, sig: int = 9) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        c = df["close"].astype(float)
        ef = c.ewm(span=fast, adjust=False).mean()
        es = c.ewm(span=slow, adjust=False).mean()
        macd = ef - es
        signal = macd.ewm(span=sig, adjust=False).mean()
        hist = (macd - signal).to_numpy()
        prev = np.roll(hist, 1)
        prev[0] = 0
        out = np.zeros(len(df), dtype=int)
        hour = _hour(ctx)
        liquid = (hour >= 7) & (hour < 21) if str(ctx.get("sym") or "") not in (
            "BTCUSD",
            "ETHUSD",
        ) else np.ones(len(df), dtype=bool)
        out[liquid & (prev <= 0) & (hist > 0)] = 1
        out[liquid & (prev >= 0) & (hist < 0)] = -1
        return out

    return fn


def mk_supertrend_proxy(mult: float = 2.5, win: int = 10) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        c = df["close"].astype(float)
        atr = df["atr"].astype(float)
        hl2 = (df["high"].astype(float) + df["low"].astype(float)) / 2.0
        upper = (hl2 + mult * atr).to_numpy()
        lower = (hl2 - mult * atr).to_numpy()
        cn = c.to_numpy()
        out = np.zeros(len(df), dtype=int)
        trend = 1
        for i in range(1, len(df)):
            if cn[i] > upper[i - 1]:
                trend = 1
            elif cn[i] < lower[i - 1]:
                trend = -1
            out[i] = trend
        # only fire on flip
        flip = np.zeros(len(df), dtype=int)
        prev = np.roll(out, 1)
        flip[(out == 1) & (prev == -1)] = 1
        flip[(out == -1) & (prev == 1)] = -1
        return flip

    return fn


def mk_keltner_break(win: int = 20, mult: float = 1.5) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        c = df["close"].astype(float)
        atr = df["atr"].astype(float)
        mid = c.ewm(span=win, adjust=False).mean()
        up = (mid + mult * atr).to_numpy()
        dn = (mid - mult * atr).to_numpy()
        cn = c.to_numpy()
        out = np.zeros(len(df), dtype=int)
        out[cn > up] = 1
        out[cn < dn] = -1
        return out

    return fn


def mk_zscore_mr(win: int = 40, z: float = 1.8) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        c = df["close"].astype(float)
        m = c.rolling(win).mean()
        s = c.rolling(win).std().replace(0, np.nan)
        zz = ((c - m) / s).fillna(0).to_numpy()
        out = np.zeros(len(df), dtype=int)
        out[zz <= -z] = 1
        out[zz >= z] = -1
        return out

    return fn


def mk_gold_session_thrust(snr_min: float = 0.35) -> SideFn:
    """GOLD: London/NY SNR thrust, flat Asia thin / Off."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        hour = _hour(ctx)
        snr = df["snr"].to_numpy(dtype=float)
        ds = df["drift_sign"].to_numpy(dtype=float)
        liquid = (hour >= 8) & (hour < 21)
        m = liquid & (snr >= snr_min) & (ds != 0)
        out = np.zeros(len(df), dtype=int)
        out[m] = ds[m].astype(int)
        return out

    return fn


def mk_btc_funding_fade(look: int = 96, thr: float = 0.02) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        ret = df["ret"].to_numpy(dtype=float)
        s = pd.Series(ret).rolling(look, min_periods=look // 2).sum().fillna(0).to_numpy()
        kz = df["kalman_z"].to_numpy(dtype=float)
        out = np.zeros(len(df), dtype=int)
        out[(s >= thr) & (kz >= 0.9)] = -1
        out[(s <= -thr) & (kz <= -0.9)] = 1
        return out

    return fn


def mk_nas_rth_only(base: SideFn) -> SideFn:
    """NAS: only 13:30–20:00 UTC approx US cash hours."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sides = base(df, ctx)
        hour = _hour(ctx)
        # rough RTH: 14–20 UTC
        out = sides.copy()
        out[(hour < 14) | (hour >= 20)] = 0
        return out

    return fn


def mk_oil_inventory_proxy(vr_lo: float = 0.85, vr_hi: float = 1.1) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        vr = df["vol_ratio"].to_numpy(dtype=float)
        vr_prev = np.roll(vr, 5)
        ds = df["drift_sign"].to_numpy(dtype=float)
        hour = _hour(ctx)
        fire = (vr_prev < vr_lo) & (vr >= vr_hi)
        liquid = (hour >= 8) & (hour < 21)
        out = np.zeros(len(df), dtype=int)
        m = fire & liquid & (ds != 0)
        out[m] = ds[m].astype(int)
        return out

    return fn


def build_mega_catalog(
    *, lean: bool = False, symbol: Optional[str] = None
) -> list[tuple[str, str, SideFn]]:
    """Build ≥400 unique modes (lean ≈ half for 1m)."""
    out: list[tuple[str, str, SideFn]] = []

    # ── Session mix dense grid ──────────────────────────────────────────
    for tb in (0.18, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45):
        for to in (0.32, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55):
            if to <= tb:
                continue
            base = mk_session_mix(tb, to)
            out.append((f"sm_t{tb:.2f}_o{to:.2f}", "session_mix", base))
            out.append((f"off_sm_t{tb:.2f}_o{to:.2f}", "offflat sm", mk_off_flat(base)))

    # ── Volmom dense ────────────────────────────────────────────────────
    for thr in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        for lb in (10, 15, 20, 25, 30, 40):
            vm = mk_volmom(thr, lb)
            out.append((f"vm_t{thr}_l{lb}", "volmom", vm))
            out.append((f"off_vm_t{thr}_l{lb}", "offflat volmom", mk_off_flat(vm)))

    # ── Circadian / entropy / SNR / OU / Fourier ────────────────────────
    for z0 in (1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0):
        for zs in (0.3, 0.5, 0.7):
            out.append((f"cir_z{z0}_s{zs}", "circadian", mk_circadian(z0, zs)))
    for p0 in (0.60, 0.68, 0.72, 0.78, 0.85):
        for sn in (0.0, 0.15, 0.25):
            out.append((f"ent_p{p0}_s{sn}", "entropy", mk_entropy(p0, sn)))
    for thr in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        out.append((f"snr_{thr}", "snr", mk_snr(thr)))
    for z in (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6):
        out.append((f"ou_{z}", "ou_fade", mk_ou_fade(z)))
    for lf in (0.50, 0.55, 0.60, 0.65, 0.70):
        for sn in (0.25, 0.30, 0.35, 0.40):
            out.append((f"ft_lf{lf}_s{sn}", "fourier", mk_fourier_trend(lf, sn)))

    # ── Classic TA invent ───────────────────────────────────────────────
    for f, s, z in (
        (8, 21, 0.5),
        (12, 36, 0.6),
        (12, 36, 0.8),
        (20, 50, 0.7),
        (20, 50, 0.9),
        (20, 50, 1.1),
        (9, 30, 0.6),
        (15, 45, 0.8),
    ):
        out.append((f"ema_{f}_{s}_z{z}", "ema_pb", mk_ema_pullback(f, s, z)))
    for look, buf in ((24, 0.08), (36, 0.1), (48, 0.12), (48, 0.15), (64, 0.15), (80, 0.2), (96, 0.12)):
        out.append((f"asia_{look}_{buf}", "asia_brk", mk_breakout_asia_range(look, buf)))
    for win, k in ((20, 1.6), (30, 1.8), (40, 2.0), (40, 2.2), (50, 2.0), (50, 2.5), (60, 2.2)):
        out.append((f"bb_{win}_{k}", "bb_mr", mk_bb_mr(win, k)))
    for thr in (0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03):
        out.append((f"cusum_{thr}", "cusum", mk_cusum_clock(thr)))
    for lo, hi in ((20, 80), (25, 75), (30, 70), (35, 65), (40, 60)):
        out.append((f"rsi_{lo}_{hi}", "rsi", mk_rsi_mr(14, float(lo), float(hi))))
    for look in (20, 30, 40, 55, 70, 80, 100):
        out.append((f"don_{look}", "donchian", mk_donchian_trend(look)))
    for f, s, sig in ((8, 17, 9), (12, 26, 9), (12, 26, 5), (16, 32, 9)):
        out.append((f"macd_{f}_{s}_{sig}", "macd", mk_macd_cross(f, s, sig)))
    for m in (1.8, 2.2, 2.5, 3.0, 3.5):
        out.append((f"st_{m}", "supertrend", mk_supertrend_proxy(m)))
    for win, m in ((15, 1.2), (20, 1.5), (20, 2.0), (30, 1.5), (40, 1.8)):
        out.append((f"kelt_{win}_{m}", "keltner", mk_keltner_break(win, m)))
    for win, z in ((20, 1.5), (30, 1.8), (40, 2.0), (50, 2.2), (60, 1.8)):
        out.append((f"zmr_{win}_{z}", "zscore_mr", mk_zscore_mr(win, z)))

    # ── Pair econ ───────────────────────────────────────────────────────
    for snr in (0.28, 0.32, 0.38, 0.45):
        out.append((f"gold_thrust_{snr}", "gold_thrust", mk_gold_session_thrust(snr)))
    for look, thr in ((64, 0.015), (96, 0.02), (120, 0.025), (80, 0.018)):
        out.append((f"btc_fund_{look}_{thr}", "btc_fund", mk_btc_funding_fade(look, thr)))
    for vr_lo, vr_hi in ((0.80, 1.05), (0.85, 1.10), (0.75, 1.15)):
        out.append((f"oil_inv_{vr_lo}_{vr_hi}", "oil_inv", mk_oil_inventory_proxy(vr_lo, vr_hi)))
    for thr in (0.00008, 0.00012, 0.00018, 0.00022):
        out.append((f"aud_carry_{thr}", "aud_carry", mk_aud_carry_proxy(thr)))
    for snr, bim in ((0.28, 0.40), (0.32, 0.45), (0.40, 0.50)):
        out.append((f"aud_risk_{snr}_{bim}", "aud_risk", mk_aud_risk_on(snr, bim)))
    out.append(("aud_tot", "aud_tot", mk_aud_tot_compress()))
    out.append(("aud_ou", "aud_ou", mk_aud_session_ou()))
    out.append(("aud_asia_lon", "aud_asia_lon", mk_aud_asia_fade_london_trend()))
    for jz, fz in ((2.2, 1.0), (2.8, 1.1), (3.2, 1.3)):
        out.append((f"eth_jump_{jz}_{fz}", "eth_jump", mk_eth_jump_fade(jz, fz)))
    for vv, sn in ((1.15, 0.25), (1.25, 0.28), (1.40, 0.32)):
        out.append((f"eth_vc_{vv}_{sn}", "eth_vol", mk_eth_vol_cluster(vv, sn)))
    for sm in (10.0, 12.0, 14.0, 16.0):
        out.append((f"eth_sp_{sm}", "eth_spread", mk_eth_spread_thrust(sm, 0.32)))
    out.append(("eth_fund", "eth_fund", mk_eth_funding_proxy()))
    out.append(("eth_hurst", "eth_hurst", mk_eth_hurst_adaptive()))
    for m, s in ((1.15, 0.28), (1.25, 0.30), (1.4, 0.35)):
        out.append((f"eth_atr_{m}_{s}", "eth_atr", mk_eth_atr_expand(m, s)))
    for bars, z in ((4, 0.8), (6, 0.9), (8, 1.1)):
        out.append((f"aud_imp_{bars}_{z}", "aud_impulse", mk_aud_london_open_impulse(bars, z)))

    # ── Adaptive composites ─────────────────────────────────────────────
    sm = mk_session_mix(0.28, 0.38)
    vm = mk_volmom(0.5, 20)
    ou = mk_ou_fade(1.8)
    snr = mk_snr(0.40)
    out.append(("agree_sm_vm", "agree", mk_dual_agree(sm, vm)))
    out.append(("agree_snr_ou", "agree", mk_dual_agree(snr, ou)))
    out.append(("agree_sm_ou", "agree", mk_dual_agree(sm, ou)))
    out.append(("triple_sm_snr_ou", "triple", mk_triple_agree(sm, snr, ou)))
    out.append(("hurst_sm", "hurst_gate", mk_hurst_gate(sm, mode="trend", thr=0.52)))
    out.append(("hurst_vm", "hurst_gate", mk_hurst_gate(vm, mode="trend", thr=0.52)))
    out.append(("lonny_sm", "lonny", mk_london_ny_only(sm)))
    out.append(("lonny_vm", "lonny", mk_london_ny_only(vm)))
    out.append(("dyn_hurst_sm_ou", "dyn", mk_dyn_hurst_switch(sm, ou, 0.52)))
    out.append(("dyn_hurst_vm_bb", "dyn", mk_dyn_hurst_switch(vm, mk_bb_mr(40, 2.0), 0.52)))
    out.append(("dyn_vol_sm_ou", "dyn", mk_dyn_vol_regime(sm, ou, 1.05)))
    out.append(("dyn_bayes_sm_vm", "dyn", mk_dyn_bayes_pair(sm, vm, 400, 0.52)))
    out.append(("dyn_bayes_snr_ou", "dyn", mk_dyn_bayes_pair(snr, ou, 400, 0.52)))
    out.append(("edge_sm", "edge", mk_rolling_edge_gate(sm, 500, 0.0)))
    out.append(("edge_vm", "edge", mk_rolling_edge_gate(vm, 400, 0.0)))
    out.append(("nas_rth_sm", "nas_rth", mk_nas_rth_only(sm)))
    out.append(("nas_rth_vm", "nas_rth", mk_nas_rth_only(vm)))
    out.append(("nas_rth_don55", "nas_rth", mk_nas_rth_only(mk_donchian_trend(55))))

    # ── Full Zenith math catalog ────────────────────────────────────────
    for fid, lab, fn in _zenith_catalog():
        out.append((f"math_{fid}", f"zenith:{lab}", _wrap_zenith(fid, fn)))

    # dedupe
    seen: set[str] = set()
    uniq: list[tuple[str, str, SideFn]] = []
    for vid, lab, fn in out:
        if vid in seen:
            continue
        seen.add(vid)
        uniq.append((vid, lab, fn))

    # symbol soft-preference: keep shared + pair-tagged
    sym = (symbol or "").upper()
    if sym:
        prefer_keep = []
        for vid, lab, fn in uniq:
            # drop clearly opposite-pair-only tags
            if sym == "AUDUSD" and vid.startswith(("eth_", "btc_", "oil_", "nas_", "gold_")):
                continue
            if sym == "ETHUSD" and vid.startswith(("aud_", "oil_", "nas_", "gold_")):
                continue
            if sym == "GOLD" and vid.startswith(("aud_", "eth_", "btc_", "oil_", "nas_")):
                continue
            if sym == "BTCUSD" and vid.startswith(("aud_", "eth_jump", "eth_vc", "eth_sp", "oil_", "nas_", "gold_")):
                continue
            if sym == "UKOIL" and vid.startswith(("aud_", "eth_", "btc_", "nas_", "gold_")):
                continue
            if sym == "NAS100" and vid.startswith(("aud_", "eth_", "btc_", "oil_", "gold_")):
                continue
            if sym in ("USDJPY", "USDCHF") and vid.startswith(
                ("aud_", "eth_", "btc_", "oil_", "nas_", "gold_")
            ):
                continue
            prefer_keep.append((vid, lab, fn))
        uniq = prefer_keep

    if lean:
        # keep ~280 for 1m tractability but still large
        return uniq[:280]
    return uniq


def catalog_size(*, lean: bool = False, symbol: Optional[str] = None) -> int:
    return len(build_mega_catalog(lean=lean, symbol=symbol))
