#!/usr/bin/env python3
"""DEEP adaptive XM-cost hunt — multi-TF · full hist · param grids · new methods.

Designed for multi-hour runs:
  · Dukascopy full available span (up to ~10y)
  · TFs: 1m (scalp, last 2y) · 3m · 5m · 15m (intraday)
  · Session-wise XM spreads on every trade
  · Expanded adaptive catalog + parameter grids (SL×RR×thr)
  · Windows: 7d 15d 30d 90d 180d 1y 3y full
  · Incremental JSON progress (resume-safe per pair×tf)

PA/OB on hold. Does NOT wire live.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import _prep  # noqa: E402
from analysis.lab_adaptive_minute_day_v3_bt import _calendar  # noqa: E402
from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    BOOK,
    _classify_regime,
    _sides_bimod,
    _sides_kalman,
    _sides_perm,
    _sides_snr,
    _static_sides_for_winner,
    _window_trades,
)
from analysis.lab_adaptive_regime_v2_bt import (  # noqa: E402
    _family,
    _gated_sides,
    _switch_sides,
)
from analysis.lab_zenith_math_invention_deep_bt import _enrich  # noqa: E402
from analysis.xm_session_costs import (  # noqa: E402
    SPREAD_RISK_MULT,
    min_risk_with_spread,
    session_name,
    spread_price,
)
from data.hist_local import (  # noqa: E402
    clear_hist_cache,
    has_local_hist,
    load_local_ohlcv,
    local_span,
)
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL, _min_sl_px  # noqa: E402

HUNT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"
OUT = Path(ROOT) / "data" / "lab_adaptive_xm_deep_hunt.json"
LOG = Path(ROOT) / "data" / "lab_adaptive_xm_deep_hunt.log"
PROGRESS = Path(ROOT) / "data" / "lab_adaptive_xm_deep_hunt.progress.json"
PARTIAL = Path(ROOT) / "data" / "lab_adaptive_xm_deep_hunt.partial.json"

RISK_USD = 100.0
WINDOWS = (
    ("7d", 7),
    ("15d", 15),
    ("30d", 30),
    ("90d", 90),
    ("180d", 180),
    ("1y", 365),
    ("3y", 365 * 3),
    ("full", None),
)

# TF → max years of hist (1m capped for tractability)
TF_YEARS = {
    "3m": 5.0,
    "5m": 10.0,
    "15m": 10.0,
    # 1m scalp pass: shorter hist + lean catalog (see SHORT_VARIANTS)
    "1m": 1.0,
}
# Main intraday first (hours), then scalp 1m lean pass
TFS = ("5m", "15m", "3m", "1m")

# Per-TF grids
SL_GRID_BY_TF = {
    "1m": (2.0, 2.5),
    "3m": (2.0, 2.5, 3.0),
    "5m": (1.5, 2.0, 2.5, 3.0),
    "15m": (1.5, 2.0, 2.5, 3.0),
}
RR_GRID = (2.0, 3.0, 4.0)
GAP_GRID_BY_TF = {
    "1m": (3, 5),
    "3m": (2, 3),
    "5m": (2, 3, 5),
    "15m": (2, 3),
}

SideFn = Callable[[pd.DataFrame, dict[str, Any]], np.ndarray]


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _load(sym: str, tf: str, years: float) -> pd.DataFrame:
    clear_hist_cache()
    if not has_local_hist(sym, tf):
        raise RuntimeError(f"no hist {sym} {tf}")
    sp = local_span(sym, tf)
    assert sp
    end = sp[1]
    start = max(sp[0], end - pd.Timedelta(days=int(years * 365.25) + 40))
    raw = load_local_ohlcv(sym, tf=tf, start=start.isoformat(), end=end.isoformat())
    if raw is None or len(raw) < 800:
        raise RuntimeError(f"short {sym} {tf}")
    return _enrich(_prep(raw[["open", "high", "low", "close"]].dropna()))


# ── Adaptive / dynamic side generators ──────────────────────────────────────


def _sign_thr(score: np.ndarray, thr: float) -> np.ndarray:
    out = np.zeros(len(score), dtype=int)
    out[score >= thr] = 1
    out[score <= -thr] = -1
    return out


def mk_session_mix(thr_base: float = 0.22, thr_ov: float = 0.35) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        cal = ctx["cal"]
        snr = df["snr"].to_numpy(dtype=float)
        ds = df["drift_sign"].to_numpy(dtype=float)
        H = df["hurst"].to_numpy(dtype=float)
        kz = df["kalman_z"].to_numpy(dtype=float)
        st = ds * (0.55 * snr + 0.45 * np.clip((H - 0.5) * 4.0, -1.0, 1.0))
        sf = -np.tanh(kz / 2.0)
        alpha = np.array([0.35, 0.75, 0.55, 0.45])[cal["sess"]]
        score = alpha * st + (1.0 - alpha) * sf
        thr = np.where(cal["sess"] == 3, thr_ov, thr_base)
        return _sign_thr(score, thr)

    return fn


def mk_circadian(z0: float = 1.7, z_sin: float = 0.6) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        cal = ctx["cal"]
        kz = df["kalman_z"].to_numpy()
        zthr = z0 + z_sin * cal["sin_tod"] + 0.2 * (cal["dow"] / 7.0)
        out = np.zeros(len(df), dtype=int)
        out[kz <= -zthr] = 1
        out[kz >= zthr] = -1
        return out

    return fn


def mk_entropy(p0: float = 0.78, snr_min: float = 0.0) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        cal = ctx["cal"]
        pe = df["perm_ent"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        snr = df["snr"].to_numpy()
        pmax = p0 + 0.10 * cal["cos_tod"] + 0.04 * cal["sin_week"]
        ok = (pe <= pmax) & (ds != 0) & (snr >= snr_min)
        out = np.zeros(len(df), dtype=int)
        out[ok] = ds[ok].astype(int)
        return out

    return fn


def mk_snr(thr: float = 0.40) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        return _sides_snr(df, thr)

    return fn


def mk_vol_clock(quiet: float = 0.85, hot: float = 1.15) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        cal = ctx["cal"]
        atr = df["atr"].to_numpy(dtype=float)
        atr = np.nan_to_num(atr, nan=0.0)
        bucket = (cal["mod"] // 30).astype(int)
        typ = np.zeros(len(df), dtype=float)
        for b in range(48):
            m = bucket == b
            if m.any():
                typ[m] = float(np.median(atr[m])) if m.sum() > 20 else float(np.median(atr))
        ratio = atr / np.maximum(typ, 1e-12)
        s = ctx["static"]
        k = _sides_kalman(df, 2.0)
        out = np.zeros(len(df), dtype=int)
        out[ratio <= quiet] = k[ratio <= quiet]
        out[ratio >= hot] = s[ratio >= hot]
        mid = (ratio > quiet) & (ratio < hot)
        out[mid] = s[mid]
        return out

    return fn


def mk_weekday_regime() -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        cal = ctx["cal"]
        reg = ctx["regime"]
        prior = np.array([0.55, 0.58, 0.52, 0.48, 0.40, 0.35, 0.35])[
            cal["dow"].astype(int).clip(0, 6)
        ]
        p_t = 0.55 * (reg == 1).astype(float) + 0.45 * prior
        p_f = 0.55 * (reg == -1).astype(float) + 0.45 * (1.0 - prior)
        s = ctx["static"]
        k = _sides_kalman(df, 2.0)
        out = np.zeros(len(df), dtype=int)
        take_t = p_t >= 0.55
        take_f = (~take_t) & (p_f >= 0.55)
        out[take_t] = s[take_t]
        out[take_f] = k[take_f]
        return out

    return fn


def mk_volmom(thr: float = 0.5, lb: int = 20) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        ret = df["close"].pct_change(lb)
        vol = df["close"].pct_change().rolling(lb).std().replace(0, np.nan)
        score = (ret / vol).fillna(0).to_numpy(dtype=float)
        out = np.zeros(len(df), dtype=int)
        # vectorized cross
        prev = np.roll(score, 1)
        prev[0] = 0
        buy = (prev <= 0) & (score > thr)
        sell = (prev >= 0) & (score < -thr)
        out[buy] = 1
        out[sell] = -1
        out[:40] = 0
        return out

    return fn


def mk_fourier_trend(lf_min: float = 0.55, snr_min: float = 0.35) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        lf = df["fourier_lf"].to_numpy() if "fourier_lf" in df.columns else np.ones(len(df))
        snr = df["snr"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        ok = (lf >= lf_min) & (snr >= snr_min) & (ds != 0)
        out = np.zeros(len(df), dtype=int)
        out[ok] = ds[ok].astype(int)
        return out

    return fn


def mk_ou_fade(z: float = 1.5) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        kz = df["kalman_z"].to_numpy()
        hl = df["ou_hl"].to_numpy() if "ou_hl" in df.columns else np.full(len(df), 20.0)
        out = np.zeros(len(df), dtype=int)
        m = (hl >= 5) & (hl <= 50)
        out[m & (kz <= -z)] = 1
        out[m & (kz >= z)] = -1
        return out

    return fn


def mk_rolling_edge_gate(base: SideFn, look: int = 500, min_er: float = 0.0) -> SideFn:
    """Dynamic: only take base signals when recent proxy edge is non-negative.

    Proxy: sign agreement of drift with next-bar return over rolling window.
    """

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sides = base(df, ctx)
        ret = df["close"].pct_change().fillna(0).to_numpy()
        ds = df["drift_sign"].to_numpy()
        # rolling hit rate of drift direction
        hit = (np.sign(ds) * np.sign(np.roll(ret, -1))) > 0
        hit = hit.astype(float)
        roll = pd.Series(hit).rolling(look, min_periods=max(50, look // 5)).mean().fillna(0.5).to_numpy()
        out = sides.copy()
        out[roll < (0.5 + min_er)] = 0
        return out

    return fn


def mk_session_book_switch() -> SideFn:
    """Asia fade · London trend · NY mix · Off flat."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        cal = ctx["cal"]
        s = ctx["static"]
        k = _sides_kalman(df, 2.0)
        snr = _sides_snr(df, 0.40)
        out = np.zeros(len(df), dtype=int)
        # sess: 0 asia 1 london 2 ny 3 overlap
        asia = cal["sess"] == 0
        lon = cal["sess"] == 1
        ny = cal["sess"] == 2
        ov = cal["sess"] == 3
        out[asia] = k[asia]
        out[lon] = snr[lon]
        out[ny] = np.where(s[ny] != 0, s[ny], k[ny])
        out[ov] = np.where(snr[ov] != 0, snr[ov], 0)
        # Off hours (21-24) already mostly asia in cal — force flat 21+
        out[cal["hour"] >= 21] = 0
        return out

    return fn


def mk_liq_only(base: SideFn) -> SideFn:
    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sides = base(df, ctx)
        hour = ctx["cal"]["hour"]
        out = sides.copy()
        out[(hour < 8) | (hour >= 21)] = 0
        return out

    return fn


def _build_variants(*, lean: bool = False) -> list[tuple[str, str, SideFn]]:
    """Adaptive catalog. lean=True → scalp 1m subset (~20) for runtime."""
    out: list[tuple[str, str, SideFn]] = []
    if lean:
        for tb, to in ((0.22, 0.35), (0.28, 0.38), (0.32, 0.42)):
            out.append((f"session_mix_t{tb:.2f}_o{to:.2f}", "session_mix", mk_session_mix(tb, to)))
        for z0 in (1.7, 2.0, 2.2):
            out.append((f"circadian_z{z0:.1f}_s0.5", "circadian", mk_circadian(z0, 0.5)))
        for p0, sn in ((0.72, 0.22), (0.78, 0.0), (0.72, 0.0)):
            out.append((f"entropy_p{p0:.2f}_s{sn:.2f}", "entropy", mk_entropy(p0, sn)))
        out.append(("vol_clock", "vol clock", mk_vol_clock()))
        out.append(("weekday_regime_prior", "weekday regime", mk_weekday_regime()))
        out.append(("session_book_switch", "session book switch", mk_session_book_switch()))
        out.append(("volmom_t0.5_l20", "volmom", mk_volmom(0.5, 20)))
        out.append(("fourier_lf0.55_s0.35", "fourier", mk_fourier_trend(0.55, 0.35)))
        out.append(("ou_fade_z1.5", "ou fade", mk_ou_fade(1.5)))
        out.append(("session_mix_liq", "session_mix liq", mk_liq_only(mk_session_mix(0.28, 0.38))))
        out.append(("circadian_edgegate", "circadian edgegate", mk_rolling_edge_gate(mk_circadian(2.2, 0.5), 400, 0.02)))
        return out

    # Full grid — intraday TFs
    for tb, to in product((0.18, 0.22, 0.28, 0.32, 0.38), (0.30, 0.35, 0.42)):
        out.append(
            (
                f"session_mix_t{tb:.2f}_o{to:.2f}",
                f"session_mix thr={tb}/{to}",
                mk_session_mix(tb, to),
            )
        )
    for z0, zs in product((1.5, 1.7, 2.0, 2.2, 2.5), (0.4, 0.6, 0.8)):
        out.append(
            (f"circadian_z{z0:.1f}_s{zs:.1f}", f"circadian z0={z0} sin={zs}", mk_circadian(z0, zs))
        )
    for p0, sn in product((0.65, 0.72, 0.78, 0.85), (0.0, 0.18, 0.22, 0.30)):
        out.append(
            (f"entropy_p{p0:.2f}_s{sn:.2f}", f"entropy p0={p0} snr>={sn}", mk_entropy(p0, sn))
        )
    for thr in (0.30, 0.40, 0.50, 0.60):
        out.append((f"snr_{thr:.2f}", f"SNR thr={thr}", mk_snr(thr)))
    for thr, lb in product((0.4, 0.5, 0.7), (10, 20, 40)):
        out.append((f"volmom_t{thr}_l{lb}", f"volmom thr={thr} lb={lb}", mk_volmom(thr, lb)))
    for lf, sn in product((0.50, 0.55, 0.65), (0.30, 0.40)):
        out.append((f"fourier_lf{lf}_s{sn}", f"fourier LF gate", mk_fourier_trend(lf, sn)))
    for z in (1.2, 1.5, 1.8, 2.2):
        out.append((f"ou_fade_z{z}", f"OU fade z={z}", mk_ou_fade(z)))
    out.append(("vol_clock", "vol clock", mk_vol_clock()))
    out.append(("vol_clock_tight", "vol clock tight", mk_vol_clock(0.90, 1.10)))
    out.append(("weekday_regime_prior", "weekday regime prior", mk_weekday_regime()))
    out.append(("session_book_switch", "Asia fade / Lon trend / NY mix", mk_session_book_switch()))
    base_sm = mk_session_mix(0.28, 0.38)
    base_cir = mk_circadian(2.2, 0.5)
    base_ent = mk_entropy(0.72, 0.22)
    out.append(("session_mix_liq", "session_mix liquid hours", mk_liq_only(base_sm)))
    out.append(("circadian_liq", "circadian liquid hours", mk_liq_only(base_cir)))
    out.append(("entropy_liq", "entropy liquid hours", mk_liq_only(base_ent)))
    out.append(
        (
            "session_mix_edgegate",
            "session_mix + rolling edge gate",
            mk_rolling_edge_gate(base_sm, 400, 0.0),
        )
    )
    out.append(
        (
            "circadian_edgegate",
            "circadian + rolling edge gate",
            mk_rolling_edge_gate(base_cir, 400, 0.02),
        )
    )
    return out


def _entries(
    df: pd.DataFrame,
    sides: np.ndarray,
    sym: str,
    *,
    sl_atr: float,
    min_gap: int,
) -> list[dict]:
    atr = df["atr"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    hours = pd.DatetimeIndex(df.index).hour.to_numpy(dtype=float)
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
        hour = float(hours[i])
        risk = min_risk_with_spread(
            sym, float(sl_atr) * a, hour_utc=hour, floor_px=floor
        )
        sp = spread_price(sym, hour_utc=hour)
        if sp > 0 and risk < 4.0 * sp:
            continue
        if s > 0:
            sl, side = entry - risk, "BUY"
        else:
            sl, side = entry + risk, "SELL"
        out.append(
            {
                "ei": i,
                "side": side,
                "entry": entry,
                "sl": sl,
                "risk": risk,
                "spread_px": sp,
                "hour": hour,
            }
        )
        busy = i + min_gap
    return out


def _simulate_fast(
    df: pd.DataFrame, entries: list[dict], rr: float, sym: str
) -> list[dict]:
    """Numpy barrier path — same economics as _simulate, much faster on long hist."""
    high = df["high"].to_numpy(dtype=float, copy=False)
    low = df["low"].to_numpy(dtype=float, copy=False)
    n = len(df)
    trades: list[dict] = []
    busy = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < busy or ei >= n - 1:
            continue
        risk = float(e["risk"])
        if risk <= 0:
            continue
        sp = float(e.get("spread_px") or 0)
        spread_r = sp / risk if risk else 0.0
        side = e["side"]
        entry = float(e["entry"])
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
            # scan forward
            exit_r = None
            exit_i = ei
            for j in range(ei + 1, n):
                hit_sl = low[j] <= sl
                hit_tp = high[j] >= tp
                if hit_sl or hit_tp:
                    exit_r = -1.0 if hit_sl else float(rr)
                    if hit_sl and hit_tp:
                        exit_r = -1.0
                    exit_i = j
                    break
        else:
            sl, tp = entry + risk, entry - rr * risk
            exit_r = None
            exit_i = ei
            for j in range(ei + 1, n):
                hit_sl = high[j] >= sl
                hit_tp = low[j] <= tp
                if hit_sl or hit_tp:
                    exit_r = -1.0 if hit_sl else float(rr)
                    if hit_sl and hit_tp:
                        exit_r = -1.0
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
                "session": session_name(float(e.get("hour") or 0)),
                "r": round(r_net, 4),
                "spread_r": round(spread_r, 4),
                "pnl": round(r_net * RISK_USD, 2),
            }
        )
        busy = exit_i + 2
    return trades


# alias
_simulate = _simulate_fast


def _summarize(trades: list[dict]) -> dict:
    if not trades:
        return {
            "n": 0,
            "exp_r": 0.0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "avg_spread_r": 0.0,
            "wf_green": 0,
            "oos_exp_r": 0.0,
            "verdict": "FAIL",
        }
    n = len(trades)
    rs = [float(t["r"]) for t in trades]
    exp_r = sum(rs) / n
    wr = sum(1 for r in rs if r > 0) / n
    pnl = sum(float(t["pnl"]) for t in trades)
    avg_sp = sum(float(t.get("spread_r") or 0) for t in trades) / n
    idxs = [int(t["i"]) for t in trades]
    i0, i1 = min(idxs), max(idxs) + 1
    span = max(i1 - i0, 1)
    wf_green = 0
    for k in range(4):
        a = i0 + int(span * (0.40 + 0.15 * k))
        b = i0 + int(span * (0.55 + 0.15 * k)) - 1
        sub = [t for t in trades if a <= int(t["i"]) <= b]
        if len(sub) >= 5 and sum(float(t["r"]) for t in sub) / len(sub) > 0:
            wf_green += 1
    split = i0 + int(span * 0.70)
    oos = [t for t in trades if int(t["i"]) >= split]
    oos_er = sum(float(t["r"]) for t in oos) / len(oos) if oos else 0.0
    verd = "FAIL"
    if n >= 20 and exp_r > 0 and oos_er > 0 and wf_green >= 3:
        verd = "PASS"
    elif n >= 12 and exp_r > 0 and wf_green >= 2:
        verd = "SOFT"
    return {
        "n": n,
        "exp_r": round(exp_r, 4),
        "win_rate": round(wr, 4),
        "pnl": round(pnl, 2),
        "avg_spread_r": round(avg_sp, 4),
        "wf_green": wf_green,
        "oos_exp_r": round(oos_er, 4),
        "verdict": verd,
    }


def _rank(sm: dict) -> tuple:
    v = sm.get("verdict")
    rank = 2 if v == "PASS" else (1 if v == "SOFT" else 0)
    return (
        rank,
        float(sm.get("pnl") or 0),
        float(sm.get("exp_r") or 0),
        float(sm.get("oos_exp_r") or 0),
        int(sm.get("wf_green") or 0),
        int(sm.get("n") or 0),
    )


def _save_partial(payload: dict) -> None:
    PARTIAL.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    if not LOG.exists() or LOG.stat().st_size == 0:
        LOG.write_text("", encoding="utf-8")
    t0 = time.time()
    hunt = json.loads(HUNT.read_text(encoding="utf-8")) if HUNT.exists() else {}
    per_hunt = hunt.get("per_pair") or {}
    variants_full = _build_variants(lean=False)
    variants_lean = _build_variants(lean=True)
    rr_grid = RR_GRID

    # resume
    done_keys = set()
    per_pair: dict[str, Any] = {}
    if PARTIAL.exists():
        try:
            prev = json.loads(PARTIAL.read_text(encoding="utf-8"))
            per_pair = prev.get("per_pair") or {}
            for sym, row in per_pair.items():
                for tf, block in (row.get("by_tf") or {}).items():
                    if block.get("done"):
                        done_keys.add(f"{sym}:{tf}")
            _log(f"RESUME · {len(done_keys)} pair×tf already done")
        except Exception:
            pass

    est = 0
    for tf in TFS:
        nv = len(variants_lean if tf == "1m" else variants_full) + 3
        est += (
            len(BOOK)
            * nv
            * len(SL_GRID_BY_TF[tf])
            * len(RR_GRID)
            * len(GAP_GRID_BY_TF[tf])
        )
    total_cells = est
    _log(
        f"=== DEEP XM adaptive hunt ===\n"
        f"pairs={len(BOOK)} tfs={TFS} full_variants={len(variants_full)} "
        f"lean_1m={len(variants_lean)} rr={rr_grid}\n"
        f"est_cells≈{total_cells:,} · risk=${RISK_USD}/R · session spreads ON\n"
        f"windows={[w for w,_ in WINDOWS]}\n"
    )
    _progress({"status": "running", "pct": 0, "cells_est": total_cells})

    cells_done = 0
    for pi, sym in enumerate(BOOK):
        wmeta = (per_hunt.get(sym) or {}).get("winner") or {}
        wid = str(wmeta.get("id") or "")
        fam = _family(wid)
        per_pair.setdefault(
            sym,
            {
                "symbol": sym,
                "static_id": wid,
                "family": fam,
                "by_tf": {},
                "best_overall": None,
            },
        )

        for tf in TFS:
            key = f"{sym}:{tf}"
            if key in done_keys:
                _log(f"skip {key} (resume)")
                continue
            years = TF_YEARS[tf]
            _log(f"\n######## {sym} @ {tf} · {years}y ########")
            _progress(
                {
                    "status": "running",
                    "pct": round(
                        100
                        * (pi * len(TFS) + TFS.index(tf))
                        / (len(BOOK) * len(TFS)),
                        1,
                    ),
                    "pair": sym,
                    "tf": tf,
                    "cells_done": cells_done,
                    "elapsed_sec": round(time.time() - t0, 1),
                }
            )
            try:
                df = _load(sym, tf, years)
            except Exception as exc:
                _log(f"  LOAD FAIL: {exc}")
                per_pair[sym]["by_tf"][tf] = {"error": str(exc), "done": True}
                _save_partial(
                    {
                        "per_pair": per_pair,
                        "elapsed_sec": round(time.time() - t0, 1),
                    }
                )
                continue

            static, label, _ = _static_sides_for_winner(df, wid)
            regime = _classify_regime(df)
            cal = _calendar(df)
            ctx = {
                "sym": sym,
                "static": static,
                "regime": regime,
                "cal": cal,
                "v2_preferred_sides": static,
                "family": fam,
            }
            # also evaluate gated/switch once
            alt_trend = (
                _sides_bimod(df, 0.55)
                if sym in ("ETHUSD", "BTCUSD")
                else _sides_perm(df, 0.80)
                if sym == "GOLD"
                else _sides_snr(df, 0.40)
            )
            alt_fade = _sides_kalman(df, 2.2)
            extra_sides = {
                "static": static,
                "gated": _gated_sides(static, regime, fam),
                "switch": _switch_sides(static, regime, fam, alt_trend, alt_fade),
            }

            results = []
            live_sl = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))
            sl_grid = SL_GRID_BY_TF[tf]
            gap_grid = GAP_GRID_BY_TF[tf]
            variants = variants_lean if tf == "1m" else variants_full

            # precompute sides for all variants (expensive once)
            sides_cache: dict[str, np.ndarray] = dict(extra_sides)
            for vid, _vlab, vfn in variants:
                try:
                    sides_cache[vid] = vfn(df, ctx)
                except Exception as exc:
                    _log(f"  sides {vid} fail: {exc}")
                    sides_cache[vid] = np.zeros(len(df), dtype=int)

            mode_ids = list(sides_cache.keys())
            n_combo = len(mode_ids) * len(sl_grid) * len(rr_grid) * len(gap_grid)
            _log(f"  bars={len(df)} modes={len(mode_ids)} combos={n_combo}")

            for mi, mode in enumerate(mode_ids):
                sides = sides_cache[mode]
                for sl_atr, gap in product(sl_grid, gap_grid):
                    if sym in ("AUDUSD", "USDCHF", "USDJPY") and sl_atr < 2.0:
                        continue
                    # entries depend on SL + gap only — reuse across RR
                    entries = _entries(
                        df, sides, sym, sl_atr=float(sl_atr), min_gap=int(gap)
                    )
                    for rr in rr_grid:
                        trades = _simulate(df, entries, float(rr), sym)
                        wins = {}
                        for wname, days in WINDOWS:
                            if days is None:
                                tw = trades
                            else:
                                tw = _window_trades(df, trades, days)
                            wins[wname] = _summarize(tw)
                        cell = {
                            "id": f"{mode}|sl{sl_atr}|rr{rr}|g{gap}",
                            "mode": mode,
                            "sl_atr": sl_atr,
                            "rr": rr,
                            "min_gap": gap,
                            "tf": tf,
                            "windows": wins,
                            "n_entries": len(entries),
                        }
                        cell["_score"] = (
                            _rank(wins["1y"])
                            + _rank(wins["180d"])
                            + _rank(wins["90d"])
                        )
                        results.append(cell)
                        cells_done += 1
                if (mi + 1) % 10 == 0 or mi == 0:
                    _log(
                        f"  … {mi+1}/{len(mode_ids)} modes · cells={cells_done} · "
                        f"{time.time()-t0:.0f}s · {mode}"
                    )
                    _progress(
                        {
                            "status": "running",
                            "pair": sym,
                            "tf": tf,
                            "mode_i": mi + 1,
                            "modes": len(mode_ids),
                            "mode_id": mode,
                            "mode_family": str(mode).split("_")[0]
                            if mode
                            not in ("static", "gated", "switch")
                            else "baseline",
                            "cells_done": cells_done,
                            "elapsed_sec": round(time.time() - t0, 1),
                        }
                    )

            results.sort(key=lambda c: c["_score"], reverse=True)
            top = []
            for c in results[:25]:
                top.append(
                    {
                        "id": c["id"],
                        "mode": c["mode"],
                        "sl_atr": c["sl_atr"],
                        "rr": c["rr"],
                        "min_gap": c["min_gap"],
                        "verdict_1y": c["windows"]["1y"]["verdict"],
                        "pnl_1y": c["windows"]["1y"]["pnl"],
                        "exp_r_1y": c["windows"]["1y"]["exp_r"],
                        "oos_1y": c["windows"]["1y"]["oos_exp_r"],
                        "wf_1y": c["windows"]["1y"]["wf_green"],
                        "n_1y": c["windows"]["1y"]["n"],
                        "spR_1y": c["windows"]["1y"]["avg_spread_r"],
                        "pnl_180d": c["windows"]["180d"]["pnl"],
                        "pnl_90d": c["windows"]["90d"]["pnl"],
                        "pnl_30d": c["windows"]["30d"]["pnl"],
                        "pnl_15d": c["windows"]["15d"]["pnl"],
                        "pnl_7d": c["windows"]["7d"]["pnl"],
                        "verdict_30d": c["windows"]["30d"]["verdict"],
                        "verdict_90d": c["windows"]["90d"]["verdict"],
                    }
                )
            best = top[0] if top else None
            per_pair[sym]["by_tf"][tf] = {
                "done": True,
                "n_bars": len(df),
                "span": [str(df.index[0]), str(df.index[-1])],
                "n_combos": len(results),
                "best": best,
                "top25": top,
                "live_sl_ref": live_sl,
            }
            # update best overall across tfs by 1y rank
            cand = best
            cur = per_pair[sym].get("best_overall")
            if cand and (
                cur is None
                or _rank(
                    {
                        "verdict": cand["verdict_1y"],
                        "pnl": cand["pnl_1y"],
                        "exp_r": cand["exp_r_1y"],
                        "oos_exp_r": cand["oos_1y"],
                        "wf_green": cand["wf_1y"],
                        "n": cand["n_1y"],
                    }
                )
                > _rank(
                    {
                        "verdict": cur["verdict_1y"],
                        "pnl": cur["pnl_1y"],
                        "exp_r": cur["exp_r_1y"],
                        "oos_exp_r": cur["oos_1y"],
                        "wf_green": cur["wf_1y"],
                        "n": cur["n_1y"],
                    }
                )
            ):
                per_pair[sym]["best_overall"] = {**cand, "tf": tf}

            _log(
                f"  BEST {tf}: {best['id'] if best else None} "
                f"{best['verdict_1y'] if best else ''} "
                f"1y ${best['pnl_1y'] if best else 0:,.0f}"
            )
            _save_partial(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "cells_done": cells_done,
                    "per_pair": per_pair,
                }
            )

    # desk from best_overall wireable
    desk = {w: 0.0 for w, _ in WINDOWS if w != "full"}
    recs = []
    for sym in BOOK:
        row = per_pair.get(sym) or {}
        b = row.get("best_overall")
        if not b:
            continue
        wireable = b.get("verdict_1y") in ("PASS", "SOFT")
        recs.append(
            {
                "symbol": sym,
                "tf": b.get("tf"),
                "id": b.get("id"),
                "mode": b.get("mode"),
                "sl_atr": b.get("sl_atr"),
                "rr": b.get("rr"),
                "verdict_1y": b.get("verdict_1y"),
                "pnl_1y": b.get("pnl_1y"),
                "exp_r_1y": b.get("exp_r_1y"),
                "wf_1y": b.get("wf_1y"),
                "wireable": wireable,
                "pnl_7d": b.get("pnl_7d"),
                "pnl_15d": b.get("pnl_15d"),
                "pnl_30d": b.get("pnl_30d"),
                "pnl_90d": b.get("pnl_90d"),
                "pnl_180d": b.get("pnl_180d"),
            }
        )
        if wireable:
            for w, _ in WINDOWS:
                if w == "full":
                    continue
                key = f"pnl_{w}" if w != "1y" else "pnl_1y"
                # map
                m = {
                    "7d": "pnl_7d",
                    "15d": "pnl_15d",
                    "30d": "pnl_30d",
                    "90d": "pnl_90d",
                    "180d": "pnl_180d",
                    "1y": "pnl_1y",
                    "3y": None,
                }.get(w)
                if m and b.get(m) is not None:
                    desk[w] = round(desk[w] + float(b[m]), 2)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "adaptive_xm_deep_hunt",
        "research_only": True,
        "recommend_live": False,
        "risk_usd": RISK_USD,
        "elapsed_sec": round(time.time() - t0, 1),
        "cells_done": cells_done,
        "tfs": list(TFS),
        "tf_years": TF_YEARS,
        "windows": [w for w, _ in WINDOWS],
        "n_variants": len(variants),
        "desk_best_wireable": desk,
        "recommendations": recs,
        "per_pair": per_pair,
        "note": (
            "Deep Dukascopy hunt with XM session spreads, multi-TF "
            "(1m scalp→15m intraday), param grids, dynamic adaptive methods."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _save_partial(payload)
    _progress(
        {
            "status": "done",
            "pct": 100,
            "elapsed_sec": round(time.time() - t0, 1),
            "cells_done": cells_done,
            "out": str(OUT),
        }
    )
    _log(f"\nDONE · {cells_done} cells · {time.time()-t0:.0f}s · wrote {OUT}")
    for r in recs:
        _log(
            f"  REC {r['symbol']:<8} {r['tf']:<4} {r['mode']:<28} "
            f"{r['verdict_1y']:<5} 1y ${r['pnl_1y']:,.0f} "
            f"30d ${r['pnl_30d']:,.0f} 90d ${r['pnl_90d']:,.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
