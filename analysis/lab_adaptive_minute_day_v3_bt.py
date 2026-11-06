#!/usr/bin/env python3
"""RESEARCH ONLY — Adaptive v3: minute/day-conditioned equations vs v2.

Invent soft adaptive policies that change every bar from:
  · minute-of-day / hour / weekday / session
  · live regime features (Hurst, SNR, Kalman z, entropy)
Then bake-off against adaptive v2 baselines (static | gated | switch | preferred).

Does NOT change live brains.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD  # noqa: E402
from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    BOOK,
    TF,
    WINDOWS,
    _classify_regime,
    _equity_curve,
    _load,
    _sides_bimod,
    _sides_kalman,
    _sides_perm,
    _sides_snr,
    _static_sides_for_winner,
    _summarize,
    _window_trades,
)
from analysis.lab_adaptive_regime_v2_bt import (  # noqa: E402
    _family,
    _gated_sides,
    _switch_sides,
)
from analysis.lab_zenith_math_invention_deep_bt import (  # noqa: E402
    _atr_sl_entries,
    _trades_with_i,
)

HUNT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"
V2 = Path(ROOT) / "data" / "lab_adaptive_regime_v2.json"
OUT = Path(ROOT) / "data" / "lab_adaptive_minute_day_v3.json"
EQ = Path(ROOT) / "data" / "lab_adaptive_minute_day_v3_equations.md"
LOG = Path(ROOT) / "data" / "lab_adaptive_minute_day_v3.log"
PROGRESS = Path(ROOT) / "data" / "lab_adaptive_minute_day_v3.progress.json"
CAP_YEARS = 3.0

SideFn = Callable[[pd.DataFrame, dict[str, Any]], np.ndarray]


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _calendar(df: pd.DataFrame) -> dict[str, np.ndarray]:
    idx = pd.DatetimeIndex(df.index)
    # hist is UTC-ish; treat as clock for session structure
    hour = idx.hour.to_numpy(dtype=float)
    minute = idx.minute.to_numpy(dtype=float)
    dow = idx.dayofweek.to_numpy(dtype=float)  # Mon=0
    mod = hour * 60.0 + minute  # minute of day 0..1439
    tod = mod / 1440.0
    ang = 2.0 * math.pi * tod
    # Sessions (UTC proxies): Asia 0-7, London 7-12, NY 13-21, overlap 12-16
    sess = np.zeros(len(df), dtype=int)  # 0 asia 1 london 2 ny 3 overlap
    sess[(hour >= 7) & (hour < 12)] = 1
    sess[(hour >= 12) & (hour < 16)] = 3
    sess[(hour >= 16) & (hour < 21)] = 2
    return {
        "hour": hour,
        "minute": minute,
        "dow": dow,
        "mod": mod,
        "tod": tod,
        "sin_tod": np.sin(ang),
        "cos_tod": np.cos(ang),
        "sin_week": np.sin(2.0 * math.pi * (dow + tod) / 7.0),
        "cos_week": np.cos(2.0 * math.pi * (dow + tod) / 7.0),
        "sess": sess,
    }


def _soft_gate(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Smooth 0→1 gate between lo/hi."""
    return np.clip((x - lo) / max(1e-9, hi - lo), 0.0, 1.0)


def _score_trend(df: pd.DataFrame) -> np.ndarray:
    snr = df["snr"].to_numpy(dtype=float)
    ds = df["drift_sign"].to_numpy(dtype=float)
    H = df["hurst"].to_numpy(dtype=float)
    return ds * (0.55 * snr + 0.45 * np.clip((H - 0.5) * 4.0, -1.0, 1.0))


def _score_fade(df: pd.DataFrame) -> np.ndarray:
    kz = df["kalman_z"].to_numpy(dtype=float)
    # fade wants mean-revert: negative score when z>0 (short), positive when z<0
    return -np.tanh(kz / 2.0)


def _sign_thr(score: np.ndarray, thr: float) -> np.ndarray:
    out = np.zeros(len(score), dtype=int)
    out[score >= thr] = 1
    out[score <= -thr] = -1
    return out


def _books(df: pd.DataFrame, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "static": ctx["static"],
        "snr": _sides_snr(df, 0.40),
        "bimod": _sides_bimod(df, 0.55),
        "perm": _sides_perm(df, 0.80),
        "kalman": _sides_kalman(df, 2.2),
        "kalman_soft": _sides_kalman(df, 1.8),
    }


# ── Invented adaptive equations (bar-level / day-conditioned) ───────────────


def eq_circadian_thr(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """SNR threshold breathes with TOD harmonic: thr(t)=τ0+α·sin(2π·tod+φ)."""
    cal = ctx["cal"]
    snr = df["snr"].to_numpy()
    ds = df["drift_sign"].to_numpy()
    thr = 0.38 + 0.12 * cal["sin_tod"] + 0.06 * cal["cos_tod"]
    return np.where((snr >= thr) & (ds != 0), ds.astype(int), 0)


def eq_dow_book(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Weekday picks book: Mon/Tue trend, Wed mix, Thu/Fri fade-lean."""
    cal = ctx["cal"]
    b = _books(df, ctx)
    dow = cal["dow"].astype(int)
    out = np.zeros(len(df), dtype=int)
    out[dow <= 1] = b["static"][dow <= 1]
    mid = dow == 2
    out[mid] = np.where(b["static"][mid] != 0, b["static"][mid], b["kalman"][mid])
    late = dow >= 3
    out[late] = np.where(b["kalman"][late] != 0, b["kalman"][late], b["static"][late])
    return out


def eq_session_mix(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Soft mix: side = sign(α(sess)·trend + β(sess)·fade)."""
    cal = ctx["cal"]
    st = _score_trend(df)
    sf = _score_fade(df)
    # Asia fade lean, London trend, NY mix, overlap cautious
    alpha = np.array([0.35, 0.75, 0.55, 0.45])[cal["sess"]]
    beta = 1.0 - alpha
    score = alpha * st + beta * sf
    # session thr: overlap stricter
    thr = np.where(cal["sess"] == 3, 0.35, 0.22)
    return _sign_thr(score, thr)


def eq_harmonic_gate(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Trade static only when |sin(ω·tod + φ_dow)| exceeds gate (active minutes)."""
    cal = ctx["cal"]
    phase = 0.6 * cal["dow"] / 7.0 * 2.0 * math.pi
    wave = np.abs(np.sin(2.0 * math.pi * cal["tod"] + phase))
    gate = wave >= (0.35 + 0.08 * (cal["dow"] >= 3))
    out = np.zeros(len(df), dtype=int)
    s = ctx["static"]
    out[gate] = s[gate]
    return out


def eq_vol_clock(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Aggressiveness vs typical vol for this minute-of-day bucket."""
    cal = ctx["cal"]
    atr = df["atr"].to_numpy(dtype=float) if "atr" in df.columns else df["close"].pct_change().abs().rolling(14).mean().to_numpy()
    atr = np.nan_to_num(atr, nan=0.0)
    # bucket by 30m
    bucket = (cal["mod"] // 30).astype(int)
    typ = np.zeros(len(df), dtype=float)
    for b in range(48):
        m = bucket == b
        if m.any():
            typ[m] = float(np.median(atr[m])) if m.sum() > 20 else float(np.median(atr))
    ratio = atr / np.maximum(typ, 1e-12)
    # quiet → fade; hot → trend static
    s = ctx["static"]
    k = _sides_kalman(df, 2.0)
    out = np.zeros(len(df), dtype=int)
    out[ratio <= 0.85] = k[ratio <= 0.85]
    out[ratio >= 1.15] = s[ratio >= 1.15]
    mid = (ratio > 0.85) & (ratio < 1.15)
    out[mid] = s[mid]
    return out


def eq_weekday_regime_prior(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Blend live regime with empirical P(trend|dow) prior; route books."""
    cal = ctx["cal"]
    reg = ctx["regime"]
    # soft prior: midweek more trend, Friday more fade
    prior_trend = np.array([0.55, 0.58, 0.52, 0.48, 0.40, 0.35, 0.35])[cal["dow"].astype(int).clip(0, 6)]
    live_t = (reg == 1).astype(float)
    live_f = (reg == -1).astype(float)
    p_t = 0.55 * live_t + 0.45 * prior_trend
    p_f = 0.55 * live_f + 0.45 * (1.0 - prior_trend)
    b = _books(df, ctx)
    out = np.zeros(len(df), dtype=int)
    take_t = p_t >= 0.55
    take_f = (~take_t) & (p_f >= 0.55)
    out[take_t] = b["static"][take_t]
    out[take_f] = b["kalman"][take_f]
    return out


def eq_tod_fade_boost(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Boost fade near London open (07-09) and NY open (13-15); else static."""
    cal = ctx["cal"]
    h = cal["hour"]
    boost = ((h >= 7) & (h < 9)) | ((h >= 13) & (h < 15))
    s = ctx["static"]
    k = _sides_kalman(df, 1.9)
    out = s.copy()
    out[boost] = np.where(k[boost] != 0, k[boost], s[boost])
    return out


def eq_entropy_tod(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Permutation-entropy ceiling adapts with TOD: pe_max(t)=0.78+0.1·cos(tod)."""
    cal = ctx["cal"]
    pe = df["perm_ent"].to_numpy()
    ds = df["drift_sign"].to_numpy()
    pmax = 0.78 + 0.10 * cal["cos_tod"] + 0.04 * cal["sin_week"]
    return np.where((pe <= pmax) & (ds != 0), ds.astype(int), 0)


def eq_minute_softmax(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Every bar: logits from calendar+features → soft pick among books / sit-out."""
    cal = ctx["cal"]
    b = _books(df, ctx)
    snr = df["snr"].to_numpy(dtype=float)
    kz = np.abs(df["kalman_z"].to_numpy(dtype=float))
    H = df["hurst"].to_numpy(dtype=float)
    # logits
    l_static = 0.8 * snr + 0.4 * (H - 0.5) + 0.25 * cal["sin_tod"] + 0.15 * (cal["dow"] <= 2)
    l_kalman = 0.7 * _soft_gate(kz, 1.4, 2.6) + 0.35 * cal["cos_tod"] + 0.2 * (cal["dow"] >= 3)
    l_snr = 0.65 * snr + 0.2 * cal["sin_week"]
    l_flat = 0.55 + 0.25 * (1.0 - snr) + 0.15 * (cal["sess"] == 3)
    logits = np.stack([l_static, l_kalman, l_snr, l_flat], axis=1)
    # numerically stable softmax argmax
    pick = np.argmax(logits - logits.max(axis=1, keepdims=True), axis=1)
    out = np.zeros(len(df), dtype=int)
    out[pick == 0] = b["static"][pick == 0]
    out[pick == 1] = b["kalman"][pick == 1]
    out[pick == 2] = b["snr"][pick == 2]
    # pick==3 sit out
    return out


def eq_hybrid_v2_tod(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """v2 preferred sides + TOD sit-out when wave inactive (fewer bad minutes)."""
    cal = ctx["cal"]
    base = ctx["v2_preferred_sides"]
    wave = np.abs(np.sin(2.0 * math.pi * cal["tod"] + 0.4 * cal["dow"]))
    # also sit Asia deep night for FX/oil
    sym = ctx["sym"]
    night = (cal["hour"] < 5) | (cal["hour"] >= 22)
    allow = wave >= 0.28
    if sym in ("AUDUSD", "USDCHF", "USDJPY", "UKOIL"):
        allow = allow & ~night
    out = np.zeros(len(df), dtype=int)
    out[allow] = base[allow]
    return out


def eq_circadian_kalman_z(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Kalman |z| entry threshold varies by minute: z(t)=1.7+0.6·sin(tod)+0.2·dow/7."""
    cal = ctx["cal"]
    kz = df["kalman_z"].to_numpy()
    zthr = 1.7 + 0.6 * cal["sin_tod"] + 0.2 * (cal["dow"] / 7.0)
    out = np.zeros(len(df), dtype=int)
    out[kz <= -zthr] = 1
    out[kz >= zthr] = -1
    return out


def eq_intraday_cusum_clock(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Reset-ish: trade static when |cusum slope| high AND hour in active band."""
    cal = ctx["cal"]
    if "cusum" in df.columns:
        c = df["cusum"].to_numpy(dtype=float)
        slope = pd.Series(c).diff(6).fillna(0).to_numpy()
    else:
        slope = df["snr"].to_numpy() * df["drift_sign"].to_numpy()
    active = (cal["hour"] >= 6) & (cal["hour"] < 20)
    hot = np.abs(slope) >= np.nanpercentile(np.abs(slope), 65)
    out = np.zeros(len(df), dtype=int)
    s = ctx["static"]
    m = active & hot
    out[m] = s[m]
    return out


def _catalog() -> list[tuple[str, str, SideFn]]:
    return [
        ("circadian_thr", "SNR thr breathes with sin/cos(TOD)", eq_circadian_thr),
        ("dow_book", "Weekday book switch (trend→fade)", eq_dow_book),
        ("session_mix", "Soft α(sess)·trend + β·fade mix", eq_session_mix),
        ("harmonic_gate", "Static only on active harmonic minutes", eq_harmonic_gate),
        ("vol_clock", "Minute-of-day vol clock → fade/trend", eq_vol_clock),
        ("weekday_regime_prior", "P(trend|dow) × live regime router", eq_weekday_regime_prior),
        ("tod_fade_boost", "London/NY open fade boost", eq_tod_fade_boost),
        ("entropy_tod", "Perm-entropy ceiling = f(TOD,week)", eq_entropy_tod),
        ("minute_softmax", "Bar-level softmax book router", eq_minute_softmax),
        ("hybrid_v2_tod", "v2 preferred + TOD/night sit-out", eq_hybrid_v2_tod),
        ("circadian_kalman_z", "Kalman |z| thr = f(TOD, DOW)", eq_circadian_kalman_z),
        ("intraday_cusum_clock", "Cusum slope × active hours gate", eq_intraday_cusum_clock),
    ]


def main() -> int:
    LOG.write_text("")
    t0 = time.time()
    hunt = json.loads(HUNT.read_text(encoding="utf-8")) if HUNT.exists() else {}
    per_hunt = hunt.get("per_pair") or {}
    v2 = json.loads(V2.read_text(encoding="utf-8")) if V2.exists() else {}
    v2_pref = {
        s: (v2.get("per_pair") or {}).get(s, {}).get("best_mode_1y") or "static"
        for s in BOOK
    }

    formulas = _catalog()
    _log(
        "=== ADAPTIVE v3 · MINUTE/DAY EQUATIONS vs v2 ===\n"
        f"TF={TF} {CAP_YEARS}y · {len(formulas)} invented adaptive eqs\n"
        "Compare vs v2 static|gated|switch|preferred\n"
        "LIVE BRAINS NOT TOUCHED\n"
    )

    desk: dict[str, dict[str, float]] = {}
    per_pair: dict[str, Any] = {}
    eq_lines = [
        "# Adaptive v3 — minute/day equations",
        "",
        "Research only. Live brains unchanged.",
        "",
        "At each bar we recompute sides from calendar (minute-of-day, weekday, session)",
        "and live features. Bake-off vs adaptive v2.",
        "",
    ]

    for pi, sym in enumerate(BOOK):
        _progress(
            {
                "status": "running",
                "pct": int(3 + 95 * pi / len(BOOK)),
                "message": f"v3 [{sym}]",
                "symbol": sym,
                "elapsed_sec": round(time.time() - t0, 1),
            }
        )
        _log(f"\n######## {sym} ########")
        try:
            df = _load(sym, TF, CAP_YEARS)
        except Exception as exc:
            _log(f"  FAIL: {exc}")
            per_pair[sym] = {"error": str(exc)}
            continue

        w = (per_hunt.get(sym) or {}).get("winner") or {}
        wid = str(w.get("id") or "")
        fam = _family(wid)
        rr = float(w.get("rr") or 2.4)
        sl = float(w.get("sl_atr") or 1.2)

        static, label, _ = _static_sides_for_winner(df, wid)
        regime = _classify_regime(df)
        alt_trend = (
            _sides_bimod(df, 0.55)
            if sym in ("ETHUSD", "BTCUSD")
            else _sides_perm(df, 0.80)
            if sym == "GOLD"
            else _sides_snr(df, 0.40)
        )
        alt_fade = _sides_kalman(df, 2.2)
        v2_sides = {
            "static": static,
            "gated": _gated_sides(static, regime, fam),
            "switch": _switch_sides(static, regime, fam, alt_trend, alt_fade),
        }
        pref_mode = v2_pref.get(sym) or "static"
        v2_sides["v2_preferred"] = v2_sides[pref_mode]

        cal = _calendar(df)
        ctx: dict[str, Any] = {
            "sym": sym,
            "static": static,
            "regime": regime,
            "cal": cal,
            "v2_preferred_sides": v2_sides["v2_preferred"],
            "family": fam,
        }

        all_sides: dict[str, np.ndarray] = dict(v2_sides)
        for fid, _flabel, fn in formulas:
            try:
                all_sides[fid] = fn(df, ctx)
            except Exception as exc:
                _log(f"  {fid} FAIL: {exc}")
                all_sides[fid] = np.zeros(len(df), dtype=int)

        trades: dict[str, list] = {}
        for mode, sides in all_sides.items():
            ent = _atr_sl_entries(df, sides, sl_atr=sl)
            trades[mode] = _trades_with_i(df, ent, rr)

        win_out: dict[str, dict] = {}
        for wname, days in WINDOWS:
            end = df.index[-1]
            start = end - pd.Timedelta(days=days)
            n_bars = int(np.sum(np.asarray(df.index >= start)))
            block = {}
            for mode in all_sides:
                tw = _window_trades(df, trades[mode], days)
                sm = _summarize(tw, n_bars)
                block[mode] = sm
                desk.setdefault(mode, {w[0]: 0.0 for w in WINDOWS})
                desk[mode][wname] = round(desk[mode][wname] + float(sm["pnl"]), 2)
            win_out[wname] = block

        # Absolute winner = best 1y PnL among all modes; also require E[R]>0 if possible
        modes_1y = win_out["1y"]
        ranked = sorted(
            modes_1y.keys(),
            key=lambda m: (
                float(modes_1y[m]["pnl"]),
                float(modes_1y[m]["exp_r"] or 0),
                int(modes_1y[m]["n"] or 0),
            ),
            reverse=True,
        )
        abs_win = ranked[0]
        v2_best_pnl = max(float(modes_1y[m]["pnl"]) for m in ("static", "gated", "switch"))
        beat_v2 = float(modes_1y[abs_win]["pnl"]) > v2_best_pnl + 1e-6

        top5 = [
            {
                "id": m,
                "pnl_1y": modes_1y[m]["pnl"],
                "exp_r": modes_1y[m]["exp_r"],
                "n": modes_1y[m]["n"],
                "wr": modes_1y[m]["win_rate"],
            }
            for m in ranked[:5]
        ]
        _log(
            f"  ABS={abs_win} ${modes_1y[abs_win]['pnl']:,.0f} "
            f"vs v2best ${v2_best_pnl:,.0f} beat={beat_v2}"
        )
        _log("  top: " + ", ".join(f"{t['id']}=${t['pnl_1y']:,.0f}" for t in top5))

        per_pair[sym] = {
            "symbol": sym,
            "family": fam,
            "static_id": wid,
            "static_label": label,
            "rr": rr,
            "sl_atr": sl,
            "v2_preferred": pref_mode,
            "absolute_winner": abs_win,
            "beat_v2": beat_v2,
            "top5_1y": top5,
            "windows": {
                wname: {
                    m: win_out[wname][m]
                    for m in (
                        "static",
                        "gated",
                        "switch",
                        "v2_preferred",
                        abs_win,
                        *ranked[:3],
                    )
                }
                for wname, _ in WINDOWS
            },
            "all_1y": {m: modes_1y[m] for m in ranked},
        }

        eq_lines.append(f"## {sym}")
        eq_lines.append(f"- v2 preferred: `{pref_mode}`")
        eq_lines.append(f"- absolute winner: `{abs_win}` (beat_v2={beat_v2})")
        eq_lines.append(
            f"- 1y: ${modes_1y[abs_win]['pnl']:,.0f} · E[R]={modes_1y[abs_win]['exp_r']} · n={modes_1y[abs_win]['n']}"
        )
        eq_lines.append("")

    # Desk absolute = sum of per-pair absolute winners on each window
    desk_abs = {w[0]: 0.0 for w in WINDOWS}
    desk_v2pref = {w[0]: 0.0 for w in WINDOWS}
    for sym, row in per_pair.items():
        if row.get("error"):
            continue
        aw = row["absolute_winner"]
        vp = row["v2_preferred"]
        for wname, _ in WINDOWS:
            desk_abs[wname] = round(
                desk_abs[wname] + float(row["windows"][wname].get(aw, {}).get("pnl") or 0),
                2,
            )
            desk_v2pref[wname] = round(
                desk_v2pref[wname]
                + float(row["windows"][wname].get("v2_preferred", {}).get("pnl") or 0),
                2,
            )

    beats = sum(1 for r in per_pair.values() if r.get("beat_v2"))
    eq_lines.extend(
        [
            "## Desk",
            f"- absolute (per-pair best): 1y ${desk_abs['1y']:,.0f}",
            f"- v2 preferred desk: 1y ${desk_v2pref['1y']:,.0f}",
            f"- pairs beating v2: {beats}/8",
            "",
            "## Equation catalog",
        ]
    )
    for fid, flabel, _ in formulas:
        eq_lines.append(f"- `{fid}` — {flabel}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "recommend_live": False,
        "elapsed_sec": round(time.time() - t0, 1),
        "version": "adaptive_minute_day_v3",
        "tf": TF,
        "risk_usd": RISK_USD,
        "note": (
            "Invented minute/day-conditioned adaptive equations vs v2. "
            "LIVE brains NOT changed. Absolute winner = best 1y PnL per pair."
        ),
        "formulas": [{"id": a, "label": b} for a, b, _ in formulas],
        "desk_modes": {
            m: desk[m]
            for m in desk
            if m in ("static", "gated", "switch", "v2_preferred")
            or m in {f[0] for f in formulas}
        },
        "desk_absolute": desk_abs,
        "desk_v2_preferred": desk_v2pref,
        "pairs_beat_v2": beats,
        "per_pair": per_pair,
        "charts": {
            "desk": [
                {
                    "window": w,
                    "v2_preferred": desk_v2pref[w],
                    "absolute": desk_abs[w],
                    "static": (desk.get("static") or {}).get(w),
                    "gated": (desk.get("gated") or {}).get(w),
                    "switch": (desk.get("switch") or {}).get(w),
                }
                for w, _ in WINDOWS
            ],
            "pair_1y_abs_vs_v2": [
                {
                    "symbol": s,
                    "absolute": per_pair[s]["absolute_winner"],
                    "abs_pnl": per_pair[s]["all_1y"][per_pair[s]["absolute_winner"]]["pnl"],
                    "v2_pref": per_pair[s]["v2_preferred"],
                    "v2_pnl": per_pair[s]["all_1y"].get(
                        "v2_preferred", per_pair[s]["all_1y"].get(per_pair[s]["v2_preferred"], {})
                    ).get("pnl"),
                    "beat_v2": per_pair[s]["beat_v2"],
                }
                for s in BOOK
                if s in per_pair and not per_pair[s].get("error")
            ],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    EQ.write_text("\n".join(eq_lines) + "\n", encoding="utf-8")
    _progress(
        {
            "status": "done",
            "pct": 100,
            "elapsed_sec": round(time.time() - t0, 1),
            "desk_abs_1y": desk_abs["1y"],
            "desk_v2_1y": desk_v2pref["1y"],
            "pairs_beat_v2": beats,
        }
    )
    _log(
        f"\nDESK abs 1y ${desk_abs['1y']:,.0f} vs v2pref ${desk_v2pref['1y']:,.0f} "
        f"· beats {beats}/8 · wrote {OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
