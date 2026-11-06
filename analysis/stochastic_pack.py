"""Stochastic stress pack: Brownian, Itô, martingale, HJM (research only).

Applied to per-pair trade R sequences + price returns. Not live gates.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def _trade_rs(trades: list[dict]) -> np.ndarray:
    return np.array([float(t["r"]) for t in trades], dtype=float)


def brownian_motion_test(
    trades: list[dict],
    *,
    paths: int = 1500,
    steps_scale: float = 1.0,
    seed: int = 21,
) -> dict[str, Any]:
    """Geometric Brownian Motion on trade R (drift μ, vol σ from sample).

    dX = μ dt + σ dW  → equity = cumsum(X). Survival under pure diffusion
    (no jumps — contrast with Lévy pack).
    """
    if len(trades) < 5:
        return {"paths": 0, "note": "too_few_trades"}
    rs = _trade_rs(trades)
    mu = float(np.mean(rs))
    sig = float(np.std(rs, ddof=1)) or 1e-6
    n = len(rs)
    rng = np.random.default_rng(seed)
    finals = []
    max_dds = []
    ruin = 0
    for _ in range(paths):
        # One synthetic trade-path of length n
        shocks = rng.normal(mu * steps_scale, sig * np.sqrt(steps_scale), size=n)
        eq = np.cumsum(shocks)
        peak = np.maximum.accumulate(eq)
        dd = float((eq - peak).min())
        finals.append(float(eq[-1]))
        max_dds.append(dd)
        if eq.min() < -10.0:
            ruin += 1
    finals_a = np.array(finals)
    return {
        "paths": paths,
        "model": "brownian_trade_R",
        "mu_R": round(mu, 4),
        "sigma_R": round(sig, 4),
        "median_R": round(float(np.median(finals_a)), 3),
        "p5_R": round(float(np.percentile(finals_a, 5)), 3),
        "mean_R": round(float(np.mean(finals_a)), 3),
        "prob_profit": round(100.0 * float(np.mean(finals_a > 0)), 1),
        "prob_ruin_10R": round(100.0 * ruin / paths, 1),
        "median_maxDD_R": round(float(np.median(max_dds)), 3),
        "layman": (
            "Imagine your edge as a smooth random walk with a slight tilt "
            "(drift) plus everyday noise — no sudden news jumps."
        ),
    }


def ito_lemma_test(
    df: pd.DataFrame,
    trades: list[dict],
    *,
    seed: int = 31,
) -> dict[str, Any]:
    """Itô check on log-price: discrete Δlog S vs Itô expansion.

    For f(S)=log(S), Itô says:
      df = (μ − ½σ²) dt + σ dW
    so mean(Δlog) should ≈ μ_price − ½σ² (drift correction).

    Also maps trade equity through f(E)=E² (convex) to show Itô convexity
    bonus/penalty vs naive chain rule.
    """
    if "close" not in df.columns or len(df) < 80:
        return {"note": "too_few_bars"}
    px = df["close"].astype(float).values
    logp = np.log(np.clip(px, 1e-12, None))
    dlog = np.diff(logp)
    rets = np.diff(px) / np.clip(px[:-1], 1e-12, None)
    mu = float(np.mean(rets))
    sig = float(np.std(rets, ddof=1)) or 1e-9
    # Empirical mean of Δlog vs Itô-predicted (μ − ½σ²)
    emp_dlog = float(np.mean(dlog))
    ito_pred = mu - 0.5 * sig * sig
    err = abs(emp_dlog - ito_pred)
    # Relative: error vs |ito_pred| or sigma scale
    tol = max(abs(ito_pred) * 0.5, sig * sig * 0.25, 1e-8)
    log_ok = err <= tol * 3.0 or err < 1e-5

    # Equity convexity: f(E)=E² — chain rule would ignore ½ f'' σ² term
    out: dict[str, Any] = {
        "model": "ito_log_price",
        "price_mu": round(mu, 6),
        "price_sigma": round(sig, 6),
        "emp_mean_dlog": round(emp_dlog, 6),
        "ito_pred_dlog": round(ito_pred, 6),
        "ito_error": round(err, 6),
        "log_ito_ok": bool(log_ok),
        "layman": (
            "Itô's lemma is the 'correct calculus' for random prices — "
            "it adds a small drift fix (−½σ²) that plain school calculus misses. "
            "We check if gold/FX log-moves respect that fix."
        ),
    }
    if len(trades) >= 5:
        rs = _trade_rs(trades)
        eq = np.cumsum(rs)
        # Discrete Δ(E²) vs chain 2E ΔE + (ΔE)²  (Itô discrete)
        dE = np.diff(eq, prepend=0.0)
        dE2 = np.diff(eq ** 2, prepend=0.0)
        chain = 2.0 * eq * dE  # naive (wrong for continuous limit)
        ito_disc = 2.0 * eq * dE + (dE ** 2)  # includes quadratic variation
        # Compare last segment means
        out["equity_quadratic_mean"] = round(float(np.mean(dE**2)), 4)
        out["naive_chain_bias"] = round(float(np.mean(chain - dE2)), 4)
        out["ito_discrete_bias"] = round(float(np.mean(ito_disc - dE2)), 4)
        out["convexity_ok"] = abs(float(np.mean(ito_disc - dE2))) < abs(
            float(np.mean(chain - dE2))
        ) + 1e-9
    verdict = "PASS" if out.get("log_ito_ok") else "SOFT"
    if out.get("convexity_ok") is False:
        verdict = "SOFT"
    out["verdict"] = verdict
    return out


def martingale_test(
    trades: list[dict],
    *,
    block: int = 3,
) -> dict[str, Any]:
    """Martingale / fair-game tests on trade R.

    Under a pure martingale, E[R_{t+1}|past]=0 (no edge).
    We want REJECT martingale (positive drift) for a real algo edge.

    Tests:
      1) Mean increment vs 0 (t-stat)
      2) Autocorr of increments ≈ 0 (no easy predictability leftover)
      3) Variance-ratio ≈ 1 for block aggregation (random-walk-ish residuals)
    """
    if len(trades) < 6:
        return {"note": "too_few_trades", "verdict": "INCONCLUSIVE"}
    rs = _trade_rs(trades)
    n = len(rs)
    mu = float(np.mean(rs))
    sig = float(np.std(rs, ddof=1)) or 1e-9
    t_stat = mu / (sig / np.sqrt(n))
    # Two-sided p approx via normal
    from math import erfc, sqrt

    p_two = float(erfc(abs(t_stat) / sqrt(2.0)))
    # Autocorr lag-1
    if n > 2:
        c0 = float(np.dot(rs - mu, rs - mu))
        c1 = float(np.dot(rs[1:] - mu, rs[:-1] - mu))
        ac1 = c1 / c0 if c0 > 0 else 0.0
    else:
        ac1 = 0.0
    # Variance ratio: Var(sum of k) / (k Var(1))
    k = max(2, min(block, n // 2))
    # overlapping blocks
    block_sums = np.array(
        [rs[i : i + k].sum() for i in range(0, n - k + 1)], dtype=float
    )
    vr = float(np.var(block_sums, ddof=1) / (k * (sig**2))) if n > k + 2 else 1.0

    # Edge present if mean > 0 and t>1.0 (lenient for small n)
    has_edge = mu > 0 and t_stat > 1.0
    # Martingale-like residuals if |ac1| small and VR near 1
    residual_ok = abs(ac1) < 0.35 and 0.5 <= vr <= 1.8

    if has_edge and residual_ok:
        verdict = "EDGE_NOT_MARTINGALE"  # good — has drift, residuals fair
    elif has_edge:
        verdict = "EDGE_PREDICTABLE"  # edge but leftover serial structure
    elif mu > 0:
        verdict = "WEAK_EDGE"
    else:
        verdict = "MARTINGALE_LIKE"  # bad for a trading system

    return {
        "model": "martingale_trade_R",
        "mean_R": round(mu, 4),
        "t_stat": round(t_stat, 3),
        "p_value_two_sided": round(p_two, 4),
        "acf_lag1": round(ac1, 3),
        "variance_ratio": round(vr, 3),
        "block_k": k,
        "has_edge": bool(has_edge),
        "verdict": verdict,
        "layman": (
            "A martingale is a fair coin-flip game — past doesn't help, "
            "expected next move is zero. We want the algo NOT to be a martingale: "
            "a positive tilt (edge) with leftovers that aren't too pattern-y."
        ),
    }


def hjm_forward_curve_test(
    trades: list[dict],
    *,
    horizons: tuple[int, ...] = (1, 2, 3, 5, 8),
) -> dict[str, Any]:
    """HJM-inspired forward expectancy curve on trade R.

    Build 'forward rates' f(0,T) = E[average R over next T trades] from the
    sample path (rolling), then check a discrete HJM-style consistency:
      forwards should not explode; drift of short end ≈ vol structure.

    Practical desk read:
      - upward forward curve → edge compounds with horizon
      - inverted / unstable → fragile term structure
    """
    if len(trades) < 8:
        return {"note": "too_few_trades", "verdict": "INCONCLUSIVE"}
    rs = _trade_rs(trades)
    n = len(rs)
    curve = {}
    for h in horizons:
        if h >= n:
            continue
        # Rolling mean of next-h trades (forward-looking in-sample proxy)
        fwd = np.array(
            [rs[i : i + h].mean() for i in range(0, n - h + 1)], dtype=float
        )
        curve[str(h)] = {
            "horizon_trades": h,
            "mean_fwd_R": round(float(np.mean(fwd)), 4),
            "std_fwd_R": round(float(np.std(fwd, ddof=1)), 4),
            "p_positive": round(100.0 * float(np.mean(fwd > 0)), 1),
        }
    if len(curve) < 2:
        return {"note": "curve_too_short", "verdict": "INCONCLUSIVE"}

    means = [curve[k]["mean_fwd_R"] for k in sorted(curve, key=int)]
    # Smoothness: consecutive forward diffs
    diffs = np.diff(means)
    max_jump = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
    # Short-end vol vs long-end vol (HJM vol structure proxy)
    keys = sorted(curve, key=int)
    vol_short = curve[keys[0]]["std_fwd_R"]
    vol_long = curve[keys[-1]]["std_fwd_R"]
    # No-arb soft: long vol shouldn't wildly exceed short without mean collapse
    structure_ok = vol_long <= vol_short * 2.5 + 0.15
    positive_curve = sum(1 for m in means if m > 0) >= max(1, len(means) // 2 + 1)
    smooth = max_jump < 0.85

    if positive_curve and structure_ok and smooth:
        verdict = "PASS"
    elif positive_curve:
        verdict = "SOFT"
    else:
        verdict = "FAIL"

    return {
        "model": "hjm_forward_R_curve",
        "curve": curve,
        "max_forward_jump": round(max_jump, 4),
        "vol_short": vol_short,
        "vol_long": vol_long,
        "structure_ok": bool(structure_ok),
        "positive_curve": bool(positive_curve),
        "verdict": verdict,
        "layman": (
            "HJM is how bond desks model the whole yield curve evolving together. "
            "Here we build a 'curve' of expected trade edge at 1,2,3… trades ahead. "
            "A healthy algo keeps that curve mostly positive and smooth — not wild."
        ),
    }


def stochastic_pack(df: pd.DataFrame, trades: list[dict]) -> dict[str, Any]:
    return {
        "brownian": brownian_motion_test(trades),
        "ito": ito_lemma_test(df, trades),
        "martingale": martingale_test(trades),
        "hjm": hjm_forward_curve_test(trades),
    }
