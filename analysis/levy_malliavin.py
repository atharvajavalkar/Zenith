"""Lévy jump paths + Malliavin-style sensitivity for trade R sequences.

Used by research batteries (not live gating). Explains fat-tail survival and
how fragile a setup is to tiny vol / drift bumps.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def levy_monte_carlo(
    trades: list[dict],
    *,
    paths: int = 1500,
    jump_intensity: float = 0.08,
    jump_scale: float = 1.5,
    seed: int = 7,
) -> dict[str, Any]:
    """Bootstrap trade R with rare Lévy jumps (news shocks).

    Base: resample historical R (like normal MC).
    Extra: with small probability each trade, add a fat-tailed jump to R
    (symmetric Merton-style) so the equity path can gap — closer to real markets
    than smooth Brownian noise alone.
    """
    if len(trades) < 5:
        return {"paths": 0, "note": "too_few_trades"}
    rs = np.array([float(t["r"]) for t in trades], dtype=float)
    rng = np.random.default_rng(seed)
    finals: list[float] = []
    max_dds: list[float] = []
    ruin = 0
    for _ in range(paths):
        sample = rng.choice(rs, size=len(rs), replace=True).astype(float)
        # Rare jumps: Poisson-like Bernoulli per trade
        jumps = rng.random(len(sample)) < jump_intensity
        if jumps.any():
            # Laplace (double-exponential) jumps — classic Lévy building block
            shock = rng.laplace(0.0, jump_scale, size=int(jumps.sum()))
            sample[jumps] = sample[jumps] + shock
        eq = np.cumsum(sample)
        peak = np.maximum.accumulate(eq)
        dd = float((eq - peak).min())
        finals.append(float(eq[-1]))
        max_dds.append(dd)
        if eq.min() < -10.0:
            ruin += 1
    finals_a = np.array(finals)
    return {
        "paths": paths,
        "model": "levy_jump_bootstrap",
        "jump_intensity": jump_intensity,
        "jump_scale": jump_scale,
        "median_R": round(float(np.median(finals_a)), 3),
        "p5_R": round(float(np.percentile(finals_a, 5)), 3),
        "p95_R": round(float(np.percentile(finals_a, 95)), 3),
        "mean_R": round(float(np.mean(finals_a)), 3),
        "prob_profit": round(100.0 * float(np.mean(finals_a > 0)), 1),
        "prob_ruin_10R": round(100.0 * ruin / paths, 1),
        "median_maxDD_R": round(float(np.median(max_dds)), 3),
        "p5_maxDD_R": round(float(np.percentile(max_dds, 5)), 3),
        "layman": (
            "Same trade history, but occasionally a 'news jump' shocks a trade's R "
            "up or down — like real markets that gap instead of only drifting."
        ),
    }


def malliavin_sensitivity(
    trades: list[dict],
    *,
    bump: float = 0.05,
    paths: int = 800,
    seed: int = 11,
) -> dict[str, Any]:
    """Finite-difference Malliavin-style weight: d P(profit) / d vol-scale.

    Practical recipe (not full SPDE Malliavin on tick paths):
      1. Bootstrap equity paths from trade R.
      2. Re-run with R scaled by (1±bump) — mimics a tiny volatility / edge bump.
      3. Sensitivity ≈ [P(+bump) − P(−bump)] / (2·bump).

    Large |sensitivity| ⇒ fragile: a small change in edge/vol flips survival.
    Near 0 ⇒ robust.
    """
    if len(trades) < 5:
        return {"paths": 0, "note": "too_few_trades"}
    rs = np.array([float(t["r"]) for t in trades], dtype=float)

    def _prob_profit(scale: float, s: int) -> float:
        rng = np.random.default_rng(s)
        finals = []
        for _ in range(paths):
            sample = rng.choice(rs, size=len(rs), replace=True) * scale
            finals.append(float(np.cumsum(sample)[-1]))
        return float(np.mean(np.array(finals) > 0))

    p_up = _prob_profit(1.0 + bump, seed)
    p_dn = _prob_profit(1.0 - bump, seed + 1)
    p0 = _prob_profit(1.0, seed + 2)
    sens = (p_up - p_dn) / (2.0 * bump)  # per unit relative edge
    # Also report dollar-R sensitivity of median terminal R
    def _median_R(scale: float, s: int) -> float:
        rng = np.random.default_rng(s)
        finals = []
        for _ in range(paths):
            sample = rng.choice(rs, size=len(rs), replace=True) * scale
            finals.append(float(np.cumsum(sample)[-1]))
        return float(np.median(finals))

    med_up = _median_R(1.0 + bump, seed + 3)
    med_dn = _median_R(1.0 - bump, seed + 4)
    d_med = (med_up - med_dn) / (2.0 * bump)

    fragility = "ROBUST"
    if abs(sens) >= 2.0:
        fragility = "FRAGILE"
    elif abs(sens) >= 0.8:
        fragility = "SENSITIVE"

    return {
        "paths": paths,
        "bump": bump,
        "prob_profit_base": round(100.0 * p0, 1),
        "prob_profit_up": round(100.0 * p_up, 1),
        "prob_profit_dn": round(100.0 * p_dn, 1),
        "d_prob_profit_d_edge": round(float(sens), 3),
        "d_median_R_d_edge": round(float(d_med), 3),
        "fragility": fragility,
        "layman": (
            "We nudge every trade's result a little better and a little worse, "
            "then see how much the chance of finishing green moves. "
            "If that chance swings a lot from a tiny nudge, the edge is fragile."
        ),
    }


def levy_malliavin_pack(trades: list[dict]) -> dict[str, Any]:
    return {
        "levy": levy_monte_carlo(trades),
        "malliavin": malliavin_sensitivity(trades),
    }
