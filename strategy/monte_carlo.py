"""Monte Carlo simulation for trade outcome probability."""

from typing import Optional

import numpy as np

from config import (
    ALLOW_SMC_STRUCTURE_OVERRIDE,
    MC_SIMULATIONS,
    MC_STEPS,
    MIN_SMC_SCORE,
    MINUTE_RR_CHOICES,
    RR_CHOICES,
    STATE_PARAMS,
)
from analysis.markov_engine import markov_monte_carlo_probability, markov_store


def _align_state_for_direction(state: str, direction: str) -> str:
    """Use drift aligned with trade direction so MC isn't biased against the setup."""
    if direction == "BUY" and "BEAR" in state:
        return state.replace("BEAR", "BULL")
    if direction == "SELL" and "BULL" in state:
        return state.replace("BULL", "BEAR")
    return state


def _structure_valid(direction: str, tp: float, zones: dict) -> bool:
    """TP may reach the opposite zone but should not blow through it."""
    if direction == "BUY":
        return tp <= zones["s_buf_high"]
    return tp >= zones["d_buf_low"]


def _active_state_params() -> dict:
    """Prefer learned Markov feedback params; fall back to config defaults."""
    blended = markov_store.blended_params()
    return blended if blended else STATE_PARAMS


def monte_carlo_probability(
    entry: float,
    sl: float,
    tp: float,
    state: str,
    atr_now: float,
    simulations: int = MC_SIMULATIONS,
    steps: int = MC_STEPS,
    use_markov: bool = True,
    seed: Optional[int] = None,
) -> float:
    direction = "BUY" if tp > entry else "SELL"
    mc_state = _align_state_for_direction(state, direction)
    params_map = _active_state_params()

    if use_markov:
        matrix = markov_store.transition_matrix()
        if matrix and markov_store.get().get("transition_source") == "learned":
            return markov_monte_carlo_probability(
                entry,
                sl,
                tp,
                mc_state,
                atr_now,
                matrix,
                params_map,
                simulations=simulations,
                steps=steps,
                seed=seed,
            )

    params = params_map.get(mc_state, params_map.get("CHOP", STATE_PARAMS["CHOP"]))
    drift = params["drift"] * atr_now
    vol = params["vol"] * atr_now

    tp_hits = 0
    sl_hits = 0

    rng = np.random.default_rng(seed)

    for _ in range(simulations):
        sim_price = entry

        for _ in range(steps):
            move = rng.normal(loc=drift, scale=vol)
            sim_price += move

            if direction == "BUY":
                if sim_price >= tp:
                    tp_hits += 1
                    break
                if sim_price <= sl:
                    sl_hits += 1
                    break
            else:
                if sim_price <= tp:
                    tp_hits += 1
                    break
                if sim_price >= sl:
                    sl_hits += 1
                    break

    total = tp_hits + sl_hits

    if total == 0:
        return 0.0

    return round((tp_hits / total) * 100, 2)


def static_monte_carlo_probability(
    entry: float,
    sl: float,
    tp: float,
    state: str,
    atr_now: float,
    simulations: int = MC_SIMULATIONS,
    steps: int = MC_STEPS,
    seed: Optional[int] = None,
) -> float:
    """Legacy fixed-state MC for comparison / calibration."""
    return monte_carlo_probability(
        entry, sl, tp, state, atr_now, simulations, steps, use_markov=False, seed=seed
    )


def choose_best_rr(
    direction: str,
    entry: float,
    sl: float,
    zones: dict,
    state: str,
    atr_now: float,
    smc_score: float = 0.0,
) -> Optional[dict]:
    best = None
    risk = abs(entry - sl)
    if risk <= 0:
        return None

    mc_state = _align_state_for_direction(state, direction)
    override_min = MIN_SMC_SCORE
    allow_override = ALLOW_SMC_STRUCTURE_OVERRIDE and smc_score >= override_min

    candidates: list[tuple[float, float]] = [(rr, rr) for rr in RR_CHOICES]

    zone_tp = zones["s_buf_low"] if direction == "BUY" else zones["d_buf_high"]
    zone_reward = abs(zone_tp - entry)
    if zone_reward > 0:
        zone_rr = round(zone_reward / risk, 2)
        if zone_rr >= 1.5:
            candidates.append((zone_rr, zone_rr))

    seen_rr: set[float] = set()
    for rr, rr_label in candidates:
        if rr in seen_rr:
            continue
        seen_rr.add(rr)

        tp = entry + rr * risk if direction == "BUY" else entry - rr * risk
        structure_ok = _structure_valid(direction, tp, zones)

        if not structure_ok and not allow_override:
            continue

        mc_prob = monte_carlo_probability(entry, sl, tp, mc_state, atr_now)
        smc_bonus = min(float(smc_score), 5.0) * 0.05
        expectancy = (mc_prob / 100.0) * rr - (1 - mc_prob / 100.0) + smc_bonus

        if best is None or expectancy > best["expectancy"]:
            best = {
                "rr": rr_label,
                "tp": tp,
                "mc_prob": mc_prob,
                "expectancy": expectancy,
                "structure_valid": structure_ok,
            }

    return best


def choose_minute_rr(
    direction: str,
    entry: float,
    sl: float,
    state: str,
    atr_now: float,
    zones: Optional[dict] = None,
    seed: Optional[int] = None,
) -> Optional[dict]:
    """Try each RR in MINUTE_RR_CHOICES (1:2, 1:3, 1:4) and return the setup
    with the highest expected value.  All intermediate TP levels are populated
    so the trade-manager can apply staged partial closes."""
    risk = abs(entry - sl)
    if risk <= 0:
        risk = atr_now * MINUTE_RR_CHOICES[0]
        sl = entry - risk if direction == "BUY" else entry + risk

    mc_state = _align_state_for_direction(state, direction)
    best: Optional[dict] = None

    for rr in MINUTE_RR_CHOICES:
        rr_f = float(rr)
        tp_final = round(entry + rr_f * risk, 2) if direction == "BUY" else round(entry - rr_f * risk, 2)

        # Intermediate levels: 1R = TP1, 2R = TP2 (for 1:3/1:4), 3R = TP3 (for 1:4)
        tp1 = round(entry + risk, 2) if direction == "BUY" else round(entry - risk, 2)
        if rr >= 3:
            tp2 = round(entry + 2 * risk, 2) if direction == "BUY" else round(entry - 2 * risk, 2)
        else:
            tp2 = None
        if rr >= 4:
            tp3 = round(entry + 3 * risk, 2) if direction == "BUY" else round(entry - 3 * risk, 2)
        else:
            tp3 = None

        # MC probability measured against the FINAL TP (hardest target — most conservative)
        static_prob = static_monte_carlo_probability(entry, sl, tp_final, mc_state, atr_now, seed=seed)
        markov_prob = monte_carlo_probability(entry, sl, tp_final, mc_state, atr_now, seed=seed)
        mc_prob = markov_prob
        expectancy = (mc_prob / 100.0) * rr_f - (1 - mc_prob / 100.0)

        if best is None or expectancy > best["expectancy"]:
            best = {
                "rr": rr_f,
                "rr_int": rr,
                "tp": tp_final,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "mc_prob": mc_prob,
                "static_mc_prob": static_prob,
                "expectancy": expectancy,
                "tp_source": f"multi_rr_{rr}",
            }

    return best
