"""Monte Carlo simulation for trade outcome probability."""

import math
from typing import Optional, Union

import numpy as np

from config import (
    ALLOW_SMC_STRUCTURE_OVERRIDE,
    MC_FLOOR_BASE,
    MC_RR_PREMIUM,
    MC_SIMULATIONS,
    MC_STEPS,
    MIN_SMC_SCORE,
    MINUTE_RR_CHOICES,
    RR_CHOICES,
    STATE_PARAMS,
)
from analysis.markov_engine import markov_monte_carlo_probability, markov_store


def _align_state_for_direction(state: str, direction: str) -> str:
    """Keep MC honest — do not flip BULL/BEAR to match the trade direction.

    Direction bias belongs in entry filters, not in the probability model.
    """
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
    forced_rr: Optional[Union[int, float]] = None,
    symbol: Optional[str] = None,
    prefer_rr: Optional[float] = None,
    structure_tp: Optional[float] = None,
    structure_mode: Optional[str] = None,
) -> Optional[dict]:
    """Try each RR in MINUTE_RR_CHOICES (1:2, 1:3, 1:4) and return the setup
    with the highest *staged* expected value — unless forced_rr / SYMBOL_RR is set.

    Staged EV uses MC hit-rates at 1R/2R/3R/4R plus RR_TP_FRACS so multi-TP
    partials are valued correctly (not just final-TP MC). Intermediate TP1/TP2/TP3
    are always populated for the trade manager (partials + dynamic trail).

    prefer_rr / symbol RL overlays bias toward 1:3–1:4 after losses.

    When ``structure_tp`` is set (structure brain), prefer that target if RR is
    in band and MC clears a soft floor — walk-forward winner for GOLD/USOIL.

    ``forced_rr`` may be float (1.5 / 2.4 / 4) from the SYMBOL_RR research map.
    """
    from config import (
        RR_TP_FRACS,
        STRUCTURE_TP_MODE,
        STRUCTURE_TP_MIN_RR,
        STRUCTURE_TP_MAX_RR,
        rr_profile_int,
        symbol_rr,
    )

    # Auto-pin from desk SYMBOL_RR when caller didn't force
    if forced_rr is None and symbol:
        try:
            mapped = float(symbol_rr(symbol, default=0.0) or 0.0)
            if mapped > 0:
                forced_rr = mapped
        except Exception:
            pass

    if prefer_rr is None and symbol:
        try:
            from analysis.instrument_rl import overlays_for

            prefer_rr = float(overlays_for(symbol).get("rr_prefer") or 3.0)
        except Exception:
            prefer_rr = 3.0
    if prefer_rr is None:
        prefer_rr = 3.0
    prefer_rr = float(prefer_rr)
    mode = (structure_mode or STRUCTURE_TP_MODE or "prefer").strip().lower()

    risk = abs(entry - sl)
    if risk <= 0:
        risk = atr_now * MINUTE_RR_CHOICES[0]
        sl = entry - risk if direction == "BUY" else entry + risk

    mc_state = _align_state_for_direction(state, direction)

    def _tp_at(r_mult: float) -> float:
        # Keep full float precision; callers round for the instrument's digits
        return (
            entry + r_mult * risk
            if direction == "BUY"
            else entry - r_mult * risk
        )

    forced_f = float(forced_rr) if forced_rr is not None else None
    # Hit probabilities before SL at each R (shared across profiles)
    max_r = (
        max(int(math.ceil(forced_f)), max(MINUTE_RR_CHOICES))
        if forced_f is not None
        else max(MINUTE_RR_CHOICES)
    )
    p_at: dict[int, float] = {}
    static_at: dict[int, float] = {}
    for r in range(1, max_r + 1):
        tp_r = _tp_at(float(r))
        static_at[r] = static_monte_carlo_probability(
            entry, sl, tp_r, mc_state, atr_now, seed=seed
        )
        p_at[r] = monte_carlo_probability(
            entry, sl, tp_r, mc_state, atr_now, seed=seed
        ) / 100.0

    def _staged_ev(rr: int) -> float:
        """EV in R-multiples with sequential partials + BE after TP1."""
        fracs = RR_TP_FRACS.get(rr, RR_TP_FRACS[2])
        p1 = p_at.get(1, 0.0)
        # Full stop before any partial
        ev = -(1.0 - p1)
        # Book each intermediate fraction at its R level when that level is reached
        booked = 0.0
        for i, frac in enumerate(fracs):
            r_lvl = i + 1
            pr = p_at.get(r_lvl, 0.0)
            ev += pr * frac * float(r_lvl)
            booked += frac
        # Runner to final TP (rr)
        runner = max(0.0, 1.0 - booked)
        if runner > 0:
            ev += p_at.get(rr, 0.0) * runner * float(rr)
        # Soft prefer not stretching too far when final hit-rate is weak
        if rr > 2:
            final_p = p_at.get(rr, 0.0) * 100.0
            soft_floor = MC_FLOOR_BASE - MC_RR_PREMIUM  # e.g. 42-2=40
            if final_p < soft_floor:
                ev -= 0.15 * (rr - 2)
        # RL bias: prefer ≥ prefer_rr (typically 3 after losses)
        if rr + 0.01 >= prefer_rr:
            ev += 0.10 + 0.04 * max(0.0, prefer_rr - 2.0)
        elif prefer_rr >= 3 and rr < prefer_rr:
            # Penalize pure 1:2 when policy wants runners + multi-TP
            ev -= 0.08 * (prefer_rr - rr)
        return ev

    # Pin exact float RR from SYMBOL_RR / caller (1.5, 2.4, 3, 4, …)
    if forced_f is not None and forced_f > 0:
        rr_i = rr_profile_int(forced_f)
        tp_final = _tp_at(forced_f)
        p_final = monte_carlo_probability(
            entry, sl, tp_final, mc_state, atr_now, seed=seed
        ) / 100.0
        p1 = p_at.get(1, 0.0)
        mc_prob = (0.55 * p1 + 0.45 * p_final) * 100.0
        static_final = static_monte_carlo_probability(
            entry, sl, tp_final, mc_state, atr_now, seed=seed
        )
        static_prob = 0.55 * float(static_at.get(1, 0.0)) + 0.45 * float(static_final)
        base_ev = _staged_ev(rr_i)
        if rr_i > 0:
            base_ev *= float(forced_f) / float(rr_i)
        return {
            "rr": round(float(forced_f), 2),
            "rr_int": rr_i,
            "tp": tp_final,
            "tp1": _tp_at(1.0),
            "tp2": _tp_at(2.0) if forced_f >= 2.0 else None,
            "tp3": _tp_at(3.0) if forced_f >= 3.0 else None,
            "mc_prob": round(mc_prob, 2),
            "static_mc_prob": round(float(static_prob), 2),
            "final_mc_prob": round(p_final * 100.0, 2),
            "expectancy": base_ev,
            "tp_source": f"symbol_rr_{forced_f:g}",
            "p_1r": round(p1 * 100.0, 1),
            "p_2r": round(p_at.get(2, 0.0) * 100.0, 1),
            "prefer_rr": prefer_rr,
        }

    choices = list(MINUTE_RR_CHOICES)
    # When preferring ≥3, evaluate higher RRs first but still pick by EV
    if prefer_rr >= 3:
        choices = sorted(choices, key=lambda x: (0 if x >= int(prefer_rr) else 1, -x))

    best: Optional[dict] = None
    for rr in choices:
        rr = int(rr)
        tp1 = _tp_at(1.0)
        tp2 = _tp_at(2.0) if rr >= 3 else None
        tp3 = _tp_at(3.0) if rr >= 4 else None
        tp_final = _tp_at(float(rr))
        # Blend TP1 + final hit-rates for the trade gate / UI MC%.
        # Pure final-TP MC understates multi-TP (partials lock value at 1R).
        p1 = p_at.get(1, 0.0)
        pf = p_at.get(rr, 0.0)
        mc_prob = (0.55 * p1 + 0.45 * pf) * 100.0
        static_prob = 0.55 * float(static_at.get(1, 0.0)) + 0.45 * float(static_at.get(rr, 0.0))
        expectancy = _staged_ev(rr)

        # Still skip absurdly weak setups unless forced
        if mc_prob < max(28.0, MC_FLOOR_BASE - 12.0):
            continue

        if best is None or expectancy > best["expectancy"]:
            best = {
                "rr": float(rr),
                "rr_int": rr,
                "tp": tp_final,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "mc_prob": round(mc_prob, 2),
                "static_mc_prob": round(float(static_prob), 2),
                "final_mc_prob": round(pf * 100.0, 2),
                "expectancy": expectancy,
                "tp_source": f"multi_rr_{rr}",
                "p_1r": round(p1 * 100.0, 1),
                "p_2r": round(p_at.get(2, 0.0) * 100.0, 1),
                "prefer_rr": prefer_rr,
            }

    # Fallback: prefer_rr first, else 1:2
    if best is None:
        for rr in (int(prefer_rr) if int(prefer_rr) in MINUTE_RR_CHOICES else 3, 3, 2):
            if rr not in MINUTE_RR_CHOICES:
                continue
            best = {
                "rr": float(rr),
                "rr_int": rr,
                "tp": _tp_at(float(rr)),
                "tp1": _tp_at(1.0),
                "tp2": _tp_at(2.0) if rr >= 3 else None,
                "tp3": _tp_at(3.0) if rr >= 4 else None,
                "mc_prob": round(p_at.get(rr, 0.0) * 100.0, 2),
                "static_mc_prob": round(float(static_at.get(rr, 0.0)), 2),
                "expectancy": _staged_ev(rr),
                "tp_source": f"multi_rr_{rr}",
                "prefer_rr": prefer_rr,
            }
            break

    # Structure brain TP — preferred exit when RR lands in the validated band.
    if (
        best is not None
        and structure_tp is not None
        and mode in ("prefer", "hybrid")
        and risk > 0
    ):
        stp = float(structure_tp)
        struct_rr = abs(stp - entry) / risk
        if STRUCTURE_TP_MIN_RR - 1e-9 <= struct_rr <= STRUCTURE_TP_MAX_RR + 1e-9:
            # Map continuous structure RR onto staged grid EV (nearest int for fracs)
            rr_nearest = int(min(MINUTE_RR_CHOICES, key=lambda x: abs(x - struct_rr)))
            # MC at the actual structure TP price
            p_struct = monte_carlo_probability(
                entry, sl, stp, mc_state, atr_now, seed=seed
            ) / 100.0
            p1 = p_at.get(1, 0.0)
            mc_struct = (0.55 * p1 + 0.45 * p_struct) * 100.0
            static_struct = 0.55 * float(static_at.get(1, 0.0)) + 0.45 * float(
                static_monte_carlo_probability(
                    entry, sl, stp, mc_state, atr_now, seed=seed
                )
            )
            # Staged EV proxy: scale nearest-profile EV by structure RR stretch
            ev_struct = _staged_ev(rr_nearest)
            # Mild bonus — structure magnets were OOS-superior vs blind grid
            ev_struct += 0.06
            # Soft floor: don't take structure into a coin-flip
            soft_ok = mc_struct >= max(28.0, MC_FLOOR_BASE - 12.0)
            take = False
            if soft_ok and mode == "prefer":
                take = True
            elif soft_ok and mode == "hybrid":
                take = ev_struct >= float(best.get("expectancy") or -9) - 0.05
            if take:
                best = {
                    "rr": round(float(struct_rr), 2),
                    "rr_int": rr_nearest,
                    "tp": stp,
                    "tp1": _tp_at(1.0),
                    "tp2": _tp_at(2.0) if rr_nearest >= 3 or struct_rr >= 2.5 else None,
                    "tp3": _tp_at(3.0) if rr_nearest >= 4 or struct_rr >= 3.5 else None,
                    "mc_prob": round(mc_struct, 2),
                    "static_mc_prob": round(float(static_struct), 2),
                    "final_mc_prob": round(p_struct * 100.0, 2),
                    "expectancy": ev_struct,
                    "tp_source": "structure",
                    "p_1r": round(p1 * 100.0, 1),
                    "p_2r": round(p_at.get(2, 0.0) * 100.0, 1),
                    "prefer_rr": prefer_rr,
                    "structure_rr": round(float(struct_rr), 2),
                }

    return best
