"""Concept attribution + auto-weighting for Zenith.

Ranks every signal ingredient (EMA, SMC, AVP, zone, MC bucket, state, …) by
realized expectancy from closed trades, then:

1. Reweights score bits toward concepts that actually pay
2. Soft-blocks concepts with persistent negative expectancy
3. Produces a concept_quality score used as an entry gate (live + backtest)

This is walk-forward friendly: call ``ingest_closed_trade`` / ``rebuild`` after
each close so any timeline teaches the algo which of *its own* concepts work
in that regime — then the same weights drive live XM.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Score-bit concepts (used as weighted ingredients in buy/sell score)
SCORE_CONCEPTS = (
    "ema_trend",
    "laplacian",
    "laminar",
    "setup",
    "graetz",
    "macd",
    "rsi",
    "atr_ok",
    "bollinger",
    "bias_1h",
    "smc_ok",
    "smc_choch",
    "smc_disp",
    "smc_fvg",
    "ema_5m",
)

# Context / regime concepts (attribution + gates, not score bits)
CONTEXT_CONCEPTS = (
    "near_zone",
    "avp_aligned",
    "avp_in_value",
    "bias_4h_aligned",
    "state_laminar",
    "state_turbulent",
    "state_chop",
    "mc_high",
    "mc_mid",
    "mc_soft",
    "rr_2",
    "rr_3",
    "rr_4",
    "session_prime",
)

ALL_CONCEPTS = SCORE_CONCEPTS + CONTEXT_CONCEPTS

_DEFAULT_WEIGHT = 1.0
_MIN_SAMPLES = 8
_NEG_EXPECTANCY = -0.15  # R-multiples; soft-block below this
_STORE_PATH = Path(os.getenv("CONCEPT_STORE_PATH", "/app/data/concept_weights.json"))
if not _STORE_PATH.parent.exists():
    _STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "concept_weights.json"

_lock = threading.Lock()
_state: dict[str, Any] = {
    "weights": {c: _DEFAULT_WEIGHT for c in SCORE_CONCEPTS},
    "blocked": [],
    "stats": {},
    "ranking": [],
    "updates": 0,
    "message": "Concept engine awaiting closed trades",
    "min_quality": 0.35,
}


def _r_multiple(trade: dict) -> Optional[float]:
    entry = trade.get("entry")
    sl = trade.get("sl")
    pnl = trade.get("pnl")
    lot = float(trade.get("lot_size") or 0.1)
    if entry is None or sl is None or pnl is None:
        return None
    risk_price = abs(float(entry) - float(sl))
    if risk_price <= 0:
        return None
    # XAU: ~$10 PnL per $1 move per 0.1 lot (100 oz * 0.1)
    risk_usd = risk_price * lot * 100.0
    if risk_usd <= 0:
        return None
    return float(pnl) / risk_usd


def _active_concepts_from_features(feats: dict) -> list[str]:
    """Resolve concept flags from a trade features blob."""
    concepts = feats.get("concepts") or feats.get("CONCEPTS") or {}
    if isinstance(concepts, str):
        try:
            concepts = json.loads(concepts)
        except json.JSONDecodeError:
            concepts = {}
    active: list[str] = []
    if isinstance(concepts, dict):
        for k, v in concepts.items():
            if v and k in ALL_CONCEPTS:
                active.append(k)
        return active
    # Fallback: infer coarse concepts from flat features
    if feats.get("STATE") in ("LAMINAR_BULL", "LAMINAR_BEAR"):
        active.append("state_laminar")
    elif feats.get("STATE") in ("TURBULENT_BULL", "TURBULENT_BEAR"):
        active.append("state_turbulent")
    elif feats.get("STATE") == "CHOP":
        active.append("state_chop")
    mc = feats.get("mc_prob") or feats.get("markov_mc_prob")
    if mc is not None:
        mc = float(mc)
        if mc >= 55:
            active.append("mc_high")
        elif mc >= 45:
            active.append("mc_mid")
        else:
            active.append("mc_soft")
    if feats.get("vp_aligned") in (1, 1.0, True):
        active.append("avp_aligned")
    if feats.get("vp_in_value") in (1, 1.0, True):
        active.append("avp_in_value")
    smc = feats.get("smc_score")
    if smc is not None and float(smc) >= 2.5:
        active.append("smc_ok")
    return active


def build_concept_flags(
    *,
    direction: str,
    ema_align: bool,
    lap_align: bool,
    laminar_ok: bool,
    setup: bool,
    graetz_align: bool,
    macd_align: bool,
    rsi_ok: bool,
    atr_ok: bool,
    bb_align: bool,
    bias_1h_align: bool,
    smc_ok: bool,
    smc_choch: bool,
    smc_disp: bool,
    smc_fvg: bool,
    ema_5m_align: bool,
    near_zone: bool,
    avp_aligned: bool,
    avp_in_value: bool,
    bias_4h_align: bool,
    state: str,
    mc_prob: Optional[float],
    rr: Optional[float],
    session_name: str = "",
) -> dict[str, bool]:
    """Bitmask of active concepts at entry — stored on every trade."""
    st = state or ""
    mc = float(mc_prob) if mc_prob is not None else None
    rr_n = int(rr) if rr is not None else None
    session = (session_name or "").upper()
    prime = any(x in session for x in ("NY", "LONDON", "SCALP", "POWER", "OVERLAP"))

    flags = {
        "ema_trend": bool(ema_align),
        "laplacian": bool(lap_align),
        "laminar": bool(laminar_ok),
        "setup": bool(setup),
        "graetz": bool(graetz_align),
        "macd": bool(macd_align),
        "rsi": bool(rsi_ok),
        "atr_ok": bool(atr_ok),
        "bollinger": bool(bb_align),
        "bias_1h": bool(bias_1h_align),
        "smc_ok": bool(smc_ok),
        "smc_choch": bool(smc_choch),
        "smc_disp": bool(smc_disp),
        "smc_fvg": bool(smc_fvg),
        "ema_5m": bool(ema_5m_align),
        "near_zone": bool(near_zone),
        "avp_aligned": bool(avp_aligned),
        "avp_in_value": bool(avp_in_value),
        "bias_4h_aligned": bool(bias_4h_align),
        "state_laminar": "LAMINAR" in st,
        "state_turbulent": "TURBULENT" in st,
        "state_chop": st == "CHOP" or "TRANSITION" in st,
        "mc_high": bool(mc is not None and mc >= 55),
        "mc_mid": bool(mc is not None and 45 <= mc < 55),
        "mc_soft": bool(mc is not None and mc < 45),
        "rr_2": rr_n == 2,
        "rr_3": rr_n == 3,
        "rr_4": rr_n == 4,
        "session_prime": prime,
    }
    # direction unused except for future per-side stats
    _ = direction
    return flags


def weighted_score(flags: dict[str, bool], side_bits: dict[str, bool]) -> float:
    """Sum weight * bit for score concepts present in side_bits."""
    weights = get_weights()
    total = 0.0
    for name, on in side_bits.items():
        if not on:
            continue
        total += float(weights.get(name, _DEFAULT_WEIGHT))
    # Always include any SCORE_CONCEPTS flipped in flags if side_bits omitted keys
    for name in SCORE_CONCEPTS:
        if name in side_bits:
            continue
        if flags.get(name):
            total += float(weights.get(name, _DEFAULT_WEIGHT))
    return total


def concept_quality(flags: dict[str, bool]) -> tuple[float, str]:
    """
    0–1 quality from active concepts' learned expectancy.
    Unknown concepts (cold start) contribute neutral 0.5.
    """
    with _lock:
        stats = _state.get("stats") or {}
        blocked = set(_state.get("blocked") or [])
        min_q = float(_state.get("min_quality") or 0.35)

    active = [c for c, on in flags.items() if on and c in ALL_CONCEPTS]
    if not active:
        return 0.5, "No concept flags"

    # Hard soft-block: if a blocked concept is the only location edge, fail
    hit_blocked = [c for c in active if c in blocked]
    scores: list[float] = []
    for c in active:
        st = stats.get(c) or {}
        n = int(st.get("n") or 0)
        if n < _MIN_SAMPLES:
            scores.append(0.55)  # slight prior optimism until learned
            continue
        exp = float(st.get("expectancy") or 0.0)
        # Map expectancy in [-1, +1] → quality in [0, 1]
        q = 0.5 + 0.5 * max(-1.0, min(1.0, exp))
        if c in blocked:
            q *= 0.25
        scores.append(q)

    quality = sum(scores) / len(scores)
    # Require core edge concepts when learned
    core = [c for c in ("near_zone", "smc_ok", "avp_aligned", "setup") if flags.get(c)]
    if core and stats:
        core_exps = []
        for c in core:
            st = stats.get(c) or {}
            if int(st.get("n") or 0) >= _MIN_SAMPLES:
                core_exps.append(float(st.get("expectancy") or 0))
        if core_exps and max(core_exps) < _NEG_EXPECTANCY:
            return 0.0, f"Core concepts negative expectancy ({', '.join(core)})"

    if hit_blocked and quality < min_q + 0.1:
        return quality, f"Weak concepts active: {', '.join(hit_blocked[:3])}"

    return quality, "Concept quality OK"


def passes_concept_gate(flags: dict[str, bool]) -> tuple[bool, float, str]:
    # Historical bars must not inherit the live desk's learned blocks / negative
    # expectancy — that freezes every backtest at 0 trades. Walk-forward learning
    # inside a backtest still works via rebuild(closed_sim) when the caller opts in.
    try:
        from analysis.session import get_backtest_bar_time

        if get_backtest_bar_time() is not None:
            return True, 0.55, "Backtest cold prior (live concept memory ignored)"
    except Exception:
        pass
    q, reason = concept_quality(flags)
    with _lock:
        min_q = float(_state.get("min_quality") or 0.35)
    if q < min_q:
        return False, q, f"Concept quality {q:.2f} < {min_q:.2f} — {reason}"
    return True, q, reason


def rebuild(closed: list[dict], min_samples: int = _MIN_SAMPLES) -> dict[str, Any]:
    """Recompute expectancy / weights / blocks from closed trades."""
    global _state

    buckets: dict[str, list[float]] = {c: [] for c in ALL_CONCEPTS}

    for row in closed:
        if row.get("status") != "closed" or row.get("signal") not in ("BUY", "SELL"):
            continue
        r_mult = _r_multiple(row)
        if r_mult is None:
            continue
        feats = row.get("features") or {}
        if isinstance(feats, str):
            try:
                feats = json.loads(feats)
            except json.JSONDecodeError:
                feats = {}
        for c in _active_concepts_from_features(feats):
            buckets[c].append(r_mult)

    stats: dict[str, dict] = {}
    ranking: list[dict] = []
    blocked: list[str] = []
    weights: dict[str, float] = {c: _DEFAULT_WEIGHT for c in SCORE_CONCEPTS}

    for c, samples in buckets.items():
        n = len(samples)
        if n == 0:
            continue
        mean_r = sum(samples) / n
        wins = sum(1 for x in samples if x > 0.05)
        wr = wins / n if n else 0.0
        # Shrink toward 0 when few samples
        shrink = n / (n + min_samples)
        exp = mean_r * shrink
        stats[c] = {
            "n": n,
            "expectancy": round(exp, 4),
            "mean_r": round(mean_r, 4),
            "win_rate": round(wr, 4),
            "total_r": round(sum(samples), 3),
        }
        ranking.append({"concept": c, **stats[c]})

        if c in SCORE_CONCEPTS and n >= min_samples:
            # Weight ∈ [0.25, 2.0] from expectancy
            w = 1.0 + max(-0.75, min(1.0, exp))
            weights[c] = round(max(0.25, min(2.0, w)), 3)

        if n >= min_samples and exp <= _NEG_EXPECTANCY:
            blocked.append(c)

    ranking.sort(key=lambda x: x["expectancy"], reverse=True)

    message = (
        f"Concepts learned from {sum(1 for r in closed if r.get('status')=='closed')} closes — "
        f"top: {ranking[0]['concept']} ({ranking[0]['expectancy']:+.2f}R)"
        if ranking
        else "Concept engine awaiting closed trades"
    )

    new_state = {
        "weights": weights,
        "blocked": blocked,
        "stats": stats,
        "ranking": ranking[:20],
        "updates": _state.get("updates", 0) + 1,
        "message": message,
        "min_quality": float(os.getenv("CONCEPT_MIN_QUALITY", "0.35")),
    }
    with _lock:
        _state = new_state
    _persist(new_state)
    return new_state


def get_weights() -> dict[str, float]:
    with _lock:
        return dict(_state.get("weights") or {})


def get_state() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def get_ranking() -> list[dict]:
    with _lock:
        return list(_state.get("ranking") or [])


def snapshot() -> dict[str, Any]:
    """Copy current state (so backtest walk-forward can restore live weights after)."""
    with _lock:
        return json.loads(json.dumps(_state))


def restore(snap: dict[str, Any]) -> None:
    global _state
    if not snap:
        return
    with _lock:
        _state = snap


def _persist(state: dict) -> None:
    try:
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": state.get("weights"),
            "blocked": state.get("blocked"),
            "stats": state.get("stats"),
            "ranking": state.get("ranking"),
            "updates": state.get("updates"),
            "min_quality": state.get("min_quality"),
        }
        _STORE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("concept persist skipped: %s", exc)


def load() -> dict[str, Any]:
    """Load persisted weights at startup."""
    global _state
    try:
        if _STORE_PATH.exists():
            data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
            with _lock:
                _state.update(
                    {
                        "weights": data.get("weights") or _state["weights"],
                        "blocked": data.get("blocked") or [],
                        "stats": data.get("stats") or {},
                        "ranking": data.get("ranking") or [],
                        "updates": int(data.get("updates") or 0),
                        "min_quality": float(data.get("min_quality") or 0.35),
                        "message": "Loaded concept weights from disk",
                    }
                )
            return get_state()
    except Exception as exc:
        logger.warning("concept load failed: %s", exc)
    return get_state()


# Load on import
try:
    load()
except Exception:
    pass
