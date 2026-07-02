"""Learned Markov state transitions + per-state drift/vol feedback for Monte Carlo."""

from __future__ import annotations

import logging
import threading
import time
from copy import deepcopy
from typing import Any, Optional

import numpy as np

from analysis.trade_outcome import is_breakeven_trade, is_winning_trade
from config import STATE_PARAMS as DEFAULT_STATE_PARAMS

logger = logging.getLogger(__name__)

MARKET_STATES = [
    "LAMINAR_BULL",
    "LAMINAR_BEAR",
    "TRANSITION_BULL",
    "TRANSITION_BEAR",
    "TURBULENT_BULL",
    "TURBULENT_BEAR",
    "CHOP",
]

BULL_STATES = {"LAMINAR_BULL", "TRANSITION_BULL", "TURBULENT_BULL"}
BEAR_STATES = {"LAMINAR_BEAR", "TRANSITION_BEAR", "TURBULENT_BEAR"}

MIN_TRANSITION_SAMPLES = 20
MIN_STATE_TRADE_SAMPLES = 15
LEARN_BLEND = 0.55  # weight on learned params vs defaults


def _align_state_for_direction(state: str, direction: str) -> str:
    if direction == "BUY" and "BEAR" in state:
        return state.replace("BEAR", "BULL")
    if direction == "SELL" and "BULL" in state:
        return state.replace("BULL", "BEAR")
    return state


def _uniform_transitions() -> dict[str, dict[str, float]]:
    n = len(MARKET_STATES)
    p = 1.0 / n
    return {s: {t: p for t in MARKET_STATES} for s in MARKET_STATES}


def learn_transition_matrix(state_sequence: list[str]) -> dict[str, dict[str, float]]:
    """Estimate P(next_state | current_state) with Laplace smoothing."""
    counts: dict[str, dict[str, float]] = {s: {t: 1.0 for t in MARKET_STATES} for s in MARKET_STATES}
    valid = [s for s in state_sequence if s in MARKET_STATES]
    for i in range(len(valid) - 1):
        cur, nxt = valid[i], valid[i + 1]
        counts[cur][nxt] += 1.0

    matrix: dict[str, dict[str, float]] = {}
    for state in MARKET_STATES:
        row = counts[state]
        total = sum(row.values())
        matrix[state] = {t: row[t] / total for t in MARKET_STATES}
    return matrix


def learn_state_params_from_trades(
    trades: list[dict],
    defaults: Optional[dict] = None,
) -> dict[str, dict[str, float]]:
    """Feedback loop: derive drift/vol per state from realised trade outcomes."""
    base = defaults or DEFAULT_STATE_PARAMS
    buckets: dict[str, list[float]] = {s: [] for s in MARKET_STATES}
    vol_buckets: dict[str, list[float]] = {s: [] for s in MARKET_STATES}

    for row in trades:
        if row.get("status") != "closed" or row.get("signal") not in ("BUY", "SELL"):
            continue
        feats = row.get("features") or {}
        if isinstance(feats, str):
            import json
            try:
                feats = json.loads(feats)
            except json.JSONDecodeError:
                feats = {}
        state = feats.get("STATE")
        if state not in MARKET_STATES:
            continue
        entry = row.get("entry")
        sl = row.get("sl")
        pnl = row.get("pnl")
        if entry is None or sl is None or pnl is None:
            continue
        risk = abs(float(entry) - float(sl))
        if risk <= 0:
            continue
        direction = 1.0 if row.get("signal") == "BUY" else -1.0
        r_mult = (float(pnl) / risk) if risk else 0.0
        signed_move = direction * r_mult
        buckets[state].append(signed_move)
        vol_buckets[state].append(abs(signed_move))

    learned: dict[str, dict[str, float]] = {}
    for state in MARKET_STATES:
        default = base.get(state, base["CHOP"])
        moves = buckets[state]
        vols = vol_buckets[state]
        if len(moves) < MIN_STATE_TRADE_SAMPLES:
            learned[state] = {"drift": default["drift"], "vol": default["vol"], "samples": len(moves)}
            continue
        mean_r = float(np.mean(moves))
        std_r = float(np.std(moves)) if len(vols) > 1 else default["vol"]
        drift = float(np.clip(mean_r * 0.12, -0.35, 0.35))
        vol = float(np.clip(max(std_r * 0.85, 0.45), 0.45, 2.2))
        learned[state] = {"drift": round(drift, 4), "vol": round(vol, 4), "samples": len(moves)}
    return learned


def blend_state_params(
    learned: dict[str, dict[str, float]],
    defaults: Optional[dict] = None,
    alpha: float = LEARN_BLEND,
) -> dict[str, dict[str, float]]:
    base = defaults or DEFAULT_STATE_PARAMS
    blended: dict[str, dict[str, float]] = {}
    for state in MARKET_STATES:
        d = base.get(state, base["CHOP"])
        l = learned.get(state, d)
        blended[state] = {
            "drift": round((1 - alpha) * d["drift"] + alpha * l["drift"], 4),
            "vol": round((1 - alpha) * d["vol"] + alpha * l["vol"], 4),
        }
    return blended


def state_win_rates(trades: list[dict]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for state in MARKET_STATES:
        subset = []
        for row in trades:
            if row.get("status") != "closed":
                continue
            feats = row.get("features") or {}
            if isinstance(feats, str):
                import json
                try:
                    feats = json.loads(feats)
                except json.JSONDecodeError:
                    feats = {}
            if feats.get("STATE") == state:
                subset.append(row)
        if not subset:
            stats[state] = {"count": 0, "win_rate": None}
            continue
        decisive = [r for r in subset if not is_breakeven_trade(r)]
        wins = sum(1 for r in decisive if is_winning_trade(r))
        stats[state] = {
            "count": len(subset),
            "win_rate": round(wins / len(decisive), 4) if decisive else None,
        }
    return stats


def _sample_next_state(
    current: str,
    matrix: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> str:
    row = matrix.get(current, matrix.get("CHOP", {}))
    states = list(row.keys())
    probs = np.array([row[s] for s in states], dtype=float)
    probs /= probs.sum()
    return str(rng.choice(states, p=probs))


def markov_monte_carlo_probability(
    entry: float,
    sl: float,
    tp: float,
    start_state: str,
    atr_now: float,
    transition_matrix: dict[str, dict[str, float]],
    state_params: dict[str, dict[str, float]],
    simulations: int = 500,
    steps: int = 12,
    seed: Optional[int] = None,
) -> float:
    """Simulate paths where market state evolves via learned Markov transitions."""
    direction = "BUY" if tp > entry else "SELL"
    mc_state = _align_state_for_direction(start_state, direction)
    rng = np.random.default_rng(seed)

    tp_hits = 0
    sl_hits = 0

    for _ in range(simulations):
        price = float(entry)
        state = mc_state
        for _ in range(steps):
            params = state_params.get(state, state_params.get("CHOP", {"drift": 0.0, "vol": 1.0}))
            drift = params["drift"] * atr_now
            vol = params["vol"] * atr_now
            price += float(rng.normal(loc=drift, scale=vol))

            if direction == "BUY":
                if price >= tp:
                    tp_hits += 1
                    break
                if price <= sl:
                    sl_hits += 1
                    break
            else:
                if price <= tp:
                    tp_hits += 1
                    break
                if price >= sl:
                    sl_hits += 1
                    break
            state = _sample_next_state(state, transition_matrix, rng)

    total = tp_hits + sl_hits
    if total == 0:
        return 0.0
    return round((tp_hits / total) * 100, 2)


def markov_features(
    current_state: str,
    direction: str,
    transition_matrix: dict[str, dict[str, float]],
    win_rates: dict[str, dict[str, Any]],
    markov_mc_prob: float,
    static_mc_prob: float,
) -> dict[str, Any]:
    row = transition_matrix.get(current_state, transition_matrix.get("CHOP", {}))
    persist = row.get(current_state, 0.0)
    bull_prob = sum(row.get(s, 0.0) for s in BULL_STATES)
    bear_prob = sum(row.get(s, 0.0) for s in BEAR_STATES)
    aligned = _align_state_for_direction(current_state, direction)
    state_wr = win_rates.get(aligned, {}).get("win_rate")

    return {
        "markov_persist": round(persist, 4),
        "markov_bull_prob": round(bull_prob, 4),
        "markov_bear_prob": round(bear_prob, 4),
        "markov_mc_prob": markov_mc_prob,
        "static_mc_prob": static_mc_prob,
        "markov_mc_delta": round(markov_mc_prob - static_mc_prob, 2),
        "state_win_rate": state_wr,
        "STATE": current_state,
        "ALIGNED_STATE": aligned,
    }


def calibration_error(trades: list[dict]) -> dict[str, Any]:
    """Compare predicted MC vs realised outcomes for closed trades."""
    pairs: list[tuple[float, float]] = []
    for row in trades:
        if row.get("status") != "closed":
            continue
        feats = row.get("features") or {}
        if isinstance(feats, str):
            import json
            try:
                feats = json.loads(feats)
            except json.JSONDecodeError:
                feats = {}
        pred = feats.get("markov_mc_prob") or feats.get("mc_prob") or feats.get("SUCCESS_PERCENT")
        if pred is None:
            continue
        if is_breakeven_trade(row):
            continue
        won = 1.0 if is_winning_trade(row) else 0.0
        pairs.append((float(pred) / 100.0, won))

    if len(pairs) < 30:
        return {"samples": len(pairs), "mean_error": None, "brier": None}

    preds = np.array([p[0] for p in pairs])
    actual = np.array([p[1] for p in pairs])
    return {
        "samples": len(pairs),
        "mean_error": round(float(np.mean(preds - actual)), 4),
        "brier": round(float(np.mean((preds - actual) ** 2)), 4),
        "predicted_wr": round(float(np.mean(preds)), 4),
        "actual_wr": round(float(np.mean(actual)), 4),
    }


def build_markov_model(
    snapshots: list[dict],
    trades: list[dict],
) -> dict[str, Any]:
    """Full feedback model from historical signal states + trade outcomes."""
    states = [s.get("state") for s in snapshots if s.get("state") in MARKET_STATES]
    if len(states) < MIN_TRANSITION_SAMPLES:
        transitions = _uniform_transitions()
        transition_source = "default_uniform"
    else:
        transitions = learn_transition_matrix(states)
        transition_source = "learned"

    learned_params = learn_state_params_from_trades(trades)
    blended_params = blend_state_params(learned_params)
    wr = state_win_rates(trades)
    cal = calibration_error(trades)

    top_transitions = []
    for from_state in MARKET_STATES:
        row = transitions[from_state]
        best = max(row.items(), key=lambda x: x[1])
        top_transitions.append({
            "from": from_state,
            "to": best[0],
            "prob": round(best[1], 3),
        })

    return {
        "transition_matrix": transitions,
        "transition_source": transition_source,
        "state_sequence_length": len(states),
        "learned_state_params": learned_params,
        "blended_state_params": blended_params,
        "state_win_rates": wr,
        "calibration": cal,
        "top_transitions": sorted(top_transitions, key=lambda x: x["prob"], reverse=True)[:5],
        "updated_at": time.time(),
    }


class MarkovStore:
    """Thread-safe in-memory store read by signal/MC during live ticks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: dict[str, Any] = self._default_model()

    @staticmethod
    def _default_model() -> dict[str, Any]:
        return {
            "transition_matrix": _uniform_transitions(),
            "transition_source": "default_uniform",
            "state_sequence_length": 0,
            "learned_state_params": {},
            "blended_state_params": deepcopy(DEFAULT_STATE_PARAMS),
            "state_win_rates": {s: {"count": 0, "win_rate": None} for s in MARKET_STATES},
            "calibration": {"samples": 0, "mean_error": None, "brier": None},
            "top_transitions": [],
            "updated_at": 0.0,
        }

    def update(self, model: dict[str, Any]) -> None:
        with self._lock:
            self._model = model

    def get(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._model)

    def blended_params(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return deepcopy(self._model.get("blended_state_params", DEFAULT_STATE_PARAMS))

    def transition_matrix(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return deepcopy(self._model.get("transition_matrix", _uniform_transitions()))


markov_store = MarkovStore()
