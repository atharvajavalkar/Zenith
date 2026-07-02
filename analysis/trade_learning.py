"""Learn patterns from closed trades and calibrate execution toward target win rate."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from analysis.adaptive_guard import check as adaptive_check, mc_floor, min_ml_prob
from analysis.trade_outcome import is_breakeven_trade, is_losing_trade, is_winning_trade
from config import CALIBRATION_TARGET_WIN_RATE, MIN_TRAIN_SAMPLES

logger = logging.getLogger(__name__)

TARGET_WIN_RATE = CALIBRATION_TARGET_WIN_RATE
FEATURE_NAMES = [
    "risk_usd",
    "rr_ratio",
    "is_buy",
    "prime_hour",
    "mc_prob",
    "markov_mc_prob",
    "markov_persist",
    "markov_bull_prob",
    "state_win_rate",
    "score_margin",
    "smc_score",
    "bias_aligned",
    "laminar_bull",
    "laminar_bear",
]

# Hard rules — only reject clearly bad setups (not normal ATR-sized stops)
HARD_RULES = [
    {
        "id": "risk_oversized",
        "label": "Oversized SL (>$6) wins only 26%",
        "check": lambda f: f.get("risk_usd") is not None and f["risk_usd"] > 6.0,
        "action": "reject",
        "reason": "SL risk > $6 — historically 26% win rate",
    },
]


def _parse_hour(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            dt = datetime.strptime(value.replace("+00:00", ""), fmt.replace("%z", ""))
            return dt.hour
        except ValueError:
            continue
    match = re.search(r"\b(\d{2}):\d{2}:\d{2}\b", value)
    if match:
        return int(match.group(1))
    return None


def extract_features_from_trade(row: dict) -> dict[str, Any]:
    """Build feature dict from a trade record row."""
    entry = row.get("entry")
    sl = row.get("sl")
    tp = row.get("tp")
    signal = row.get("signal", "")
    risk = abs(float(entry) - float(sl)) if entry is not None and sl is not None else None
    reward = abs(float(tp) - float(entry)) if tp is not None and entry is not None else None
    rr = (reward / risk) if risk and risk > 0 and reward is not None else None
    hour = _parse_hour(row.get("candle_time") or row.get("log_time"))
    if hour is None and row.get("created_at"):
        try:
            hour = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00")).hour
        except ValueError:
            hour = None

    features = row.get("features") or {}
    if isinstance(features, str):
        try:
            features = json.loads(features)
        except json.JSONDecodeError:
            features = {}

    return {
        "risk_usd": round(risk, 3) if risk is not None else None,
        "rr_ratio": round(rr, 3) if rr is not None else None,
        "is_buy": 1.0 if signal == "BUY" else 0.0,
        "hour_utc": hour,
        "prime_hour": 1.0 if hour in {6, 9, 10, 11} else 0.0,
        "mc_prob": _float(features.get("mc_prob") or features.get("SUCCESS_PERCENT")),
        "markov_mc_prob": _float(features.get("markov_mc_prob") or features.get("MARKOV_MC_PROB")),
        "markov_persist": _float(features.get("markov_persist") or features.get("MARKOV_PERSIST")),
        "markov_bull_prob": _float(features.get("markov_bull_prob") or features.get("MARKOV_BULL_PROB")),
        "state_win_rate": _float(features.get("state_win_rate") or features.get("STATE_WIN_RATE")),
        "score_margin": _float(features.get("score_margin")),
        "smc_score": _float(features.get("smc_score") or features.get("SMC_SCORE")),
        "bias_aligned": _float(features.get("bias_aligned")),
        "laminar_bull": 1.0 if features.get("STATE") == "LAMINAR_BULL" else 0.0,
        "laminar_bear": 1.0 if features.get("STATE") == "LAMINAR_BEAR" else 0.0,
        "signal": signal,
    }


def extract_features_from_signal(sig: dict, trade_sig: Optional[dict] = None) -> dict[str, Any]:
    """Build feature dict at execution time from live signal payload."""
    ts = trade_sig or sig
    entry = ts.get("ENTRY")
    sl = ts.get("SL")
    tp = ts.get("TP")
    signal = ts.get("SIGNAL") or sig.get("SIGNAL")
    risk = abs(float(entry) - float(sl)) if entry is not None and sl is not None else None
    reward = abs(float(tp) - float(entry)) if tp is not None and entry is not None else None
    rr = (reward / risk) if risk and risk > 0 and reward is not None else None

    buy_score = float(sig.get("BUY_SCORE") or 0)
    sell_score = float(sig.get("SELL_SCORE") or 0)
    bias_1h = sig.get("BIAS_1H")
    aligned = (
        (signal == "BUY" and bias_1h == "BULLISH")
        or (signal == "SELL" and bias_1h == "BEARISH")
    )

    hour = _parse_hour(sig.get("CANDLE_TIME") or sig.get("LIVE_TIME"))

    return {
        "risk_usd": round(risk, 3) if risk is not None else None,
        "rr_ratio": round(rr, 3) if rr is not None else None,
        "is_buy": 1.0 if signal == "BUY" else 0.0,
        "hour_utc": hour,
        "prime_hour": 1.0 if hour in {6, 9, 10, 11} else 0.0,
        "mc_prob": _float(ts.get("SUCCESS_PERCENT") or sig.get("SUCCESS_PERCENT")),
        "markov_mc_prob": _float(ts.get("MARKOV_MC_PROB") or sig.get("MARKOV_MC_PROB")),
        "markov_persist": _float(sig.get("MARKOV_PERSIST")),
        "markov_bull_prob": _float(sig.get("MARKOV_BULL_PROB")),
        "state_win_rate": _float(sig.get("STATE_WIN_RATE")),
        "score_margin": abs(buy_score - sell_score),
        "smc_score": _float(sig.get("SMC_SCORE")),
        "bias_aligned": 1.0 if aligned else 0.0,
        "laminar_bull": 1.0 if sig.get("STATE") == "LAMINAR_BULL" else 0.0,
        "laminar_bear": 1.0 if sig.get("STATE") == "LAMINAR_BEAR" else 0.0,
        "signal": signal,
        "STATE": sig.get("STATE"),
        "BUY_SCORE": buy_score,
        "SELL_SCORE": sell_score,
        "SUCCESS_PERCENT": _float(ts.get("SUCCESS_PERCENT") or sig.get("SUCCESS_PERCENT")),
        "MARKOV_MC_PROB": _float(ts.get("MARKOV_MC_PROB") or sig.get("MARKOV_MC_PROB")),
        "MARKOV_PERSIST": _float(sig.get("MARKOV_PERSIST")),
        "MARKOV_BULL_PROB": _float(sig.get("MARKOV_BULL_PROB")),
        "STATE_WIN_RATE": _float(sig.get("STATE_WIN_RATE")),
        "SMC_SCORE": _float(sig.get("SMC_SCORE")),
    }


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_vector(features: dict[str, Any], means: dict, stds: dict) -> np.ndarray:
    vec = []
    for name in FEATURE_NAMES:
        val = features.get(name)
        if val is None:
            val = means.get(name, 0.0)
        mean = means.get(name, 0.0)
        std = stds.get(name, 1.0) or 1.0
        vec.append((float(val) - mean) / std)
    return np.array(vec, dtype=float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticCalibrator:
    """Lightweight logistic regression — no sklearn dependency."""

    def __init__(self) -> None:
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.threshold: float = 0.55
        self.train_win_rate: Optional[float] = None
        self.filtered_win_rate: Optional[float] = None
        self.sample_count: int = 0

    def fit(self, rows: list[dict]) -> None:
        if len(rows) < MIN_TRAIN_SAMPLES:
            self.weights = None
            self.sample_count = len(rows)
            return

        X_raw = []
        y = []
        for row in rows:
            if is_breakeven_trade(row):
                continue
            feats = extract_features_from_trade(row)
            X_raw.append(feats)
            y.append(1.0 if is_winning_trade(row) else 0.0)

        if len(X_raw) < MIN_TRAIN_SAMPLES:
            self.weights = None
            self.sample_count = len(X_raw)
            return

        for name in FEATURE_NAMES:
            vals = [f.get(name) for f in X_raw if f.get(name) is not None]
            self.means[name] = float(np.mean(vals)) if vals else 0.0
            self.stds[name] = float(np.std(vals)) if vals else 1.0
            if self.stds[name] < 1e-6:
                self.stds[name] = 1.0

        X = np.array([_feature_vector(f, self.means, self.stds) for f in X_raw])
        y_arr = np.array(y, dtype=float)
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        lr = 0.08

        for _ in range(400):
            z = X @ w + b
            pred = _sigmoid(z)
            grad_w = (X.T @ (pred - y_arr)) / n + 0.02 * w
            grad_b = float(np.mean(pred - y_arr))
            w -= lr * grad_w
            b -= lr * grad_b

        self.weights = w
        self.bias = b
        self.sample_count = len(rows)
        self.train_win_rate = float(np.mean(y_arr))

        probs = _sigmoid(X @ w + b)
        self.threshold = _find_threshold(probs, y_arr, TARGET_WIN_RATE)
        mask = probs >= self.threshold
        if mask.any():
            self.filtered_win_rate = float(np.mean(y_arr[mask]))
        else:
            self.filtered_win_rate = self.train_win_rate

    def predict_proba(self, features: dict[str, Any]) -> float:
        if self.weights is None:
            return 0.55
        x = _feature_vector(features, self.means, self.stds)
        return float(_sigmoid(np.array([x @ self.weights + self.bias]))[0])

    def to_dict(self) -> dict:
        return {
            "trained": self.weights is not None,
            "sample_count": self.sample_count,
            "train_win_rate": self.train_win_rate,
            "filtered_win_rate": self.filtered_win_rate,
            "threshold": self.threshold,
            "target_win_rate": TARGET_WIN_RATE,
            "feature_weights": {
                name: round(float(self.weights[i]), 4)
                for i, name in enumerate(FEATURE_NAMES)
            }
            if self.weights is not None
            else {},
        }


def _find_threshold(probs: np.ndarray, y: np.ndarray, target: float) -> float:
    """Lowest threshold that keeps the most trades while win rate >= target."""
    best_t = 0.45
    best_count = 0
    min_samples = max(5, int(len(y) * 0.08))
    for t in np.linspace(0.45, 0.82, 38):
        mask = probs >= t
        count = int(mask.sum())
        if count < min_samples:
            continue
        wr = float(np.mean(y[mask]))
        if wr >= target and count >= best_count:
            best_t = float(t)
            best_count = count
    return max(0.45, min(0.80, best_t))


def analyze_patterns(rows: list[dict]) -> dict[str, Any]:
    """Summarize win-rate patterns from historical trades."""
    closed = [r for r in rows if r.get("status") == "closed" and r.get("signal") in ("BUY", "SELL")]
    if not closed:
        return {"total_closed": 0, "overall_win_rate": 0.0, "insights": []}

    wins = sum(1 for r in closed if is_winning_trade(r))
    decisive = [r for r in closed if not is_breakeven_trade(r)]
    overall_wr = wins / len(decisive) if decisive else 0.0

    def bucket_wr(label: str, subset: list[dict]) -> dict:
        if not subset:
            return {"label": label, "count": 0, "win_rate": 0.0}
        decisive_subset = [r for r in subset if not is_breakeven_trade(r)]
        if not decisive_subset:
            return {"label": label, "count": len(subset), "win_rate": 0.0}
        w = sum(1 for r in decisive_subset if is_winning_trade(r))
        return {
            "label": label,
            "count": len(subset),
            "win_rate": round(w / len(decisive_subset), 3),
        }

    risk_buckets = []
    for lo, hi, name in [(0, 1.5, "Tight SL (<$1.50)"), (1.5, 3, "Medium SL ($1.50–3)"), (3, 6, "Wide SL ($3–6)"), (6, 999, "Oversized SL (>$6)")]:
        subset = [
            r for r in closed
            if r.get("entry") and r.get("sl")
            and lo <= abs(float(r["entry"]) - float(r["sl"])) < hi
        ]
        risk_buckets.append(bucket_wr(name, subset))

    hour_buckets = []
    for hour in range(24):
        subset = [r for r in closed if extract_features_from_trade(r).get("hour_utc") == hour]
        if len(subset) >= 10:
            hour_buckets.append(bucket_wr(f"Hour {hour:02d} UTC", subset))
    hour_buckets.sort(key=lambda x: x["win_rate"], reverse=True)

    direction = [
        bucket_wr("BUY", [r for r in closed if r.get("signal") == "BUY"]),
        bucket_wr("SELL", [r for r in closed if r.get("signal") == "SELL"]),
    ]

    insights = [
        f"Overall win rate: {overall_wr:.1%} across {len(closed)} closed trades",
        f"Target calibration: {TARGET_WIN_RATE:.0%} (architecture equilibrium, not hard-coded)",
        f"Best risk bucket: {risk_buckets[0]['label']} at {risk_buckets[0]['win_rate']:.1%}" if risk_buckets else "",
        f"Worst risk bucket: {min(risk_buckets, key=lambda x: x['win_rate'])['label']} at {min(risk_buckets, key=lambda x: x['win_rate'])['win_rate']:.1%}" if len(risk_buckets) > 1 else "",
        "Recommendation: adaptive guard blocks weak states/hours after losses",
        "Recommendation: ML + Markov refresh runs after every closed trade",
    ]

    return {
        "total_closed": len(closed),
        "overall_win_rate": round(overall_wr, 4),
        "target_win_rate": TARGET_WIN_RATE,
        "risk_buckets": risk_buckets,
        "best_hours": hour_buckets[:5],
        "worst_hours": hour_buckets[-3:] if len(hour_buckets) >= 3 else [],
        "direction": direction,
        "insights": [i for i in insights if i],
        "hard_rules": [{"id": r["id"], "label": r["label"]} for r in HARD_RULES],
    }


def evaluate_trade(
    features: dict[str, Any],
    model: LogisticCalibrator,
    use_ml_gate: bool = True,
) -> tuple[bool, float, str]:
    """Return (allow, probability, reason)."""
    ok, adaptive_reason = adaptive_check(features)
    if not ok:
        return False, 0.0, adaptive_reason

    for rule in HARD_RULES:
        if rule["check"](features):
            return False, 0.0, rule["reason"]

    prob = model.predict_proba(features)

    if use_ml_gate and model.weights is not None:
        floor_prob = max(model.threshold, min_ml_prob())
        if prob < floor_prob:
            return (
                False,
                prob,
                f"ML win probability {prob:.0%} below gate {floor_prob:.0%} (target {TARGET_WIN_RATE:.0%})",
            )

    if features.get("rr_ratio") is not None and features["rr_ratio"] < 1.5:
        return False, prob, f"RR {features['rr_ratio']:.2f} below 1:1.5 minimum"

    mc_val = features.get("markov_mc_prob") or features.get("mc_prob")
    floor_mc = mc_floor()
    if mc_val is not None and mc_val < floor_mc:
        label = "Markov MC" if features.get("markov_mc_prob") is not None else "Monte Carlo"
        return False, prob, f"{label} {mc_val:.0f}% below adaptive floor {floor_mc:.0f}%"

    return True, prob, f"Approved — ML win probability {prob:.0%}"
