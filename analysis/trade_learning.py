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
    "news_bias_score",
    "news_aligned",
    "vp_score",
    "vp_aligned",
    "vp_dist_poc_atr",
    "vp_in_value",
    "vp_above_value",
    "vp_below_value",
    "laminar_bull",
    "laminar_bear",
    "lot_size",
    "capital_band",
    "asset_vol",
    "trail_lock",
    "tp1_frac",
]

# Hard rules — only reject clearly bad setups
# SL cap raised to $10 — user accepts up to $10 risk per trade;
# data shows $3–$10 is tradeable, only >$10 has poor win rate.
HARD_RULES = [
    {
        "id": "risk_oversized",
        "label": "Oversized $ risk vs SL — poor historical win rate",
        "check": lambda f: f.get("risk_usd") is not None and f["risk_usd"] > 250.0,
        "action": "reject",
        "reason": "SL risk > $250 — historically poor win rate",
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
        "prime_hour": 1.0 if hour in {6, 8, 9, 10, 11, 13, 14, 15, 16, 17} else 0.0,
        "mc_prob": _float(features.get("mc_prob") or features.get("SUCCESS_PERCENT")),
        "markov_mc_prob": _float(features.get("markov_mc_prob") or features.get("MARKOV_MC_PROB")),
        "markov_persist": _float(features.get("markov_persist") or features.get("MARKOV_PERSIST")),
        "markov_bull_prob": _float(features.get("markov_bull_prob") or features.get("MARKOV_BULL_PROB")),
        "state_win_rate": _float(features.get("state_win_rate") or features.get("STATE_WIN_RATE")),
        "score_margin": _float(features.get("score_margin")),
        "smc_score": _float(features.get("smc_score") or features.get("SMC_SCORE")),
        "bias_aligned": _float(features.get("bias_aligned")),
        "news_bias_score": _float(
            features.get("news_bias_score") or features.get("WORLD_NEWS_SCORE")
        ),
        "news_aligned": (
            _float(features.get("news_aligned"))
            if features.get("news_aligned") is not None
            else (
                1.0
                if features.get("WORLD_NEWS_ALIGNED") is True
                else 0.0
                if features.get("WORLD_NEWS_ALIGNED") is False
                else None
            )
        ),
        "vp_score": _float(features.get("vp_score") or features.get("VP_SCORE")),
        "vp_aligned": (
            _float(features.get("vp_aligned"))
            if features.get("vp_aligned") is not None
            else (
                1.0
                if features.get("VP_ALIGNED") is True
                else 0.0
                if features.get("VP_ALIGNED") is False
                else None
            )
        ),
        "vp_dist_poc_atr": _float(
            features.get("vp_dist_poc_atr") or features.get("VP_DIST_POC_ATR")
        ),
        "vp_in_value": _float(
            features.get("vp_in_value")
            if features.get("vp_in_value") is not None
            else (1.0 if features.get("VP_IN_VALUE") else 0.0 if features.get("VP_IN_VALUE") is False else None)
        ),
        "vp_above_value": _float(
            features.get("vp_above_value")
            if features.get("vp_above_value") is not None
            else (1.0 if features.get("VP_ABOVE_VALUE") else 0.0 if features.get("VP_ABOVE_VALUE") is False else None)
        ),
        "vp_below_value": _float(
            features.get("vp_below_value")
            if features.get("vp_below_value") is not None
            else (1.0 if features.get("VP_BELOW_VALUE") else 0.0 if features.get("VP_BELOW_VALUE") is False else None)
        ),
        "laminar_bull": 1.0 if features.get("STATE") == "LAMINAR_BULL" else 0.0,
        "laminar_bear": 1.0 if features.get("STATE") == "LAMINAR_BEAR" else 0.0,
        "lot_size": _float(features.get("lot_size") or row.get("lot_size")),
        "capital_band": _capital_band(features.get("capital")),
        "asset_vol": _asset_vol_score(features.get("asset_class") or features.get("symbol") or row.get("symbol")),
        "trail_lock": _float(features.get("trail_lock")),
        "tp1_frac": _float(features.get("tp1_frac")),
        "signal": signal,
    }


def _capital_band(capital: Any) -> Optional[float]:
    """Normalize capital into 0..1 learning feature (log-ish)."""
    try:
        c = float(capital)
    except (TypeError, ValueError):
        return None
    if c <= 0:
        return 0.0
    import math

    return round(min(1.0, math.log10(max(c, 10.0)) / 5.0), 4)


def _asset_vol_score(symbol_or_class: Any) -> Optional[float]:
    try:
        from strategy.dynamic_lots import asset_class, _VOL_MULT

        raw = str(symbol_or_class or "").upper()
        cls = raw if raw in _VOL_MULT else asset_class(raw)
        return float(_VOL_MULT.get(cls, 1.0))
    except Exception:
        return None


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

    news_score = _float(ts.get("WORLD_NEWS_SCORE") or sig.get("WORLD_NEWS_SCORE"))
    news_aligned_raw = ts.get("WORLD_NEWS_ALIGNED")
    if news_aligned_raw is None:
        news_aligned_raw = sig.get("WORLD_NEWS_ALIGNED")
    if news_aligned_raw is None and news_score is not None and signal in ("BUY", "SELL"):
        # Derive alignment if signal payload lacked the flag
        news_aligned_raw = (signal == "BUY" and news_score > 0.15) or (
            signal == "SELL" and news_score < -0.15
        )

    vp_score = _float(ts.get("VP_SCORE") or sig.get("VP_SCORE"))
    vp_aligned_raw = ts.get("VP_ALIGNED")
    if vp_aligned_raw is None:
        vp_aligned_raw = sig.get("VP_ALIGNED")
    vp_in = ts.get("VP_IN_VALUE")
    if vp_in is None:
        vp_in = sig.get("VP_IN_VALUE")
    vp_above = ts.get("VP_ABOVE_VALUE")
    if vp_above is None:
        vp_above = sig.get("VP_ABOVE_VALUE")
    vp_below = ts.get("VP_BELOW_VALUE")
    if vp_below is None:
        vp_below = sig.get("VP_BELOW_VALUE")

    hour = _parse_hour(sig.get("CANDLE_TIME") or sig.get("LIVE_TIME"))
    sym = str(ts.get("SYMBOL") or ts.get("INSTRUMENT") or sig.get("SYMBOL") or "GOLD").upper()
    lot = _float(ts.get("LOT_SIZE") or ts.get("lot_size"))
    capital = _float(ts.get("capital") or sig.get("capital"))
    trail = _float(ts.get("trail_lock"))
    tp1f = _float(ts.get("tp1_frac"))
    asset_cls = ts.get("asset_class")

    # Prefer true $ risk at SL when lot is known
    risk_usd = None
    if entry is not None and sl is not None:
        if lot is not None:
            try:
                from strategy.position_sizing import leg_pnl

                risk_usd = abs(leg_pnl(signal or "BUY", float(entry), float(sl), lot, symbol=sym))
            except Exception:
                risk_usd = round(risk, 3) if risk is not None else None
        else:
            risk_usd = round(risk, 3) if risk is not None else None

    return {
        "risk_usd": risk_usd,
        "rr_ratio": round(rr, 3) if rr is not None else None,
        "is_buy": 1.0 if signal == "BUY" else 0.0,
        "hour_utc": hour,
        "prime_hour": 1.0 if hour in {6, 8, 9, 10, 11, 13, 14, 15, 16, 17} else 0.0,
        "mc_prob": _float(ts.get("SUCCESS_PERCENT") or sig.get("SUCCESS_PERCENT")),
        "markov_mc_prob": _float(ts.get("MARKOV_MC_PROB") or sig.get("MARKOV_MC_PROB")),
        "markov_persist": _float(sig.get("MARKOV_PERSIST")),
        "markov_bull_prob": _float(sig.get("MARKOV_BULL_PROB")),
        "state_win_rate": _float(sig.get("STATE_WIN_RATE")),
        "score_margin": abs(buy_score - sell_score),
        "smc_score": _float(sig.get("SMC_SCORE")),
        "bias_aligned": 1.0 if aligned else 0.0,
        "news_bias_score": news_score,
        "news_aligned": 1.0 if news_aligned_raw else 0.0 if news_aligned_raw is not None else None,
        "vp_score": vp_score,
        "vp_aligned": 1.0 if vp_aligned_raw else 0.0 if vp_aligned_raw is not None else None,
        "vp_dist_poc_atr": _float(ts.get("VP_DIST_POC_ATR") or sig.get("VP_DIST_POC_ATR")),
        "vp_in_value": 1.0 if vp_in else 0.0 if vp_in is not None else None,
        "vp_above_value": 1.0 if vp_above else 0.0 if vp_above is not None else None,
        "vp_below_value": 1.0 if vp_below else 0.0 if vp_below is not None else None,
        "laminar_bull": 1.0 if sig.get("STATE") == "LAMINAR_BULL" else 0.0,
        "laminar_bear": 1.0 if sig.get("STATE") == "LAMINAR_BEAR" else 0.0,
        "lot_size": lot,
        "capital_band": _capital_band(capital),
        "asset_vol": _asset_vol_score(asset_cls or sym),
        "trail_lock": trail,
        "tp1_frac": tp1f,
        "signal": signal,
        "STATE": sig.get("STATE") or ts.get("STATE"),
        "exec_tf": ts.get("EXEC_TF") or sig.get("EXEC_TF"),
        "EXEC_TF": ts.get("EXEC_TF") or sig.get("EXEC_TF"),
        "mtf_independent": bool(ts.get("MTF_INDEPENDENT") or sig.get("MTF_INDEPENDENT")),
        "BUY_SCORE": buy_score,
        "SELL_SCORE": sell_score,
        "concepts": sig.get("CONCEPTS") or {},
        "symbol": sym,
        "SYMBOL": sym,
        "concept_quality": _float(sig.get("CONCEPT_QUALITY")),
        "SUCCESS_PERCENT": _float(ts.get("SUCCESS_PERCENT") or sig.get("SUCCESS_PERCENT")),
        "MARKOV_MC_PROB": _float(ts.get("MARKOV_MC_PROB") or sig.get("MARKOV_MC_PROB")),
        "MARKOV_PERSIST": _float(sig.get("MARKOV_PERSIST")),
        "MARKOV_BULL_PROB": _float(sig.get("MARKOV_BULL_PROB")),
        "STATE_WIN_RATE": _float(sig.get("STATE_WIN_RATE")),
        "SMC_SCORE": _float(sig.get("SMC_SCORE")),
        "WORLD_NEWS_SCORE": news_score,
        "WORLD_NEWS_ALIGNED": news_aligned_raw,
        "WORLD_NEWS_DIRECTION": ts.get("WORLD_NEWS_DIRECTION") or sig.get("WORLD_NEWS_DIRECTION"),
        "WORLD_NEWS_MC_ADJUST": _float(
            ts.get("WORLD_NEWS_MC_ADJUST") or sig.get("WORLD_NEWS_MC_ADJUST")
        ),
        "VP_SCORE": vp_score,
        "VP_ALIGNED": vp_aligned_raw,
        "VP_LOCATION": ts.get("VP_LOCATION") or sig.get("VP_LOCATION"),
        "VP_POC": ts.get("VP_POC") or sig.get("VP_POC"),
        "VP_VAH": ts.get("VP_VAH") or sig.get("VP_VAH"),
        "VP_VAL": ts.get("VP_VAL") or sig.get("VP_VAL"),
        "VP_MC_ADJUST": _float(ts.get("VP_MC_ADJUST") or sig.get("VP_MC_ADJUST")),
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
        # Floor std at 0.1 — prevents catastrophic extrapolation for features rarely
        # seen in training (e.g. laminar_bull/bear present in <0.1% of historical rows)
        std = max(stds.get(name, 1.0) or 1.0, 0.1)
        # Clip to ±5 std — standard practice to cap rare-feature explosion at inference
        normed = np.clip((float(val) - mean) / std, -5.0, 5.0)
        vec.append(normed)
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
            self.stds[name] = max(float(np.std(vals)) if vals else 1.0, 0.1)

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
        # Cap so a 6% train WR model cannot demand 55%+ and freeze the desk
        self.threshold = min(float(self.threshold or 0.40), 0.40)
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

    # Concept quality (learned expectancy across EMA/SMC/AVP/MC/…)
    cq = features.get("concept_quality")
    concepts = features.get("concepts")
    if concepts and cq is None:
        try:
            from analysis.concept_engine import passes_concept_gate
            ok_c, cq, creason = passes_concept_gate(concepts if isinstance(concepts, dict) else {})
            if not ok_c:
                return False, 0.0, creason
        except Exception:
            pass
    elif cq is not None:
        try:
            from analysis.concept_engine import get_state as concept_state
            min_q = float(concept_state().get("min_quality") or 0.35)
            if float(cq) < min_q:
                return False, float(cq), f"Concept quality {float(cq):.2f} below {min_q:.2f}"
        except Exception:
            pass

    for rule in HARD_RULES:
        if rule["check"](features):
            return False, 0.0, rule["reason"]

    prob = model.predict_proba(features)

    mc_val = features.get("markov_mc_prob") or features.get("mc_prob")
    sym = str(features.get("symbol") or features.get("SYMBOL") or "").upper() or None
    try:
        from analysis.instrument_rl import mc_floor_for, min_ml_for

        floor_mc = mc_floor_for(sym)
        floor_ml_extra = min_ml_for(sym)
    except Exception:
        floor_mc = mc_floor()
        floor_ml_extra = min_ml_prob()

    if use_ml_gate and model.weights is not None:
        exec_tf = str(features.get("exec_tf") or features.get("EXEC_TF") or "").lower()
        is_swing = exec_tf in ("15m", "1h", "4h") or bool(features.get("mtf_independent"))
        if is_swing:
            floor_prob = min(float(model.threshold or 0.40), 0.34)
        else:
            floor_prob = min(
                0.36,
                max(0.26, max(float(model.threshold or 0.32), float(floor_ml_extra))),
            )

        # Learn mode: if a real setup has MC near the floor, never kill on ML.
        # RL needs closes — risk control is lots/RR from overlays, not a dead gate.
        has_setup = str(features.get("signal") or features.get("SIGNAL") or "").upper() in (
            "BUY",
            "SELL",
        )
        mc_present = mc_val is not None
        mc_ok = mc_present and float(mc_val) + 1e-6 >= float(floor_mc) - 3.0  # 3pp grace
        skip_ml_hard = bool(has_setup and (mc_ok or not mc_present))

        if not skip_ml_hard and prob < floor_prob:
            return (
                False,
                prob,
                f"ML win probability {prob:.0%} below gate {floor_prob:.0%} "
                f"({sym or 'desk'} target {TARGET_WIN_RATE:.0%})",
            )

    if features.get("rr_ratio") is not None and features["rr_ratio"] < 1.15:
        return False, prob, f"RR {features['rr_ratio']:.2f} below 1:1.15 minimum"

    # Per-entry Markov/path-MC floor — OFF when MARKOV_MC_GATE_ENABLED=false so
    # live matches Dukascopy hunt brains (bootstrap MC only, no entry Markov gate).
    try:
        from config import MARKOV_MC_GATE_ENABLED
    except Exception:
        MARKOV_MC_GATE_ENABLED = False
    if MARKOV_MC_GATE_ENABLED:
        # 3pp grace so borderline zone setups aren't sniped by floating MC floors
        if mc_val is not None and float(mc_val) + 3.0 + 1e-6 < float(floor_mc):
            label = (
                "Markov MC"
                if features.get("markov_mc_prob") is not None
                else "Monte Carlo"
            )
            return (
                False,
                prob,
                f"{label} {float(mc_val):.0f}% below {sym or 'desk'} floor {float(floor_mc):.0f}%",
            )

    # Soft structure hints only — do not hard-kill setups (RL needs the closes)
    try:
        from analysis.correlation_matrix import correlation_bias_for
        from analysis.minute_learn import get_minute_state

        sig_dir = str(features.get("signal") or features.get("SIGNAL") or "")
        cb = correlation_bias_for(sym or "GOLD", sig_dir)
        # Extreme anti-gold correlation alone is not enough to kill; just annotate
        mstate = get_minute_state(sym)
        if mstate.get("structure_bias") and sig_dir in ("BUY", "SELL"):
            sb = mstate["structure_bias"]
            vp_sc = float(mstate.get("vp_score") or 0)
            fights_stack = (sig_dir == "BUY" and sb == "BEARISH" and vp_sc < -0.85) or (
                sig_dir == "SELL" and sb == "BULLISH" and vp_sc > 0.85
            )
            # Only block obvious fights — leave mild conflicts for RL to sort
            if fights_stack and abs(vp_sc) >= 0.9:
                return False, prob, f"Minute stack conflicts trade ({mstate.get('message')})"
        _ = cb  # retained for logging / future soft score
    except Exception:
        pass

    tag = f"{sym} · " if sym else ""
    return True, prob, f"Approved — {tag}ML win probability {prob:.0%}"
