"""Continuous minute learning from forming candles on every timeframe.

Learns structure (impulse/chop), FRVP/POC location, and universe correlations
every minute — not only from closed trades. Soft state overlays into
instrument_rl and evaluate_trade.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PATH = Path(__file__).resolve().parents[1] / "data" / "minute_learn_state.json"
_TFS = ("1m", "3m", "5m", "15m", "1h", "4h")

_STATE: dict[str, Any] = {
    "updated_at": None,
    "last_1m": None,
    "instruments": {},
    "correlation": {},
}


def _blank_sym(sym: str) -> dict[str, Any]:
    return {
        "symbol": sym,
        "tf": {},
        "poc": None,
        "vah": None,
        "val": None,
        "vp_location": None,
        "vp_session": None,
        "vp_score": 0.0,
        "structure_bias": "NEUTRAL",  # from multi-TF candle vote
        "structure_score": 0.0,
        "mc_floor_delta": 0.0,
        "min_ml_delta": 0.0,
        "aggression": 0.0,
        "message": "awaiting candles",
        "candles_seen": 0,
    }


def _candle_vote(df: pd.DataFrame) -> tuple[float, str]:
    """Score forming+last closed candles: +bull / -bear."""
    if df is None or len(df) < 3:
        return 0.0, "NEUTRAL"
    tail = df.tail(5)
    score = 0.0
    for _, row in tail.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body = c - o
        rng = max(h - l, 1e-9)
        score += (body / rng) * 0.6
        # close location in range
        score += ((c - l) / rng - 0.5) * 0.4
    score = max(-2.0, min(2.0, score))
    if score > 0.35:
        return score, "BULLISH"
    if score < -0.35:
        return score, "BEARISH"
    return score, "NEUTRAL"


def _learn_symbol(
    sym: str,
    frames: dict[str, pd.DataFrame],
    *,
    price: Optional[float] = None,
) -> dict[str, Any]:
    from analysis.volume_profile import build_dual_profile

    row = _blank_sym(sym)
    tf_scores: list[float] = []
    tf_detail: dict[str, Any] = {}

    for tf in _TFS:
        df = frames.get(tf)
        if df is None or len(df) < 3:
            continue
        sc, bias = _candle_vote(df)
        # Weight higher TFs more
        w = {"1m": 0.5, "3m": 0.7, "5m": 1.0, "15m": 1.3, "1h": 1.6, "4h": 2.0}.get(tf, 1.0)
        tf_scores.append(sc * w)
        last = df.iloc[-1]
        tf_detail[tf] = {
            "bias": bias,
            "score": round(sc, 3),
            "close": float(last["close"]),
            "time": str(df.index[-1]),
        }
        row["candles_seen"] += 1

    row["tf"] = tf_detail
    if tf_scores:
        struct = sum(tf_scores) / max(len(tf_scores), 1)
        row["structure_score"] = round(struct, 3)
        row["structure_bias"] = (
            "BULLISH" if struct > 0.25 else ("BEARISH" if struct < -0.25 else "NEUTRAL")
        )

    df5 = frames.get("5m")
    if df5 is None:
        df5 = frames.get("1m")
    atr = None
    px = price
    if df5 is not None and len(df5):
        px = px or float(df5.iloc[-1]["close"])
        atr = float((df5["high"] - df5["low"]).tail(20).mean() or 0) or None
        vp = build_dual_profile(df5, price=px, atr=atr, symbol=sym)
        if vp.get("available"):
            row["poc"] = vp.get("poc")
            row["vah"] = vp.get("vah")
            row["val"] = vp.get("val")
            row["vp_location"] = vp.get("location")
            row["vp_session"] = vp.get("session_name") or vp.get("source")
            row["vp_score"] = float(vp.get("score") or 0)
            row["vp_summary"] = vp.get("summary")

    # Soft overlays: chop / against POC → raise gates; stacked structure+VP → ease slightly
    mc_d = 0.0
    ml_d = 0.0
    agg = 0.0
    loc = row.get("vp_location")
    struct_b = row.get("structure_bias")
    if loc in ("ABOVE_VALUE", "BELOW_VALUE") and struct_b == "NEUTRAL":
        mc_d += 2.0  # extended without TF confirm — pickier
        ml_d += 0.02
        agg -= 0.04
    if loc == "AT_POC" and struct_b != "NEUTRAL":
        # POC + candle stack — allow mean-reversion / continuation with slight ease
        mc_d -= 1.5
        agg += 0.03
    if struct_b == "NEUTRAL" and abs(float(row.get("structure_score") or 0)) < 0.15:
        mc_d += 1.5  # chop across TFs
        agg -= 0.05

    # Align structure with VP direction
    vp_dir = "BULLISH" if float(row.get("vp_score") or 0) > 0.2 else (
        "BEARISH" if float(row.get("vp_score") or 0) < -0.2 else "NEUTRAL"
    )
    if struct_b != "NEUTRAL" and vp_dir != "NEUTRAL" and struct_b == vp_dir:
        mc_d -= 2.0
        ml_d -= 0.02
        agg += 0.05
        row["message"] = f"{struct_b} stack · {loc} · POC {row.get('poc')}"
    elif struct_b != "NEUTRAL" and vp_dir != "NEUTRAL" and struct_b != vp_dir:
        mc_d += 3.0
        ml_d += 0.03
        agg -= 0.06
        row["message"] = f"conflict {struct_b} vs VP {vp_dir} · picky"
    else:
        row["message"] = f"{struct_b} · {loc or 'no VP'} · sess {row.get('vp_session')}"

    row["mc_floor_delta"] = round(mc_d, 2)
    row["min_ml_delta"] = round(ml_d, 3)
    row["aggression"] = round(agg, 3)
    return row


def run_minute_learn(
    *,
    primary_frames: Optional[dict[str, pd.DataFrame]] = None,
    primary_symbol: str = "GOLD",
    primary_signal: Optional[dict] = None,
    candle_key: Optional[str] = None,
    scan_universe: bool = True,
) -> dict[str, Any]:
    """
    Called every worker poll. Heavy universe work only when 1m candle_key changes.
    """
    from analysis.correlation_matrix import compute_correlation_state, get_correlation_state
    from config import INSTRUMENT_UNIVERSE, MULTI_INSTRUMENT_ENABLED

    with _LOCK:
        last = _STATE.get("last_1m")
        if candle_key and candle_key == last:
            pass
        elif candle_key:
            _STATE["last_1m"] = candle_key

    instruments: dict[str, Any] = {}
    frames_by_sym: dict[str, dict[str, pd.DataFrame]] = {}

    if primary_frames:
        frames_by_sym[str(primary_symbol).upper()] = primary_frames
        px = None
        if primary_signal:
            px = primary_signal.get("LOG_PRICE_1M") or primary_signal.get("LIVE_PRICE")
        instruments[str(primary_symbol).upper()] = _learn_symbol(
            str(primary_symbol).upper(), primary_frames, price=float(px) if px else None
        )

    do_universe = (
        bool(scan_universe)
        and bool(candle_key)
        and (candle_key != last or not _STATE.get("instruments"))
        and MULTI_INSTRUMENT_ENABLED
    )
    if do_universe:
        try:
            from strategy.instrument_scanner import fetch_frames_for_symbol

            for sym in INSTRUMENT_UNIVERSE:
                su = str(sym).upper()
                if su in frames_by_sym:
                    continue
                try:
                    fr = fetch_frames_for_symbol(su)
                    if fr:
                        frames_by_sym[su] = fr
                        instruments[su] = _learn_symbol(su, fr)
                except Exception as exc:
                    logger.debug("minute_learn %s: %s", su, exc)
        except Exception as exc:
            logger.debug("minute_learn universe: %s", exc)

    # Correlation across whatever we have
    corr = get_correlation_state()
    if frames_by_sym and (do_universe or not corr.get("matrix")):
        try:
            corr = compute_correlation_state(frames_by_sym, tf="5m")
        except Exception as exc:
            logger.debug("corr compute: %s", exc)

    # Merge with previous instruments if this tick only refreshed primary
    with _LOCK:
        if not do_universe and _STATE.get("instruments"):
            merged = dict(_STATE["instruments"])
            merged.update(instruments)
            instruments = merged
        _STATE["instruments"] = instruments
        _STATE["correlation"] = {
            "pairs": corr.get("pairs"),
            "leaders": corr.get("leaders"),
            "updated_at": corr.get("updated_at"),
        }
        _STATE["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            _PATH.write_text(json.dumps(_STATE, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
        return {
            "updated_at": _STATE["updated_at"],
            "last_1m": _STATE.get("last_1m"),
            "instruments": {k: dict(v) for k, v in instruments.items()},
            "correlation": dict(_STATE["correlation"]),
        }


def get_minute_state(symbol: Optional[str] = None) -> dict[str, Any]:
    with _LOCK:
        if symbol:
            return dict((_STATE.get("instruments") or {}).get(str(symbol).upper()) or {})
        return {
            "updated_at": _STATE.get("updated_at"),
            "last_1m": _STATE.get("last_1m"),
            "instruments": dict(_STATE.get("instruments") or {}),
            "correlation": dict(_STATE.get("correlation") or {}),
        }


def minute_overlays_for(symbol: Optional[str] = None) -> dict[str, float]:
    """Soft deltas for instrument_rl — capped so close-trade RL stays primary."""
    row = get_minute_state(symbol)
    if not row:
        return {"mc_floor_delta": 0.0, "min_ml_delta": 0.0, "aggression": 0.0}
    return {
        "mc_floor_delta": max(-3.0, min(3.0, float(row.get("mc_floor_delta") or 0))),
        "min_ml_delta": max(-0.03, min(0.04, float(row.get("min_ml_delta") or 0))),
        "aggression": max(-0.10, min(0.10, float(row.get("aggression") or 0))),
        "structure_score": float(row.get("structure_score") or 0),
        "vp_score": float(row.get("vp_score") or 0),
    }
