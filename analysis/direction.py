"""5-minute movement direction and AI bias alignment gates."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from analysis.bias import get_bias_detail


def get_5m_direction(df_5m: pd.DataFrame) -> dict[str, Any]:
    """
    Determine trade direction from 5M price movement (not 1M noise).
    Returns BUY when 5M structure is bullish, SELL when bearish, else NEUTRAL.
    """
    if df_5m is None or len(df_5m) < 3:
        return {
            "direction": "NEUTRAL",
            "movement": "FLAT",
            "confidence": 0.0,
            "reason": "Insufficient 5M data",
        }

    row = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]
    prev2 = df_5m.iloc[-3]

    close = float(row["close"])
    open_ = float(row["open"])
    prev_close = float(prev["close"])
    prev2_close = float(prev2["close"])

    bull_votes = sum(
        [
            float(row["EMA9"]) > float(row["EMA21"]),
            close > open_,
            close > prev_close,
            prev_close > prev2_close,
            float(row.get("MACD_HIST", 0)) > 0,
            float(row.get("RSI", 50)) > 50,
        ]
    )
    bear_votes = sum(
        [
            float(row["EMA9"]) < float(row["EMA21"]),
            close < open_,
            close < prev_close,
            prev_close < prev2_close,
            float(row.get("MACD_HIST", 0)) < 0,
            float(row.get("RSI", 50)) < 50,
        ]
    )

    bias_detail = get_bias_detail(df_5m)

    if bull_votes >= 4 and bull_votes > bear_votes:
        movement = "UP"
        direction = "BUY"
        confidence = round((bull_votes / 6.0) * 100.0, 1)
        reason = f"5M movement UP ({bull_votes}/6 bullish signals, bias {bias_detail['confidence']:.0f}%)"
    elif bear_votes >= 4 and bear_votes > bull_votes:
        movement = "DOWN"
        direction = "SELL"
        confidence = round((bear_votes / 6.0) * 100.0, 1)
        reason = f"5M movement DOWN ({bear_votes}/6 bearish signals, bias {bias_detail['confidence']:.0f}%)"
    else:
        movement = "FLAT"
        direction = "NEUTRAL"
        confidence = 0.0
        reason = f"5M choppy — bull {bull_votes} vs bear {bear_votes}, no clear direction"

    return {
        "direction": direction,
        "movement": movement,
        "confidence": confidence,
        "bias_confidence": bias_detail["confidence"],
        "reason": reason,
    }


def bias_allows_trade(
    direction: str,
    bias_data: Optional[dict[str, Any]],
    min_confidence: float = 60.0,
    symbol: Optional[str] = None,
) -> tuple[bool, str]:
    """Check algo bias agrees with intended trade direction (per-symbol when available)."""
    if not bias_data or direction not in ("BUY", "SELL"):
        return True, "AI bias unavailable — allowing trade"

    pack = bias_data
    sym = str(symbol or "").upper().strip()
    if sym and isinstance(bias_data.get("instruments"), list):
        for row in bias_data["instruments"]:
            if str(row.get("symbol") or "").upper() == sym:
                pack = row
                break

    overall = pack.get("overall") or {}
    tf5 = (pack.get("timeframes") or {}).get("5m") or {}

    checks = [
        ("Overall", overall.get("direction"), overall.get("confidence")),
        ("5M", tf5.get("direction"), tf5.get("confidence")),
    ]

    for label, bias_dir, conf in checks:
        if bias_dir not in ("BULLISH", "BEARISH") or conf is None:
            continue
        if float(conf) < min_confidence:
            continue
        if direction == "BUY" and bias_dir == "BEARISH":
            return False, f"{sym or 'Desk'} {label} bias BEARISH ({conf:.0f}%) — blocks BUY"
        if direction == "SELL" and bias_dir == "BULLISH":
            return False, f"{sym or 'Desk'} {label} bias BULLISH ({conf:.0f}%) — blocks SELL"

    overall_label = overall.get("label", "Neutral")
    overall_conf = overall.get("confidence", 0)
    tag = f"{sym} " if sym else ""
    return True, f"{tag}AI bias aligned — {overall_label} ({overall_conf:.0f}%)"
