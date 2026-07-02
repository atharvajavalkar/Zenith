"""Multi-indicator trend bias detection."""

import pandas as pd


def _indicator_votes(row: pd.Series) -> tuple[int, int, float, float, float]:
    bull = 0
    bear = 0

    if row["EMA9"] > row["EMA21"]:
        bull += 1
    elif row["EMA9"] < row["EMA21"]:
        bear += 1

    if row["close"] > row["EMA50"]:
        bull += 1
    elif row["close"] < row["EMA50"]:
        bear += 1

    if row["MACD_HIST"] > 0:
        bull += 1
    elif row["MACD_HIST"] < 0:
        bear += 1

    if row["RSI"] > 50:
        bull += 1
    elif row["RSI"] < 50:
        bear += 1

    rsi = float(row["RSI"])
    macd_hist = float(row["MACD_HIST"])
    atr = float(row.get("ATR") or 1.0)
    macd_strength = min(1.0, abs(macd_hist) / max(atr * 0.15, 0.01))
    rsi_bull = max(0.0, (rsi - 50.0) / 50.0)
    rsi_bear = max(0.0, (50.0 - rsi) / 50.0)
    return bull, bear, rsi_bull, rsi_bear, macd_strength


def get_bias(df: pd.DataFrame) -> str:
    bull, bear, _, _, _ = _indicator_votes(df.iloc[-1])
    if bull >= 3:
        return "BULLISH"
    if bear >= 3:
        return "BEARISH"
    return "NEUTRAL"


def get_bias_detail(df: pd.DataFrame) -> dict:
    """Direction + confidence % for one timeframe."""
    row = df.iloc[-1]
    bull, bear, rsi_bull, rsi_bear, macd_strength = _indicator_votes(row)

    if bull >= 3:
        direction = "BULLISH"
        base = bull / 4.0 * 100.0
        confidence = base + rsi_bull * 10.0 + macd_strength * 8.0
    elif bear >= 3:
        direction = "BEARISH"
        base = bear / 4.0 * 100.0
        confidence = base + rsi_bear * 10.0 + macd_strength * 8.0
    else:
        direction = "NEUTRAL"
        confidence = 50.0 + abs(bull - bear) * 6.0

    confidence = round(min(95.0, max(52.0, confidence)), 1)
    return {
        "direction": direction,
        "confidence": confidence,
        "bull_votes": bull,
        "bear_votes": bear,
    }


def get_context_4h(df: pd.DataFrame) -> str:
    row = df.iloc[-1]

    if row["ATR_RATIO"] > 2.50:
        return "TOO_VOLATILE"

    if row["BB_WIDTH"] < row["BB_WIDTH_MA"] or row["ATR_RATIO"] < 0.60:
        return "CONSOLIDATING"

    return "TRADABLE"
