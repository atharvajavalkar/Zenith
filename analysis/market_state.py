"""Market regime classification from laminar flow and volatility metrics."""

import pandas as pd


def get_market_state(row: pd.Series) -> str:
    atr_ratio = float(row["ATR_RATIO"])
    laminar = float(row["LAMINAR"]) > 0.05

    ema_bull = row["EMA9"] > row["EMA21"]
    ema_bear = row["EMA9"] < row["EMA21"]

    lap_bull = row["LAP"] > 0
    lap_bear = row["LAP"] < 0

    if laminar and ema_bull and lap_bull:
        return "LAMINAR_BULL"
    if laminar and ema_bear and lap_bear:
        return "LAMINAR_BEAR"

    if atr_ratio > 1.80 and ema_bull:
        return "TURBULENT_BULL"
    if atr_ratio > 1.80 and ema_bear:
        return "TURBULENT_BEAR"

    if ema_bull or lap_bull:
        return "TRANSITION_BULL"
    if ema_bear or lap_bear:
        return "TRANSITION_BEAR"

    return "CHOP"
