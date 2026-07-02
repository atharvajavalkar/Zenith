"""Enrich raw OHLCV data with all indicators used by the strategy."""

import numpy as np
import pandas as pd

from indicators.technical import (
    atr,
    bollinger,
    donchian,
    ema,
    laplacian,
    macd,
    rsi,
)
from config import SWING_LOOKBACK


def swing_points(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> tuple[pd.Series, pd.Series]:
    highs = df["high"]
    lows = df["low"]
    swing_high = pd.Series(True, index=df.index)
    swing_low = pd.Series(True, index=df.index)

    for i in range(1, lookback + 1):
        swing_high &= highs > highs.shift(i)
        swing_high &= highs >= highs.shift(-i)
        swing_low &= lows < lows.shift(i)
        swing_low &= lows <= lows.shift(-i)

    return swing_high.fillna(False), swing_low.fillna(False)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["EMA9"] = ema(df["close"], 9)
    df["EMA21"] = ema(df["close"], 21)
    df["EMA50"] = ema(df["close"], 50)
    df["ATR"] = atr(df)
    df["RSI"] = rsi(df["close"])
    df["LAP"] = laplacian(df["close"])

    _, _, macd_hist = macd(df)
    df["MACD_HIST"] = macd_hist
    df["MACD_HIST_CHANGE"] = df["MACD_HIST"].diff()

    _, bb_mid, _, bb_width = bollinger(df)
    df["BB_MID"] = bb_mid
    df["BB_WIDTH"] = bb_width
    df["BB_WIDTH_MA"] = df["BB_WIDTH"].rolling(50).mean()

    don_high, don_low = donchian(df)
    df["DONCHIAN_HIGH"] = don_high
    df["DONCHIAN_LOW"] = don_low

    df["LAMINAR"] = abs(df["EMA9"] - df["EMA21"]) / df["ATR"].replace(0, np.nan)
    df["ATR_MA"] = df["ATR"].rolling(50).mean()
    df["ATR_RATIO"] = df["ATR"] / df["ATR_MA"].replace(0, np.nan)

    df["BODY"] = (df["close"] - df["open"]).abs()
    df["RANGE"] = (df["high"] - df["low"]).replace(0, np.nan)
    df["BODY_RATIO"] = df["BODY"] / df["RANGE"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df["SWING_HIGH"], df["SWING_LOW"] = swing_points(df)
    df["BULL_FVG_LOW"] = np.where(
        df["low"] > df["high"].shift(2), df["high"].shift(2), np.nan
    )
    df["BULL_FVG_HIGH"] = np.where(
        df["low"] > df["high"].shift(2), df["low"], np.nan
    )
    df["BEAR_FVG_LOW"] = np.where(
        df["high"] < df["low"].shift(2), df["high"], np.nan
    )
    df["BEAR_FVG_HIGH"] = np.where(
        df["high"] < df["low"].shift(2), df["low"].shift(2), np.nan
    )

    return df
