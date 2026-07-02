"""Core technical indicator calculations."""

from typing import Tuple

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(df["close"], 12)
    slow = ema(df["close"], 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def bollinger(
    df: pd.DataFrame, n: int = 20, k: int = 2
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width


def donchian(df: pd.DataFrame, n: int = 20) -> Tuple[pd.Series, pd.Series]:
    return df["high"].rolling(n).max(), df["low"].rolling(n).min()


def laplacian(s: pd.Series) -> pd.Series:
    return s - 2 * s.shift(1) + s.shift(2)
