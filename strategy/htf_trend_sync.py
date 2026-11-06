"""With-trend gate: only allow trades aligned with higher-TF bias.

Used so BB mean-revert becomes *pullbacks with the trend*:
  • 15m + 1h bull → BUY dips OK, block SELL
  • 15m + 1h bear → SELL rips OK, block BUY
  • disagree / flat → no trade

Retail-sized desks shouldn't fade a clean HTF trend.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def _ema_bias(df: Optional[pd.DataFrame], *, fast: int = 8, slow: int = 14) -> int:
    """+1 bull, -1 bear, 0 unknown. Uses last *closed* bar (iloc[-2] if possible)."""
    if df is None or len(df) < slow + 3:
        return 0
    # prefer completed bar
    i = -2 if len(df) >= slow + 4 else -1
    c = df["close"].astype(float)
    ema_f = c.ewm(span=fast, adjust=False).mean()
    ema_s = c.ewm(span=slow, adjust=False).mean()
    ef, es = float(ema_f.iloc[i]), float(ema_s.iloc[i])
    if ef > es:
        return 1
    if ef < es:
        return -1
    return 0


def _candle_bias(df: Optional[pd.DataFrame]) -> int:
    if df is None or len(df) < 3:
        return 0
    i = -2 if len(df) >= 4 else -1
    o, c = float(df["open"].iloc[i]), float(df["close"].iloc[i])
    if c > o:
        return 1
    if c < o:
        return -1
    return 0


def htf_sync_bias(
    df_15m: Optional[pd.DataFrame],
    df_1h: Optional[pd.DataFrame],
    *,
    mode: str = "ema",
) -> tuple[int, str]:
    """Return (bias, reason). bias in {+1, -1, 0}."""
    mode_l = (mode or "ema").strip().lower()
    if mode_l == "candle":
        b15, b1h = _candle_bias(df_15m), _candle_bias(df_1h)
        tag = "candle"
    else:
        b15, b1h = _ema_bias(df_15m), _ema_bias(df_1h)
        tag = "ema8/14"
    if b15 == 0 or b1h == 0:
        return 0, f"htf {tag} incomplete (15m={b15}, 1h={b1h})"
    if b15 != b1h:
        return 0, f"htf {tag} disagree (15m={b15}, 1h={b1h})"
    label = "BULL" if b15 == 1 else "BEAR"
    return b15, f"htf {tag} sync {label}"


def allows_with_trend(
    signal: str,
    df_15m: Optional[pd.DataFrame],
    df_1h: Optional[pd.DataFrame],
    *,
    mode: str = "ema",
) -> tuple[bool, str, dict[str, Any]]:
    """True if signal is a with-trend pullback vs 15m+1h sync."""
    side = str(signal or "").upper()
    bias, reason = htf_sync_bias(df_15m, df_1h, mode=mode)
    meta = {"htf_bias": bias, "htf_sync": reason, "htf_mode": mode}
    if bias == 0:
        return False, f"no HTF sync — {reason}", meta
    if side == "BUY" and bias != 1:
        return False, f"block BUY — {reason}", meta
    if side == "SELL" and bias != -1:
        return False, f"block SELL — {reason}", meta
    return True, f"with-trend OK — {reason}", meta
