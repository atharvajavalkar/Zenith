"""Session volume profile — POC, value area (VAH/VAL), and location vs price.

Supports rolling lookback AVP and fixed-range VP (FRVP) from a previous session
(Asia / London / NY). Broker/tick volume is relative, not exchange-certified.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

DEFAULT_LOOKBACK = 288  # 24h * 12 bars/hour at 5m
VALUE_AREA_PCT = 0.70
N_BINS = 48


def _distribute_bar_volume(
    low: float,
    high: float,
    close: float,
    volume: float,
    edges: np.ndarray,
) -> np.ndarray:
    """Spread bar volume across overlapping price bins (slight close bias)."""
    n = len(edges) - 1
    out = np.zeros(n, dtype=np.float64)
    if volume <= 0 or not np.isfinite(volume):
        return out
    if high < low:
        low, high = high, low
    if high == low:
        mid = close if np.isfinite(close) else low
        idx = int(np.clip(np.searchsorted(edges, mid, side="right") - 1, 0, n - 1))
        out[idx] = volume
        return out

    span = high - low
    for i in range(n):
        b_lo, b_hi = edges[i], edges[i + 1]
        overlap = max(0.0, min(high, b_hi) - max(low, b_lo))
        if overlap <= 0:
            continue
        share = overlap / span
        bin_mid = 0.5 * (b_lo + b_hi)
        close_w = 1.0 + 0.35 * (1.0 - min(1.0, abs(bin_mid - close) / max(span, 1e-9)))
        out[i] += volume * share * close_w
    total = out.sum()
    if total > 0:
        out *= volume / total
    return out


def _empty_profile(summary: str = "Volume profile unavailable") -> dict[str, Any]:
    return {
        "available": False,
        "poc": None,
        "vah": None,
        "val": None,
        "location": "UNKNOWN",
        "direction": "NEUTRAL",
        "score": 0.0,
        "confidence": 50.0,
        "dist_to_poc_atr": None,
        "in_value": False,
        "above_value": False,
        "below_value": False,
        "at_poc": False,
        "mc_adjust_hint": 0.0,
        "summary": summary,
        "bars_used": 0,
        "source": None,
        "session_name": None,
    }


def _round_px(x: float, symbol: Optional[str] = None) -> float:
    from strategy.instrument_meta import price_digits

    return round(float(x), price_digits(symbol))


def _profile_from_window(
    window: pd.DataFrame,
    *,
    price: Optional[float],
    atr: Optional[float],
    n_bins: int,
    value_area_pct: float,
    symbol: Optional[str] = None,
    source: str = "rolling",
    session_name: Optional[str] = None,
) -> dict[str, Any]:
    empty = _empty_profile()
    if window is None or len(window) < 12:
        return empty

    win = window.copy()
    if "volume" not in win.columns:
        win["volume"] = 1.0
    vol = win["volume"].astype(float).fillna(0.0)
    if float(vol.sum()) <= 0:
        rng = (win["high"] - win["low"]).astype(float).clip(lower=1e-6)
        win = win.assign(volume=rng)

    lo = float(win["low"].min())
    hi = float(win["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return empty

    edges = np.linspace(lo, hi, n_bins + 1)
    hist = np.zeros(n_bins, dtype=np.float64)
    for _, row in win.iterrows():
        hist += _distribute_bar_volume(
            float(row["low"]),
            float(row["high"]),
            float(row["close"]),
            float(row["volume"]),
            edges,
        )

    if hist.sum() <= 0:
        return empty

    poc_idx = int(np.argmax(hist))
    poc = float(0.5 * (edges[poc_idx] + edges[poc_idx + 1]))

    target = float(hist.sum()) * float(value_area_pct)
    left = right = poc_idx
    covered = float(hist[poc_idx])
    while covered < target and (left > 0 or right < n_bins - 1):
        left_vol = hist[left - 1] if left > 0 else -1.0
        right_vol = hist[right + 1] if right < n_bins - 1 else -1.0
        if right_vol >= left_vol and right < n_bins - 1:
            right += 1
            covered += float(hist[right])
        elif left > 0:
            left -= 1
            covered += float(hist[left])
        else:
            break

    val = float(edges[left])
    vah = float(edges[right + 1])

    px = float(price if price is not None else win["close"].iloc[-1])
    atr_v = float(atr if atr is not None else (win["high"] - win["low"]).mean() or 1.0)
    atr_v = max(atr_v, 1e-9)
    dist_poc = (px - poc) / atr_v

    poc_buf = 0.15 * atr_v
    at_poc = abs(px - poc) <= poc_buf
    in_value = val <= px <= vah
    above_value = px > vah
    below_value = px < val

    if at_poc:
        location = "AT_POC"
    elif above_value:
        location = "ABOVE_VALUE"
    elif below_value:
        location = "BELOW_VALUE"
    elif in_value and px >= poc:
        location = "IN_VALUE_UPPER"
    elif in_value:
        location = "IN_VALUE_LOWER"
    else:
        location = "IN_VALUE"

    if above_value:
        score = min(2.0, 0.8 + abs(dist_poc) * 0.25)
        direction = "BULLISH"
    elif below_value:
        score = -min(2.0, 0.8 + abs(dist_poc) * 0.25)
        direction = "BEARISH"
    elif location == "IN_VALUE_UPPER":
        score = 0.35
        direction = "BULLISH"
    elif location == "IN_VALUE_LOWER":
        score = -0.35
        direction = "BEARISH"
    else:
        score = 0.0
        direction = "NEUTRAL"

    confidence = round(min(88.0, 58.0 + abs(score) * 12.0), 1)
    mc_hint = max(-2.5, min(2.5, score * 1.1))

    tag = session_name or source
    labels = {
        "AT_POC": f"At {tag} POC {_round_px(poc, symbol)} — balance / magnet",
        "ABOVE_VALUE": f"Above {tag} value (VAH {_round_px(vah, symbol)}) — bullish acceptance",
        "BELOW_VALUE": f"Below {tag} value (VAL {_round_px(val, symbol)}) — bearish acceptance",
        "IN_VALUE_UPPER": f"In {tag} value upper (POC {_round_px(poc, symbol)}) — mild bullish lean",
        "IN_VALUE_LOWER": f"In {tag} value lower (POC {_round_px(poc, symbol)}) — mild bearish lean",
        "IN_VALUE": f"In {tag} value {_round_px(val, symbol)}–{_round_px(vah, symbol)}",
    }

    return {
        "available": True,
        "poc": _round_px(poc, symbol),
        "vah": _round_px(vah, symbol),
        "val": _round_px(val, symbol),
        "location": location,
        "direction": direction,
        "score": round(score, 3),
        "confidence": confidence,
        "dist_to_poc_atr": round(float(dist_poc), 3),
        "in_value": bool(in_value and not above_value and not below_value),
        "above_value": bool(above_value),
        "below_value": bool(below_value),
        "at_poc": bool(at_poc),
        "mc_adjust_hint": round(mc_hint, 2),
        "summary": labels.get(location, "Volume profile"),
        "bars_used": int(len(win)),
        "source": source,
        "session_name": session_name,
    }


def build_volume_profile(
    df: pd.DataFrame,
    *,
    price: Optional[float] = None,
    atr: Optional[float] = None,
    lookback: int = DEFAULT_LOOKBACK,
    n_bins: int = N_BINS,
    value_area_pct: float = VALUE_AREA_PCT,
    symbol: Optional[str] = None,
) -> dict[str, Any]:
    """Rolling lookback AVP from recent OHLCV bars."""
    if df is None or len(df) < 30:
        return _empty_profile()
    window = df.tail(min(lookback, len(df)))
    return _profile_from_window(
        window,
        price=price,
        atr=atr,
        n_bins=n_bins,
        value_area_pct=value_area_pct,
        symbol=symbol,
        source="rolling",
    )


def _slice_session(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    try:
        df.index = pd.to_datetime(df.index, utc=True)
    except Exception:
        return df.iloc[0:0]
    start_u = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_u = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
    return df.loc[(df.index >= start_u) & (df.index < end_u)]


def build_frvp_previous_session(
    df: pd.DataFrame,
    *,
    price: Optional[float] = None,
    atr: Optional[float] = None,
    n_bins: int = N_BINS,
    value_area_pct: float = VALUE_AREA_PCT,
    symbol: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Fixed-range VP from last completed Asia/London/NY session."""
    from analysis.session_windows import previous_session_window, prior_day_window

    if df is None or len(df) < 30:
        return _empty_profile("FRVP unavailable")

    start, end, name = previous_session_window(now)
    window = _slice_session(df, start, end)
    if window is None or len(window) < 12:
        start, end, name = prior_day_window(now)
        window = _slice_session(df, start, end)

    if window is None or len(window) < 12:
        out = build_volume_profile(
            df, price=price, atr=atr, symbol=symbol, n_bins=n_bins, value_area_pct=value_area_pct
        )
        out["source"] = "rolling_fallback"
        return out

    return _profile_from_window(
        window,
        price=price,
        atr=atr,
        n_bins=n_bins,
        value_area_pct=value_area_pct,
        symbol=symbol,
        source="frvp_prev_session",
        session_name=name,
    )


def build_dual_profile(
    df: pd.DataFrame,
    *,
    price: Optional[float] = None,
    atr: Optional[float] = None,
    lookback: int = DEFAULT_LOOKBACK,
    symbol: Optional[str] = None,
) -> dict[str, Any]:
    """Prefer previous-session FRVP POC; blend score with rolling AVP."""
    frvp = build_frvp_previous_session(df, price=price, atr=atr, symbol=symbol)
    roll = build_volume_profile(df, price=price, atr=atr, lookback=lookback, symbol=symbol)

    if frvp.get("available"):
        out = dict(frvp)
        if roll.get("available"):
            if (frvp.get("direction") == "BULLISH" and roll.get("direction") == "BEARISH") or (
                frvp.get("direction") == "BEARISH" and roll.get("direction") == "BULLISH"
            ):
                out["score"] = round(float(out.get("score") or 0) * 0.45, 3)
                out["mc_adjust_hint"] = round(float(out.get("mc_adjust_hint") or 0) * 0.45, 2)
                out["summary"] = f"{frvp.get('summary')} · rolling conflict"
            out["rolling_poc"] = roll.get("poc")
            out["rolling_location"] = roll.get("location")
        out["profile_mode"] = "frvp+rolling"
        return out

    out = dict(roll)
    out["profile_mode"] = "rolling_only"
    return out


def profile_score_nudge(profile: dict[str, Any]) -> tuple[float, float]:
    if not profile.get("available"):
        return 0.0, 0.0
    score = float(profile.get("score") or 0.0)
    buy = max(-0.75, min(0.75, score * 0.40))
    sell = max(-0.75, min(0.75, -score * 0.40))
    if profile.get("at_poc"):
        buy *= 0.5
        sell *= 0.5
    return buy, sell


def profile_mc_adjust(profile: dict[str, Any], signal: str, max_adj: float) -> float:
    if not profile.get("available") or max_adj <= 0 or signal not in ("BUY", "SELL"):
        return 0.0
    hint = float(profile.get("mc_adjust_hint") or 0.0)
    raw = hint if signal == "BUY" else -hint
    if profile.get("at_poc") or abs(float(profile.get("dist_to_poc_atr") or 99)) < 0.35:
        raw += 0.4
    return round(max(-max_adj, min(max_adj, raw)), 2)
