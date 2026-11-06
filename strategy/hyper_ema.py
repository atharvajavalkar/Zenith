"""Hyper EMA(8/14) scalp strategy — fully separate from the Zenith zone algo.

Rules (grounded in EMA-ribbon / ChoCh practice + user feedback):
  • Directional bias from 5m EMA8/14 — only trade WITH that bias.
  • Entry timing on fast candles (default 10s from ticks).
  • TP/SL distances from 1m ATR (stable stops; not 10s noise).
  • SKIP when price sits BETWEEN the EMAs — that's a Change of Character zone.
  • SKIP when the ribbon is flat/braided (EMAs too close) or a fresh cross.
  • Trade WITH the ribbon on a pullback: EMA8>EMA14 → BUY near the band;
    EMA8<EMA14 → SELL near the band.
  • Micro FRVP (~30 × 10s / 5m) must agree: BUY above POC, SELL below POC;
    skip the POC magnet zone.
  • Hyper RL (strategy/hyper_rl.py) then overlays lot scale + threshold tweaks
    learned from past hyper trade outcomes.

This module never touches strategy/signal.py.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, float] = {
    "ema_fast": 8,
    "ema_slow": 14,
    "tf": "10s",        # seconds-based candles built live from ticks (entry timing)
    "bias_tf": "5m",    # higher-timeframe bias — Hyper only trades WITH this
    "atr_tf": "1m",     # TP/SL sized off 1-minute ATR (not 10s noise)
    "bars": 120,
    "bias_bars": 120,
    "atr_bars": 120,
    "near_atr": 0.35,   # only tight pullbacks into the ribbon
    "max_dist_atr": 1.2,  # skip chase
    "confirm_bars": 3,
    "flat_atr": 0.22,   # min ribbon spread (|EMA8-EMA14|/ATR) — below = braided/ChoCh
    "choch_lookback": 4,  # bars to look back for a fresh EMA8/EMA14 cross (ChoCh)
    "tp_atr": 0.85,     # TP wider than SL so winners can cover losers + spread
    "sl_atr": 0.55,
    "min_atr_ratio": 0.05,
}

_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 1.5
_BIAS_CACHE_TTL = 45.0  # 5m bias does not need sub-second refresh
_ATR_CACHE_TTL = 20.0   # 1m ATR refresh cadence
_bias_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_last_bias: dict[str, dict[str, Any]] = {}
_BIAS_LOCK = threading.Lock()
_atr_cache: dict[str, tuple[float, float]] = {}  # key -> (ts, atr)

# ── Live seconds-candle builder (tick-fed) ───────────────────────────────
_SEC_LOCK = threading.Lock()
_sec_bars: dict[str, deque] = {}   # key -> deque[[t0,o,h,l,c,tick_volume]]
_MAX_BARS = 80


def _tf_seconds(tf: str) -> Optional[int]:
    """Return bar length in seconds for '5s'/'10s'/'30s' style tf, else None."""
    t = str(tf).lower().strip()
    if t.endswith("s") and t[:-1].isdigit():
        return max(1, int(t[:-1]))
    return None


def feed_tick(symbol: str, price: float, bar_sec: int, ts: Optional[float] = None) -> None:
    """Aggregate a tick into the live seconds-candle for `symbol`."""
    s = str(symbol).upper()
    key = f"{s}:{int(bar_sec)}"
    t = float(ts if ts is not None else time.time())
    bucket = int(t // int(bar_sec)) * int(bar_sec)
    px = float(price)
    with _SEC_LOCK:
        dq = _sec_bars.get(key)
        if dq is None:
            dq = deque(maxlen=_MAX_BARS)
            _sec_bars[key] = dq
        if dq and dq[-1][0] == bucket:
            bar = dq[-1]
            if px > bar[2]:
                bar[2] = px
            if px < bar[3]:
                bar[3] = px
            bar[4] = px
            bar[5] += 1.0
        else:
            dq.append([bucket, px, px, px, px, 1.0])


def _seconds_frame(symbol: str, bar_sec: int):
    import pandas as pd

    key = f"{str(symbol).upper()}:{int(bar_sec)}"
    with _SEC_LOCK:
        dq = _sec_bars.get(key)
        rows = list(dq) if dq else []
    if len(rows) < 3:
        return None
    df = pd.DataFrame(rows, columns=["t0", "open", "high", "low", "close", "volume"])
    return df


def _fetch_ohlcv_fast(symbol: str, tf: str, bars: int):
    """Prefer MetaAPI candles when configured (even in paper) — TV is too slow for Hyper."""
    try:
        from data.mt5_client import mt5_client

        if mt5_client.configured and not mt5_client._in_backoff():  # noqa: SLF001
            df = mt5_client.fetch_candles(tf, bars=min(int(bars), 500), symbol=symbol)
            if df is not None and len(df) >= 10:
                return df
    except Exception as exc:
        logger.debug("hyper MetaAPI candles %s %s failed: %s", symbol, tf, exc)
    from data.market_data import fetch_ohlcv

    return fetch_ohlcv(tf, bars, symbol=symbol)


def _fetch_frame(symbol: str, tf: str, bars: int):
    bar_sec = _tf_seconds(tf)
    if bar_sec is not None:
        df = _seconds_frame(symbol, bar_sec)
        # Need enough bars for EMA14 + ATR. In paper mode TradingView ticks are
        # too slow to fill the live 10s cache, so fall back to 1m OHLC until
        # the seconds candles are warm enough.
        need = 20
        if df is not None and len(df) >= need:
            return df
        try:
            fb = _fetch_ohlcv_fast(symbol, "1m", max(bars, 120))
            if fb is not None and len(fb) >= need:
                return fb
        except Exception as exc:
            logger.debug("hyper 1m entry fallback failed %s: %s", symbol, exc)
        return df

    key = f"{symbol}:{tf}:{bars}"
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    df = _fetch_ohlcv_fast(symbol, tf, bars)
    _cache[key] = (now, df)
    return df


def compute_htf_bias(
    symbol: str,
    *,
    settings: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """EMA8/14 bias on the higher timeframe (default 5m).

    BUY/BULLISH when EMA8 >= EMA14, else SELL/BEARISH.
    Cached ~45s so the 1s Hyper loop does not hammer MetaAPI for 5m candles.
    """
    cfg = dict(DEFAULTS)
    if settings:
        for k in ("ema_fast", "ema_slow", "bias_tf", "bias_bars"):
            if k in settings and settings[k] is not None:
                cfg[k] = settings[k]

    bias_tf = str(cfg.get("bias_tf") or "5m")
    bars = int(cfg.get("bias_bars") or 120)
    key = f"{str(symbol).upper()}:{bias_tf}:{bars}"
    now = time.time()
    hit = _bias_cache.get(key)
    if hit and now - hit[0] < _BIAS_CACHE_TTL:
        return dict(hit[1])

    try:
        # Prefer MetaAPI 5m when configured (paper mode otherwise hangs on TV)
        df = _fetch_ohlcv_fast(symbol, bias_tf, bars)
    except Exception as exc:
        logger.debug("hyper 5m bias fetch failed %s: %s", symbol, exc)
        return None
    if df is None or len(df) < int(cfg["ema_slow"]) + 5:
        return None

    from indicators.technical import ema as _ema

    close = df["close"].astype(float)
    ema_f = _ema(close, int(cfg["ema_fast"]))
    ema_s = _ema(close, int(cfg["ema_slow"]))
    ef = float(ema_f.iloc[-1])
    es = float(ema_s.iloc[-1])
    up = ef >= es
    out = {
        "symbol": str(symbol).upper(),
        "tf": bias_tf,
        "signal": "BUY" if up else "SELL",
        "direction": "BULLISH" if up else "BEARISH",
        "ema_fast": round(ef, 6),
        "ema_slow": round(es, 6),
        "updated_at": now,
    }
    _bias_cache[key] = (now, out)
    with _BIAS_LOCK:
        _last_bias[str(symbol).upper()] = dict(out)
    return dict(out)


def bias_snapshot() -> dict[str, Any]:
    with _BIAS_LOCK:
        rows = [dict(v) for v in _last_bias.values()]
    rows.sort(key=lambda r: str(r.get("symbol") or ""))
    return {"symbols": rows}


def compute_stop_atr(
    symbol: str,
    *,
    settings: Optional[dict[str, Any]] = None,
    fallback: float = 0.0,
) -> float:
    """1-minute ATR used for Hyper TP/SL distances (cached ~20s)."""
    cfg = dict(DEFAULTS)
    if settings:
        for k in ("atr_tf", "atr_bars"):
            if k in settings and settings[k] is not None:
                cfg[k] = settings[k]
    atr_tf = str(cfg.get("atr_tf") or "1m")
    bars = int(cfg.get("atr_bars") or 120)
    key = f"{str(symbol).upper()}:{atr_tf}:{bars}"
    now = time.time()
    hit = _atr_cache.get(key)
    if hit and now - hit[0] < _ATR_CACHE_TTL:
        return float(hit[1])
    try:
        df = _fetch_frame(symbol, atr_tf, bars)
        if df is None or len(df) < 20:
            return float(fallback or 0.0)
        from indicators.technical import atr as _atr

        series = _atr(df, 14)
        a = float(series.iloc[-1]) if series.iloc[-1] == series.iloc[-1] else 0.0
        if a <= 0:
            return float(fallback or 0.0)
        _atr_cache[key] = (now, a)
        return a
    except Exception as exc:
        logger.debug("hyper 1m ATR fetch failed %s: %s", symbol, exc)
        return float(fallback or 0.0)


def evaluate_ema_scalp(
    symbol: str,
    *,
    price: Optional[float] = None,
    settings: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Return a scalp decision dict or None.

    Rules (grounded in EMA-ribbon + ChoCh practice):
      • Higher-TF bias (default 5m EMA8/14) must agree — no counter-trend punches.
      • Skip when ribbon is flat/braided OR price sits in the middle (ChoCh zone).
      • Skip on a fresh EMA8/EMA14 cross (character change in progress).
      • Trade WITH the ribbon: EMA8 > EMA14 → BUY on a pullback into the near
        band; EMA8 < EMA14 → SELL on a pullback into the near band.
    """
    cfg = dict(DEFAULTS)
    if settings:
        for k in cfg:
            if k in settings and settings[k] is not None:
                cfg[k] = settings[k]

    # 0. Higher-timeframe bias first (5m) — blocks counter-trend scam
    htf = compute_htf_bias(symbol, settings=cfg)
    if not htf:
        return None
    bias_signal = str(htf.get("signal") or "")

    tf = str(cfg["tf"])
    bars = int(cfg["bars"])
    try:
        df = _fetch_frame(symbol, tf, bars)
    except Exception as exc:
        logger.debug("hyper_ema fetch failed %s: %s", symbol, exc)
        return None
    if df is None or len(df) < int(cfg["ema_slow"]) + 5:
        return None

    from indicators.technical import atr as _atr, ema as _ema

    close = df["close"].astype(float)
    ema_f = _ema(close, int(cfg["ema_fast"]))
    ema_s = _ema(close, int(cfg["ema_slow"]))
    atr_series = _atr(df, 14)

    last_close = float(price) if price is not None else float(close.iloc[-1])
    ef = float(ema_f.iloc[-1])
    es = float(ema_s.iloc[-1])
    # Entry-timing ATR from the fast (10s) frame — drives ribbon/near/flat gates
    a_entry = float(atr_series.iloc[-1]) if atr_series.iloc[-1] == atr_series.iloc[-1] else 0.0
    if a_entry <= 0:
        return None
    # Stop-sizing ATR from 1m — TP/SL distances stay stable vs 10s noise
    a_stop = compute_stop_atr(symbol, settings=cfg, fallback=a_entry)
    a = a_entry

    # Mid of the EMA ribbon
    band = (ef + es) / 2.0
    dist_atr = (last_close - band) / a
    abs_dist = abs(dist_atr)
    ribbon_spread = abs(ef - es) / a

    # Hyper-only rolling FRVP: 30 × 10s bars by default (~5 minutes).
    # Build it before EMA gates so the dashboard profile remains current even
    # while ChoCh / ribbon filters are blocking entries.
    from strategy.hyper_vp import build_micro_profile

    micro_vp = build_micro_profile(
        df,
        symbol=symbol,
        price=last_close,
        atr=a,
        timeframe=tf,
    )

    # 1. Flat / braided ribbon → chop — no edge (algo-matrix / tradingcompendium)
    if ribbon_spread < float(cfg["flat_atr"]):
        return None

    # 2. Change of Character — price sitting BETWEEN the two EMAs
    #    (the "middle of EMA" zone the user flagged in the GBPUSD chart)
    ribbon_lo = min(ef, es)
    ribbon_hi = max(ef, es)
    in_middle = ribbon_lo <= last_close <= ribbon_hi
    if in_middle:
        return None

    # 3. Fresh EMA cross in lookback = ChoCh in progress — wait for it to settle
    lb = max(1, int(cfg["choch_lookback"]))
    if len(ema_f) > lb + 1:
        signs = [1 if float(ema_f.iloc[-i]) >= float(ema_s.iloc[-i]) else -1 for i in range(1, lb + 2)]
        if any(signs[i] != signs[i + 1] for i in range(len(signs) - 1)):
            return None  # cross occurred within lookback

    # 4. Too stretched → don't chase; wait for a pullback into the near band
    if abs_dist > float(cfg["max_dist_atr"]):
        return None
    if abs_dist > float(cfg["near_atr"]):
        return None  # only enter on a pullback / hug

    # 5. Trade WITH the ribbon (trend) on a pullback into the near band
    up = ef >= es
    if up:
        # Bullish ribbon — only BUY when price is near/above the band (no shorting uptrend)
        if last_close < ribbon_lo:
            return None
        signal = "BUY"
        side = "above"
    else:
        # Bearish ribbon — only SELL when price is near/below the band
        if last_close > ribbon_hi:
            return None
        signal = "SELL"
        side = "below"

    # Must match 5m bias — no buying into a 5m bearish ribbon (and vice versa)
    if signal != bias_signal:
        return None

    # Confirm recent closes agree with the trend direction
    n_conf = max(1, int(cfg["confirm_bars"]))
    recent = close.iloc[-n_conf:]
    if signal == "BUY":
        ok = all(float(c) >= ribbon_lo * 0.9995 for c in recent)  # didn't dump under ribbon
    else:
        ok = all(float(c) <= ribbon_hi * 1.0005 for c in recent)
    if not ok:
        return None

    # EMA direction must agree with the current micro auction: BUY above POC,
    # SELL below POC; skip the POC magnet/chop zone.
    from strategy.hyper_vp import allows_signal

    vp_ok, vp_reason = allows_signal(micro_vp, signal)
    if not vp_ok:
        return None

    return {
        "signal": signal,
        "mode": "pullback",
        "side": side,
        "choch": False,
        "bias_tf": htf.get("tf"),
        "bias_5m": htf.get("direction"),
        "bias_signal": bias_signal,
        "ribbon_spread": round(ribbon_spread, 3),
        "ema_fast": round(ef, 6),
        "ema_slow": round(es, 6),
        "ema_band": round(band, 6),
        "atr": round(a_stop, 6),           # TP/SL sizing (1m ATR)
        "atr_entry": round(a_entry, 6),    # 10s ribbon ATR
        "atr_tf": str(cfg.get("atr_tf") or "1m"),
        "dist_atr": round(dist_atr, 3),
        "vp_poc": micro_vp.get("poc"),
        "vp_vah": micro_vp.get("vah"),
        "vp_val": micro_vp.get("val"),
        "vp_location": micro_vp.get("location"),
        "vp_dist_poc_atr": micro_vp.get("dist_to_poc_atr"),
        "vp_reason": vp_reason,
        "tp_atr": float(cfg["tp_atr"]),
        "sl_atr": float(cfg["sl_atr"]),
        "entry": last_close,
        "tf": tf,
    }
