"""Hyper Strategy B — 1m up/down volume imbalance (poor-man's heatmap).

Desk idea (GS / Bloomberg auction style):
  • Aggressor proxy: green candle volume = buy pressure, red = sell pressure.
  • Trade WITH rolling imbalance (not against the tape).
  • Enter on a pullback into micro POC / value (balance → continuation).
  • Stops from swing structure — NO ATR.
  • Higher-TF (5m) imbalance must agree — no counter-flow punches.

Separate from Zenith physics (strategy/signal.py) and the old EMA/ATR Hyper.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "tf": "1m",
    "bias_tf": "5m",
    "bars": 120,
    "bias_bars": 80,
    "imb_lookback": 20,       # rolling heatmap window (1m bars)
    "min_imbalance": 0.18,    # |buy-sell|/(buy+sell) to have a clear heat
    "poc_pullback_bps": 8.0,  # max distance to POC in bps of price (no ATR)
    "swing_lookback": 10,     # bars for structure SL/TP
    "min_rr": 1.25,           # TP must clear this vs structure risk
    "confirm_bars": 2,
}

_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 2.0
_BIAS_TTL = 40.0
_bias_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_last_bias: dict[str, dict[str, Any]] = {}
_last_flow: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _fetch_ohlcv(symbol: str, tf: str, bars: int):
    key = f"{str(symbol).upper()}:{tf}:{bars}"
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    try:
        from data.mt5_client import mt5_client

        if mt5_client.configured and not mt5_client._in_backoff():  # noqa: SLF001
            df = mt5_client.fetch_candles(tf, bars=min(int(bars), 500), symbol=symbol)
            if df is not None and len(df) >= 20:
                _cache[key] = (now, df)
                return df
    except Exception as exc:
        logger.debug("hyper_flow MetaAPI %s %s: %s", symbol, tf, exc)
    try:
        from data.market_data import fetch_ohlcv

        df = fetch_ohlcv(tf, bars, symbol=symbol)
        _cache[key] = (now, df)
        return df
    except Exception as exc:
        logger.debug("hyper_flow fetch %s %s: %s", symbol, tf, exc)
        return None


def _up_down_volumes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Poor-man's heatmap: close>=open → buy vol, else sell vol."""
    o = df["open"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    if "volume" in df.columns:
        v = df["volume"].astype(float).fillna(0.0).to_numpy()
        if float(np.nansum(v)) <= 0:
            v = (df["high"].astype(float) - df["low"].astype(float)).clip(lower=1e-9).to_numpy()
    else:
        v = (df["high"].astype(float) - df["low"].astype(float)).clip(lower=1e-9).to_numpy()
    buy = np.where(c >= o, v, 0.0)
    sell = np.where(c < o, v, 0.0)
    return buy, sell


def rolling_imbalance(df: pd.DataFrame, lookback: int) -> Optional[float]:
    if df is None or len(df) < max(5, lookback):
        return None
    buy, sell = _up_down_volumes(df.tail(int(lookback)))
    b, s = float(buy.sum()), float(sell.sum())
    tot = b + s
    if tot <= 0:
        return None
    return (b - s) / tot


def _swing_levels(df: pd.DataFrame, lookback: int) -> tuple[float, float]:
    win = df.tail(max(3, int(lookback)))
    return float(win["low"].min()), float(win["high"].max())


def compute_flow_bias(
    symbol: str,
    *,
    settings: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """5m imbalance bias — BUY if heat > 0 else SELL."""
    cfg = dict(DEFAULTS)
    if settings:
        for k in ("bias_tf", "bias_bars", "imb_lookback", "min_imbalance"):
            if k in settings and settings[k] is not None:
                cfg[k] = settings[k]
    bias_tf = str(cfg.get("bias_tf") or "5m")
    bars = int(cfg.get("bias_bars") or 80)
    key = f"{str(symbol).upper()}:{bias_tf}:{bars}"
    now = time.time()
    hit = _bias_cache.get(key)
    if hit and now - hit[0] < _BIAS_TTL:
        return dict(hit[1])
    df = _fetch_ohlcv(symbol, bias_tf, bars)
    if df is None or len(df) < 20:
        return None
    lb = min(int(cfg["imb_lookback"]), len(df))
    imb = rolling_imbalance(df, lb)
    if imb is None:
        return None
    # Soft bias: any signed heat; strength still gated at entry by min_imbalance
    up = imb >= 0
    out = {
        "symbol": str(symbol).upper(),
        "tf": bias_tf,
        "signal": "BUY" if up else "SELL",
        "direction": "BULLISH" if up else "BEARISH",
        "imbalance": round(float(imb), 4),
        "updated_at": now,
    }
    _bias_cache[key] = (now, out)
    with _LOCK:
        _last_bias[str(symbol).upper()] = dict(out)
    return dict(out)


def bias_snapshot() -> dict[str, Any]:
    with _LOCK:
        rows = [dict(v) for v in _last_bias.values()]
    rows.sort(key=lambda r: str(r.get("symbol") or ""))
    return {"symbols": rows}


def flow_snapshot() -> dict[str, Any]:
    with _LOCK:
        rows = [dict(v) for v in _last_flow.values()]
    rows.sort(key=lambda r: str(r.get("symbol") or ""))
    return {"flows": rows}


def _record_flow(symbol: str, **fields: Any) -> None:
    row = {
        "symbol": str(symbol).upper(),
        "updated_at": time.time(),
        **fields,
    }
    with _LOCK:
        prev = dict(_last_flow.get(str(symbol).upper()) or {})
        prev.update(row)
        _last_flow[str(symbol).upper()] = prev


def probe_flow_board(
    symbol: str,
    *,
    price: Optional[float] = None,
    settings: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Always refresh the 1m heat board; return a trade only when fully ready."""
    return evaluate_flow_scalp(symbol, price=price, settings=settings, record_board=True)


def evaluate_flow_scalp(
    symbol: str,
    *,
    price: Optional[float] = None,
    settings: Optional[dict[str, Any]] = None,
    record_board: bool = True,
) -> Optional[dict[str, Any]]:
    """Return a Hyper B decision or None. No ATR."""
    cfg = dict(DEFAULTS)
    if settings:
        for k in cfg:
            if k in settings and settings[k] is not None:
                cfg[k] = settings[k]

    sym = str(symbol).upper()

    def _board(**kw: Any) -> None:
        if record_board:
            _record_flow(sym, **kw)

    htf = compute_flow_bias(symbol, settings=cfg)
    if not htf:
        _board(status="NO_BIAS", signal=None, imbalance=None, poc_dist_bps=None, bias_5m=None)
        return None
    bias_signal = str(htf.get("signal") or "")

    tf = str(cfg["tf"])
    bars = int(cfg["bars"])
    df = _fetch_ohlcv(symbol, tf, bars)
    if df is None or len(df) < max(25, int(cfg["imb_lookback"]) + 5):
        _board(
            status="WARMING",
            signal=None,
            imbalance=None,
            poc_dist_bps=None,
            bias_5m=htf.get("direction"),
        )
        return None

    last_close = float(price) if price is not None else float(df["close"].iloc[-1])
    lb = int(cfg["imb_lookback"])
    imb = rolling_imbalance(df, lb)
    if imb is None:
        _board(status="NO_VOL", signal=None, imbalance=None, bias_5m=htf.get("direction"))
        return None

    signal = "BUY" if imb > 0 else "SELL"
    min_imb = float(cfg["min_imbalance"])
    base = {
        "imbalance": round(float(imb), 4),
        "signal": signal,
        "bias_5m": htf.get("direction"),
        "bias_imbalance": htf.get("imbalance"),
    }

    if abs(imb) < min_imb:
        _board(status="WEAK_HEAT", skip=f"|imb|<{min_imb}", **base)
        return None

    if signal != bias_signal:
        _board(status="VS_BIAS", skip=f"1m {signal} vs 5m {bias_signal}", **base)
        return None

    from strategy.hyper_vp import allows_signal, build_micro_profile

    rng = float((df["high"] - df["low"]).tail(20).mean() or abs(last_close) * 0.0005)
    micro_vp = build_micro_profile(
        df,
        symbol=symbol,
        price=last_close,
        atr=max(rng, abs(last_close) * 1e-6),
        bars=30,
        timeframe=tf,
    )
    poc = micro_vp.get("poc")
    if not micro_vp.get("available") or poc is None:
        _board(status="NO_POC", skip="micro VP warming", **base)
        return None
    poc = float(poc)
    dist_bps = abs(last_close - poc) / max(abs(last_close), 1e-9) * 1e4
    base["poc_dist_bps"] = round(dist_bps, 2)
    base["vp_location"] = micro_vp.get("location")

    if dist_bps > float(cfg["poc_pullback_bps"]):
        _board(
            status="FAR_POC",
            skip=f"{dist_bps:.1f}bps > {cfg['poc_pullback_bps']}bps",
            **base,
        )
        return None

    vp_ok, vp_reason = allows_signal(micro_vp, signal)
    if not vp_ok:
        loc = str(micro_vp.get("location") or "")
        if signal == "BUY" and loc in ("BELOW_VALUE",):
            _board(status="VS_VALUE", skip=vp_reason, **base)
            return None
        if signal == "SELL" and loc in ("ABOVE_VALUE",):
            _board(status="VS_VALUE", skip=vp_reason, **base)
            return None
        if "POC" in vp_reason and dist_bps < float(cfg["poc_pullback_bps"]) * 0.35:
            _board(status="AT_POC", skip=vp_reason, **base)
            return None

    swing_lo, swing_hi = _swing_levels(df, int(cfg["swing_lookback"]))
    pad = max(abs(last_close) * 0.00015, 1e-6)
    if signal == "BUY":
        sl = swing_lo - pad
        risk = last_close - sl
        if risk <= 0:
            _board(status="BAD_SL", skip="no buy risk", **base)
            return None
        struct_tp = swing_hi
        rr_tp = last_close + float(cfg["min_rr"]) * risk
        tp = max(struct_tp, rr_tp)
        if tp <= last_close:
            _board(status="BAD_TP", skip="no buy target", **base)
            return None
    else:
        sl = swing_hi + pad
        risk = sl - last_close
        if risk <= 0:
            _board(status="BAD_SL", skip="no sell risk", **base)
            return None
        struct_tp = swing_lo
        rr_tp = last_close - float(cfg["min_rr"]) * risk
        tp = min(struct_tp, rr_tp)
        if tp >= last_close:
            _board(status="BAD_TP", skip="no sell target", **base)
            return None

    n_conf = max(1, int(cfg["confirm_bars"]))
    recent = df["close"].astype(float).iloc[-n_conf:]
    if signal == "BUY":
        if not all(float(c) >= swing_lo for c in recent):
            _board(status="NO_CONFIRM", skip="closes broke swing low", **base)
            return None
    else:
        if not all(float(c) <= swing_hi for c in recent):
            _board(status="NO_CONFIRM", skip="closes broke swing high", **base)
            return None

    out = {
        "signal": signal,
        "mode": "flow_imbalance",
        "side": "buy_heat" if signal == "BUY" else "sell_heat",
        "imbalance": round(float(imb), 4),
        "bias_tf": htf.get("tf"),
        "bias_5m": htf.get("direction"),
        "bias_signal": bias_signal,
        "bias_imbalance": htf.get("imbalance"),
        "poc": poc,
        "poc_dist_bps": round(dist_bps, 2),
        "vp_location": micro_vp.get("location"),
        "vp_vah": micro_vp.get("vah"),
        "vp_val": micro_vp.get("val"),
        "vp_reason": vp_reason,
        "swing_low": round(swing_lo, 8),
        "swing_high": round(swing_hi, 8),
        "sl": float(sl),
        "tp": float(tp),
        "entry": last_close,
        "tf": tf,
        "rr": round(abs(tp - last_close) / max(abs(last_close - sl), 1e-12), 2),
    }
    _board(
        status="READY",
        skip=None,
        imbalance=out["imbalance"],
        signal=signal,
        poc_dist_bps=out["poc_dist_bps"],
        bias_5m=out["bias_5m"],
        vp_location=out["vp_location"],
    )
    return out
