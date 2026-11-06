"""Hyper — ZENITH ENGINE on 1m (trend pullback, the only OOS-green setup).

Walk-forward backtests (45d, XM spreads deducted) showed the failed-auction
fade was negative-expectancy on every symbol, while the Zenith structure
pullback was green OOS on USOIL (PF 1.70) and GOLD (PF 1.21). This module
runs that exact setup at Hyper cadence:

  1. 5m structure bias — close vs SMA20 + 3-bar slope (both must agree).
  2. 1m displacement candle in the bias direction (range ≥ disp_atr × ATR14,
     close in the leading 25% of the bar).
  3. Entry on pullback: price within pull_max × ATR of the EMA8/EMA21 mid,
     still on the right side of EMA8 and the bar open.
  4. SL beyond the 10-bar swing (± 0.1 ATR pad); TP = tp_r × risk (1.5R).
  5. TP1 partial at +1R → book tp1_frac, stop to breakeven (worker manages).
  6. Hard XM cost gate: TP distance must clear cost_mult × round-trip spread.

No auction fade, no POC magnet — pure momentum continuation with partials.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "entry_tf": "1m",
    "bias_tf": "5m",
    "entry_bars": 120,
    "bias_bars": 80,
    "disp_atr": 0.65,        # displacement bar ≥ this × ATR14 (backtest-validated default)
    "pull_max_atr": 0.85,    # entry must sit within this × ATR of EMA mid
    "swing_lookback": 10,
    "sl_pad_atr": 0.10,
    "min_risk_atr": 0.15,    # too-tight structural stops get spread-eaten
    "tp_r": 1.5,             # full target (1.5R beat 2R OOS on every symbol)
    "tp1_r": 1.0,            # partial trigger
    "tp1_frac": 0.5,         # fraction booked at TP1 (then SL → breakeven)
    "cost_mult": 3.5,        # TP distance ≥ this × round-trip spread (XM gate)
    "min_range_bps": 1.0,    # dead-tape filter on 1m median range
}

# Typical XM round-trip cost proxy (bps of price)
_SPREAD_BPS: dict[str, float] = {
    "GOLD": 1.8,
    "SILVER": 3.5,
    "EURUSD": 1.0,
    "GBPUSD": 1.2,
    "USDJPY": 1.0,
    "USDCHF": 1.4,
    "AUDUSD": 1.2,
    "USOIL": 2.5,
    "UKOIL": 2.8,
    "US30": 1.5,
    "NAS100": 1.6,
    "US500": 1.5,
    "BTCUSD": 2.0,
    "ETHUSD": 3.5,
}

# Same-book heat: one directional risk per cluster
_CORR_GROUP: dict[str, str] = {
    "GOLD": "metals",
    "SILVER": "metals",
    "EURUSD": "usd_fx",
    "GBPUSD": "usd_fx",
    "USDJPY": "usd_fx",
    "USDCHF": "usd_fx",
    "AUDUSD": "usd_fx",
    "USOIL": "energy",
    "UKOIL": "energy",
    "US30": "index",
    "NAS100": "index",
    "US500": "index",
    "BTCUSD": "crypto",
    "ETHUSD": "crypto",
}

_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 2.0
_last_board: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()

# Composite punch order — CHF/JPY/NAS/UKOIL core, AUD smaller, BTC satellite
SYMBOL_PRIORITY = (
    "USDCHF",
    "USDJPY",
    "NAS100",
    "UKOIL",
    "AUDUSD",
    "BTCUSD",
    "USOIL",
    "EURUSD",
    "GOLD",
    "US30",
    "US500",
    "ETHUSD",
    "SILVER",
    "GBPUSD",
)


def punch_order(symbols: Sequence[str]) -> list[str]:
    rank = {s: i for i, s in enumerate(SYMBOL_PRIORITY)}
    return sorted([str(s).upper() for s in symbols], key=lambda s: rank.get(s, 100))


def corr_group(symbol: str) -> str:
    return _CORR_GROUP.get(str(symbol).upper(), str(symbol).upper())


def estimate_spread_price(symbol: str, price: float) -> float:
    bps = float(_SPREAD_BPS.get(str(symbol).upper(), 2.0))
    return abs(float(price)) * bps / 1e4


def score_rank(symbols: Sequence[str]) -> list[str]:
    """Punch highest edge-per-cost first; USOIL tie-break via priority."""
    with _LOCK:
        board = {k: dict(v) for k, v in _last_board.items()}
    pri = {s: i for i, s in enumerate(SYMBOL_PRIORITY)}

    def key(sym: str):
        row = board.get(str(sym).upper()) or {}
        ready = 1 if row.get("status") == "READY" else 0
        score = float(row.get("score") or 0.0)
        return (-ready, -score, pri.get(str(sym).upper(), 100))

    return sorted([str(s).upper() for s in symbols], key=key)


def blocks_correlated(symbol: str, signal: str, open_metas: Sequence[dict[str, Any]]) -> Optional[str]:
    """One risk cluster, one direction."""
    g = corr_group(symbol)
    sig = str(signal).upper()
    for m in open_metas:
        osym = str(m.get("symbol") or "").upper()
        osig = str(m.get("signal") or "").upper()
        if not osym or not osig:
            continue
        if corr_group(osym) == g and osig == sig:
            return f"corr heat {g}: already {osig} {osym}"
    return None


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
        logger.debug("hyper_momentum MetaAPI %s %s: %s", symbol, tf, exc)
    try:
        from data.market_data import fetch_ohlcv

        df = fetch_ohlcv(tf, bars, symbol=symbol)
        _cache[key] = (now, df)
        return df
    except Exception as exc:
        logger.debug("hyper_momentum fetch %s %s: %s", symbol, tf, exc)
        return None


def _record(symbol: str, **fields: Any) -> None:
    row = {"symbol": str(symbol).upper(), "updated_at": time.time(), **fields}
    with _LOCK:
        prev = dict(_last_board.get(str(symbol).upper()) or {})
        prev.update(row)
        _last_board[str(symbol).upper()] = prev


def board_snapshot() -> dict[str, Any]:
    with _LOCK:
        rows = [dict(v) for v in _last_board.values()]
    rows.sort(key=lambda r: str(r.get("symbol") or ""))
    return {"symbols": rows}


def _atr(df: pd.DataFrame, n: int = 14) -> Optional[float]:
    if df is None or len(df) < n + 2:
        return None
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    v = float(tr.rolling(n).mean().iloc[-1])
    return v if v > 0 else None


def evaluate_momentum(
    symbol: str,
    *,
    price: Optional[float] = None,
    settings: Optional[dict[str, Any]] = None,
    record_board: bool = True,
) -> Optional[dict[str, Any]]:
    """Return a pullback-continuation decision, else None (board records why)."""
    cfg = dict(DEFAULTS)
    if settings:
        for k, v in settings.items():
            if v is not None and k in cfg:
                cfg[k] = v
            # aliases from hyper_scalp state
            if k == "ema_tf" and v is not None:
                cfg["entry_tf"] = v
            if k == "bias_tf" and v is not None:
                cfg["bias_tf"] = v

    sym = str(symbol).upper()

    def board(**kw: Any) -> None:
        if record_board:
            _record(sym, **kw)

    df1 = _fetch_ohlcv(sym, str(cfg["entry_tf"]), int(cfg["entry_bars"]))
    df5 = _fetch_ohlcv(sym, str(cfg["bias_tf"]), int(cfg["bias_bars"]))
    if df1 is None or df5 is None or len(df1) < 40 or len(df5) < 26:
        board(status="WARMING", signal=None, skip="candles")
        return None

    last = float(price) if price is not None else float(df1["close"].iloc[-1])

    # Dead-tape filter — 1m median range must be worth trading vs costs
    ranges_1m = (df1["high"].astype(float) - df1["low"].astype(float)).tail(20)
    med_bps = float(ranges_1m.median()) / max(abs(last), 1e-9) * 1e4
    if med_bps < float(cfg["min_range_bps"]):
        board(status="DEAD_TAPE", signal=None, skip=f"1m med range {med_bps:.2f}bps")
        return None

    # 5m structure bias: close vs SMA20 + slope over 3 bars must agree
    c5 = df5["close"].astype(float)
    sma20 = float(c5.rolling(20).mean().iloc[-1])
    slope = float(c5.iloc[-1] - c5.iloc[-4])
    if c5.iloc[-1] > sma20 and slope > 0:
        bias = "BUY"
    elif c5.iloc[-1] < sma20 and slope < 0:
        bias = "SELL"
    else:
        board(status="NO_BIAS", signal=None, skip="5m structure flat/conflicted", bias=None)
        return None

    atr = _atr(df1, 14)
    if not atr:
        board(status="WARMING", signal=bias, skip="ATR warming", bias=bias)
        return None

    # Displacement: last closed 1m bar must be an impulse in the bias direction
    o_bar = float(df1["open"].iloc[-1])
    h_bar = float(df1["high"].iloc[-1])
    l_bar = float(df1["low"].iloc[-1])
    c_bar = float(df1["close"].iloc[-1])
    rng = h_bar - l_bar
    disp_need = float(cfg["disp_atr"]) * atr
    disp_ratio = rng / max(atr, 1e-12)
    if rng < disp_need:
        board(
            status="NO_DISPLACEMENT",
            signal=bias,
            skip=f"1m range {disp_ratio:.2f}×ATR < {cfg['disp_atr']}×",
            bias=bias,
            disp_atr=round(disp_ratio, 2),
        )
        return None
    bull_impulse = c_bar > o_bar and c_bar >= h_bar - 0.25 * rng
    bear_impulse = c_bar < o_bar and c_bar <= l_bar + 0.25 * rng
    if bias == "BUY" and not bull_impulse:
        board(status="NO_DISPLACEMENT", signal=bias, skip="bar not closing strong with bias",
              bias=bias, disp_atr=round(disp_ratio, 2))
        return None
    if bias == "SELL" and not bear_impulse:
        board(status="NO_DISPLACEMENT", signal=bias, skip="bar not closing strong with bias",
              bias=bias, disp_atr=round(disp_ratio, 2))
        return None

    # Pullback location: near the EMA8/21 mid, right side of EMA8 and bar open
    c1 = df1["close"].astype(float)
    e8 = float(c1.ewm(span=8, adjust=False).mean().iloc[-1])
    e21 = float(c1.ewm(span=21, adjust=False).mean().iloc[-1])
    mid = (e8 + e21) / 2.0
    pull_atr = abs(last - mid) / max(atr, 1e-12)
    if pull_atr > float(cfg["pull_max_atr"]):
        board(
            status="NO_PULLBACK",
            signal=bias,
            skip=f"price {pull_atr:.2f}×ATR from EMA mid (max {cfg['pull_max_atr']})",
            bias=bias,
            pull_atr=round(pull_atr, 2),
        )
        return None
    if bias == "BUY" and not (last >= e8 and last >= o_bar):
        board(status="NO_PULLBACK", signal=bias, skip="below EMA8/open — not holding bid",
              bias=bias, pull_atr=round(pull_atr, 2))
        return None
    if bias == "SELL" and not (last <= e8 and last <= o_bar):
        board(status="NO_PULLBACK", signal=bias, skip="above EMA8/open — not holding offer",
              bias=bias, pull_atr=round(pull_atr, 2))
        return None

    # Structure stop beyond the swing, padded
    look = max(3, int(cfg["swing_lookback"]))
    pad = float(cfg["sl_pad_atr"]) * atr
    if bias == "BUY":
        sl = float(df1["low"].tail(look).min()) - pad
        risk = last - sl
    else:
        sl = float(df1["high"].tail(look).max()) + pad
        risk = sl - last
    if risk < float(cfg["min_risk_atr"]) * atr:
        board(status="TIGHT_STOP", signal=bias, skip=f"risk {risk / max(atr, 1e-12):.2f}×ATR too tight",
              bias=bias)
        return None

    tp_r = float(cfg["tp_r"])
    tp1_r = float(cfg["tp1_r"])
    if bias == "BUY":
        tp = last + tp_r * risk
        tp1 = last + tp1_r * risk
    else:
        tp = last - tp_r * risk
        tp1 = last - tp1_r * risk

    # XM cost gate — the tweak that keeps the fast book solvent
    spread = estimate_spread_price(sym, last)
    cost_need = float(cfg["cost_mult"]) * spread
    tp_dist = abs(tp - last)
    if tp_dist < cost_need:
        board(
            status="COST_KILL",
            signal=bias,
            skip=f"TP {tp_dist:.5g} < {cfg['cost_mult']}×spread {cost_need:.5g}",
            bias=bias,
            edge_vs_cost=round(tp_dist / max(spread, 1e-12), 2),
        )
        return None

    edge_vs_cost = tp_dist / max(spread, 1e-12)
    # Score: edge-per-cost × displacement quality (rank punches across the book)
    score = round(min(edge_vs_cost / 3.0, 3.0) * min(disp_ratio, 2.5), 4)
    if sym == "USOIL":
        score = round(score * 1.15, 4)  # oil carries the book (OOS PF 1.70)

    out = {
        "signal": bias,
        "mode": "zenith_momentum",
        "bias": bias,
        "sl": float(sl),
        "tp": float(tp),
        "tp1": float(tp1),
        "tp1_frac": float(cfg["tp1_frac"]),
        "entry": last,
        "tf": cfg["entry_tf"],
        "bias_tf": cfg["bias_tf"],
        "rr": round(tp_r, 2),
        "risk": float(risk),
        "atr": float(atr),
        "disp_atr": round(disp_ratio, 2),
        "pull_atr": round(pull_atr, 2),
        "spread": round(spread, 8),
        "edge_vs_cost": round(edge_vs_cost, 2),
        "score": score,
        "med_range_bps": round(med_bps, 2),
    }
    board(
        status="READY",
        skip=None,
        signal=bias,
        bias=bias,
        disp_atr=out["disp_atr"],
        pull_atr=out["pull_atr"],
        rr=out["rr"],
        score=score,
        edge_vs_cost=out["edge_vs_cost"],
    )
    return out


def probe_momentum_board(
    symbol: str,
    *,
    price: Optional[float] = None,
    settings: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    return evaluate_momentum(symbol, price=price, settings=settings, record_board=True)
