"""Hyper — LIQUIDATION MAGNET (failed auction / trapped inventory).

Desk idea:
  1. 5m value (VAH/VAL/POC) defines fair price.
  2. Price breaks outside value → crowd piles in.
  3. 1m close *rejects* (reclaim back through the broken edge).
  4. Poor-man's tape (up/down volume) flips *against* the breakout.
  5. Fade the trapped side → TP at POC, SL beyond the failed extreme.
  6. No ATR. No EMA ribbon.

Separate from Zenith physics and the old EMA / flow-continuation Hyper.
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
    "value_tf": "5m",
    "entry_bars": 80,
    "value_bars": 60,        # ~5h of 5m for rolling value
    "break_lookback": 8,     # recent bars that may have pierced value
    "reject_bars": 3,        # 1m bars for reclaim confirmation
    "imb_lookback": 12,
    "min_imbalance": 0.12,   # tape must lean against the breakout
    "min_break_bps": 2.0,    # floor in bps
    "min_break_z": 0.85,     # break must clear this many σ of recent 5m range
    "swing_pad_bps": 1.5,    # SL pad beyond failed extreme
    "min_rr": 1.20,
    "cost_mult": 2.5,        # reward must be ≥ this × estimated round-trip spread
    "min_range_bps": 1.2,    # skip dead 1m tape (median range)
    "imb_accel": 0.04,       # tape must be accelerating against the break
    "n_bins": 36,
}

# Typical XM round-trip cost proxy (bps of price) — numbers desk, not chart art
_SPREAD_BPS: dict[str, float] = {
    "GOLD": 1.8,
    "SILVER": 3.5,
    "EURUSD": 1.0,
    "GBPUSD": 1.2,
    "USDJPY": 1.0,
    "USOIL": 2.5,
    "US30": 1.5,
    "BTCUSD": 2.0,
}

# Same-book heat: one directional risk per cluster
_CORR_GROUP: dict[str, str] = {
    "GOLD": "metals",
    "SILVER": "metals",
    "EURUSD": "usd_fx",
    "GBPUSD": "usd_fx",
    "USDJPY": "usd_fx",
    "USOIL": "energy",
    "US30": "index",
    "BTCUSD": "crypto",
}

_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 2.0
_last_board: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()

# Punch priority — GOLD leads the book
SYMBOL_PRIORITY = (
    "GOLD",
    "SILVER",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USOIL",
    "US30",
    "BTCUSD",
)


def punch_order(symbols: Sequence[str]) -> list[str]:
    """Static fallback order — prefer score_rank when board has READY rows."""
    rank = {s: i for i, s in enumerate(SYMBOL_PRIORITY)}
    return sorted(
        [str(s).upper() for s in symbols],
        key=lambda s: rank.get(s, 100),
    )


def corr_group(symbol: str) -> str:
    return _CORR_GROUP.get(str(symbol).upper(), str(symbol).upper())


def estimate_spread_price(symbol: str, price: float) -> float:
    bps = float(_SPREAD_BPS.get(str(symbol).upper(), 2.0))
    return abs(float(price)) * bps / 1e4


def score_rank(symbols: Sequence[str]) -> list[str]:
    """Punch highest quant score first; GOLD tie-break via priority."""
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
    """Numbers desk: one risk cluster, one direction."""
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
        logger.debug("failed_auction MetaAPI %s %s: %s", symbol, tf, exc)
    try:
        from data.market_data import fetch_ohlcv

        df = fetch_ohlcv(tf, bars, symbol=symbol)
        _cache[key] = (now, df)
        return df
    except Exception as exc:
        logger.debug("failed_auction fetch %s %s: %s", symbol, tf, exc)
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


def _build_value(df_5m: pd.DataFrame, price: float, symbol: str, n_bins: int) -> Optional[dict[str, Any]]:
    if df_5m is None or len(df_5m) < 24:
        return None
    from analysis.volume_profile import _profile_from_window

    rng = float((df_5m["high"] - df_5m["low"]).tail(20).mean() or abs(price) * 0.0005)
    atr = max(rng, abs(price) * 1e-6)
    return _profile_from_window(
        df_5m.tail(min(len(df_5m), 80)).copy(),
        price=float(price),
        atr=atr,
        n_bins=int(n_bins),
        value_area_pct=0.70,
        symbol=symbol,
        source="hyper_failed_auction",
        session_name="5m value",
    )


def evaluate_failed_auction(
    symbol: str,
    *,
    price: Optional[float] = None,
    settings: Optional[dict[str, Any]] = None,
    record_board: bool = True,
) -> Optional[dict[str, Any]]:
    """Return a fade decision when a value break fails, else None."""
    cfg = dict(DEFAULTS)
    if settings:
        for k, v in settings.items():
            if v is not None and k in cfg:
                cfg[k] = v
            # accept aliases from hyper_scalp state
            if k == "ema_tf" and v is not None:
                cfg["entry_tf"] = v
            if k == "bias_tf" and v is not None:
                cfg["value_tf"] = v

    sym = str(symbol).upper()

    def board(**kw: Any) -> None:
        if record_board:
            _record(sym, **kw)

    df5 = _fetch_ohlcv(sym, str(cfg["value_tf"]), int(cfg["value_bars"]))
    df1 = _fetch_ohlcv(sym, str(cfg["entry_tf"]), int(cfg["entry_bars"]))
    if df5 is None or df1 is None:
        board(status="WARMING", signal=None, skip="candles")
        return None

    last = float(price) if price is not None else float(df1["close"].iloc[-1])

    # Dead-tape filter — no edge when 1m median range is tiny vs costs
    ranges_1m = (df1["high"].astype(float) - df1["low"].astype(float)).tail(20)
    med_range = float(ranges_1m.median()) if len(ranges_1m) else 0.0
    med_bps = med_range / max(abs(last), 1e-9) * 1e4
    if med_bps < float(cfg["min_range_bps"]):
        board(status="DEAD_TAPE", signal=None, skip=f"1m med range {med_bps:.2f}bps")
        return None

    value = _build_value(df5, last, sym, int(cfg["n_bins"]))
    if not value or not value.get("available"):
        board(status="NO_VALUE", signal=None, skip="value profile warming")
        return None

    vah = float(value["vah"])
    val = float(value["val"])
    poc = float(value["poc"])
    if vah <= val:
        board(status="FLAT_VALUE", signal=None, skip="vah<=val", vah=vah, val=val, poc=poc)
        return None

    lb = max(3, int(cfg["break_lookback"]))
    win5 = df5.tail(lb)
    hi5 = float(win5["high"].max())
    lo5 = float(win5["low"].min())

    # σ of recent 5m bar ranges — break must be statistically meaningful
    rng5 = (df5["high"].astype(float) - df5["low"].astype(float)).tail(30)
    sigma = float(rng5.std()) if len(rng5) >= 8 else float(rng5.mean() or abs(last) * 0.0004)
    sigma = max(sigma, abs(last) * 1e-6)
    up_ext = max(0.0, hi5 - vah)
    dn_ext = max(0.0, val - lo5)
    up_z = up_ext / sigma
    dn_z = dn_ext / sigma
    up_bps = up_ext / max(abs(last), 1e-9) * 1e4
    dn_bps = dn_ext / max(abs(last), 1e-9) * 1e4
    min_z = float(cfg["min_break_z"])
    min_bps = float(cfg["min_break_bps"])
    up_break = up_z >= min_z and up_bps >= min_bps
    dn_break = dn_z >= min_z and dn_bps >= min_bps

    if up_break and dn_break:
        if up_z >= dn_z:
            dn_break = False
        else:
            up_break = False

    if not up_break and not dn_break:
        board(
            status="NO_BREAK",
            signal=None,
            skip=f"need z≥{min_z} & bps≥{min_bps} outside value",
            vah=vah,
            val=val,
            poc=poc,
            break_z=round(max(up_z, dn_z), 2),
            location=value.get("location"),
        )
        return None

    # Failed upside = fade longs → SELL; failed downside = fade shorts → BUY
    fade_sell = bool(up_break)
    fade_buy = bool(dn_break)
    break_z = float(up_z if fade_sell else dn_z)

    n_rej = max(2, int(cfg["reject_bars"]))
    recent1 = df1.tail(n_rej)
    c_last = float(recent1["close"].iloc[-1])
    o_last = float(recent1["open"].iloc[-1])
    h_win = float(recent1["high"].max())
    l_win = float(recent1["low"].min())

    rejected = False
    signal: Optional[str] = None
    failed_extreme: Optional[float] = None

    if fade_sell:
        # Need pierce above VAH then close back below VAH (reclaim)
        pierced = h_win > vah
        reclaimed = c_last < vah
        if len(recent1) >= 2:
            bear_close = c_last <= o_last or c_last < float(recent1["close"].iloc[-2])
        else:
            bear_close = c_last <= o_last
        if pierced and reclaimed and bear_close:
            rejected = True
            signal = "SELL"
            failed_extreme = h_win
        else:
            board(
                status="NO_REJECT",
                signal="SELL",
                skip="upside break not reclaimed on 1m",
                vah=vah,
                val=val,
                poc=poc,
                break_side="ABOVE_VAH",
            )
            return None
    elif fade_buy:
        pierced = l_win < val
        reclaimed = c_last > val
        if len(recent1) >= 2:
            bull_close = c_last >= o_last or c_last > float(recent1["close"].iloc[-2])
        else:
            bull_close = c_last >= o_last
        if pierced and reclaimed and bull_close:
            rejected = True
            signal = "BUY"
            failed_extreme = l_win
        else:
            board(
                status="NO_REJECT",
                signal="BUY",
                skip="downside break not reclaimed on 1m",
                vah=vah,
                val=val,
                poc=poc,
                break_side="BELOW_VAL",
            )
            return None

    if not rejected or signal is None or failed_extreme is None:
        board(status="NO_REJECT", signal=None, skip="no reclaim", vah=vah, val=val, poc=poc)
        return None

    # Tape: level + acceleration against the breakout (trapped inventory)
    from strategy.hyper_flow import rolling_imbalance

    lb_imb = int(cfg["imb_lookback"])
    imb = rolling_imbalance(df1, lb_imb)
    imb_fast = rolling_imbalance(df1, max(4, lb_imb // 2))
    imb_prev = rolling_imbalance(df1.iloc[:-max(4, lb_imb // 2)], lb_imb) if len(df1) > lb_imb + 4 else None
    if imb is None:
        board(status="NO_TAPE", signal=signal, skip="no imbalance", vah=vah, val=val, poc=poc)
        return None
    min_imb = float(cfg["min_imbalance"])
    accel = float(cfg["imb_accel"])
    if signal == "SELL":
        if imb > -min_imb:
            board(
                status="TAPE_MISS",
                signal=signal,
                skip=f"need sell heat ≤{-min_imb:.2f}, got {imb:.3f}",
                imbalance=round(float(imb), 4),
                vah=vah,
                val=val,
                poc=poc,
                break_z=round(break_z, 2),
            )
            return None
        if imb_prev is not None and imb_fast is not None:
            if float(imb_fast) > float(imb_prev) - accel:
                board(
                    status="TAPE_FLAT",
                    signal=signal,
                    skip=f"sell heat not accelerating ({imb_fast:.3f} vs {imb_prev:.3f})",
                    imbalance=round(float(imb), 4),
                    vah=vah,
                    poc=poc,
                    break_z=round(break_z, 2),
                )
                return None
    else:
        if imb < min_imb:
            board(
                status="TAPE_MISS",
                signal=signal,
                skip=f"need buy heat ≥{min_imb:.2f}, got {imb:.3f}",
                imbalance=round(float(imb), 4),
                vah=vah,
                val=val,
                poc=poc,
                break_z=round(break_z, 2),
            )
            return None
        if imb_prev is not None and imb_fast is not None:
            if float(imb_fast) < float(imb_prev) + accel:
                board(
                    status="TAPE_FLAT",
                    signal=signal,
                    skip=f"buy heat not accelerating ({imb_fast:.3f} vs {imb_prev:.3f})",
                    imbalance=round(float(imb), 4),
                    vah=vah,
                    poc=poc,
                    break_z=round(break_z, 2),
                )
                return None

    pad = max(abs(last) * float(cfg["swing_pad_bps"]) / 1e4, abs(last) * 1e-6)
    if signal == "SELL":
        sl = float(failed_extreme) + pad
        tp = float(poc)
        risk = sl - last
        reward = last - tp
        if risk <= 0 or tp >= last:
            board(status="BAD_LEVELS", signal=signal, skip="sell sl/tp invalid", vah=vah, poc=poc)
            return None
        min_rr = float(cfg["min_rr"])
        if reward / risk < min_rr:
            need = last - min_rr * risk
            tp = min(tp, need, float(val))
            reward = last - tp
            if reward / risk < min_rr * 0.85:
                board(
                    status="BAD_RR",
                    signal=signal,
                    skip=f"rr {reward/risk:.2f} < {min_rr}",
                    imbalance=round(float(imb), 4),
                    vah=vah,
                    val=val,
                    poc=poc,
                    break_z=round(break_z, 2),
                )
                return None
    else:
        sl = float(failed_extreme) - pad
        tp = float(poc)
        risk = last - sl
        reward = tp - last
        if risk <= 0 or tp <= last:
            board(status="BAD_LEVELS", signal=signal, skip="buy sl/tp invalid", val=val, poc=poc)
            return None
        min_rr = float(cfg["min_rr"])
        if reward / risk < min_rr:
            need = last + min_rr * risk
            tp = max(tp, need, float(vah))
            reward = tp - last
            if reward / risk < min_rr * 0.85:
                board(
                    status="BAD_RR",
                    signal=signal,
                    skip=f"rr {reward/risk:.2f} < {min_rr}",
                    imbalance=round(float(imb), 4),
                    vah=vah,
                    val=val,
                    poc=poc,
                    break_z=round(break_z, 2),
                )
                return None

    # Cost gate: edge to POC must clear estimated round-trip spread
    spread = estimate_spread_price(sym, last)
    cost_need = float(cfg["cost_mult"]) * spread
    if reward < cost_need:
        board(
            status="COST_KILL",
            signal=signal,
            skip=f"reward {reward:.5g} < {cfg['cost_mult']}×spread {cost_need:.5g}",
            imbalance=round(float(imb), 4),
            vah=vah,
            val=val,
            poc=poc,
            break_z=round(break_z, 2),
        )
        return None

    rr = abs(tp - last) / max(abs(last - sl), 1e-12)
    edge_vs_cost = reward / max(spread, 1e-12)
    # Quant score — higher = punch first
    score = round(rr * abs(float(imb)) * min(break_z, 3.0) * min(edge_vs_cost / 3.0, 2.0), 4)
    # Mild GOLD book preference in the score, not a hard rule
    if sym == "GOLD":
        score = round(score * 1.15, 4)

    out = {
        "signal": signal,
        "mode": "failed_auction",
        "side": "fade_longs" if signal == "SELL" else "fade_shorts",
        "imbalance": round(float(imb), 4),
        "imb_fast": round(float(imb_fast), 4) if imb_fast is not None else None,
        "vah": round(vah, 8),
        "val": round(val, 8),
        "poc": round(poc, 8),
        "failed_extreme": round(float(failed_extreme), 8),
        "break_side": "ABOVE_VAH" if signal == "SELL" else "BELOW_VAL",
        "break_z": round(break_z, 3),
        "vp_location": value.get("location"),
        "sl": float(sl),
        "tp": float(tp),
        "entry": last,
        "tf": cfg["entry_tf"],
        "value_tf": cfg["value_tf"],
        "rr": round(rr, 2),
        "spread": round(spread, 8),
        "edge_vs_cost": round(edge_vs_cost, 2),
        "score": score,
        "med_range_bps": round(med_bps, 2),
    }
    board(
        status="READY",
        skip=None,
        signal=signal,
        imbalance=out["imbalance"],
        vah=out["vah"],
        val=out["val"],
        poc=out["poc"],
        break_side=out["break_side"],
        break_z=out["break_z"],
        rr=out["rr"],
        score=score,
        edge_vs_cost=out["edge_vs_cost"],
    )
    return out


def probe_auction_board(
    symbol: str,
    *,
    price: Optional[float] = None,
    settings: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    return evaluate_failed_auction(symbol, price=price, settings=settings, record_board=True)
