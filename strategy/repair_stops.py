"""Repair truncated / missing SL-TP on open XM positions."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _digits_for(symbol: str) -> int:
    from strategy.instrument_meta import price_digits

    return price_digits(symbol)


def _looks_truncated(symbol: str, level: Optional[float], entry: float) -> bool:
    """True when stop was rounded like gold (2dp) on an FX pair, or too tight."""
    if level is None:
        return True
    s = str(symbol or "").upper()
    digits = _digits_for(s)
    dist = abs(float(level) - float(entry))
    # Too tight for FX (< ~5 pips) — force rewrite
    if digits >= 5 and dist < 0.0005:
        return True
    if digits == 3 and dist < 0.04:
        return True
    text = f"{float(level):.10f}".rstrip("0")
    if "." in text:
        frac = text.split(".")[1]
        if digits >= 5 and len(frac) <= 2:
            return True
    return False


def compute_stops(
    *,
    symbol: str,
    direction: str,
    entry: float,
    atr: Optional[float] = None,
) -> tuple[float, float]:
    """ATR-based SL / 2R TP with symbol digits + floor distances."""
    s = str(symbol).upper()
    digits = _digits_for(s)
    atr_v = float(atr) if atr and atr > 0 else 0.0

    from strategy.instrument_meta import min_stop_distance

    floor = min_stop_distance(s)

    risk = max(atr_v * 1.25, floor)
    sl_dist = risk
    tp_dist = risk * 2.0
    rnd = lambda x: round(float(x), digits)  # noqa: E731
    if direction == "BUY":
        return rnd(entry - sl_dist), rnd(entry + tp_dist)
    return rnd(entry + sl_dist), rnd(entry - tp_dist)


def repair_open_stops(broker_client: Any) -> dict[str, Any]:
    """
    Scan open XM positions and rewrite SL/TP when missing or truncated.
    Returns counts of fixed / ok / failed.
    """
    from data.mt5_client import mt5_client

    client = broker_client or mt5_client
    fixed, ok, failed = [], [], []

    try:
        positions = client.get_open_positions()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    atr_cache: dict[str, float] = {}
    for pos in positions:
        sym = str(pos.get("symbol") or "")
        pid = str(pos.get("id") or pos.get("positionId") or "")
        entry = float(pos.get("openPrice") or 0)
        sl = pos.get("stopLoss")
        tp = pos.get("takeProfit")
        ptype = str(pos.get("type") or "").upper()
        direction = "BUY" if "BUY" in ptype else "SELL"

        needs = (sl is None or tp is None) or _looks_truncated(sym, sl, entry) or _looks_truncated(
            sym, tp, entry
        )
        if not needs:
            ok.append(pid)
            continue

        if sym not in atr_cache:
            try:
                df = client.fetch_candles("5m", 40, symbol=sym)
                atr_cache[sym] = float((df["high"] - df["low"]).tail(14).mean())
            except Exception:
                atr_cache[sym] = 0.0

        new_sl, new_tp = compute_stops(
            symbol=sym, direction=direction, entry=entry, atr=atr_cache.get(sym) or None
        )
        try:
            client.modify_position(pid, stop_loss=new_sl, take_profit=new_tp)
            fixed.append(
                {
                    "id": pid,
                    "symbol": sym,
                    "sl": new_sl,
                    "tp": new_tp,
                    "old_sl": sl,
                    "old_tp": tp,
                }
            )
            logger.info(
                "Repaired XM stops %s pos=%s SL %s→%s TP %s→%s",
                sym,
                pid,
                sl,
                new_sl,
                tp,
                new_tp,
            )
        except Exception as exc:
            failed.append({"id": pid, "symbol": sym, "error": str(exc)})
            logger.warning("Stop repair failed %s %s: %s", sym, pid, exc)

    return {
        "ok": True,
        "fixed": fixed,
        "already_ok": len(ok),
        "failed": failed,
        "fixed_count": len(fixed),
    }
