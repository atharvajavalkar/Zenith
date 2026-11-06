"""Timeframe-hop trade stacking.

When the parent setup fires (same zone/SMC/AVP logic), we hop
15m → 5m → 3m → 1m and open multiple legs with SL/TP scaled to each TF's ATR.

Target: fill up to MAX_TRADES_PER_BUCKET (~10) per 5m bucket from one setup.
"""

from __future__ import annotations

from typing import Any, Optional

from config import MINUTE_SL_ATR, MAX_TRADES_PER_BUCKET
from strategy.monte_carlo import choose_minute_rr

# Hop order: higher TF first (structure), then finer execution TFs
TF_HOP_ORDER = ("15m", "5m", "3m", "1m")

# SL distance as ATR multiples per timeframe (wider on higher TF)
TF_SL_ATR: dict[str, float] = {
    "15m": 1.45,
    "5m": 1.15,
    "3m": 0.95,
    "1m": 0.80,
}

# How many legs to allocate per TF inside one 5m bucket (sums to 10)
TF_LEG_QUOTA: dict[str, int] = {
    "15m": 2,
    "5m": 3,
    "3m": 3,
    "1m": 2,
}


def _atr_map_from_signal(sig: dict) -> dict[str, float]:
    """Pull per-TF ATR from signal payload with sensible fallbacks."""
    base = float(sig.get("ATR") or 1.0)
    return {
        "15m": float(sig.get("ATR_15M") or base * 1.6),
        "5m": float(sig.get("ATR_5M") or base),
        "3m": float(sig.get("ATR_3M") or base * 0.85),
        "1m": float(sig.get("ATR_1M") or base * 0.65),
    }


def build_tf_hop_legs(
    sig: dict,
    live_price: float,
    max_legs: Optional[int] = None,
    already_opened: int = 0,
) -> list[dict]:
    """
    Expand a BUY/SELL parent setup into TF-scaled stacked legs.

    Same direction / setup as parent; each leg gets its own SL/TP from that TF's ATR.
    """
    direction = sig.get("SIGNAL")
    if direction not in ("BUY", "SELL"):
        return []

    cap = int(max_legs if max_legs is not None else MAX_TRADES_PER_BUCKET)
    remaining = max(0, cap - int(already_opened))
    if remaining <= 0:
        return []

    atr_map = _atr_map_from_signal(sig)
    state = sig.get("STATE") or "CHOP"
    parent_entry = float(sig.get("ENTRY") or live_price)
    parent_sl = float(sig.get("SL") or live_price)
    parent_risk = abs(parent_entry - parent_sl)

    legs: list[dict] = []
    for tf in TF_HOP_ORDER:
        quota = int(TF_LEG_QUOTA.get(tf, 1))
        atr = max(float(atr_map.get(tf) or 1.0), 0.05)
        sl_mult = float(TF_SL_ATR.get(tf, MINUTE_SL_ATR))

        for k in range(quota):
            if len(legs) >= remaining:
                break

            # Micro scale-in: stagger entry slightly by TF ATR so legs aren't identical fills
            stagger = atr * 0.04 * k
            if direction == "BUY":
                entry = float(live_price) - stagger
                # Prefer TF ATR stop; never tighter than ~60% of parent risk
                risk = max(atr * sl_mult, parent_risk * 0.60)
                sl = entry - risk
            else:
                entry = float(live_price) + stagger
                risk = max(atr * sl_mult, parent_risk * 0.60)
                sl = entry + risk

            best = choose_minute_rr(direction, entry, sl, state, atr, zones=None, symbol=sig.get("SYMBOL") or sig.get("INSTRUMENT"))
            if best is None:
                continue

            out = dict(sig)
            out["ENTRY"] = round(entry, 2)
            out["SL"] = round(sl, 2)
            out["TP"] = best["tp"]
            out["TP1"] = best.get("tp1")
            out["TP2"] = best.get("tp2")
            out["TP3"] = best.get("tp3")
            rr_int = best.get("rr_int", int(best["rr"]))
            out["RR"] = f"1:{rr_int}"
            out["SUCCESS_PERCENT"] = best["mc_prob"]
            out["MARKOV_MC_PROB"] = best.get("mc_prob")
            out["LIVE_PRICE"] = round(float(live_price), 2)
            out["LOG_PRICE_1M"] = round(float(live_price), 2)
            out["EXEC_TF"] = tf
            out["TF_HOP"] = True
            out["TF_SL_ATR"] = sl_mult
            out["HOP_INDEX"] = len(legs) + 1
            out["ZONE_TFS"] = f"{tf.upper()} hop"
            out["REASON"] = (
                f"TF-hop {tf} leg {k + 1}/{quota} — SL {sl_mult:.2f}×ATR_{tf}, "
                f"RR 1:{rr_int}, same {direction} setup"
            )
            legs.append(out)

        if len(legs) >= remaining:
            break

    return legs[:remaining]
