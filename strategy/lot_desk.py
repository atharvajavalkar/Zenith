"""Profit-target lot desk for the universe — API-facing snapshot."""

from __future__ import annotations

from typing import Any, Optional

from config import INSTRUMENT_UNIVERSE, TARGET_PROFIT_USD
from strategy.dynamic_lots import (
    fetch_account_capital,
    sizing_snapshot,
)


def lot_desk_payload(capital: Optional[float] = None) -> dict[str, Any]:
    """Live desk lots. Optional capital overrides XM balance (backtest preview)."""
    from config import SYMBOL_ALLOCATION, SYMBOL_RR, symbol_rr

    cap = float(capital) if capital is not None and float(capital) > 0 else fetch_account_capital()
    rows = []
    for sym in INSTRUMENT_UNIVERSE:
        snap = sizing_snapshot(sym, capital=cap)
        target_rr = float(symbol_rr(sym, default=float(snap.get("rr") or 2)))
        rows.append(
            {
                "symbol": sym,
                "lot_size": snap["lot_size"],
                "asset_class": snap["asset_class"],
                "expected_profit_usd": snap.get("expected_profit_usd"),
                "expected_loss_usd": snap.get("expected_loss_usd"),
                "rr": target_rr,
                "tp1_frac": snap["tp1_frac"],
                "trail_lock": round(float(snap["trail_lock"]), 2),
                "vol_mult": snap["vol_mult"],
                "allocation_mult": snap.get("allocation_mult", 1.0),
                "swing_lot_mult": snap.get("swing_lot_mult", 1.0),
                "risk_usd_cap": snap.get("risk_usd_cap"),
                "note": _class_note(snap["asset_class"], target_rr),
            }
        )
    return {
        "capital": round(cap, 2),
        "dynamic_lots": True,
        "target_profit_usd": float(TARGET_PROFIT_USD),
        "mode": "target_profit",
        "symbol_rr": {k: float(v) for k, v in SYMBOL_RR.items()},
        "symbol_allocation": {k: float(v) for k, v in (SYMBOL_ALLOCATION or {}).items()},
        "aggression": (
            sizing_snapshot(rows[0]["symbol"], capital=cap).get("aggression") if rows else None
        ),
        "instruments": rows,
        "capital_source": "override" if capital is not None and float(capital) > 0 else "account",
    }


def _class_note(cls: str, rr: Any = None) -> str:
    rr_s = f"1:{rr:g}" if isinstance(rr, (int, float)) else "1:2"
    return {
        "gold": f"TP ≥ ${TARGET_PROFIT_USD:.0f} · SL ≈ profit/{rr_s}",
        "fx": f"TP ≥ ${TARGET_PROFIT_USD:.0f} · SL scales with {rr_s}",
        "crypto": f"TP ≥ ${TARGET_PROFIT_USD:.0f} · tighter risk vs vol",
        "silver": f"TP ≥ ${TARGET_PROFIT_USD:.0f} · SL by {rr_s}",
        "index": f"TP ≥ ${TARGET_PROFIT_USD:.0f} · SL by {rr_s}",
        "energy": f"TP ≥ ${TARGET_PROFIT_USD:.0f} · SL by {rr_s}",
    }.get(cls, f"Sized for ~${TARGET_PROFIT_USD:.0f} TP")
