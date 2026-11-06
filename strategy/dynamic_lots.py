"""Profit-target lot sizing + volatility-scaled partials / trail.

Primary rule: size so TP reward ≈ TARGET_PROFIT_USD ($400).
SL loss follows the setup RR (1:2 → ~$200 risk, 1:3 → ~$133).

Capital tables + max-risk % act as safety ceilings so protect mode / small
accounts cannot oversize into account ruin.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from config import TRADE_LOT_SIZE

logger = logging.getLogger(__name__)

# Soft capital ceilings (never size above these mid-band guides)
_FX_LOT_TABLE: list[tuple[float, float]] = [
    (50, 0.01),
    (100, 0.02),
    (500, 0.06),
    (1000, 0.10),
    (2500, 0.25),
    (5000, 0.60),
    (10000, 1.00),
    (25000, 2.25),
    (50000, 4.50),
    (100000, 9.00),
]

_GOLD_LOT_TABLE: list[tuple[float, float]] = [
    (50, 0.01),
    (100, 0.01),
    (500, 0.015),
    (1000, 0.015),
    (2500, 0.03),
    (5000, 0.075),
    (10000, 0.15),
    (25000, 0.375),
    (50000, 0.75),
    (100000, 1.50),
]

_VOL_MULT: dict[str, float] = {
    "fx": 0.20,
    "gold": 1.0,
    "crypto": 1.4,
    "silver": 1.15,
    "index": 0.45,
    "energy": 0.55,
}

# Typical SL/TP distances when no live setup is passed (desk preview)
_DEFAULT_RISK_REWARD: dict[str, tuple[float, float, float]] = {
    # risk_pts, reward_pts, rr
    "fx": (0.0015, 0.0030, 2.0),
    "gold": (8.0, 16.0, 2.0),
    "crypto": (250.0, 500.0, 2.0),
    "silver": (0.12, 0.24, 2.0),
    "index": (25.0, 50.0, 2.0),
    "energy": (0.35, 0.70, 2.0),
}

_capital_cache: dict[str, Any] = {"balance": None, "ts": 0.0}
_capital_lock = threading.Lock()
_CAPITAL_TTL = 20.0


def asset_class(symbol: Optional[str]) -> str:
    from strategy.instrument_meta import asset_class as _cls

    return _cls(symbol)

def _interp(table: list[tuple[float, float]], capital: float) -> float:
    if capital <= table[0][0]:
        return table[0][1]
    if capital >= table[-1][0]:
        c0, l0 = table[-2]
        c1, l1 = table[-1]
        slope = (l1 - l0) / max(c1 - c0, 1.0)
        return l1 + slope * (capital - c1)
    for i in range(1, len(table)):
        c0, l0 = table[i - 1]
        c1, l1 = table[i]
        if capital <= c1:
            t = (capital - c0) / max(c1 - c0, 1e-9)
            return l0 + t * (l1 - l0)
    return table[-1][1]


def _aggression() -> float:
    try:
        from analysis.instrument_rl import aggression_for

        # Global fallback when no symbol in context — dynamic_lots passes symbol separately
        return float(aggression_for(None))
    except Exception:
        pass
    try:
        from analysis.adaptive_guard import get_state

        st = get_state() or {}
        mode = str(st.get("mode") or "")
        if mode in ("protect", "bank", "daily"):
            return 0.15
        if mode in ("seek", "aggressive"):
            return 0.85
    except Exception:
        pass
    return 0.5


def allocation_mult(symbol: Optional[str]) -> float:
    """Desk allocation weight → sizing multiplier (1.0 = equal weight).

    SYMBOL_ALLOCATION holds risk fractions (e.g. USOIL:0.60, GOLD:0.40).
    Normalized against equal weight so a 60/40 two-symbol book sizes
    oil at 1.2× and gold at 0.8× of the base profit target.
    """
    try:
        from config import INSTRUMENT_UNIVERSE, SYMBOL_ALLOCATION

        sym = str(symbol or "").upper()
        weights = {k: float(v) for k, v in (SYMBOL_ALLOCATION or {}).items() if float(v) > 0}
        if not weights or sym not in weights:
            return 1.0
        total = sum(weights.values())
        n = max(len(INSTRUMENT_UNIVERSE), 1)
        equal = total / n
        if equal <= 0:
            return 1.0
        return max(0.25, min(2.0, weights[sym] / equal))
    except Exception:
        return 1.0


def swing_lot_mult(symbol: Optional[str]) -> float:
    """1h swing sleeve size bump (longer holds → larger book).

    Primary SYMBOL_SCAN_TFS == 1h → SWING_LOT_MULT (default 1.5).
    Faster TFs (5m/15m) stay at 1.0.
    """
    import os

    try:
        raw = os.getenv("SWING_LOT_MULT", "1.5")
        mult = float(raw) if raw not in ("", None) else 1.5
    except Exception:
        mult = 1.5
    mult = max(1.0, min(2.5, mult))
    if mult <= 1.0 + 1e-9:
        return 1.0
    try:
        from config import symbol_scan_tfs

        tfs = symbol_scan_tfs(symbol) or ()
        primary = str(tfs[0]).lower() if tfs else ""
    except Exception:
        primary = ""
    if primary == "1h":
        return mult
    return 1.0


def _target_profit_usd() -> float:
    try:
        import runtime_config

        return float(runtime_config.get("target_profit_usd", 400.0))
    except Exception:
        try:
            from config import TARGET_PROFIT_USD

            return float(TARGET_PROFIT_USD)
        except Exception:
            return 400.0


def _max_risk_pct() -> float:
    try:
        import runtime_config

        return float(runtime_config.get("max_risk_pct_per_trade", 0.04))
    except Exception:
        try:
            from config import MAX_RISK_PCT_PER_TRADE

            return float(MAX_RISK_PCT_PER_TRADE)
        except Exception:
            return 0.04


def fetch_account_capital(force: bool = False) -> float:
    """Live XM balance (demo/live) — cached ~20s."""
    now = time.time()
    with _capital_lock:
        if (
            not force
            and _capital_cache["balance"] is not None
            and now - float(_capital_cache["ts"]) < _CAPITAL_TTL
        ):
            return float(_capital_cache["balance"])

    balance: Optional[float] = None
    try:
        import runtime_config
        from data.broker import broker_client

        if runtime_config.is_broker_mode() and broker_client.configured:
            info = broker_client.account_information()
            raw = info.get("balance")
            if raw is not None:
                balance = float(raw)
    except Exception as exc:
        logger.debug("Capital fetch failed: %s", exc)

    if balance is None:
        try:
            import os

            balance = float(os.getenv("PAPER_CAPITAL", "10000"))
        except Exception:
            balance = 10000.0

    with _capital_lock:
        _capital_cache["balance"] = balance
        _capital_cache["ts"] = now
    return float(balance)


def _capital_ceiling(symbol: Optional[str], capital: float) -> float:
    """Hard ceiling from safe-lot tables (must still allow ≥$400 TP)."""
    cls = asset_class(symbol)
    if cls == "fx":
        mid = _interp(_FX_LOT_TABLE, capital)
        # Was max(mid*2.5, 3.0) — blew AUD/CHF to 3.0 lots on noise SLs
        return max(min(mid * 1.25, 0.50), 0.10)
    if cls == "gold":
        mid = _interp(_GOLD_LOT_TABLE, capital)
        return max(mid * 3.0, 0.75)
    gold_mid = _interp(_GOLD_LOT_TABLE, capital)
    vol = _VOL_MULT.get(cls, 1.0)
    mid = gold_mid * (1.0 / max(vol, 0.15))
    # Floors sized so default R/R tables can actually reach TARGET_PROFIT_USD
    floor = {
        "crypto": 1.5,
        "silver": 0.75,
        # Index: scale with capital (was flat 3.0 — starved NAS on ~$9k+ desks)
        "index": max(4.0, capital / 1200.0),
        "energy": 1.5,   # soft-capped — UKOIL/USOIL blow-up protection
    }.get(cls, 0.75)
    if cls == "energy":
        return max(min(mid * 3.0, floor * 2.0), floor)
    if cls == "index":
        # Allow capital expression; LOT_HARD_CAP_INDEX is the absolute brake
        return max(mid * 5.0, floor)
    return max(mid * 3.0, floor)


def _rr_from_levels(entry: float, sl: float, tp: float) -> float:
    risk = abs(float(entry) - float(sl))
    reward = abs(float(tp) - float(entry))
    if risk <= 0:
        return 2.0
    return max(0.5, reward / risk)


def _parse_rr(rr: Any) -> float:
    if rr is None:
        return 2.0
    try:
        if isinstance(rr, (int, float)):
            return max(0.5, float(rr))
        text = str(rr)
        if ":" in text:
            return max(0.5, float(text.split(":")[-1]))
        return max(0.5, float(text))
    except Exception:
        return 2.0


def lot_for_profit_target(
    symbol: Optional[str],
    *,
    entry: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    rr: Any = None,
    capital: Optional[float] = None,
    target_profit: Optional[float] = None,
) -> dict[str, Any]:
    """
    Size so |TP − entry| × units × lot ≥ target_profit ($400 minimum).
    Planned SL loss ≈ target / setup_RR.
    """
    from strategy.position_sizing import contract_units_per_lot, leg_pnl

    sym = str(symbol or "GOLD").upper()
    cls = asset_class(sym)
    cap = float(capital) if capital is not None else fetch_account_capital()
    target = float(target_profit) if target_profit is not None else _target_profit_usd()
    # Never shrink the configured $400 floor — RL only trims oversize, not the floor
    min_target = float(target)
    try:
        from analysis.instrument_rl import aggression_for, overlays_for

        agg = float(aggression_for(sym))
        risk_scale = float(overlays_for(sym).get("risk_scale") or 1.0)
    except Exception:
        agg = _aggression()
        risk_scale = 1.0
    # RL risk_scale: tight (<1) shrinks book; aggressive can bump up to 1.15×
    scale = float(risk_scale) * (0.7 + 0.5 * float(agg))
    scale = max(0.50, min(1.15, scale))
    target = min_target * scale
    # Desk allocation (e.g. GOLD 10% / BTC 12%) scales the per-symbol book
    alloc = allocation_mult(sym)
    target *= alloc
    min_target *= alloc
    # 1h swing pairs (GOLD/NAS/BTC/ETH) — longer holds, larger size
    swing = swing_lot_mult(sym)
    target *= swing
    min_target *= swing

    units = contract_units_per_lot(sym)
    risk_pts = reward_pts = None
    try:
        from config import symbol_rr

        rr_n = _parse_rr(rr) if rr is not None else float(symbol_rr(sym, default=2.0))
    except Exception:
        rr_n = _parse_rr(rr)

    if entry is not None and sl is not None and tp is not None:
        risk_pts = abs(float(entry) - float(sl))
        reward_pts = abs(float(tp) - float(entry))
        if risk_pts > 0 and reward_pts > 0:
            rr_n = reward_pts / risk_pts
    else:
        # No live prices — use research RR for this pair (desk SYMBOL_RR)
        cls_defaults = _DEFAULT_RISK_REWARD.get(cls, (8.0, 16.0, 2.0))
        risk_pts = float(cls_defaults[0])
        reward_pts = float(risk_pts) * float(rr_n)

    risk_pts = max(float(risk_pts or 0), 1e-9)
    reward_pts = max(float(reward_pts or 0), 1e-9)

    # Exact lot for ≥$target at TP
    lot_profit = target / (reward_pts * units)

    # Prefer meeting TP $; capital risk is a soft check after sizing
    planned_loss = target / max(rr_n, 0.5)
    capital_risk_cap = cap * _max_risk_pct()
    ceiling = _capital_ceiling(sym, cap)
    import os

    hard_cap = {
        "fx": float(os.getenv("LOT_HARD_CAP_FX", "0.50")),
        "crypto": 3.0,
        "index": float(os.getenv("LOT_HARD_CAP_INDEX", "10.0")),
        "energy": float(os.getenv("LOT_HARD_CAP_ENERGY", "1.5")),
        "silver": 3.0,
        "gold": 5.0,
    }.get(cls, 8.0)

    lot = min(lot_profit, ceiling, hard_cap)
    # Round up to 2dp so we don't land at $396 / $399 from float trunc
    import math

    lot = max(0.01, math.ceil(lot * 100 - 1e-9) / 100.0)
    lot = min(lot, ceiling, hard_cap)
    lot = max(0.01, round(lot, 2))

    def _exp(lot_v: float) -> tuple[float, float]:
        if entry is not None and sl is not None and tp is not None:
            return (
                abs(leg_pnl("BUY", float(entry), float(tp), lot_v, symbol=sym)),
                abs(leg_pnl("BUY", float(entry), float(sl), lot_v, symbol=sym)),
            )
        return (
            round(reward_pts * units * lot_v, 2),
            round(risk_pts * units * lot_v, 2),
        )

    exp_profit, exp_loss = _exp(lot)

    # Enforce $400+ TP when caps allow — bump (ceil) until we clear min_target
    guard = 0
    while exp_profit + 0.01 < min_target and guard < 8:
        guard += 1
        need = min_target / (reward_pts * units)
        bumped = min(math.ceil(need * 100 - 1e-9) / 100.0, ceiling, hard_cap)
        bumped = max(0.01, round(bumped, 2))
        if bumped <= lot + 1e-9:
            # one more tick if still short and under caps
            bumped = min(round(lot + 0.01, 2), ceiling, hard_cap)
        if bumped <= lot + 1e-9:
            break
        lot = bumped
        exp_profit, exp_loss = _exp(lot)

    # Soft risk: warn-only shrink if SL risk >> capital_risk_cap *and* TP still ≥ min
    if capital_risk_cap > 0 and exp_loss > capital_risk_cap * 1.35:
        lot_risk = capital_risk_cap / (risk_pts * units)
        shrunk = max(0.01, round(min(lot, lot_risk), 2))
        sp, sloss = _exp(shrunk)
        if sp + 0.5 >= min_target * 0.85 or sloss <= capital_risk_cap:
            lot, exp_profit, exp_loss = shrunk, sp, sloss

    # Hard absolute SL $ cap (desk mastery A) — always enforce, even if TP shrinks
    try:
        from config import MAX_RISK_USD_PER_TRADE

        abs_cap = float(MAX_RISK_USD_PER_TRADE or 0)
    except Exception:
        abs_cap = float(os.getenv("MAX_RISK_USD_PER_TRADE", "200") or 200)
    # Small accounts: tighter of abs/$% . Larger desks: capital % (abs is a floor only).
    if capital_risk_cap > 0 and abs_cap > 0:
        if cap >= 3000:
            risk_usd_cap = max(capital_risk_cap, abs_cap * 0.5)
        else:
            risk_usd_cap = min(abs_cap, capital_risk_cap)
    elif capital_risk_cap > 0:
        risk_usd_cap = capital_risk_cap
    else:
        risk_usd_cap = abs_cap
    # Let swing sleeve actually express 1.5× (else $200 abs cap clips it back)
    if risk_usd_cap > 0 and swing > 1.0:
        risk_usd_cap = float(risk_usd_cap) * float(swing)
    if risk_usd_cap > 0 and exp_loss > risk_usd_cap + 0.5:
        lot_risk = risk_usd_cap / (risk_pts * units)
        shrunk = max(0.01, round(min(lot, lot_risk), 2))
        sp, sloss = _exp(shrunk)
        lot, exp_profit, exp_loss = shrunk, sp, sloss
        # One more tick down if still over (float)
        guard = 0
        while exp_loss > risk_usd_cap + 0.5 and lot > 0.01 and guard < 12:
            guard += 1
            lot = max(0.01, round(lot - 0.01, 2))
            exp_profit, exp_loss = _exp(lot)

    return {
        "lot_size": lot,
        "target_profit_usd": round(min_target, 2),
        "expected_profit_usd": round(exp_profit, 2),
        "expected_loss_usd": round(exp_loss, 2),
        "rr": round(rr_n, 2),
        "risk_pts": risk_pts,
        "reward_pts": reward_pts,
        "capital": round(cap, 2),
        "capital_ceiling_lot": round(ceiling, 2),
        "asset_class": cls,
        "aggression": agg,
        "planned_loss_usd": round(planned_loss, 2),
        "allocation_mult": round(alloc, 2),
        "swing_lot_mult": round(swing, 2),
        "risk_usd_cap": round(risk_usd_cap, 2) if risk_usd_cap else None,
    }


def resolve_lot_size(
    symbol: Optional[str] = None,
    capital: Optional[float] = None,
    *,
    aggression: Optional[float] = None,
    entry: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    rr: Any = None,
) -> float:
    try:
        import runtime_config

        if not bool(runtime_config.get("dynamic_lots_enabled", True)):
            return float(runtime_config.get("trade_lot_size", TRADE_LOT_SIZE))
    except Exception:
        pass

    sized = lot_for_profit_target(
        symbol,
        entry=entry,
        sl=sl,
        tp=tp,
        rr=rr,
        capital=capital,
    )
    return float(sized["lot_size"])


def partial_fracs_for_symbol(rr_n: int, symbol: Optional[str] = None) -> list[float]:
    from config import RR_TP_FRACS

    base = list(RR_TP_FRACS.get(rr_n, RR_TP_FRACS[2]))
    cls = asset_class(symbol)
    if cls in ("gold", "crypto", "silver"):
        scale = 1.15 if cls != "crypto" else 1.20
    elif cls == "fx":
        scale = 0.80
    else:
        scale = 1.0

    try:
        from analysis.instrument_rl import overlays_for

        boost = float(overlays_for(symbol).get("partial_boost") or 1.0)
        scale *= max(0.75, min(1.50, boost))
    except Exception:
        pass

    out = [min(0.70, max(0.25, f * scale)) for f in base]
    booked = sum(out)
    if booked > 0.88:
        out = [f * (0.88 / booked) for f in out]
    return [round(f, 4) for f in out]


def trail_lock_for_symbol(symbol: Optional[str] = None) -> float:
    from config import TRAIL_PROFIT_LOCK

    cls = asset_class(symbol)
    lock = float(TRAIL_PROFIT_LOCK)
    if cls == "crypto":
        lock = min(0.85, lock + 0.12)
    elif cls in ("gold", "silver"):
        lock = min(0.82, lock + 0.08)
    elif cls == "fx":
        lock = max(0.55, lock - 0.12)

    try:
        from analysis.instrument_rl import overlays_for

        lock += float(overlays_for(symbol).get("trail_delta") or 0.0)
    except Exception:
        pass
    return max(0.50, min(0.90, lock))


def sizing_snapshot(
    symbol: Optional[str] = None,
    *,
    entry: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    rr: Any = None,
    capital: Optional[float] = None,
) -> dict[str, Any]:
    """UI / learning / execute metadata."""
    sized = lot_for_profit_target(
        symbol, entry=entry, sl=sl, tp=tp, rr=rr, capital=capital
    )
    return {
        "capital": sized["capital"],
        "lot_size": sized["lot_size"],
        "asset_class": sized["asset_class"],
        "vol_mult": _VOL_MULT.get(sized["asset_class"], 1.0),
        "tp1_frac": partial_fracs_for_symbol(2, symbol)[0],
        "trail_lock": trail_lock_for_symbol(symbol),
        "aggression": sized["aggression"],
        "target_profit_usd": sized["target_profit_usd"],
        "expected_profit_usd": sized["expected_profit_usd"],
        "expected_loss_usd": sized["expected_loss_usd"],
        "rr": sized["rr"],
        "allocation_mult": sized.get("allocation_mult", 1.0),
        "swing_lot_mult": sized.get("swing_lot_mult", 1.0),
        "risk_usd_cap": sized.get("risk_usd_cap"),
        "mode": "target_profit",
    }
