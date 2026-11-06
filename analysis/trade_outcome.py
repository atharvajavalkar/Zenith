"""Win / loss / breakeven classification for closed trades."""

from __future__ import annotations

from typing import Any, Optional


def _get(row: Any, key: str, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _symbol(row: Any) -> str:
    return str(_get(row, "symbol") or _get(row, "SYMBOL") or "GOLD").upper().strip()


def be_price_tolerance(symbol: Optional[str] = None) -> float:
    """
    Max |SL − entry| to treat a stop as “parked at entry” (true BE).

    Must stay tight vs typical 1R — ETH SL≈$5 used to sit inside a $15 band
    so every micro full-stop was counted as breakeven.
    """
    s = str(symbol or "GOLD").upper()
    if s in ("GOLD", "XAUUSD", "SILVER", "XAGUSD"):
        return 0.05  # ~5 cents
    if s in ("ETHUSD", "ETH"):
        return 0.75
    if s in ("BTCUSD", "BTC") or s.startswith("BTC"):
        return 5.0
    if s in ("NAS100", "US100", "USTEC"):
        return 1.5
    if s in ("UKOIL", "USOIL", "XTIUSD", "XBRUSD"):
        return 0.05
    if s.endswith("JPY"):
        return 0.03  # ~3 points
    # FX majors (5-digit)
    return 0.00025  # ~2.5 pips


def _exit_price_floor(symbol: Optional[str] = None) -> float:
    """Tight floor for SL/TP exit proximity (not the BE SL≈entry band)."""
    s = str(symbol or "GOLD").upper()
    if s in ("GOLD", "XAUUSD", "SILVER", "XAGUSD"):
        return 0.05
    if s in ("ETHUSD", "ETH"):
        return 1.0
    if s in ("BTCUSD", "BTC") or s.startswith("BTC"):
        return 10.0
    if s.endswith("JPY"):
        return 0.02
    if s in ("NAS100", "US100", "USTEC"):
        return 1.0
    if s in ("UKOIL", "USOIL", "XTIUSD", "XBRUSD"):
        return 0.05
    return 0.00015


def _estimated_one_r_usd(row: Any) -> Optional[float]:
    """Rough 1R cash from entry/SL/lot — used to size BE pnl band."""
    entry = _get(row, "entry")
    sl = _get(row, "sl")
    if entry is None or sl is None:
        return None
    try:
        risk_px = abs(float(entry) - float(sl))
        lot = float(_get(row, "lot_size") or _get(row, "entry_lot") or 0.1)
    except Exception:
        return None
    if risk_px <= 0 or lot <= 0:
        return None
    # Prefer stamped entry_lot from features when present
    feats = _get(row, "features")
    if isinstance(feats, str):
        try:
            import json

            feats = json.loads(feats)
        except Exception:
            feats = None
    if isinstance(feats, dict):
        for k in ("entry_lot", "lot_size"):
            if feats.get(k) is not None:
                try:
                    lot = float(feats[k])
                    break
                except Exception:
                    pass
    try:
        from strategy.position_sizing import contract_units_per_lot

        return risk_px * lot * float(contract_units_per_lot(_symbol(row)))
    except Exception:
        return risk_px * lot


def be_pnl_tolerance(row: Any) -> float:
    """Near-zero cash after costs still counts as BE — never swallow a full 1R."""
    one_r = _estimated_one_r_usd(row)
    if one_r is not None and one_r > 0:
        # ≤25% of 1R, hard-capped so ETH -$7 stops are losses not BE
        return max(0.5, min(one_r * 0.25, 4.0))
    try:
        lot = float(_get(row, "lot_size") or 0.1)
    except Exception:
        lot = 0.1
    sym = _symbol(row)
    if sym in ("ETHUSD", "ETH", "BTCUSD", "BTC") or sym.startswith("BTC"):
        return max(0.5, min(3.0, 1.5 * max(lot, 0.01)))
    return max(1.0, min(8.0, 3.0 * max(lot, 0.01)))


def classify_broker_exit(
    signal: Optional[str],
    entry: Optional[float],
    sl: Optional[float],
    tp: Optional[float],
    exit_price: Optional[float],
    *,
    symbol: Optional[str] = None,
    tp1: Optional[float] = None,
    tp2: Optional[float] = None,
    tp3: Optional[float] = None,
) -> str:
    """Infer closed_sl / closed_tp / closed_broker from exit vs stops.

    XM sync previously stamped everything closed_manual even when the broker
    hit SL/TP. Use price proximity (and stop-through) to classify.
    """
    if signal not in ("BUY", "SELL") or entry is None or exit_price is None:
        return "closed_broker"

    entry_f = float(entry)
    exit_f = float(exit_price)
    sym = str(symbol or "GOLD").upper()
    risk = abs(entry_f - float(sl)) if sl is not None else None
    # BE tol must NOT drive exit labels (old ETH path: $15×4=$60 → fake TP4).
    if risk and risk > 0:
        tol = max(risk * 0.20, _exit_price_floor(sym))
    else:
        tol = _exit_price_floor(sym) * 4.0
    slip = min(tol, (risk * 0.10) if risk and risk > 0 else tol)

    def _near(level: Optional[float]) -> bool:
        if level is None:
            return False
        return abs(exit_f - float(level)) <= tol

    near_sl = _near(sl)
    if sl is not None:
        sl_f = float(sl)
        if signal == "BUY" and exit_f <= sl_f + slip:
            near_sl = True
        if signal == "SELL" and exit_f >= sl_f - slip:
            near_sl = True

    tp_levels = [x for x in (tp, tp1, tp2, tp3) if x is not None]
    profitable = (signal == "BUY" and exit_f > entry_f) or (
        signal == "SELL" and exit_f < entry_f
    )
    near_tp = False
    if profitable and tp_levels:
        near_tp = any(_near(x) for x in tp_levels)
        for level in tp_levels:
            lv = float(level)
            if signal == "BUY" and exit_f >= lv - slip:
                near_tp = True
            if signal == "SELL" and exit_f <= lv + slip:
                near_tp = True

    if near_sl and near_tp:
        d_sl = abs(exit_f - float(sl)) if sl is not None else 1e18
        d_tp = min(abs(exit_f - float(x)) for x in tp_levels) if tp_levels else 1e18
        return "closed_sl" if d_sl <= d_tp else "closed_tp"
    if near_sl:
        return "closed_sl"
    if near_tp:
        return "closed_tp"

    if sl is not None and risk and risk > 0:
        adverse = (signal == "BUY" and exit_f < entry_f) or (
            signal == "SELL" and exit_f > entry_f
        )
        if adverse and abs(exit_f - float(sl)) <= max(tol, risk * 0.35):
            return "closed_sl"

    return "closed_broker"


def sl_at_breakeven(
    signal: str | None,
    entry: float | None,
    sl: float | None,
    symbol: str | None = None,
) -> bool:
    """Stop placed essentially at entry (not trailed into profit)."""
    if signal not in ("BUY", "SELL") or entry is None or sl is None:
        return False
    return abs(float(sl) - float(entry)) <= be_price_tolerance(symbol)


def sl_trailed_profit(
    signal: str | None,
    entry: float | None,
    sl: float | None,
    symbol: str | None = None,
) -> bool:
    """Stop trailed beyond entry — counts as a win when stopped out."""
    if signal not in ("BUY", "SELL") or entry is None or sl is None:
        return False
    tol = be_price_tolerance(symbol)
    if abs(float(sl) - float(entry)) <= tol:
        return False
    entry_f = float(entry)
    sl_f = float(sl)
    if signal == "BUY":
        return sl_f > entry_f + tol
    return sl_f < entry_f - tol


def _pnl_value(row: Any) -> Optional[float]:
    # Prefer stamped XM deal cash when present
    feats = _get(row, "features")
    if isinstance(feats, str):
        try:
            import json

            feats = json.loads(feats)
        except Exception:
            feats = None
    if isinstance(feats, dict) and feats.get("pnl_source") == "xm_deal":
        if feats.get("xm_deal_pnl") is not None:
            try:
                return float(feats["xm_deal_pnl"])
            except Exception:
                pass
    pnl = _get(row, "pnl")
    if pnl is None:
        return None
    try:
        return float(pnl)
    except Exception:
        return None


def is_breakeven_trade(row: Any) -> bool:
    """
    Flat exit: stop essentially at entry AND cash result near zero.
    Full micro-stops (e.g. ETH −$7 on a $5 SL) must never count as BE.
    """
    if _get(row, "signal") not in ("BUY", "SELL"):
        return False
    if _get(row, "status") != "closed":
        return False
    if _get(row, "pnl_status") != "closed_sl":
        return False
    sym = _symbol(row)
    if not sl_at_breakeven(
        _get(row, "signal"), _get(row, "entry"), _get(row, "sl"), sym
    ):
        return False
    pnl = _pnl_value(row)
    if pnl is None:
        return True
    return abs(pnl) <= be_pnl_tolerance(row)


def is_winning_trade(row: Any) -> bool:
    if is_breakeven_trade(row):
        return False
    if _get(row, "signal") not in ("BUY", "SELL"):
        return False
    if _get(row, "status") != "closed":
        return False

    pnl = _pnl_value(row)
    # Cash is truth — never crown a losing close a win via status label
    if pnl is not None and pnl < 0:
        return False
    if pnl is not None and pnl > be_pnl_tolerance(row):
        return True
    if pnl is not None and pnl > 0:
        return True

    pnl_status = _get(row, "pnl_status")
    if pnl_status == "closed_tp":
        return True

    if pnl_status == "closed_sl" and sl_trailed_profit(
        _get(row, "signal"),
        _get(row, "entry"),
        _get(row, "sl"),
        _symbol(row),
    ):
        return True

    return False


def is_losing_trade(row: Any) -> bool:
    if is_breakeven_trade(row):
        return False
    if _get(row, "signal") not in ("BUY", "SELL"):
        return False
    if _get(row, "status") != "closed":
        return False

    pnl = _pnl_value(row)
    if pnl is not None and pnl < 0:
        return True

    if _get(row, "pnl_status") == "closed_sl":
        return not sl_trailed_profit(
            _get(row, "signal"),
            _get(row, "entry"),
            _get(row, "sl"),
            _symbol(row),
        )

    return False


def outcome_label(row: Any) -> str:
    if is_breakeven_trade(row):
        return "breakeven"
    if is_winning_trade(row):
        return "win"
    if is_losing_trade(row):
        return "loss"
    return "flat"
