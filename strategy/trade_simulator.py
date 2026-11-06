"""
Bar-by-bar trade management for backtesting.

Mirrors the live tick logic in signal_worker._update_open_trades_from_quote:
  - TP1 / TP2 / TP3 partial booking (RR_TP_FRACS)
  - Breakeven SL after TP1, SL → TP1 after TP2, SL → TP2 after TP3
  - Dynamic trailing stop on the runner (TRAIL_PROFIT_LOCK)
  - SL / final-TP exit resolution
"""
from __future__ import annotations

from typing import Any, Optional

from analysis.trade_outcome import (
    is_breakeven_trade,
    is_losing_trade,
    is_winning_trade,
    outcome_label,
    sl_trailed_profit,
)
from config import RR_TP_FRACS, TRADE_LOT_SIZE
from strategy.position_sizing import leg_pnl
from strategy.trailing_stop import apply_trailing_stop, sl_trail_label, trailing_status_message

OPEN_STATUSES = ("open", "partial", "partial2", "partial3")


def rr_int(rr_type: Any) -> int:
    try:
        return int(str(rr_type or "1:2").split(":")[-1])
    except (ValueError, AttributeError):
        return 2


def tp_fracs(rr_n: int, symbol: Optional[str] = None) -> list[float]:
    try:
        from strategy.dynamic_lots import partial_fracs_for_symbol

        return partial_fracs_for_symbol(rr_n, symbol)
    except Exception:
        return RR_TP_FRACS.get(rr_n, RR_TP_FRACS[2])


def price_hit(signal: str, level: float, high: float, low: float) -> bool:
    if signal == "BUY":
        return float(high) >= float(level)
    return float(low) <= float(level)


def runner_fraction(trade: dict) -> float:
    rr_n = rr_int(trade.get("rr_type"))
    fracs = tp_fracs(rr_n, trade.get("symbol") or trade.get("SYMBOL"))
    stage_idx = {"open": 0, "partial": 1, "partial2": 2, "partial3": 3}.get(
        trade.get("pnl_status") or "open", 0
    )
    booked = sum(fracs[:stage_idx])
    return max(0.0, round(1.0 - booked, 6))


def total_pnl(trade: dict, price: float, lot_size: float | None = None) -> float:
    lot = lot_size if lot_size is not None else trade.get("lot_size") or TRADE_LOT_SIZE
    sym = trade.get("symbol") or trade.get("SYMBOL")
    partial = trade.get("partial_pnl") or 0.0
    runner = leg_pnl(trade["signal"], trade["entry"], price, lot, symbol=sym) * runner_fraction(trade)
    return round(partial + runner, 2)


def book_partial(trade: dict, tp_price: float, frac: float, next_status: str, lot: float) -> None:
    sym = trade.get("symbol") or trade.get("SYMBOL")
    booked_pnl = leg_pnl(trade["signal"], trade["entry"], tp_price, lot, symbol=sym) * frac
    trade["partial_pnl"] = round((trade.get("partial_pnl") or 0.0) + booked_pnl, 2)
    trade["pnl_status"] = next_status


def next_tp_target(trade: dict) -> Optional[float]:
    rr_n = rr_int(trade.get("rr_type"))
    status = trade.get("pnl_status")
    if status == "partial":
        return trade.get("tp2") if rr_n >= 3 else trade.get("tp")
    if status == "partial2":
        return trade.get("tp3") if rr_n >= 4 else trade.get("tp")
    return trade.get("tp")


def resolve_exit(
    signal: str,
    sl: Optional[float],
    tp1: Optional[float],
    tp_final: Optional[float],
    pnl_status: Optional[str],
    low: float,
    high: float,
    close: float,
) -> tuple[Optional[str], Optional[float]]:
    """SL hit and final TP hit — intermediate TPs handled separately."""
    if signal not in ("BUY", "SELL"):
        return None, close

    sl_f = float(sl) if sl is not None else None
    tp1_f = float(tp1) if tp1 is not None else None
    tp_f = float(tp_final) if tp_final is not None else None

    if signal == "BUY":
        if sl_f is not None and low <= sl_f:
            return "closed_sl", sl_f
        if tp_f is not None and high >= tp_f:
            if pnl_status in ("partial", "partial2", "partial3"):
                return "closed_tp", tp_f
            if pnl_status == "open" and tp1_f is None:
                return "closed_tp", tp_f
    else:
        if sl_f is not None and high >= sl_f:
            return "closed_sl", sl_f
        if tp_f is not None and low <= tp_f:
            if pnl_status in ("partial", "partial2", "partial3"):
                return "closed_tp", tp_f
            if pnl_status == "open" and tp1_f is None:
                return "closed_tp", tp_f
    return None, close


def new_open_trade(
    signal: str,
    entry: float,
    sl: float,
    tp: Optional[float],
    tp1: Optional[float],
    tp2: Optional[float],
    tp3: Optional[float],
    rr_type: Optional[str],
    lot_size: float,
    mc_prob: float,
    **extra: Any,
) -> dict[str, Any]:
    """Create an open trade dict with all fields needed for live-style management."""
    from strategy.price_format import round_price

    sym = extra.get("symbol") or extra.get("SYMBOL")

    def _r(v):
        return round_price(v, sym)

    trade = {
        "signal": signal,
        "entry": _r(entry),
        "sl": _r(sl),
        "tp": _r(tp),
        "tp1": _r(tp1),
        "tp2": _r(tp2),
        "tp3": _r(tp3),
        "rr_type": rr_type,
        "lot_size": lot_size,
        "mc_prob": round(float(mc_prob), 1),
        "pnl_status": "open",
        "partial_pnl": 0.0,
        "extreme_price": None,
        "status": "open",
        "result": "open",
        "pnl": None,
        "exit_price": None,
        "exit_time": None,
        "sl_trail_status": f"Fixed @ {_r(sl)}",
        "message": "",
        "trail_log": [],
        "symbol": sym,
        "SYMBOL": sym,
    }
    trade.update(extra)
    return trade


def update_trade_on_bar(
    trade: dict,
    high: float,
    low: float,
    close: float,
    lot_size: float | None = None,
    mgmt_mode: str | None = None,
) -> bool:
    """
    Advance one bar of live-style trade management.
    Returns True if the trade closed on this bar.

    mgmt_mode:
      - None / "live" / "partials_trail" — TP partials + dynamic trail (default)
      - "fixed" — full size to final TP or original SL (no partials / trail)
      - "partials_be" — book TP1(+TP2) and SL→BE/TP1, but no dynamic trail
    """
    lot = lot_size if lot_size is not None else trade.get("lot_size") or TRADE_LOT_SIZE
    signal = trade["signal"]
    entry = trade["entry"]
    rr_n = rr_int(trade.get("rr_type"))
    sym = trade.get("symbol") or trade.get("SYMBOL")
    fracs = tp_fracs(rr_n, sym)
    mode = (mgmt_mode or trade.get("mgmt_mode") or "live").lower().strip()
    if mode in ("live", "partials_trail", "trail"):
        mode = "partials_trail"
    allow_partials = mode != "fixed"
    allow_trail = mode == "partials_trail"

    # Fixed RR exit: ignore staged TPs so resolve_exit can close at final TP.
    if mode == "fixed":
        trade["tp1"] = None
        trade["tp2"] = None
        trade["tp3"] = None

    # Same-bar realism: while still fully open, original SL beats TP1.
    # Booking TP1 first then moving SL→BE overstates early wins in chop.
    if trade.get("pnl_status") == "open" and trade.get("sl") is not None:
        sl_open = float(trade["sl"])
        sl_hit = (signal == "BUY" and low <= sl_open) or (signal == "SELL" and high >= sl_open)
        if sl_hit:
            exit_price = sl_open
            trade["pnl"] = total_pnl(trade, exit_price, lot)
            trade["pnl_status"] = "closed_sl"
            trade["status"] = "closed"
            trade["exit_price"] = round(float(exit_price), 2)
            trade["sl_trail_status"] = "Closed"
            trade["outcome"] = outcome_label(trade)
            trade["result"] = trade["outcome"]
            trade["message"] = f"Closed at SL {sl_open:.2f}"
            trade.setdefault("trail_log", []).append(f"Closed at SL {sl_open:.2f} (loss)")
            return True

    # --- TP1 hit (all setups with tp1) ---
    if allow_partials and trade.get("pnl_status") == "open" and trade.get("tp1") is not None:
        if price_hit(signal, float(trade["tp1"]), high, low):
            book_partial(trade, float(trade["tp1"]), fracs[0], "partial", lot)
            trade["sl"] = entry
            pct = int(fracs[0] * 100)
            trade["message"] = f"TP1 hit @ {trade['tp1']:.2f} — {pct}% booked, SL → breakeven"
            trade.setdefault("trail_log", []).append(
                f"TP1 {trade['tp1']:.2f}: booked {pct}%, SL→BE"
            )
            trade["sl_trail_status"] = sl_trail_label(signal, float(entry), float(entry), "partial")

    # --- TP2 hit (1:3 and 1:4) ---
    if allow_partials and trade.get("pnl_status") == "partial" and rr_n >= 3 and trade.get("tp2") is not None:
        if price_hit(signal, float(trade["tp2"]), high, low):
            book_partial(trade, float(trade["tp2"]), fracs[1], "partial2", lot)
            trade["sl"] = float(trade["tp1"])
            pct = int(fracs[1] * 100)
            trade["message"] = (
                f"TP2 hit @ {trade['tp2']:.2f} — {pct}% booked, SL → TP1 ({trade['tp1']:.2f})"
            )
            trade.setdefault("trail_log", []).append(
                f"TP2 {trade['tp2']:.2f}: booked {pct}%, SL→TP1"
            )
            trade["sl_trail_status"] = sl_trail_label(
                signal, float(entry), float(trade["sl"]), "partial2"
            )

    # --- TP3 hit (1:4 only) ---
    if allow_partials and trade.get("pnl_status") == "partial2" and rr_n >= 4 and trade.get("tp3") is not None:
        if price_hit(signal, float(trade["tp3"]), high, low):
            book_partial(trade, float(trade["tp3"]), fracs[2], "partial3", lot)
            trade["sl"] = float(trade["tp2"])
            pct = int(fracs[2] * 100)
            trade["message"] = (
                f"TP3 hit @ {trade['tp3']:.2f} — {pct}% booked, SL → TP2 ({trade['tp2']:.2f})"
            )
            trade.setdefault("trail_log", []).append(
                f"TP3 {trade['tp3']:.2f}: booked {pct}%, SL→TP2"
            )
            trade["sl_trail_status"] = sl_trail_label(
                signal, float(entry), float(trade["sl"]), "partial3"
            )

    # --- Dynamic trailing on runner ---
    if allow_trail and trade.get("pnl_status") in ("partial", "partial2", "partial3"):
        next_tp = next_tp_target(trade)
        if next_tp is not None:
            try:
                from strategy.dynamic_lots import trail_lock_for_symbol

                lock = float(
                    trade.get("trail_lock")
                    or trail_lock_for_symbol(sym)
                )
            except Exception:
                from config import TRAIL_PROFIT_LOCK

                lock = TRAIL_PROFIT_LOCK
            new_sl, new_extreme, trail_changed = apply_trailing_stop(
                signal,
                entry,
                next_tp,
                trade.get("sl"),
                trade.get("extreme_price"),
                high,
                low,
                lock_fraction=lock,
                active=True,
            )
            if trail_changed or new_extreme != trade.get("extreme_price"):
                trade["sl"] = new_sl
                trade["extreme_price"] = new_extreme
                if trail_changed:
                    trade.setdefault("trail_log", []).append(
                        f"Trail SL→{float(new_sl):.2f} (extreme {float(new_extreme):.2f})"
                    )
                trade["sl_trail_status"] = sl_trail_label(
                    signal,
                    float(entry),
                    float(new_sl),
                    trade["pnl_status"],
                    float(new_extreme) if new_extreme is not None else None,
                )
                trade["message"] = trailing_status_message(
                    signal,
                    float(entry),
                    float(next_tp),
                    float(new_sl),
                    float(new_extreme),
                    partial=True,
                    stage=trade["pnl_status"],
                )

    # --- SL / final TP exit ---
    exit_status, exit_price = resolve_exit(
        signal,
        trade.get("sl"),
        trade.get("tp1"),
        trade.get("tp"),
        trade.get("pnl_status"),
        low,
        high,
        close,
    )

    trade["pnl"] = total_pnl(trade, exit_price if exit_price is not None else close, lot)

    if exit_status:
        last_trail = trade.get("sl_trail_status")
        trade["pnl_status"] = exit_status
        trade["status"] = "closed"
        trade["exit_price"] = round(float(exit_price), 2) if exit_price is not None else None
        # Keep a readable trail summary instead of only "Closed"
        if last_trail and last_trail != "Closed":
            trade["sl_trail_status"] = f"Closed · {last_trail}"
        else:
            trade["sl_trail_status"] = "Closed"
        trade["outcome"] = outcome_label(trade)
        trade["result"] = trade["outcome"]

        if exit_status == "closed_tp":
            final_label = f"TP{rr_n}" if rr_n > 1 else "TP"
            trade["message"] = f"Closed at {final_label} {exit_price:.2f}"
            trade.setdefault("trail_log", []).append(f"Closed at {final_label} {float(exit_price):.2f}")
        elif exit_status == "closed_sl":
            if is_breakeven_trade(trade):
                trade["message"] = f"Closed at SL {trade['sl']:.2f} (breakeven)"
                trade.setdefault("trail_log", []).append(f"Closed at SL {float(trade['sl']):.2f} (BE)")
            elif sl_trailed_profit(signal, entry, trade.get("sl")):
                trade["message"] = f"Closed at SL {trade['sl']:.2f} (trailed — win)"
                trade.setdefault("trail_log", []).append(f"Closed at SL {float(trade['sl']):.2f} (trailed win)")
            else:
                trade["message"] = f"Closed at SL {trade['sl']:.2f}"
                trade.setdefault("trail_log", []).append(f"Closed at SL {float(trade['sl']):.2f} (loss)")
        return True

    if trade.get("pnl_status") in OPEN_STATUSES and entry and trade.get("sl"):
        trade["sl_trail_status"] = sl_trail_label(
            signal,
            float(entry),
            float(trade["sl"]),
            trade.get("pnl_status"),
            float(trade["extreme_price"]) if trade.get("extreme_price") is not None else None,
        )
    return False


def close_at_market(trade: dict, price: float, exit_time: Any = None) -> dict:
    """Mark-to-market close at end of backtest range (includes booked partials)."""
    trade["exit_price"] = round(float(price), 2)
    trade["pnl"] = total_pnl(trade, float(price))
    trade["pnl_status"] = "closed_manual"
    trade["status"] = "closed"
    trade["sl_trail_status"] = "Closed"
    trade["outcome"] = outcome_label(trade)
    trade["result"] = trade["outcome"]
    trade["message"] = f"Closed at market {price:.2f}"
    trade.setdefault("trail_log", []).append(f"Closed at market {float(price):.2f}")
    if exit_time is not None:
        trade["exit_time"] = exit_time
    return trade


def apply_rr_mode(result: dict, rr_mode: Optional[str]) -> Optional[dict]:
    """
    Overlay a forced RR profile onto a signal result (backtest).

    rr_mode: 'auto' | '1:2' | '1:3' | '1:4'
    Uses the same choose_minute_rr() math as live — only the selection changes.
    """
    if not rr_mode or str(rr_mode).lower() == "auto":
        return result

    signal = result.get("SIGNAL")
    if signal not in ("BUY", "SELL"):
        return result

    try:
        rr_int = int(str(rr_mode).split(":")[-1])
    except (ValueError, TypeError):
        return result
    if rr_int not in (2, 3, 4):
        return result

    entry = result.get("ENTRY")
    sl = result.get("SL")
    if entry is None or sl is None:
        return result

    state = result.get("STATE") or "CHOP"
    atr = float(result.get("ATR") or abs(float(entry) - float(sl)) or 1.0)

    from strategy.monte_carlo import choose_minute_rr

    best = choose_minute_rr(
        signal,
        float(entry),
        float(sl),
        state,
        atr,
        zones=None,
        forced_rr=rr_int,
    )
    if best is None:
        return None

    out = dict(result)
    out["SL"] = round(float(sl), 2)
    out["TP"] = best["tp"]
    out["TP1"] = best.get("tp1")
    out["TP2"] = best.get("tp2")
    out["TP3"] = best.get("tp3")
    out["RR"] = f"1:{best['rr_int']}"
    out["SUCCESS_PERCENT"] = best["mc_prob"]
    out["MARKOV_MC_PROB"] = best["mc_prob"]
    out["STATIC_MC_PROB"] = best.get("static_mc_prob")
    return out


def stats_from_trades(trades: list[dict]) -> dict[str, Any]:
    closed = [t for t in trades if t.get("status") == "closed"]
    wins = sum(1 for t in closed if is_winning_trade(t))
    losses = sum(1 for t in closed if is_losing_trade(t))
    bes = sum(1 for t in closed if is_breakeven_trade(t))
    realized = round(sum(float(t.get("pnl") or 0) for t in closed), 2)
    decisive = wins + losses
    return {
        "closed": len(closed),
        "wins": wins,
        "losses": losses,
        "breakevens": bes,
        "realized_pnl": realized,
        "win_rate": round(wins / decisive, 4) if decisive else 0.0,
    }
