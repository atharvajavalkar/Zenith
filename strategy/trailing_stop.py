"""Dual-TP trade management: TP1 partial, breakeven, 50% profit trail to TP2."""

from typing import Optional, Tuple

from config import TRAIL_PROFIT_LOCK


def one_r_price(entry: float, tp2: float) -> float:
    """TP1 level — 1R on a 1:2 setup (midpoint entry→TP2)."""
    return (float(entry) + float(tp2)) / 2.0


def progress_toward_tp(signal: str, entry: float, tp: float, extreme: float) -> float:
    """Fraction of entry→TP distance covered (0..1)."""
    entry_f = float(entry)
    tp_f = float(tp)
    extreme_f = float(extreme)
    total = abs(tp_f - entry_f)
    if total <= 0:
        return 0.0
    if signal == "BUY":
        return max(0.0, min(1.0, (extreme_f - entry_f) / (tp_f - entry_f)))
    return max(0.0, min(1.0, (entry_f - extreme_f) / (entry_f - tp_f)))


def apply_trailing_stop(
    signal: str,
    entry: Optional[float],
    tp: Optional[float],
    sl: Optional[float],
    extreme_price: Optional[float],
    high: Optional[float],
    low: Optional[float],
    lock_fraction: float = TRAIL_PROFIT_LOCK,
    active: bool = True,
) -> Tuple[Optional[float], Optional[float], bool]:
    """
    After TP1 partial: SL at breakeven minimum, then trail to lock `lock_fraction`
    of the max favorable move (default 50% of floating profit on the runner).
    """
    if not active or signal not in ("BUY", "SELL") or entry is None or tp is None or sl is None:
        return sl, extreme_price, False

    entry_f = float(entry)
    sl_f = float(sl)
    high_f = float(high) if high is not None else None
    low_f = float(low) if low is not None else None
    lock = max(0.0, min(1.0, float(lock_fraction)))

    changed = False
    new_sl = sl_f
    new_extreme = extreme_price

    if signal == "BUY":
        if high_f is None:
            return sl, extreme_price, False

        if new_extreme is None:
            new_extreme = high_f
        elif high_f > float(new_extreme):
            new_extreme = high_f
            changed = True

        move = float(new_extreme) - entry_f
        if move <= 0:
            return new_sl, new_extreme, changed

        candidate_sl = entry_f + lock * move
        candidate_sl = max(entry_f, candidate_sl)

        if candidate_sl > new_sl + 1e-9:
            new_sl = round(candidate_sl, 2)
            changed = True

    else:
        if low_f is None:
            return sl, extreme_price, False

        if new_extreme is None:
            new_extreme = low_f
        elif low_f < float(new_extreme):
            new_extreme = low_f
            changed = True

        move = entry_f - float(new_extreme)
        if move <= 0:
            return new_sl, new_extreme, changed

        candidate_sl = entry_f - lock * move
        candidate_sl = min(entry_f, candidate_sl)

        if candidate_sl < new_sl - 1e-9:
            new_sl = round(candidate_sl, 2)
            changed = True

    return new_sl, new_extreme, changed


_PARTIAL_STAGES = ("partial", "partial2", "partial3")


def sl_trail_label(
    signal: str,
    entry: float,
    sl: float,
    pnl_status: str | None,
    extreme: float | None = None,
) -> str:
    """Short label for dashboard SL trail column."""
    if pnl_status in ("closed_tp", "closed_sl", "closed_manual"):
        return "Closed"

    if pnl_status == "open":
        return f"Fixed @ {sl:.2f}"

    if pnl_status not in _PARTIAL_STAGES:
        return "—"

    stage_map = {"partial": "TP1", "partial2": "TP2", "partial3": "TP3"}
    stage = stage_map.get(pnl_status, "TP1")

    if abs(sl - entry) < 0.02:
        return f"Breakeven ({stage} done)"

    # SL above entry for BUY or below for SELL → locking profit
    if (signal == "BUY" and sl > entry) or (signal == "SELL" and sl < entry):
        locked_pts = abs(sl - entry)
        lock_pct = int(TRAIL_PROFIT_LOCK * 100)
        return f"Trail {lock_pct}% @ {sl:.2f} (+{locked_pts:.2f}) [{stage}]"

    locked_pts = abs(sl - entry)
    lock_pct = int(TRAIL_PROFIT_LOCK * 100)
    return f"Trail {lock_pct}% @ {sl:.2f} (+{locked_pts:.2f})"


def trailing_status_message(
    signal: str,
    entry: float,
    tp: float,
    sl: float,
    extreme: float,
    partial: bool = False,
    stage: str = "partial",
) -> str:
    progress = progress_toward_tp(signal, entry, tp, extreme)
    pct = round(progress * 100, 1)
    lock_pct = int(TRAIL_PROFIT_LOCK * 100)

    if not partial:
        return f"Open 0.1 lot — {pct}% toward TP1 ({one_r_price(entry, tp):.2f})"

    stage_num = {"partial": 1, "partial2": 2, "partial3": 3}.get(stage, 1)
    next_tp_num = stage_num + 1

    if abs(sl - entry) < 0.02:
        return (
            f"TP{stage_num} done — breakeven SL, trailing {lock_pct}% lock "
            f"({pct}% to TP{next_tp_num})"
        )

    return (
        f"Trailing runner [{stage_num}→{next_tp_num}] — SL {sl:.2f} locks {lock_pct}% of profit "
        f"({pct}% toward TP{next_tp_num}, extreme {extreme:.2f})"
    )
