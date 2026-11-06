"""Independent multi-timeframe scanner.

Scans each TF for its *own* setup (not cloned legs from one 5m signal).
A 1m trade can close in minutes; a 4h trade can run for days — each with
its own SL / TP / partials / trailing.

Available frames (live): 1m, 3m, 5m, 15m, 1h, 4h
Backtest from 5m base: 5m, 15m, 1h, 4h (1m/3m approximated when absent)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from config import MINUTE_SL_ATR
from strategy.monte_carlo import choose_minute_rr

logger = logging.getLogger(__name__)

# Full ladder — higher TFs hold longer with wider stops
MTF_SCAN_ORDER = ("1m", "3m", "5m", "15m", "1h", "4h")

TF_SL_ATR: dict[str, float] = {
    "1m": 0.85,
    "3m": 0.95,
    "5m": 1.10,
    "15m": 1.35,
    "1h": 1.65,
    "4h": 2.10,
}

# Max simultaneous opens per TF (independent holds)
MTF_MAX_PER_TF = 1


def _remap_frames(
    exec_tf: str,
    frames: dict[str, Optional[pd.DataFrame]],
) -> Optional[dict[str, Any]]:
    """
    Map the chosen exec TF into _compute_signal's (1m/3m/5m/15m/1h/4h) slots
    so indicators, zones, and SMC run on that TF's bar structure.
    """
    primary = frames.get(exec_tf)
    if primary is None or len(primary) < 40:
        return None

    f1 = frames.get("1m")
    f3 = frames.get("3m")
    f5 = frames.get("5m")
    f15 = frames.get("15m")
    f1h = frames.get("1h")
    f4h = frames.get("4h")

    # Always keep true higher-TF bias context when available
    ctx_1h = f1h if f1h is not None and len(f1h) >= 30 else primary
    ctx_4h = f4h if f4h is not None and len(f4h) >= 20 else ctx_1h

    if exec_tf == "1m":
        return {
            "df_1m": primary,
            "df_3m": f3 if f3 is not None else f5,
            "df_5m": f5 if f5 is not None else primary,
            "df_15m": f15 if f15 is not None else primary,
            "df_1h": ctx_1h,
            "df_4h": ctx_4h,
        }
    if exec_tf == "3m":
        return {
            "df_1m": primary,
            "df_3m": primary,
            "df_5m": f5 if f5 is not None else primary,
            "df_15m": f15 if f15 is not None else primary,
            "df_1h": ctx_1h,
            "df_4h": ctx_4h,
        }
    if exec_tf == "5m":
        return {
            "df_1m": primary,
            "df_3m": f3 if f3 is not None else primary,
            "df_5m": primary,
            "df_15m": f15 if f15 is not None else primary,
            "df_1h": ctx_1h,
            "df_4h": ctx_4h,
        }
    if exec_tf == "15m":
        return {
            "df_1m": primary,
            "df_3m": primary,
            "df_5m": primary,
            "df_15m": primary,
            "df_1h": ctx_1h,
            "df_4h": ctx_4h,
        }
    if exec_tf == "1h":
        return {
            "df_1m": primary,
            "df_3m": primary,
            "df_5m": primary,
            "df_15m": primary,
            "df_1h": primary,
            "df_4h": ctx_4h,
        }
    if exec_tf == "4h":
        return {
            "df_1m": primary,
            "df_3m": primary,
            "df_5m": primary,
            "df_15m": primary,
            "df_1h": primary,
            "df_4h": primary,
        }
    return None


def _apply_tf_risk(sig: dict, exec_tf: str, df: pd.DataFrame) -> dict:
    """Re-size SL/TP to this TF's ATR so a 4h trade isn't sized like a 1m scalp."""
    if sig.get("SIGNAL") not in ("BUY", "SELL"):
        return sig

    # Preserve forced-RR brain stops (already sized to entry wick / ATR)
    brain = str(sig.get("BRAIN") or "").lower()
    tp_src = str(sig.get("TP_SOURCE") or "").lower()
    entry_brain = str(sig.get("ENTRY_BRAIN") or "").lower()
    try:
        from config import is_forced_2r_brain

        is_forced = (
            is_forced_2r_brain(brain)
            or is_forced_2r_brain(entry_brain)
            or is_forced_2r_brain(tp_src)
            or tp_src.startswith("ema_20_50")
        )
    except Exception:
        is_forced = brain in (
            "fib_liq",
            "ema_20_50",
            "ema_20_50_tight",
            "sma_20_50",
            "breakout_20",
            "don_100",
            "pa_liq_engulf",
            "pa_opp_gold",
            "usdjpy_opp_gold",
            "bb_2",
            "bb_2.5",
            "cta_10_40",
            "volmom",
            "session_mix",
            "circadian_kalman_z",
            "entropy_tod",
        ) or tp_src.startswith("ema_20_50")
    if is_forced:
        out = dict(sig)
        out["EXEC_TF"] = exec_tf
        out["MTF_INDEPENDENT"] = True
        return out

    direction = sig["SIGNAL"]
    entry = float(sig.get("ENTRY") or df.iloc[-1]["close"])
    atr = float(df.iloc[-1].get("ATR") or sig.get("ATR") or 1.0)
    sl_mult = float(TF_SL_ATR.get(exec_tf, MINUTE_SL_ATR))
    risk = max(atr * sl_mult, atr * 0.5)
    sl = entry - risk if direction == "BUY" else entry + risk
    state = sig.get("STATE") or "CHOP"
    sym = sig.get("SYMBOL") or sig.get("INSTRUMENT")

    struct_tp_price = None
    try:
        from analysis.structure_targets import structure_enabled_for, structure_take_profit
        from config import STRUCTURE_TP_MAX_RR, STRUCTURE_TP_MIN_RR, STRUCTURE_TP_MODE

        if structure_enabled_for(sym):
            pack = structure_take_profit(
                direction,
                entry,
                sl,
                atr,
                min_rr=STRUCTURE_TP_MIN_RR,
                max_rr=STRUCTURE_TP_MAX_RR,
            )
            if pack:
                struct_tp_price = float(pack["tp"])
        mode = STRUCTURE_TP_MODE
    except Exception:
        mode = "prefer"

    best = choose_minute_rr(
        direction,
        entry,
        sl,
        state,
        atr,
        zones=None,
        symbol=sym,
        structure_tp=struct_tp_price,
        structure_mode=mode,
    )
    if best is None:
        return sig

    out = dict(sig)
    from strategy.price_format import round_price

    out["ENTRY"] = round_price(entry, sym)
    out["SL"] = round_price(sl, sym)
    out["TP"] = round_price(best["tp"], sym)
    out["TP1"] = round_price(best.get("tp1"), sym)
    out["TP2"] = round_price(best.get("tp2"), sym)
    out["TP3"] = round_price(best.get("tp3"), sym)
    rr_v = best.get("rr")
    if rr_v is not None and float(rr_v) == int(float(rr_v)):
        out["RR"] = f"1:{int(rr_v)}"
    else:
        out["RR"] = f"1:{float(rr_v):.2f}" if rr_v is not None else None
    out["TP_SOURCE"] = best.get("tp_source")
    out["SUCCESS_PERCENT"] = best["mc_prob"]
    out["MARKOV_MC_PROB"] = best.get("mc_prob")
    out["ATR"] = round(atr, 4)
    out["TF_SL_ATR"] = sl_mult
    out["EXEC_TF"] = exec_tf
    out["MTF_INDEPENDENT"] = True
    out["ZONE_TFS"] = f"{exec_tf.upper()} setup"
    out["REASON"] = (
        f"MTF {exec_tf} independent setup — SL {sl_mult:.2f}×ATR, "
        f"RR {out['RR']}, hold on {exec_tf} structure"
    )
    return out


def scan_all_timeframes(
    frames: dict[str, Optional[pd.DataFrame]],
    *,
    only_tfs: Optional[tuple[str, ...]] = None,
    skip_tfs: Optional[set[str]] = None,
    symbol: Optional[str] = None,
) -> list[dict]:
    """
    Run the live signal engine once per TF. Returns 0..N independent BUY/SELL payloads.
    """
    from strategy.signal import _compute_signal

    order = only_tfs or MTF_SCAN_ORDER
    skip = skip_tfs or set()
    out: list[dict] = []
    sym = str(symbol).upper().strip() if symbol else None

    for tf in order:
        if tf in skip:
            continue
        kwargs = _remap_frames(tf, frames)
        if kwargs is None:
            continue
        try:
            sig = _compute_signal(**kwargs, symbol=sym)
        except Exception as exc:
            logger.debug("MTF scan %s failed: %s", tf, exc)
            continue

        if sig.get("SIGNAL") not in ("BUY", "SELL"):
            continue

        primary = frames.get(tf)
        if primary is None:
            continue
        if sym:
            sig["SYMBOL"] = sym
            sig["INSTRUMENT"] = sym
        tuned = _apply_tf_risk(sig, tf, primary)
        # Tag candle time of the exec TF for bucket / dedupe
        tuned["CANDLE_TIME"] = str(primary.index[-1])
        tuned["CANDLE_TIME_TF"] = str(primary.index[-1])
        tuned["EXEC_TF"] = tf
        out.append(tuned)

    return out


def tf_bar_closed(bar_time: pd.Timestamp, tf: str) -> bool:
    """True when `bar_time` (usually a 5m stamp) aligns with a closed TF candle."""
    minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240}.get(tf)
    if minutes is None:
        return False
    total = int(bar_time.hour) * 60 + int(bar_time.minute)
    return total % minutes == 0


def open_tfs(open_trades: list[dict]) -> set[str]:
    """Which exec TFs already have an open position."""
    return {
        str(t.get("exec_tf") or t.get("EXEC_TF") or "")
        for t in open_trades
        if t.get("status") in (None, "open", "partial", "partial2", "partial3")
        or t.get("pnl_status") in ("open", "partial", "partial2", "partial3")
    }
