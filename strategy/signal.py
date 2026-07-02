"""Main signal generation orchestrating all analysis layers."""

from typing import Optional

import hashlib

import numpy as np
import pandas as pd
from tvDatafeed import Interval

from analysis.bias import get_bias, get_context_4h
from analysis.direction import get_5m_direction
from analysis.market_state import get_market_state
from analysis.markov_engine import markov_features, markov_store
from analysis.session import get_ny_time_and_session
from analysis.zones import get_local_zones, get_zones
from config import (
    ALLOW_ALL_SESSIONS,
    BARS_1M,
    BARS_3M,
    BARS_15M,
    BARS_1H,
    BARS_4H,
    BARS_5M,
    CONSOLIDATION_MIN_SMC_SCORE,
    DIRECTION_LOCK_5M,
    DISPLACEMENT_ATR_MULT,
    FVG_LOOKBACK,
    MIN_SMC_SCORE,
    MINUTE_SL_ATR,
    MAX_SL_ATR,
    SL_ZONE_EXTRA,
    SMC_LOOKBACK,
    TRADE_5M_MODE,
)
from data.fetcher import fetch_tv_with_pause as fetch_tv
from indicators.prepare import prepare_df
from strategy.monte_carlo import choose_minute_rr
from strategy.trailing_stop import one_r_price

CANDLE_LIMIT = 200


def _seed_mc(candle_time: str) -> None:
    digest = hashlib.md5(candle_time.encode()).hexdigest()
    np.random.seed(int(digest[:8], 16) % (2**32))


def _serialize_candles(df: pd.DataFrame, limit: int = CANDLE_LIMIT) -> list:
    candles = []
    for idx, row in df.tail(limit).iterrows():
        ts = int(pd.Timestamp(idx).timestamp())
        candles.append(
            {
                "time": ts,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": round(float(row.get("volume", 0)), 0),
            }
        )
    return candles


def get_last_swing_levels(df: pd.DataFrame, before_index: int = -1, lookback: int = SMC_LOOKBACK):
    recent = df.iloc[:before_index].tail(lookback)
    swing_highs = recent[recent["SWING_HIGH"]]
    swing_lows = recent[recent["SWING_LOW"]]

    last_high = None if swing_highs.empty else float(swing_highs.iloc[-1]["high"])
    last_low = None if swing_lows.empty else float(swing_lows.iloc[-1]["low"])
    return last_high, last_low


def get_recent_fvg(df: pd.DataFrame, direction: str, lookback: int = FVG_LOOKBACK):
    recent = df.tail(lookback)

    if direction == "BUY":
        gaps = recent.dropna(subset=["BULL_FVG_LOW", "BULL_FVG_HIGH"])
        if gaps.empty:
            return None
        row = gaps.iloc[-1]
        return {
            "type": "BULLISH_FVG",
            "low": float(row["BULL_FVG_LOW"]),
            "high": float(row["BULL_FVG_HIGH"]),
            "mid": float((row["BULL_FVG_LOW"] + row["BULL_FVG_HIGH"]) / 2),
        }

    gaps = recent.dropna(subset=["BEAR_FVG_LOW", "BEAR_FVG_HIGH"])
    if gaps.empty:
        return None
    row = gaps.iloc[-1]
    return {
        "type": "BEARISH_FVG",
        "low": float(row["BEAR_FVG_LOW"]),
        "high": float(row["BEAR_FVG_HIGH"]),
        "mid": float((row["BEAR_FVG_LOW"] + row["BEAR_FVG_HIGH"]) / 2),
    }


def get_smc_profile(
    df: pd.DataFrame,
    direction: str,
    zones: dict,
    df_swings: Optional[pd.DataFrame] = None,
) -> dict:
    row = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(row["close"])
    atr_now = float(row["ATR"])
    swing_df = df_swings if df_swings is not None else df
    last_swing_high, last_swing_low = get_last_swing_levels(swing_df, lookback=40)

    bullish_displacement = bool(
        row["close"] > row["open"]
        and row["BODY"] >= atr_now * DISPLACEMENT_ATR_MULT
        and row["BODY_RATIO"] >= 0.55
    )
    bearish_displacement = bool(
        row["close"] < row["open"]
        and row["BODY"] >= atr_now * DISPLACEMENT_ATR_MULT
        and row["BODY_RATIO"] >= 0.55
    )

    bull_sweep = bool(
        last_swing_low is not None
        and prev["low"] < last_swing_low
        and price > last_swing_low
    )
    bear_sweep = bool(
        last_swing_high is not None
        and prev["high"] > last_swing_high
        and price < last_swing_high
    )

    bull_mss = bool(last_swing_high is not None and price > last_swing_high)
    bear_mss = bool(last_swing_low is not None and price < last_swing_low)

    if direction == "BUY":
        fvg = get_recent_fvg(df, "BUY")
        fvg_near = bool(
            fvg is not None and fvg["low"] <= price <= fvg["high"] + atr_now * 0.35
        )
        score = sum([bull_sweep, bull_mss, bullish_displacement, fvg is not None, fvg_near])
        refined_entry = fvg["mid"] if fvg is not None and price >= fvg["mid"] else price
        structural_sl = float(last_swing_low) - atr_now * 0.05 if last_swing_low is not None else price - atr_now * MINUTE_SL_ATR
        if last_swing_low is not None and last_swing_low < price:
            buf = min(zones["d_width"], atr_now * 0.5) * SL_ZONE_EXTRA
            structural_sl = last_swing_low - buf - atr_now * 0.05
        return {
            "score": float(score),
            "valid": bool(score >= MIN_SMC_SCORE),
            "sweep": bull_sweep,
            "mss": bull_mss,
            "choch": bull_mss and bull_sweep,
            "displacement": bullish_displacement,
            "fvg": fvg,
            "fvg_near": fvg_near,
            "entry": float(refined_entry),
            "sl": float(structural_sl),
            "last_swing_high": last_swing_high,
            "last_swing_low": last_swing_low,
        }

    fvg = get_recent_fvg(df, "SELL")
    fvg_near = bool(
        fvg is not None and fvg["low"] - atr_now * 0.35 <= price <= fvg["high"]
    )
    score = sum([bear_sweep, bear_mss, bearish_displacement, fvg is not None, fvg_near])
    refined_entry = fvg["mid"] if fvg is not None and price <= fvg["mid"] else price
    structural_sl = float(last_swing_high) + atr_now * 0.05 if last_swing_high is not None else price + atr_now * MINUTE_SL_ATR
    if last_swing_high is not None and last_swing_high > price:
        buf = min(zones["s_width"], atr_now * 0.5) * SL_ZONE_EXTRA
        structural_sl = last_swing_high + buf + atr_now * 0.05
    return {
        "score": float(score),
        "valid": bool(score >= MIN_SMC_SCORE),
        "sweep": bear_sweep,
        "mss": bear_mss,
        "choch": bear_mss and bear_sweep,
        "displacement": bearish_displacement,
        "fvg": fvg,
        "fvg_near": fvg_near,
        "entry": float(refined_entry),
        "sl": float(structural_sl),
        "last_swing_high": last_swing_high,
        "last_swing_low": last_swing_low,
    }


def get_smc_read(smc: dict, min_smc_score: float) -> str:
    confirmations = []
    if smc["sweep"]:
        confirmations.append("liquidity sweep")
    if smc["mss"]:
        confirmations.append("market structure shift")
    if smc["choch"]:
        confirmations.append("CHoCH")
    if smc["displacement"]:
        confirmations.append("displacement")
    if smc["fvg_near"]:
        confirmations.append("price near FVG")
    text = ", ".join(confirmations) if confirmations else "no major SMC confirmation yet"
    status = "PASS" if smc["score"] >= min_smc_score else "WAIT"
    return f"{status} - SMC score {smc['score']:.1f}/{min_smc_score:.1f}; {text}"


def _entry_near_zone(direction: str, entry: float, zones: dict, atr: float) -> bool:
    """True when entry is at or inside a supply/demand zone (with ATR margin)."""
    margin = atr * 0.75
    if direction == "BUY":
        return entry <= zones["d_buf_high"] + margin
    return entry >= zones["s_buf_low"] - margin


def _local_micro_sl(
    direction: str,
    entry: float,
    atr: float,
    zones: dict,
    smc: dict,
) -> float:
    """Tight SL just beyond local 1M swing; capped at MAX_SL_ATR."""
    swing_low = smc.get("last_swing_low")
    swing_high = smc.get("last_swing_high")
    buf = min(zones["d_width"], zones["s_width"], atr * 0.5) * SL_ZONE_EXTRA

    if direction == "BUY":
        candidates = [entry - atr * MINUTE_SL_ATR]
        if swing_low is not None and swing_low < entry:
            candidates.append(swing_low - buf - atr * 0.05)
        sl = max(c for c in candidates if c < entry)
        floor = entry - atr * MAX_SL_ATR
        return round(max(sl, floor), 2)

    candidates = [entry + atr * MINUTE_SL_ATR]
    if swing_high is not None and swing_high > entry:
        candidates.append(swing_high + buf + atr * 0.05)
    sl = min(c for c in candidates if c > entry)
    ceiling = entry + atr * MAX_SL_ATR
    return round(min(sl, ceiling), 2)


def build_exit_rule(direction: str, entry: float, sl: float, tp: float) -> str:
    risk = abs(entry - sl)
    if risk <= 0:
        return "WAIT"
    one_r = one_r_price(entry, tp)
    return (
        f"DUAL TP: TP1 1R ({one_r:.2f}) BOOK 50% LOT + BREAKEVEN | "
        f"TP2 2R ({tp:.2f}) TRAIL SL 50% PROFIT LOCK"
    )


def get_no_trade_reason(
    session_mult,
    session_name,
    context_4h,
) -> str:
    if session_mult == 0 and not ALLOW_ALL_SESSIONS:
        return f"No trade: outside approved sessions ({session_name})."
    if context_4h == "TOO_VOLATILE":
        return "No trade: 4H market is too volatile for this model."
    return "No trade: conditions blocked."


def _fetch_all_timeframes():
    df_1m = prepare_df(fetch_tv(Interval.in_1_minute, BARS_1M))
    df_3m = (
        prepare_df(fetch_tv(Interval.in_3_minute, BARS_3M))
        if TRADE_5M_MODE
        else None
    )
    df_5m = prepare_df(fetch_tv(Interval.in_5_minute, BARS_5M))
    df_15m = prepare_df(fetch_tv(Interval.in_15_minute, BARS_15M))
    df_1h = prepare_df(fetch_tv(Interval.in_1_hour, BARS_1H))
    df_4h = prepare_df(fetch_tv(Interval.in_4_hour, BARS_4H))
    return df_1m, df_3m, df_5m, df_15m, df_1h, df_4h


def _direction_5m_allows(trade_direction: str, direction_5m: str) -> bool:
    """Hard lock in 5M mode; soft opposite-block in 1M test mode."""
    if DIRECTION_LOCK_5M or TRADE_5M_MODE:
        return trade_direction == direction_5m
    if trade_direction == "BUY":
        return direction_5m != "SELL"
    return direction_5m != "BUY"


def _compute_signal(df_1m, df_3m, df_5m, df_15m, df_1h, df_4h) -> dict:
    bias_1h = get_bias(df_1h)
    bias_4h = get_bias(df_4h)
    context_4h = get_context_4h(df_4h)

    row = df_1m.iloc[-1]
    prev = df_1m.iloc[-2]
    row_5m = df_5m.iloc[-1]
    state_row = row_5m if TRADE_5M_MODE else row

    candle_time = str(df_5m.index[-1] if TRADE_5M_MODE else df_1m.index[-1])
    _seed_mc(candle_time)

    price = float(row["close"])
    atr_now = float(state_row["ATR"])

    ny_time, session_mult, session_name = get_ny_time_and_session()

    zones = get_zones(df_15m, df_1h, price)
    micro_zones = get_local_zones(
        df_1m, df_5m, price, df_3m=df_3m, use_3m5m=TRADE_5M_MODE
    )
    zone_label = "3M/5M" if TRADE_5M_MODE else "1M/5M"

    zone_proximity = atr_now * 0.35
    near_demand = (
        micro_zones["d_buf_low"] - zone_proximity
        <= price
        <= micro_zones["d_buf_high"] + zone_proximity
    )
    near_supply = (
        micro_zones["s_buf_low"] - zone_proximity
        <= price
        <= micro_zones["s_buf_high"] + zone_proximity
    )
    macro_near_demand = zones["d_buf_low"] <= price <= zones["d_buf_high"]
    macro_near_supply = zones["s_buf_low"] <= price <= zones["s_buf_high"]

    bull_reclaim = bool(
        prev["low"] <= prev["DONCHIAN_LOW"] and price > row["DONCHIAN_LOW"]
    )
    bear_reject = bool(
        prev["high"] >= prev["DONCHIAN_HIGH"] and price < row["DONCHIAN_HIGH"]
    )

    atr_ok = bool(0.50 <= float(row["ATR_RATIO"]) <= 2.50)

    macd_bull = bool(row["MACD_HIST"] > 0 or row["MACD_HIST_CHANGE"] > 0)
    macd_bear = bool(row["MACD_HIST"] < 0 or row["MACD_HIST_CHANGE"] < 0)

    rsi_buy_ok = bool(32 <= float(row["RSI"]) <= 75)
    rsi_sell_ok = bool(25 <= float(row["RSI"]) <= 68)

    ema_bull = bool(row["EMA9"] > row["EMA21"])
    ema_bear = bool(row["EMA9"] < row["EMA21"])

    lap_bull = bool(row["LAP"] > 0)
    lap_bear = bool(row["LAP"] < 0)

    bb_bull = bool(price > row["BB_MID"])
    bb_bear = bool(price < row["BB_MID"])

    laminar_ok = bool(row["LAMINAR"] > 0.035)

    tube_width = max(zones["s_buf_low"] - zones["d_buf_high"], 0.01)
    distance_to_supply = max(zones["s_buf_low"] - price, 0.01)
    distance_to_demand = max(price - zones["d_buf_high"], 0.01)

    velocity = abs(float(df_1m["close"].diff().iloc[-1]))
    k_laminar = max(float(row["LAMINAR"]), 0.01)

    graetz_long = (velocity * tube_width**2) / (k_laminar * distance_to_supply)
    graetz_short = (velocity * tube_width**2) / (k_laminar * distance_to_demand)

    pullback_long = bool(
        bias_1h == "BULLISH"
        and price > row["EMA21"]
        and price <= row["EMA9"] + atr_now * 0.60
    )

    pullback_short = bool(
        bias_1h == "BEARISH"
        and price < row["EMA21"]
        and price >= row["EMA9"] - atr_now * 0.60
    )

    smc_long = get_smc_profile(df_5m, "BUY", micro_zones, df_swings=df_1m)
    smc_short = get_smc_profile(df_5m, "SELL", micro_zones, df_swings=df_1m)
    min_smc_score = (
        CONSOLIDATION_MIN_SMC_SCORE
        if context_4h == "CONSOLIDATING"
        else MIN_SMC_SCORE
    )

    smc_long_ok = bool(smc_long["score"] >= min_smc_score)
    smc_short_ok = bool(smc_short["score"] >= min_smc_score)
    setup_long = (
        near_demand
        or macro_near_demand
        or bull_reclaim
        or pullback_long
        or ema_bull
        or graetz_long > graetz_short
        or smc_long["sweep"]
        or smc_long_ok
    )
    setup_short = (
        near_supply
        or macro_near_supply
        or bear_reject
        or pullback_short
        or ema_bear
        or graetz_short > graetz_long
        or smc_short["sweep"]
        or smc_short_ok
    )

    buy_score = sum(
        [
            ema_bull,
            lap_bull,
            laminar_ok,
            setup_long,
            graetz_long > graetz_short,
            macd_bull,
            rsi_buy_ok,
            atr_ok,
            bb_bull,
            bias_1h == "BULLISH",
            smc_long_ok,
            smc_long["choch"],
            smc_long["displacement"],
            smc_long["fvg_near"],
            row_5m["EMA9"] > row_5m["EMA21"],
        ]
    )

    sell_score = sum(
        [
            ema_bear,
            lap_bear,
            laminar_ok,
            setup_short,
            graetz_short > graetz_long,
            macd_bear,
            rsi_sell_ok,
            atr_ok,
            bb_bear,
            bias_1h == "BEARISH",
            smc_short_ok,
            smc_short["choch"],
            smc_short["displacement"],
            smc_short["fvg_near"],
            row_5m["EMA9"] < row_5m["EMA21"],
        ]
    )

    if bias_4h == "BULLISH":
        buy_score += 0.5
    elif bias_4h == "BEARISH":
        sell_score += 0.5

    score_session_mult = session_mult if session_mult != 0 or not ALLOW_ALL_SESSIONS else 1.0
    buy_score *= score_session_mult
    sell_score *= score_session_mult

    state = get_market_state(state_row)

    candle_time_5m = str(df_5m.index[-1])
    direction_5m_info = get_5m_direction(df_5m)
    direction_5m = direction_5m_info["direction"]

    signal = "NO TRADE"
    success = None
    entry = None
    sl = None
    tp = None
    tp1 = None
    rr_used = None
    exit_rule = "WAIT"
    smc_used = None
    best = None

    tradable_session = session_mult != 0 or ALLOW_ALL_SESSIONS
    zone_tag = None

    if tradable_session and context_4h != "TOO_VOLATILE":
        direction = None

        # 3M/5M or 1M/5M supply & demand (micro_zones picks TF by mode)
        if near_demand and buy_score >= sell_score and _direction_5m_allows("BUY", direction_5m):
            direction = "BUY"
            zone_tag = f"{micro_zones['demand_tf']} demand"
        elif near_supply and sell_score >= buy_score and _direction_5m_allows("SELL", direction_5m):
            direction = "SELL"
            zone_tag = f"{micro_zones['supply_tf']} supply"
        elif setup_long and buy_score >= sell_score + 0.5 and _direction_5m_allows("BUY", direction_5m):
            direction = "BUY"
            zone_tag = f"{micro_zones['demand_tf']} setup"
        elif setup_short and sell_score >= buy_score + 0.5 and _direction_5m_allows("SELL", direction_5m):
            direction = "SELL"
            zone_tag = f"{micro_zones['supply_tf']} setup"

        if direction is not None:
            smc = smc_long if direction == "BUY" else smc_short
            entry = price
            test_sl = _local_micro_sl(direction, entry, atr_now, micro_zones, smc)

            if direction == "BUY" and test_sl < entry:
                best = choose_minute_rr(direction, entry, test_sl, state, atr_now, micro_zones)
            elif direction == "SELL" and test_sl > entry:
                best = choose_minute_rr(direction, entry, test_sl, state, atr_now, micro_zones)
            else:
                test_sl = (
                    entry - atr_now * MINUTE_SL_ATR
                    if direction == "BUY"
                    else entry + atr_now * MINUTE_SL_ATR
                )
                best = choose_minute_rr(direction, entry, test_sl, state, atr_now, micro_zones)

            if best is not None:
                signal = direction
                success = best["mc_prob"]
                sl = test_sl
                tp = best["tp"]
                tp1 = best.get("tp1")
                rr_used = best["rr"]
                smc_used = smc
                exit_rule = build_exit_rule(direction, entry, sl, tp)

    active_smc = smc_used or (smc_long if buy_score >= sell_score else smc_short)
    fvg = active_smc["fvg"]
    decision = (
        "TAKE BUY"
        if signal == "BUY"
        else "TAKE SELL"
        if signal == "SELL"
        else "NO TRADE"
    )
    setup_direction = signal if signal in ("BUY", "SELL") else (
        "BUY" if buy_score >= sell_score else "SELL"
    )

    static_mc = best.get("static_mc_prob") if best is not None else None
    markov_mc = float(best["mc_prob"]) if best is not None else None
    model = markov_store.get()
    markov_ctx = markov_features(
        state,
        setup_direction,
        model.get("transition_matrix", {}),
        model.get("state_win_rates", {}),
        markov_mc if markov_mc is not None else 0.0,
        float(static_mc if static_mc is not None else markov_mc or 0),
    )
    if markov_mc is None:
        markov_ctx["markov_mc_prob"] = None
        markov_ctx["markov_mc_delta"] = None
        markov_ctx["static_mc_prob"] = None

    exec_tf = "5M" if TRADE_5M_MODE else "1M"

    if signal == "BUY":
        action = f"{exec_tf} long at {zone_label} zone — dual TP 1R/2R."
        mk_delta = markov_ctx.get("markov_mc_delta")
        mk_suffix = (
            f" (Markov {markov_ctx.get('markov_mc_prob')}% Δ{mk_delta:+.1f})"
            if mk_delta is not None
            else ""
        )
        reason = (
            f"{exec_tf} BUY @ {zone_tag or micro_zones['demand_tf'] + ' demand'}, {state}, "
            f"buy_score={buy_score:.1f}, 5M {direction_5m_info['movement']}, "
            f"RR 1:{rr_used}, TP1={tp1}, TP2={tp}, MC {success}%{mk_suffix}."
        )
    elif signal == "SELL":
        action = f"{exec_tf} short at {zone_label} zone — dual TP 1R/2R."
        mk_delta = markov_ctx.get("markov_mc_delta")
        mk_suffix = (
            f" (Markov {markov_ctx.get('markov_mc_prob')}% Δ{mk_delta:+.1f})"
            if mk_delta is not None
            else ""
        )
        reason = (
            f"{exec_tf} SELL @ {zone_tag or micro_zones['supply_tf'] + ' supply'}, {state}, "
            f"sell_score={sell_score:.1f}, 5M {direction_5m_info['movement']}, "
            f"RR 1:{rr_used}, TP1={tp1}, TP2={tp}, MC {success}%{mk_suffix}."
        )
    else:
        action = f"Wait — not at {zone_label} zone, session filter, or volatility blocked."
        if tradable_session and context_4h != "TOO_VOLATILE" and not near_demand and not near_supply:
            reason = (
                f"No zone touch — demand {micro_zones['demand_tf']} / supply {micro_zones['supply_tf']}, "
                f"5M {direction_5m_info['movement']}"
            )
        else:
            reason = get_no_trade_reason(session_mult, session_name, context_4h)

    return {
        "LIVE_TIME": ny_time.strftime("%Y-%m-%d %H:%M:%S"),
        "CANDLE_TIME": candle_time,
        "CANDLE_TIME_5M": candle_time_5m,
        "EXEC_CADENCE": "5m" if TRADE_5M_MODE else "1m",
        "ZONE_TFS": zone_label,
        "DIRECTION_5M": direction_5m,
        "MOVEMENT_5M": direction_5m_info["movement"],
        "CONFIDENCE_5M": direction_5m_info["confidence"],
        "DIRECTION_5M_REASON": direction_5m_info["reason"],
        "LIVE_PRICE": round(price, 2),
        "ATR": round(atr_now, 4),
        "SIGNAL": signal,
        "DECISION": decision,
        "ACTION": action,
        "REASON": reason,
        "SESSION": session_name,
        "SETUP_DIRECTION": setup_direction,
        "BUY_SCORE": round(buy_score, 2),
        "SELL_SCORE": round(sell_score, 2),
        "BIAS_1H": bias_1h,
        "BIAS_4H": bias_4h,
        "CONTEXT_4H": context_4h,
        "STATE": state,
        "SUCCESS_PERCENT": success,
        "MARKOV_MC_PROB": markov_ctx.get("markov_mc_prob"),
        "MARKOV_PERSIST": markov_ctx.get("markov_persist"),
        "MARKOV_BULL_PROB": markov_ctx.get("markov_bull_prob"),
        "MARKOV_BEAR_PROB": markov_ctx.get("markov_bear_prob"),
        "MARKOV_MC_DELTA": markov_ctx.get("markov_mc_delta"),
        "STATE_WIN_RATE": markov_ctx.get("state_win_rate"),
        "STATIC_MC_PROB": static_mc,
        "RR": None if rr_used is None else f"1:{int(rr_used)}",
        "ENTRY": None if entry is None else round(entry, 2),
        "SL": None if sl is None else round(sl, 2),
        "TP": None if tp is None else round(tp, 2),
        "TP1": None if tp1 is None else round(tp1, 2),
        "TP2": None if best is None else (None if best.get("tp2") is None else round(best["tp2"], 2)),
        "TP3": None if best is None else (None if best.get("tp3") is None else round(best["tp3"], 2)),
        "DEMAND_TF": micro_zones["demand_tf"],
        "SUPPLY_TF": micro_zones["supply_tf"],
        "MACRO_DEMAND_TF": zones["demand_tf"],
        "MACRO_SUPPLY_TF": zones["supply_tf"],
        "MICRO_S_BUF_LOW": round(micro_zones["s_buf_low"], 2),
        "MICRO_D_BUF_HIGH": round(micro_zones["d_buf_high"], 2),
        "SMC_SCORE": active_smc["score"],
        "SMC_READ": get_smc_read(active_smc, min_smc_score),
        "SMC_SWEEP": active_smc["sweep"],
        "SMC_MSS": active_smc["mss"],
        "SMC_CHOCH": active_smc["choch"],
        "SMC_DISPLACEMENT": active_smc["displacement"],
        "SMC_FVG": None
        if fvg is None
        else f"{fvg['type']} {round(fvg['low'], 2)}-{round(fvg['high'], 2)}",
        "EXIT": exit_rule,
    }


def build_stacked_trade(sig: dict, live_price: float) -> dict:
    """Build a stacked trade at live price — same zone-based risk, fixed 1:2 RR."""
    direction = sig["SIGNAL"]
    atr = float(sig.get("ATR") or abs(float(sig["ENTRY"]) - float(sig["SL"])) or 1.0)
    state = sig.get("STATE", "CHOP")
    entry = float(live_price)

    parent_entry = float(sig["ENTRY"])
    parent_sl = float(sig["SL"])
    risk = abs(parent_entry - parent_sl)
    if risk <= 0:
        risk = atr * MINUTE_SL_ATR

    test_sl = entry - risk if direction == "BUY" else entry + risk
    micro_zones = {
        "s_buf_low": sig.get("MICRO_S_BUF_LOW"),
        "d_buf_high": sig.get("MICRO_D_BUF_HIGH"),
    }
    if micro_zones["s_buf_low"] is None or micro_zones["d_buf_high"] is None:
        micro_zones = None
    best = choose_minute_rr(direction, entry, test_sl, state, atr, micro_zones)

    out = dict(sig)
    out["ENTRY"] = round(entry, 2)
    out["SL"] = round(test_sl, 2)
    out["TP"] = best["tp"]
    out["TP1"] = best.get("tp1")
    out["TP2"] = best.get("tp2")
    out["TP3"] = best.get("tp3")
    rr_int = best.get("rr_int", int(best["rr"]))
    out["RR"] = f"1:{rr_int}"
    out["SUCCESS_PERCENT"] = best["mc_prob"]
    out["LIVE_PRICE"] = round(entry, 2)
    out["LOG_PRICE_1M"] = round(entry, 2)
    return out


def signal_once() -> dict:
    df_1m, df_3m, df_5m, df_15m, df_1h, df_4h = _fetch_all_timeframes()
    return _compute_signal(df_1m, df_3m, df_5m, df_15m, df_1h, df_4h)


def run_signal_cycle() -> dict:
    df_1m, df_3m, df_5m, df_15m, df_1h, df_4h = _fetch_all_timeframes()
    signal = _compute_signal(df_1m, df_3m, df_5m, df_15m, df_1h, df_4h)
    signal["LOG_TIME_1M"] = str(df_1m.index[-1])
    signal["LOG_TIME_3M"] = str(df_3m.index[-1])
    signal["LOG_TIME_5M"] = str(df_5m.index[-1])
    last_1m = df_1m.iloc[-1]
    signal["LOG_PRICE_1M"] = round(float(last_1m["close"]), 2)
    signal["LOG_HIGH_1M"] = round(float(last_1m["high"]), 2)
    signal["LOG_LOW_1M"] = round(float(last_1m["low"]), 2)
    return {
        "signal": signal,
        "candles": {
            "1m": _serialize_candles(df_1m),
            "5m": _serialize_candles(df_5m),
            "15m": _serialize_candles(df_15m),
            "1h": _serialize_candles(df_1h),
            "4h": _serialize_candles(df_4h),
        },
    }
