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
from analysis.market_hours import get_gold_market_status
from analysis.session import get_ny_time_and_session
from analysis.zones import get_local_zones, get_zones
from config import (
    ALLOW_ALL_SESSIONS,
    ATR_RATIO_MAX,
    ATR_RATIO_MIN,
    AVP_HARD_GATE,
    BARS_1M,
    BARS_3M,
    BARS_15M,
    BARS_1H,
    BARS_4H,
    BARS_5M,
    CONSOLIDATION_MIN_SMC_SCORE,
    DIRECTION_LOCK_5M,
    DISPLACEMENT_ATR_MULT,
    ENTRY_BRAIN,
    ENTRY_MIN_SCORE,
    ENTRY_SCORE_MARGIN_SETUP,
    ENTRY_SCORE_MARGIN_ZONE,
    FVG_LOOKBACK,
    MIN_SMC_SCORE,
    MINUTE_SL_ATR,
    NEWS_MC_ADJUST_MAX,
    PHYSICS_FILTER_ENABLED,
    PHYSICS_HARD_SYMBOLS,
    PHYSICS_SOFT_SCORE_EXTRA,
    PHYSICS_SOFT_SYMBOLS,
    PROFILE_LOOKBACK_BARS,
    PROFILE_MC_ADJUST_MAX,
    MAX_SL_ATR,
    REQUIRE_DISPLACEMENT_ENTRY,
    REQUIRE_ZONE_AND_PULLBACK,
    SL_ZONE_EXTRA,
    SMC_LOOKBACK,
    STRUCTURE_ENTRY_MIN_SCORE,
    STRUCTURE_SOFT_ENTRY,
    STRUCTURE_TP_MAX_RR,
    STRUCTURE_TP_MIN_RR,
    STRUCTURE_TP_MODE,
    TRADE_5M_MODE,
)
from data.market_data import fetch_ohlcv_with_pause as fetch_tv
from indicators.prepare import prepare_df
from strategy.monte_carlo import choose_minute_rr
from strategy.price_format import round_price

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


def build_exit_rule(
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    *,
    tp_source: Optional[str] = None,
    rr: Optional[float] = None,
) -> str:
    risk = abs(entry - sl)
    if risk <= 0:
        return "WAIT"
    # True 1R from risk distance (not midpoint — structure RR is often ≠ 2)
    one_r = entry + risk if direction == "BUY" else entry - risk
    rr_n = float(rr) if rr is not None else (abs(tp - entry) / risk)
    label = "STRUCTURE" if (tp_source or "").startswith("structure") else f"{rr_n:.1f}R"
    return (
        f"DUAL TP: TP1 1R ({one_r:.2f}) BOOK 50% LOT + BREAKEVEN | "
        f"TP2 {label} ({tp:.2f} · {rr_n:.2f}R) TRAIL SL PROFIT LOCK"
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


def _compute_signal(df_1m, df_3m, df_5m, df_15m, df_1h, df_4h, symbol: Optional[str] = None) -> dict:
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

    atr_ok = bool(ATR_RATIO_MIN <= float(row["ATR_RATIO"]) <= ATR_RATIO_MAX)

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
    # Location + confirmation. FVG/MSS count as confirmation when SMC already clears the floor
    # (bare EMA/Graetz alone caused late-range chop — keep those out).
    setup_long = near_demand or (
        (macro_near_demand or smc_long_ok or smc_long["sweep"])
        and (
            bull_reclaim
            or pullback_long
            or smc_long["displacement"]
            or smc_long["choch"]
            or smc_long["mss"]
            or smc_long["fvg_near"]
        )
    )
    setup_short = near_supply or (
        (macro_near_supply or smc_short_ok or smc_short["sweep"])
        and (
            bear_reject
            or pullback_short
            or smc_short["displacement"]
            or smc_short["choch"]
            or smc_short["mss"]
            or smc_short["fvg_near"]
        )
    )

    buy_bits = {
        "ema_trend": ema_bull,
        "laplacian": lap_bull,
        "laminar": laminar_ok,
        "setup": setup_long,
        "graetz": graetz_long > graetz_short,
        "macd": macd_bull,
        "rsi": rsi_buy_ok,
        "atr_ok": atr_ok,
        "bollinger": bb_bull,
        "bias_1h": bias_1h == "BULLISH",
        "smc_ok": smc_long_ok,
        "smc_choch": bool(smc_long["choch"]),
        "smc_disp": bool(smc_long["displacement"]),
        "smc_fvg": bool(smc_long["fvg_near"]),
        "ema_5m": bool(row_5m["EMA9"] > row_5m["EMA21"]),
    }
    sell_bits = {
        "ema_trend": ema_bear,
        "laplacian": lap_bear,
        "laminar": laminar_ok,
        "setup": setup_short,
        "graetz": graetz_short > graetz_long,
        "macd": macd_bear,
        "rsi": rsi_sell_ok,
        "atr_ok": atr_ok,
        "bollinger": bb_bear,
        "bias_1h": bias_1h == "BEARISH",
        "smc_ok": smc_short_ok,
        "smc_choch": bool(smc_short["choch"]),
        "smc_disp": bool(smc_short["displacement"]),
        "smc_fvg": bool(smc_short["fvg_near"]),
        "ema_5m": bool(row_5m["EMA9"] < row_5m["EMA21"]),
    }

    from analysis.concept_engine import weighted_score as _concept_weighted_score

    buy_score = _concept_weighted_score({}, buy_bits)
    sell_score = _concept_weighted_score({}, sell_bits)

    if bias_4h == "BULLISH":
        buy_score += 0.5
    elif bias_4h == "BEARISH":
        sell_score += 0.5

    # Previous-session FRVP (POC/VAH/VAL) + rolling blend
    from analysis.volume_profile import (
        build_dual_profile,
        profile_mc_adjust,
        profile_score_nudge,
    )

    vp = build_dual_profile(
        df_5m,
        price=price,
        atr=atr_now,
        lookback=PROFILE_LOOKBACK_BARS,
        symbol=symbol,
    )
    buy_nudge, sell_nudge = profile_score_nudge(vp)
    buy_score += buy_nudge
    sell_score += sell_nudge

    # Universe correlation soft lean (vs GOLD)
    try:
        from analysis.correlation_matrix import correlation_bias_for

        # provisional — refined once direction chosen; mild lean into scores now
        cb_buy = correlation_bias_for(str(symbol or "GOLD"), "BUY")
        cb_sell = correlation_bias_for(str(symbol or "GOLD"), "SELL")
        buy_score += float(cb_buy.get("score") or 0) * 0.5
        sell_score += float(cb_sell.get("score") or 0) * 0.5
    except Exception:
        pass

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
    concept_flags: dict = {}
    concept_q = None  # type: Optional[float]
    concept_reason = None  # type: Optional[str]

    tradable_session = session_mult != 0 or ALLOW_ALL_SESSIONS
    zone_tag = None
    brain_tag = "physics"
    ema_brain_sl = None  # type: Optional[float]
    crypto_firm_meta: dict = {}
    sym_u0 = str(symbol or "").upper().strip()
    from config import (
        is_forced_2r_brain,
        is_fib_entry_brain,
        is_trading_halted,
        symbol_entry_brain,
    )

    entry_brain = symbol_entry_brain(sym_u0)
    # XM-cost WF: hard-halt pairs with no survivable edge
    if is_trading_halted(sym_u0):
        entry_brain = "halted"
    # Lab hunt aliases
    if entry_brain in ("ema_tight",):
        entry_brain = "ema_20_50_tight"
    if entry_brain in ("ema_pull",):
        entry_brain = "ema_20_50_pull"
    use_fib = is_fib_entry_brain(entry_brain)
    use_ema_pull = entry_brain == "ema_20_50_pull"
    use_ema_tight = entry_brain == "ema_20_50_tight"
    use_sma_cross = entry_brain == "sma_20_50_cross"
    use_breakout = entry_brain == "breakout_20"
    use_don_100 = entry_brain in ("don_100", "donchian_100")
    use_pa_liq_engulf = entry_brain in (
        "pa_liq_engulf",
        "pa_liq_1h25_engulf5",
        "gold_pa_liq",
    )
    use_pa_opp_gold = entry_brain in (
        "pa_opp_gold",
        "usdjpy_opp_gold",
        "opp_gold_liq",
    )
    use_cta_10_40 = entry_brain in ("cta_10_40", "cta_10/40")
    use_cta_50_200 = entry_brain in ("cta_50_200", "cta_50/200")
    use_rsi = entry_brain in ("rsi", "rsi_mr")
    use_bb_15 = entry_brain in ("bb_1.5", "bb_1_5", "bb15")
    use_bb_2 = entry_brain in ("bb_2", "bb_2.0", "bb20")
    use_bb_25 = entry_brain in ("bb_2.5", "bb_2_5", "bb25")
    use_roc_120 = entry_brain in ("roc_120", "roc120")
    use_st_3 = entry_brain in ("st_3", "supertrend_3", "supertrend3")
    use_volmom = entry_brain in ("volmom", "vol_scaled_mom", "vol_mom")
    use_session_mix = entry_brain in ("session_mix", "adaptive_session_mix")
    use_circadian_kalman = entry_brain in (
        "circadian_kalman_z",
        "circadian_kalman",
        "adaptive_circadian_kalman_z",
    )
    use_entropy_tod = entry_brain in ("entropy_tod", "adaptive_entropy_tod")
    use_omni = entry_brain in ("omni", "omni_horizon", "omnihorizon")
    use_adaptive_v3 = use_session_mix or use_circadian_kalman or use_entropy_tod
    use_forced_2r_brain = is_forced_2r_brain(entry_brain)
    # Legacy aliases used downstream for TP_SOURCE tagging
    use_fib_gold = use_fib
    use_ema_2050 = use_ema_pull or use_ema_tight

    def _lab_exec_df():
        """Prefer symbol scan TF frame for lab brains; fall back 15m→5m→1m."""
        try:
            from config import symbol_scan_tfs

            pref = symbol_scan_tfs(sym_u0) or ()
        except Exception:
            pref = ()
        frame_map = {
            "1m": df_1m,
            "3m": df_3m,
            "5m": df_5m,
            "15m": df_15m,
            "1h": df_1h,
            "4h": df_4h,
        }
        for tf in pref:
            fr = frame_map.get(tf)
            if fr is not None and len(fr) >= 40:
                return fr, tf
        for tf, fr in (
            ("15m", df_15m),
            ("5m", df_5m),
            ("3m", df_3m),
            ("1m", df_1m),
        ):
            if fr is not None and len(fr) >= 40:
                return fr, tf
        return (df_5m if TRADE_5M_MODE else df_1m), "exec"

    if tradable_session and context_4h != "TOO_VOLATILE":
        direction = None
        vp_below = bool(isinstance(vp, dict) and vp.get("below_value"))
        vp_above = bool(isinstance(vp, dict) and vp.get("above_value"))

        # ── Fib 50–61.8 + liquidity sweep ──
        if use_fib:
            from strategy.fib_liq_brain import evaluate_fib_liq_sweep

            hit = None
            try:
                from config import fib_exec_tfs, symbol_scan_tfs

                allowed = set(fib_exec_tfs())
                mapped = symbol_scan_tfs(sym_u0)
                if mapped:
                    allowed |= set(mapped)
            except Exception:
                allowed = {"15m", "1h", "3m"}
            candidates = []
            frame_map = {
                "1m": df_1m,
                "3m": df_3m,
                "5m": df_5m,
                "15m": df_15m,
                "1h": df_1h,
            }
            for tf in ("3m", "15m", "5m", "1h", "1m"):
                if tf in allowed and frame_map.get(tf) is not None:
                    candidates.append((tf, frame_map[tf]))
            if df_15m is not None and not candidates:
                candidates.append(("15m_slot", df_15m))
            for _label, frame in candidates:
                hit = evaluate_fib_liq_sweep(frame)
                if hit is not None:
                    break
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                zone_tag = "Fib50-618 + liq sweep"
                brain_tag = "fib_liq"

        # ── EMA20/50 pull (standard or tight research knobs) ──
        elif use_ema_pull or use_ema_tight:
            from strategy.ema_pull_brain import evaluate_ema_2050_pull

            exec_df, _tf = _lab_exec_df()
            brain_key = "ema_20_50_tight" if use_ema_tight else "ema_20_50_pull"
            if use_ema_tight:
                pull_atr, disp_atr = 0.75, 0.70
            else:
                pull_atr, disp_atr = 1.0, 0.55
            try:
                from analysis.autopsy_brain import live_ema_knobs

                promo = live_ema_knobs(sym_u0, brain_key)
                if promo.get("pull_atr") is not None:
                    pull_atr = float(promo["pull_atr"])
                if promo.get("disp_atr") is not None:
                    disp_atr = float(promo["disp_atr"])
            except Exception:
                pass
            if use_ema_pull and not use_ema_tight:
                try:
                    from analysis.candle_edge import zenith_overlays_for
                    from analysis.autopsy_brain import get as autopsy_brain_get

                    if not autopsy_brain_get(sym_u0):
                        cov = zenith_overlays_for(sym_u0)
                        if int(cov.get("candle_n") or 0) >= 6:
                            pull_atr = float(cov.get("pull_atr") or pull_atr)
                            disp_atr = float(cov.get("disp_atr") or disp_atr)
                except Exception:
                    pass
            hit = evaluate_ema_2050_pull(
                exec_df, pull_atr=pull_atr, disp_atr=disp_atr
            )
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                zone_tag = "EMA20/50 tight" if use_ema_tight else "EMA20/50 pull"
                brain_tag = "ema_20_50_tight" if use_ema_tight else "ema_20_50"

        # ── SMA20/50 cross ──
        elif use_sma_cross:
            from strategy.sma_cross_brain import evaluate_sma_2050_cross

            exec_df, _tf = _lab_exec_df()
            hit = evaluate_sma_2050_cross(exec_df)
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                zone_tag = "SMA20/50 cross"
                brain_tag = "sma_20_50"

        # ── Donchian breakout ──
        elif use_breakout or use_don_100:
            from strategy.breakout_brain import evaluate_breakout

            exec_df, _tf = _lab_exec_df()
            lookback = 100 if use_don_100 else 20
            firm_sl_atr = 1.2
            use_firm = False
            try:
                from config import CRYPTO_FIRM_FORMULA_ENABLED
                from strategy.crypto_firm_formula import (
                    apply_firm_sl as _crypto_apply_firm_sl,
                    firm_spec as _crypto_firm_spec,
                    is_firm_crypto_symbol,
                    passes_firm_filters as _crypto_passes_firm,
                )

                if bool(CRYPTO_FIRM_FORMULA_ENABLED) and is_firm_crypto_symbol(sym_u0):
                    fs = _crypto_firm_spec(sym_u0) or {}
                    # Only apply crypto-firm stack on Donchian family specs
                    if str(fs.get("family") or "") in ("don_break", "breakout", "donchian"):
                        use_firm = True
                        firm_sl_atr = float(fs.get("sl_atr") or 1.2)
                        if fs.get("don_n"):
                            lookback = int(fs["don_n"])
            except Exception:
                use_firm = False

            hit = evaluate_breakout(exec_df, lookback=lookback, sl_atr=firm_sl_atr)
            if hit is not None and use_firm:
                ok_f, f_reason, f_meta = _crypto_passes_firm(exec_df, sym_u0)
                crypto_firm_meta = dict(f_meta or {})
                crypto_firm_meta["firm_gate"] = f_reason
                crypto_firm_meta["firm_kind"] = "crypto"
                if not ok_f:
                    hit = None
                    zone_tag = f"Don firm blocked ({f_reason})"
                else:
                    widened = _crypto_apply_firm_sl(
                        hit["signal"],
                        float(hit["entry"]),
                        float(hit.get("atr") or 0),
                        sym_u0,
                    )
                    if widened is not None:
                        hit["sl"] = float(widened)
                    crypto_firm_meta["firm_formula"] = True
                    crypto_firm_meta["sl_atr"] = firm_sl_atr
            # With-trend: Donchian only when 15m+1h agree (ETH trend-run firm)
            if hit is not None and use_firm:
                try:
                    from config import (
                        CRYPTO_TREND_SYNC_ENABLED,
                        CRYPTO_TREND_SYNC_MODE,
                        CRYPTO_TREND_SYNC_SYMBOLS,
                    )
                    from strategy.htf_trend_sync import allows_with_trend

                    sync_syms = {
                        str(s).upper() for s in (CRYPTO_TREND_SYNC_SYMBOLS or ())
                    }
                    if "ETHUSD" in sync_syms:
                        sync_syms.add("ETH")
                    if "BTCUSD" in sync_syms:
                        sync_syms.add("BTC")
                    if CRYPTO_TREND_SYNC_ENABLED and sym_u0 in sync_syms:
                        ok_t, t_reason, t_meta = allows_with_trend(
                            hit["signal"],
                            df_15m,
                            df_1h,
                            mode=CRYPTO_TREND_SYNC_MODE,
                        )
                        crypto_firm_meta.update(t_meta or {})
                        crypto_firm_meta["trend_sync"] = t_reason
                        if not ok_t:
                            hit = None
                            zone_tag = f"Don blocked (with-trend: {t_reason})"
                        else:
                            crypto_firm_meta["with_trend"] = True
                except Exception:
                    pass
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                if crypto_firm_meta.get("with_trend"):
                    zone_tag = f"Donchian {lookback} with-trend firm"
                elif use_firm:
                    zone_tag = f"Donchian {lookback} firm"
                else:
                    zone_tag = f"Donchian {lookback}" if use_don_100 else "Breakout 20"
                brain_tag = "don_100" if use_don_100 else "breakout_20"

        # ── USDJPY firm: JPY liq opp-to-GOLD + corr + Fourier ──
        elif use_pa_opp_gold:
            from config import (
                USDJPY_OPP_GOLD_FIRM_ENABLED,
                symbol_pa_bias_tfs,
                symbol_pa_liq_tf,
            )
            from strategy.usdjpy_opp_gold_firm import (
                evaluate_opp_gold_force,
                firm_spec as _opp_gold_spec,
                is_opp_gold_symbol,
                load_gold_frames,
            )

            if not bool(USDJPY_OPP_GOLD_FIRM_ENABLED) or not is_opp_gold_symbol(
                sym_u0
            ):
                zone_tag = "opp-gold disabled"
            else:
                fs = _opp_gold_spec()
                liq_tf = symbol_pa_liq_tf(sym_u0, default=str(fs.get("liq_tf") or "15m"))
                bias_a_tf, bias_b_tf = symbol_pa_bias_tfs(
                    sym_u0,
                    default=(
                        str(fs.get("gold_bias_a") or "15m"),
                        str(fs.get("gold_bias_b") or "5m"),
                    ),
                )
                frame_map = {
                    "1m": df_1m,
                    "3m": df_3m,
                    "5m": df_5m,
                    "15m": df_15m,
                }
                df_e = frame_map.get("1m")
                if df_e is None or len(df_e) < 60:
                    df_e = df_5m
                df_l = frame_map.get(liq_tf) or df_15m
                df_jpy_corr = frame_map.get(str(fs.get("corr_tf") or "15m")) or df_15m
                gold_frames = load_gold_frames(
                    need_tfs=(bias_a_tf, bias_b_tf, str(fs.get("corr_tf") or "15m")),
                    bars=220,
                )
                df_ga = gold_frames.get(bias_a_tf)
                df_gb = gold_frames.get(bias_b_tf)
                df_gc = gold_frames.get(str(fs.get("corr_tf") or "15m"))
                hit, og_reason, og_meta = evaluate_opp_gold_force(
                    df_e,
                    df_l,
                    df_ga,
                    df_gb,
                    df_gc,
                    df_jpy_corr,
                    entry_tf_min=1 if df_e is df_1m else 5,
                    spec=fs,
                )
                crypto_firm_meta.update(og_meta or {})
                crypto_firm_meta["firm_gate"] = og_reason
                if hit is not None:
                    direction = hit["signal"]
                    ema_brain_sl = float(hit["sl"])
                    zone_tag = (
                        f"JPY {liq_tf} liq opp-GOLD {bias_a_tf}+{bias_b_tf} "
                        f"corr={float(hit.get('corr_vs_gold') or 0):+.2f} "
                        f"HF={float(hit.get('fourier_hf') or 0):.2f} "
                        f"@ {float(hit.get('liq') or 0):.5g}"
                    )
                    brain_tag = "pa_opp_gold"
                    crypto_firm_meta["firm_formula"] = True
                    crypto_firm_meta["firm_kind"] = "usdjpy_opp_gold"
                else:
                    zone_tag = f"opp-gold blocked ({og_reason})"

        # ── PA: liq sweep → dual bias → 1m engulf (per-symbol TF-grid) ──
        elif use_pa_liq_engulf:
            from config import symbol_pa_bias_tfs, symbol_pa_liq_tf
            from strategy.pa_liq_engulf_brain import evaluate_pa_liq_engulf

            liq_tf = symbol_pa_liq_tf(sym_u0, default="3m")
            bias_a_tf, bias_b_tf = symbol_pa_bias_tfs(sym_u0, default=("15m", "5m"))
            frame_map = {
                "1m": df_1m,
                "3m": df_3m,
                "5m": df_5m,
                "15m": df_15m,
            }
            df_e = frame_map.get("1m")
            if df_e is None or len(df_e) < 60:
                df_e = df_5m
            df_l = frame_map.get(liq_tf)
            if df_l is None or len(df_l) < 80:
                df_l = df_3m if df_3m is not None and len(df_3m) >= 80 else df_5m
            df_ba = frame_map.get(bias_a_tf) or df_15m
            df_bb = frame_map.get(bias_b_tf) or df_5m
            hit = evaluate_pa_liq_engulf(
                df_e,
                df_l,
                df_ba,
                df_bb,
                entry_tf_min=1 if df_e is df_1m else 5,
            )
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                zone_tag = (
                    f"{liq_tf} liq / {bias_a_tf}+{bias_b_tf} bias "
                    f"@ {float(hit.get('liq') or 0):.5g} → 1m engulf"
                )
                brain_tag = "pa_liq_engulf"

        # ── Lab / deep-hunt winners: CTA / RSI / BB / ROC / Supertrend ──
        elif use_cta_10_40 or use_cta_50_200:
            from strategy.lab_brains import evaluate_cta_cross

            exec_df, _tf = _lab_exec_df()
            fast, slow = (50, 200) if use_cta_50_200 else (10, 40)
            hit = evaluate_cta_cross(exec_df, fast=fast, slow=slow)
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                zone_tag = f"CTA {fast}/{slow}"
                brain_tag = "cta_50_200" if use_cta_50_200 else "cta_10_40"

        elif use_rsi:
            from strategy.lab_brains import evaluate_rsi_mr

            exec_df, _tf = _lab_exec_df()
            hit = evaluate_rsi_mr(exec_df)
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                zone_tag = "RSI MR"
                brain_tag = "rsi"

        elif use_bb_15 or use_bb_2 or use_bb_25:
            from strategy.lab_brains import evaluate_bb_mr

            exec_df, _tf = _lab_exec_df()
            if use_bb_25:
                mult = 2.5
                brain_tag = "bb_2.5"
            elif use_bb_15:
                mult = 1.5
                brain_tag = "bb_1.5"
            else:
                mult = 2.0
                brain_tag = "bb_2"
            # Firm hunt winners — wider SL + Fourier(/dist) gates (crypto + FX)
            firm_sl_atr = 1.2
            use_firm = False
            firm_kind = None  # "crypto" | "fx"
            apply_firm_sl = None
            passes_firm_filters = None
            try:
                from config import CRYPTO_FIRM_FORMULA_ENABLED, FX_FIRM_FORMULA_ENABLED
                from strategy.crypto_firm_formula import (
                    apply_firm_sl as _crypto_apply_firm_sl,
                    firm_spec as _crypto_firm_spec,
                    is_firm_crypto_symbol,
                    passes_firm_filters as _crypto_passes_firm,
                )
                from strategy.fx_firm_formula import (
                    apply_firm_sl as _fx_apply_firm_sl,
                    firm_spec as _fx_firm_spec,
                    is_firm_fx_symbol,
                    passes_firm_filters as _fx_passes_firm,
                )

                if bool(CRYPTO_FIRM_FORMULA_ENABLED) and is_firm_crypto_symbol(sym_u0):
                    use_firm = True
                    firm_kind = "crypto"
                    apply_firm_sl = _crypto_apply_firm_sl
                    passes_firm_filters = _crypto_passes_firm
                    fs = _crypto_firm_spec(sym_u0) or {}
                elif bool(FX_FIRM_FORMULA_ENABLED) and is_firm_fx_symbol(sym_u0):
                    use_firm = True
                    firm_kind = "fx"
                    apply_firm_sl = _fx_apply_firm_sl
                    passes_firm_filters = _fx_passes_firm
                    fs = _fx_firm_spec(sym_u0) or {}
                else:
                    fs = {}

                if use_firm:
                    firm_sl_atr = float(fs.get("sl_atr") or 1.2)
                    if fs.get("bb_mult") is not None:
                        mult = float(fs["bb_mult"])
                        if abs(mult - 2.5) < 1e-9:
                            brain_tag = "bb_2.5"
                        elif abs(mult - 1.5) < 1e-9:
                            brain_tag = "bb_1.5"
                        else:
                            brain_tag = f"bb_{mult:g}"
            except Exception:
                use_firm = False
                firm_kind = None

            hit = evaluate_bb_mr(exec_df, std_mult=mult, sl_atr=firm_sl_atr)
            if hit is not None and use_firm and apply_firm_sl and passes_firm_filters:
                ok_f, f_reason, f_meta = passes_firm_filters(exec_df, sym_u0)
                crypto_firm_meta = dict(f_meta or {})
                crypto_firm_meta["firm_gate"] = f_reason
                crypto_firm_meta["firm_kind"] = firm_kind
                if not ok_f:
                    hit = None
                    zone_tag = f"BB firm blocked ({f_reason})"
                else:
                    widened = apply_firm_sl(
                        hit["signal"],
                        float(hit["entry"]),
                        float(hit.get("atr") or 0),
                        sym_u0,
                    )
                    if widened is not None:
                        hit["sl"] = float(widened)
                    crypto_firm_meta["firm_formula"] = True
                    crypto_firm_meta["sl_atr"] = firm_sl_atr
            # With-trend surf: BB pullbacks only when 15m+1h agree (ETH/BTC)
            if hit is not None:
                try:
                    from config import (
                        CRYPTO_TREND_SYNC_ENABLED,
                        CRYPTO_TREND_SYNC_MODE,
                        CRYPTO_TREND_SYNC_SYMBOLS,
                    )
                    from strategy.htf_trend_sync import allows_with_trend

                    sync_syms = {
                        str(s).upper() for s in (CRYPTO_TREND_SYNC_SYMBOLS or ())
                    }
                    # aliases
                    if "ETHUSD" in sync_syms:
                        sync_syms.add("ETH")
                    if "BTCUSD" in sync_syms:
                        sync_syms.add("BTC")
                    if CRYPTO_TREND_SYNC_ENABLED and sym_u0 in sync_syms:
                        ok_t, t_reason, t_meta = allows_with_trend(
                            hit["signal"],
                            df_15m,
                            df_1h,
                            mode=CRYPTO_TREND_SYNC_MODE,
                        )
                        crypto_firm_meta.update(t_meta or {})
                        crypto_firm_meta["trend_sync"] = t_reason
                        if not ok_t:
                            hit = None
                            zone_tag = f"BB blocked (with-trend: {t_reason})"
                        else:
                            crypto_firm_meta["with_trend"] = True
                except Exception:
                    pass
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                if crypto_firm_meta.get("with_trend"):
                    zone_tag = (
                        f"BB {mult:g}σ with-trend"
                        + (" firm" if use_firm else "")
                    )
                else:
                    zone_tag = f"BB {mult:g}σ firm" if use_firm else f"BB {mult:g}σ"

        elif use_roc_120:
            from strategy.lab_brains import evaluate_roc_cross

            exec_df, _tf = _lab_exec_df()
            hit = evaluate_roc_cross(exec_df, period=120)
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                zone_tag = "ROC 120"
                brain_tag = "roc_120"

        elif use_st_3:
            from strategy.lab_brains import evaluate_supertrend

            exec_df, _tf = _lab_exec_df()
            hit = evaluate_supertrend(exec_df, mult=3.0)
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                zone_tag = "Supertrend 3"
                brain_tag = "st_3"

        elif use_volmom:
            from strategy.lab_brains import evaluate_vol_scaled_mom

            exec_df, _tf = _lab_exec_df()
            hit = evaluate_vol_scaled_mom(exec_df, symbol=sym_u0)
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                zone_tag = "Vol-scaled mom"
                brain_tag = "volmom"

        elif use_omni:
            from strategy.omni_live import evaluate_omni

            try:
                from config import symbol_scan_tfs

                pref_tfs = [t for t in (symbol_scan_tfs(sym_u0) or ()) if t in ("3m", "5m")]
                if not pref_tfs:
                    pref_tfs = ["5m"]
            except Exception:
                pref_tfs = ["5m"]
            frame_map_o = {
                "1m": df_1m,
                "3m": df_3m,
                "5m": df_5m,
                "15m": df_15m,
            }
            hit = None
            exec_tf = pref_tfs[0]
            for tf in pref_tfs:
                fr = frame_map_o.get(tf)
                if fr is None or len(fr) < 160:
                    continue
                hit = evaluate_omni(fr, symbol=sym_u0, tf=tf)
                exec_tf = tf
                if hit is not None:
                    break
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                brain_tag = str(hit.get("brain_active") or "omni")
                zone_tag = f"Omni {brain_tag} @{exec_tf}"

        elif use_adaptive_v3:
            from strategy.adaptive_v3_brains import evaluate_adaptive_v3

            # Honor configured scan TFs (3m/5m) — omni desk uses both
            try:
                from config import symbol_scan_tfs

                pref_tfs = [
                    t
                    for t in (symbol_scan_tfs(sym_u0) or ())
                    if t in ("1m", "3m", "5m", "15m")
                ]
                if not pref_tfs:
                    pref_tfs = ["5m"]
            except Exception:
                pref_tfs = ["5m"]
            frame_map_a = {
                "1m": df_1m,
                "3m": df_3m,
                "5m": df_5m,
                "15m": df_15m,
            }
            hit = None
            exec_tf = "5m"
            tried = []
            for tf in (pref_tfs or ["5m"]):
                fr = frame_map_a.get(tf)
                if fr is None or len(fr) < 80:
                    continue
                tried.append(tf)
                hit = evaluate_adaptive_v3(fr, entry_brain, symbol=sym_u0)
                exec_tf = tf
                if hit is not None:
                    break
            if hit is None and not tried:
                exec_df, exec_tf = _lab_exec_df()
                hit = evaluate_adaptive_v3(exec_df, entry_brain, symbol=sym_u0)
            if hit is not None:
                direction = hit["signal"]
                ema_brain_sl = float(hit["sl"])
                brain_tag = str(hit.get("brain_active") or entry_brain)
                zone_tag = f"Adaptive v3 {brain_tag} @{exec_tf}"

        # Zone / SMC / physics path — skipped for forced-2R brains
        if not use_forced_2r_brain:
            zone_buy_ok = (
                near_demand
                and buy_score >= ENTRY_MIN_SCORE
                and buy_score >= sell_score + ENTRY_SCORE_MARGIN_ZONE
                and (smc_long_ok or buy_score >= ENTRY_MIN_SCORE + 1.5)
                and _direction_5m_allows("BUY", direction_5m)
                and not (vp_below and not smc_long_ok)
            )
            zone_sell_ok = (
                near_supply
                and sell_score >= ENTRY_MIN_SCORE
                and sell_score >= buy_score + ENTRY_SCORE_MARGIN_ZONE
                and (smc_short_ok or sell_score >= ENTRY_MIN_SCORE + 1.5)
                and _direction_5m_allows("SELL", direction_5m)
                and not (vp_above and not smc_short_ok)
            )
            setup_buy_ok = (
                setup_long
                and (near_demand or smc_long_ok)
                and buy_score >= ENTRY_MIN_SCORE
                and buy_score >= sell_score + ENTRY_SCORE_MARGIN_SETUP
                and _direction_5m_allows("BUY", direction_5m)
            )
            setup_sell_ok = (
                setup_short
                and (near_supply or smc_short_ok)
                and sell_score >= ENTRY_MIN_SCORE
                and sell_score >= buy_score + ENTRY_SCORE_MARGIN_SETUP
                and _direction_5m_allows("SELL", direction_5m)
            )

            if AVP_HARD_GATE:
                if zone_buy_ok or setup_buy_ok:
                    if vp_above and not (near_demand or smc_long_ok):
                        zone_buy_ok = False
                        setup_buy_ok = False
                if zone_sell_ok or setup_sell_ok:
                    if vp_below and not (near_supply or smc_short_ok):
                        zone_sell_ok = False
                        setup_sell_ok = False
                if isinstance(vp, dict) and vp.get("in_value") and not vp_above and not vp_below:
                    if setup_buy_ok and not (near_demand and smc_long_ok):
                        setup_buy_ok = False
                    if setup_sell_ok and not (near_supply and smc_short_ok):
                        setup_sell_ok = False

            candle_disp_buy = bool(
                row["close"] > row["open"]
                and float(row["BODY"]) >= atr_now * DISPLACEMENT_ATR_MULT
                and float(row["BODY_RATIO"]) >= 0.55
            )
            candle_disp_sell = bool(
                row["close"] < row["open"]
                and float(row["BODY"]) >= atr_now * DISPLACEMENT_ATR_MULT
                and float(row["BODY_RATIO"]) >= 0.55
            )
            disp_buy = bool(smc_long["displacement"] or candle_disp_buy or bull_reclaim)
            disp_sell = bool(smc_short["displacement"] or candle_disp_sell or bear_reject)

            sym_u = str(symbol or "").upper().strip()
            from analysis.structure_targets import structure_enabled_for as _struct_on

            struct_book = _struct_on(symbol)

            if REQUIRE_DISPLACEMENT_ENTRY:
                if not disp_buy:
                    zone_buy_ok = False
                    setup_buy_ok = False
                if not disp_sell:
                    zone_sell_ok = False
                    setup_sell_ok = False
            if REQUIRE_ZONE_AND_PULLBACK and struct_book:
                if not (near_demand or pullback_long):
                    zone_buy_ok = False
                    setup_buy_ok = False
                if not (near_supply or pullback_short):
                    zone_sell_ok = False
                    setup_sell_ok = False

            physics_buy_ok = laminar_ok and graetz_long > graetz_short
            physics_sell_ok = laminar_ok and graetz_short > graetz_long
            hard_phys = PHYSICS_FILTER_ENABLED and (
                (not PHYSICS_HARD_SYMBOLS and not PHYSICS_SOFT_SYMBOLS)
                or sym_u in PHYSICS_HARD_SYMBOLS
            )
            soft_phys = PHYSICS_FILTER_ENABLED and sym_u in PHYSICS_SOFT_SYMBOLS

            if hard_phys:
                if not physics_buy_ok:
                    zone_buy_ok = False
                    setup_buy_ok = False
                if not physics_sell_ok:
                    zone_sell_ok = False
                    setup_sell_ok = False
            elif soft_phys:
                extra = PHYSICS_SOFT_SCORE_EXTRA
                if not physics_buy_ok and buy_score < ENTRY_MIN_SCORE + extra:
                    zone_buy_ok = False
                    setup_buy_ok = False
                if not physics_sell_ok and sell_score < ENTRY_MIN_SCORE + extra:
                    zone_sell_ok = False
                    setup_sell_ok = False

            if zone_buy_ok:
                direction = "BUY"
                zone_tag = f"{micro_zones['demand_tf']} demand"
            elif zone_sell_ok:
                direction = "SELL"
                zone_tag = f"{micro_zones['supply_tf']} supply"
            elif setup_buy_ok:
                direction = "BUY"
                zone_tag = f"{micro_zones['demand_tf']} setup"
            elif setup_sell_ok:
                direction = "SELL"
                zone_tag = f"{micro_zones['supply_tf']} setup"

            brain_tag = "physics"
            if direction is None and STRUCTURE_SOFT_ENTRY and struct_book:
                physics_buy = (not hard_phys) or physics_buy_ok
                physics_sell = (not hard_phys) or physics_sell_ok
                if soft_phys and not physics_buy_ok:
                    physics_buy = buy_score >= STRUCTURE_ENTRY_MIN_SCORE + PHYSICS_SOFT_SCORE_EXTRA
                if soft_phys and not physics_sell_ok:
                    physics_sell = sell_score >= STRUCTURE_ENTRY_MIN_SCORE + PHYSICS_SOFT_SCORE_EXTRA
                struct_buy_ok = (
                    bias_1h == "BULLISH"
                    and near_demand
                    and pullback_long
                    and disp_buy
                    and buy_score >= STRUCTURE_ENTRY_MIN_SCORE
                    and buy_score >= sell_score + 0.75
                    and physics_buy
                    and atr_ok
                    and _direction_5m_allows("BUY", direction_5m)
                )
                struct_sell_ok = (
                    bias_1h == "BEARISH"
                    and near_supply
                    and pullback_short
                    and disp_sell
                    and sell_score >= STRUCTURE_ENTRY_MIN_SCORE
                    and sell_score >= buy_score + 0.75
                    and physics_sell
                    and atr_ok
                    and _direction_5m_allows("SELL", direction_5m)
                )
                if struct_buy_ok:
                    direction = "BUY"
                    zone_tag = f"{micro_zones['demand_tf']} structure"
                    brain_tag = "structure"
                elif struct_sell_ok:
                    direction = "SELL"
                    zone_tag = f"{micro_zones['supply_tf']} structure"
                    brain_tag = "structure"

        if direction is not None:
            smc = smc_long if direction == "BUY" else smc_short
            entry = price
            if use_forced_2r_brain and ema_brain_sl is not None:
                test_sl = float(ema_brain_sl)
            else:
                test_sl = _local_micro_sl(direction, entry, atr_now, micro_zones, smc)

            # Structure TP candidate (preferred exit for GOLD/USOIL)
            struct_tp_price = None
            try:
                from analysis.structure_targets import (
                    htf_extremes,
                    structure_enabled_for,
                    structure_take_profit,
                )

                if structure_enabled_for(symbol) and not use_forced_2r_brain:
                    htf_hi, htf_lo = htf_extremes(df_1h, lookback=8)
                    pack = structure_take_profit(
                        direction,
                        entry,
                        test_sl
                        if (
                            (direction == "BUY" and test_sl < entry)
                            or (direction == "SELL" and test_sl > entry)
                        )
                        else (
                            entry - atr_now * MINUTE_SL_ATR
                            if direction == "BUY"
                            else entry + atr_now * MINUTE_SL_ATR
                        ),
                        atr_now,
                        zones=micro_zones,
                        swing_high=smc.get("last_swing_high"),
                        swing_low=smc.get("last_swing_low"),
                        htf_high=htf_hi,
                        htf_low=htf_lo,
                        min_rr=STRUCTURE_TP_MIN_RR,
                        max_rr=STRUCTURE_TP_MAX_RR,
                    )
                    if pack:
                        struct_tp_price = float(pack["tp"])
            except Exception:
                struct_tp_price = None

            rr_kwargs = dict(
                symbol=symbol,
                structure_tp=None if use_forced_2r_brain else struct_tp_price,
                structure_mode="off" if use_forced_2r_brain else STRUCTURE_TP_MODE,
            )
            if use_forced_2r_brain:
                from config import symbol_rr

                # Pin TP to research-backed per-pair RR (was hard 1:2)
                rr_kwargs["forced_rr"] = symbol_rr(symbol, default=2.0)
            if direction == "BUY" and test_sl < entry:
                best = choose_minute_rr(
                    direction, entry, test_sl, state, atr_now, micro_zones, **rr_kwargs
                )
            elif direction == "SELL" and test_sl > entry:
                best = choose_minute_rr(
                    direction, entry, test_sl, state, atr_now, micro_zones, **rr_kwargs
                )
            else:
                test_sl = (
                    entry - atr_now * MINUTE_SL_ATR
                    if direction == "BUY"
                    else entry + atr_now * MINUTE_SL_ATR
                )
                best = choose_minute_rr(
                    direction, entry, test_sl, state, atr_now, micro_zones, **rr_kwargs
                )

            if best is not None:
                from analysis.concept_engine import build_concept_flags, passes_concept_gate

                avp_aligned_pre = False
                if isinstance(vp, dict) and vp.get("available"):
                    if direction == "BUY":
                        avp_aligned_pre = bool(vp.get("below_value") or vp.get("in_value"))
                    else:
                        avp_aligned_pre = bool(vp.get("above_value") or vp.get("in_value"))

                if best.get("tp_source") == "structure" and not use_forced_2r_brain:
                    brain_tag = "structure" if brain_tag == "physics" else f"{brain_tag}+structure_tp"

                concept_flags = build_concept_flags(
                    direction=direction,
                    ema_align=ema_bull if direction == "BUY" else ema_bear,
                    lap_align=lap_bull if direction == "BUY" else lap_bear,
                    laminar_ok=laminar_ok,
                    setup=setup_long if direction == "BUY" else setup_short,
                    graetz_align=(graetz_long > graetz_short) if direction == "BUY" else (graetz_short > graetz_long),
                    macd_align=macd_bull if direction == "BUY" else macd_bear,
                    rsi_ok=rsi_buy_ok if direction == "BUY" else rsi_sell_ok,
                    atr_ok=atr_ok,
                    bb_align=bb_bull if direction == "BUY" else bb_bear,
                    bias_1h_align=(bias_1h == "BULLISH") if direction == "BUY" else (bias_1h == "BEARISH"),
                    smc_ok=smc_long_ok if direction == "BUY" else smc_short_ok,
                    smc_choch=bool((smc_long if direction == "BUY" else smc_short)["choch"]),
                    smc_disp=bool((smc_long if direction == "BUY" else smc_short)["displacement"]),
                    smc_fvg=bool((smc_long if direction == "BUY" else smc_short)["fvg_near"]),
                    ema_5m_align=(
                        bool(row_5m["EMA9"] > row_5m["EMA21"])
                        if direction == "BUY"
                        else bool(row_5m["EMA9"] < row_5m["EMA21"])
                    ),
                    near_zone=near_demand if direction == "BUY" else near_supply,
                    avp_aligned=avp_aligned_pre,
                    avp_in_value=bool(isinstance(vp, dict) and vp.get("in_value")),
                    bias_4h_align=(bias_4h == "BULLISH") if direction == "BUY" else (bias_4h == "BEARISH"),
                    state=state,
                    mc_prob=best.get("mc_prob"),
                    rr=best.get("rr"),
                    session_name=session_name,
                )
                if use_forced_2r_brain:
                    ok_concepts, concept_q, concept_reason = (
                        True,
                        1.0,
                        f"{brain_tag or entry_brain} — concept gate bypassed",
                    )
                else:
                    ok_concepts, concept_q, concept_reason = passes_concept_gate(concept_flags)
                if ok_concepts:
                    signal = direction
                    success = best["mc_prob"]
                    if use_forced_2r_brain:
                        # Research alts did not use a hard MC floor — keep tradable.
                        success = max(float(success or 0), 48.0)
                    sl = test_sl
                    tp = best["tp"]
                    tp1 = best.get("tp1")
                    rr_used = best["rr"]
                    smc_used = smc
                    exit_rule = build_exit_rule(
                        direction,
                        entry,
                        sl,
                        tp,
                        tp_source=(
                            "fib_liq"
                            if use_fib
                            else "ema_20_50_tight"
                            if use_ema_tight
                            else "ema_20_50"
                            if use_ema_pull
                            else "sma_20_50"
                            if use_sma_cross
                            else "breakout_20"
                            if use_breakout
                            else str(best.get("tp_source") or "")
                        ),
                        rr=float(rr_used) if rr_used is not None else None,
                    )
                else:
                    best = None
                    direction = None
                    zone_tag = None
                    if not use_forced_2r_brain:
                        brain_tag = "physics"

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

    # World-news variable: soft MC nudge only (physics MC kept in STATIC_MC_PROB)
    # Skip live news during backtest (would be lookahead + wall-clock contamination).
    news_ctx = {
        "available": False,
        "direction": "NEUTRAL",
        "score": 0.0,
        "confidence": 50.0,
        "summary": None,
        "mc_adjust": 0.0,
        "aligned": None,
    }
    try:
        from analysis.session import get_backtest_bar_time
        from data.gold_news import aggregate_world_news_bias

        if get_backtest_bar_time() is None:
            world = aggregate_world_news_bias()
            news_score = float(world.get("score") or 0.0)
            news_dir = world.get("direction") or "NEUTRAL"
            news_ctx.update(
                {
                    "available": bool(world.get("available")),
                    "direction": news_dir,
                    "score": news_score,
                    "confidence": float(world.get("confidence") or 50.0),
                    "summary": world.get("summary"),
                    "bullish_hubs": world.get("bullish_hubs"),
                    "bearish_hubs": world.get("bearish_hubs"),
                    "neutral_hubs": world.get("neutral_hubs"),
                }
            )
            max_adj = float(NEWS_MC_ADJUST_MAX)
            if signal in ("BUY", "SELL") and max_adj > 0 and news_ctx["available"]:
                # Positive news_score = gold-bullish world → helps BUY, hurts SELL
                raw = news_score * (1.0 if signal == "BUY" else -1.0)
                # Scale: score ~1.0 → about 1.2pp; capped by NEWS_MC_ADJUST_MAX
                adj = max(-max_adj, min(max_adj, raw * 1.2))
                news_ctx["mc_adjust"] = round(adj, 2)
                news_ctx["aligned"] = adj > 0.15
                if success is not None:
                    success = float(max(1.0, min(99.0, float(success) + adj)))
                if markov_ctx.get("markov_mc_prob") is not None:
                    markov_ctx["markov_mc_prob"] = round(
                        float(max(1.0, min(99.0, float(markov_ctx["markov_mc_prob"]) + adj))),
                        2,
                    )
                    if markov_ctx.get("static_mc_prob") is not None:
                        markov_ctx["markov_mc_delta"] = round(
                            float(markov_ctx["markov_mc_prob"]) - float(markov_ctx["static_mc_prob"]),
                            2,
                        )
    except Exception:
        pass

    # Volume profile soft MC nudge (STATIC_MC stays physics-only)
    vp_mc_adj = 0.0
    vp_aligned = None
    try:
        vp_mc_adj = profile_mc_adjust(vp, signal, float(PROFILE_MC_ADJUST_MAX))
        if signal in ("BUY", "SELL") and vp.get("available"):
            vp_aligned = vp_mc_adj > 0.1
            if success is not None and vp_mc_adj != 0.0:
                success = float(max(1.0, min(99.0, float(success) + vp_mc_adj)))
            if markov_ctx.get("markov_mc_prob") is not None and vp_mc_adj != 0.0:
                markov_ctx["markov_mc_prob"] = round(
                    float(max(1.0, min(99.0, float(markov_ctx["markov_mc_prob"]) + vp_mc_adj))),
                    2,
                )
                if markov_ctx.get("static_mc_prob") is not None:
                    markov_ctx["markov_mc_delta"] = round(
                        float(markov_ctx["markov_mc_prob"]) - float(markov_ctx["static_mc_prob"]),
                        2,
                    )
    except Exception:
        pass

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
        if concept_reason and concept_q is not None and concept_q < 0.5:
            reason = f"Concept gate — {concept_reason}"
        elif tradable_session and context_4h != "TOO_VOLATILE":
            prefer = "SELL" if sell_score >= buy_score else "BUY"
            bits = []
            if prefer == "SELL":
                if not near_supply:
                    bits.append("not at supply")
                if not smc_short_ok:
                    bits.append(f"SMC {smc_short['score']:.1f}/{min_smc_score:.1f}")
                if not setup_short:
                    bits.append("no short setup confirm")
                if vp_below and not (near_supply or smc_short_ok):
                    bits.append("AVP blocks sell in discount")
            else:
                if not near_demand:
                    bits.append("not at demand")
                if not smc_long_ok:
                    bits.append(f"SMC {smc_long['score']:.1f}/{min_smc_score:.1f}")
                if not setup_long:
                    bits.append("no long setup confirm")
                if vp_above and not (near_demand or smc_long_ok):
                    bits.append("AVP blocks buy in premium")
            detail = ", ".join(bits) if bits else "filters not met"
            lean = sell_score if prefer == "SELL" else buy_score
            reason = (
                f"No trade ({prefer} lean score {lean:.1f}) — {detail}; "
                f"VP={vp.get('location') if isinstance(vp, dict) else 'n/a'}, "
                f"5M {direction_5m_info['movement']}"
            )
        else:
            reason = get_no_trade_reason(session_mult, session_name, context_4h)

    out = {
        "LIVE_TIME": ny_time.strftime("%Y-%m-%d %H:%M:%S"),
        "CANDLE_TIME": candle_time,
        "CANDLE_TIME_5M": candle_time_5m,
        "EXEC_CADENCE": "5m" if TRADE_5M_MODE else "1m",
        "ZONE_TFS": zone_label,
        "DIRECTION_5M": direction_5m,
        "MOVEMENT_5M": direction_5m_info["movement"],
        "CONFIDENCE_5M": direction_5m_info["confidence"],
        "DIRECTION_5M_REASON": direction_5m_info["reason"],
        "LIVE_PRICE": round_price(price, symbol),
        "ATR": round(atr_now, 4),
        "ATR_1M": round(float(row["ATR"]), 4),
        "ATR_3M": round(float(df_3m.iloc[-1]["ATR"]), 4) if df_3m is not None and len(df_3m) else round(atr_now * 0.85, 4),
        "ATR_5M": round(float(row_5m["ATR"]), 4),
        "ATR_15M": round(float(df_15m.iloc[-1]["ATR"]), 4) if df_15m is not None and len(df_15m) else round(atr_now * 1.6, 4),
        "SIGNAL": signal,
        "DECISION": decision,
        "ACTION": action,
        "REASON": reason,
        "SESSION": session_name,
        "SETUP_DIRECTION": setup_direction,
        "BUY_SCORE": round(buy_score, 2),
        "SELL_SCORE": round(sell_score, 2),
        "CONCEPTS": concept_flags or {},
        "CONCEPT_QUALITY": round(float(concept_q), 3) if concept_q is not None else None,
        "CONCEPT_REASON": concept_reason,
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
        "WORLD_NEWS_DIRECTION": news_ctx.get("direction"),
        "WORLD_NEWS_SCORE": news_ctx.get("score"),
        "WORLD_NEWS_CONFIDENCE": news_ctx.get("confidence"),
        "WORLD_NEWS_SUMMARY": news_ctx.get("summary"),
        "WORLD_NEWS_MC_ADJUST": news_ctx.get("mc_adjust"),
        "WORLD_NEWS_ALIGNED": news_ctx.get("aligned"),
        "WORLD_NEWS_AVAILABLE": news_ctx.get("available"),
        "VP_AVAILABLE": bool(vp.get("available")),
        "VP_POC": vp.get("poc"),
        "VP_VAH": vp.get("vah"),
        "VP_VAL": vp.get("val"),
        "VP_LOCATION": vp.get("location"),
        "VP_DIRECTION": vp.get("direction"),
        "VP_SCORE": vp.get("score"),
        "VP_DIST_POC_ATR": vp.get("dist_to_poc_atr"),
        "VP_IN_VALUE": vp.get("in_value"),
        "VP_ABOVE_VALUE": vp.get("above_value"),
        "VP_BELOW_VALUE": vp.get("below_value"),
        "VP_AT_POC": vp.get("at_poc"),
        "VP_SUMMARY": vp.get("summary"),
        "VP_MC_ADJUST": vp_mc_adj,
        "VP_ALIGNED": vp_aligned,
        "VP_SESSION": vp.get("session_name"),
        "VP_SOURCE": vp.get("source") or vp.get("profile_mode"),
        "VP_ROLLING_POC": vp.get("rolling_poc"),
        "RR": (
            None
            if rr_used is None
            else (
                f"1:{int(rr_used)}"
                if float(rr_used) == int(float(rr_used))
                else f"1:{float(rr_used):.2f}"
            )
        ),
        "ENTRY": round_price(entry, symbol),
        "SL": round_price(sl, symbol),
        "TP": round_price(tp, symbol),
        "TP1": round_price(tp1, symbol),
        "TP2": None if best is None else round_price(best.get("tp2"), symbol),
        "TP3": None if best is None else round_price(best.get("tp3"), symbol),
        "TP_SOURCE": None if best is None else best.get("tp_source"),
        # Configured pair brain (always) + fired brain tag on entries
        "ENTRY_BRAIN": entry_brain,
        "BRAIN": brain_tag if signal in ("BUY", "SELL") else (brain_tag if brain_tag != "physics" else None),
        "CRYPTO_FIRM": bool(crypto_firm_meta.get("firm_formula"))
        and crypto_firm_meta.get("firm_kind") == "crypto",
        "FX_FIRM": bool(crypto_firm_meta.get("firm_formula"))
        and crypto_firm_meta.get("firm_kind") == "fx",
        "USDJPY_OPP_GOLD": bool(crypto_firm_meta.get("firm_formula"))
        and crypto_firm_meta.get("firm_kind") == "usdjpy_opp_gold",
        "FIRM_FORMULA": bool(crypto_firm_meta.get("firm_formula")),
        "FIRM_KIND": crypto_firm_meta.get("firm_kind"),
        "CRYPTO_FIRM_FOURIER_HF": crypto_firm_meta.get("fourier_hf"),
        "CRYPTO_FIRM_RET_Z": crypto_firm_meta.get("ret_z"),
        "CRYPTO_FIRM_KURT": crypto_firm_meta.get("kurt60"),
        "CRYPTO_FIRM_EFF": crypto_firm_meta.get("eff"),
        "CRYPTO_FIRM_GATE": crypto_firm_meta.get("firm_gate"),
        "CORR_VS_GOLD": crypto_firm_meta.get("corr_vs_gold"),
        "WITH_TREND": bool(crypto_firm_meta.get("with_trend")),
        "WITH_OPP_GOLD": bool(crypto_firm_meta.get("with_opp_gold")),
        "HTF_SYNC": crypto_firm_meta.get("htf_sync") or crypto_firm_meta.get("trend_sync"),
        "HTF_BIAS": crypto_firm_meta.get("htf_bias"),
        "SYMBOL": str(symbol).upper() if symbol else None,
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

    # Per-symbol hours — crypto stays open weekends; do not blanket-kill on gold close
    from strategy.instrument_meta import instrument_tradable, uses_gold_session

    sym_u = str(symbol or "GOLD").upper().strip()
    sym_open = instrument_tradable(sym_u)
    market = get_gold_market_status()
    out["MARKET_OPEN"] = sym_open
    out["MARKET_STATUS"] = (
        "open"
        if sym_open
        else market["status"]
    )
    if sym_open and not market["is_open"] and not uses_gold_session(sym_u):
        out["MARKET_LABEL"] = f"{sym_u} open"
    else:
        out["MARKET_LABEL"] = market["label"]
    out["MARKET_NEXT_OPEN"] = None if sym_open else market.get("next_open_display")
    out["MARKET_SERVER_TIME"] = market.get("server_time_display")
    if not sym_open:
        out["SIGNAL"] = "NO TRADE"
        out["DECISION"] = "WAIT"
        out["ACTION"] = f"{sym_u} market closed"
        # Don't paste gold-session copy onto FX/crypto closes
        if uses_gold_session(sym_u):
            out["REASON"] = market["message"]
        else:
            out["REASON"] = f"{sym_u} market closed (session hours)"
        out["ENTRY"] = None
        out["SL"] = None
        out["TP"] = None
        out["TP1"] = None
        out["TP2"] = None
        out["TP3"] = None
        out["RR"] = None
        out["EXIT"] = None

    return out


def build_stacked_trade(sig: dict, live_price: float) -> dict:
    """Build a stacked trade at live price — same zone risk; structure TP preferred."""
    direction = sig["SIGNAL"]
    atr = float(sig.get("ATR") or abs(float(sig["ENTRY"]) - float(sig["SL"])) or 1.0)
    state = sig.get("STATE", "CHOP")
    entry = float(live_price)
    symbol = sig.get("SYMBOL") or sig.get("INSTRUMENT")

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

    struct_tp_price = None
    try:
        from analysis.structure_targets import structure_enabled_for, structure_take_profit

        if structure_enabled_for(symbol):
            pack = structure_take_profit(
                direction,
                entry,
                test_sl,
                atr,
                zones=micro_zones,
                min_rr=STRUCTURE_TP_MIN_RR,
                max_rr=STRUCTURE_TP_MAX_RR,
            )
            if pack:
                struct_tp_price = float(pack["tp"])
    except Exception:
        struct_tp_price = None

    best = choose_minute_rr(
        direction,
        entry,
        test_sl,
        state,
        atr,
        micro_zones,
        symbol=symbol,
        structure_tp=struct_tp_price,
        structure_mode=STRUCTURE_TP_MODE,
    )

    out = dict(sig)
    out["ENTRY"] = round(entry, 2)
    out["SL"] = round(test_sl, 2)
    out["TP"] = best["tp"]
    out["TP1"] = best.get("tp1")
    out["TP2"] = best.get("tp2")
    out["TP3"] = best.get("tp3")
    rr_v = best.get("rr")
    out["RR"] = (
        f"1:{int(rr_v)}"
        if rr_v is not None and float(rr_v) == int(float(rr_v))
        else f"1:{float(rr_v):.2f}"
    )
    out["TP_SOURCE"] = best.get("tp_source")
    out["SUCCESS_PERCENT"] = best["mc_prob"]
    out["LIVE_PRICE"] = round(entry, 2)
    out["LOG_PRICE_1M"] = round(entry, 2)
    return out


def signal_once() -> dict:
    df_1m, df_3m, df_5m, df_15m, df_1h, df_4h = _fetch_all_timeframes()
    return _compute_signal(df_1m, df_3m, df_5m, df_15m, df_1h, df_4h)


def run_signal_cycle() -> dict:
    from data.market_data import get_source_info, last_fetch_source

    df_1m, df_3m, df_5m, df_15m, df_1h, df_4h = _fetch_all_timeframes()
    signal = _compute_signal(df_1m, df_3m, df_5m, df_15m, df_1h, df_4h)
    signal["LOG_TIME_1M"] = str(df_1m.index[-1])
    signal["LOG_TIME_3M"] = str(df_3m.index[-1]) if df_3m is not None else None
    signal["LOG_TIME_5M"] = str(df_5m.index[-1])
    last_1m = df_1m.iloc[-1]
    signal["LOG_PRICE_1M"] = round(float(last_1m["close"]), 2)
    signal["LOG_HIGH_1M"] = round(float(last_1m["high"]), 2)
    signal["LOG_LOW_1M"] = round(float(last_1m["low"]), 2)
    info = get_source_info()
    # Prefer what actually filled the last candle request (may be TV fallback)
    source = last_fetch_source() or info["data_source"]
    signal["DATA_SOURCE"] = source
    signal["SYMBOL"] = info["symbol"] if source == "xm" else "XAUUSD"
    return {
        "signal": signal,
        "data_source": source,
        "symbol": signal["SYMBOL"],
        "frames": {
            "1m": df_1m,
            "3m": df_3m,
            "5m": df_5m,
            "15m": df_15m,
            "1h": df_1h,
            "4h": df_4h,
        },
        "candles": {
            "1m": _serialize_candles(df_1m),
            "5m": _serialize_candles(df_5m),
            "15m": _serialize_candles(df_15m),
            "1h": _serialize_candles(df_1h),
            "4h": _serialize_candles(df_4h),
        },
    }


def scan_mtf_signals(
    frames: Optional[dict] = None,
    only_tfs: Optional[tuple] = None,
) -> list:
    """Independent setups across all configured timeframes."""
    from config import MTF_SCAN_TFS
    from strategy.mtf_scanner import scan_all_timeframes

    if frames is None:
        df_1m, df_3m, df_5m, df_15m, df_1h, df_4h = _fetch_all_timeframes()
        frames = {
            "1m": df_1m,
            "3m": df_3m,
            "5m": df_5m,
            "15m": df_15m,
            "1h": df_1h,
            "4h": df_4h,
        }
    tfs = only_tfs or MTF_SCAN_TFS
    return scan_all_timeframes(frames, only_tfs=tuple(tfs))
