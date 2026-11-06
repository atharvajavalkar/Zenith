"""Multi-instrument market analyst — physics + macro + news + volume profile.

Runs the same algo bias stack for every symbol in INSTRUMENT_UNIVERSE
(GOLD, FX majors, BTC, …). Primary payload stays backward-compatible for
the bias gate; `instruments` holds the full desk board.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

from tvDatafeed import Interval

from analysis.bias import get_bias_detail
from analysis.volume_profile import build_dual_profile
from config import INSTRUMENT_UNIVERSE, NEWS_BIAS_WEIGHT, PROFILE_BIAS_WEIGHT, PROFILE_LOOKBACK_BARS
from data.market_data import fetch_ohlcv_with_pause as fetch_tv
from data.market_data import get_source_info
from indicators.prepare import prepare_df

logger = logging.getLogger(__name__)

TF_WEIGHTS = {"5m": 0.10, "15m": 0.15, "1h": 0.30, "4h": 0.45}
MACRO_WEIGHT = 0.22

YAHOO_SYMBOLS = {
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "tnx": "^TNX",
    "tyx": "^IRX",
    "spx": "^GSPC",
}


def _asset_class(symbol: str) -> str:
    from strategy.instrument_meta import bias_asset_class

    return bias_asset_class(symbol)


def _yahoo_daily_change(symbol: str) -> Optional[float]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&range=5d"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode())
        closes = payload["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        closes = [c for c in closes if c is not None]
        if len(closes) < 2:
            return None
        prev, last = closes[-2], closes[-1]
        if prev == 0:
            return None
        return (last - prev) / prev * 100.0
    except Exception as exc:
        logger.debug("Yahoo fetch failed for %s: %s", symbol, exc)
        return None


def _fetch_macro_changes() -> dict[str, Optional[float]]:
    return {key: _yahoo_daily_change(sym) for key, sym in YAHOO_SYMBOLS.items()}


def _macro_context(symbol: str = "GOLD", changes: Optional[dict] = None) -> dict:
    """Intermarket read — notes/scoring oriented to the given instrument."""
    changes = changes if changes is not None else _fetch_macro_changes()
    cls = _asset_class(symbol)
    bull = 0.0
    bear = 0.0
    notes: list[str] = []

    dxy = changes.get("dxy")
    vix = changes.get("vix")
    tnx = changes.get("tnx")
    tyx = changes.get("tyx")
    spx = changes.get("spx")

    if cls == "metals":
        if dxy is not None:
            if dxy > 0.15:
                bear += 1.2
                notes.append(f"DXY +{dxy:.2f}% (USD strength → metal headwind)")
            elif dxy < -0.15:
                bull += 1.2
                notes.append(f"DXY {dxy:.2f}% (USD weakness → metal tailwind)")
        if vix is not None:
            if vix > 3:
                bull += 0.9
                notes.append(f"VIX rising (+{vix:.1f}%) — safe-haven bid")
            elif vix < -3:
                bear += 0.4
                notes.append("VIX falling — risk-on tone")
        if tnx is not None:
            if tnx > 0.8:
                bear += 1.0
                notes.append(f"10Y yields +{tnx:.2f}% — real-rate pressure")
            elif tnx < -0.8:
                bull += 1.0
                notes.append(f"10Y yields {tnx:.2f}% — supportive for metals")
        if spx is not None:
            if spx < -0.5:
                bull += 0.6
                notes.append("Equities soft — defensive metal flows")
            elif spx > 0.5:
                bear += 0.3

    elif cls == "fx":
        # Quote as XXXUSD except USDJPY (USD in base)
        usd_base = symbol.upper().startswith("USD")  # USDJPY
        if dxy is not None:
            if dxy > 0.15:
                if usd_base:
                    bull += 1.2
                    notes.append(f"DXY +{dxy:.2f}% — USD firm (supportive for {symbol})")
                else:
                    bear += 1.2
                    notes.append(f"DXY +{dxy:.2f}% — USD firm (headwind for {symbol})")
            elif dxy < -0.15:
                if usd_base:
                    bear += 1.2
                    notes.append(f"DXY {dxy:.2f}% — USD soft (headwind for {symbol})")
                else:
                    bull += 1.2
                    notes.append(f"DXY {dxy:.2f}% — USD soft (tailwind for {symbol})")
        if tnx is not None:
            if tnx > 0.8:
                if usd_base:
                    bull += 0.7
                    notes.append(f"Yields up +{tnx:.2f}% — USD support")
                else:
                    bear += 0.7
                    notes.append(f"Yields up +{tnx:.2f}% — USD carry headwind")
            elif tnx < -0.8:
                if usd_base:
                    bear += 0.7
                else:
                    bull += 0.7
                    notes.append(f"Yields {tnx:.2f}% — softer USD tone")
        if vix is not None and symbol.upper() == "USDJPY":
            if vix > 3:
                bear += 0.8  # risk-off yen bid
                notes.append(f"VIX +{vix:.1f}% — yen safe-haven bid")
            elif vix < -3:
                bull += 0.5
                notes.append("VIX soft — risk FX / USDJPY constructive")
        if spx is not None and symbol.upper() in ("EURUSD", "GBPUSD"):
            if spx > 0.5:
                bull += 0.4
                notes.append("Equities firm — risk FX tone")
            elif spx < -0.5:
                bear += 0.4
                notes.append("Equities soft — risk-off FX tone")

    elif cls == "crypto":
        if spx is not None:
            if spx > 0.5:
                bull += 1.2
                notes.append(f"Equities +{spx:.2f}% — risk-on crypto bid")
            elif spx < -0.5:
                bear += 1.2
                notes.append(f"Equities {spx:.2f}% — risk-off crypto pressure")
        if vix is not None:
            if vix > 3:
                bear += 1.0
                notes.append(f"VIX +{vix:.1f}% — crypto risk-off")
            elif vix < -3:
                bull += 0.8
                notes.append("VIX falling — crypto risk appetite")
        if dxy is not None:
            if dxy > 0.3:
                bear += 0.6
                notes.append(f"DXY +{dxy:.2f}% — mild crypto headwind")
            elif dxy < -0.3:
                bull += 0.6
                notes.append(f"DXY {dxy:.2f}% — mild crypto tailwind")

    elif cls == "indices":
        if spx is not None:
            if spx > 0.4:
                bull += 1.4
                notes.append(f"S&P +{spx:.2f}% — equity / US30 bid")
            elif spx < -0.4:
                bear += 1.4
                notes.append(f"S&P {spx:.2f}% — equity / US30 pressure")
        if vix is not None:
            if vix > 3:
                bear += 1.2
                notes.append(f"VIX +{vix:.1f}% — index risk-off")
            elif vix < -3:
                bull += 0.9
                notes.append("VIX soft — index risk-on")
        if dxy is not None:
            if dxy > 0.25:
                bear += 0.4
            elif dxy < -0.25:
                bull += 0.4
        if tnx is not None:
            if tnx > 1.0:
                bear += 0.6
                notes.append(f"Yields +{tnx:.2f}% — growth / index headwind")
            elif tnx < -1.0:
                bull += 0.5

    else:  # energy
        if spx is not None:
            if spx > 0.5:
                bull += 0.6
                notes.append(f"Equities +{spx:.2f}% — demand tone for oil")
            elif spx < -0.5:
                bear += 0.6
                notes.append(f"Equities {spx:.2f}% — oil demand soft")
        if dxy is not None:
            if dxy > 0.2:
                bear += 0.5
                notes.append(f"DXY +{dxy:.2f}% — oil in USD headwind")
            elif dxy < -0.2:
                bull += 0.5
                notes.append(f"DXY {dxy:.2f}% — oil USD tailwind")

    if tyx is not None and cls == "metals":
        if tyx > 0.5:
            bear += 0.5
        elif tyx < -0.5:
            bull += 0.5

    total = bull + bear
    if total <= 0:
        direction = "NEUTRAL"
        confidence = 55.0
    elif bull > bear:
        direction = "BULLISH"
        confidence = min(92.0, 58.0 + (bull / total) * 32.0)
    else:
        direction = "BEARISH"
        confidence = min(92.0, 58.0 + (bear / total) * 32.0)

    return {
        "direction": direction,
        "confidence": round(confidence, 1),
        "notes": notes[:4],
        "available": any(v is not None for v in changes.values()),
    }


def _news_context(symbol: Optional[str] = None) -> dict:
    """World-desk average, optionally for one instrument from the markets board."""
    try:
        from data.gold_news import aggregate_world_news_bias, build_world_desk_news

        if symbol:
            desk = build_world_desk_news()
            for m in desk.get("markets") or []:
                if str(m.get("symbol") or "").upper() == str(symbol).upper():
                    return {
                        "direction": m.get("bias", "NEUTRAL"),
                        "confidence": float(m.get("confidence") or 50.0),
                        "score": float(m.get("score") or 0.0),
                        "notes": [m.get("summary") or ""],
                        "summary": m.get("summary"),
                        "bullish_hubs": None,
                        "bearish_hubs": None,
                        "neutral_hubs": None,
                        "hub_count": int(m.get("count") or 0),
                        "available": True,
                        "label": m.get("label", "Neutral"),
                    }

        world = aggregate_world_news_bias()
        return {
            "direction": world.get("direction", "NEUTRAL"),
            "confidence": float(world.get("confidence") or 50.0),
            "score": float(world.get("score") or 0.0),
            "notes": list(world.get("notes") or [])[:4],
            "summary": world.get("summary"),
            "bullish_hubs": world.get("bullish_hubs", 0),
            "bearish_hubs": world.get("bearish_hubs", 0),
            "neutral_hubs": world.get("neutral_hubs", 0),
            "hub_count": world.get("hub_count", 0),
            "available": bool(world.get("available")),
            "label": world.get("label", "Neutral"),
        }
    except Exception as exc:
        logger.warning("News context unavailable: %s", exc)
        return {
            "direction": "NEUTRAL",
            "confidence": 50.0,
            "score": 0.0,
            "notes": [],
            "summary": "News layer offline",
            "bullish_hubs": 0,
            "bearish_hubs": 0,
            "neutral_hubs": 0,
            "hub_count": 0,
            "available": False,
            "label": "Neutral",
        }


def _overall_bias(
    tf_results: dict[str, dict],
    macro: dict,
    news: Optional[dict] = None,
    profile: Optional[dict] = None,
) -> dict:
    weighted_sign = 0.0
    weighted_conf = 0.0
    total_w = 0.0

    for tf, detail in tf_results.items():
        w = TF_WEIGHTS[tf]
        sign = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}.get(detail["direction"], 0.0)
        weighted_sign += sign * detail["confidence"] * w
        weighted_conf += detail["confidence"] * w
        total_w += w

    if macro.get("available"):
        w = MACRO_WEIGHT
        sign = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}.get(macro["direction"], 0.0)
        weighted_sign += sign * macro["confidence"] * w
        weighted_conf += macro["confidence"] * w
        total_w += w

    news = news or {}
    if news.get("available") and NEWS_BIAS_WEIGHT > 0:
        w = float(NEWS_BIAS_WEIGHT)
        sign = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}.get(news.get("direction"), 0.0)
        score = float(news.get("score") or 0.0)
        if abs(score) > 0.05:
            sign = max(-1.5, min(1.5, score / 2.0))
        weighted_sign += sign * float(news.get("confidence") or 55.0) * w
        weighted_conf += float(news.get("confidence") or 55.0) * w
        total_w += w

    profile = profile or {}
    profile_w_used = 0.0
    if profile.get("available") and PROFILE_BIAS_WEIGHT > 0:
        w = float(PROFILE_BIAS_WEIGHT)
        profile_w_used = w
        score = float(profile.get("score") or 0.0)
        sign = max(-1.5, min(1.5, score / 1.2)) if abs(score) > 0.05 else 0.0
        if sign == 0.0:
            sign = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}.get(
                profile.get("direction"), 0.0
            )
        weighted_sign += sign * float(profile.get("confidence") or 55.0) * w
        weighted_conf += float(profile.get("confidence") or 55.0) * w
        total_w += w

    avg_conf = round(weighted_conf / total_w, 1) if total_w else 55.0
    sign_strength = abs(weighted_sign) / total_w if total_w else 0.0

    if weighted_sign >= 6:
        label = "Strong Bullish" if avg_conf >= 78 and sign_strength >= 12 else "Bullish"
        direction = "BULLISH"
    elif weighted_sign <= -6:
        label = "Strong Bearish" if avg_conf >= 78 and sign_strength >= 12 else "Bearish"
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
        label = "Neutral"
        avg_conf = max(52.0, min(65.0, avg_conf))

    news_w = float(NEWS_BIAS_WEIGHT if news.get("available") else 0.0)
    return {
        "direction": direction,
        "confidence": avg_conf,
        "label": label,
        "physics_weight": round(max(0.0, 1.0 - news_w - profile_w_used), 3),
        "news_weight": round(news_w, 3),
        "profile_weight": round(profile_w_used, 3),
    }


def _analyze_symbol(symbol: str, macro_changes: dict) -> dict[str, Any]:
    """Full bias stack for one instrument."""
    sym = str(symbol).upper().strip()
    timeframes = [
        ("5m", Interval.in_5_minute, 200),
        ("15m", Interval.in_15_minute, 200),
        ("1h", Interval.in_1_hour, 200),
        ("4h", Interval.in_4_hour, 200),
    ]

    tf_results: dict[str, dict] = {}
    df_5m = None
    error = None

    try:
        for label, interval, bars in timeframes:
            df = prepare_df(fetch_tv(interval, bars, symbol=sym))
            detail = get_bias_detail(df)
            tf_results[label] = detail
            if label == "5m":
                df_5m = df
    except Exception as exc:
        logger.warning("Physics unavailable for %s: %s", sym, exc)
        error = str(exc)
        for label, _, _ in timeframes:
            tf_results.setdefault(
                label,
                {"direction": "NEUTRAL", "confidence": 50.0, "bull_votes": 0, "bear_votes": 0},
            )

    macro = _macro_context(sym, macro_changes)
    news = _news_context(sym)

    profile: dict[str, Any] = {"available": False}
    try:
        if df_5m is not None and len(df_5m) >= 30:
            row = df_5m.iloc[-1]
            profile = build_dual_profile(
                df_5m,
                price=float(row["close"]),
                atr=float(row.get("ATR") or 1.0),
                lookback=min(PROFILE_LOOKBACK_BARS, 120),
                symbol=sym,
            )
    except Exception as exc:
        logger.warning("Volume profile unavailable for %s: %s", sym, exc)

    overall = _overall_bias(tf_results, macro, news, profile)

    return {
        "symbol": sym,
        "asset_class": _asset_class(sym),
        "error": error,
        "timeframes": {
            tf: {
                "label": tf,
                "direction": tf_results[tf]["direction"],
                "confidence": tf_results[tf]["confidence"],
            }
            for tf in tf_results
        },
        "macro": {
            "direction": macro["direction"],
            "confidence": macro["confidence"],
            "notes": macro["notes"],
            "available": macro["available"],
        },
        "news": news,
        "profile": {
            "available": bool(profile.get("available")),
            "direction": profile.get("direction", "NEUTRAL"),
            "confidence": profile.get("confidence", 50.0),
            "score": profile.get("score", 0.0),
            "location": profile.get("location"),
            "poc": profile.get("poc"),
            "vah": profile.get("vah"),
            "val": profile.get("val"),
            "summary": profile.get("summary"),
            "dist_to_poc_atr": profile.get("dist_to_poc_atr"),
            "bars_used": profile.get("bars_used", 0),
        },
        "overall": overall,
    }


def _pack_primary(primary: dict, info: dict) -> dict:
    """Legacy top-level shape (used by bias gate + old UI)."""
    return {
        "symbol": primary["symbol"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": info["data_source"],
        "description": (
            f"Physics (multi-TF + intermarket) plus world-news and session volume profile "
            f"for {primary['symbol']} and the full desk universe. Bias only — no trade levels."
        ),
        "timeframes": primary["timeframes"],
        "macro": primary["macro"],
        "news": primary["news"],
        "profile": primary["profile"],
        "overall": primary["overall"],
        "sources": [
            "Yahoo Finance (DXY, VIX, yields, equities)",
            f"{'XM MetaAPI' if info['data_source'] == 'xm' else 'TradingView'} multi-symbol candles",
            "Multi-TF EMA, MACD, RSI structure",
            "World news desks (Forex Factory + Yahoo)",
            "Session volume profile (POC / VAH / VAL)",
        ],
    }


def run_market_analysis() -> dict:
    """Scan INSTRUMENT_UNIVERSE — return primary + instruments[]."""
    info = get_source_info()
    universe = list(INSTRUMENT_UNIVERSE) or ["GOLD"]
    macro_changes = _fetch_macro_changes()

    instruments: list[dict] = []
    by_symbol: dict[str, dict] = {}
    for sym in universe:
        try:
            packed = _analyze_symbol(sym, macro_changes)
        except Exception as exc:
            logger.exception("Bias scan failed for %s: %s", sym, exc)
            packed = {
                "symbol": sym,
                "asset_class": _asset_class(sym),
                "error": str(exc),
                "timeframes": {
                    tf: {"label": tf, "direction": "NEUTRAL", "confidence": 50.0}
                    for tf in TF_WEIGHTS
                },
                "macro": {"direction": "NEUTRAL", "confidence": 50.0, "notes": [], "available": False},
                "news": _news_context(sym),
                "profile": {"available": False},
                "overall": {
                    "direction": "NEUTRAL",
                    "confidence": 55.0,
                    "label": "Neutral",
                    "physics_weight": 1.0,
                    "news_weight": 0.0,
                    "profile_weight": 0.0,
                },
            }
        instruments.append(packed)
        by_symbol[sym] = packed

    primary_sym = "GOLD" if "GOLD" in by_symbol else universe[0]
    primary = by_symbol[primary_sym]
    out = _pack_primary(primary, info)
    out["instruments"] = instruments
    out["universe"] = universe
    return out
