"""Hourly macro + technical XAU/USD market analyst (bias only — no trade levels)."""

import json
import logging
import time
import urllib.request
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from tvDatafeed import Interval

from analysis.bias import get_bias_detail
from data.fetcher import fetch_tv_with_pause as fetch_tv
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


def _yahoo_daily_change(symbol: str) -> Optional[float]:
    """Percent change over last 2 daily closes."""
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


def _macro_context() -> dict:
    """Intermarket read: gold vs USD, yields, risk sentiment."""
    changes = {key: _yahoo_daily_change(sym) for key, sym in YAHOO_SYMBOLS.items()}
    bull = 0.0
    bear = 0.0
    notes: list[str] = []

    dxy = changes.get("dxy")
    if dxy is not None:
        if dxy > 0.15:
            bear += 1.2
            notes.append(f"DXY +{dxy:.2f}% (USD strength → gold headwind)")
        elif dxy < -0.15:
            bull += 1.2
            notes.append(f"DXY {dxy:.2f}% (USD weakness → gold tailwind)")

    vix = changes.get("vix")
    if vix is not None:
        if vix > 3:
            bull += 0.9
            notes.append(f"VIX rising (+{vix:.1f}%) — safe-haven bid")
        elif vix < -3:
            bear += 0.4
            notes.append("VIX falling — risk-on tone")

    tnx = changes.get("tnx")
    if tnx is not None:
        if tnx > 0.8:
            bear += 1.0
            notes.append(f"10Y yields +{tnx:.2f}% — real-rate pressure on gold")
        elif tnx < -0.8:
            bull += 1.0
            notes.append(f"10Y yields {tnx:.2f}% — supportive for gold")

    tyx = changes.get("tyx")
    if tyx is not None:
        if tyx > 0.5:
            bear += 0.5
        elif tyx < -0.5:
            bull += 0.5

    spx = changes.get("spx")
    if spx is not None:
        if spx < -0.5:
            bull += 0.6
            notes.append("Equities soft — defensive gold flows")
        elif spx > 0.5:
            bear += 0.3

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


def _overall_bias(tf_results: dict[str, dict], macro: dict) -> dict:
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

    return {
        "direction": direction,
        "confidence": avg_conf,
        "label": label,
    }


def run_market_analysis() -> dict:
    """Full hourly macro + technical scan for XAU/USD directional bias."""
    timeframes = [
        ("5m", Interval.in_5_minute, 300),
        ("15m", Interval.in_15_minute, 300),
        ("1h", Interval.in_1_hour, 300),
        ("4h", Interval.in_4_hour, 300),
    ]

    tf_results: dict[str, dict] = {}

    for label, interval, bars in timeframes:
        df = prepare_df(fetch_tv(interval, bars))
        detail = get_bias_detail(df)
        tf_results[label] = detail

    macro = _macro_context()
    overall = _overall_bias(tf_results, macro)

    return {
        "symbol": "XAU/USD",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Macro + technical market analyst for XAU/USD. Scans intermarket data, "
            "sentiment, and multi-timeframe structure hourly. Directional bias only — "
            "no trade execution or entry/exit levels."
        ),
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
        "overall": overall,
        "sources": [
            "Yahoo Finance (DXY, VIX, yields, equities)",
            "TradingView / XM XAUUSD price action",
            "Multi-TF EMA, MACD, RSI structure",
            "Supply & demand zones (macro context)",
        ],
    }
