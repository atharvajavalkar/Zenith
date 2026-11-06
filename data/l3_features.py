"""Live Coinbase book features for Zenith crypto (BTCUSD / ETHUSD).

Uses public REST book level=2 (aggregated from the L3 matching engine) —
fast enough for the execute path. Full order-id L3 snapshots stay in
`data/l3_ingest.py` (research / offline).

Fail-open by default: if Coinbase is down, do not block the desk.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

COINBASE_REST = "https://api.exchange.coinbase.com"
UA = "ZenithL3Live/1.0"

_DESK_TO_PRODUCT = {
    "BTCUSD": "BTC-USD",
    "BTC": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "ETH": "ETH-USD",
}

_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_lock = threading.Lock()
_CACHE_TTL = 2.0  # seconds — share across stacked candidates


def coinbase_product(symbol: Optional[str]) -> Optional[str]:
    s = str(symbol or "").upper().strip()
    return _DESK_TO_PRODUCT.get(s)


def is_crypto_l3_symbol(symbol: Optional[str]) -> bool:
    return coinbase_product(symbol) is not None


def _http_json(url: str, timeout: float = 4.0) -> Any:
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_coinbase_book_l2(product: str) -> dict[str, Any]:
    """Aggregated top-of-book depth from Coinbase (L3 engine → L2 view)."""
    url = f"{COINBASE_REST}/products/{product}/book?level=2"
    raw = _http_json(url)
    # level=2 rows are [price, size, num_orders]; tolerate 2-tuples too
    def _ps(row) -> tuple[float, float]:
        return float(row[0]), float(row[1])

    bids = [_ps(r) for r in (raw.get("bids") or [])]
    asks = [_ps(r) for r in (raw.get("asks") or [])]
    bids.sort(key=lambda x: x[0], reverse=True)
    asks.sort(key=lambda x: x[0])
    return {
        "ok": True,
        "product": product,
        "sequence": raw.get("sequence"),
        "bids": bids,
        "asks": asks,
    }


def book_features_from_l2(
    book: dict[str, Any],
    *,
    depth_levels: int = 10,
) -> dict[str, Any]:
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks:
        return {"ok": False, "error": "empty book"}

    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    spread_px = max(0.0, best_ask - best_bid)
    spread_bps = (spread_px / mid) * 10_000.0 if mid > 0 else 0.0

    n = max(1, int(depth_levels))
    bid_depth = sum(float(s) for _, s in bids[:n])
    ask_depth = sum(float(s) for _, s in asks[:n])
    bid_notional = sum(float(p) * float(s) for p, s in bids[:n])
    ask_notional = sum(float(p) * float(s) for p, s in asks[:n])
    tot = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / tot if tot > 0 else 0.0

    return {
        "ok": True,
        "venue": "coinbase",
        "schema": "L2_FROM_L3_ENGINE",
        "product": book.get("product"),
        "sequence": book.get("sequence"),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_px": round(spread_px, 6),
        "spread_bps": round(spread_bps, 3),
        "bid_depth": round(bid_depth, 6),
        "ask_depth": round(ask_depth, 6),
        "bid_notional_usd": round(bid_notional, 2),
        "ask_notional_usd": round(ask_notional, 2),
        "imbalance": round(imbalance, 4),  # + = bid-heavy
        "depth_levels": n,
    }


def get_crypto_book_features(
    symbol: Optional[str],
    *,
    depth_levels: int = 10,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Cached Coinbase book features for a Zenith crypto desk symbol."""
    product = coinbase_product(symbol)
    if not product:
        return {"ok": False, "error": "not a crypto L3 symbol", "symbol": symbol}

    now = time.time()
    if use_cache:
        with _lock:
            hit = _cache.get(product)
            if hit and now - hit[0] <= _CACHE_TTL:
                return dict(hit[1])

    try:
        book = fetch_coinbase_book_l2(product)
        feats = book_features_from_l2(book, depth_levels=depth_levels)
        feats["symbol"] = str(symbol or "").upper()
        feats["ts"] = now
    except Exception as exc:
        logger.warning("Coinbase book fetch failed (%s): %s", product, exc)
        return {"ok": False, "error": str(exc), "symbol": symbol, "product": product}

    if use_cache and feats.get("ok"):
        with _lock:
            _cache[product] = (now, dict(feats))
    return feats


def _xm_spread_floor(symbol: Optional[str]) -> float:
    """XM CFD typical half-spread proxy — Coinbase is far tighter than the fill venue."""
    s = str(symbol or "").upper()
    if s in ("ETHUSD", "ETH"):
        return 1.5
    if s in ("BTCUSD", "BTC"):
        return 8.0
    return 1.0


def evaluate_crypto_l3_gate(
    symbol: Optional[str],
    side: Optional[str],
    *,
    entry: Optional[float] = None,
    sl: Optional[float] = None,
    max_spread_bps: float = 8.0,
    min_top_notional_usd: float = 5_000.0,
    min_risk_spread_mult: float = 3.0,
    max_adverse_imbalance: float = 0.55,
    depth_levels: int = 10,
    fail_open: bool = True,
    xm_spread_floor: Optional[float] = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Return (allowed, reason, features).

    Blocks when:
      • Coinbase spread too wide (venue stress)
      • top-of-book notional too thin
      • planned risk < min_risk_spread_mult × max(cb_spread, xm_floor)
        (XM fill costs — same math as ETH micro-stop autopsy)
      • book imbalance strongly against the trade
    """
    feats = get_crypto_book_features(symbol, depth_levels=depth_levels)
    if not feats.get("ok"):
        if fail_open:
            return True, f"l3 fail-open: {feats.get('error')}", feats
        return False, f"l3 unavailable: {feats.get('error')}", feats

    spread_bps = float(feats.get("spread_bps") or 0)
    if spread_bps > float(max_spread_bps):
        return (
            False,
            f"l3 spread {spread_bps:.2f}bps > max {max_spread_bps}",
            feats,
        )

    top_n = min(
        float(feats.get("bid_notional_usd") or 0),
        float(feats.get("ask_notional_usd") or 0),
    )
    if top_n < float(min_top_notional_usd):
        return (
            False,
            f"l3 thin book ${top_n:,.0f} < min ${min_top_notional_usd:,.0f}",
            feats,
        )

    cb_spread = float(feats.get("spread_px") or 0)
    floor = (
        float(xm_spread_floor)
        if xm_spread_floor is not None
        else _xm_spread_floor(symbol)
    )
    eff_spread = max(cb_spread, floor)
    feats["xm_spread_floor"] = floor
    feats["effective_spread_px"] = eff_spread

    if entry is not None and sl is not None and eff_spread > 0:
        risk = abs(float(entry) - float(sl))
        need = float(min_risk_spread_mult) * eff_spread
        feats["risk_px"] = round(risk, 4)
        feats["min_risk_px"] = round(need, 4)
        if risk < need:
            return (
                False,
                f"l3 risk ${risk:.2f} < {min_risk_spread_mult:g}×eff_spread "
                f"(${need:.2f}; xm_floor={floor})",
                feats,
            )

    imb = float(feats.get("imbalance") or 0)
    side_u = str(side or "").upper()
    if side_u == "BUY" and imb < -float(max_adverse_imbalance):
        return False, f"l3 adverse imbalance {imb:.2f} vs BUY", feats
    if side_u == "SELL" and imb > float(max_adverse_imbalance):
        return False, f"l3 adverse imbalance {imb:.2f} vs SELL", feats

    return True, "l3 ok", feats


def features_for_trade_log(feats: dict[str, Any], gate_reason: str) -> dict[str, Any]:
    """Flatten into TradeRecord.features keys."""
    if not feats:
        return {"l3_gate": gate_reason, "l3_ok": False}
    return {
        "l3_gate": gate_reason,
        "l3_ok": bool(feats.get("ok")),
        "l3_venue": feats.get("venue"),
        "l3_schema": feats.get("schema"),
        "l3_product": feats.get("product"),
        "l3_spread_bps": feats.get("spread_bps"),
        "l3_spread_px": feats.get("spread_px"),
        "l3_imbalance": feats.get("imbalance"),
        "l3_bid_notional": feats.get("bid_notional_usd"),
        "l3_ask_notional": feats.get("ask_notional_usd"),
        "l3_best_bid": feats.get("best_bid"),
        "l3_best_ask": feats.get("best_ask"),
        "l3_xm_spread_floor": feats.get("xm_spread_floor"),
        "l3_effective_spread": feats.get("effective_spread_px"),
        "l3_risk_px": feats.get("risk_px"),
        "l3_min_risk_px": feats.get("min_risk_px"),
    }
