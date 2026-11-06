#!/usr/bin/env python3
"""Ingest depth / L3 for the Zenith universe → data/l3/.

Sources (best available without paid keys):
  • BTCUSD / ETHUSD — Coinbase Exchange true L3 snapshot (order_id) + optional
    websocket order-chain capture (full channel).
  • All pairs — XM MetaAPI current-book (broker DOM = aggregated L2, no order_id).

Optional (env):
  DATABENTO_API_KEY — CME/FX MBO or MBP historical/live
  TARDIS_API_KEY   — crypto L3 history (Coinbase/Bitfinex/Bitstamp)

Usage:
  python -m data.l3_ingest
  python -m data.l3_ingest --symbols BTCUSD,ETHUSD --chain-sec 45
  python -m data.l3_ingest --no-xm --no-chain
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.l3_universe import L3_MAP, ZENITH_UNIVERSE  # noqa: E402

L3_DIR = Path(ROOT) / "data" / "l3"
UA = "ZenithL3/1.0"
COINBASE_REST = "https://api.exchange.coinbase.com"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _http_json(url: str, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_coinbase_l3(product: str, *, out_dir: Optional[Path] = None) -> dict[str, Any]:
    """True L3 snapshot: each level is [price, size, order_id].

    Full book is written to gzip JSONL (not embedded) — books are 10–20MB.
    """
    url = f"{COINBASE_REST}/products/{product}/book?level=3"
    raw = _http_json(url)
    bids_raw = raw.get("bids") or []
    asks_raw = raw.get("asks") or []
    bids = [
        {"price": float(p), "size": float(s), "order_id": str(oid)}
        for p, s, oid in bids_raw
    ]
    asks = [
        {"price": float(p), "size": float(s), "order_id": str(oid)}
        for p, s, oid in asks_raw
    ]
    orders_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        orders_path = out_dir / f"{product.replace('-', '')}_l3_orders.jsonl.gz"
        with gzip.open(orders_path, "wt", encoding="utf-8") as f:
            for side, rows in (("bid", bids), ("ask", asks)):
                for r in rows:
                    f.write(
                        json.dumps(
                            {
                                "side": side,
                                "price": r["price"],
                                "size": r["size"],
                                "order_id": r["order_id"],
                            }
                        )
                        + "\n"
                    )
    # Aggregate L2 view (sum size by price) for quick analysis
    def _agg(rows: list[dict], reverse: bool) -> list[dict]:
        bucket: dict[float, float] = {}
        for r in rows:
            bucket[r["price"]] = bucket.get(r["price"], 0.0) + r["size"]
        levels = [{"price": p, "size": s, "n_orders": 0} for p, s in bucket.items()]
        # recount orders
        counts: dict[float, int] = {}
        for r in rows:
            counts[r["price"]] = counts.get(r["price"], 0) + 1
        for lv in levels:
            lv["n_orders"] = counts[lv["price"]]
        levels.sort(key=lambda x: x["price"], reverse=reverse)
        return levels[:50]

    return {
        "ok": True,
        "venue": "coinbase",
        "schema": "L3_MBO",
        "product": product,
        "sequence": raw.get("sequence"),
        "auction_mode": raw.get("auction_mode"),
        "bid_orders": len(bids),
        "ask_orders": len(asks),
        "best_bid": bids[0]["price"] if bids else None,
        "best_ask": asks[0]["price"] if asks else None,
        "top50_bid_levels": _agg(bids, True),
        "top50_ask_levels": _agg(asks, False),
        "orders_file": str(orders_path.relative_to(ROOT)) if orders_path else None,
        "fetched_at": _utc_now(),
    }


def fetch_binance_l2(symbol: str) -> dict[str, Any]:
    """Binance spot depth (aggregated L2) — free, no key."""
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100"
    raw = _http_json(url)
    bids = [{"price": float(p), "size": float(s)} for p, s in (raw.get("bids") or [])]
    asks = [{"price": float(p), "size": float(s)} for p, s in (raw.get("asks") or [])]
    return {
        "ok": True,
        "venue": "binance",
        "schema": "L2_MBP",
        "symbol": symbol,
        "last_update_id": raw.get("lastUpdateId"),
        "bids": bids,
        "asks": asks,
        "bid_levels": len(bids),
        "ask_levels": len(asks),
        "best_bid": bids[0]["price"] if bids else None,
        "best_ask": asks[0]["price"] if asks else None,
        "fetched_at": _utc_now(),
    }


def capture_coinbase_matches(
    products: list[str],
    *,
    seconds: float = 30.0,
    max_events: int = 50_000,
) -> dict[str, Any]:
    """Public trade tape via Coinbase `matches` (no auth).

    Note: `full` / L3 websocket channels now require Coinbase API auth.
    REST level=3 snapshots remain public.
    """
    try:
        import websocket  # websocket-client
    except ImportError as exc:
        return {
            "ok": False,
            "error": f"websocket-client not installed: {exc}",
            "hint": "pip install websocket-client",
        }

    events: list[dict[str, Any]] = []
    by_type: dict[str, int] = {}
    done = {"stop": False}

    def on_message(_ws, message: str) -> None:
        if done["stop"] or len(events) >= max_events:
            return
        try:
            msg = json.loads(message)
        except Exception:
            return
        t = msg.get("type")
        if t in ("subscriptions", "heartbeat"):
            return
        if t == "error":
            by_type["error"] = by_type.get("error", 0) + 1
            events.append({"t": "error", "message": msg.get("message"), "reason": msg.get("reason")})
            return
        by_type[t] = by_type.get(t, 0) + 1
        events.append(
            {
                "t": t,
                "time": msg.get("time"),
                "product_id": msg.get("product_id"),
                "trade_id": msg.get("trade_id"),
                "maker_order_id": msg.get("maker_order_id"),
                "taker_order_id": msg.get("taker_order_id"),
                "side": msg.get("side"),
                "price": msg.get("price"),
                "size": msg.get("size"),
                "sequence": msg.get("sequence"),
            }
        )

    def on_open(ws) -> None:
        ws.send(
            json.dumps(
                {
                    "type": "subscribe",
                    "product_ids": products,
                    "channels": ["matches"],
                }
            )
        )

    def on_error(_ws, err) -> None:
        by_type["_error"] = by_type.get("_error", 0) + 1
        events.append({"t": "error", "error": str(err)})

    ws = websocket.WebSocketApp(
        COINBASE_WS,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )

    import threading

    th = threading.Thread(target=ws.run_forever, kwargs={"ping_interval": 20}, daemon=True)
    th.start()
    time.sleep(max(1.0, float(seconds)))
    done["stop"] = True
    try:
        ws.close()
    except Exception:
        pass
    th.join(timeout=3.0)

    return {
        "ok": True,
        "venue": "coinbase",
        "schema": "TRADE_TAPE",
        "channel": "matches",
        "products": products,
        "seconds": seconds,
        "n_events": len(events),
        "by_type": by_type,
        "events": events,
        "note": "Public matches tape. Full L3 order-chain WS needs Coinbase API key auth.",
        "fetched_at": _utc_now(),
    }


def fetch_xm_book(desk: str) -> dict[str, Any]:
    """XM MetaAPI DOM — aggregated levels, no order_id."""
    from data.mt5_client import MetaApiClient
    from strategy.instrument_meta import broker_candidates, resolve_broker_symbol

    client = MetaApiClient()
    if not client.configured:
        return {
            "ok": False,
            "zenith": desk,
            "schema": "L2_AGG_BROKER_DOM",
            "error": "MetaAPI not configured",
        }

    def probe(sym: str) -> bool:
        try:
            client._request(  # noqa: SLF001 — intentional probe
                "GET",
                f"/symbols/{sym}/specification",
                cache_key=f"l3probe:{sym}",
                cache_ttl=300.0,
            )
            return True
        except Exception:
            return False

    wire = resolve_broker_symbol(desk, probe=probe)
    last_err: Optional[str] = None
    for cand in (wire, *broker_candidates(desk)):
        try:
            raw = client.current_book(cand, keep_subscription=True)
        except Exception as exc:
            last_err = str(exc)
            raw = None
        if not raw:
            continue
        book = raw.get("book") or []
        bids, asks = [], []
        for e in book:
            side = str(e.get("type") or "")
            row = {
                "price": float(e.get("price") or 0),
                "size": float(e.get("volume") or e.get("volume_dbl") or 0),
            }
            if "BUY" in side:
                bids.append(row)
            elif "SELL" in side:
                asks.append(row)
        bids.sort(key=lambda x: x["price"], reverse=True)
        asks.sort(key=lambda x: x["price"])
        return {
            "ok": True,
            "zenith": desk,
            "venue": "xm_metaapi",
            "schema": "L2_AGG_BROKER_DOM",
            "broker_symbol": cand,
            "time": raw.get("time"),
            "broker_time": raw.get("brokerTime"),
            "bids": bids,
            "asks": asks,
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            "best_bid": bids[0]["price"] if bids else None,
            "best_ask": asks[0]["price"] if asks else None,
            "raw_entries": len(book),
            "fetched_at": _utc_now(),
            "note": "XM depth is aggregated by price — not exchange order_id L3",
        }

    return {
        "ok": False,
        "zenith": desk,
        "schema": "L2_AGG_BROKER_DOM",
        "broker_tried": list(broker_candidates(desk)),
        "error": last_err or "empty book / symbol unsupported on this XM account",
        "fetched_at": _utc_now(),
    }


def try_databento_hint(desk: str) -> dict[str, Any]:
    key = os.getenv("DATABENTO_API_KEY", "").strip()
    meta = L3_MAP.get(desk) or {}
    db = meta.get("databento")
    if not db:
        return {"ok": False, "skipped": True, "reason": "no databento map"}
    if not key:
        return {
            "ok": False,
            "skipped": True,
            "reason": "DATABENTO_API_KEY not set",
            "would_fetch": db,
        }
    try:
        import databento as db_mod  # type: ignore
    except ImportError:
        return {
            "ok": False,
            "error": "databento package not installed (pip install databento)",
            "would_fetch": db,
        }
    # Live snapshot via historical recent window is heavy; store a tiny live MBO sample
    try:
        client = db_mod.Historical(key)
        # Last ~2 minutes of MBO can be huge — pull mbp-1 as smoke if mbo too large
        schema = db.get("schema") or "mbo"
        data = client.timeseries.get_range(
            dataset=db["dataset"],
            symbols=[db["symbol"]],
            schema=schema,
            start=(datetime.now(timezone.utc).strftime("%Y-%m-%d")),
            limit=500,
        )
        rows = []
        for i, rec in enumerate(data):
            if i >= 500:
                break
            rows.append(str(rec))
        return {
            "ok": True,
            "venue": "databento",
            "schema": schema,
            "dataset": db["dataset"],
            "symbol": db["symbol"],
            "n": len(rows),
            "sample": rows[:20],
            "fetched_at": _utc_now(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "would_fetch": db}


_BINANCE_SPOT = {"BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT"}


def ingest_symbol(
    desk: str,
    *,
    use_xm: bool = True,
    use_coinbase: bool = True,
    use_databento: bool = True,
    run_dir: Optional[Path] = None,
) -> dict[str, Any]:
    meta = L3_MAP.get(desk) or {}
    out: dict[str, Any] = {
        "zenith": desk,
        "map": meta,
        "sources": {},
        "fetched_at": _utc_now(),
    }

    if use_coinbase and meta.get("primary") == "coinbase" and meta.get("product"):
        try:
            out["sources"]["coinbase_l3"] = fetch_coinbase_l3(
                str(meta["product"]), out_dir=run_dir
            )
        except Exception as exc:
            out["sources"]["coinbase_l3"] = {"ok": False, "error": str(exc)}
        bsym = _BINANCE_SPOT.get(desk)
        if bsym:
            try:
                out["sources"]["binance_l2"] = fetch_binance_l2(bsym)
            except Exception as exc:
                out["sources"]["binance_l2"] = {"ok": False, "error": str(exc)}

    if use_xm:
        try:
            out["sources"]["xm_dom"] = fetch_xm_book(desk)
        except Exception as exc:
            out["sources"]["xm_dom"] = {"ok": False, "error": str(exc)}

    if use_databento and meta.get("databento"):
        out["sources"]["databento"] = try_databento_hint(desk)

    # Best available flag
    if (
        out["sources"].get("coinbase_l3")
        and out["sources"]["coinbase_l3"].get("schema") == "L3_MBO"
        and out["sources"]["coinbase_l3"].get("ok")
    ):
        out["best_schema"] = "L3_MBO"
        out["best_source"] = "coinbase_l3"
    elif out["sources"].get("binance_l2", {}).get("ok"):
        out["best_schema"] = "L2_MBP"
        out["best_source"] = "binance_l2"
    elif out["sources"].get("xm_dom", {}).get("ok"):
        out["best_schema"] = "L2_AGG_BROKER_DOM"
        out["best_source"] = "xm_dom"
    else:
        out["best_schema"] = None
        out["best_source"] = None
        out["blocked_reason"] = (
            "XM account has no Market Depth for this CFD; "
            "set DATABENTO_API_KEY for CME/FX proxy L3/L2"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Zenith L3 / depth ingest")
    ap.add_argument(
        "--symbols",
        default=",".join(ZENITH_UNIVERSE),
        help="Comma desk symbols (default: full Zenith universe)",
    )
    ap.add_argument("--no-xm", action="store_true", help="Skip XM MetaAPI DOM")
    ap.add_argument("--no-coinbase", action="store_true", help="Skip Coinbase L3")
    ap.add_argument("--no-databento", action="store_true", help="Skip Databento attempt")
    ap.add_argument(
        "--chain-sec",
        type=float,
        default=30.0,
        help="Seconds of Coinbase public matches tape (0=skip)",
    )
    ap.add_argument("--no-chain", action="store_true", help="Skip websocket matches tape")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run_id = _stamp()
    run_dir = L3_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Zenith L3 ingest · {run_id} ===", flush=True)
    print(f"out: {run_dir}", flush=True)

    results: list[dict[str, Any]] = []
    for desk in symbols:
        print(f"\n--- {desk} ---", flush=True)
        payload = ingest_symbol(
            desk,
            use_xm=not args.no_xm,
            use_coinbase=not args.no_coinbase,
            use_databento=not args.no_databento,
            run_dir=run_dir,
        )
        path = run_dir / f"{desk}.json"
        _write_json(path, payload)
        best = payload.get("best_source")
        schema = payload.get("best_schema")
        src = payload.get("sources") or {}
        if "coinbase_l3" in src and src["coinbase_l3"].get("ok"):
            c = src["coinbase_l3"]
            print(
                f"  Coinbase L3: {c.get('bid_orders')} bid orders / "
                f"{c.get('ask_orders')} ask orders  "
                f"BBO {c.get('best_bid')} / {c.get('best_ask')}  "
                f"file={c.get('orders_file')}",
                flush=True,
            )
        if "binance_l2" in src and src["binance_l2"].get("ok"):
            b = src["binance_l2"]
            print(
                f"  Binance L2: {b.get('bid_levels')}x{b.get('ask_levels')}  "
                f"BBO {b.get('best_bid')} / {b.get('best_ask')}",
                flush=True,
            )
        if "xm_dom" in src:
            x = src["xm_dom"]
            if x.get("ok"):
                print(
                    f"  XM DOM ({x.get('broker_symbol')}): "
                    f"{x.get('bid_levels')}x{x.get('ask_levels')} levels  "
                    f"BBO {x.get('best_bid')} / {x.get('best_ask')}",
                    flush=True,
                )
            else:
                print(f"  XM DOM unavailable: {x.get('error')}", flush=True)
        db = src.get("databento")
        if db and not db.get("skipped"):
            print(f"  Databento: ok={db.get('ok')} n={db.get('n')} err={db.get('error')}", flush=True)
        elif db and db.get("skipped"):
            print(f"  Databento skipped: {db.get('reason')}", flush=True)
        if payload.get("blocked_reason"):
            print(f"  BLOCKED: {payload['blocked_reason']}", flush=True)
        print(f"  best={best} schema={schema} → {path.name}", flush=True)
        results.append(
            {
                "zenith": desk,
                "best_source": best,
                "best_schema": schema,
                "file": str(path.relative_to(ROOT)),
                "xm_ok": bool((src.get("xm_dom") or {}).get("ok")),
                "coinbase_orders": (src.get("coinbase_l3") or {}).get("bid_orders"),
                "orders_file": (src.get("coinbase_l3") or {}).get("orders_file"),
                "blocked_reason": payload.get("blocked_reason"),
            }
        )
        time.sleep(0.35)  # be gentle on MetaAPI

    chain_path = None
    if not args.no_chain and args.chain_sec > 0:
        products = []
        for desk in symbols:
            m = L3_MAP.get(desk) or {}
            if m.get("primary") == "coinbase" and m.get("product"):
                products.append(str(m["product"]))
        if products:
            print(
                f"\n--- Coinbase matches tape ({args.chain_sec}s) {products} ---",
                flush=True,
            )
            chain = capture_coinbase_matches(products, seconds=args.chain_sec)
            chain_path = run_dir / "coinbase_matches.jsonl"
            summary = {k: v for k, v in chain.items() if k != "events"}
            _write_json(run_dir / "coinbase_matches_summary.json", summary)
            with chain_path.open("w", encoding="utf-8") as f:
                for ev in chain.get("events") or []:
                    f.write(json.dumps(ev) + "\n")
            print(
                f"  events={chain.get('n_events')} by_type={chain.get('by_type')} "
                f"→ {chain_path.name}",
                flush=True,
            )
            if not chain.get("ok"):
                print(f"  tape error: {chain.get('error')} {chain.get('hint', '')}", flush=True)

    manifest = {
        "run_id": run_id,
        "generated_at": _utc_now(),
        "symbols": symbols,
        "results": results,
        "matches_tape": str(chain_path.relative_to(ROOT)) if chain_path else None,
        "notes": {
            "true_l3_now": "BTCUSD + ETHUSD Coinbase REST level=3 (order_id gzip JSONL)",
            "xm_dom": "This XM/MetaAPI account returns 404 — no Market Depth on these CFDs",
            "futures_fx_true_l3": "Set DATABENTO_API_KEY (+ pip install databento) for CME GC/NQ/CL MBO",
            "crypto_history_l3": "Set TARDIS_API_KEY for historical Coinbase L3",
            "live_order_chain_ws": "Coinbase full/L3 WS needs API key auth; matches tape is public",
        },
    }
    man_path = run_dir / "manifest.json"
    _write_json(man_path, manifest)
    # latest pointer
    latest = {
        "run_id": run_id,
        "dir": str(run_dir.relative_to(ROOT)),
        "generated_at": manifest["generated_at"],
        "results": results,
        "matches_tape": manifest.get("matches_tape"),
        "notes": manifest.get("notes"),
    }
    _write_json(L3_DIR / "latest.json", latest)
    print(f"\nWrote {man_path}", flush=True)
    print(
        f"True L3: {sum(1 for r in results if r['best_schema']=='L3_MBO')}/"
        f"{len(results)} · XM DOM ok: {sum(1 for r in results if r['xm_ok'])}/{len(results)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
