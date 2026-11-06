"""XM average spread by pair × session since June (MetaAPI candle.spread).

MetaAPI: candles?startTime=T&limit=N returns up to N bars *ending at* T
(walking backward). We page backward from now until June 1.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, timedelta
import time

from data.broker import broker_client as c

PAIRS = ["ETHUSD", "BTCUSD", "GOLD", "NAS100", "USDJPY", "AUDUSD", "USDCHF", "UKOIL"]
# MT5 point size: spread field is in points
POINT = {
    "ETHUSD": 0.01,
    "BTCUSD": 0.01,
    "GOLD": 0.01,
    "NAS100": 0.01,
    "USDJPY": 0.001,
    "AUDUSD": 0.00001,
    "USDCHF": 0.00001,
    "UKOIL": 0.001,
}
PIP = {
    "ETHUSD": 0.01,
    "BTCUSD": 1.0,
    "GOLD": 0.1,
    "NAS100": 1.0,
    "USDJPY": 0.01,
    "AUDUSD": 0.0001,
    "USDCHF": 0.0001,
    "UKOIL": 0.01,
}
SESSIONS = [
    ("Asia", 0, 8),
    ("London", 8, 13),
    ("Overlap", 13, 16),
    ("NY", 16, 21),
    ("Off", 21, 24),
]
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


def session_of(h: int) -> str:
    for name, a, b in SESSIONS:
        if a <= h < b:
            return name
    return "Off"


def fetch_candles_with_spread(
    symbol: str, start: datetime, end: datetime, tf: str = "1h"
) -> list:
    """Page MetaAPI candles backward from end until start. Keep spread."""
    step = timedelta(minutes=TF_MINUTES.get(tf, 60))
    by_ts: dict[datetime, float] = {}
    cursor = end
    safety = 0
    while cursor > start and safety < 200:
        safety += 1
        st = cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        path = (
            f"/historical-market-data/symbols/{symbol}/timeframes/{tf}"
            f"/candles?startTime={st}&limit=1000"
        )
        try:
            raw = c._market_data_request(
                path, cache_key=f"spb:{symbol}:{tf}:{st}", cache_ttl=60
            )
        except Exception as e:
            print(f"  FAIL {symbol} {st}: {e}", flush=True)
            break
        if not isinstance(raw, list) or not raw:
            break
        oldest = None
        newest = None
        for r in raw:
            t = r.get("time") or ""
            try:
                ts = datetime.fromisoformat(t.replace("Z", "+00:00"))
            except Exception:
                continue
            if oldest is None or ts < oldest:
                oldest = ts
            if newest is None or ts > newest:
                newest = ts
            if ts < start or ts >= end:
                continue
            sp = r.get("spread")
            if sp is None:
                continue
            by_ts[ts] = float(sp)
        print(
            f"  {symbol}: {len(by_ts)} unique bars | batch {len(raw)} "
            f"({raw[0].get('time')} → {raw[-1].get('time')})",
            flush=True,
        )
        if oldest is None or oldest <= start:
            break
        nxt = oldest - step
        if nxt >= cursor:
            break
        cursor = nxt
        if len(raw) < 1000:
            break
        time.sleep(0.1)
    return sorted(by_ts.items(), key=lambda x: x[0])


def fmt(vals, sym: str):
    if not vals:
        return None
    avg_pts = sum(vals) / len(vals)
    avg_price = avg_pts * POINT[sym]
    avg_pips = avg_price / PIP[sym]
    return {
        "n": len(vals),
        "avg_points": round(avg_pts, 2),
        "avg_price": avg_price,
        "avg_pips": round(avg_pips, 3),
        "min_pts": min(vals),
        "max_pts": max(vals),
    }


def main():
    # Desk deals start ~2026-07-09; still request from June 1 for coverage
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    tf = "1h"
    print(
        f"Window: {start.date()} → {end.date()} UTC | TF={tf} | "
        "source=MetaAPI candle.spread (points)",
        flush=True,
    )
    print(
        "Sessions UTC: Asia 00-08 | London 08-13 | Overlap 13-16 | NY 16-21 | Off 21-24",
        flush=True,
    )
    print(flush=True)

    results = {}
    for sym in PAIRS:
        print(f"Fetching {sym}…", flush=True)
        bars = fetch_candles_with_spread(sym, start, end, tf)
        print(f"{sym}: {len(bars)} bars with spread", flush=True)
        if not bars:
            results[sym] = None
            continue
        by_sess = defaultdict(list)
        all_sp = []
        for ts, sp_pts in bars:
            by_sess[session_of(ts.hour)].append(sp_pts)
            all_sp.append(sp_pts)
        results[sym] = {
            "all": fmt(all_sp, sym),
            "sessions": {s: fmt(by_sess.get(s, []), sym) for s, _, _ in SESSIONS},
            "first": bars[0][0].isoformat(),
            "last": bars[-1][0].isoformat(),
        }

    print("\n" + "=" * 100, flush=True)
    print(
        f"{'PAIR':<10} {'SESSION':<10} {'N bars':>8} {'Avg pts':>10} "
        f"{'Avg pips':>10} {'Avg price':>14} {'min':>6} {'max':>6}",
        flush=True,
    )
    print("=" * 100, flush=True)
    for sym in PAIRS:
        r = results[sym]
        if not r:
            print(f"{sym:<10} NO DATA", flush=True)
            continue
        for sess in ["ALL"] + [s for s, _, _ in SESSIONS]:
            row = r["all"] if sess == "ALL" else r["sessions"].get(sess)
            if not row:
                print(f"{sym:<10} {sess:<10} {'—':>8}", flush=True)
                continue
            print(
                f"{sym:<10} {sess:<10} {row['n']:>8} {row['avg_points']:>10.2f} "
                f"{row['avg_pips']:>10.3f} {row['avg_price']:>14.6g} "
                f"{row['min_pts']:>6.0f} {row['max_pts']:>6.0f}",
                flush=True,
            )
        print(f"  coverage: {r['first'][:16]} → {r['last'][:16]}", flush=True)
        print("-" * 100, flush=True)

    print("\nLIVE snapshot (current bid/ask) for sanity:", flush=True)
    for sym in PAIRS:
        try:
            q = c.current_price(sym)
            bid, ask = float(q["bid"]), float(q["ask"])
            sp = ask - bid
            pip = PIP[sym]
            print(
                f"  {sym}: spread={sp:.6g} ({sp / pip:.2f} pips)  bid={bid} ask={ask}",
                flush=True,
            )
        except Exception as e:
            print(f"  {sym}: {e}", flush=True)


if __name__ == "__main__":
    main()
