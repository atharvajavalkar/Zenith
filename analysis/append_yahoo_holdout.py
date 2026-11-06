#!/usr/bin/env python3
"""Append the post-2026-07-23 hole onto local Dukascopy CSVs.

Dukascopy's public datafeed was returning 429/503. Yahoo's chart API still
has 5m/15m through today. XM session spreads stay the cost model; this only
extends the price path so the holdout is the period the old hunts never saw.

Does not touch 1m files or .env.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "hist"
OUT = ROOT / "data" / "hist_yahoo_join.json"

# Desk symbol → Yahoo chart ticker. Checked against the 2026-07-23 Dukascopy close.
TICKERS = {
    "GOLD": "GC=F",
    "BTCUSD": "BTC-USD",
    "UKOIL": "BZ=F",
    "NAS100": "NQ=F",
}
TFS = ("5m", "15m")


def _fetch(ticker: str, interval: str) -> pd.DataFrame:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(ticker)}?interval={interval}&range=60d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    result = payload["chart"]["result"][0]
    ts = result.get("timestamp") or []
    if not ts:
        raise RuntimeError(f"no bars {ticker} {interval}")
    q = result["indicators"]["quote"][0]
    idx = pd.to_datetime(ts, unit="s", utc=True)
    df = pd.DataFrame(
        {
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q.get("volume") or [0.0] * len(ts),
        },
        index=idx,
    )
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _append(symbol: str, tf: str, fresh: pd.DataFrame) -> dict:
    path = HIST / f"{symbol}_{tf}.csv"
    old = pd.read_csv(path, parse_dates=["time"])
    old = old.set_index("time")
    if old.index.tz is None:
        old.index = old.index.tz_localize("UTC")
    else:
        old.index = old.index.tz_convert("UTC")
    last = old.index[-1]
    add = fresh[fresh.index > last]
    if add.empty:
        return {"symbol": symbol, "tf": tf, "added": 0, "end": str(last)}
    gap_h = (add.index[0] - last).total_seconds() / 3600.0
    # Reject a scale break (wrong instrument), not a real weekend gap.
    ref = float(old["close"].iloc[-1])
    nxt = float(add["close"].iloc[0])
    jump = abs(nxt - ref) / ref if ref else 1.0
    if jump > 0.08 and gap_h < 48:
        raise RuntimeError(
            f"{symbol} {tf} jump {jump:.1%} across {gap_h:.1f}h "
            f"({ref} → {nxt}) — not appending"
        )
    merged = pd.concat([old, add]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    dump = merged.reset_index().rename(columns={"index": "time", "time": "time"})
    if dump.columns[0] != "time":
        dump = dump.rename(columns={dump.columns[0]: "time"})
    dump.to_csv(path, index=False)
    return {
        "symbol": symbol,
        "tf": tf,
        "source": TICKERS[symbol],
        "added": int(len(add)),
        "join": str(add.index[0]),
        "gap_hours": round(gap_h, 2),
        "prev_close": ref,
        "join_close": nxt,
        "end": str(merged.index[-1]),
    }


def main() -> int:
    report: dict = {}
    for sym, ticker in TICKERS.items():
        report[sym] = {}
        for tf in TFS:
            print(f"fetch {sym} {tf} via {ticker}", flush=True)
            fresh = _fetch(ticker, tf)
            info = _append(sym, tf, fresh)
            report[sym][tf] = info
            print(
                f"  added {info.get('added')} gap {info.get('gap_hours')}h "
                f"→ {info.get('end')}",
                flush=True,
            )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["note"] = (
        "Yahoo chart path appended after local Dukascopy. "
        "XM session spreads are still applied in the hunt, not in these prices."
    )
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
