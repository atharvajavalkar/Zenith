#!/usr/bin/env python3
"""
Download multi-year M1 from Dukascopy public datafeed → data/hist/.

Uses native BID_candles_min_1.bi5 (1 file/day). Resamples to 3m/5m/15m.
Live trading path is not changed. Spreads ignored (bid only) for stress tests.

Usage (host or container):
  python -m data.dukascopy_ingest --years 10
  python -m data.dukascopy_ingest --symbols GOLD,AUDUSD --years 5
  python -m data.dukascopy_ingest --start 2015-01-01 --end 2026-07-01
"""
from __future__ import annotations

import argparse
import lzma
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hist_local import (  # noqa: E402
    DUKA_INSTRUMENTS,
    DUKA_POINT_DIV,
    HIST_DIR,
    clear_hist_cache,
    hist_path,
    save_ohlcv_csv,
)

UA = "Mozilla/5.0 (compatible; ZenithHist/1.0)"
BASE = "https://datafeed.dukascopy.com/datafeed"
_PACE_LOCK = threading.Lock()
_PACE_AT = 0.0


def _pace() -> None:
    """Stay under the public datafeed's 429 limit."""
    global _PACE_AT
    with _PACE_LOCK:
        now = time.time()
        wait = 0.45 - (now - _PACE_AT)
        if wait > 0:
            time.sleep(wait)
        _PACE_AT = time.time()
DAY_FMT = ">iiiii f"  # sec, o, c, l, h, vol  — big-endian
REC_SIZE = 24
DEFAULT_BOOK = (
    "GOLD",
    "USDCHF",
    "USDJPY",
    "NAS100",
    "UKOIL",
    "AUDUSD",
    "BTCUSD",
    "ETHUSD",
)


def _day_url(instrument: str, d: date) -> str:
    # Dukascopy months are 0-indexed
    return (
        f"{BASE}/{instrument}/{d.year}/{d.month - 1:02d}/{d.day:02d}/"
        f"BID_candles_min_1.bi5"
    )


def _fetch_bytes(url: str, retries: int = 6) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(retries):
        _pace()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return b""
            last = exc
            # Public datafeed returns 429 if days are pulled in parallel.
            wait = 6.0 + attempt * 4.0 if exc.code == 429 else min(2**attempt * 0.35, 12.0)
            time.sleep(wait)
        except Exception as exc:
            last = exc
            time.sleep(min(2**attempt * 0.35, 12.0))
    if last:
        raise last
    return b""


def _decode_day(blob: bytes, instrument: str, d: date) -> list[tuple]:
    if not blob:
        return []
    try:
        raw = lzma.decompress(blob)
    except Exception:
        return []
    if len(raw) % REC_SIZE:
        return []
    div = float(DUKA_POINT_DIV.get(instrument, 100_000.0))
    day0 = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    rows: list[tuple] = []
    n = len(raw) // REC_SIZE
    for i in range(n):
        sec, o, c, l, h, vol = struct.unpack_from(DAY_FMT, raw, i * REC_SIZE)
        ts = day0 + timedelta(seconds=int(sec))
        rows.append(
            (
                ts,
                o / div,
                h / div,
                l / div,
                c / div,
                float(vol),
            )
        )
    return rows


def _download_day(instrument: str, d: date) -> list[tuple]:
    return _decode_day(_fetch_bytes(_day_url(instrument, d)), instrument, d)


def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _load_existing_1m(symbol: str) -> Optional[pd.DataFrame]:
    path = hist_path(symbol, "1m")
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.columns = [str(c).lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def _resample_all(m1: pd.DataFrame, symbol: str) -> None:
    rules = {
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
    }
    for tf, rule in rules.items():
        out = pd.DataFrame(
            {
                "open": m1["open"].resample(rule).first(),
                "high": m1["high"].resample(rule).max(),
                "low": m1["low"].resample(rule).min(),
                "close": m1["close"].resample(rule).last(),
                "volume": m1["volume"].resample(rule).sum(),
            }
        ).dropna()
        path = save_ohlcv_csv(out, symbol, tf)
        print(f"    wrote {path.name}: {len(out):,} bars", flush=True)


def ingest_symbol(
    symbol: str,
    start: date,
    end: date,
    workers: int = 8,
    resume: bool = True,
) -> Path:
    sym = symbol.upper().strip()
    instrument = DUKA_INSTRUMENTS.get(sym)
    if not instrument:
        raise ValueError(f"no Dukascopy map for {sym}")

    existing = _load_existing_1m(sym) if resume else None
    # Only skip days already present (UTC calendar days with ≥100 M1 bars)
    have_days: set[date] = set()
    if existing is not None and not existing.empty:
        counts = existing.groupby(existing.index.date).size()
        have_days = {d for d, n in counts.items() if int(n) >= 100}
        # Always refresh the last 3 calendar days of the requested window
        for i in range(3):
            have_days.discard(end - timedelta(days=i))
        print(
            f"  {sym}: resume — keeping {len(have_days)} complete days "
            f"(file {existing.index[0].date()}→{existing.index[-1].date()}, "
            f"{len(existing):,} bars)",
            flush=True,
        )

    days = [d for d in _daterange(start, end) if d not in have_days]
    print(
        f"  {sym} ({instrument}): fetching {len(days)} days "
        f"{start}→{end} workers={workers}",
        flush=True,
    )

    # Chunked download + checkpoint so a kill doesn't lose months of work
    chunk_size = 90
    m1 = existing
    total_done = 0
    errors = 0
    for chunk_i in range(0, max(len(days), 1), chunk_size):
        chunk = days[chunk_i : chunk_i + chunk_size]
        if not chunk:
            break
        rows: list[tuple] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(_download_day, instrument, d): d for d in chunk}
            for fut in as_completed(futs):
                d = futs[fut]
                total_done += 1
                try:
                    rows.extend(fut.result())
                except Exception as exc:
                    errors += 1
                    if errors <= 8:
                        print(f"    warn {sym} {d}: {exc}", flush=True)
                if total_done % 60 == 0 or total_done == len(days):
                    print(
                        f"    {sym}: {total_done}/{len(days)} days "
                        f"(err={errors})",
                        flush=True,
                    )

        if rows:
            new_df = pd.DataFrame(
                rows, columns=["time", "open", "high", "low", "close", "volume"]
            )
            new_df = new_df.set_index("time").sort_index()
            new_df = new_df[~new_df.index.duplicated(keep="last")]
            if m1 is not None and not m1.empty:
                m1 = pd.concat([m1, new_df]).sort_index()
                m1 = m1[~m1.index.duplicated(keep="last")]
            else:
                m1 = new_df
            # Checkpoint 1m after every chunk
            save_ohlcv_csv(m1, sym, "1m")
            print(
                f"    {sym}: checkpoint 1m {len(m1):,} bars "
                f"→ {m1.index[0].date()}…{m1.index[-1].date()}",
                flush=True,
            )

    if m1 is None or m1.empty:
        raise RuntimeError(f"no bars downloaded for {sym}")

    m1 = m1[(m1.index.date >= start) & (m1.index.date <= end)]
    if m1.empty:
        raise RuntimeError(f"{sym}: empty after merge")

    path = save_ohlcv_csv(m1, sym, "1m")
    print(
        f"  {sym}: saved 1m {len(m1):,} bars "
        f"{m1.index[0].date()}→{m1.index[-1].date()} → {path}",
        flush=True,
    )
    _resample_all(m1, sym)
    clear_hist_cache()
    return path


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ingest Dukascopy M1 → data/hist/")
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_BOOK),
        help="Comma-separated desk symbols",
    )
    p.add_argument("--years", type=float, default=10.0, help="Lookback years")
    p.add_argument("--start", default="", help="YYYY-MM-DD (overrides --years)")
    p.add_argument("--end", default="", help="YYYY-MM-DD (default: yesterday UTC)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)

    end = (
        date.fromisoformat(args.end)
        if args.end
        else (datetime.now(timezone.utc).date() - timedelta(days=1))
    )
    if args.start:
        start = date.fromisoformat(args.start)
    else:
        start = end - timedelta(days=int(args.years * 365.25))

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Dukascopy ingest → {HIST_DIR}\n"
        f"  window {start} → {end}\n"
        f"  symbols {symbols}",
        flush=True,
    )

    failed: list[str] = []
    for sym in symbols:
        try:
            ingest_symbol(
                sym,
                start,
                end,
                workers=args.workers,
                resume=not args.no_resume,
            )
        except Exception as exc:
            failed.append(sym)
            print(f"  FAIL {sym}: {exc}", flush=True)

    print(
        f"\nDone. hist dir: {HIST_DIR}  failed={failed or 'none'}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
