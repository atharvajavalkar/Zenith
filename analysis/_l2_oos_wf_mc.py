#!/usr/bin/env python3
"""RESEARCH ONLY — OOS / walk-forward / Monte Carlo with best free L2 we can get.

True free L2:
  BTC/ETH → Binance USDM bookDepth (percentage-band depth around mid)
Elsewhere → none free (FX/gold/NAS/oil). Baseline OOS/WF/MC only.

No live algo tweaks.
"""
from __future__ import annotations

import json
import sys
import zipfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import (  # noqa: E402
    BRAINS,
    BOOK,
    RISK_USD,
    _fetch_1h,
    _prep,
    _stats,
    monte_carlo,
    run_breakout_trades,
    run_ema_trades,
    run_fib_trades,
    run_sma_trades,
    verdict,
)

CACHE = Path("/tmp/l2_oos_cache")
OUT = Path("/tmp/l2_oos_wf_mc.json")
CACHE.mkdir(parents=True, exist_ok=True)

BINANCE_BOOK = {
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
}


def gen_trades(symbol: str, df: pd.DataFrame) -> tuple[str, list[dict]]:
    brain = BRAINS.get(symbol, "ema_20_50_pull")
    if brain == "fib_liq":
        return brain, run_fib_trades(df)
    if brain == "sma_20_50_cross":
        return brain, run_sma_trades(df)
    if brain == "breakout_20":
        return brain, run_breakout_trades(df)
    pull, disp = (0.75, 0.70) if brain == "ema_20_50_tight" else (1.0, 0.55)
    return brain, run_ema_trades(df, pull_atr=pull, disp_atr=disp)


def download_book_day(binance_sym: str, day: str) -> Optional[Path]:
    name = f"{binance_sym}-bookDepth-{day}.zip"
    url = (
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        f"{binance_sym}/{name}"
    )
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 500:
        return dest
    try:
        urllib.request.urlretrieve(url, dest)
        if dest.stat().st_size < 200:
            dest.unlink(missing_ok=True)
            return None
        return dest
    except Exception:
        return None


def load_book_imbalance(binance_sym: str, days: list[str]) -> Optional[pd.Series]:
    """Hourly L2 imbalance = (bid_notional - ask_notional) / total near mid (|pct|<=1)."""
    parts = []
    for day in days:
        z = download_book_day(binance_sym, day)
        if z is None:
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    df = pd.read_csv(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["percentage"] = df["percentage"].astype(float)
            df["notional"] = df["notional"].astype(float)
            # near-touch bands only
            near = df[df["percentage"].abs() <= 1.0].copy()
            if near.empty:
                continue
            near["side_bid"] = near["percentage"] < 0
            g = near.groupby([near["timestamp"].dt.floor("h"), "side_bid"])["notional"].sum()
            g = g.unstack(fill_value=0.0)
            if True not in g.columns:
                g[True] = 0.0
            if False not in g.columns:
                g[False] = 0.0
            bid = g[True]
            ask = g[False]
            imb = (bid - ask) / (bid + ask).replace(0, np.nan)
            parts.append(imb)
            print(f"    bookDepth {binance_sym} {day} ok", flush=True)
        except Exception as exc:
            print(f"    bookDepth fail {day}: {exc}", flush=True)
    if not parts:
        return None
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def l2_filter(trades: list[dict], imb: pd.Series, thr: float = 0.05) -> list[dict]:
    kept = []
    for tr in trades:
        ts = pd.Timestamp(tr["entry_t"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        sub = imb.loc[:ts]
        if sub.empty or pd.isna(sub.iloc[-1]):
            continue
        v = float(sub.iloc[-1])
        if tr["side"] == "BUY" and v > thr:
            kept.append(tr)
        elif tr["side"] == "SELL" and v < -thr:
            kept.append(tr)
    return kept


def oos_wf_mc(trades: list[dict], n_bars: int, label: str) -> dict[str, Any]:
    """Approximate OOS/WF using trade index fractions (same spirit as bar splits)."""
    full = _stats(trades)
    if not trades:
        return {
            "label": label,
            "full": full,
            "oos": _stats([]),
            "walk_forward": [],
            "monte_carlo": {"paths": 0, "note": "no_trades"},
            "verdict": "DEAD",
        }

    # Sort by entry index if present else keep order
    tr = sorted(trades, key=lambda t: t.get("i", 0))
    n = len(tr)
    split = max(1, int(n * 0.70))
    oos = _stats(tr[split:])
    is_ = _stats(tr[:split])

    wf = []
    for k in range(4):
        a = int(n * (0.40 + 0.15 * k))
        b = int(n * (0.55 + 0.15 * k))
        if b <= a:
            continue
        train = _stats(tr[:a])
        test = _stats(tr[a:b])
        wf.append({"fold": k + 1, "train": train, "oos": test})

    mc = monte_carlo(tr)
    # torture-lite: skip — keep verdict compatible
    torture = {"baseline": full}
    v = verdict(full, oos, wf, torture, mc)
    return {
        "label": label,
        "full": full,
        "in_sample": is_,
        "oos": oos,
        "walk_forward": wf,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": sum(1 for f in wf if f["oos"].get("exp_r", 0) > 0),
        "wf_total": len(wf),
    }


def layman(sym: str, has_l2: bool, base: dict, l2: Optional[dict]) -> str:
    if not has_l2:
        return (
            f"No free Level-2 for {sym}. "
            f"Candle algo alone: OOS E[R]={base['oos'].get('exp_r')} · "
            f"WF {base['wf_green']}/{base['wf_total']} · "
            f"MC profit {base['monte_carlo'].get('prob_profit', 'n/a')}% · "
            f"verdict {base['verdict']}."
        )
    assert l2 is not None
    b_er, l_er = base["oos"].get("exp_r"), l2["oos"].get("exp_r")
    b_mc = base["monte_carlo"].get("prob_profit")
    l_mc = l2["monte_carlo"].get("prob_profit")
    if l2["full"]["n"] < 4:
        return (
            f"Real Binance L2 loaded, but too few trades survived the book filter "
            f"(n={l2['full']['n']}). Not enough to trust. Keep candle algo."
        )
    better = (l_er is not None and b_er is not None and l_er > b_er + 0.05) or (
        l_mc is not None and b_mc is not None and l_mc > b_mc + 5
    )
    worse = (l_er is not None and b_er is not None and l_er < b_er - 0.05) or (
        l_mc is not None and b_mc is not None and l_mc + 5 < b_mc
    )
    if better:
        return (
            f"Binance L2 filter helped: OOS E[R] {b_er}→{l_er}, "
            f"MC {b_mc}%→{l_mc}%, verdict {base['verdict']}→{l2['verdict']}. "
            f"Worth researching — do not live-wire yet."
        )
    if worse:
        return (
            f"Binance L2 filter hurt or weakened the edge: OOS {b_er}→{l_er}, "
            f"MC {b_mc}%→{l_mc}%. Do not add L2 gate."
        )
    return (
        f"Binance L2 filter looked similar to candles alone "
        f"(OOS {b_er}→{l_er}, MC {b_mc}%→{l_mc}%). No clear reason to change."
    )


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=91)).isoformat()
    end_s = end.isoformat()
    print(
        f"L2 OOS/WF/MC research {start}→{end_s}  ($200/R · 1h brains · no live tweaks)",
        flush=True,
    )

    rows = []
    for sym in BOOK:
        print(f"\n######## {sym} ########", flush=True)
        row: dict[str, Any] = {
            "symbol": sym,
            "brain": BRAINS[sym],
            "l2_available": sym in BINANCE_BOOK,
            "l2_source": (
                f"Binance USDM bookDepth ({BINANCE_BOOK[sym]})"
                if sym in BINANCE_BOOK
                else None
            ),
        }
        try:
            raw = _fetch_1h(sym, start, end_s)
            df = _prep(raw)
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            brain, trades = gen_trades(sym, df)
            print(f"  baseline trades={len(trades)}", flush=True)
            base = oos_wf_mc(trades, len(df), "L1_candle_only")
            row["baseline"] = base
            print(
                f"  BASE OOS E[R]={base['oos'].get('exp_r')} "
                f"WF {base['wf_green']}/{base['wf_total']} "
                f"MC={base['monte_carlo'].get('prob_profit')}% "
                f"→ {base['verdict']}",
                flush=True,
            )

            l2_pack = None
            if sym in BINANCE_BOOK and trades:
                days = sorted(
                    {
                        str(
                            (
                                pd.Timestamp(t["entry_t"]).tz_localize("UTC")
                                if pd.Timestamp(t["entry_t"]).tzinfo is None
                                else pd.Timestamp(t["entry_t"]).tz_convert("UTC")
                            )
                        )[:10]
                        for t in trades
                    }
                )
                # also pull ±1 day for coverage
                extra = set(days)
                for d in list(days):
                    dt = datetime.fromisoformat(d)
                    extra.add((dt - timedelta(days=1)).date().isoformat())
                    extra.add((dt + timedelta(days=1)).date().isoformat())
                days2 = sorted(extra)
                print(f"  loading L2 bookDepth days={len(days2)}", flush=True)
                imb = load_book_imbalance(BINANCE_BOOK[sym], days2)
                if imb is not None and len(imb):
                    filtered = l2_filter(trades, imb)
                    print(f"  L2-kept trades={len(filtered)}/{len(trades)}", flush=True)
                    l2_pack = oos_wf_mc(filtered, len(df), "L2_book_confirm")
                    row["l2_filtered"] = l2_pack
                    row["l2_keep_pct"] = round(
                        100.0 * len(filtered) / max(1, len(trades)), 1
                    )
                    print(
                        f"  L2   OOS E[R]={l2_pack['oos'].get('exp_r')} "
                        f"WF {l2_pack['wf_green']}/{l2_pack['wf_total']} "
                        f"MC={l2_pack['monte_carlo'].get('prob_profit')}% "
                        f"→ {l2_pack['verdict']}",
                        flush=True,
                    )
                else:
                    row["l2_error"] = "bookDepth download/parse failed"
            row["layman"] = layman(sym, bool(l2_pack), base, l2_pack)
            print(f"  → {row['layman']}", flush=True)
        except Exception as exc:
            row["error"] = str(exc)
            row["layman"] = f"Failed: {exc}"
            print(f"  FAIL {exc}", flush=True)
        rows.append(row)

    payload = {
        "window": {"start": start, "end": end_s, "tf": "1h_proxy"},
        "risk_usd": RISK_USD,
        "note": (
            "True free L2 only for BTC/ETH via Binance bookDepth. "
            "FX/gold/NAS/oil have no free L2 — baseline only. No live changes."
        ),
        "results": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### L2 OOS / WF / MC SCORECARD\n", flush=True)
    print(
        "| Pair | L2? | Base OOS | Base WF | Base MC | Base | "
        "L2 OOS | L2 WF | L2 MC | L2 | Layman |",
        flush=True,
    )
    print("|---|---|---:|---:|---:|---|---:|---:|---:|---|---|", flush=True)
    for r in rows:
        b = r.get("baseline") or {}
        l = r.get("l2_filtered") or {}
        mc_b = (b.get("monte_carlo") or {}).get("prob_profit", "—")
        mc_l = (l.get("monte_carlo") or {}).get("prob_profit", "—")
        print(
            f"| {r['symbol']} | {r.get('l2_available')} | "
            f"{(b.get('oos') or {}).get('exp_r', '—')} | "
            f"{b.get('wf_green', '—')}/{b.get('wf_total', '—')} | {mc_b}% | "
            f"{b.get('verdict', '—')} | "
            f"{(l.get('oos') or {}).get('exp_r', '—')} | "
            f"{l.get('wf_green', '—')}/{l.get('wf_total', '—')} | {mc_l}% | "
            f"{l.get('verdict', '—')} | {(r.get('layman') or '')[:70]} |",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
