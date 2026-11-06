#!/usr/bin/env python3
"""RESEARCH ONLY — Tweak GOLD 15m OB+FVG → 5m entry. No live wiring.

Prior baseline was FRAGILE (full +$5.4k, OOS $0). This bake-off tightens:
  overlap · fresh age · EMA trend · stronger disp · 50% mean · session · RR
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import (  # noqa: E402
    RISK_USD,
    _fetch_1h,
    _prep,
    _stats,
    monte_carlo,
    verdict,
)
from analysis._htf_trend_5m_exit_bt import _fetch_5m  # noqa: E402
from analysis._rr_best_by_pair import _collect_fib, simulate  # noqa: E402
from analysis.levy_malliavin import levy_malliavin_pack  # noqa: E402
from analysis.stochastic_pack import stochastic_pack  # noqa: E402
from config import symbol_rr  # noqa: E402
from indicators.prepare import prepare_df  # noqa: E402

OUT = Path("/tmp/gold_ob_fvg_tweak_battery.json")
RR_LIVE = float(symbol_rr("GOLD", default=4.0))
SL_PAD = 0.08


def _resample_ohlc(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    m = {"5m": "5min", "15m": "15min"}[rule]
    out = pd.DataFrame(
        {
            "open": df5["open"].resample(m).first(),
            "high": df5["high"].resample(m).max(),
            "low": df5["low"].resample(m).min(),
            "close": df5["close"].resample(m).last(),
        }
    ).dropna()
    return prepare_df(out)


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(hour=23, minute=59, tzinfo=timezone.utc)
    return df[(df.index >= start_dt) & (df.index <= end_dt)].copy()


def _build_zones(df15: pd.DataFrame, disp_atr: float) -> list[dict]:
    zones: list[dict] = []
    for i in range(5, len(df15)):
        row = df15.iloc[i]
        atr = float(row["ATR"] or 0) or 1e-6
        for side, lo_k, hi_k in (
            ("BUY", "BULL_FVG_LOW", "BULL_FVG_HIGH"),
            ("SELL", "BEAR_FVG_LOW", "BEAR_FVG_HIGH"),
        ):
            lo, hi = row.get(lo_k), row.get(hi_k)
            if pd.notna(lo) and pd.notna(hi):
                zones.append(
                    {
                        "kind": "FVG",
                        "side": side,
                        "i": i,
                        "t": df15.index[i],
                        "low": float(lo),
                        "high": float(hi),
                        "mid": 0.5 * (float(lo) + float(hi)),
                    }
                )

        body = float(row["BODY"] or 0)
        br = float(row["BODY_RATIO"] or 0)
        bull_d = float(row["close"]) > float(row["open"]) and body >= disp_atr * atr and br >= 0.55
        bear_d = float(row["close"]) < float(row["open"]) and body >= disp_atr * atr and br >= 0.55
        if bull_d:
            for k in range(1, 5):
                j = i - k
                if j < 1:
                    break
                prev = df15.iloc[j]
                if float(prev["close"]) < float(prev["open"]):
                    zones.append(
                        {
                            "kind": "OB",
                            "side": "BUY",
                            "i": i,
                            "t": df15.index[i],
                            "low": float(prev["low"]),
                            "high": float(prev["high"]),
                            "mid": 0.5 * (float(prev["low"]) + float(prev["high"])),
                        }
                    )
                    break
        if bear_d:
            for k in range(1, 5):
                j = i - k
                if j < 1:
                    break
                prev = df15.iloc[j]
                if float(prev["close"]) > float(prev["open"]):
                    zones.append(
                        {
                            "kind": "OB",
                            "side": "SELL",
                            "i": i,
                            "t": df15.index[i],
                            "low": float(prev["low"]),
                            "high": float(prev["high"]),
                            "mid": 0.5 * (float(prev["low"]) + float(prev["high"])),
                        }
                    )
                    break
    return zones


def _overlap(a: dict, b: dict) -> Optional[dict]:
    lo = max(a["low"], b["low"])
    hi = min(a["high"], b["high"])
    if hi <= lo:
        return None
    return {
        "kind": "OB∩FVG",
        "side": a["side"],
        "i": max(a["i"], b["i"]),
        "t": max(a["t"], b["t"]),
        "low": lo,
        "high": hi,
        "mid": 0.5 * (lo + hi),
    }


def _combined_zones(
    zones: list[dict],
    pos15: int,
    *,
    max_age: int,
    require_overlap: bool,
) -> list[dict]:
    lo = pos15 - max_age
    cand = [z for z in zones if lo <= z["i"] <= pos15]
    out: list[dict] = []
    for side in ("BUY", "SELL"):
        obs = [z for z in cand if z["kind"] == "OB" and z["side"] == side]
        fvgs = [z for z in cand if z["kind"] == "FVG" and z["side"] == side]
        if not obs or not fvgs:
            continue
        # freshest pair
        ob, fvg = obs[-1], fvgs[-1]
        if require_overlap:
            ov = _overlap(ob, fvg)
            if ov is None:
                # try any overlapping pair among recent
                found = None
                for o in reversed(obs[-4:]):
                    for f in reversed(fvgs[-6:]):
                        found = _overlap(o, f)
                        if found:
                            break
                    if found:
                        break
                if not found:
                    continue
                out.append(found)
            else:
                out.append(ov)
        else:
            out.append(
                {
                    "kind": "OB+FVG",
                    "side": side,
                    "i": max(ob["i"], fvg["i"]),
                    "t": max(ob["t"], fvg["t"]),
                    "low": min(ob["low"], fvg["low"]),
                    "high": max(ob["high"], fvg["high"]),
                    "mid": 0.5 * (ob["mid"] + fvg["mid"]),
                }
            )
    return out


def _mitigated(df15: pd.DataFrame, z: dict, pos15: int) -> bool:
    """True if zone fully traded through after formation (close beyond far side)."""
    for j in range(z["i"] + 1, pos15 + 1):
        c = float(df15.iloc[j]["close"])
        if z["side"] == "BUY" and c < z["low"]:
            return True
        if z["side"] == "SELL" and c > z["high"]:
            return True
    return False


def collect(
    df15: pd.DataFrame,
    df5: pd.DataFrame,
    zones: list[dict],
    *,
    max_age: int = 40,
    require_overlap: bool = False,
    ema_trend: bool = False,
    mean_frac: float = 0.50,  # reclaim past this fraction of zone
    session_only: bool = False,  # London+NY 07–20 UTC
    unmitigated: bool = False,
    min_zone_atr: float = 0.0,
) -> list[dict]:
    entries: list[dict] = []
    for ei in range(40, len(df5) - 1):
        ts = df5.index[ei]
        if session_only:
            h = int(ts.tz_convert("UTC").hour)
            if h < 7 or h >= 20:
                continue
        pos15 = int(df15.index.searchsorted(ts, side="right")) - 1
        if pos15 < 55:
            continue
        active = _combined_zones(
            zones, pos15, max_age=max_age, require_overlap=require_overlap
        )
        if not active:
            continue

        row5 = df5.iloc[ei]
        row15 = df15.iloc[pos15]
        hh, ll, c, o = (
            float(row5["high"]),
            float(row5["low"]),
            float(row5["close"]),
            float(row5["open"]),
        )
        atr5 = float(row5["ATR"] or 0) or max(hh - ll, 1e-6)
        atr15 = float(row15["ATR"] or atr5)

        if ema_trend:
            e20, e50 = float(row15["EMA20"]), float(row15["EMA50"])
        else:
            e20 = e50 = None

        signal = None
        chosen = None
        entry = sl = 0.0
        for z in reversed(active):
            if unmitigated and _mitigated(df15, z, pos15):
                continue
            width = z["high"] - z["low"]
            if width < min_zone_atr * atr15:
                continue
            if ema_trend:
                if z["side"] == "BUY" and not (e20 > e50):
                    continue
                if z["side"] == "SELL" and not (e20 < e50):
                    continue

            tapped = ll <= z["high"] and hh >= z["low"]
            if not tapped:
                continue
            # threshold inside zone
            if z["side"] == "BUY":
                thr = z["low"] + mean_frac * width
                if c > thr and c > o and ll <= z["high"]:
                    signal = "BUY"
                    entry = c
                    sl = min(ll, z["low"]) - SL_PAD * atr5
                    chosen = z
                    break
            else:
                thr = z["high"] - mean_frac * width
                if c < thr and c < o and hh >= z["low"]:
                    signal = "SELL"
                    entry = c
                    sl = max(hh, z["high"]) + SL_PAD * atr5
                    chosen = z
                    break
        if not signal or chosen is None:
            continue
        risk = abs(entry - sl)
        if risk <= 0.12 * atr5 or risk > 2.8 * atr5:
            continue
        entries.append(
            {
                "ei": ei,
                "side": signal,
                "entry": float(entry),
                "sl": float(sl),
                "entry_t": ts,
                "htf_i": pos15,
            }
        )
    return entries


def simulate_5m(df5: pd.DataFrame, entries: list[dict], rr: float) -> list[dict]:
    trades: list[dict] = []
    busy = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < busy:
            continue
        side, entry, sl0 = e["side"], float(e["entry"]), float(e["sl"])
        risk = abs(entry - sl0)
        if risk <= 0:
            continue
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk
        exit_r = None
        exit_j = ei
        for j in range(ei + 1, len(df5)):
            row = df5.iloc[j]
            h, l = float(row["high"]), float(row["low"])
            hit_sl = (side == "BUY" and l <= sl) or (side == "SELL" and h >= sl)
            hit_tp = (side == "BUY" and h >= tp) or (side == "SELL" and l <= tp)
            if hit_sl or hit_tp:
                exit_r = -1.0 if hit_sl else float(rr)
                if hit_sl and hit_tp:
                    exit_r = -1.0
                exit_j = j
                break
        if exit_r is None:
            continue
        trades.append(
            {
                "i": int(e.get("htf_i", ei)),
                "entry_t": str(e["entry_t"])[:16],
                "side": side,
                "r": round(exit_r, 4),
                "pnl": round(exit_r * RISK_USD, 2),
            }
        )
        busy = exit_j + 2
    return trades


def _battery(full_tr, df_ito, n_fold, arm, label, rr) -> dict:
    split = int(n_fold * 0.70)

    def slice_tr(start_i=0, end_i=None):
        end_i = end_i if end_i is not None else 10**9
        return [t for t in full_tr if start_i <= int(t["i"]) <= end_i]

    full = _stats(full_tr)
    oos = _stats(slice_tr(start_i=split))
    wf = []
    for k in range(4):
        a = int(n_fold * (0.40 + 0.15 * k))
        b = int(n_fold * (0.55 + 0.15 * k)) - 1
        if b <= a or a < 40:
            continue
        wf.append(
            {
                "fold": k + 1,
                "oos": _stats(slice_tr(start_i=a, end_i=b)),
            }
        )
    torture = {
        "spread_0.15R": _stats(
            [
                {
                    **t,
                    "r": round(t["r"] - 0.15, 4),
                    "pnl": round((t["r"] - 0.15) * RISK_USD, 2),
                }
                for t in full_tr
            ]
        )
    }
    mc = monte_carlo(full_tr, paths=1200)
    lm = levy_malliavin_pack(full_tr)
    stoch = stochastic_pack(df_ito, full_tr)
    v = verdict(full, oos, wf, torture, mc)
    wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
    return {
        "arm": arm,
        "label": label,
        "rr": rr,
        "full": full,
        "oos": oos,
        "wf_green": f"{wf_g}/{len(wf)}",
        "monte_carlo": mc,
        "levy": lm["levy"],
        "malliavin": lm["malliavin"],
        "brownian": stoch["brownian"],
        "ito": stoch["ito"],
        "martingale": stoch["martingale"],
        "hjm": stoch["hjm"],
        "verdict": v,
        "pnl_usd": full.get("pnl"),
        "oos_pnl_usd": oos.get("pnl"),
    }


# Tweak grid (research)
TWEAKS: list[dict[str, Any]] = [
    {
        "arm": "baseline",
        "label": "prior OB+FVG union age40",
        "disp": 1.15,
        "max_age": 40,
        "overlap": False,
        "ema": False,
        "mean": 0.50,
        "session": False,
        "unmit": False,
        "min_z": 0.0,
        "rr": RR_LIVE,
    },
    {
        "arm": "overlap",
        "label": "OB∩FVG price overlap only",
        "disp": 1.15,
        "max_age": 40,
        "overlap": True,
        "ema": False,
        "mean": 0.50,
        "session": False,
        "unmit": False,
        "min_z": 0.0,
        "rr": RR_LIVE,
    },
    {
        "arm": "fresh16",
        "label": "fresh zones ≤16 bars (~4h)",
        "disp": 1.15,
        "max_age": 16,
        "overlap": False,
        "ema": False,
        "mean": 0.50,
        "session": False,
        "unmit": False,
        "min_z": 0.0,
        "rr": RR_LIVE,
    },
    {
        "arm": "ema_trend",
        "label": "EMA20/50 15m trend filter",
        "disp": 1.15,
        "max_age": 40,
        "overlap": False,
        "ema": True,
        "mean": 0.50,
        "session": False,
        "unmit": False,
        "min_z": 0.0,
        "rr": RR_LIVE,
    },
    {
        "arm": "overlap_ema_fresh",
        "label": "overlap + EMA + age16",
        "disp": 1.15,
        "max_age": 16,
        "overlap": True,
        "ema": True,
        "mean": 0.50,
        "session": False,
        "unmit": False,
        "min_z": 0.0,
        "rr": RR_LIVE,
    },
    {
        "arm": "strict_stack",
        "label": "overlap+EMA+age16+unmit+minW+sess",
        "disp": 1.35,
        "max_age": 16,
        "overlap": True,
        "ema": True,
        "mean": 0.50,
        "session": True,
        "unmit": True,
        "min_z": 0.15,
        "rr": RR_LIVE,
    },
    {
        "arm": "disp_strong",
        "label": "disp 1.5ATR + overlap + unmit",
        "disp": 1.50,
        "max_age": 24,
        "overlap": True,
        "ema": True,
        "mean": 0.50,
        "session": False,
        "unmit": True,
        "min_z": 0.10,
        "rr": RR_LIVE,
    },
    {
        "arm": "mean62",
        "label": "overlap + reclaim past 62% of zone",
        "disp": 1.15,
        "max_age": 24,
        "overlap": True,
        "ema": True,
        "mean": 0.62,
        "session": False,
        "unmit": True,
        "min_z": 0.0,
        "rr": RR_LIVE,
    },
    {
        "arm": "best_rr2",
        "label": "strict_stack @ RR 1:2",
        "disp": 1.35,
        "max_age": 16,
        "overlap": True,
        "ema": True,
        "mean": 0.50,
        "session": True,
        "unmit": True,
        "min_z": 0.15,
        "rr": 2.0,
    },
    {
        "arm": "best_rr3",
        "label": "strict_stack @ RR 1:3",
        "disp": 1.35,
        "max_age": 16,
        "overlap": True,
        "ema": True,
        "mean": 0.50,
        "session": True,
        "unmit": True,
        "min_z": 0.15,
        "rr": 3.0,
    },
]


def main() -> int:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_5 = (end - timedelta(days=55)).isoformat()
    start_1h = (end - timedelta(days=182)).isoformat()
    end_s = end.isoformat()
    print(
        f"GOLD OB+FVG TWEAK bake-off · {start_5}→{end_s} · ${RISK_USD}/R · NO LIVE WIRE",
        flush=True,
    )

    raw5 = _fetch_5m("GOLD", start_5, end_s)
    clipped = _clip(raw5, start_5, end_s)
    df5 = _resample_ohlc(clipped, "5m")
    df15 = _resample_ohlc(clipped, "15m")
    print(f"5m={len(df5)} 15m={len(df15)}", flush=True)

    # cache zones by disp
    zone_cache: dict[float, list] = {}
    results = []

    for tw in TWEAKS:
        disp = float(tw["disp"])
        if disp not in zone_cache:
            zone_cache[disp] = _build_zones(df15, disp)
        zones = zone_cache[disp]
        entries = collect(
            df15,
            df5,
            zones,
            max_age=int(tw["max_age"]),
            require_overlap=bool(tw["overlap"]),
            ema_trend=bool(tw["ema"]),
            mean_frac=float(tw["mean"]),
            session_only=bool(tw["session"]),
            unmitigated=bool(tw["unmit"]),
            min_zone_atr=float(tw["min_z"]),
        )
        rr = float(tw["rr"])
        trades = simulate_5m(df5, entries, rr)
        print(
            f"\n{tw['arm']} ({tw['label']}) entries={len(entries)} fills={len(trades)} RR=1:{rr:g}",
            flush=True,
        )
        row = _battery(trades, df5, len(df15), tw["arm"], tw["label"], rr)
        row["raw_entries"] = len(entries)
        row["fills"] = len(trades)
        f, o = row["full"], row["oos"]
        print(
            f"  n={f['n']} E[R]={f['exp_r']} pnl=${f['pnl']} | "
            f"OOS E[R]={o['exp_r']} pnl=${o['pnl']} | WF {row['wf_green']} | "
            f"MC={row['monte_carlo'].get('prob_profit')}% | {row['verdict']}",
            flush=True,
        )
        results.append(row)

    # fib control same window
    raw1 = _fetch_1h("GOLD", start_5, end_s)
    df1 = _prep(_clip(raw1, start_5, end_s))
    e1 = _collect_fib(df1)
    tr1 = simulate(df1, e1, RR_LIVE)
    row = _battery(tr1, df1, len(df1), "fib_1h_55d", "current fib control 55d", RR_LIVE)
    row["raw_entries"] = len(e1)
    row["fills"] = len(tr1)
    print(
        f"\nfib_1h_55d fills={len(tr1)} E[R]={row['full']['exp_r']} "
        f"pnl=${row['full']['pnl']} OOS=${row['oos']['pnl']} | {row['verdict']}",
        flush=True,
    )
    results.append(row)

    raw1L = _fetch_1h("GOLD", start_1h, end_s)
    df1L = _prep(_clip(raw1L, start_1h, end_s))
    e1L = _collect_fib(df1L)
    tr1L = simulate(df1L, e1L, RR_LIVE)
    rowL = _battery(
        tr1L, df1L, len(df1L), "fib_1h_182d", "current fib control 182d", RR_LIVE
    )
    rowL["raw_entries"] = len(e1L)
    rowL["fills"] = len(tr1L)
    print(
        f"fib_1h_182d fills={len(tr1L)} E[R]={rowL['full']['exp_r']} "
        f"pnl=${rowL['full']['pnl']} OOS=${rowL['oos']['pnl']} | {rowL['verdict']}",
        flush=True,
    )
    results.append(rowL)

    ranked = sorted(
        [r for r in results if not str(r["arm"]).startswith("fib")],
        key=lambda r: (
            float(r.get("oos_pnl_usd") or -1e9),
            float((r.get("full") or {}).get("exp_r") or -1e9),
        ),
        reverse=True,
    )

    payload = {
        "note": "RESEARCH ONLY — no live config changes",
        "window": {"start": start_5, "end": end_s},
        "results": results,
        "rank_oos": [r["arm"] for r in ranked],
        "best_tweak": ranked[0]["arm"] if ranked else None,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n### OB+FVG TWEAK BAKE-OFF\n", flush=True)
    print(
        "| Arm | RR | n | Full E[R] | Full PnL | OOS PnL | WF | MC% | Lévy% | Mall | Verdict |",
        flush=True,
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|", flush=True)
    for r in results:
        f = r["full"]
        print(
            f"| {r['arm']} | 1:{r['rr']:g} | {f['n']} | {f['exp_r']} | "
            f"${f['pnl']:+.0f} | ${r['oos']['pnl']:+.0f} | {r['wf_green']} | "
            f"{r['monte_carlo'].get('prob_profit')} | {r['levy'].get('prob_profit')} | "
            f"{r['malliavin'].get('fragility')} | **{r['verdict']}** |",
            flush=True,
        )
    print("\n### RANK (OB+FVG tweaks by OOS PnL)\n", flush=True)
    for i, r in enumerate(ranked, 1):
        print(
            f"{i}. {r['arm']}: OOS ${r['oos_pnl_usd']:+.0f} · Full ${r['pnl_usd']:+.0f} · {r['verdict']}",
            flush=True,
        )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
