#!/usr/bin/env python3
"""RESEARCH ONLY — Manual demand-sweep/OB reclaim vs live GOLD PA brain.

Manual strategy (from live chart punch 2026-08-06):
  15m demand / bullish OB ← liquidity sweep (wick below zone)
  → reclaim close back above zone mid on 5m (or 1m)
  → SL under sweep wick
  → TP fixed RR=2  OR  opposing bearish OB (structure)

Live PA (wired):
  liq=3m sweep → bias 15m+5m EMA agree → 1m engulf · RR=2

Windows: 2m / 6m / 12m / 2y · Dukascopy · $200/R · 8-test battery + BE timing.
Does NOT change live wiring.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD, _prep  # noqa: E402
from analysis._gold_ob_fvg_15_enter5_bt import (  # noqa: E402
    OB_MAX_AGE_BARS,
    SL_PAD,
    _active_zones,
    _build_zones,
)
from analysis.lab_agentic_1y_battery import _battery_from_trades  # noqa: E402
from analysis.lab_gold_pa_liq_engulf_bt import _is_engulf, _resample_ohlc  # noqa: E402
from analysis.lab_gold_pa_tf_grid_bt import (  # noqa: E402
    _collect_entries,
    _fast_sweeps,
)
from analysis.lab_usdchf_pa_horizons_tf_bt import (  # noqa: E402
    _equity_milestones,
    _fmt_ms,
    _parse_t,
)
from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv  # noqa: E402
from indicators.prepare import prepare_df  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_gold_demand_sweep_vs_pa_bt.json"
LOG = Path(ROOT) / "data" / "lab_gold_demand_sweep_vs_pa_bt.log"

SYM = "GOLD"
WINDOWS = (("2m", 61), ("6m", 183), ("12m", 365), ("2y", 730))
PA_LIVE = {
    "liq_tf": "3m",
    "bias_a": "15m",
    "bias_b": "5m",
    "entry_tf": "1m",
    "rr": 2.0,
}
SWEEP_PAD_ATR = 0.02  # pierce below zone
RECLAIM_BARS = 12  # 5m bars (~1h) to reclaim after sweep


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _strip_index_tz(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if getattr(df.index, "tz", None) is not None:
        out = df.copy()
        out.index = out.index.tz_convert("UTC").tz_localize(None)
        return out
    return df


def _load_frames(window_days: int) -> tuple[dict[str, pd.DataFrame], pd.Timestamp, pd.Timestamp]:
    clear_hist_cache()
    probe = load_local_ohlcv(SYM, tf="5m") if has_local_hist(SYM, "5m") else None
    if probe is None or len(probe) < 400:
        raise RuntimeError("GOLD 5m hist missing")
    # Hist CSVs are UTC-aware — keep bounds UTC-aware for load_local_ohlcv
    data_end_aware = pd.Timestamp(probe.index[-1])
    if data_end_aware.tzinfo is None:
        data_end_aware = data_end_aware.tz_localize("UTC")
    else:
        data_end_aware = data_end_aware.tz_convert("UTC")
    start_aware = data_end_aware - pd.Timedelta(days=window_days)
    load_start_aware = start_aware - pd.Timedelta(days=14)
    data_end = _to_naive(data_end_aware)
    start = _to_naive(start_aware)

    frames: dict[str, pd.DataFrame] = {}
    for tf in ("1m", "3m", "5m", "15m"):
        if not has_local_hist(SYM, tf):
            continue
        df = load_local_ohlcv(
            SYM,
            tf=tf,
            start=load_start_aware.isoformat(),
            end=data_end_aware.isoformat(),
        )
        if df is None or len(df) < 200:
            continue
        df = _strip_index_tz(df)
        need = [c for c in ("open", "high", "low", "close") if c in df.columns]
        df = _prep(df[need].dropna())
        try:
            df = prepare_df(df)
        except Exception:
            pass
        warm = start - pd.Timedelta(days=7)
        frames[tf] = df[df.index >= warm]
        _log(
            f"  load {SYM} {tf}: {len(frames[tf]):,} "
            f"{frames[tf].index[0]}→{frames[tf].index[-1]}"
        )
    for tf, rule in (("3m", "3min"), ("5m", "5min"), ("15m", "15min")):
        if tf not in frames:
            src = frames.get("1m") or frames.get("5m")
            frames[tf] = prepare_df(_resample_ohlc(src, rule))
            _log(f"  resample {tf}: {len(frames[tf]):,}")
    return frames, start, data_end


def _nearest_opposing_ob(
    zones: list[dict],
    pos15: int,
    side: str,
    entry: float,
) -> Optional[float]:
    """Nearest opposing OB mid/edge for structure TP."""
    active = _active_zones(zones, pos15, kind="OB")
    opp = "SELL" if side == "BUY" else "BUY"
    cands = [z for z in active if z["side"] == opp]
    if side == "BUY":
        above = [z for z in cands if z["low"] > entry]
        if not above:
            return None
        z = min(above, key=lambda z: z["low"])
        return float(z["low"])  # tap into bearish OB
    below = [z for z in cands if z["high"] < entry]
    if not below:
        return None
    z = max(below, key=lambda z: z["high"])
    return float(z["high"])


def _collect_demand_sweep(
    df15: pd.DataFrame,
    df_e: pd.DataFrame,
    zones: list[dict],
    *,
    mode: str,  # 'ob' | 'both'
    window_start: pd.Timestamp,
    tp_mode: str,  # 'rr2' | 'struct'
    fixed_rr: float = 2.0,
) -> list[dict]:
    """
    Sweep into demand/supply OB → reclaim → entry.
    BUY: wick below zone.low, then close > zone.mid within RECLAIM_BARS.
    """
    entries: list[dict] = []
    pending: list[dict] = []  # open sweep arms awaiting reclaim

    for ei in range(40, len(df_e) - 1):
        ts = df_e.index[ei]
        if ts < window_start:
            continue
        pos15 = int(df15.index.searchsorted(ts, side="right")) - 1
        if pos15 < 10:
            continue
        if mode == "both":
            active = _active_zones(zones, pos15, require_both=True)
        else:
            active = _active_zones(zones, pos15, kind="OB")
        if not active:
            pending = [p for p in pending if ei - p["sweep_ei"] <= RECLAIM_BARS]
            continue

        row = df_e.iloc[ei]
        h, l, c, o = (
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["open"]),
        )
        atr = float(row.get("atr") or row.get("ATR") or 0) or max(h - l, 1e-6)

        # 1) Detect new sweeps into zones
        for z in reversed(active):
            if z["side"] == "BUY":
                pierced = l < (z["low"] - SWEEP_PAD_ATR * atr)
                if pierced:
                    pending.append(
                        {
                            "side": "BUY",
                            "zone": z,
                            "sweep_ei": ei,
                            "sweep_low": l,
                            "pos15": pos15,
                        }
                    )
            else:
                pierced = h > (z["high"] + SWEEP_PAD_ATR * atr)
                if pierced:
                    pending.append(
                        {
                            "side": "SELL",
                            "zone": z,
                            "sweep_ei": ei,
                            "sweep_high": h,
                            "pos15": pos15,
                        }
                    )

        # expire
        pending = [p for p in pending if ei - p["sweep_ei"] <= RECLAIM_BARS]

        # 2) Reclaim entries (no EMA bias — that's the point vs PA)
        hit = None
        for p in reversed(pending):
            z = p["zone"]
            if p["side"] == "BUY":
                reclaimed = c > z["mid"] and c > o and l <= z["high"]
                if not reclaimed:
                    continue
                # prefer engulf if present, not required
                entry = c
                sl = min(float(p["sweep_low"]), l) - SL_PAD * atr
                if sl >= entry or abs(entry - sl) <= 0.12 * atr:
                    continue
                if abs(entry - sl) > 3.5 * atr:
                    continue
                hit = {
                    "ei": ei,
                    "side": "BUY",
                    "entry": entry,
                    "sl": sl,
                    "entry_t": ts,
                    "zone_kind": z["kind"],
                    "sweep_ei": p["sweep_ei"],
                    "pos15": p["pos15"],
                }
            else:
                reclaimed = c < z["mid"] and c < o and h >= z["low"]
                if not reclaimed:
                    continue
                entry = c
                sl = max(float(p["sweep_high"]), h) + SL_PAD * atr
                if sl <= entry or abs(entry - sl) <= 0.12 * atr:
                    continue
                if abs(entry - sl) > 3.5 * atr:
                    continue
                hit = {
                    "ei": ei,
                    "side": "SELL",
                    "entry": entry,
                    "sl": sl,
                    "entry_t": ts,
                    "zone_kind": z["kind"],
                    "sweep_ei": p["sweep_ei"],
                    "pos15": p["pos15"],
                }
            if hit:
                break
        if not hit:
            continue

        risk = abs(hit["entry"] - hit["sl"])
        if tp_mode == "struct":
            tp_px = _nearest_opposing_ob(zones, hit["pos15"], hit["side"], hit["entry"])
            if tp_px is None:
                rr = fixed_rr
            else:
                reward = abs(tp_px - hit["entry"])
                rr = reward / risk if risk > 0 else fixed_rr
                # only take if structure offers ≥1.5R; else fall back
                if rr < 1.5:
                    rr = fixed_rr
                else:
                    rr = min(rr, 4.0)
            hit["rr"] = float(rr)
            hit["tp_mode"] = "struct" if tp_px is not None and rr != fixed_rr else "rr_fallback"
        else:
            hit["rr"] = float(fixed_rr)
            hit["tp_mode"] = "fixed"

        entries.append(hit)
        # clear pending same side to avoid spam
        pending = [p for p in pending if p["side"] != hit["side"]]

    return entries


def _simulate_var_rr(df: pd.DataFrame, entries: list[dict]) -> list[dict]:
    """Per-entry RR (or default 2)."""
    trades: list[dict] = []
    busy = -1
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    idx = df.index
    n = len(df)
    for e in entries:
        ei = int(e["ei"])
        if ei < busy or ei >= n - 1:
            continue
        side = e["side"]
        entry = float(e["entry"])
        sl0 = float(e["sl"])
        rr = float(e.get("rr") or 2.0)
        risk = abs(entry - sl0)
        if risk <= 0:
            continue
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk
        exit_r = None
        exit_i = ei
        for j in range(ei + 1, n):
            hit_sl = (side == "BUY" and low[j] <= sl) or (side == "SELL" and high[j] >= sl)
            hit_tp = (side == "BUY" and high[j] >= tp) or (side == "SELL" and low[j] <= tp)
            if hit_sl or hit_tp:
                exit_r = -1.0 if hit_sl else rr
                if hit_sl and hit_tp:
                    exit_r = -1.0
                exit_i = j
                break
        if exit_r is None:
            continue
        trades.append(
            {
                "i": ei,
                "entry_t": str(e.get("entry_t", idx[ei]))[:19],
                "exit_t": str(idx[exit_i])[:19],
                "side": side,
                "r": round(float(exit_r), 4),
                "pnl": round(float(exit_r) * RISK_USD, 2),
                "rr_used": rr,
                "tp_mode": e.get("tp_mode"),
            }
        )
        busy = exit_i + 2
    return trades


def _pack_arm(
    name: str,
    df: pd.DataFrame,
    trades: list[dict],
    window_start: pd.Timestamp,
    meta: Optional[dict] = None,
) -> dict[str, Any]:
    bat = _battery_from_trades(trades, len(df))
    full = bat.get("full") or {}
    tests = bat.get("tests") or {}
    eq = _equity_milestones(trades, window_start)
    return {
        "name": name,
        **(meta or {}),
        "trades": full.get("n", 0),
        "pnl": full.get("pnl", 0.0),
        "exp_r": full.get("exp_r", 0.0),
        "win_rate": full.get("win_rate", 0.0),
        "verdict": bat.get("verdict"),
        "wf_green": bat.get("wf_green"),
        "wf_n": bat.get("wf_n"),
        "mc_prob_profit": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "equity": eq,
        "tests": {
            k: {
                "n": (tests.get(k) or {}).get("n"),
                "pnl": (tests.get(k) or {}).get("pnl"),
                "exp_r": (tests.get(k) or {}).get("exp_r"),
            }
            for k in (
                "1_full",
                "2_oos70",
                "3_wf_fold_1",
                "4_wf_fold_2",
                "5_wf_fold_3",
                "6_wf_fold_4",
                "7_torture_spread",
                "8_torture_skip",
            )
        },
    }


def _run_pa_live(
    frames: dict[str, pd.DataFrame], window_start: pd.Timestamp
) -> dict[str, Any]:
    df_e = frames["1m"]
    sweeps = [
        s
        for s in _fast_sweeps(frames["3m"])
        if _parse_t(s.get("t")) is None or _parse_t(s["t"]) >= window_start
    ]
    entries = _collect_entries(
        df_e, sweeps, frames["15m"], frames["5m"], "1m"
    )
    entries = [
        e
        for e in entries
        if _parse_t(e.get("entry_t")) is None or _parse_t(e["entry_t"]) >= window_start
    ]
    for e in entries:
        e["rr"] = PA_LIVE["rr"]
    trades = _simulate_var_rr(df_e, entries)
    return _pack_arm(
        "pa_live_liq3m_bias15+5_entry1m_rr2",
        df_e,
        trades,
        window_start,
        meta={**PA_LIVE, "family": "pa_liq_engulf"},
    )


def _run_window(label: str, days: int) -> dict[str, Any]:
    t0 = time.time()
    _log(f"\n{'='*64}\n===== WINDOW {label} ({days}d) =====")
    frames, start, end = _load_frames(days)
    _log(f"  trade window: {start.date()} → {end.date()}")

    df15 = frames["15m"]
    # OB zones on full loaded 15m (includes warmup)
    t1 = time.time()
    zones = _build_zones(df15)
    _log(f"  15m zones built: {len(zones)} ({time.time()-t1:.1f}s)")

    arms: list[dict] = []

    # --- Live PA ---
    t1 = time.time()
    pa = _run_pa_live(frames, start)
    arms.append(pa)
    eq = pa.get("equity") or {}
    _log(
        f"  PA live: n={pa['trades']} pnl=${pa['pnl']:,.0f} E[R]={pa['exp_r']} "
        f"{pa['verdict']} ({time.time()-t1:.1f}s) | "
        f"BE={_fmt_ms(eq.get('be_after_dd'))} locked={_fmt_ms(eq.get('locked_be'))}"
    )

    # --- Manual demand-sweep variants ---
    variants = [
        ("demand_sweep_ob_5m_rr2", "5m", "ob", "rr2", 2.0),
        ("demand_sweep_ob_5m_struct", "5m", "ob", "struct", 2.0),
        ("demand_sweep_obfvg_5m_rr2", "5m", "both", "rr2", 2.0),
        ("demand_sweep_ob_1m_rr2", "1m", "ob", "rr2", 2.0),
        ("demand_sweep_ob_5m_rr2.4", "5m", "ob", "rr2", 2.4),
    ]
    for name, etf, mode, tp_mode, rr in variants:
        t1 = time.time()
        df_e = frames[etf]
        ents = _collect_demand_sweep(
            df15,
            df_e,
            zones,
            mode=mode,
            window_start=start,
            tp_mode=tp_mode,
            fixed_rr=rr,
        )
        tr = _simulate_var_rr(df_e, ents)
        cell = _pack_arm(
            name,
            df_e,
            tr,
            start,
            meta={
                "family": "demand_sweep_ob",
                "entry_tf": etf,
                "zone_mode": mode,
                "tp_mode": tp_mode,
                "rr": rr,
            },
        )
        arms.append(cell)
        eq = cell.get("equity") or {}
        _log(
            f"  {name}: entries={len(ents)} n={cell['trades']} "
            f"pnl=${cell['pnl']:,.0f} E[R]={cell['exp_r']} {cell['verdict']} "
            f"({time.time()-t1:.1f}s) | BE={_fmt_ms(eq.get('be_after_dd'))} "
            f"→ profit={_fmt_ms(eq.get('profit_after_be'))} "
            f"locked={_fmt_ms(eq.get('locked_be'))} maxDD=${eq.get('max_dd_usd')}"
        )

    # rank
    def key(c):
        return (
            1 if c.get("verdict") == "SURVIVES" else 0,
            float(c.get("pnl") or 0),
            float(c.get("exp_r") or 0),
        )

    ranked = sorted(arms, key=key, reverse=True)
    best = ranked[0]
    _log(
        f"\n--- {label} BEST: {best['name']} pnl=${best['pnl']:,.0f} "
        f"{best['verdict']} E[R]={best['exp_r']}"
    )
    # delta manual best vs PA
    manuals = [a for a in arms if a.get("family") == "demand_sweep_ob" or "demand_sweep" in a["name"]]
    # fix family on pack
    for a in arms:
        if a["name"].startswith("demand_sweep"):
            a["family"] = "demand_sweep_ob"
        if a["name"].startswith("pa_live"):
            a["family"] = "pa_liq_engulf"
    manuals = [a for a in arms if a.get("family") == "demand_sweep_ob"]
    pa_arm = next(a for a in arms if a.get("family") == "pa_liq_engulf")
    man_best = max(manuals, key=key) if manuals else None
    delta = None
    if man_best:
        delta = round(float(man_best["pnl"]) - float(pa_arm["pnl"]), 2)
        _log(
            f"Δ best_manual − PA: ${delta:,.0f} "
            f"({man_best['name']} vs {pa_arm['name']})"
        )

    return {
        "label": label,
        "days": days,
        "start": str(start)[:19],
        "end": str(end)[:19],
        "arms": arms,
        "best": best,
        "pa_live": pa_arm,
        "best_manual": man_best,
        "delta_manual_minus_pa": delta,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass
    _log("=== GOLD demand-sweep/OB vs live PA · Dukascopy · RESEARCH ===")
    _log(f"Risk ${RISK_USD}/R · Live wiring NOT changed")
    _log(
        "Manual: 15m OB(+FVG) ← wick sweep → reclaim on 5m/1m · "
        "no EMA bias · TP RR2 or opposing OB"
    )
    _log(
        f"PA live: liq={PA_LIVE['liq_tf']} bias={PA_LIVE['bias_a']}+{PA_LIVE['bias_b']} "
        f"entry={PA_LIVE['entry_tf']} RR={PA_LIVE['rr']}"
    )

    windows = [_run_window(lab, d) for lab, d in WINDOWS]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "symbol": SYM,
        "risk_usd": RISK_USD,
        "manual_rules": {
            "zone": "15m bullish/bearish OB (optional OB+FVG)",
            "trigger": "wick pierce beyond zone + reclaim close > mid",
            "no_ema_bias": True,
            "sl": "under/over sweep wick + pad",
            "tp": "fixed RR or opposing OB",
        },
        "pa_live": PA_LIVE,
        "windows": windows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _log(f"\nWrote {OUT} ({payload['elapsed_s']}s)")

    _log("\n======== CROSS-HORIZON SCOREBOARD ========")
    _log(f"{'win':>4} {'arm':40} {'pnl':>10} {'n':>5} {'E[R]':>7} {'verdict':>10}")
    for w in windows:
        for a in w["arms"]:
            _log(
                f"{w['label']:>4} {a['name'][:40]:40} "
                f"${a['pnl']:>9,.0f} {a['trades']:>5} {a['exp_r']:>7.3f} "
                f"{a['verdict'] or '?':>10}"
            )
        _log(
            f"     → Δ best_manual−PA = ${w.get('delta_manual_minus_pa')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
