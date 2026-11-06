#!/usr/bin/env python3
"""RESEARCH ONLY — Adaptive regime router vs static invented math.

At each bar classify market state:
  TREND | FADE | FLAT
Then route to the matching formula family (or sit out).

Compares adaptive vs each pair's static invention winner on:
  · last 7d · 90d · 1y · 3y
Builds equity curves + chart payloads for canvas / Strategy Lab style graphs.

Does NOT change live brains.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD, _prep, _stats  # noqa: E402
from analysis.lab_agentic_1y_battery import _battery_from_trades  # noqa: E402
from analysis.lab_zenith_math_invention_deep_bt import (  # noqa: E402
    BOOK,
    _atr_sl_entries,
    _enrich,
    _trades_with_i,
)
from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv, local_span  # noqa: E402

HUNT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"
OUT = Path(ROOT) / "data" / "lab_adaptive_regime_router.json"
LOG = Path(ROOT) / "data" / "lab_adaptive_regime_router.log"
CHARTS = Path(ROOT) / "data" / "lab_adaptive_regime_charts.json"
PROGRESS = Path(ROOT) / "data" / "lab_adaptive_regime_router.progress.json"

# Primary TF for adaptive (balance speed vs resolution)
TF = "5m"
CAP_YEARS = 3.0
WINDOWS = (
    ("7d", 7),
    ("90d", 90),
    ("1y", 365),
    ("3y", 365 * 3),
)


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(payload: dict) -> None:
    PROGRESS.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load(sym: str, tf: str, years: float) -> pd.DataFrame:
    clear_hist_cache()
    if not has_local_hist(sym, tf):
        raise RuntimeError(f"no hist {sym} {tf}")
    sp = local_span(sym, tf)
    assert sp
    end = sp[1]
    start = max(sp[0], end - pd.Timedelta(days=int(years * 365.25) + 50))
    raw = load_local_ohlcv(sym, tf=tf, start=start.isoformat(), end=end.isoformat())
    if raw is None or len(raw) < 500:
        raise RuntimeError(f"short {sym} {tf}")
    return _enrich(_prep(raw[["open", "high", "low", "close"]].dropna()))


# ── Formula side generators ─────────────────────────────────────────────────


def _sides_snr(df: pd.DataFrame, thr: float = 0.40) -> np.ndarray:
    snr = df["snr"].to_numpy()
    ds = df["drift_sign"].to_numpy()
    return np.where((snr >= thr) & (ds != 0), ds.astype(int), 0)


def _sides_bimod(df: pd.DataFrame, thr: float = 0.55) -> np.ndarray:
    b = df["bimod"].to_numpy()
    ds = df["drift_sign"].to_numpy()
    return np.where((b >= thr) & (ds != 0), ds.astype(int), 0)


def _sides_perm(df: pd.DataFrame, pmax: float = 0.80) -> np.ndarray:
    pe = df["perm_ent"].to_numpy()
    ds = df["drift_sign"].to_numpy()
    return np.where((pe <= pmax) & (ds != 0), ds.astype(int), 0)


def _sides_kalman(df: pd.DataFrame, z: float = 2.0) -> np.ndarray:
    kz = df["kalman_z"].to_numpy()
    sides = np.zeros(len(df), dtype=int)
    sides[kz <= -z] = 1
    sides[kz >= z] = -1
    return sides


def _sides_hurst_trend(df: pd.DataFrame, hmin: float = 0.55) -> np.ndarray:
    H = df["hurst"].to_numpy()
    ds = df["drift_sign"].to_numpy()
    return np.where((H >= hmin) & (ds != 0), ds.astype(int), 0)


def _static_sides_for_winner(df: pd.DataFrame, wid: str) -> tuple[np.ndarray, str, float]:
    """Return sides, label, default sl from static winner id."""
    wid = str(wid or "")
    if "bimod" in wid:
        return _sides_bimod(df, 0.55), "static:bimod_0.55", 1.5
    if "perm_order" in wid:
        return _sides_perm(df, 0.80), "static:perm_order_0.80", 1.2
    if "snr_drift" in wid:
        return _sides_snr(df, 0.40), "static:snr_drift_0.40", 1.5
    if "kalman_fade_2.4" in wid or wid.endswith("2.4") or "2.4" in wid:
        return _sides_kalman(df, 2.4), "static:kalman_fade_2.4", 1.0
    if "kalman" in wid:
        return _sides_kalman(df, 2.0), "static:kalman_fade_2.0", 1.0
    return _sides_snr(df, 0.40), "static:snr_fallback", 1.2


def _classify_regime(df: pd.DataFrame) -> np.ndarray:
    """
    Per-bar regime:
      1 = TREND, -1 = FADE, 0 = FLAT
    """
    H = df["hurst"].to_numpy()
    snr = df["snr"].to_numpy()
    lf = df["fourier_lf"].to_numpy()
    hf = df["fourier_hf"].to_numpy()
    pe = df["perm_ent"].to_numpy()
    kz = np.abs(df["kalman_z"].to_numpy())
    fd = df["higuchi"].to_numpy()
    vr = df["vol_ratio"].to_numpy()

    trend = (
        (H >= 0.52)
        | ((snr >= 0.35) & (lf >= 0.55) & (pe <= 0.88))
        | ((snr >= 0.45) & (vr <= 1.15))
    )
    fade = (kz >= 1.8) & ((H <= 0.48) | (hf >= 0.58) | (fd >= 1.60))
    # conflict → FLAT (sit out) — key adaptive edge
    regime = np.zeros(len(df), dtype=int)
    only_trend = trend & ~fade
    only_fade = fade & ~trend
    regime[only_trend] = 1
    regime[only_fade] = -1
    return regime


def _adaptive_sides(df: pd.DataFrame, sym: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Route:
      TREND → crypto: bimod, metals: perm, index: snr, else snr|bimod vote
      FADE  → kalman
      FLAT  → 0
    """
    reg = _classify_regime(df)
    snr = _sides_snr(df, 0.40)
    bim = _sides_bimod(df, 0.55)
    perm = _sides_perm(df, 0.80)
    kal = _sides_kalman(df, 2.2)
    hur = _sides_hurst_trend(df, 0.55)

    if sym in ("ETHUSD", "BTCUSD"):
        trend_side = bim
        trend_name = "bimod"
    elif sym == "GOLD":
        trend_side = np.where(perm != 0, perm, snr)
        trend_name = "perm|snr"
    elif sym == "NAS100":
        trend_side = snr
        trend_name = "snr"
    else:
        # FX/oil: prefer snr, fallback hurst
        trend_side = np.where(snr != 0, snr, hur)
        trend_name = "snr|hurst"

    out = np.zeros(len(df), dtype=int)
    out[reg == 1] = trend_side[reg == 1]
    out[reg == -1] = kal[reg == -1]
    # FLAT stays 0

    meta = {
        "trend_book": trend_name,
        "fade_book": "kalman_2.2",
        "pct_trend": round(100.0 * float(np.mean(reg == 1)), 2),
        "pct_fade": round(100.0 * float(np.mean(reg == -1)), 2),
        "pct_flat": round(100.0 * float(np.mean(reg == 0)), 2),
    }
    return out, reg, meta


def _window_trades(
    df: pd.DataFrame, trades: list[dict], days: Optional[int]
) -> list[dict]:
    if days is None:
        return trades
    end = df.index[-1]
    start = end - pd.Timedelta(days=days)
    i0 = int(np.searchsorted(np.asarray(df.index), start))
    return [t for t in trades if int(t["i"]) >= i0]


def _equity_curve(trades: list[dict], df: pd.DataFrame, max_points: int = 400) -> dict:
    if not trades:
        return {"times": [], "equity": [], "final": 0.0}
    # sort by exit index
    ordered = sorted(trades, key=lambda t: int(t.get("exit_i") or t["i"]))
    eq = 0.0
    times = []
    series = []
    for t in ordered:
        eq = round(eq + float(t["pnl"]), 2)
        ei = int(t.get("exit_i") or t["i"])
        ei = min(ei, len(df) - 1)
        times.append(str(df.index[ei])[:19])
        series.append(eq)
    if len(times) > max_points:
        step = max(1, len(times) // max_points)
        keep = list(range(0, len(times), step))
        if keep[-1] != len(times) - 1:
            keep.append(len(times) - 1)
        times = [times[i] for i in keep]
        series = [series[i] for i in keep]
    return {"times": times, "equity": series, "final": series[-1] if series else 0.0}


def _summarize(trades: list[dict], n_bars: int) -> dict:
    st = _stats(trades)
    bat = _battery_from_trades(trades, max(n_bars, 250)) if trades else {}
    wins = sum(1 for t in trades if float(t["r"]) > 0)
    n = len(trades)
    return {
        "n": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": round(100.0 * wins / n, 1) if n else 0.0,
        "exp_r": st.get("exp_r") or 0.0,
        "pnl": st.get("pnl") or 0.0,
        "wf_green": bat.get("wf_green"),
        "mc": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "verdict": bat.get("verdict"),
        "oos_exp_r": (bat.get("oos") or {}).get("exp_r"),
    }


def main() -> int:
    LOG.write_text("")
    t0 = time.time()
    hunt = json.loads(HUNT.read_text(encoding="utf-8")) if HUNT.exists() else {}
    per_hunt = hunt.get("per_pair") or {}

    _log(
        "=== ADAPTIVE REGIME ROUTER BT ===\n"
        f"TF={TF} cap={CAP_YEARS}y windows={[w[0] for w in WINDOWS]}\n"
        "Router: TREND→family book | FADE→kalman | FLAT→sit out\n"
        "LIVE BRAINS NOT TOUCHED\n"
    )

    per_pair: dict[str, Any] = {}
    desk_adapt: dict[str, float] = {w[0]: 0.0 for w in WINDOWS}
    desk_static: dict[str, float] = {w[0]: 0.0 for w in WINDOWS}
    equity_by_sym: dict[str, Any] = {}

    for pi, sym in enumerate(BOOK):
        _progress(
            {
                "status": "running",
                "pct": int(5 + 90 * pi / len(BOOK)),
                "message": f"adaptive [{sym}]",
                "symbol": sym,
                "elapsed_sec": round(time.time() - t0, 1),
            }
        )
        _log(f"\n######## {sym} ########")
        try:
            df = _load(sym, TF, CAP_YEARS)
        except Exception as exc:
            _log(f"  FAIL load: {exc}")
            per_pair[sym] = {"error": str(exc)}
            continue

        wid = ((per_hunt.get(sym) or {}).get("winner") or {}).get("id") or ""
        static_sides, static_label, static_sl = _static_sides_for_winner(df, wid)
        # RR from hunt winner if present
        wrow = (per_hunt.get(sym) or {}).get("winner") or {}
        rr_static = float(wrow.get("rr") or 2.4)
        rr_adapt = 2.4  # balanced adaptive RR
        sl_adapt = 1.2

        adapt_sides, regime, reg_meta = _adaptive_sides(df, sym)

        ent_s = _atr_sl_entries(df, static_sides, sl_atr=float(wrow.get("sl_atr") or static_sl))
        ent_a = _atr_sl_entries(df, adapt_sides, sl_atr=sl_adapt)
        tr_s = _trades_with_i(df, ent_s, rr_static)
        tr_a = _trades_with_i(df, ent_a, rr_adapt)

        windows_out = {}
        for wname, days in WINDOWS:
            ts = _window_trades(df, tr_s, days)
            ta = _window_trades(df, tr_a, days)
            # bars in window
            end = df.index[-1]
            start = end - pd.Timedelta(days=days)
            n_bars = int(np.sum(np.asarray(df.index >= start)))
            ss = _summarize(ts, n_bars)
            sa = _summarize(ta, n_bars)
            windows_out[wname] = {"static": ss, "adaptive": sa}
            desk_static[wname] = round(desk_static[wname] + float(ss["pnl"]), 2)
            desk_adapt[wname] = round(desk_adapt[wname] + float(sa["pnl"]), 2)
            _log(
                f"  {wname:4} static n={ss['n']:5} WR={ss['win_rate']:5} E[R]={ss['exp_r']:+.3f} pnl=${ss['pnl']:>8,.0f}  |  "
                f"adapt n={sa['n']:5} WR={sa['win_rate']:5} E[R]={sa['exp_r']:+.3f} pnl=${sa['pnl']:>8,.0f}"
            )

        eq_s = _equity_curve(tr_s, df)
        eq_a = _equity_curve(tr_a, df)
        equity_by_sym[sym] = {
            "static": eq_s,
            "adaptive": eq_a,
            "static_label": static_label,
            "rr_static": rr_static,
            "rr_adaptive": rr_adapt,
        }

        # regime occupancy in last 90d
        end = df.index[-1]
        i90 = int(np.searchsorted(np.asarray(df.index), end - pd.Timedelta(days=90)))
        reg90 = regime[i90:]
        per_pair[sym] = {
            "symbol": sym,
            "tf": TF,
            "static_winner_id": wid,
            "static_label": static_label,
            "regime": reg_meta,
            "regime_90d": {
                "pct_trend": round(100.0 * float(np.mean(reg90 == 1)), 2) if len(reg90) else 0,
                "pct_fade": round(100.0 * float(np.mean(reg90 == -1)), 2) if len(reg90) else 0,
                "pct_flat": round(100.0 * float(np.mean(reg90 == 0)), 2) if len(reg90) else 0,
            },
            "windows": windows_out,
            "delta_pnl": {
                w: round(
                    float(windows_out[w]["adaptive"]["pnl"])
                    - float(windows_out[w]["static"]["pnl"]),
                    2,
                )
                for w in windows_out
            },
        }

    # Desk equity race (1y adaptive vs static) — concatenate by time approx via per-symbol finals
    charts = {
        "desk_pnl_by_window": [
            {
                "window": w,
                "static_pnl": desk_static[w],
                "adaptive_pnl": desk_adapt[w],
                "delta": round(desk_adapt[w] - desk_static[w], 2),
            }
            for w, _ in WINDOWS
        ],
        "pair_delta_1y": [
            {
                "symbol": s,
                "delta": (per_pair.get(s) or {}).get("delta_pnl", {}).get("1y"),
                "adapt_pnl": ((per_pair.get(s) or {}).get("windows") or {})
                .get("1y", {})
                .get("adaptive", {})
                .get("pnl"),
                "static_pnl": ((per_pair.get(s) or {}).get("windows") or {})
                .get("1y", {})
                .get("static", {})
                .get("pnl"),
            }
            for s in BOOK
            if s in per_pair and "error" not in per_pair[s]
        ],
        "pair_delta_7d": [
            {
                "symbol": s,
                "delta": (per_pair.get(s) or {}).get("delta_pnl", {}).get("7d"),
                "adapt_pnl": ((per_pair.get(s) or {}).get("windows") or {})
                .get("7d", {})
                .get("adaptive", {})
                .get("pnl"),
                "static_pnl": ((per_pair.get(s) or {}).get("windows") or {})
                .get("7d", {})
                .get("static", {})
                .get("pnl"),
            }
            for s in BOOK
            if s in per_pair and "error" not in per_pair[s]
        ],
        "regime_flat_pct": [
            {
                "symbol": s,
                "flat": (per_pair.get(s) or {}).get("regime", {}).get("pct_flat"),
                "trend": (per_pair.get(s) or {}).get("regime", {}).get("pct_trend"),
                "fade": (per_pair.get(s) or {}).get("regime", {}).get("pct_fade"),
            }
            for s in BOOK
            if s in per_pair and "error" not in per_pair[s]
        ],
        "equity_by_symbol": equity_by_sym,
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "recommend_live": False,
        "elapsed_sec": round(time.time() - t0, 1),
        "tf": TF,
        "cap_years": CAP_YEARS,
        "risk_usd": RISK_USD,
        "router": {
            "TREND": "pair family (bimod/perm/snr)",
            "FADE": "kalman_fade |z|≥2.2",
            "FLAT": "sit out (no trade)",
        },
        "desk": {
            "static_pnl": desk_static,
            "adaptive_pnl": desk_adapt,
            "delta": {w: round(desk_adapt[w] - desk_static[w], 2) for w, _ in WINDOWS},
        },
        "per_pair": per_pair,
        "charts": charts,
        "note": "Adaptive regime router vs static invention winners. Live brains unchanged.",
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    CHARTS.write_text(json.dumps(charts, indent=2, default=str), encoding="utf-8")
    _progress(
        {
            "status": "done",
            "pct": 100,
            "message": f"done {(time.time()-t0)/60:.1f}m",
            "elapsed_sec": round(time.time() - t0, 1),
        }
    )
    _log(
        f"\nDESK deltas: "
        + ", ".join(f"{w} ${desk_adapt[w]-desk_static[w]:+,.0f}" for w, _ in WINDOWS)
    )
    _log(f"Wrote {OUT}\nCharts {CHARTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
