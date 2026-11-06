#!/usr/bin/env python3
"""RESEARCH ONLY — ETHUSD with-trend deep battery.

Bias ONLY from 15m + 1h (EMA 8/14 must agree).
Entry ONLY on 1m / 3m / 5m — pick the clear winner.

For each (entry_tf × strategy × window):
  full 8-test battery (full, OOS70, WF×4, torture spread, torture skip + MC)

Windows: 1m / 3m / 6m / 1y.

Does NOT change live SYMBOL_ENTRY_BRAIN.
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

from analysis._edge_survive_bt import RISK_USD, _prep  # noqa: E402
from analysis.lab_agentic_1y_battery import _battery_from_trades  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import _simulate_fast  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_eth_withtrend_entry_tf_deep_bt.json"
LOG = Path(ROOT) / "data" / "lab_eth_withtrend_entry_tf_deep_bt.log"

SYMBOL = "ETHUSD"
ENTRY_TFS = ("1m", "3m", "5m")
WINDOWS = (("1m", 30), ("3m", 91), ("6m", 183), ("1y", 365))


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _load_tf(sym: str, tf: str, days: int = 400) -> pd.DataFrame:
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    clear_hist_cache()
    if not has_local_hist(sym, tf):
        raise RuntimeError(f"no hist {sym} {tf}")
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    raw = load_local_ohlcv(
        sym,
        tf=tf,
        start=(end - pd.Timedelta(days=days + 40)).isoformat(),
        end=end.isoformat(),
    )
    if raw is None or len(raw) < 400:
        raise RuntimeError(f"{sym} {tf} too short")
    df = _prep(raw[["open", "high", "low", "close"]].dropna())
    data_end = df.index[-1]
    df = df[df.index >= data_end - pd.Timedelta(days=days)]
    _log(f"load {sym} {tf}: {len(df):,} {df.index[0].date()}→{df.index[-1].date()}")
    return df


def _slice_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    end = df.index[-1]
    return df[df.index >= end - pd.Timedelta(days=days)].copy()


def _align_bias(exec_index: pd.DatetimeIndex, htf: pd.Series) -> np.ndarray:
    s = htf.shift(1)  # no peek
    return s.reindex(exec_index, method="ffill").fillna(0).astype(int).to_numpy()


def _htf_bias_series(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    ema8 = c.ewm(span=8, adjust=False).mean()
    ema14 = c.ewm(span=14, adjust=False).mean()
    sig = pd.Series(0, index=df.index, dtype=int)
    return sig.mask(ema8 > ema14, 1).mask(ema8 < ema14, -1)


def _build_sync(exec_df: pd.DataFrame, df_15: pd.DataFrame, df_1h: pd.DataFrame) -> np.ndarray:
    a15 = _align_bias(exec_df.index, _htf_bias_series(df_15))
    a1h = _align_bias(exec_df.index, _htf_bias_series(df_1h))
    out = np.zeros(len(exec_df), dtype=int)
    agree = (a15 == a1h) & (a15 != 0)
    out[agree] = a15[agree]
    return out


def _bb_with_trend(
    df: pd.DataFrame,
    sync: np.ndarray,
    *,
    std_mult: float,
    sl_atr: float,
    min_risk_px: float,
) -> list[dict]:
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    lo = (mid - std_mult * std).to_numpy(dtype=float)
    hi = (mid + std_mult * std).to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    entries: list[dict] = []
    for i in range(60, len(df) - 1):
        bias = int(sync[i])
        if bias == 0:
            continue
        a = float(atr[i] or 0) or 1e-6
        c, o = float(close[i]), float(open_[i])
        if not np.isfinite(lo[i]) or not np.isfinite(hi[i]):
            continue
        side = None
        # with-trend pullbacks only
        if bias == 1 and c < lo[i] and c > o:
            side = "BUY"
            sl = c - sl_atr * a
        elif bias == -1 and c > hi[i] and c < o:
            side = "SELL"
            sl = c + sl_atr * a
        if side is None:
            continue
        risk = abs(c - sl)
        if risk < min_risk_px:
            continue
        entries.append(
            {"ei": i, "side": side, "entry": c, "sl": sl, "entry_t": df.index[i], "risk_px": risk}
        )
    return entries


def _ema_pull_with_trend(
    df: pd.DataFrame,
    sync: np.ndarray,
    *,
    sl_atr: float,
    min_risk_px: float,
) -> list[dict]:
    c = df["close"].astype(float)
    ema8 = c.ewm(span=8, adjust=False).mean().to_numpy(dtype=float)
    ema14 = c.ewm(span=14, adjust=False).mean().to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    close = c.to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    entries: list[dict] = []
    for i in range(60, len(df) - 1):
        bias = int(sync[i])
        if bias == 0:
            continue
        a = float(atr[i] or 0) or 1e-6
        px = float(close[i])
        side = None
        if bias == 1 and ema8[i] > ema14[i]:
            if low[i] <= ema8[i] + 0.05 * a and px > ema8[i] and px > float(open_[i]):
                side = "BUY"
        elif bias == -1 and ema8[i] < ema14[i]:
            if high[i] >= ema8[i] - 0.05 * a and px < ema8[i] and px < float(open_[i]):
                side = "SELL"
        if side is None:
            continue
        sl = px - sl_atr * a if side == "BUY" else px + sl_atr * a
        risk = abs(px - sl)
        if risk < min_risk_px:
            continue
        entries.append(
            {"ei": i, "side": side, "entry": px, "sl": sl, "entry_t": df.index[i], "risk_px": risk}
        )
    return entries


def _ema_cross_with_trend(
    df: pd.DataFrame,
    sync: np.ndarray,
    *,
    sl_atr: float,
    min_risk_px: float,
) -> list[dict]:
    c = df["close"].astype(float)
    ema8 = c.ewm(span=8, adjust=False).mean().to_numpy(dtype=float)
    ema14 = c.ewm(span=14, adjust=False).mean().to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    close = c.to_numpy(dtype=float)
    entries: list[dict] = []
    for i in range(60, len(df) - 1):
        bias = int(sync[i])
        if bias == 0:
            continue
        side = None
        if bias == 1 and ema8[i - 1] <= ema14[i - 1] and ema8[i] > ema14[i]:
            side = "BUY"
        elif bias == -1 and ema8[i - 1] >= ema14[i - 1] and ema8[i] < ema14[i]:
            side = "SELL"
        if side is None:
            continue
        a = float(atr[i] or 0) or 1e-6
        px = float(close[i])
        sl = px - sl_atr * a if side == "BUY" else px + sl_atr * a
        risk = abs(px - sl)
        if risk < min_risk_px:
            continue
        entries.append(
            {"ei": i, "side": side, "entry": px, "sl": sl, "entry_t": df.index[i], "risk_px": risk}
        )
    return entries


def _bb_no_bias(
    df: pd.DataFrame,
    *,
    std_mult: float,
    sl_atr: float,
    min_risk_px: float,
) -> list[dict]:
    """Ablation: live-like MR with no HTF bias (for comparison)."""
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    lo = (mid - std_mult * std).to_numpy(dtype=float)
    hi = (mid + std_mult * std).to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    entries: list[dict] = []
    for i in range(60, len(df) - 1):
        a = float(atr[i] or 0) or 1e-6
        c, o = float(close[i]), float(open_[i])
        if not np.isfinite(lo[i]) or not np.isfinite(hi[i]):
            continue
        side = None
        if c < lo[i] and c > o:
            side, sl = "BUY", c - sl_atr * a
        elif c > hi[i] and c < o:
            side, sl = "SELL", c + sl_atr * a
        else:
            continue
        risk = abs(c - sl)
        if risk < min_risk_px:
            continue
        entries.append(
            {"ei": i, "side": side, "entry": c, "sl": sl, "entry_t": df.index[i], "risk_px": risk}
        )
    return entries


SPECS: list[dict[str, Any]] = [
    {
        "name": "BB15_withtrend_firm_rr4",
        "family": "bb_wt",
        "bb_mult": 1.5,
        "sl_atr": 2.0,
        "rr": 4.0,
        "min_risk_px": 6.0,
        "note": "BB1.5 pullback WITH 15m+1h bias · firm SL · RR4",
    },
    {
        "name": "BB15_withtrend_sl1.5_rr2",
        "family": "bb_wt",
        "bb_mult": 1.5,
        "sl_atr": 1.5,
        "rr": 2.0,
        "min_risk_px": 6.0,
        "note": "BB1.5 with-trend · scalp RR2",
    },
    {
        "name": "BB25_withtrend_firm_rr4",
        "family": "bb_wt",
        "bb_mult": 2.5,
        "sl_atr": 2.0,
        "rr": 4.0,
        "min_risk_px": 6.0,
        "note": "BB2.5 with-trend · firm",
    },
    {
        "name": "EMA814_pull_withtrend_rr2",
        "family": "ema_pull",
        "sl_atr": 1.5,
        "rr": 2.0,
        "min_risk_px": 6.0,
        "note": "EMA8/14 pullback with HTF bias · RR2",
    },
    {
        "name": "EMA814_pull_withtrend_sl2_rr2.4",
        "family": "ema_pull",
        "sl_atr": 2.0,
        "rr": 2.4,
        "min_risk_px": 6.0,
        "note": "EMA pull with-trend · wider",
    },
    {
        "name": "EMA814_cross_withtrend_rr2",
        "family": "ema_cross",
        "sl_atr": 1.5,
        "rr": 2.0,
        "min_risk_px": 6.0,
        "note": "EMA8/14 cross with HTF bias",
    },
    {
        "name": "LIVE_bb15_NO_bias_rr4",
        "family": "bb_nobias",
        "bb_mult": 1.5,
        "sl_atr": 1.2,
        "rr": 4.0,
        "min_risk_px": 0.0,
        "note": "ablation: live MR no HTF bias",
    },
    {
        "name": "LIVE_bb15_firm_NO_bias_rr4",
        "family": "bb_nobias",
        "bb_mult": 1.5,
        "sl_atr": 2.0,
        "rr": 4.0,
        "min_risk_px": 6.0,
        "note": "ablation: firm MR no HTF bias",
    },
]


def _entries(spec: dict, df: pd.DataFrame, sync: np.ndarray) -> list[dict]:
    fam = spec["family"]
    if fam == "bb_wt":
        return _bb_with_trend(
            df,
            sync,
            std_mult=float(spec["bb_mult"]),
            sl_atr=float(spec["sl_atr"]),
            min_risk_px=float(spec["min_risk_px"]),
        )
    if fam == "ema_pull":
        return _ema_pull_with_trend(
            df,
            sync,
            sl_atr=float(spec["sl_atr"]),
            min_risk_px=float(spec["min_risk_px"]),
        )
    if fam == "ema_cross":
        return _ema_cross_with_trend(
            df,
            sync,
            sl_atr=float(spec["sl_atr"]),
            min_risk_px=float(spec["min_risk_px"]),
        )
    return _bb_no_bias(
        df,
        std_mult=float(spec["bb_mult"]),
        sl_atr=float(spec["sl_atr"]),
        min_risk_px=float(spec["min_risk_px"]),
    )


def _rank_key(cell: dict) -> tuple:
    return (
        1 if cell.get("verdict") == "SURVIVES" else 0,
        float(cell.get("exp_r") or -99),
        float(cell.get("oos_exp_r") or -99),
        int(cell.get("wf_green") or 0),
        float(cell.get("pnl") or -1e18),
    )


def _slim_tests(tests: dict) -> dict:
    out = {}
    for k, v in (tests or {}).items():
        if k == "mc_prob_profit":
            out[k] = v
            continue
        if isinstance(v, dict):
            out[k] = {
                "n": v.get("n"),
                "pnl": v.get("pnl"),
                "exp_r": v.get("exp_r"),
                "win_rate": v.get("win_rate"),
            }
    return out


def run_cell(
    spec: dict,
    entry_tf: str,
    window_tag: str,
    df: pd.DataFrame,
    sync: np.ndarray,
) -> dict:
    ents = _entries(spec, df, sync)
    tr = _simulate_fast(df, ents, float(spec["rr"]))
    bat = _battery_from_trades(tr, len(df))
    full = bat.get("full") or {}
    return {
        "name": spec["name"],
        "note": spec.get("note", ""),
        "entry_tf": entry_tf,
        "window": window_tag,
        "spec": {k: v for k, v in spec.items() if k != "note"},
        "entries_raw": len(ents),
        "trades": full.get("n", 0),
        "pnl": full.get("pnl", 0.0),
        "exp_r": full.get("exp_r", 0.0),
        "win_rate": full.get("win_rate", 0.0),
        "verdict": bat.get("verdict"),
        "wf_green": bat.get("wf_green"),
        "wf_n": bat.get("wf_n"),
        "oos_exp_r": ((bat.get("oos") or {}).get("exp_r")),
        "oos_pnl": ((bat.get("oos") or {}).get("pnl")),
        "mc_prob_profit": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "tests": _slim_tests(bat.get("tests") or {}),
        "torture": bat.get("torture"),
    }


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass

    _log("=== ETHUSD with-trend deep · bias=15m+1h · entry=1m/3m/5m · 8-test battery ===")
    _log("Windows: 1m / 3m / 6m / 1y · RESEARCH only")

    df_15 = _load_tf(SYMBOL, "15m", days=400)
    df_1h = _load_tf(SYMBOL, "1h", days=400)
    frames = {tf: _load_tf(SYMBOL, tf, days=400) for tf in ENTRY_TFS}

    all_cells: list[dict] = []
    by_window: dict[str, list] = {w: [] for w, _ in WINDOWS}
    by_tf: dict[str, list] = {tf: [] for tf in ENTRY_TFS}

    for entry_tf in ENTRY_TFS:
        df_full = frames[entry_tf]
        for wtag, wdays in WINDOWS:
            # need enough bars for WF — skip tiny
            ex = _slice_days(df_full, wdays)
            d15 = _slice_days(df_15, wdays + 10)
            d1h = _slice_days(df_1h, wdays + 10)
            if len(ex) < 250:
                _log(f"SKIP {entry_tf}/{wtag}: only {len(ex)} bars")
                continue
            sync = _build_sync(ex, d15, d1h)
            sync_rate = float((sync != 0).mean())
            _log(f"\n===== entry={entry_tf} window={wtag} bars={len(ex):,} htf_sync_rate={sync_rate:.1%} =====")
            for spec in SPECS:
                cell = run_cell(spec, entry_tf, wtag, ex, sync)
                all_cells.append(cell)
                by_window[wtag].append(cell)
                by_tf[entry_tf].append(cell)
                _log(
                    f"  {cell['name']}: n={cell['trades']} E[R]={cell['exp_r']} "
                    f"WR={cell['win_rate']}% OOS_E[R]={cell['oos_exp_r']} "
                    f"WF={cell['wf_green']}/{cell['wf_n']} MC={cell['mc_prob_profit']}% "
                    f"{cell['verdict']} · ${cell['pnl']:,.0f}"
                )

    # Clear winners
    _log("\n########## CLEAR WINNERS ##########")

    winners_by_window = {}
    for wtag, _ in WINDOWS:
        cells = by_window.get(wtag) or []
        if not cells:
            continue
        ranked = sorted(cells, key=_rank_key, reverse=True)
        best = ranked[0]
        winners_by_window[wtag] = {
            "name": best["name"],
            "entry_tf": best["entry_tf"],
            "verdict": best["verdict"],
            "exp_r": best["exp_r"],
            "oos_exp_r": best["oos_exp_r"],
            "pnl": best["pnl"],
            "trades": best["trades"],
            "wf_green": best["wf_green"],
            "mc_prob_profit": best["mc_prob_profit"],
            "note": best["note"],
            "tests": best["tests"],
            "top3": [
                {
                    "name": c["name"],
                    "entry_tf": c["entry_tf"],
                    "verdict": c["verdict"],
                    "exp_r": c["exp_r"],
                    "pnl": c["pnl"],
                    "trades": c["trades"],
                }
                for c in ranked[:3]
            ],
        }
        _log(
            f"WINDOW {wtag}: {best['entry_tf']} · {best['name']} · "
            f"{best['verdict']} E[R]={best['exp_r']} OOS={best['oos_exp_r']} "
            f"n={best['trades']} ${best['pnl']:,.0f} WF={best['wf_green']}/{best['wf_n']}"
        )

    # Best entry TF: count SURVIVES + sum exp_r across windows for with-trend families only
    _log("\n### BEST ENTRY TF (with-trend specs only, across windows)")
    tf_score: dict[str, dict] = {}
    for tf in ENTRY_TFS:
        cells = [
            c
            for c in by_tf[tf]
            if c["spec"].get("family") in ("bb_wt", "ema_pull", "ema_cross")
        ]
        survives = sum(1 for c in cells if c.get("verdict") == "SURVIVES")
        er_sum = sum(float(c.get("exp_r") or 0) for c in cells)
        pnl_sum = sum(float(c.get("pnl") or 0) for c in cells)
        tf_score[tf] = {
            "survives": survives,
            "cells": len(cells),
            "exp_r_sum": round(er_sum, 4),
            "pnl_sum": round(pnl_sum, 2),
        }
        _log(
            f"  {tf}: SURVIVES={survives}/{len(cells)} ΣE[R]={er_sum:+.3f} Σpnl=${pnl_sum:,.0f}"
        )

    best_tf = max(
        ENTRY_TFS,
        key=lambda t: (
            tf_score[t]["survives"],
            tf_score[t]["exp_r_sum"],
            tf_score[t]["pnl_sum"],
        ),
    )
    _log(f"\n>>> BEST ENTRY TIMEFRAME: {best_tf}")

    # Overall firm pick: with-trend only, prefer SURVIVES on 6m AND 1y same name+tf
    _log("\n### FIRM OVERALL (must show on 6m + 1y)")
    key_cells: dict[tuple, dict] = {}
    for c in all_cells:
        if c["spec"].get("family") not in ("bb_wt", "ema_pull", "ema_cross"):
            continue
        if c["window"] not in ("6m", "1y"):
            continue
        k = (c["entry_tf"], c["name"])
        key_cells.setdefault(k, {})[c["window"]] = c

    firm = []
    for (tf, name), wins in key_cells.items():
        if "6m" not in wins or "1y" not in wins:
            continue
        a, b = wins["6m"], wins["1y"]
        if a["verdict"] == "SURVIVES" and b["verdict"] == "SURVIVES":
            firm.append(
                {
                    "entry_tf": tf,
                    "name": name,
                    "6m": {
                        "exp_r": a["exp_r"],
                        "pnl": a["pnl"],
                        "trades": a["trades"],
                        "oos_exp_r": a["oos_exp_r"],
                        "wf_green": a["wf_green"],
                        "mc": a["mc_prob_profit"],
                    },
                    "1y": {
                        "exp_r": b["exp_r"],
                        "pnl": b["pnl"],
                        "trades": b["trades"],
                        "oos_exp_r": b["oos_exp_r"],
                        "wf_green": b["wf_green"],
                        "mc": b["mc_prob_profit"],
                    },
                    "score": float(a["exp_r"]) + float(b["exp_r"]),
                    "note": a["note"],
                }
            )
    firm = sorted(firm, key=lambda x: x["score"], reverse=True)
    if firm:
        for f in firm[:5]:
            _log(
                f"  FIRM {f['entry_tf']} {f['name']}: "
                f"6m E[R]={f['6m']['exp_r']} 1y E[R]={f['1y']['exp_r']} "
                f"score={f['score']:.3f}"
            )
        clear = firm[0]
        _log(
            f"\n>>> CLEAR WINNER: entry={clear['entry_tf']} · {clear['name']}\n"
            f"    {clear['note']}"
        )
    else:
        clear = None
        _log("\n>>> NO FIRM WINNER (no with-trend spec SURVIVES both 6m and 1y)")
        # fallback: best single cell on 1y with-trend
        yt = [
            c
            for c in by_window.get("1y", [])
            if c["spec"].get("family") in ("bb_wt", "ema_pull", "ema_cross")
        ]
        if yt:
            best = max(yt, key=_rank_key)
            clear = {
                "entry_tf": best["entry_tf"],
                "name": best["name"],
                "note": best["note"],
                "fallback": True,
                "1y": {
                    "verdict": best["verdict"],
                    "exp_r": best["exp_r"],
                    "pnl": best["pnl"],
                },
            }
            _log(
                f"    FALLBACK 1y best: {best['entry_tf']} {best['name']} "
                f"{best['verdict']} E[R]={best['exp_r']}"
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "symbol": SYMBOL,
        "risk_usd": RISK_USD,
        "method": {
            "bias": "15m + 1h EMA8/14 must agree (bias only — no entry on those TFs)",
            "entry": "1m / 3m / 5m",
            "battery": "8 tests: full, OOS70, WF×4, torture spread 0.15R, torture skip 30% + MC",
            "windows": ["1m", "3m", "6m", "1y"],
        },
        "best_entry_tf": best_tf,
        "tf_scores": tf_score,
        "winners_by_window": winners_by_window,
        "firm_candidates": firm,
        "clear_winner": clear,
        "cells": all_cells,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"\nWrote {OUT} in {payload['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
