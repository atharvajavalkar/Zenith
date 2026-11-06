#!/usr/bin/env python3
"""RESEARCH ONLY — Master USDJPY opp-to-GOLD liquidity with math/probability.

Core process (from prior deep lab winners):
  JPY liquidity sweep + engulf, side = opposite GOLD dual EMA bias
  (OPP_FORCE) or JPY PA gated by opposite GOLD bias (OPP_GATE).

Math / probability stack (ablations + soft score):
  Fourier HF/LF · return z · kurtosis · skew · efficiency · trend_str
  · GOLD return z (move significance) · rolling corr opp gate
  · soft logistic-style score threshold (feature blend)

Windows: 1m / 3m / 6m / 12m / 2y · full 8-test + MC.
Baseline control: wired bb_2.5 RR4.

Firm recommend: SURVIVES 6m+12m, OOS E[R]>0, WF≥2, MC≥55%, n_12m≥80,
AND 12m E[R] ≥ wired BB 12m E[R] − 0.05 (must be competitive).

Does NOT auto-wire live.
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
from analysis.lab_gold_pa_liq_engulf_bt import RRS, _bias_series, _resample_ohlc  # noqa: E402
from analysis.lab_gold_pa_tf_grid_bt import TF_MIN, _fast_sweeps  # noqa: E402
from analysis.lab_usdjpy_gold_pa_opp_corr_deep_bt import (  # noqa: E402
    _apply_corr_opp_gate,
    _apply_gold_bias_opp_gate,
    _bb_entries,
    _collect_gold_bias_opp_force,
    _filter_window,
    _gold_dual_bias,
    _gold_ret_sum,
    _hist_end,
    _load_sym_frames,
    _parse_t,
    _rolling_corr,
)
from analysis.lab_gold_pa_tf_grid_bt import _collect_entries  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import _simulate_fast  # noqa: E402
from data.hist_local import clear_hist_cache  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_usdjpy_opp_gold_math_dist_deep_bt.json"
LOG = Path(ROOT) / "data" / "lab_usdjpy_opp_gold_math_dist_deep_bt.log"
PROGRESS = Path(ROOT) / "data" / "lab_usdjpy_opp_gold_math_dist_deep_bt.progress.json"

SYM = "USDJPY"
GOLD = "GOLD"
TFS = ("1m", "3m", "5m", "15m")
WINDOWS = (
    ("1m", 30),
    ("3m", 91),
    ("6m", 183),
    ("12m", 365),
    ("2y", 730),
)
FIRM_WINDOWS = ("6m", "12m")

# Seed stacks from prior firm opp winners + GOLD live clone
SEEDS = (
    {"mode": "OPP_FORCE", "liq_tf": "15m", "bias_a": "15m", "bias_b": "5m", "entry_tf": "1m"},
    {"mode": "OPP_FORCE", "liq_tf": "15m", "bias_a": "15m", "bias_b": "3m", "entry_tf": "1m"},
    {"mode": "OPP_FORCE", "liq_tf": "15m", "bias_a": "5m", "bias_b": "3m", "entry_tf": "1m"},
    {"mode": "OPP_FORCE", "liq_tf": "15m", "bias_a": "15m", "bias_b": "5m", "entry_tf": "3m"},
    {"mode": "OPP_GATE", "liq_tf": "5m", "bias_a": "15m", "bias_b": "5m", "entry_tf": "1m"},
    {"mode": "OPP_GATE", "liq_tf": "3m", "bias_a": "15m", "bias_b": "5m", "entry_tf": "1m"},
    {"mode": "OPP_GATE", "liq_tf": "3m", "bias_a": "15m", "bias_b": "5m", "entry_tf": "3m"},
    {"mode": "OPP_GATE", "liq_tf": "15m", "bias_a": "15m", "bias_b": "5m", "entry_tf": "1m"},
)

WIRED_BB_RR = 4.0


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _progress(payload: dict) -> None:
    try:
        PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _rolling_fourier_hf(ret: np.ndarray, win: int = 64, stride: int = 8) -> np.ndarray:
    n = len(ret)
    out = np.full(n, np.nan, dtype=float)
    for i in range(win, n, stride):
        seg = ret[i - win : i]
        if not np.isfinite(seg).all():
            continue
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg)) ** 2
        tot = float(spec.sum())
        if tot <= 0:
            continue
        cut = max(2, len(spec) // 2)
        out[i] = float(spec[cut:].sum() / tot)
    return pd.Series(out).ffill().fillna(0.5).to_numpy(dtype=float)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    o = out["open"].astype(float)
    rng = (h - l).replace(0, np.nan)
    out["eff"] = ((c - o).abs() / rng).clip(0, 1).fillna(0)
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ret"] = ret
    out["ret_z"] = (
        (ret - ret.rolling(60).mean()) / ret.rolling(60).std().replace(0, np.nan)
    ).fillna(0.0)
    out["kurt60"] = ret.rolling(60).kurt().fillna(0.0)
    out["vol60"] = ret.rolling(60).std().fillna(0.0)
    out["vol_z"] = (
        (out["vol60"] - out["vol60"].rolling(120).mean())
        / out["vol60"].rolling(120).std().replace(0, np.nan)
    ).fillna(0.0)
    out["fourier_hf"] = _rolling_fourier_hf(ret.to_numpy(), 64, 8)
    out["fourier_lf"] = 1.0 - out["fourier_hf"]
    ema8 = c.ewm(span=8, adjust=False).mean()
    ema14 = c.ewm(span=14, adjust=False).mean()
    atr = out["atr"].replace(0, np.nan)
    out["ema8"] = ema8
    out["ema14"] = ema14
    out["trend_str"] = ((ema8 - ema14).abs() / atr).fillna(0.0)
    out["skew60"] = ret.rolling(60).skew().fillna(0.0)
    return out


def _math_gate_ok(
    i: int,
    side: str,
    df: pd.DataFrame,
    gold_ret_z: pd.Series,
    gate: dict[str, Any],
) -> bool:
    fh = float(df["fourier_hf"].iloc[i])
    fl = float(df["fourier_lf"].iloc[i])
    rz = float(df["ret_z"].iloc[i])
    kurt = float(df["kurt60"].iloc[i])
    eff = float(df["eff"].iloc[i])
    tstr = float(df["trend_str"].iloc[i])
    skew = float(df["skew60"].iloc[i])
    vz = float(df["vol_z"].iloc[i])
    grz = float(gold_ret_z.iloc[i]) if i < len(gold_ret_z) else 0.0

    if gate.get("fourier_hf_max") is not None and fh > float(gate["fourier_hf_max"]):
        return False
    if gate.get("fourier_hf_min") is not None and fh < float(gate["fourier_hf_min"]):
        return False
    if gate.get("fourier_lf_min") is not None and fl < float(gate["fourier_lf_min"]):
        return False
    if gate.get("kurt_max") is not None and abs(kurt) > float(gate["kurt_max"]):
        return False
    if gate.get("min_eff") is not None and eff < float(gate["min_eff"]):
        return False
    if gate.get("min_trend_str") is not None and tstr < float(gate["min_trend_str"]):
        return False
    if gate.get("max_vol_z") is not None and vz > float(gate["max_vol_z"]):
        return False
    # GOLD move significance (opposite world: want |gold_ret_z| large)
    if gate.get("min_abs_gold_z") is not None and abs(grz) < float(gate["min_abs_gold_z"]):
        return False
    # pullback z on JPY with side
    zmin, zmax = gate.get("abs_z_min"), gate.get("abs_z_max")
    if zmin is not None or zmax is not None:
        if side == "BUY":
            if rz >= 0:
                return False
            depth = -rz
        else:
            if rz <= 0:
                return False
            depth = rz
        if zmin is not None and depth < float(zmin):
            return False
        if zmax is not None and depth > float(zmax):
            return False
    if gate.get("skew_align"):
        if side == "BUY" and skew < -0.5:
            return False
        if side == "SELL" and skew > 0.5:
            return False
    return True


def _soft_score(
    i: int,
    side: str,
    df: pd.DataFrame,
    gold_ret_z: pd.Series,
    corr_v: float,
) -> float:
    """0–1ish score: higher = cleaner opp-liq setup."""
    fh = float(df["fourier_hf"].iloc[i])
    fl = 1.0 - fh
    rz = float(df["ret_z"].iloc[i])
    kurt = abs(float(df["kurt60"].iloc[i]))
    eff = float(df["eff"].iloc[i])
    tstr = float(df["trend_str"].iloc[i])
    skew = float(df["skew60"].iloc[i])
    grz = float(gold_ret_z.iloc[i]) if i < len(gold_ret_z) else 0.0
    # components in [0,1]
    s_lf = float(np.clip(fl, 0, 1))
    s_eff = float(np.clip(eff, 0, 1))
    s_t = float(np.clip(tstr / 0.5, 0, 1))
    s_kurt = float(np.clip(1.0 - kurt / 8.0, 0, 1))
    # gold significance + correct opposite polarity
    if side == "BUY":
        s_gold = float(np.clip((-grz) / 1.5, 0, 1))  # want gold weak
        s_z = float(np.clip((-rz) / 1.5, 0, 1))
        s_sk = float(np.clip((skew + 0.5) / 1.5, 0, 1))
    else:
        s_gold = float(np.clip((grz) / 1.5, 0, 1))
        s_z = float(np.clip((rz) / 1.5, 0, 1))
        s_sk = float(np.clip((0.5 - skew) / 1.5, 0, 1))
    s_corr = float(np.clip((-corr_v) / 0.4, 0, 1)) if np.isfinite(corr_v) else 0.3
    score = (
        0.18 * s_lf
        + 0.12 * s_eff
        + 0.12 * s_t
        + 0.12 * s_kurt
        + 0.18 * s_gold
        + 0.12 * s_z
        + 0.08 * s_sk
        + 0.08 * s_corr
    )
    return float(score)


def _apply_math(
    entries: list[dict],
    df: pd.DataFrame,
    gold_ret_z: pd.Series,
    corr: pd.Series,
    gate: dict[str, Any],
) -> list[dict]:
    out = []
    for e in entries:
        ei = int(e["ei"])
        if ei < 0 or ei >= len(df):
            continue
        if not _math_gate_ok(ei, e["side"], df, gold_ret_z, gate):
            continue
        cv = float(corr.iloc[ei]) if ei < len(corr) and np.isfinite(corr.iloc[ei]) else 0.0
        sc = _soft_score(ei, e["side"], df, gold_ret_z, cv)
        if gate.get("min_score") is not None and sc < float(gate["min_score"]):
            continue
        ee = dict(e)
        ee["math_score"] = round(sc, 4)
        out.append(ee)
    return out


def _gate_sets() -> list[dict[str, Any]]:
    """Ablation math/prob gates — prefer sparse so n doesn't die."""
    gates: list[dict[str, Any]] = [
        {"name": "bare"},
        # Fourier
        {"name": "hf_max0.55", "fourier_hf_max": 0.55},
        {"name": "hf_max0.50", "fourier_hf_max": 0.50},
        {"name": "hf_max0.45", "fourier_hf_max": 0.45},
        {"name": "lf_min0.50", "fourier_lf_min": 0.50},
        {"name": "lf_min0.55", "fourier_lf_min": 0.55},
        {"name": "hf_min0.35", "fourier_hf_min": 0.35},  # chop-accept (MR-ish)
        # distribution
        {"name": "kurt6", "kurt_max": 6.0},
        {"name": "kurt8", "kurt_max": 8.0},
        {"name": "z0.4-2.5", "abs_z_min": 0.4, "abs_z_max": 2.5},
        {"name": "z0.7-2.5", "abs_z_min": 0.7, "abs_z_max": 2.5},
        {"name": "skew", "skew_align": True},
        {"name": "eff0.25", "min_eff": 0.25},
        {"name": "eff0.35", "min_eff": 0.35},
        {"name": "tstr0.15", "min_trend_str": 0.15},
        {"name": "tstr0.25", "min_trend_str": 0.25},
        {"name": "volz2", "max_vol_z": 2.0},
        {"name": "goldz0.8", "min_abs_gold_z": 0.8},
        {"name": "goldz1.2", "min_abs_gold_z": 1.2},
        # soft score
        {"name": "score0.45", "min_score": 0.45},
        {"name": "score0.55", "min_score": 0.55},
        {"name": "score0.62", "min_score": 0.62},
        # dual soft
        {"name": "lf0.52+goldz0.8", "fourier_lf_min": 0.52, "min_abs_gold_z": 0.8},
        {"name": "hf0.50+kurt6", "fourier_hf_max": 0.50, "kurt_max": 6.0},
        {"name": "hf0.48+eff0.25+skew", "fourier_hf_max": 0.48, "min_eff": 0.25, "skew_align": True},
        {"name": "lf0.55+tstr0.15+score0.5", "fourier_lf_min": 0.55, "min_trend_str": 0.15, "min_score": 0.50},
        {"name": "z0.5-2.5+goldz1+score0.5", "abs_z_min": 0.5, "abs_z_max": 2.5, "min_abs_gold_z": 1.0, "min_score": 0.50},
        {"name": "hf0.45+kurt6+goldz1+score0.55", "fourier_hf_max": 0.45, "kurt_max": 6.0, "min_abs_gold_z": 1.0, "min_score": 0.55},
        {"name": "master_v1", "fourier_lf_min": 0.52, "kurt_max": 7.0, "min_eff": 0.22, "min_abs_gold_z": 0.7, "min_score": 0.48, "skew_align": True},
        {"name": "master_v2", "fourier_hf_max": 0.50, "abs_z_min": 0.3, "abs_z_max": 2.8, "min_trend_str": 0.12, "min_abs_gold_z": 0.9, "min_score": 0.52},
    ]
    return gates


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


def _pack(
    name: str,
    family: str,
    df: pd.DataFrame,
    entries: list[dict],
    rr: float,
    meta: dict,
    window_start: pd.Timestamp,
) -> dict[str, Any]:
    ents = _filter_window(entries, window_start)
    tr = _simulate_fast(df, ents, rr)
    bat = _battery_from_trades(tr, len(df))
    full = bat.get("full") or {}
    return {
        "name": name,
        "family": family,
        **meta,
        "rr": rr,
        "entries_raw": len(ents),
        "trades": full.get("n", 0),
        "pnl": full.get("pnl", 0.0),
        "exp_r": full.get("exp_r", 0.0),
        "win_rate": full.get("win_rate", 0.0),
        "verdict": bat.get("verdict"),
        "wf_green": bat.get("wf_green"),
        "wf_n": bat.get("wf_n"),
        "oos_exp_r": ((bat.get("oos") or {}).get("exp_r")),
        "mc_prob_profit": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "tests": _slim_tests(bat.get("tests") or {}),
    }


def _rank_key(c: dict) -> tuple:
    return (
        1 if c.get("verdict") == "SURVIVES" else 0,
        float(c.get("exp_r") or -99),
        float(c.get("oos_exp_r") or -99),
        int(c.get("wf_green") or 0),
        float(c.get("mc_prob_profit") or 0),
        float(c.get("pnl") or -1e18),
    )


def _is_firm_pair(a: dict, b: dict, bb_12m_er: float) -> bool:
    if a.get("verdict") != "SURVIVES" or b.get("verdict") != "SURVIVES":
        return False
    if float(a.get("oos_exp_r") or -1) <= 0 or float(b.get("oos_exp_r") or -1) <= 0:
        return False
    if int(a.get("wf_green") or 0) < 2 or int(b.get("wf_green") or 0) < 2:
        return False
    for cell in (a, b):
        mc = cell.get("mc_prob_profit")
        if mc is not None and float(mc) < 55:
            return False
    if int(b.get("trades") or 0) < 80:
        return False
    # competitive vs BB
    if float(b.get("exp_r") or -99) < float(bb_12m_er) - 0.05:
        return False
    return True


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass

    clear_hist_cache()
    data_end = _hist_end()
    gates = _gate_sets()
    rrs = (2.0, 2.4, 3.0, 4.0)
    corr_params = (
        ("15m", 96, -0.25),
        ("15m", 96, -0.15),
        ("5m", 96, -0.25),
        (None, None, None),  # no corr filter
    )

    _log(
        f"=== USDJPY opp-GOLD + MATH/DIST deep · seeds={len(SEEDS)} · "
        f"gates={len(gates)} · windows={[w for w, _ in WINDOWS]} ==="
    )
    _log("Core: liquidity opp-to-GOLD · math: Fourier/z/kurt/skew/eff/score/gold_z")
    _log(f"hist_end={data_end} · 8-test battery · RESEARCH")

    all_cells: list[dict] = []
    done = 0
    # seeds * gates * corr_variants * rrs * windows  (corr none + 3)
    total_est = (
        len(SEEDS) * len(gates) * len(corr_params) * len(rrs) * len(WINDOWS)
        + len(rrs) * len(WINDOWS)  # BB baseline
    )

    bb_12m_er = 0.0

    for wtag, wdays in WINDOWS:
        start = data_end - pd.Timedelta(days=wdays)
        _log(f"\n{'=' * 64}\n===== WINDOW {wtag} ({wdays}d) =====")
        jpy_raw = _load_sym_frames(SYM, wdays, data_end)
        gold_raw = _load_sym_frames(GOLD, wdays, data_end)
        jpy = {tf: _enrich(df) for tf, df in jpy_raw.items()}
        gold = {tf: _enrich(df) for tf, df in gold_raw.items()}

        sweeps: dict[str, list] = {}
        for liq in sorted({s["liq_tf"] for s in SEEDS}):
            sw = _fast_sweeps(jpy[liq])
            sweeps[liq] = [
                s
                for s in sw
                if _parse_t(s.get("t")) is None or _parse_t(s["t"]) >= start
            ]
            _log(f"  sweeps {liq}: {len(sweeps[liq])}")

        corr_cache: dict[tuple, pd.Series] = {}
        for ctf, lb, _th in corr_params:
            if ctf is None:
                continue
            corr_cache[(ctf, lb)] = _rolling_corr(gold[ctf], jpy[ctf], int(lb))

        # BB baseline
        bb_ents = _bb_entries(jpy["1m"], bb_mult=2.5)
        for rr in rrs:
            cell = _pack(
                f"BASE_bb_2.5_1m_rr{rr:g}",
                "BASE_BB",
                jpy["1m"],
                bb_ents,
                rr,
                {"window": wtag, "entry_tf": "1m"},
                start,
            )
            all_cells.append(cell)
            done += 1
            if abs(rr - WIRED_BB_RR) < 1e-9 and wtag == "12m":
                bb_12m_er = float(cell["exp_r"] or 0)
            _log(
                f"  BASE bb RR={rr}: n={cell['trades']} E[R]={cell['exp_r']} "
                f"{cell['verdict']}"
            )

        for seed in SEEDS:
            mode = seed["mode"]
            liq, ea, eb, ent = (
                seed["liq_tf"],
                seed["bias_a"],
                seed["bias_b"],
                seed["entry_tf"],
            )
            tag = f"{mode}|liq={liq}|bias={ea}+{eb}|entry={ent}"
            gb = _gold_dual_bias(gold[ea], gold[eb], jpy[ent].index)
            if mode == "OPP_FORCE":
                base = _collect_gold_bias_opp_force(
                    jpy[ent], sweeps[liq], gb, ent
                )
            else:
                bare = _collect_entries(
                    jpy[ent], sweeps[liq], jpy[ea], jpy[eb], ent
                )
                base = _apply_gold_bias_opp_gate(bare, jpy[ent], gb)

            # GOLD ret z on entry tf
            gret = gold[ent]["ret"]
            gold_ret_z = (
                (gret - gret.rolling(60).mean())
                / gret.rolling(60).std().replace(0, np.nan)
            ).fillna(0.0)
            gold_ret_z = gold_ret_z.reindex(jpy[ent].index, method="ffill").fillna(0.0)

            _log(f"  {tag}: base_entries={len(base)}")

            for ctf, lb, th in corr_params:
                if ctf is None:
                    corr_e = pd.Series(0.0, index=jpy[ent].index)
                    base_c = base
                    corr_tag = "nocorr"
                else:
                    corr_e = corr_cache[(ctf, lb)].reindex(
                        jpy[ent].index, method="ffill"
                    )
                    gsum = _gold_ret_sum(gold[ent], 3)
                    base_c = _apply_corr_opp_gate(
                        base, jpy[ent], corr_e, gsum, float(th)
                    )
                    corr_tag = f"corr={ctf}/{lb}/θ{th}"

                for gate in gates:
                    gname = gate["name"]
                    gkw = {k: v for k, v in gate.items() if k != "name"}
                    ents = _apply_math(
                        base_c, jpy[ent], gold_ret_z, corr_e, gkw
                    )
                    name = f"{tag}|{corr_tag}|math={gname}"
                    meta = {
                        "window": wtag,
                        "mode": mode,
                        "liq_tf": liq,
                        "bias_a": ea,
                        "bias_b": eb,
                        "entry_tf": ent,
                        "corr_tf": ctf,
                        "corr_lb": lb,
                        "corr_thresh": th,
                        "math_gate": gname,
                        "math": gkw,
                    }
                    for rr in rrs:
                        cell = _pack(
                            name, "OPP_MATH", jpy[ent], ents, rr, meta, start
                        )
                        all_cells.append(cell)
                        done += 1
                        if done % 50 == 0 or cell.get("verdict") == "SURVIVES":
                            if cell.get("verdict") == "SURVIVES" or done % 100 == 0:
                                _log(
                                    f"    [{done}/{total_est}] {wtag} {name[:55]} "
                                    f"RR={rr} n={cell['trades']} E[R]={cell['exp_r']} "
                                    f"{cell['verdict']} t={time.time()-t0:.0f}s"
                                )
                        if done % 80 == 0:
                            _progress(
                                {
                                    "done": done,
                                    "total": total_est,
                                    "elapsed_sec": round(time.time() - t0, 1),
                                    "pct": round(
                                        100.0 * done / max(total_est, 1), 1
                                    ),
                                    "window": wtag,
                                }
                            )

    if bb_12m_er <= 0:
        # fallback from cells
        for c in all_cells:
            if (
                c.get("family") == "BASE_BB"
                and c.get("window") == "12m"
                and abs(float(c.get("rr") or 0) - WIRED_BB_RR) < 1e-9
            ):
                bb_12m_er = float(c.get("exp_r") or 0)
                break

    _log(f"\nBB 12m E[R] benchmark = {bb_12m_er}")
    _log("########## FIRM SELECTION (6m+12m, competitive vs BB) ##########")

    by_key: dict[tuple, dict] = {}
    for c in all_cells:
        if c.get("window") not in FIRM_WINDOWS:
            continue
        if c.get("family") != "OPP_MATH":
            continue
        k = (c["name"], float(c["rr"]))
        by_key.setdefault(k, {})[c["window"]] = c

    firm = []
    firm_soft = []  # survives both but may lag BB
    for (name, rr), wins in by_key.items():
        if "6m" not in wins or "12m" not in wins:
            continue
        a, b = wins["6m"], wins["12m"]
        row = {
            "name": name,
            "rr": rr,
            "mode": a.get("mode"),
            "math_gate": a.get("math_gate"),
            "spec": {
                k: a.get(k)
                for k in (
                    "liq_tf",
                    "bias_a",
                    "bias_b",
                    "entry_tf",
                    "corr_tf",
                    "corr_lb",
                    "corr_thresh",
                    "math",
                )
                if a.get(k) is not None
            },
            "6m": {
                "exp_r": a["exp_r"],
                "oos_exp_r": a["oos_exp_r"],
                "trades": a["trades"],
                "wf_green": a["wf_green"],
                "mc": a["mc_prob_profit"],
                "pnl": a["pnl"],
            },
            "12m": {
                "exp_r": b["exp_r"],
                "oos_exp_r": b["oos_exp_r"],
                "trades": b["trades"],
                "wf_green": b["wf_green"],
                "mc": b["mc_prob_profit"],
                "pnl": b["pnl"],
            },
            "score": float(a["exp_r"])
            + float(b["exp_r"])
            + 0.5 * float(a.get("oos_exp_r") or 0)
            + 0.5 * float(b.get("oos_exp_r") or 0),
            "vs_bb_12m": round(float(b["exp_r"]) - bb_12m_er, 4),
        }
        # soft firm without BB beat
        soft_ok = (
            a.get("verdict") == "SURVIVES"
            and b.get("verdict") == "SURVIVES"
            and float(a.get("oos_exp_r") or -1) > 0
            and float(b.get("oos_exp_r") or -1) > 0
            and int(a.get("wf_green") or 0) >= 2
            and int(b.get("wf_green") or 0) >= 2
            and int(b.get("trades") or 0) >= 80
        )
        if soft_ok:
            firm_soft.append(row)
        if _is_firm_pair(a, b, bb_12m_er):
            firm.append(row)

    firm = sorted(firm, key=lambda x: x["score"], reverse=True)
    firm_soft = sorted(firm_soft, key=lambda x: x["score"], reverse=True)

    winners_by_window = {}
    for wtag, _ in WINDOWS:
        cells = [c for c in all_cells if c.get("window") == wtag]
        if not cells:
            continue
        best = max(cells, key=_rank_key)
        winners_by_window[wtag] = {
            "name": best["name"][:80],
            "family": best["family"],
            "rr": best["rr"],
            "verdict": best["verdict"],
            "exp_r": best["exp_r"],
            "trades": best["trades"],
        }
        _log(
            f"WINDOW {wtag}: [{best['family']}] {best['name'][:55]} "
            f"RR={best['rr']} {best['verdict']} E[R]={best['exp_r']} n={best['trades']}"
        )

    # best opp_math on 12m
    opp12 = [
        c
        for c in all_cells
        if c.get("family") == "OPP_MATH" and c.get("window") == "12m"
    ]
    best_opp = max(opp12, key=_rank_key) if opp12 else None
    if best_opp:
        _log(
            f"BEST OPP_MATH 12m: {best_opp['name'][:70]} RR={best_opp['rr']} "
            f"{best_opp['verdict']} E[R]={best_opp['exp_r']} n={best_opp['trades']} "
            f"vsBB={float(best_opp['exp_r']) - bb_12m_er:+.3f}"
        )

    if firm:
        _log(f"\nFIRM (beats/near BB) COUNT={len(firm)}")
        for f in firm[:10]:
            _log(
                f"  {f['name'][:55]} RR={f['rr']} score={f['score']:.3f} "
                f"vsBB={f['vs_bb_12m']:+.3f} 6m={f['6m']['exp_r']} "
                f"12m={f['12m']['exp_r']} n12={f['12m']['trades']}"
            )
        clear = firm[0]
        _log(f"\n>>> CLEAR FIRM WINNER vs BB: {clear['name']} RR={clear['rr']}")
    else:
        clear = None
        _log("\n>>> NO math-opp arm competitive with BB under firm gates")
        if firm_soft:
            s = firm_soft[0]
            _log(
                f"    soft best: {s['name'][:60]} RR={s['rr']} "
                f"12m E[R]={s['12m']['exp_r']} vsBB={s['vs_bb_12m']:+.3f}"
            )

    elapsed = round(time.time() - t0, 1)
    cells_out = all_cells
    if len(all_cells) > 8000:
        keep = {id(c) for c in all_cells if c.get("verdict") == "SURVIVES"}
        ranked = sorted(all_cells, key=_rank_key, reverse=True)[:4000]
        keep |= {id(c) for c in ranked}
        cells_out = [c for c in all_cells if id(c) in keep]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "symbol": SYM,
        "elapsed_sec": elapsed,
        "n_cells": len(all_cells),
        "n_cells_saved": len(cells_out),
        "bb_12m_exp_r": bb_12m_er,
        "method": {
            "core": "JPY liq opp-to-GOLD (FORCE/GATE)",
            "math": "Fourier HF/LF, ret_z, kurt, skew, eff, trend_str, gold_ret_z, soft score, optional corr opp",
            "battery": "8 tests + MC",
            "firm_rule": "SURVIVES 6m+12m, OOS>0, WF≥2, MC≥55%, n12≥80, 12m E[R]≥BB−0.05",
        },
        "winners_by_window": winners_by_window,
        "best_opp_math_12m": {
            "name": best_opp["name"],
            "rr": best_opp["rr"],
            "verdict": best_opp["verdict"],
            "exp_r": best_opp["exp_r"],
            "oos_exp_r": best_opp["oos_exp_r"],
            "trades": best_opp["trades"],
            "mc_prob_profit": best_opp["mc_prob_profit"],
            "math_gate": best_opp.get("math_gate"),
            "tests": best_opp.get("tests"),
            "vs_bb": round(float(best_opp["exp_r"]) - bb_12m_er, 4),
        }
        if best_opp
        else None,
        "firm_candidates": firm[:20],
        "firm_soft_top": firm_soft[:15],
        "clear_winner": clear,
        "recommend_live": bool(clear),
        "cells": cells_out,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress({"done": done, "total": done, "elapsed_sec": elapsed, "pct": 100.0})
    _log(f"\nWrote {OUT} in {elapsed}s ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
