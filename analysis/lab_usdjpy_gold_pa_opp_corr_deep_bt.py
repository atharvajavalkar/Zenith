#!/usr/bin/env python3
"""RESEARCH ONLY — USDJPY deep: GOLD PA strategy vs opposite-to-GOLD correlation.

Arms
  A) GOLD_PA — same process as GOLD: liq sweep → dual-TF EMA bias → engulf
     Full TF grid {1m,3m,5m,15m} · liq∈{3m,5m,15m} · RR∈{2,2.4,3,4}
  B) OPP_CORR — same PA entries/gates with GOLD inverse filters:
     • gold_bias_opp_gate  — keep JPY trade only if GOLD dual bias opposite
     • gold_bias_opp_force — require JPY side = opposite of GOLD dual bias
     • corr_opp_gate       — keep only when rolling corr(GOLD,JPY)<θ and
                             recent GOLD return opposes the JPY side
  C) BASE — wired bb_2.5 RR4 on 1m (control)

Windows: 1m / 3m / 6m / 12m / 2y · full 8-test + MC battery each cell.

Firm rule (recommend): SURVIVES on BOTH 6m and 12m, OOS E[R]>0, WF≥2,
MC≥55%, n_12m≥80.

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
from analysis.lab_gold_pa_tf_grid_bt import (  # noqa: E402
    LIQ_TFS,
    TFS,
    TF_MIN,
    _collect_entries,
    _combo_iter,
    _fast_sweeps,
)
from analysis.strategy_lab_duka_deep_hunt import _simulate_fast  # noqa: E402
from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_usdjpy_gold_pa_opp_corr_deep_bt.json"
LOG = Path(ROOT) / "data" / "lab_usdjpy_gold_pa_opp_corr_deep_bt.log"
PROGRESS = Path(ROOT) / "data" / "lab_usdjpy_gold_pa_opp_corr_deep_bt.progress.json"

SYM = "USDJPY"
GOLD = "GOLD"
WINDOWS = (
    ("1m", 30),
    ("3m", 91),
    ("6m", 183),
    ("12m", 365),
    ("2y", 730),
)
FIRM_WINDOWS = ("6m", "12m")
CORR_TFS = ("5m", "15m")
CORR_LOOKBACKS = (48, 96)
CORR_THRESHOLDS = (-0.15, -0.25, -0.35)

# Live GOLD PA clone (desk)
GOLD_LIVE = {
    "liq_tf": "3m",
    "bias_a": "15m",
    "bias_b": "5m",
    "entry_tf": "1m",
    "rr": 2.0,
}
WIRED_BB = {"brain": "bb_2.5", "tf": "1m", "rr": 4.0, "bb_mult": 2.5}


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


def _parse_t(s: Any) -> Optional[pd.Timestamp]:
    if s is None or s == "":
        return None
    try:
        ts = pd.Timestamp(s)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts
    except Exception:
        return None


def _load_sym_frames(
    symbol: str, window_days: int, data_end: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    start = data_end - pd.Timedelta(days=window_days)
    load_start = start - pd.Timedelta(days=21)
    frames: dict[str, pd.DataFrame] = {}
    for tf in TFS:
        if not has_local_hist(symbol, tf):
            continue
        try:
            df = load_local_ohlcv(
                symbol,
                tf=tf,
                start=load_start.isoformat(),
                end=data_end.isoformat(),
            )
        except Exception as exc:
            _log(f"  load {symbol} {tf} FAIL: {exc}")
            continue
        if df is None or len(df) < 200:
            continue
        need = [c for c in ("open", "high", "low", "close") if c in df.columns]
        df = _prep(df[need].dropna())
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        frames[tf] = df
    for tf, rule in (("3m", "3min"), ("5m", "5min"), ("15m", "15min")):
        if tf not in frames:
            src = frames.get("1m") or frames.get("5m")
            if src is None:
                raise RuntimeError(f"{symbol}: need 1m or 5m")
            frames[tf] = _resample_ohlc(src, rule)
    warm = start - pd.Timedelta(days=10)
    return {tf: df[df.index >= warm].copy() for tf, df in frames.items()}


def _hist_end() -> pd.Timestamp:
    clear_hist_cache()
    for sym, tf in ((SYM, "1m"), (SYM, "5m"), (GOLD, "1m")):
        if has_local_hist(sym, tf):
            df = load_local_ohlcv(sym, tf=tf)
            if df is not None and len(df) >= 400:
                end = df.index[-1]
                if getattr(end, "tzinfo", None) is not None:
                    end = end.tz_localize(None)
                return end
    raise RuntimeError("no hist end")


def _gold_dual_bias(
    g_a: pd.DataFrame, g_b: pd.DataFrame, idx: pd.DatetimeIndex
) -> pd.Series:
    b1 = _bias_series(g_a).reindex(idx, method="ffill").fillna(0).astype(int)
    b2 = _bias_series(g_b).reindex(idx, method="ffill").fillna(0).astype(int)
    out = pd.Series(0, index=idx, dtype=int)
    m = (b1 == b2) & (b1 != 0)
    out.loc[m] = b1.loc[m]
    return out


def _rolling_corr(
    gold_df: pd.DataFrame, jpy_df: pd.DataFrame, lookback: int
) -> pd.Series:
    """Pearson corr of log-returns, aligned on jpy index."""
    gc = gold_df["close"].astype(float)
    jc = jpy_df["close"].astype(float)
    gr = np.log(gc / gc.shift(1)).replace([np.inf, -np.inf], np.nan)
    jr = np.log(jc / jc.shift(1)).replace([np.inf, -np.inf], np.nan)
    aligned = pd.concat([gr, jr], axis=1, join="inner")
    aligned.columns = ["g", "j"]
    corr = aligned["g"].rolling(lookback, min_periods=max(20, lookback // 2)).corr(
        aligned["j"]
    )
    return corr.reindex(jpy_df.index, method="ffill")


def _gold_ret_sum(gold_df: pd.DataFrame, n: int = 3) -> pd.Series:
    c = gold_df["close"].astype(float)
    r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r.rolling(n).sum()


def _filter_window(entries: list[dict], start: pd.Timestamp) -> list[dict]:
    out = []
    for e in entries:
        et = _parse_t(e.get("entry_t"))
        if et is not None and et < start:
            continue
        out.append(e)
    return out


def _apply_gold_bias_opp_gate(
    entries: list[dict],
    df_e: pd.DataFrame,
    gold_bias: pd.Series,
) -> list[dict]:
    """Keep JPY entry only when GOLD dual bias is opposite to JPY side."""
    out = []
    for e in entries:
        ei = int(e["ei"])
        if ei < 0 or ei >= len(df_e):
            continue
        gb = int(gold_bias.iloc[ei]) if ei < len(gold_bias) else 0
        if gb == 0:
            continue
        side = e["side"]
        # opposite: GOLD BUY ↔ JPY SELL
        if side == "BUY" and gb < 0:
            out.append(e)
        elif side == "SELL" and gb > 0:
            out.append(e)
    return out


def _collect_gold_bias_opp_force(
    df_e: pd.DataFrame,
    sweeps: list[dict],
    gold_bias: pd.Series,
    entry_tf: str,
) -> list[dict]:
    """JPY liq+engulf, but side forced to opposite of GOLD dual bias (ignore JPY bias)."""
    from collections import deque

    from analysis.lab_gold_pa_liq_engulf_bt import _is_engulf

    if not sweeps:
        return []
    max_bars = max(3, int(90 / TF_MIN[entry_tf]))
    sweep_pos = []
    for sw in sweeps:
        pos = int(df_e.index.searchsorted(sw["t"], side="left"))
        if pos < len(df_e):
            sweep_pos.append((pos, sw))
    if not sweep_pos:
        return []
    sweep_pos.sort(key=lambda x: x[0])
    entries: list[dict] = []
    used: set[int] = set()
    active: deque = deque()
    sp_i = 0
    n_sw = len(sweep_pos)
    for i in range(40, len(df_e) - 1):
        while sp_i < n_sw and sweep_pos[sp_i][0] <= i:
            active.append(sweep_pos[sp_i])
            sp_i += 1
        while active and active[0][0] < i - max_bars:
            active.popleft()
        if not active:
            continue
        gb = int(gold_bias.iloc[i]) if i < len(gold_bias) else 0
        if gb == 0:
            continue
        # opposite of GOLD
        side = "SELL" if gb > 0 else "BUY"
        matched = None
        for sp, sw in reversed(active):
            if sp in used:
                continue
            if sw["side"] == side:
                matched = (sp, sw)
                break
        if matched is None:
            continue
        row, prev = df_e.iloc[i], df_e.iloc[i - 1]
        if not _is_engulf(row, prev, side):
            continue
        atr = float(row["atr"] or 0) or 1e-6
        sp, sw = matched
        c = float(row["close"])
        if side == "BUY":
            sl = min(float(sw["sl"]), float(row["low"]) - 0.08 * atr)
            if sl >= c or abs(c - sl) <= 0.12 * atr:
                continue
        else:
            sl = max(float(sw["sl"]), float(row["high"]) + 0.08 * atr)
            if sl <= c or abs(c - sl) <= 0.12 * atr:
                continue
        entries.append(
            {
                "ei": i,
                "side": side,
                "entry": c,
                "sl": sl,
                "entry_t": df_e.index[i],
            }
        )
        used.add(sp)
    return entries


def _apply_corr_opp_gate(
    entries: list[dict],
    df_e: pd.DataFrame,
    corr: pd.Series,
    gold_ret: pd.Series,
    thresh: float,
) -> list[dict]:
    """Keep when corr < thresh and GOLD recent return opposes JPY side."""
    out = []
    gret = gold_ret.reindex(df_e.index, method="ffill").fillna(0.0)
    for e in entries:
        ei = int(e["ei"])
        if ei < 0 or ei >= len(df_e):
            continue
        c = corr.iloc[ei] if ei < len(corr) else np.nan
        if not np.isfinite(c) or float(c) >= thresh:
            continue
        gr = float(gret.iloc[ei])
        side = e["side"]
        # inverse: JPY BUY wants GOLD weak; JPY SELL wants GOLD strong
        if side == "BUY" and gr < 0:
            out.append(e)
        elif side == "SELL" and gr > 0:
            out.append(e)
    return out


def _bb_entries(df: pd.DataFrame, bb_mult: float = 2.5, sl_atr: float = 1.2) -> list[dict]:
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    lo = (mid - bb_mult * std).to_numpy()
    hi = (mid + bb_mult * std).to_numpy()
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    out = []
    for i in range(40, len(df) - 1):
        a = float(atr[i] or 0) or 1e-6
        c, o = float(close[i]), float(open_[i])
        if np.isfinite(lo[i]) and c < lo[i] and c > o:
            out.append(
                {
                    "ei": i,
                    "side": "BUY",
                    "entry": c,
                    "sl": c - sl_atr * a,
                    "entry_t": df.index[i],
                }
            )
        elif np.isfinite(hi[i]) and c > hi[i] and c < o:
            out.append(
                {
                    "ei": i,
                    "side": "SELL",
                    "entry": c,
                    "sl": c + sl_atr * a,
                    "entry_t": df.index[i],
                }
            )
    return out


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


def _is_firm_pair(a: dict, b: dict) -> bool:
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
    return True


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass

    data_end = _hist_end()
    combos = _combo_iter()
    # corr gate variants on GOLD live + top PA denseness: all combos × corr params is huge
    # Run corr gates on GOLD_LIVE stack + every combo's bare PA + opp bias modes
    n_corr_variants = len(CORR_TFS) * len(CORR_LOOKBACKS) * len(CORR_THRESHOLDS)
    _log(
        f"=== USDJPY GOLD-PA vs OPP-CORR deep · combos={len(combos)} · "
        f"RR={list(RRS)} · windows={[w for w, _ in WINDOWS]} ==="
    )
    _log(
        f"Arms: GOLD_PA · gold_bias_opp_gate · gold_bias_opp_force · "
        f"corr_opp_gate×{n_corr_variants} (on GOLD live + each combo) · BASE bb"
    )
    _log(f"hist_end={data_end} · 8-test battery · RESEARCH ONLY")
    _log(f"GOLD live clone: {GOLD_LIVE}")

    all_cells: list[dict] = []
    done = 0
    # rough total for progress
    # per window: combos*(1 bare + 1 gate + 1 force)*len(RRS) + gold_live*corr*RRS + bb
    per_w = (
        len(combos) * 3 * len(RRS)
        + len(combos) * n_corr_variants * len(RRS)  # corr on all combos — deep
        + len(RRS)  # bb
    )
    # corr on all combos is very deep (~30*18*4*5 ≈ 10k) — keep corr on GOLD_LIVE + best-name clones only
    # Override: corr only on GOLD_LIVE stack to keep runtime sane while still deep on PA/opp
    per_w = len(combos) * 3 * len(RRS) + n_corr_variants * len(RRS) + len(RRS)
    total_est = per_w * len(WINDOWS)

    for wtag, wdays in WINDOWS:
        start = data_end - pd.Timedelta(days=wdays)
        _log(f"\n{'=' * 64}\n===== WINDOW {wtag} ({wdays}d) {start.date()}→{data_end.date()} =====")
        jpy = _load_sym_frames(SYM, wdays, data_end)
        gold = _load_sym_frames(GOLD, wdays, data_end)
        for tf in TFS:
            _log(
                f"  JPY {tf}: {len(jpy[tf]):,} | GOLD {tf}: {len(gold[tf]):,}"
            )

        sweeps: dict[str, list] = {}
        for liq in LIQ_TFS:
            t1 = time.time()
            sw = _fast_sweeps(jpy[liq])
            sweeps[liq] = [
                s
                for s in sw
                if _parse_t(s.get("t")) is None or _parse_t(s["t"]) >= start
            ]
            _log(f"  sweeps {liq}: {len(sweeps[liq])} ({time.time() - t1:.1f}s)")

        def gbias(ea: str, eb: str, idx: pd.DatetimeIndex) -> pd.Series:
            return _gold_dual_bias(gold[ea], gold[eb], idx)

        # Precompute corr series for corr gates (on GOLD live entry tf)
        corr_series: dict[tuple, pd.Series] = {}
        for ctf in CORR_TFS:
            for lb in CORR_LOOKBACKS:
                corr_series[(ctf, lb)] = _rolling_corr(gold[ctf], jpy[ctf], lb)

        for ci, cfg in enumerate(combos, 1):
            liq, ea, eb, ent = (
                cfg["liq_tf"],
                cfg["bias_a"],
                cfg["bias_b"],
                cfg["entry_tf"],
            )
            tag = f"liq={liq}|bias={ea}+{eb}|entry={ent}"
            t1 = time.time()
            base_ents = _collect_entries(
                jpy[ent], sweeps[liq], jpy[ea], jpy[eb], ent
            )
            gb = gbias(ea, eb, jpy[ent].index)
            gate_ents = _apply_gold_bias_opp_gate(base_ents, jpy[ent], gb)
            force_ents = _collect_gold_bias_opp_force(
                jpy[ent], sweeps[liq], gb, ent
            )
            _log(
                f"  [{ci}/{len(combos)}] {tag}: bare={len(base_ents)} "
                f"opp_gate={len(gate_ents)} opp_force={len(force_ents)} "
                f"({time.time() - t1:.1f}s)"
            )

            arms = (
                ("GOLD_PA", tag, base_ents),
                ("OPP_GATE", f"OPP_GATE|{tag}", gate_ents),
                ("OPP_FORCE", f"OPP_FORCE|{tag}", force_ents),
            )
            for family, name, ents in arms:
                meta = {
                    "liq_tf": liq,
                    "bias_a": ea,
                    "bias_b": eb,
                    "entry_tf": ent,
                    "window": wtag,
                }
                for rr in RRS:
                    cell = _pack(
                        name, family, jpy[ent], ents, rr, meta, start
                    )
                    all_cells.append(cell)
                    done += 1
                    if done % 40 == 0 or cell.get("verdict") == "SURVIVES":
                        _log(
                            f"    [{done}/{total_est}] {wtag} {family} {name[:40]} "
                            f"RR={rr} n={cell['trades']} E[R]={cell['exp_r']} "
                            f"{cell['verdict']} t={time.time() - t0:.0f}s"
                        )
                    if done % 80 == 0:
                        _progress(
                            {
                                "done": done,
                                "total": total_est,
                                "elapsed_sec": round(time.time() - t0, 1),
                                "pct": round(100.0 * done / max(total_est, 1), 1),
                                "window": wtag,
                            }
                        )

        # Corr opp gates on GOLD live stack (deep across corr params)
        gl = GOLD_LIVE
        ent = gl["entry_tf"]
        live_tag = (
            f"liq={gl['liq_tf']}|bias={gl['bias_a']}+{gl['bias_b']}|entry={ent}"
        )
        live_base = _collect_entries(
            jpy[ent],
            sweeps[gl["liq_tf"]],
            jpy[gl["bias_a"]],
            jpy[gl["bias_b"]],
            ent,
        )
        # GOLD return on entry tf
        gret = _gold_ret_sum(gold[ent], 3)
        for ctf in CORR_TFS:
            for lb in CORR_LOOKBACKS:
                # map corr from ctf onto entry index
                corr_e = corr_series[(ctf, lb)].reindex(
                    jpy[ent].index, method="ffill"
                )
                for th in CORR_THRESHOLDS:
                    name = f"CORR_OPP|{live_tag}|corr={ctf}/{lb}/θ{th}"
                    ents = _apply_corr_opp_gate(
                        live_base, jpy[ent], corr_e, gret, th
                    )
                    for rr in RRS:
                        cell = _pack(
                            name,
                            "CORR_OPP",
                            jpy[ent],
                            ents,
                            rr,
                            {
                                "liq_tf": gl["liq_tf"],
                                "bias_a": gl["bias_a"],
                                "bias_b": gl["bias_b"],
                                "entry_tf": ent,
                                "corr_tf": ctf,
                                "corr_lb": lb,
                                "corr_thresh": th,
                                "window": wtag,
                            },
                            start,
                        )
                        all_cells.append(cell)
                        done += 1

        # Wired BB baseline
        bb_ents = _bb_entries(jpy["1m"], bb_mult=float(WIRED_BB["bb_mult"]))
        for rr in RRS:
            cell = _pack(
                f"BASE_bb_2.5_1m_rr{rr:g}",
                "BASE_BB",
                jpy["1m"],
                bb_ents,
                rr,
                {"entry_tf": "1m", "window": wtag, "bb_mult": 2.5},
                start,
            )
            all_cells.append(cell)
            done += 1
            _log(
                f"  BASE bb RR={rr}: n={cell['trades']} E[R]={cell['exp_r']} "
                f"{cell['verdict']}"
            )

        _progress(
            {
                "done": done,
                "total": total_est,
                "elapsed_sec": round(time.time() - t0, 1),
                "pct": round(100.0 * done / max(total_est, 1), 1),
                "window": wtag,
            }
        )

    # Firm selection
    _log("\n########## FIRM SELECTION (6m+12m) ##########")
    by_key: dict[tuple, dict] = {}
    for c in all_cells:
        if c.get("window") not in FIRM_WINDOWS:
            continue
        k = (c["family"], c["name"], float(c["rr"]))
        by_key.setdefault(k, {})[c["window"]] = c

    firm = []
    for (fam, name, rr), wins in by_key.items():
        if "6m" not in wins or "12m" not in wins:
            continue
        a, b = wins["6m"], wins["12m"]
        if not _is_firm_pair(a, b):
            continue
        firm.append(
            {
                "family": fam,
                "name": name,
                "rr": rr,
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
                    )
                    if a.get(k) is not None
                },
                "6m": {
                    "exp_r": a["exp_r"],
                    "oos_exp_r": a["oos_exp_r"],
                    "pnl": a["pnl"],
                    "trades": a["trades"],
                    "wf_green": a["wf_green"],
                    "mc": a["mc_prob_profit"],
                    "verdict": a["verdict"],
                },
                "12m": {
                    "exp_r": b["exp_r"],
                    "oos_exp_r": b["oos_exp_r"],
                    "pnl": b["pnl"],
                    "trades": b["trades"],
                    "wf_green": b["wf_green"],
                    "mc": b["mc_prob_profit"],
                    "verdict": b["verdict"],
                },
                "score": float(a["exp_r"])
                + float(b["exp_r"])
                + 0.5 * float(a.get("oos_exp_r") or 0)
                + 0.5 * float(b.get("oos_exp_r") or 0),
            }
        )
    firm = sorted(firm, key=lambda x: x["score"], reverse=True)

    winners_by_window: dict[str, Any] = {}
    for wtag, _ in WINDOWS:
        cells = [c for c in all_cells if c.get("window") == wtag]
        if not cells:
            continue
        best = max(cells, key=_rank_key)
        winners_by_window[wtag] = {
            "name": best["name"],
            "family": best["family"],
            "rr": best["rr"],
            "verdict": best["verdict"],
            "exp_r": best["exp_r"],
            "oos_exp_r": best["oos_exp_r"],
            "pnl": best["pnl"],
            "trades": best["trades"],
            "mc_prob_profit": best["mc_prob_profit"],
        }
        _log(
            f"WINDOW {wtag}: [{best['family']}] {best['name'][:55]} "
            f"RR={best['rr']} {best['verdict']} E[R]={best['exp_r']} n={best['trades']}"
        )

    # Best per family on 12m
    best_by_family: dict[str, Any] = {}
    for fam in ("GOLD_PA", "OPP_GATE", "OPP_FORCE", "CORR_OPP", "BASE_BB"):
        cells = [
            c
            for c in all_cells
            if c.get("family") == fam and c.get("window") == "12m"
        ]
        if cells:
            best_by_family[fam] = max(cells, key=_rank_key)
            b = best_by_family[fam]
            _log(
                f"FAMILY 12m {fam}: {b['name'][:50]} RR={b['rr']} "
                f"{b['verdict']} E[R]={b['exp_r']} n={b['trades']}"
            )

    if firm:
        _log(f"\nFIRM COUNT={len(firm)}")
        for f in firm[:12]:
            _log(
                f"  [{f['family']}] {f['name'][:50]} RR={f['rr']} "
                f"score={f['score']:.3f} 6m E[R]={f['6m']['exp_r']} "
                f"12m E[R]={f['12m']['exp_r']} n12={f['12m']['trades']}"
            )
        clear = firm[0]
        _log(
            f"\n>>> CLEAR FIRM WINNER: [{clear['family']}] {clear['name']} "
            f"RR={clear['rr']}\n    spec={clear['spec']}"
        )
    else:
        clear = None
        _log("\n>>> NO FIRM WINNER under hard gates (6m+12m)")

    elapsed = round(time.time() - t0, 1)
    # slim cells if huge
    cells_out = all_cells
    if len(all_cells) > 6000:
        keep = {id(c) for c in all_cells if c.get("verdict") == "SURVIVES"}
        ranked = sorted(all_cells, key=_rank_key, reverse=True)[:3000]
        keep |= {id(c) for c in ranked}
        cells_out = [c for c in all_cells if id(c) in keep]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "symbol": SYM,
        "gold_ref": GOLD,
        "risk_usd": RISK_USD,
        "elapsed_sec": elapsed,
        "n_cells": len(all_cells),
        "n_cells_saved": len(cells_out),
        "method": {
            "gold_pa": "liq sweep → dual EMA bias → engulf (GOLD process on USDJPY)",
            "opp_gate": "JPY PA only when GOLD dual bias opposite",
            "opp_force": "JPY side forced opposite GOLD dual bias",
            "corr_opp": "GOLD live PA + rolling corr gate + opposing GOLD return",
            "battery": "8 tests + MC",
            "firm_rule": "SURVIVES 6m+12m, OOS E[R]>0, WF≥2, MC≥55%, n_12m≥80",
            "windows": [w for w, _ in WINDOWS],
        },
        "gold_live_clone": GOLD_LIVE,
        "wired_bb": WIRED_BB,
        "winners_by_window": winners_by_window,
        "best_by_family_12m": {
            k: {
                "name": v["name"],
                "rr": v["rr"],
                "verdict": v["verdict"],
                "exp_r": v["exp_r"],
                "oos_exp_r": v["oos_exp_r"],
                "trades": v["trades"],
                "pnl": v["pnl"],
                "mc_prob_profit": v["mc_prob_profit"],
                "tests": v["tests"],
            }
            for k, v in best_by_family.items()
        },
        "firm_candidates": firm[:25],
        "clear_winner": clear,
        "recommend_live": bool(clear),
        "cells": cells_out,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress({"done": done, "total": done, "elapsed_sec": elapsed, "pct": 100.0})
    _log(f"\nWrote {OUT} in {elapsed}s ({elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
