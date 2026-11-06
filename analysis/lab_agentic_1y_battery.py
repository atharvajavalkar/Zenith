#!/usr/bin/env python3
"""Agentic-policy Dukascopy 1y scalp battery (research only — no live wiring).

Stage A: current SYMBOL_ENTRY_BRAIN map ± agentic overlays → PnL deltas.
Stage B: full catalog (~44) × TFs × RR with best agent stack + 8-test battery.

Agentic overlays are deterministic proxies for the five LLM agents (no LLM-per-bar).

Usage:
  docker compose exec -T -w /app -e PYTHONPATH=/app backend \\
    python -m analysis.lab_agentic_1y_battery
  # Stage A only:
  python -m analysis.lab_agentic_1y_battery --stage a
  # Stage B only (uses A6 combo):
  python -m analysis.lab_agentic_1y_battery --stage b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import BOOK, RISK_USD, monte_carlo, verdict  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import (  # noqa: E402
    HEAVY,
    _battery,
    _catalog,
    _rank_score,
)
from analysis.strategy_lab_duka_scalp_1y_hunt import (  # noqa: E402
    HEAVY_OK_TF,
    RECENT_DAYS,
    RRS,
    TFS,
    WINDOW_DAYS,
    _load_frames_1y,
    _passes_scalp_gate,
    _slice_days,
)
from analysis.strategy_lab_intraday_hunt import _prog  # noqa: E402
from config import (  # noqa: E402
    SYMBOL_ENTRY_BRAIN,
    SYMBOL_SCAN_TFS,
    symbol_rr,
)

OUT = Path(ROOT) / "data" / "lab_agentic_1y_results.json"
LOG = Path(ROOT) / "data" / "lab_agentic_1y_battery.log"
LOCK = Path(ROOT) / "data" / "lab_agentic_1y_battery.lock"
PROGRESS = Path(ROOT) / "data" / "strategy_lab_progress.json"


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _stats(tr: list[dict]) -> dict[str, Any]:
    if not tr:
        return {"n": 0, "pnl": 0.0, "exp_r": 0.0, "win_rate": 0.0}
    rs = np.array([float(t["r"]) for t in tr], dtype=float)
    wins = int((rs > 0).sum())
    return {
        "n": int(len(tr)),
        "pnl": round(float(rs.sum() * RISK_USD), 2),
        "exp_r": round(float(rs.mean()), 4),
        "win_rate": round(100.0 * wins / len(tr), 1),
    }


def _simulate_detailed(df: pd.DataFrame, entries: list, rr: float) -> list[dict]:
    """Trades with exit index + ATR at entry for agentic filters."""
    high = df["high"].to_numpy(dtype=float, copy=False)
    low = df["low"].to_numpy(dtype=float, copy=False)
    close = df["close"].to_numpy(dtype=float, copy=False)
    n = len(df)
    prev_c = np.roll(close, 1)
    prev_c[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=5).mean().to_numpy()

    trades: list[dict] = []
    busy_until = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < busy_until or ei >= n - 1:
            continue
        side = e["side"]
        entry = float(e["entry"])
        sl0 = float(e["sl"])
        risk = abs(entry - sl0)
        if risk <= 0:
            continue
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk
        exit_r = None
        exit_i = ei
        if side == "BUY":
            for j in range(ei + 1, n):
                if low[j] <= sl or high[j] >= tp:
                    exit_r = -1.0 if low[j] <= sl else float(rr)
                    exit_i = j
                    break
        else:
            for j in range(ei + 1, n):
                if high[j] >= sl or low[j] <= tp:
                    exit_r = -1.0 if high[j] >= sl else float(rr)
                    exit_i = j
                    break
        if exit_r is None:
            continue
        a = float(atr[ei]) if atr[ei] == atr[ei] and atr[ei] > 0 else risk
        trades.append(
            {
                "i": ei,
                "exit_i": exit_i,
                "side": side,
                "r": round(float(exit_r), 4),
                "pnl": round(float(exit_r) * RISK_USD, 2),
                "atr": a,
                "risk_px": risk,
                "entry_t": str(df.index[ei])[:19],
                "exit_t": str(df.index[exit_i])[:19],
            }
        )
        busy_until = exit_i + 2
    return trades


def filter_baseline(trades: list[dict], **_kw) -> list[dict]:
    return list(trades)


def filter_risk(
    trades: list[dict],
    *,
    halt_r: float = -10.0,
    cooldown: int = 15,
    **_kw,
) -> list[dict]:
    """Risk Sentinel: pause after cum R ≤ halt_r; resume after `cooldown` skips.

    Skipped trades do not update equity (no look-ahead). Cooldown avoids
    permanent halt after an early drawdown.
    """
    out = []
    cum = 0.0
    skip_left = 0
    for t in trades:
        if skip_left > 0:
            skip_left -= 1
            continue
        out.append(t)
        cum += float(t["r"])
        if cum <= halt_r:
            skip_left = cooldown
            # soft reset so a later window can trade again
            cum = 0.0
    return out


def filter_research(
    trades: list[dict], *, streak: int = 2, skip_after: int = 1, **_kw
) -> list[dict]:
    """Research/self-heal: after `streak` losses, skip next `skip_after` trades."""
    out = []
    loss_streak = 0
    skip_left = 0
    for t in trades:
        if skip_left > 0:
            skip_left -= 1
            continue
        out.append(t)
        if float(t["r"]) < 0:
            loss_streak += 1
            if loss_streak >= streak:
                skip_left = skip_after
                loss_streak = 0
        else:
            loss_streak = 0
    return out


def filter_scout(
    trades: list[dict],
    *,
    window: int = 20,
    pause_er: float = -0.15,
    cooldown: int = 10,
    **_kw,
) -> list[dict]:
    """Hunt Scout: pause while rolling E[R] is deeply negative; cooldown resume."""
    out = []
    recent: list[float] = []
    skip_left = 0
    for t in trades:
        if skip_left > 0:
            skip_left -= 1
            continue
        out.append(t)
        recent.append(float(t["r"]))
        if len(recent) >= window:
            er = sum(recent[-window:]) / window
            if er <= pause_er:
                skip_left = cooldown
                recent = recent[-(window // 2) :]
    return out


def filter_debate(
    trades: list[dict], *, min_risk_atr: float = 0.85, **_kw
) -> list[dict]:
    """Bull/Bear conviction proxy: only take trades with risk ≥ min_risk_atr · ATR."""
    out = []
    for t in trades:
        atr = float(t.get("atr") or 0) or 1e-9
        if float(t.get("risk_px") or 0) / atr >= min_risk_atr:
            out.append(t)
    return out


def filter_news(
    trades: list[dict], df: Optional[pd.DataFrame] = None, **_kw
) -> list[dict]:
    """News proxy: skip entries when ATR is in top decile of prior 200 bars."""
    if df is None or len(df) < 50:
        return list(trades)
    close = df["close"].to_numpy(dtype=float, copy=False)
    high = df["high"].to_numpy(dtype=float, copy=False)
    low = df["low"].to_numpy(dtype=float, copy=False)
    prev_c = np.roll(close, 1)
    prev_c[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=5).mean()
    out = []
    for t in trades:
        ei = int(t["i"])
        if ei < 50:
            out.append(t)
            continue
        window = atr.iloc[max(0, ei - 200) : ei]
        if window.isna().all():
            out.append(t)
            continue
        thr = float(window.quantile(0.90))
        a = float(atr.iloc[ei]) if atr.iloc[ei] == atr.iloc[ei] else 0
        if a <= thr:
            out.append(t)
    return out


def filter_combo(trades: list[dict], df: Optional[pd.DataFrame] = None, **_kw) -> list[dict]:
    """A6: Risk + Research + Debate (primary agentic desk)."""
    t = filter_risk(trades)
    t = filter_research(t)
    t = filter_debate(t)
    return t


POLICIES: dict[str, Any] = {
    "A0_baseline": filter_baseline,
    "A1_risk": filter_risk,
    "A2_research": filter_research,
    "A3_scout": filter_scout,
    "A4_debate": filter_debate,
    "A5_news": filter_news,
    "A6_combo": filter_combo,
}


def _battery_from_trades(full_tr: list[dict], n_bars: int) -> dict[str, Any]:
    """Rebuild 8-test-style battery from an already-filtered trade list."""
    if not full_tr or n_bars < 250:
        return {
            "bars": n_bars,
            "entries": 0,
            "full": _stats([]),
            "oos": _stats([]),
            "monte_carlo": {},
            "verdict": "DEAD",
            "wf_green": 0,
            "wf_n": 0,
            "torture": {},
            "tests": {},
        }

    def by_range(start_i=0, end_i=None):
        end_i = n_bars - 1 if end_i is None else end_i
        return [t for t in full_tr if start_i <= int(t["i"]) <= end_i]

    full = _stats(full_tr)
    split = int(n_bars * 0.70)
    oos_tr = by_range(start_i=split)
    oos = _stats(oos_tr)

    wf = []
    wf_details = {}
    for k in range(4):
        ts = int(n_bars * (0.40 + 0.15 * k))
        te = int(n_bars * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 80:
            continue
        fold_oos = _stats(by_range(start_i=ts, end_i=te))
        wf.append({"fold": k + 1, "oos": fold_oos})
        wf_details[f"wf_fold_{k+1}"] = fold_oos

    rng = random.Random(7)
    spread_tr = [
        {
            **t,
            "r": round(float(t["r"]) - 0.15, 4),
            "pnl": round((float(t["r"]) - 0.15) * RISK_USD, 2),
        }
        for t in full_tr
    ]
    skip_tr = [t for t in full_tr if rng.random() >= 0.30]
    torture = {
        "spread_0.15R": _stats(spread_tr),
        "skip_30pct": _stats(skip_tr),
    }
    mc = monte_carlo(full_tr)
    v = verdict(full, oos, wf, torture, mc)

    tests = {
        "1_full": full,
        "2_oos70": oos,
        "3_wf_fold_1": wf_details.get("wf_fold_1") or {"n": 0, "exp_r": 0, "pnl": 0},
        "4_wf_fold_2": wf_details.get("wf_fold_2") or {"n": 0, "exp_r": 0, "pnl": 0},
        "5_wf_fold_3": wf_details.get("wf_fold_3") or {"n": 0, "exp_r": 0, "pnl": 0},
        "6_wf_fold_4": wf_details.get("wf_fold_4") or {"n": 0, "exp_r": 0, "pnl": 0},
        "7_torture_spread": torture["spread_0.15R"],
        "8_torture_skip": torture["skip_30pct"],
        "mc_prob_profit": mc.get("prob_profit"),
    }

    return {
        "bars": n_bars,
        "entries": len(full_tr),
        "full": full,
        "oos": oos,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": sum(1 for f in wf if f["oos"]["exp_r"] > 0),
        "wf_n": len(wf),
        "torture": {
            k: {"n": x["n"], "pnl": x["pnl"], "exp_r": x["exp_r"]} for k, x in torture.items()
        },
        "tests": tests,
    }


def _primary_tf(raw: Any, default: str = "15m") -> str:
    """SYMBOL_SCAN_TFS values are often tuples/lists like ('15m',)."""
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        return str(raw[0]) if raw else default
    return str(raw)


def _wired_map() -> dict[str, dict[str, Any]]:
    out = {}
    for sym in BOOK:
        brain = SYMBOL_ENTRY_BRAIN.get(sym) or SYMBOL_ENTRY_BRAIN.get(str(sym).upper())
        tf_raw = SYMBOL_SCAN_TFS.get(sym) or SYMBOL_SCAN_TFS.get(str(sym).upper())
        tf = _primary_tf(tf_raw, "15m")
        rr = float(symbol_rr(sym, default=2.0))
        if brain:
            out[sym] = {"strategy": brain, "tf": tf, "rr": rr}
    return out


def run_stage_a(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    catalog = {sid: (family, label, fn) for sid, family, label, fn in _catalog()}
    wired = _wired_map()
    _log(f"\n===== STAGE A · wired brains ± agents · {len(wired)} pairs =====")
    results: dict[str, Any] = {"policies": {}, "per_pair": {}}

    for pol_name in POLICIES:
        results["policies"][pol_name] = {"desk_full_pnl": 0.0, "pairs": {}}

    for pi, sym in enumerate(BOOK):
        spec = wired.get(sym)
        if not spec:
            continue
        sid, tf, rr = spec["strategy"], spec["tf"], spec["rr"]
        _log(f"\n## {sym} {sid}@{tf} RR={rr}")
        _prog(
            status="running",
            phase="agentic_stage_a",
            pct=int(5 + 40 * pi / len(BOOK)),
            message=f"Stage A [{sym}] {sid}@{tf}",
            symbol=sym,
        )
        if sid not in catalog:
            _log(f"  skip unknown strategy {sid}")
            continue
        frames = _load_frames_1y(sym, start, end)
        df = frames.get(tf)
        if df is None or len(df) < 400:
            _log(f"  no {tf} bars")
            continue
        family, label, fn = catalog[sid]
        entries = fn(df)
        raw_tr = _simulate_detailed(df, entries, rr)
        pair_row: dict[str, Any] = {
            "strategy": sid,
            "family": family,
            "label": label,
            "tf": tf,
            "rr": rr,
            "raw_n": len(raw_tr),
            "policies": {},
        }

        for pol_name, pol_fn in POLICIES.items():
            if pol_name == "A5_news":
                filtered = pol_fn(raw_tr, df=df)
            else:
                filtered = pol_fn(raw_tr)
            bat = _battery_from_trades(filtered, len(df))
            df_r = _slice_days(df, RECENT_DAYS, end)
            if len(df_r):
                cut = str(df_r.index[0].date())
                recent_tr = [t for t in filtered if t["exit_t"][:10] >= cut]
            else:
                recent_tr = []
            bat_r = _battery_from_trades(recent_tr, len(df_r) if len(df_r) else 0)
            if bat_r.get("full", {}).get("n", 0):
                gate = _passes_scalp_gate(bat, bat_r)
            else:
                gate = bat.get("verdict") == "SURVIVES" and bat.get("wf_green", 0) >= 2
            cell = {
                "n": bat["full"]["n"],
                "full_pnl": bat["full"]["pnl"],
                "oos_pnl": bat["oos"]["pnl"],
                "oos_exp_r": bat["oos"]["exp_r"],
                "win_rate": bat["full"]["win_rate"],
                "verdict": bat["verdict"],
                "wf_green": bat["wf_green"],
                "wf_n": bat["wf_n"],
                "deep_gate": gate,
                "tests": bat.get("tests"),
                "mc_prob": (bat.get("monte_carlo") or {}).get("prob_profit"),
            }
            pair_row["policies"][pol_name] = cell
            results["policies"][pol_name]["pairs"][sym] = cell
            results["policies"][pol_name]["desk_full_pnl"] = round(
                results["policies"][pol_name]["desk_full_pnl"] + float(cell["full_pnl"]),
                2,
            )
            _log(
                f"  {pol_name}: n={cell['n']} pnl=${cell['full_pnl']:,.0f} "
                f"OOS_E[R]={cell['oos_exp_r']:+.3f} {cell['verdict']} gate={gate}"
            )

        base = pair_row["policies"].get("A0_baseline") or {}
        base_pnl = float(base.get("full_pnl") or 0)
        for pol_name, cell in pair_row["policies"].items():
            cell["delta_vs_baseline"] = round(float(cell["full_pnl"]) - base_pnl, 2)
        results["per_pair"][sym] = pair_row

    base_desk = float(results["policies"]["A0_baseline"]["desk_full_pnl"])
    for pol_name, block in results["policies"].items():
        block["delta_vs_baseline"] = round(float(block["desk_full_pnl"]) - base_desk, 2)
    return results


def _cell_battery(
    df_all: pd.DataFrame,
    entries: list,
    rr: float,
    *,
    agent_policy: str,
    pol_fn,
) -> dict[str, Any]:
    """A0 uses native fast _battery; agent policies use filtered trade stream."""
    if agent_policy == "A0_baseline":
        return _battery(df_all, entries, rr)
    raw = _simulate_detailed(df_all, entries, rr)
    if agent_policy == "A5_news":
        filtered = pol_fn(raw, df=df_all)
    else:
        filtered = pol_fn(raw)
    return _battery_from_trades(filtered, len(df_all))


def run_stage_b(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    agent_policy: str = "A6_combo",
) -> dict[str, Any]:
    catalog = _catalog()
    pol_fn = POLICIES[agent_policy]
    _log(f"\n===== STAGE B · full catalog + {agent_policy} =====")
    per_pair: dict[str, Any] = {}
    survivors: dict[str, Any] = {}
    cells_total = 0
    survives_cells = 0

    for pi, sym in enumerate(BOOK):
        _log(f"\n######## {sym} ########")
        frames = _load_frames_1y(sym, start, end)
        if not frames:
            per_pair[sym] = {"error": "no_hist"}
            continue
        scored: list[dict] = []
        for sid, family, label, fn in catalog:
            for tf in TFS:
                if tf not in frames:
                    continue
                if sid in HEAVY and tf not in HEAVY_OK_TF:
                    continue
                df_all = frames[tf]
                cells_total += 1
                _prog(
                    status="running",
                    phase="agentic_stage_b",
                    pct=int(50 + 45 * (pi + 0.3) / len(BOOK)),
                    message=f"Stage B [{sym}] {sid}@{tf}",
                    symbol=sym,
                )
                try:
                    entries = fn(df_all)
                    if not entries or len(entries) < 30:
                        continue
                    df_r = _slice_days(df_all, RECENT_DAYS, end)
                    try:
                        entries_r = fn(df_r) if len(df_r) >= 200 else []
                    except Exception:
                        entries_r = []
                    best_cell = None
                    for rr in RRS:
                        bat = _cell_battery(
                            df_all, entries, rr, agent_policy=agent_policy, pol_fn=pol_fn
                        )
                        if len(df_r) < 200 or len(entries_r) < 15:
                            bat_r = {
                                "verdict": "DEAD",
                                "oos": {"exp_r": 0},
                                "full": {"n": 0},
                                "wf_green": 0,
                            }
                        else:
                            bat_r = _cell_battery(
                                df_r,
                                entries_r,
                                rr,
                                agent_policy=agent_policy,
                                pol_fn=pol_fn,
                            )
                        gate = _passes_scalp_gate(bat, bat_r)
                        cell = {
                            "strategy": sid,
                            "family": family,
                            "label": label,
                            "tf": tf,
                            "rr": rr,
                            "full_span": {
                                "verdict": bat["verdict"],
                                "full": bat["full"],
                                "oos": bat["oos"],
                                "wf_green": bat["wf_green"],
                                "wf_n": bat["wf_n"],
                                "monte_carlo": bat.get("monte_carlo"),
                                "tests": bat.get("tests"),
                            },
                            "last_90d": {
                                "verdict": bat_r.get("verdict"),
                                "full": bat_r.get("full"),
                                "oos": bat_r.get("oos"),
                                "wf_green": bat_r.get("wf_green"),
                            },
                            "deep_gate": gate,
                            "score": round(
                                _rank_score(bat, family)
                                + (2.5 if gate else 0)
                                + min(2.0, int(bat["full"]["n"]) / 200.0),
                                3,
                            ),
                        }
                        if best_cell is None or cell["score"] > best_cell["score"]:
                            best_cell = cell
                    if best_cell:
                        scored.append(best_cell)
                        if best_cell["full_span"].get("verdict") == "SURVIVES":
                            survives_cells += 1
                        if best_cell.get("deep_gate"):
                            _log(
                                f"  ✓ GATE {sid}@{tf} RR={best_cell['rr']} "
                                f"n={best_cell['full_span']['full']['n']} "
                                f"pnl=${best_cell['full_span']['full']['pnl']:,.0f}"
                            )
                except Exception as exc:
                    _log(f"  skip {sid}@{tf}: {exc}")

        scored.sort(key=lambda r: float(r.get("score") or -1e9), reverse=True)
        gated = [r for r in scored if r.get("deep_gate")]
        pool = gated if gated else [r for r in scored if r["full_span"].get("verdict") == "SURVIVES"]
        if not pool:
            pool = [r for r in scored if r["full_span"].get("verdict") != "DEAD"]
        winner = pool[0] if pool else None
        per_pair[sym] = {
            "gated_count": len(gated),
            "winner": winner,
            "top10": scored[:10],
        }
        if winner and winner.get("deep_gate"):
            survivors[sym] = {
                "strategy": winner["strategy"],
                "tf": winner["tf"],
                "rr": winner["rr"],
                "label": winner["label"],
                "full_pnl": winner["full_span"]["full"]["pnl"],
                "n": winner["full_span"]["full"]["n"],
                "oos_exp_r": winner["full_span"]["oos"]["exp_r"],
                "verdict": winner["full_span"]["verdict"],
                "score": winner["score"],
            }
            _log(
                f"  ★ {sym} WINNER {winner['strategy']}@{winner['tf']} "
                f"RR={winner['rr']} pnl=${winner['full_span']['full']['pnl']:,.0f}"
            )

    brain = ",".join(f"{s}:{survivors[s]['strategy']}" for s in BOOK if s in survivors)
    tfs = ",".join(f"{s}:{survivors[s]['tf']}" for s in BOOK if s in survivors)
    rrs = ",".join(f"{s}:{survivors[s]['rr']}" for s in BOOK if s in survivors)
    desk_pnl = sum(float(survivors[s]["full_pnl"]) for s in survivors)

    return {
        "agent_policy": agent_policy,
        "cells_total": cells_total,
        "survives_cells": survives_cells,
        "survivors_book": survivors,
        "per_pair": {
            k: {
                "gated_count": v.get("gated_count"),
                "winner": {
                    "strategy": (v.get("winner") or {}).get("strategy"),
                    "tf": (v.get("winner") or {}).get("tf"),
                    "rr": (v.get("winner") or {}).get("rr"),
                    "label": (v.get("winner") or {}).get("label"),
                    "full_pnl": ((v.get("winner") or {}).get("full_span") or {})
                    .get("full", {})
                    .get("pnl"),
                    "n": ((v.get("winner") or {}).get("full_span") or {})
                    .get("full", {})
                    .get("n"),
                    "verdict": ((v.get("winner") or {}).get("full_span") or {}).get(
                        "verdict"
                    ),
                    "deep_gate": (v.get("winner") or {}).get("deep_gate"),
                    "score": (v.get("winner") or {}).get("score"),
                    "tests": ((v.get("winner") or {}).get("full_span") or {}).get(
                        "tests"
                    ),
                }
                if v.get("winner")
                else None,
                "top5": [
                    {
                        "strategy": t["strategy"],
                        "tf": t["tf"],
                        "rr": t["rr"],
                        "score": t["score"],
                        "deep_gate": t.get("deep_gate"),
                        "full_pnl": t["full_span"]["full"]["pnl"],
                        "verdict": t["full_span"]["verdict"],
                    }
                    for t in (v.get("top10") or [])[:5]
                ],
            }
            for k, v in per_pair.items()
        },
        "proposed_SYMBOL_ENTRY_BRAIN": brain,
        "proposed_SYMBOL_SCAN_TFS": tfs,
        "proposed_SYMBOL_RR": rrs,
        "desk_full_pnl": round(desk_pnl, 2),
        "live_map": _wired_map(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("a", "b", "all"), default="all")
    ap.add_argument("--policy", default="A6_combo", help="Stage B agent policy key")
    args = ap.parse_args()

    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip() or "0")
            if pid and Path(f"/proc/{pid}").exists():
                _log(f"another agentic battery running (pid {pid})")
                return 2
        except Exception:
            pass
    LOCK.write_text(str(os.getpid()))
    if LOG.exists():
        LOG.write_text("")

    t0 = time.time()
    end = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=1), tz="UTC")
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    _log(
        f"AGENTIC 1y BATTERY · {start.date()}→{end.date()} · ${RISK_USD}/R\n"
        f"stage={args.stage} · LIVE BRAINS NOT MODIFIED\n"
    )

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "agentic_1y_battery",
        "window_days": WINDOW_DAYS,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "risk_usd": RISK_USD,
        "note": "Deterministic agentic policy proxies — no LLM-per-bar; no live wiring",
    }
    # Preserve prior stage_a when running --stage b alone
    if OUT.exists() and args.stage == "b":
        try:
            prior = json.loads(OUT.read_text(encoding="utf-8"))
            if isinstance(prior.get("stage_a"), dict):
                payload["stage_a"] = prior["stage_a"]
        except Exception:
            pass

    try:
        if args.stage in ("a", "all"):
            payload["stage_a"] = run_stage_a(start, end)
            pols = payload["stage_a"]["policies"]
            best = max(pols.items(), key=lambda kv: float(kv[1]["desk_full_pnl"]))
            payload["stage_a"]["best_desk_policy"] = best[0]
            payload["stage_a"]["best_desk_pnl"] = best[1]["desk_full_pnl"]
            _log(f"\nStage A best desk policy: {best[0]} @ ${best[1]['desk_full_pnl']:,.0f}")
            OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if args.stage in ("b", "all"):
            pol = args.policy
            if args.stage == "all" and payload.get("stage_a", {}).get("best_desk_policy"):
                best_p = payload["stage_a"]["best_desk_policy"]
                a6 = float(payload["stage_a"]["policies"]["A6_combo"]["desk_full_pnl"])
                best_v = float(payload["stage_a"]["policies"][best_p]["desk_full_pnl"])
                pol = best_p if best_v > a6 * 1.02 else "A6_combo"
            payload["stage_b"] = run_stage_b(start, end, agent_policy=pol)

        payload["elapsed_s"] = round(time.time() - t0, 1)
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _prog(
            status="done",
            phase="agentic_1y_battery",
            pct=100,
            message="Agentic 1y battery complete",
        )
        _log(f"\nWrote {OUT} in {payload['elapsed_s']}s")
        return 0
    except Exception as exc:
        _log(f"FAIL: {exc}")
        raise
    finally:
        try:
            LOCK.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
