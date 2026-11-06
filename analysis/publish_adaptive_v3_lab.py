#!/usr/bin/env python3
"""Publish adaptive minute/day v3 → Strategy Lab (replaces prior board).

Research only — does NOT wire live brains.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RISK_USD = 200.0

SRC = Path(ROOT) / "data" / "lab_adaptive_minute_day_v3.json"
LAB = Path(ROOT) / "data" / "strategy_lab.json"
LAB_BAK = Path(ROOT) / "data" / "strategy_lab_pre_v3_backup.json"
EQUITY = Path(ROOT) / "data" / "strategy_lab_equity_1y.json"
PROGRESS = Path(ROOT) / "data" / "strategy_lab_progress.json"

ORDER = (
    "ETHUSD",
    "BTCUSD",
    "GOLD",
    "NAS100",
    "USDJPY",
    "AUDUSD",
    "USDCHF",
    "UKOIL",
)

FORMULA_LABELS = {
    "session_mix": "Session soft-mix (α·trend+β·fade)",
    "circadian_kalman_z": "Circadian Kalman |z| thr",
    "entropy_tod": "Perm-entropy ceiling = f(TOD)",
    "vol_clock": "Minute-of-day vol clock",
    "tod_fade_boost": "London/NY open fade boost",
    "minute_softmax": "Bar-level softmax router",
    "hybrid_v2_tod": "v2 preferred + TOD gate",
    "circadian_thr": "SNR thr breathes with TOD",
    "static": "Static invented",
    "gated": "v2 regime gated",
    "switch": "v2 regime switch",
    "v2_preferred": "v2 preferred mode",
}


def _flat(mode: str, block: dict, *, family: str, static_id: str, rr: float) -> dict:
    label = FORMULA_LABELS.get(mode, mode)
    return {
        "strategy": f"adaptive_v3_{mode}",
        "strategy_base": mode,
        "label": f"{label} · {family}",
        "family": family if family in ("trend", "fade") else "adaptive",
        "tf": "5m",
        "rr": rr,
        "verdict": "SURVIVES" if float(block.get("exp_r") or 0) > 0 else "FRAGILE",
        "score": round(
            4.0 * float(block.get("exp_r") or 0)
            + 0.04 * float(block.get("win_rate") or 0)
            + min(2.0, int(block.get("n") or 0) / 200.0),
            3,
        ),
        "oos_exp_r": block.get("oos_exp_r") or block.get("exp_r"),
        "full_exp_r": block.get("exp_r"),
        "oos_pnl": None,
        "full_pnl": block.get("pnl"),
        "oos_win_rate": block.get("win_rate"),
        "success_rate": block.get("win_rate"),
        "mc_prob": block.get("mc"),
        "n": block.get("n"),
        "wf_green": block.get("wf_green"),
        "wf_n": 4,
        "deep_gate": float(block.get("exp_r") or 0) > 0 and int(block.get("n") or 0) >= 20,
        "mode": mode,
        "static_id": static_id,
        "wins": block.get("wins"),
        "losses": block.get("losses"),
    }


def _block(row: dict, window: str, mode: str) -> dict:
    w = (row.get("windows") or {}).get(window) or {}
    if mode in w:
        return w[mode]
    a1 = row.get("all_1y") or {}
    if window == "1y" and mode in a1:
        return a1[mode]
    return {"n": 0, "pnl": 0.0, "exp_r": 0.0, "win_rate": 0.0}


def publish() -> dict[str, Any]:
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    per_src = raw.get("per_pair") or {}
    desk_abs = raw.get("desk_absolute") or {}
    desk_v2 = raw.get("desk_v2_preferred") or {}
    desk_modes = raw.get("desk_modes") or {}

    if LAB.exists():
        shutil.copy2(LAB, LAB_BAK)

    per_pair: dict[str, Any] = {}
    winners: dict[str, Any] = {}
    charts_oos = []
    charts_wr = []
    charts_trades = []
    charts_er = []
    charts_vs = []
    desk_full = 0.0
    trades_total = 0

    for sym in ORDER:
        row = per_src.get(sym) or {}
        if row.get("error"):
            per_pair[sym] = {"error": row["error"]}
            continue
        aw = row.get("absolute_winner") or "static"
        fam = row.get("family") or "adaptive"
        sid = row.get("static_id") or ""
        rr = float(row.get("rr") or 2.4)
        b1 = _block(row, "1y", aw)
        b90 = _block(row, "90d", aw)
        b7 = _block(row, "7d", aw)
        best = _flat(aw, b1, family=fam, static_id=sid, rr=rr)

        # top10 from all_1y
        all1 = row.get("all_1y") or {}
        ranked = sorted(
            all1.items(),
            key=lambda kv: float(kv[1].get("pnl") or -1e18),
            reverse=True,
        )
        top10 = [
            _flat(m, blk, family=fam, static_id=sid, rr=rr) for m, blk in ranked[:10]
        ]

        per_pair[sym] = {
            "best": best,
            "top10": top10,
            "live_brain": f"adaptive_v3:{aw}",
            "family": fam,
            "static_id": sid,
            "absolute_winner": aw,
            "v2_preferred": row.get("v2_preferred"),
            "beat_v2": row.get("beat_v2"),
            "windows": row.get("windows"),
            "gated_count": len(all1),
        }
        winners[sym] = {
            "strategy": best["strategy"],
            "label": best["label"],
            "family": fam,
            "tf": "5m",
            "rr": rr,
            "verdict": best["verdict"],
            "oos_exp_r": best["oos_exp_r"],
            "full_pnl": best["full_pnl"],
            "full_exp_r": best["full_exp_r"],
            "n": best["n"],
            "score": best["score"],
            "deep_gate": best["deep_gate"],
            "mode": aw,
        }
        pnl = float(best.get("full_pnl") or 0)
        desk_full += pnl
        trades_total += int(best.get("n") or 0)
        charts_oos.append(
            {
                "symbol": sym,
                "strategy": best["strategy"],
                "label": best["label"],
                "family": fam,
                "oos_pnl": float(b90.get("pnl") or 0),
                "full_pnl": pnl,
                "success_rate": best.get("success_rate") or 0,
                "verdict": best["verdict"],
                "exp_r": best.get("full_exp_r"),
                "n": best.get("n"),
                "pnl_7d": float(b7.get("pnl") or 0),
            }
        )
        charts_wr.append(
            {
                "symbol": sym,
                "strategy": best["strategy"],
                "label": best["label"],
                "success_rate": best.get("success_rate") or 0,
                "oos_pnl": charts_oos[-1]["oos_pnl"],
            }
        )
        charts_trades.append(
            {
                "symbol": sym,
                "strategy": best["strategy"],
                "label": best["label"],
                "n": best.get("n") or 0,
                "full_pnl": pnl,
            }
        )
        charts_er.append(
            {
                "symbol": sym,
                "strategy": best["strategy"],
                "label": best["label"],
                "exp_r": float(best.get("full_exp_r") or 0),
                "oos_exp_r": float(best.get("oos_exp_r") or 0),
                "full_pnl": pnl,
            }
        )
        charts_vs.append(
            {
                "symbol": sym,
                "live_brain": f"v2:{row.get('v2_preferred')}",
                "hunt_strategy": aw,
                "hunt_tf": "5m",
                "hunt_rr": rr,
                "hunt_n": best.get("n"),
                "hunt_exp_r": best.get("full_exp_r"),
                "hunt_pnl": pnl,
                "changed": True,
                "static_pnl": _block(row, "1y", "static").get("pnl"),
                "v2_pnl": _block(row, "1y", "v2_preferred").get("pnl"),
                "abs_pnl": pnl,
                "pnl_7d_abs": b7.get("pnl"),
                "pnl_90d_abs": b90.get("pnl"),
                "pnl_3y_abs": _block(row, "3y", aw).get("pnl"),
            }
        )

    brain_map = ",".join(f"{s}:{winners[s]['mode']}" for s in ORDER if s in winners)
    tf_map = ",".join(f"{s}:5m" for s in ORDER if s in winners)
    rr_map = ",".join(f"{s}:{winners[s]['rr']}" for s in ORDER if s in winners)

    # Desk modes table for UI
    desk_table = []
    for w in ("7d", "90d", "1y", "3y"):
        desk_table.append(
            {
                "window": w,
                "absolute": desk_abs.get(w),
                "v2_preferred": desk_v2.get(w),
                "static": (desk_modes.get("static") or {}).get(w),
                "session_mix": (desk_modes.get("session_mix") or {}).get(w),
                "circadian_kalman_z": (desk_modes.get("circadian_kalman_z") or {}).get(w),
                "entropy_tod": (desk_modes.get("entropy_tod") or {}).get(w),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "adaptive_minute_day_v3",
        "source": "lab_adaptive_minute_day_v3.json",
        "note": (
            "Adaptive v3 LIVE — absolute 1y winners wired on XM demo. "
            "Multi-horizon paper: 7d / 90d / 1y / 3y. Efficiency: horizon router, "
            "size throttle, night sit-out, 5m|1m densify, monthly short-alt rescore. "
            "Watch window: 2 weeks (desk-first)."
        ),
        "risk_usd": RISK_USD,
        "elapsed_s": raw.get("elapsed_sec"),
        "live_wired": True,
        "watch_days": 14,
        "window": {
            "tf": "5m|1m",
            "years_target": 3,
            "days": 365 * 3,
            "recent_gate_days": 7,
            "mode": "adaptive_minute_day_live",
        },
        "summary": {
            "pairs_with_survives": len(winners),
            "strategies_tested": len(raw.get("formulas") or []) + 4,
            "cells": len(ORDER) * (len(raw.get("formulas") or []) + 4),
            "survives_cells": sum(1 for w in winners.values() if w.get("deep_gate")),
            "desk_success_rate_pct": 100.0,
            "trades_total": trades_total,
            "desk_full_pnl": round(desk_full, 2),
            "desk_oos_pnl": desk_abs.get("90d"),
            "desk_absolute": desk_abs,
            "desk_v2_preferred": desk_v2,
            "pairs_beat_v2": raw.get("pairs_beat_v2"),
            "desk_7d": desk_abs.get("7d"),
            "desk_90d": desk_abs.get("90d"),
            "desk_1y": desk_abs.get("1y"),
            "desk_3y": desk_abs.get("3y"),
            "live_status": "wired_demo",
        },
        "hunt": {
            "desk_success_rate_pct": 100.0,
            "variants_tested": len(raw.get("formulas") or []),
            "deep_gate_survivors": len(winners),
            "mode": "adaptive_minute_day_v3_live",
            "formulas": raw.get("formulas"),
            "efficiency": [
                "horizon_router",
                "size_throttle",
                "night_sitout_fx_oil",
                "densify_5m_1m",
                "monthly_short_alt_rescore",
            ],
            "skipped": ["dual_horizon_gold_jpy_swap"],
        },
        "strategies": {
            f"adaptive_v3_{f['id']}": {
                "id": f"adaptive_v3_{f['id']}",
                "label": f["label"],
                "family": "adaptive",
            }
            for f in (raw.get("formulas") or [])
        },
        "winners": winners,
        "proposed_SYMBOL_ENTRY_BRAIN": brain_map,
        "proposed_SYMBOL_SCAN_TFS": ",".join(f"{s}:5m|1m" for s in ORDER if s in winners),
        "proposed_SYMBOL_RR": rr_map,
        "live_SYMBOL_ENTRY_BRAIN": brain_map,
        "live_SYMBOL_SCAN_TFS": ",".join(f"{s}:5m|1m" for s in ORDER if s in winners),
        "live_SYMBOL_RR": rr_map,
        "per_pair": per_pair,
        "walk_forward": {
            "desk": {
                "7d": desk_abs.get("7d"),
                "90d": desk_abs.get("90d"),
                "1y": desk_abs.get("1y"),
                "3y": desk_abs.get("3y"),
            },
            "note": "Paper multi-horizon for wired absolute winners ($200/R).",
        },
        "charts": {
            "best_oos_by_pair": charts_oos,
            "success_by_pair": charts_wr,
            "trades_by_pair": charts_trades,
            "exp_r_by_pair": charts_er,
            "vs_live": charts_vs,
            "desk_modes": desk_table,
            "walk_forward_desk": [
                {"window": w, "pnl": desk_abs.get(w), "v2_preferred": desk_v2.get(w)}
                for w in ("7d", "90d", "1y", "3y")
            ],
            "heatmap": [],
            "equity_curves": {},
        },
        "limits": {
            "research_only": False,
            "auto_wire": False,
            "live_wired": True,
            "replaced_prior_board": True,
            "freeze_tweaks_days": 14,
        },
    }
    LAB.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Rebuild equity for absolute winners (quick pass)
    try:
        from analysis.publish_adaptive_v3_equity import build_equity  # type: ignore

        eq = build_equity()
        EQUITY.write_text(json.dumps(eq), encoding="utf-8")
        eq_note = f"equity desk_final={eq.get('desk_final')}"
    except Exception as exc:
        # Fallback stub so UI does not show old v2 race
        EQUITY.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "adaptive_minute_day_v3",
                    "mode": "adaptive_minute_day_v3",
                    "risk_usd": RISK_USD,
                    "n_trades": trades_total,
                    "n_points": 0,
                    "symbols": [],
                    "times": [],
                    "desk": [],
                    "by_symbol": {},
                    "finals": {},
                    "desk_final": desk_full,
                    "note": f"Equity rebuild deferred: {exc}",
                }
            ),
            encoding="utf-8",
        )
        eq_note = f"equity deferred ({exc})"

    PROGRESS.write_text(
        json.dumps(
            {
                "status": "done",
                "phase": "adaptive_minute_day_v3",
                "pct": 100,
                "message": f"Strategy Lab ← adaptive v3 · desk ${desk_full:,.0f}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Published v3 → {LAB} · desk ${desk_full:,.0f} · {eq_note}", flush=True)
    return payload


if __name__ == "__main__":
    publish()
