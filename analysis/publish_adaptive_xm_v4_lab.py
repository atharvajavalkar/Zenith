#!/usr/bin/env python3
"""Publish adaptive XM-cost v4 → Strategy Lab (replaces board).

Research board only — does NOT enable AUTO_TRADE / live wire.
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

SRC = Path(ROOT) / "data" / "lab_adaptive_xm_cost_v4.json"
LAB = Path(ROOT) / "data" / "strategy_lab.json"
LAB_BAK = Path(ROOT) / "data" / "strategy_lab_pre_xm_v4_backup.json"
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
    "session_mix_no_off": "Session mix · flat Off",
    "circadian_kalman_z": "Circadian Kalman |z| thr",
    "circadian_strict": "Circadian Kalman strict (live 2.2)",
    "entropy_tod": "Perm-entropy ceiling = f(TOD)",
    "entropy_strict": "Entropy TOD + SNR floor",
    "vol_clock": "Minute-of-day vol clock",
    "tod_fade_boost": "London/NY open fade boost",
    "minute_softmax": "Bar-level softmax router",
    "hybrid_v2_tod": "v2 preferred + TOD gate",
    "circadian_thr": "SNR thr breathes with TOD",
    "liq_sessions_only": "Static only 08–21 UTC",
    "volmom": "Vol-scaled momentum",
    "static": "Static invented",
    "gated": "v2 regime gated",
    "switch": "v2 regime switch",
    "v2_preferred": "v2 preferred mode",
}


def _flat(mode: str, block: dict, *, family: str, static_id: str, rr: float) -> dict:
    label = FORMULA_LABELS.get(mode, mode)
    verd = block.get("verdict") or (
        "SURVIVES" if float(block.get("exp_r") or 0) > 0 else "FRAGILE"
    )
    if verd == "PASS":
        verd = "SURVIVES"
    elif verd == "SOFT":
        verd = "SOFT"
    elif verd == "FAIL":
        verd = "FRAGILE"
    return {
        "strategy": f"adaptive_xm_v4_{mode}",
        "strategy_base": mode,
        "label": f"{label} · XM-cost",
        "family": family if family in ("trend", "fade") else "adaptive",
        "tf": "5m",
        "rr": rr,
        "verdict": verd,
        "score": round(
            4.0 * float(block.get("exp_r") or 0)
            + 0.04 * float(block.get("win_rate") or 0)
            + min(2.0, int(block.get("n") or 0) / 200.0)
            + 0.5 * int(block.get("wf_green") or 0),
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
        "avg_spread_r": block.get("avg_spread_r"),
        "deep_gate": block.get("verdict") in ("PASS", "SOFT")
        or (
            float(block.get("exp_r") or 0) > 0 and int(block.get("n") or 0) >= 20
        ),
        "mode": mode,
        "static_id": static_id,
    }


def _block(row: dict, window: str, mode: str) -> dict:
    w = (row.get("windows") or {}).get(window) or {}
    if mode in w:
        return w[mode]
    a1 = row.get("all_1y") or {}
    if window == "1y" and mode in a1:
        return a1[mode]
    return {"n": 0, "pnl": 0.0, "exp_r": 0.0, "win_rate": 0.0, "verdict": "FAIL"}


def publish() -> dict[str, Any]:
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    per_src = raw.get("per_pair") or {}
    desk_abs = raw.get("desk_absolute") or {}
    desk_wire = raw.get("desk_wireable_only") or {}
    risk = float(raw.get("risk_usd") or 100.0)

    if LAB.exists():
        shutil.copy2(LAB, LAB_BAK)

    per_pair: dict[str, Any] = {}
    winners: dict[str, Any] = {}
    charts_oos = []
    charts_wr = []
    charts_trades = []
    charts_er = []
    desk_full = 0.0
    trades_total = 0
    wire_map = []

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
        best = _flat(aw, b1, family=fam, static_id=sid, rr=rr)
        all1 = row.get("all_1y") or {}
        ranked = sorted(
            all1.items(),
            key=lambda kv: (
                2
                if kv[1].get("verdict") == "PASS"
                else (1 if kv[1].get("verdict") == "SOFT" else 0),
                float(kv[1].get("pnl") or -1e18),
            ),
            reverse=True,
        )
        top10 = [
            _flat(m, blk, family=fam, static_id=sid, rr=rr) for m, blk in ranked[:10]
        ]
        per_pair[sym] = {
            "best": best,
            "top10": top10,
            "live_brain": f"adaptive_xm_v4:{aw}" if row.get("wireable") else "HALTED",
            "family": fam,
            "static_id": sid,
            "absolute_winner": aw,
            "absolute_verdict": row.get("absolute_verdict"),
            "wireable": row.get("wireable"),
            "v2_preferred": row.get("v2_preferred"),
            "beat_v2": row.get("beat_v2"),
            "sl_atr": row.get("sl_atr"),
            "spread_risk_mult": row.get("spread_risk_mult"),
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
            "wireable": row.get("wireable"),
            "avg_spread_r": best.get("avg_spread_r"),
        }
        if row.get("wireable"):
            wire_map.append(f"{sym}:{aw}")
        desk_full += float(best.get("full_pnl") or 0)
        trades_total += int(best.get("n") or 0)
        charts_oos.append(
            {
                "symbol": sym,
                "strategy": best["strategy"],
                "label": best["label"],
                "oos_exp_r": best["oos_exp_r"],
                "full_pnl": best["full_pnl"],
                "wireable": row.get("wireable"),
            }
        )
        charts_wr.append(
            {
                "symbol": sym,
                "win_rate": best.get("success_rate"),
                "verdict": best["verdict"],
            }
        )
        charts_trades.append({"symbol": sym, "n": best.get("n")})
        charts_er.append(
            {
                "symbol": sym,
                "exp_r": best.get("full_exp_r"),
                "avg_spread_r": best.get("avg_spread_r"),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "adaptive_xm_cost_v4",
        "title": "Adaptive XM-cost v4 (Dukascopy + XM spreads)",
        "subtitle": (
            "Re-hunt of adaptive minute/day brains with measured XM session spreads. "
            "PA/OB on hold. Live auto-trade NOT enabled."
        ),
        "risk_usd": risk,
        "tf": "5m",
        "research_only": True,
        "live_wired": False,
        "note": raw.get("note"),
        "source": str(SRC),
        "summary": {
            "pairs_with_survives": raw.get("pairs_wireable"),
            "strategies_tested": len(raw.get("formulas") or []) + 4,
            "desk_full_pnl": desk_abs.get("1y"),
            "desk_1y": desk_abs.get("1y"),
            "desk_90d": desk_abs.get("90d"),
            "desk_7d": desk_abs.get("7d"),
            "desk_3y": desk_abs.get("3y"),
            "desk_wireable_1y": desk_wire.get("1y"),
            "trades_total": trades_total,
            "pairs_wireable": raw.get("pairs_wireable"),
            "live_status": "paused_research",
        },
        "desk_summary": {
            "absolute_1y": desk_abs.get("1y"),
            "wireable_1y": desk_wire.get("1y"),
            "absolute_90d": desk_abs.get("90d"),
            "absolute_3y": desk_abs.get("3y"),
            "pairs_wireable": raw.get("pairs_wireable"),
            "desk_full_1y": round(desk_full, 2),
            "trades_1y": trades_total,
        },
        "hunt": {
            "mode": "adaptive_xm_cost_v4",
            "formulas": raw.get("formulas"),
            "spreads": "xm_session_costs",
            "pa_ob": "on_hold",
        },
        "proposed_SYMBOL_ENTRY_BRAIN": ",".join(wire_map),
        "walk_forward": {
            "desk": [
                {
                    "window": w,
                    "absolute": desk_abs.get(w),
                    "wireable": desk_wire.get(w),
                }
                for w in ("7d", "90d", "1y", "3y")
            ]
        },
        "winners": winners,
        "per_pair": per_pair,
        "charts": {
            "best_oos_by_pair": [
                {
                    "symbol": r["symbol"],
                    "strategy": f"adaptive_xm_v4_{r.get('absolute')}",
                    "label": r.get("absolute"),
                    "oos_exp_r": r.get("exp_r"),
                    "full_pnl": r.get("pnl"),
                    "wireable": r.get("wireable"),
                }
                for r in (raw.get("charts") or {}).get("pair_1y") or []
            ],
            "oos_exp_r": charts_oos,
            "success_by_pair": [
                {
                    "symbol": r["symbol"],
                    "success_rate": (per_pair.get(r["symbol"]) or {})
                    .get("best", {})
                    .get("success_rate"),
                    "verdict": r.get("verdict"),
                }
                for r in (raw.get("charts") or {}).get("pair_1y") or []
            ],
            "win_rate": charts_wr,
            "trades_by_pair": [
                {"symbol": r["symbol"], "n": r.get("n")}
                for r in (raw.get("charts") or {}).get("pair_1y") or []
            ],
            "trades": charts_trades,
            "exp_r_by_pair": [
                {
                    "symbol": r["symbol"],
                    "exp_r": r.get("exp_r"),
                    "avg_spread_r": r.get("avg_spread_r"),
                }
                for r in (raw.get("charts") or {}).get("pair_1y") or []
            ],
            "exp_r_vs_spread": charts_er,
            "desk_modes": (raw.get("charts") or {}).get("desk"),
            "desk": (raw.get("charts") or {}).get("desk"),
            "walk_forward_desk": [
                {
                    "window": w,
                    "absolute": desk_abs.get(w),
                    "wireable": desk_wire.get(w),
                }
                for w in ("7d", "90d", "1y", "3y")
            ],
            "pair_1y": (raw.get("charts") or {}).get("pair_1y"),
        },
        "formulas": raw.get("formulas"),
        "elapsed_sec": raw.get("elapsed_sec"),
        "elapsed_s": raw.get("elapsed_sec"),
    }
    LAB.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    PROGRESS.write_text(
        json.dumps(
            {
                "status": "done",
                "mode": "adaptive_xm_cost_v4",
                "desk_1y": desk_abs.get("1y"),
                "wireable_1y": desk_wire.get("1y"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Published Strategy Lab · desk 1y ${desk_abs.get('1y')} · "
        f"wireable ${desk_wire.get('1y')} · map={','.join(wire_map) or 'none'}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    publish()
