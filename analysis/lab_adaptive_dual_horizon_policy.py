#!/usr/bin/env python3
"""RESEARCH — Dual-horizon efficiency policy from adaptive v3 results.

Goal: find how the engine can work on SHORT (7d/90d) and LONG (1y/3y)
without wiring live brains.

Outputs:
  data/lab_adaptive_dual_horizon_policy.json
  data/lab_adaptive_dual_horizon_policy.md
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "lab_adaptive_minute_day_v3.json"
OUT = ROOT / "data" / "lab_adaptive_dual_horizon_policy.json"
MD = ROOT / "data" / "lab_adaptive_dual_horizon_policy.md"

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

# Modes we can score (need presence in windows or all_1y)
CORE = (
    "static",
    "gated",
    "switch",
    "v2_preferred",
    "session_mix",
    "circadian_kalman_z",
    "entropy_tod",
    "vol_clock",
    "tod_fade_boost",
    "minute_softmax",
    "hybrid_v2_tod",
    "circadian_thr",
    "harmonic_gate",
    "dow_book",
    "weekday_regime_prior",
    "intraday_cusum_clock",
)


def _get(row: dict, window: str, mode: str) -> dict:
    w = (row.get("windows") or {}).get(window) or {}
    if mode in w:
        return w[mode]
    if window == "1y":
        return (row.get("all_1y") or {}).get(mode) or {}
    return {}


def _pnl(row: dict, window: str, mode: str) -> float:
    return float(_get(row, window, mode).get("pnl") or 0.0)


def _er(row: dict, window: str, mode: str) -> float:
    return float(_get(row, window, mode).get("exp_r") or 0.0)


def _modes_for_pair(row: dict) -> list[str]:
    modes = set((row.get("all_1y") or {}).keys())
    for w in ("7d", "90d", "1y", "3y"):
        modes |= set(((row.get("windows") or {}).get(w) or {}).keys())
    # keep only those with at least 1y data
    return [m for m in CORE if m in modes or m in (row.get("all_1y") or {})]


def score_mode(row: dict, mode: str) -> dict[str, Any]:
    """Composite: short + long efficiency (not just 1y max)."""
    p7 = _pnl(row, "7d", mode)
    p90 = _pnl(row, "90d", mode)
    p1 = _pnl(row, "1y", mode)
    p3 = _pnl(row, "3y", mode)
    e7 = _er(row, "7d", mode)
    e90 = _er(row, "90d", mode)
    e1 = _er(row, "1y", mode)
    e3 = _er(row, "3y", mode)

    short_pnl = 0.45 * p7 + 0.55 * p90
    long_pnl = 0.40 * p1 + 0.60 * p3
    short_er = 0.45 * e7 + 0.55 * e90
    long_er = 0.40 * e1 + 0.60 * e3

    # Penalize if short is red while long is green (fragile recovery) and vice versa
    sign_align = 1.0
    if (short_pnl < 0) != (long_pnl < 0) and abs(short_pnl) > 500:
        sign_align = 0.75
    if short_pnl < 0 and long_pnl < 0:
        sign_align = 0.5

    # Dual score: equal weight short/long PnL + ER bonus
    dual = sign_align * (0.5 * short_pnl + 0.5 * long_pnl) + 2000.0 * (
        0.5 * short_er + 0.5 * long_er
    )
    return {
        "mode": mode,
        "pnl_7d": p7,
        "pnl_90d": p90,
        "pnl_1y": p1,
        "pnl_3y": p3,
        "short_pnl": round(short_pnl, 2),
        "long_pnl": round(long_pnl, 2),
        "short_er": round(short_er, 4),
        "long_er": round(long_er, 4),
        "dual_score": round(dual, 2),
        "short_green": short_pnl > 0,
        "long_green": long_pnl > 0,
        "both_green": short_pnl > 0 and long_pnl > 0,
    }


def main() -> int:
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    per = raw.get("per_pair") or {}

    per_out: dict[str, Any] = {}
    desk_abs_1y = 0.0
    desk_dual_1y = 0.0
    desk_dual_7d = 0.0
    desk_dual_3y = 0.0
    desk_short_1y = 0.0  # if we picked best short-only
    both_green_n = 0

    lines = [
        "# Dual-horizon adaptive policy",
        "",
        "Research only — live brains not wired.",
        "",
        "Problem: absolute 1y winners can be weak on 7d (or vice versa).",
        "Solution candidates:",
        "1. **Dual-score pick** — maximize short+long composite per pair",
        "2. **Horizon router** — short book for 7–90d recovery, long book for compound",
        "3. **Both-green filter** — only trade formula if short AND long green",
        "",
    ]

    for sym in ORDER:
        row = per.get(sym) or {}
        if row.get("error"):
            per_out[sym] = {"error": row["error"]}
            continue
        modes = _modes_for_pair(row)
        scored = [score_mode(row, m) for m in modes]
        scored.sort(key=lambda x: x["dual_score"], reverse=True)
        abs_1y = row.get("absolute_winner")
        best_dual = scored[0]
        best_short = max(scored, key=lambda x: x["short_pnl"])
        best_long = max(scored, key=lambda x: x["long_pnl"])
        both = [s for s in scored if s["both_green"]]
        both_best = both[0] if both else None
        if both_best:
            both_green_n += 1

        abs_score = next((s for s in scored if s["mode"] == abs_1y), None)

        per_out[sym] = {
            "absolute_1y": abs_1y,
            "best_dual": best_dual,
            "best_short": best_short,
            "best_long": best_long,
            "best_both_green": both_best,
            "absolute_1y_score": abs_score,
            "top5_dual": scored[:5],
            "recommendation": {
                "wire_candidate": (both_best or best_dual)["mode"],
                "reason": (
                    "both-green dual winner"
                    if both_best
                    else "best dual score (short/long conflict)"
                ),
            },
        }

        desk_abs_1y += _pnl(row, "1y", abs_1y)
        dm = best_dual["mode"]
        desk_dual_1y += _pnl(row, "1y", dm)
        desk_dual_7d += _pnl(row, "7d", dm)
        desk_dual_3y += _pnl(row, "3y", dm)
        desk_short_1y += _pnl(row, "1y", best_short["mode"])

        lines.append(f"## {sym}")
        lines.append(f"- 1y absolute: `{abs_1y}` (${_pnl(row,'1y',abs_1y):,.0f})")
        lines.append(
            f"- dual pick: `{best_dual['mode']}` "
            f"(7d ${best_dual['pnl_7d']:,.0f} · 90d ${best_dual['pnl_90d']:,.0f} · "
            f"1y ${best_dual['pnl_1y']:,.0f} · 3y ${best_dual['pnl_3y']:,.0f})"
        )
        if both_best:
            lines.append(f"- both-green: `{both_best['mode']}`")
        else:
            lines.append("- both-green: none (horizon conflict)")
        lines.append(
            f"- recommend wire candidate: `{per_out[sym]['recommendation']['wire_candidate']}`"
        )
        lines.append("")

    # Engine design summary
    design = {
        "short_term": {
            "goal": "Survive 7d–90d without blowing recovery",
            "levers": [
                "Raise entry thresholds when short window E[R]<0",
                "Session mix / fade boost around London+NY opens",
                "Sit-out Asia deep night on FX/oil",
                "Prefer formulas with both_green on 7d+90d",
            ],
        },
        "long_term": {
            "goal": "Compound 1y–3y with stable E[R]",
            "levers": [
                "Keep circadian Kalman / session_mix as core books",
                "Do not over-gate (v2 gated lost vs static on long)",
                "Per-pair formula (not one global router)",
                "Re-score dual horizon monthly; freeze RR/SL from invention",
            ],
        },
        "unified_engine": {
            "name": "dual_horizon_adaptive_v3",
            "rules": [
                "At each bar: compute calendar + regime features (already in v3)",
                "Pick per-symbol formula from dual-score table (not pure 1y max)",
                "If short book ≠ long book: use long book by default; "
                "switch to short book only when 7d rolling PnL of long book < −2R AND short book both-green",
                "Hard sit-out if live regime FLAT AND formula is trend-family conflict",
                "Never auto-wire until user confirms",
            ],
        },
    }

    lines.extend(
        [
            "## Desk totals",
            f"- absolute 1y mix: ${desk_abs_1y:,.0f}",
            f"- dual-pick 1y: ${desk_dual_1y:,.0f}",
            f"- dual-pick 7d: ${desk_dual_7d:,.0f}",
            f"- dual-pick 3y: ${desk_dual_3y:,.0f}",
            f"- pairs with both-green formula: {both_green_n}/8",
            "",
            "## How to be efficient short AND long",
            "1. Stop optimizing only 1y PnL — use dual score (0.5 short + 0.5 long).",
            "2. Prefer both-green formulas; if none, keep long book and throttle size short-term.",
            "3. Per-pair books beat one global adaptive router.",
            "4. v2 gated/switch alone is not enough — calendar equations (session_mix, circadian_kalman_z) carry the edge.",
            "5. Live wire only after confirm — proposed map is research.",
            "",
        ]
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "recommend_live": False,
        "source": "lab_adaptive_minute_day_v3.json",
        "desk": {
            "absolute_1y": round(desk_abs_1y, 2),
            "dual_1y": round(desk_dual_1y, 2),
            "dual_7d": round(desk_dual_7d, 2),
            "dual_3y": round(desk_dual_3y, 2),
            "short_opt_1y": round(desk_short_1y, 2),
            "both_green_pairs": both_green_n,
        },
        "design": design,
        "per_pair": per_out,
        "proposed_wire_map": {
            s: per_out[s]["recommendation"]["wire_candidate"]
            for s in ORDER
            if s in per_out and not per_out[s].get("error")
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"dual_1y=${desk_dual_1y:,.0f} abs_1y=${desk_abs_1y:,.0f} "
        f"dual_7d=${desk_dual_7d:,.0f} both_green={both_green_n}/8 → {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
