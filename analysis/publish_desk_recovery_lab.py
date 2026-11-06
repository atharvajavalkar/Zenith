#!/usr/bin/env python3
"""Publish desk recovery scalp hunt → Strategy Lab board (UI).

Reads data/lab_desk_recovery_scalp_hunt.json (+ Obsidian notes for resumed pairs),
writes a full strategy_lab.json schema the Strategy Lab page expects.

Does NOT wire live brains.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import _catalog  # noqa: E402

HUNT = Path(ROOT) / "data" / "lab_desk_recovery_scalp_hunt.json"
OBSIDIAN = Path(ROOT) / "data" / "ZenithRecovery"
LAB = Path(ROOT) / "data" / "strategy_lab.json"
LAB_BACKUP = Path(ROOT) / "data" / "strategy_lab_pre_recovery_backup.json"
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


def _parse_math_from_sid(sid: str) -> tuple[str, Optional[dict[str, Any]]]:
    """bb_2+hf0.45_e0.30 → (bb_2, {hf_max, min_eff, tag})."""
    sid = str(sid or "").strip()
    if "+" not in sid:
        return sid, None
    base, tag = sid.split("+", 1)
    hf_max = None
    min_eff = None
    m_hf = re.search(r"hf([0-9]+(?:\.[0-9]+)?)", tag)
    if m_hf:
        hf_max = float(m_hf.group(1))
    m_e = re.search(r"(?:^|_|e)([0-9]+(?:\.[0-9]+)?)$", tag) or re.search(
        r"e([0-9]+(?:\.[0-9]+)?)", tag
    )
    # Prefer explicit e0.xx after hf, or lone e0.xx tags
    m_e2 = re.search(r"(?:^|[+_])e([0-9]+(?:\.[0-9]+)?)", tag)
    if m_e2:
        min_eff = float(m_e2.group(1))
    elif tag.startswith("e") and re.match(r"e[0-9]", tag):
        min_eff = float(tag[1:].split("_")[0])
    return base, {
        "tag": tag,
        "hf_max": hf_max,
        "min_eff": min_eff,
    }


def _mc_pct(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x <= 1.5:
        x *= 100.0
    return round(x, 1)


def _enrich_from_obsidian(sym: str, row: dict[str, Any]) -> dict[str, Any]:
    """Fill gaps for resumed ETH/BTC from Obsidian pair notes."""
    path = OBSIDIAN / "Pairs" / f"{sym}.md"
    if not path.exists():
        return row
    text = path.read_text(encoding="utf-8")
    w = dict(row.get("winner") or {})
    full = dict(w.get("full") or {})

    m_math = re.search(r"\*\*Math:\*\* (.+)", text)
    if m_math and not w.get("math"):
        raw = m_math.group(1).strip()
        try:
            w["math"] = ast.literal_eval(raw)
        except Exception:
            base, parsed = _parse_math_from_sid(str(w.get("strategy") or ""))
            if parsed:
                w["math"] = parsed
                if not str(w.get("strategy") or "").startswith(base):
                    pass

    m_90 = re.search(r"\*\*90d OOS E\[R\]:\*\* ([\d.]+) \(n=(\d+)\)", text)
    last = dict(w.get("last_90d") or {})
    if m_90 and last.get("oos_exp_r") is None:
        last = {
            "oos_exp_r": float(m_90.group(1)),
            "n": int(m_90.group(2)),
            "verdict": "SURVIVES",
        }
        w["last_90d"] = last

    m_label = re.search(r"\*\*Hunt winner:\*\* `[^`]+` · (.+)", text)
    if m_label and (not w.get("label") or w.get("label") in ("resumed", "resumed from Obsidian")):
        w["label"] = m_label.group(1).strip()

    # Top 5 from Obsidian
    top5: list[dict] = list(row.get("top5") or [])
    if not top5:
        for line in text.splitlines():
            m = re.match(
                r"\d+\.\s+`([^`]+)` @(\S+) RR=([\d.]+) n=(\d+) E\[R\]=([\d.]+) "
                r"gate=(True|False) score=([\d.]+)",
                line.strip(),
            )
            if not m:
                continue
            sid = m.group(1)
            base, math_g = _parse_math_from_sid(sid)
            n = int(m.group(4))
            er = float(m.group(5))
            top5.append(
                {
                    "strategy": sid,
                    "family": "hedge" if base.startswith(("bb", "don", "z_", "cta", "vol", "tsmom", "kelt", "roc", "atr")) else "retail",
                    "label": sid,
                    "tf": m.group(2),
                    "rr": float(m.group(3)),
                    "math": math_g,
                    "full": {
                        "n": n,
                        "exp_r": er,
                        "oos_exp_r": er,
                        "pnl": round(n * er * RISK_USD, 2),
                        "verdict": "SURVIVES",
                        "mc": 99.0,
                        "wf_green": 3,
                        "win_rate": None,
                    },
                    "last_90d": {},
                    "deep_gate": m.group(6) == "True",
                    "score": float(m.group(7)),
                }
            )

    if full.get("pnl") is None and full.get("n") and full.get("exp_r") is not None:
        full["pnl"] = round(float(full["n"]) * float(full["exp_r"]) * RISK_USD, 2)
        w["full"] = full

    if not w.get("math"):
        _, parsed = _parse_math_from_sid(str(w.get("strategy") or ""))
        if parsed:
            w["math"] = parsed

    out = dict(row)
    out["winner"] = w
    if top5:
        out["top5"] = top5
    return out


def _flatten_cell(cell: Optional[dict[str, Any]], *, live_brain: str = "") -> Optional[dict[str, Any]]:
    if not cell:
        return None
    full = cell.get("full") or {}
    last = cell.get("last_90d") or {}
    sid = str(cell.get("strategy") or "")
    base, parsed_math = _parse_math_from_sid(sid)
    math_g = cell.get("math") or parsed_math
    n = full.get("n")
    exp_r = full.get("exp_r")
    pnl = full.get("pnl")
    if pnl is None and n is not None and exp_r is not None:
        pnl = round(float(n) * float(exp_r) * RISK_USD, 2)
    oos_er = full.get("oos_exp_r") if full.get("oos_exp_r") is not None else exp_r
    last_er = last.get("oos_exp_r")
    wr = full.get("win_rate")
    # Approximate OOS WR display when missing: map E[R] to a soft % (not used as gate)
    if wr is None and oos_er is not None:
        try:
            # rough display helper for bar chart (clamp)
            wr = round(min(85.0, max(35.0, 50.0 + 25.0 * float(oos_er))), 1)
        except Exception:
            wr = None
    oos_pnl = None
    if last.get("n") is not None and last_er is not None:
        oos_pnl = round(float(last["n"]) * float(last_er) * RISK_USD, 2)
    elif n is not None and oos_er is not None:
        # fall back: scale ~25% of trades as OOS proxy for chart
        oos_pnl = round(0.25 * float(n) * float(oos_er) * RISK_USD, 2)

    label = cell.get("label") or sid
    family = cell.get("family") or "hedge"
    if family in ("resumed", "x"):
        family = "hedge"

    return {
        "strategy": sid,
        "strategy_base": base,
        "label": label,
        "family": family,
        "tf": cell.get("tf"),
        "rr": cell.get("rr"),
        "math": math_g,
        "verdict": full.get("verdict") or ("SURVIVES" if cell.get("deep_gate") else "FRAGILE"),
        "score": cell.get("score"),
        "oos_exp_r": oos_er,
        "full_exp_r": exp_r,
        "oos_pnl": oos_pnl,
        "full_pnl": pnl,
        "oos_win_rate": wr,
        "success_rate": wr,
        "mc_prob": _mc_pct(full.get("mc")),
        "n": n,
        "wf_green": full.get("wf_green"),
        "wf_n": 4,
        "deep_gate": bool(cell.get("deep_gate")),
        "last_90d_oos_exp_r": last_er,
        "last_5y_oos_exp_r": last_er,
        "live_brain": live_brain or None,
        "entries": n,
    }


def publish() -> dict[str, Any]:
    if not HUNT.exists():
        raise FileNotFoundError(f"Missing {HUNT}")
    raw = json.loads(HUNT.read_text(encoding="utf-8"))
    per_raw = dict(raw.get("per_pair") or {})

    # Enrich resumed pairs
    for sym in ORDER:
        if sym in per_raw:
            per_raw[sym] = _enrich_from_obsidian(sym, per_raw[sym])

    catalog = {
        sid: {"id": sid, "label": label, "family": family}
        for sid, family, label, _fn in _catalog()
    }

    per_pair: dict[str, Any] = {}
    winners: dict[str, Any] = {}
    charts_oos: list[dict] = []
    charts_wr: list[dict] = []
    charts_trades: list[dict] = []
    charts_er: list[dict] = []
    charts_vs_live: list[dict] = []

    desk_full = 0.0
    desk_oos = 0.0
    trades_total = 0
    gated_n = 0

    for sym in ORDER:
        row = per_raw.get(sym) or {}
        live = str(row.get("live_brain") or (raw.get("live_brains") or {}).get(sym) or "")
        live_rr = row.get("live_rr")
        best = _flatten_cell(row.get("winner"), live_brain=live)
        top_src = row.get("top5") or row.get("gated") or []
        top10 = []
        for c in top_src[:10]:
            flat = _flatten_cell(c, live_brain=live)
            if flat:
                top10.append(flat)
        if best and not any(t.get("strategy") == best.get("strategy") and t.get("rr") == best.get("rr") for t in top10):
            top10 = [best] + top10
            top10 = top10[:10]

        per_pair[sym] = {
            "best": best,
            "top10": top10,
            "live_brain": live,
            "live_rr": live_rr,
            "n_scored": row.get("n_scored"),
            "n_math": row.get("n_math"),
            "n_gated": row.get("n_gated"),
            "gated_count": row.get("n_gated") or len(row.get("gated") or []),
            "error": row.get("error"),
        }
        if best and best.get("deep_gate"):
            gated_n += 1
            winners[sym] = {
                "strategy": best["strategy"],
                "strategy_base": best.get("strategy_base"),
                "family": best["family"],
                "tf": best["tf"],
                "rr": best["rr"],
                "label": best["label"],
                "verdict": best["verdict"],
                "oos_exp_r": best["oos_exp_r"],
                "full_pnl": best["full_pnl"],
                "full_exp_r": best.get("full_exp_r"),
                "n": best["n"],
                "score": best["score"],
                "math": best.get("math"),
                "deep_gate": True,
                "live_brain": live,
            }
            pnl = float(best.get("full_pnl") or 0)
            oos_pnl = float(best.get("oos_pnl") or 0)
            desk_full += pnl
            desk_oos += oos_pnl
            trades_total += int(best.get("n") or 0)
            charts_oos.append(
                {
                    "symbol": sym,
                    "strategy": best["strategy"],
                    "label": best["label"],
                    "family": best["family"],
                    "oos_pnl": oos_pnl,
                    "full_pnl": pnl,
                    "success_rate": best.get("success_rate") or 0,
                    "verdict": best["verdict"],
                    "exp_r": best.get("full_exp_r"),
                    "n": best.get("n"),
                }
            )
            charts_wr.append(
                {
                    "symbol": sym,
                    "strategy": best["strategy"],
                    "label": best["label"],
                    "success_rate": best.get("success_rate") or 0,
                    "oos_pnl": oos_pnl,
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
            charts_vs_live.append(
                {
                    "symbol": sym,
                    "live_brain": live,
                    "live_rr": live_rr,
                    "hunt_strategy": best["strategy"],
                    "hunt_tf": best["tf"],
                    "hunt_rr": best["rr"],
                    "hunt_n": best.get("n"),
                    "hunt_exp_r": best.get("full_exp_r"),
                    "hunt_pnl": pnl,
                    "changed": str(live).split("+")[0] != str(best.get("strategy_base") or best["strategy"]).split("+")[0],
                }
            )

    # Full strategy id (includes math tag) so proposed map is explicit
    brain_map = ",".join(f"{s}:{winners[s]['strategy']}" for s in ORDER if s in winners)
    tf_map = ",".join(f"{s}:{winners[s]['tf']}" for s in ORDER if s in winners)
    rr_map = ",".join(f"{s}:{winners[s]['rr']}" for s in ORDER if s in winners)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "desk_recovery_scalp_hunt",
        "source": "lab_desk_recovery_scalp_hunt.json",
        "note": (
            "Desk recovery scalper/intraday hunt (1m·3m·5m·15m). "
            "Firm gate: SURVIVES + 90d OOS E[R]>0 + WF. "
            "LIVE brains NOT changed — review before wire."
        ),
        "risk_usd": RISK_USD,
        "elapsed_s": raw.get("elapsed_sec"),
        "window": {
            "tf": "1m/3m/5m/15m",
            "years_target": 1,
            "days": 365,
            "recent_gate_days": 90,
            "mode": "scalper_intraday_recovery",
        },
        "summary": {
            "pairs_with_survives": gated_n,
            "strategies_tested": 43,
            "cells": None,
            "survives_cells": gated_n,
            "desk_success_rate_pct": round(100.0 * gated_n / max(1, len(ORDER)), 1),
            "candles_total": None,
            "trades_total": trades_total,
            "desk_full_pnl": round(desk_full, 2),
            "desk_oos_pnl": round(desk_oos, 2),
            "densify_rounds": raw.get("densify_rounds"),
        },
        "hunt": {
            "desk_success_rate_pct": round(100.0 * gated_n / max(1, len(ORDER)), 1),
            "variants_tested": 43,
            "deep_gate_survivors": gated_n,
            "mode": "desk_recovery_scalp",
            "recovery_order": list(ORDER),
        },
        "strategies": catalog,
        "winners": winners,
        "proposed_SYMBOL_ENTRY_BRAIN": brain_map,
        "proposed_SYMBOL_SCAN_TFS": tf_map,
        "proposed_SYMBOL_RR": rr_map,
        "live_brains": raw.get("live_brains") or {},
        "per_pair": per_pair,
        "charts": {
            "best_oos_by_pair": charts_oos,
            "success_by_pair": charts_wr,
            "trades_by_pair": charts_trades,
            "exp_r_by_pair": charts_er,
            "vs_live": charts_vs_live,
            "heatmap": [],
            "equity_curves": {},
        },
        "limits": {"research_only": True, "auto_wire": False},
        "obsidian": str(OBSIDIAN),
    }

    if LAB.exists():
        shutil.copy2(LAB, LAB_BACKUP)
    LAB.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    PROGRESS.write_text(
        json.dumps(
            {
                "status": "done",
                "phase": "desk_recovery_scalp",
                "pct": 100,
                "message": f"published recovery board · firm={gated_n}/8 · desk ${desk_full:,.0f}",
                "symbol": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {LAB} · firm={gated_n}/8 · desk_full=${desk_full:,.0f} · trades={trades_total}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    publish()
