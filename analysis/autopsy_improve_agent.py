#!/usr/bin/env python3
"""STEP 3 — Agentic autopsy improve loop.

Takes autopsy tips → paper A/B (OOS / WF / MC) → ranked proposals.
Soft-apply then re-runs OOS/WF/MC vs the previous setup:
  • if better → promote knobs into the live brain (autopsy_brain_promoted)
  • if not   → rollback soft overlays to the previous setup

Never rewrites .env / SYMBOL_ENTRY_BRAIN string map (brain type stays);
EMA pull/disp knobs are what get rewritten on promote.

Usage (Docker):
  docker compose exec -T -e PYTHONPATH=/app backend \\
    python /app/analysis/autopsy_improve_agent.py --days 14
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PROPOSALS_PATH = Path(__file__).resolve().parents[1] / "data" / "autopsy_proposals.json"
OUT_MD = Path("/tmp/autopsy_improve_report.md")

# Tip tag → concrete experiments the agent can paper-test
TIP_EXPERIMENTS: dict[str, list[dict[str, Any]]] = {
    "soft_mc_entry": [
        {
            "id": "raise_mc_soft",
            "label": "Raise MC soft floor (+2)",
            "kind": "soft_overlay",
            "patch": {"mc_floor_delta": 2.0},
            # Paper proxy: fewer / cleaner entries via tighter pull
            "brain_kw": {"pull_atr": 0.70, "disp_atr": 0.70},
        },
        {
            "id": "raise_mc_soft_hard",
            "label": "Raise MC soft floor (+3.5)",
            "kind": "soft_overlay",
            "patch": {"mc_floor_delta": 3.5},
            "brain_kw": {"pull_atr": 0.65, "disp_atr": 0.75},
        },
    ],
    "stretched_rr_loss": [
        {
            "id": "prefer_2r",
            "label": "Prefer forced ~2R targets",
            "kind": "soft_overlay",
            "patch": {"rr_prefer": 2.0},
            "brain_kw": {},  # already 2R in survive BT — score tip frequency
            "score_only": True,
        },
    ],
    "stop_loss_hit": [
        {
            "id": "tighter_entry",
            "label": "Tighter EMA pull / displacement",
            "kind": "brain_param",
            "patch": {},
            "brain_kw": {"pull_atr": 0.70, "disp_atr": 0.70},
            "soft_follow": {"mc_floor_delta": 1.0, "aggression": -0.04},
        },
        {
            "id": "wider_sl",
            "label": "Slightly wider stops (less noise stop-outs)",
            "kind": "brain_param",
            "patch": {},
            "brain_kw": {"sl_widen": 1.15},
            "soft_follow": {"risk_scale": 0.92},
        },
    ],
    "off_prime_hour": [
        {
            "id": "cut_size_off_prime",
            "label": "Cut risk off-prime (soft)",
            "kind": "soft_overlay",
            "patch": {"risk_scale": 0.85, "aggression": -0.06},
            "score_only": True,
        },
    ],
    "fast_tf_loss": [
        {
            "id": "raise_gate_fast_tf",
            "label": "Stricter gate after fast-TF losses",
            "kind": "soft_overlay",
            "patch": {"mc_floor_delta": 2.0, "risk_scale": 0.88},
            "score_only": True,
        },
    ],
    "large_htf_loss": [
        {
            "id": "cap_htf_risk",
            "label": "Cap risk after large HTF losses",
            "kind": "soft_overlay",
            "patch": {"risk_scale": 0.82, "mc_floor_delta": 1.5},
            "score_only": True,
        },
    ],
    "news_against": [
        {
            "id": "news_gate_soft",
            "label": "Harder news-alignment via higher MC + lower aggression",
            "kind": "soft_overlay",
            "patch": {"mc_floor_delta": 2.5, "aggression": -0.08},
            "score_only": True,
        },
    ],
    "chop_state": [
        {
            "id": "chop_tighten",
            "label": "Tighten entries in chop",
            "kind": "brain_param",
            "patch": {},
            "brain_kw": {"pull_atr": 0.65, "disp_atr": 0.75},
            "soft_follow": {"mc_floor_delta": 1.5, "aggression": -0.05},
        },
    ],
    "smc_weak": [
        {
            "id": "smc_raise_mc",
            "label": "Raise MC when SMC weak",
            "kind": "soft_overlay",
            "patch": {"mc_floor_delta": 2.0},
            "score_only": True,
        },
    ],
}


def _load_proposals() -> dict[str, Any]:
    if not PROPOSALS_PATH.exists():
        return {"updated_at": None, "proposals": [], "runs": []}
    try:
        return json.loads(PROPOSALS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"updated_at": None, "proposals": [], "runs": []}


def save_proposals(payload: dict[str, Any]) -> None:
    PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROPOSALS_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _tip_tags_from_pair(pair: dict) -> list[str]:
    tags = []
    for iss in pair.get("issues") or []:
        t = iss.get("tag")
        if t and t not in ("ok", "no_closes", "clean_win_profile", "take_profit_hit"):
            tags.append(t)
    tip = str(pair.get("tip") or "")
    # Infer from tip text if tags thin
    low = tip.lower()
    if "mc soft" in low or "monte carlo" in low or "raise mc" in low:
        tags.append("soft_mc_entry")
    if "1:2" in low or "stretched" in low or "rr" in low:
        tags.append("stretched_rr_loss")
    if "stop" in low and "placement" in low:
        tags.append("stop_loss_hit")
    if "prime" in low or "session" in low:
        tags.append("off_prime_hour")
    # de-dupe preserve order
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _baseline_kwargs(brain: str) -> dict[str, Any]:
    if brain == "ema_20_50_tight":
        return {"pull_atr": 0.75, "disp_atr": 0.70}
    if brain.startswith("ema_"):
        return {"pull_atr": 1.0, "disp_atr": 0.55}
    return {}


def _make_gen(brain: str, df, extra: dict[str, Any]) -> Callable:
    from analysis._edge_survive_bt import (
        run_breakout_trades,
        run_ema_trades,
        run_fib_trades,
        run_sma_trades,
    )

    kw = dict(extra)

    def gen(start_i=0, end_i=None, **more):
        m = {**kw, **more}
        if brain == "fib_liq":
            return run_fib_trades(df, start_i=start_i, end_i=end_i, **m)
        if brain == "sma_20_50_cross":
            return run_sma_trades(df, start_i=start_i, end_i=end_i, **m)
        if brain == "breakout_20":
            return run_breakout_trades(df, start_i=start_i, end_i=end_i, **m)
        pull = float(m.pop("pull_atr", _baseline_kwargs(brain).get("pull_atr", 1.0)))
        disp = float(m.pop("disp_atr", _baseline_kwargs(brain).get("disp_atr", 0.55)))
        return run_ema_trades(
            df, start_i=start_i, end_i=end_i, pull_atr=pull, disp_atr=disp, **m
        )

    return gen


def _quick_battery(df, gen: Callable, *, mc_paths: int = 400) -> dict[str, Any]:
    """Faster than full edge_survive — full / OOS / light WF / MC."""
    from analysis._edge_survive_bt import _stats, monte_carlo, verdict

    n = len(df)
    full_tr = gen()
    full = _stats(full_tr)
    split = int(n * 0.70)
    oos_tr = gen(start_i=split)
    oos = _stats(oos_tr)
    wf = []
    for k in range(3):
        test_start = int(n * (0.45 + 0.15 * k))
        test_end = int(n * (0.60 + 0.15 * k)) - 1
        if test_end <= test_start or test_start < 80:
            continue
        test_tr = gen(start_i=test_start, end_i=test_end)
        wf.append({"fold": k + 1, "oos": _stats(test_tr)})
    # Light torture sample
    torture = {
        "spread_0.15R": _stats(gen(spread_r=0.15)),
        "late_1bar": _stats(gen(late_bars=1)),
    }
    mc = monte_carlo(full_tr, paths=mc_paths)
    v = verdict(full, oos, wf, torture, mc)
    return {
        "full": full,
        "oos": oos,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "verdict": v,
        "n_trades": int(full.get("n") or 0),
    }


def _improves(base: dict, alt: dict) -> tuple[bool, str]:
    """Gate: alt must not die, and beat baseline on OOS E[R] (or tie + better MC)."""
    bv, av = base.get("verdict"), alt.get("verdict")
    if av == "DEAD" and bv != "DEAD":
        return False, "alt_dead"
    b_oos = float((base.get("oos") or {}).get("exp_r") or 0)
    a_oos = float((alt.get("oos") or {}).get("exp_r") or 0)
    b_full = float((base.get("full") or {}).get("exp_r") or 0)
    a_full = float((alt.get("full") or {}).get("exp_r") or 0)
    b_mc = float((base.get("monte_carlo") or {}).get("prob_profit") or 50)
    a_mc = float((alt.get("monte_carlo") or {}).get("prob_profit") or 50)

    if a_oos > b_oos + 0.02 and a_full > -0.05:
        return True, f"oos_er {b_oos:.3f}→{a_oos:.3f}"
    if a_oos >= b_oos - 0.01 and a_mc >= b_mc + 3 and a_full >= b_full:
        return True, f"mc {b_mc:.0f}%→{a_mc:.0f}%"
    if bv in ("FRAGILE", "DEAD") and av == "SURVIVES" and a_oos > 0:
        return True, f"verdict {bv}→{av}"
    return False, "no_edge_gain"


def _score_only_proposal(
    pair: dict, tag: str, exp: dict
) -> Optional[dict[str, Any]]:
    """For tips that don't change brain params — propose soft knobs if pair is weak."""
    eff = int(pair.get("efficiency") or 5)
    losses = int(pair.get("losses") or 0)
    wins = int(pair.get("wins") or 0)
    if losses < 2 and eff >= 6:
        return None
    if eff >= 8:
        return None
    conf = "medium"
    if eff <= 4 and losses > wins:
        conf = "high"
    elif eff <= 5:
        conf = "medium"
    else:
        conf = "low"
    return {
        "id": str(uuid.uuid4())[:8],
        "symbol": pair["symbol"],
        "tag": tag,
        "experiment_id": exp["id"],
        "label": exp["label"],
        "kind": "soft_overlay",
        "patch": dict(exp.get("patch") or {}),
        "status": "proposed",
        "apply_safe": True,
        "confidence": conf,
        "reason": (
            f"Autopsy tip '{tag}' on {pair['symbol']} "
            f"(eff {eff}/10, {wins}W/{losses}L) — soft gate only, no brain rewrite."
        ),
        "ab": None,
        "gate": "score_only_soft",
    }


def run_improve(
    *,
    days: int = 14,
    symbols: Optional[list[str]] = None,
    max_pairs: int = 6,
    max_exps_per_pair: int = 2,
    mc_paths: int = 400,
    lookback_days: int = 120,
) -> dict[str, Any]:
    """Main agent loop. Returns proposals payload."""
    from config import INSTRUMENT_UNIVERSE
    from analysis.trade_autopsy_agent import build_dashboard_payload
    from analysis._edge_survive_bt import BRAINS, _fetch_1h, _prep

    # Prefer live DB via autopsy agent fetch when available
    try:
        from analysis.trade_autopsy_agent import _fetch_closes

        rows = _fetch_closes(days, 400)
    except Exception as exc:
        rows = []
        fetch_err = str(exc)
    else:
        fetch_err = None

    if rows:
        board = build_dashboard_payload(
            rows, days=days, prefer_symbols=tuple(INSTRUMENT_UNIVERSE)
        )
    else:
        board = {
            "pairs": [
                {
                    "symbol": s,
                    "n": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0,
                    "efficiency": 5,
                    "issues": [],
                    "tip": None,
                    "no_data": True,
                }
                for s in INSTRUMENT_UNIVERSE
            ],
            "n": 0,
            "days": days,
            "fetch_error": fetch_err,
        }

    from config import is_autopsy_excluded

    pairs = list(board.get("pairs") or [])
    if symbols:
        allow = {s.upper() for s in symbols}
        pairs = [p for p in pairs if p.get("symbol", "").upper() in allow]
    # Fresh PA brains (GOLD/ETH/BTC) — no self-heal / improve proposals
    pairs = [
        p
        for p in pairs
        if not is_autopsy_excluded(str(p.get("symbol") or ""))
    ]
    # Prioritize weak pairs with tips/issues
    pairs.sort(
        key=lambda p: (
            0 if p.get("tip") or (p.get("issues") and not p.get("no_data")) else 1,
            int(p.get("efficiency") or 5),
            -int(p.get("losses") or 0),
        )
    )
    pairs = pairs[:max_pairs]

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=lookback_days)).isoformat()
    end_s = end.isoformat()

    proposals: list[dict[str, Any]] = []
    run_log: list[dict[str, Any]] = []

    for pair in pairs:
        sym = str(pair.get("symbol") or "").upper()
        if not sym or pair.get("no_data"):
            continue
        tags = _tip_tags_from_pair(pair)
        if not tags:
            # Still try stop_loss if losses dominate
            if int(pair.get("losses") or 0) > int(pair.get("wins") or 0):
                tags = ["stop_loss_hit"]
            else:
                continue

        brain = BRAINS.get(sym, "ema_20_50_pull")
        try:
            df = _prep(_fetch_1h(sym, start, end_s))
        except Exception as exc:
            run_log.append({"symbol": sym, "error": f"data:{exc}"})
            # Fall back to score-only soft proposals
            for tag in tags[:2]:
                for exp in (TIP_EXPERIMENTS.get(tag) or [])[:1]:
                    if exp.get("kind") == "soft_overlay" or exp.get("score_only"):
                        p = _score_only_proposal(pair, tag, exp)
                        if p:
                            proposals.append(p)
            continue

        from analysis.autopsy_brain import live_ema_knobs

        base_kw = live_ema_knobs(sym, brain) if str(brain).startswith("ema_") else _baseline_kwargs(brain)
        try:
            base = _quick_battery(df, _make_gen(brain, df, base_kw), mc_paths=mc_paths)
        except Exception as exc:
            run_log.append({"symbol": sym, "error": f"baseline:{exc}"})
            continue

        tried = 0
        for tag in tags:
            if tried >= max_exps_per_pair:
                break
            for exp in TIP_EXPERIMENTS.get(tag) or []:
                if tried >= max_exps_per_pair:
                    break
                tried += 1
                if exp.get("score_only") or (
                    exp.get("kind") == "soft_overlay" and not exp.get("brain_kw")
                ):
                    p = _score_only_proposal(pair, tag, exp)
                    if p:
                        proposals.append(p)
                    continue

                alt_kw = {**base_kw, **(exp.get("brain_kw") or {})}
                # Non-EMA brains: skip pull/disp-only experiments
                if brain in ("fib_liq", "sma_20_50_cross", "breakout_20"):
                    if set(alt_kw.keys()) <= {"pull_atr", "disp_atr"} and not any(
                        k in (exp.get("brain_kw") or {}) for k in ("sl_widen", "lookback", "fibs")
                    ):
                        # Map to soft overlay follow if present
                        if exp.get("soft_follow") or exp.get("patch"):
                            soft_exp = {
                                **exp,
                                "kind": "soft_overlay",
                                "patch": {
                                    **(exp.get("patch") or {}),
                                    **(exp.get("soft_follow") or {}),
                                },
                                "score_only": True,
                            }
                            p = _score_only_proposal(pair, tag, soft_exp)
                            if p:
                                proposals.append(p)
                        continue

                try:
                    alt = _quick_battery(
                        df, _make_gen(brain, df, alt_kw), mc_paths=mc_paths
                    )
                except Exception as exc:
                    run_log.append(
                        {"symbol": sym, "exp": exp["id"], "error": str(exc)}
                    )
                    continue

                ok, why = _improves(base, alt)
                soft_patch = dict(exp.get("patch") or {})
                soft_patch.update(exp.get("soft_follow") or {})
                apply_safe = bool(soft_patch) and ok
                status = "proposed" if ok else "rejected"
                proposals.append(
                    {
                        "id": str(uuid.uuid4())[:8],
                        "symbol": sym,
                        "tag": tag,
                        "experiment_id": exp["id"],
                        "label": exp["label"],
                        "kind": exp.get("kind") or "brain_param",
                        "brain": brain,
                        "patch": soft_patch,
                        "brain_kw": alt_kw,
                        "status": status,
                        "apply_safe": apply_safe,
                        "confidence": "high" if ok and alt.get("verdict") == "SURVIVES" else (
                            "medium" if ok else "low"
                        ),
                        "reason": why if ok else f"rejected:{why}",
                        "gate": "oos_wf_mc",
                        "ab": {
                            "baseline": {
                                "verdict": base.get("verdict"),
                                "full_er": (base.get("full") or {}).get("exp_r"),
                                "oos_er": (base.get("oos") or {}).get("exp_r"),
                                "mc_profit": (base.get("monte_carlo") or {}).get(
                                    "prob_profit"
                                ),
                                "n": (base.get("full") or {}).get("n"),
                            },
                            "alt": {
                                "verdict": alt.get("verdict"),
                                "full_er": (alt.get("full") or {}).get("exp_r"),
                                "oos_er": (alt.get("oos") or {}).get("exp_r"),
                                "mc_profit": (alt.get("monte_carlo") or {}).get(
                                    "prob_profit"
                                ),
                                "n": (alt.get("full") or {}).get("n"),
                            },
                        },
                    }
                )
                run_log.append(
                    {
                        "symbol": sym,
                        "exp": exp["id"],
                        "ok": ok,
                        "why": why,
                        "base_v": base.get("verdict"),
                        "alt_v": alt.get("verdict"),
                    }
                )

    # Prefer proposed + apply_safe first
    proposals.sort(
        key=lambda p: (
            0 if p.get("status") == "proposed" else 1,
            0 if p.get("apply_safe") else 1,
            {"high": 0, "medium": 1, "low": 2}.get(p.get("confidence"), 3),
            p.get("symbol") or "",
        )
    )

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "lookback_days": lookback_days,
        "board_n": board.get("n"),
        "desk_efficiency": board.get("desk_efficiency"),
        "proposals": proposals,
        "runs": run_log,
        "mode": "agentic_research",
        "note": (
            "Soft-apply runs OOS/WF/MC vs previous setup. "
            "Better → brain knobs promoted; worse → soft rolled back."
        ),
    }
    save_proposals(payload)
    _write_md(payload)
    return payload


def _write_md(payload: dict) -> None:
    lines = [
        "# Autopsy improve agent",
        "",
        f"Generated: {payload.get('updated_at')}",
        f"Window: {payload.get('days')}d closes · {payload.get('lookback_days')}d paper",
        "",
        "## Proposals",
        "",
    ]
    props = payload.get("proposals") or []
    if not props:
        lines.append("_No proposals this run._")
    for p in props:
        mark = "✅" if p.get("status") == "proposed" else "❌"
        soft = " · soft-apply OK" if p.get("apply_safe") else ""
        lines.append(
            f"- {mark} **{p.get('symbol')}** · {p.get('label')} "
            f"({p.get('confidence')}){soft}"
        )
        lines.append(f"  - {p.get('reason')}")
        ab = p.get("ab") or {}
        if ab:
            b, a = ab.get("baseline") or {}, ab.get("alt") or {}
            lines.append(
                f"  - A/B OOS E[R] {b.get('oos_er')}→{a.get('oos_er')} · "
                f"verdict {b.get('verdict')}→{a.get('verdict')}"
            )
        lines.append("")
    try:
        OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


def _brain_kw_meaningful(prev: dict, cand: dict) -> bool:
    """True if candidate knobs differ enough to change paper trades."""
    keys = ("pull_atr", "disp_atr", "sl_widen", "lookback", "fibs")
    for k in keys:
        if k not in cand and k not in prev:
            continue
        a, b = prev.get(k), cand.get(k)
        if a is None and b is None:
            continue
        if a is None or b is None:
            return True
        try:
            if abs(float(a) - float(b)) > 1e-6:
                return True
        except (TypeError, ValueError):
            if a != b:
                return True
    return False


def _summarize_battery(bat: dict) -> dict[str, Any]:
    return {
        "verdict": bat.get("verdict"),
        "full_er": (bat.get("full") or {}).get("exp_r"),
        "oos_er": (bat.get("oos") or {}).get("exp_r"),
        "mc_profit": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "mc_p5": (bat.get("monte_carlo") or {}).get("p5_R"),
        "n": (bat.get("full") or {}).get("n"),
        "wf_green": sum(
            1
            for f in (bat.get("walk_forward") or [])
            if (f.get("oos") or {}).get("exp_r", 0) > 0
        ),
        "wf_folds": len(bat.get("walk_forward") or []),
    }


def apply_proposal(
    proposal_id: str,
    *,
    lookback_days: int = 100,
    mc_paths: int = 400,
) -> dict[str, Any]:
    """Soft-apply → OOS/WF/MC vs previous → promote brain or rollback soft."""
    from analysis.autopsy_soft import apply_soft, restore as restore_soft, snapshot as snap_soft
    from analysis.autopsy_brain import (
        defaults_for_brain,
        get as brain_get,
        live_ema_knobs,
        promote as promote_brain,
        snapshot as snap_brain,
    )
    from analysis._edge_survive_bt import BRAINS, _fetch_1h, _prep

    st = _load_proposals()
    props = st.get("proposals") or []
    hit = None
    for p in props:
        if str(p.get("id")) == str(proposal_id):
            hit = p
            break
    if not hit:
        raise ValueError(f"proposal {proposal_id} not found")
    if hit.get("status") not in ("proposed", "applied", "rolled_back"):
        raise ValueError(f"proposal status is {hit.get('status')} — not applyable")
    if not hit.get("apply_safe") or not hit.get("patch"):
        raise ValueError(
            "Not soft-applyable — missing soft patch"
        )

    from config import is_autopsy_excluded

    sym = str(hit["symbol"]).upper()
    if is_autopsy_excluded(sym):
        raise ValueError(
            f"{sym} is autopsy-excluded (fresh PA brain) — self-heal apply blocked"
        )
    brain = str(hit.get("brain") or BRAINS.get(sym, "ema_20_50_pull"))
    soft_before = snap_soft(sym)
    brain_before = snap_brain(sym)

    # 1) Soft-apply trial
    overlay = apply_soft(
        sym,
        hit["patch"],
        proposal_id=hit["id"],
        note=hit.get("label"),
    )
    hit["soft_applied_at"] = datetime.now(timezone.utc).isoformat()

    # 2) Build previous vs candidate knobs
    prev_kw = live_ema_knobs(sym, brain) if brain.startswith("ema_") else defaults_for_brain(brain)
    # If no promote yet, live_ema_knobs returns defaults — also merge any existing promote
    promo = brain_get(sym)
    if promo.get("pull_atr") is not None:
        prev_kw["pull_atr"] = float(promo["pull_atr"])
    if promo.get("disp_atr") is not None:
        prev_kw["disp_atr"] = float(promo["disp_atr"])
    if promo.get("sl_widen") is not None:
        prev_kw["sl_widen"] = float(promo["sl_widen"])

    cand_kw = dict(prev_kw)
    for k, v in (hit.get("brain_kw") or {}).items():
        cand_kw[k] = v

    validation: dict[str, Any] = {
        "previous_knobs": dict(prev_kw),
        "candidate_knobs": dict(cand_kw),
        "brain": brain,
    }

    # 3) Paper OOS / WF / MC compare (when knobs differ)
    decision = "soft_retained"
    brain_row = None
    compare_ok = False
    why = "no_brain_kw_delta"

    if _brain_kw_meaningful(prev_kw, cand_kw) and brain.startswith("ema_"):
        end = datetime.now(timezone.utc).date() - timedelta(days=1)
        start = (end - timedelta(days=lookback_days)).isoformat()
        end_s = end.isoformat()
        try:
            df = _prep(_fetch_1h(sym, start, end_s))
            prev_bat = _quick_battery(
                df, _make_gen(brain, df, prev_kw), mc_paths=mc_paths
            )
            cand_bat = _quick_battery(
                df, _make_gen(brain, df, cand_kw), mc_paths=mc_paths
            )
            compare_ok, why = _improves(prev_bat, cand_bat)
            validation["previous"] = _summarize_battery(prev_bat)
            validation["candidate"] = _summarize_battery(cand_bat)
            validation["compare_reason"] = why
        except Exception as exc:
            # Data failure → rollback soft (safe default)
            restore_soft(sym, soft_before)
            hit["status"] = "rolled_back"
            hit["rollback_at"] = datetime.now(timezone.utc).isoformat()
            hit["validation"] = {
                **validation,
                "decision": "rolled_back",
                "error": str(exc),
            }
            save_proposals(st)
            return {
                "ok": False,
                "decision": "rolled_back",
                "reason": f"validation_data_error:{exc}",
                "proposal": hit,
                "overlay": soft_before,
            }

        if compare_ok:
            brain_row = promote_brain(
                sym,
                cand_kw,
                proposal_id=hit["id"],
                note=hit.get("label"),
                previous=brain_before,
            )
            decision = "promoted"
            hit["status"] = "promoted"
            hit["promoted_at"] = datetime.now(timezone.utc).isoformat()
        else:
            restore_soft(sym, soft_before)
            # brain unchanged (we never wrote until promote)
            decision = "rolled_back"
            hit["status"] = "rolled_back"
            hit["rollback_at"] = datetime.now(timezone.utc).isoformat()
    else:
        # Soft-only (no EMA knob delta) — keep soft; no brain rewrite
        # Re-validate using stored A/B if present; else retain soft as gate tweak
        ab = hit.get("ab") or {}
        if ab.get("baseline") and ab.get("alt"):
            # Reconstruct minimal dicts for _improves
            def _from_sum(s: dict) -> dict:
                return {
                    "verdict": s.get("verdict"),
                    "full": {"exp_r": s.get("full_er"), "n": s.get("n")},
                    "oos": {"exp_r": s.get("oos_er")},
                    "monte_carlo": {"prob_profit": s.get("mc_profit")},
                    "walk_forward": [],
                    "torture": {"x": {"exp_r": 0}},
                }

            compare_ok, why = _improves(_from_sum(ab["baseline"]), _from_sum(ab["alt"]))
            validation["previous"] = ab["baseline"]
            validation["candidate"] = ab["alt"]
            validation["compare_reason"] = why
            if compare_ok:
                decision = "soft_retained"
                hit["status"] = "promoted_soft"
            else:
                restore_soft(sym, soft_before)
                decision = "rolled_back"
                hit["status"] = "rolled_back"
                hit["rollback_at"] = datetime.now(timezone.utc).isoformat()
        else:
            decision = "soft_retained"
            hit["status"] = "promoted_soft"
            why = "soft_only_no_brain_kw"

    validation["decision"] = decision
    validation["compare_ok"] = compare_ok
    validation["compare_reason"] = why
    hit["validation"] = validation
    hit["applied_at"] = datetime.now(timezone.utc).isoformat()
    save_proposals(st)

    return {
        "ok": decision in ("promoted", "soft_retained", "promoted_soft"),
        "decision": decision,
        "reason": why,
        "proposal": hit,
        "overlay": overlay if decision != "rolled_back" else soft_before,
        "brain": brain_row,
        "validation": validation,
    }


def get_proposals() -> dict[str, Any]:
    """Return proposals, hiding autopsy-excluded symbols (wired PA pairs)."""
    from config import AUTOPSY_EXCLUDE_SYMBOLS, is_autopsy_excluded

    payload = _load_proposals()
    props = list(payload.get("proposals") or [])
    visible = [
        p for p in props if not is_autopsy_excluded(str(p.get("symbol") or ""))
    ]
    out = dict(payload)
    out["proposals"] = visible
    out["excluded_symbols"] = sorted(AUTOPSY_EXCLUDE_SYMBOLS)
    if len(visible) != len(props):
        out["note"] = (
            (out.get("note") or "")
            + f" · Hidden proposals for excluded: {', '.join(sorted(AUTOPSY_EXCLUDE_SYMBOLS))}"
        ).strip(" ·")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Autopsy improve agent")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--max-pairs", type=int, default=6)
    ap.add_argument("--apply", type=str, default="", help="proposal id to soft-apply + validate")
    args = ap.parse_args()
    if args.apply:
        out = apply_proposal(args.apply)
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("ok") or out.get("decision") == "rolled_back" else 1
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    payload = run_improve(days=args.days, symbols=syms, max_pairs=args.max_pairs)
    n_ok = sum(1 for p in payload.get("proposals") or [] if p.get("status") == "proposed")
    print(
        f"Improve agent done: {n_ok} proposed / "
        f"{len(payload.get('proposals') or [])} total → {PROPOSALS_PATH}",
        flush=True,
    )
    for p in (payload.get("proposals") or [])[:12]:
        print(
            f"  [{p.get('status')}] {p.get('symbol')} {p.get('label')} "
            f"· {p.get('reason')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
