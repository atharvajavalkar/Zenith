"""Live deep-hunt status for the Zenith Nebula view."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends

from app.core.security import get_current_user

router = APIRouter(prefix="/api/hunt-live", tags=["hunt-live"])

BOOK = ("ETHUSD", "BTCUSD", "GOLD", "NAS100", "USDJPY", "AUDUSD", "USDCHF", "UKOIL")
TFS = ("5m", "15m", "3m", "1m")
CELLS_EST = 52464

# Search-stack layers shown in the Nebula (what the hunt actually sweeps)
SEARCH_LAYERS = [
    {
        "id": "pairs",
        "label": "Instrument book",
        "detail": "8 XM desk pairs with Dukascopy hist + session-wise spreads",
    },
    {
        "id": "timeframes",
        "label": "Timeframes",
        "detail": "5m / 15m / 3m intraday full catalog · 1m scalp lean catalog",
    },
    {
        "id": "brains",
        "label": "Adaptive brains",
        "detail": "Static / gated / switch baselines + dynamic session·circadian·entropy·SNR·volmom·Fourier·OU·vol-clock·weekday·edge-gate",
    },
    {
        "id": "stops",
        "label": "Stop lattice",
        "detail": "SL × ATR grid per TF (live-safe floors + XM spread gate)",
    },
    {
        "id": "exits",
        "label": "Exit lattice",
        "detail": "RR ∈ {2,3,4} reused across entry gaps",
    },
    {
        "id": "gaps",
        "label": "Entry gaps",
        "detail": "Min bars between entries — anti-overtrade",
    },
    {
        "id": "windows",
        "label": "Horizon windows",
        "detail": "7d · 15d · 30d · 90d · 180d · 1y · 3y · full — scored with XM costs",
    },
    {
        "id": "gates",
        "label": "Survival gates",
        "detail": "PASS / SOFT / FAIL via exp R, OOS, walk-forward greens",
    },
]


def _data_roots() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        Path("/app/data"),
        here.parents[3] / "data",
        here.parents[2] / "data",
    ]


def _find(name: str) -> Optional[Path]:
    for root in _data_roots():
        p = root / name
        if p.exists():
            return p
    return None


def _read_json(name: str) -> Optional[dict[str, Any]]:
    p = _find(name)
    if not p:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None




def _family_of(mode_id: str) -> str:
    m = str(mode_id or "")
    if m in ("static", "gated", "switch"):
        return "baseline"
    if m.startswith("dyn_") or m.startswith("agree_") or m.startswith("triple_") or m.startswith("edge_"):
        return "adaptive"
    if m.startswith("sm_") or m.startswith("off_sm_") or m.startswith("session_mix"):
        return "session_mix"
    if m.startswith("vm_") or m.startswith("off_vm_") or m.startswith("volmom"):
        return "volmom"
    if m.startswith("ema_") or m.startswith("bb_") or m.startswith("rsi_") or m.startswith("don_") or m.startswith("macd_") or m.startswith("st_") or m.startswith("kelt_") or m.startswith("zmr_") or m.startswith("asia_") or m.startswith("cusum_"):
        return "classic_ta"
    if m.startswith("gold_") or m.startswith("btc_") or m.startswith("oil_") or m.startswith("nas_"):
        return "pair_econ"
    if m.startswith("aud_"):
        return "aud_econ"
    if m.startswith("eth_"):
        return "eth_econ"
    if m.startswith("cir_") or m.startswith("ent_") or m.startswith("snr_") or m.startswith("ou_") or m.startswith("ft_"):
        return "math_signal"
    if m.startswith("dyn_"):
        return "adaptive"
    if m.startswith("math_") or m.startswith("zenith"):
        return "zenith_math"
    if m.startswith("cog_bayes") or m.startswith("cog_hmm") or m.startswith("cog_info") or m.startswith("cog_math") or m.startswith("cog_"):
        return "cognitive"
    if m.startswith("omni"):
        return "omni"
    if m.startswith("agree"):
        return "agree"
    if m.startswith("hurst"):
        return "hurst"
    if m.startswith("offflat") or m.startswith("off_flat"):
        return "off_flat"
    if m.startswith("lonny") or m.startswith("asia_fade"):
        return "liquid_hours"
    if m.startswith("tod_blend"):
        return "circadian"
    if m.startswith("session_mix"):
        return "session_mix"
    if m.startswith("circadian"):
        return "circadian"
    if m.startswith("entropy"):
        return "entropy"
    if m.startswith("snr_"):
        return "snr"
    if m.startswith("volmom"):
        return "volmom"
    if m.startswith("fourier"):
        return "fourier"
    if m.startswith("ou_fade"):
        return "ou_fade"
    if m.startswith("vol_clock"):
        return "vol_clock"
    if m.startswith("weekday"):
        return "weekday"
    if m.startswith("session_book"):
        return "session_book"
    if "edgegate" in m or m.endswith("_edge"):
        return "edgegate"
    if m.endswith("_liq") or "_liq" in m:
        return "liquid_hours"
    return "other"


def _is_live_running(progress: dict, max_age_hours: float = 36.0) -> bool:
    """A progress file left at 'running' after a dead process is not live."""
    if str(progress.get("status") or "") != "running":
        return False
    raw = str(progress.get("updated_at") or "")
    if not raw:
        return False
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_s = (datetime.now(timezone.utc) - ts).total_seconds()
    return age_s < max_age_hours * 3600.0


def _active_hunt_files() -> tuple[str, str, str, str]:
    """Prefer the solution hunt, then mega / invent / cog / REST-5 when actually running."""
    sol_p = _read_json("lab_solution_xm_hunt.progress.json") or {}
    mega_p = _read_json("lab_mega_desk_invent_hunt.progress.json") or {}
    invent_p = _read_json("lab_adaptive_xm_aud_eth_invent_hunt.progress.json") or {}
    cog_p = _read_json("lab_adaptive_xm_rest5_cog_hunt.progress.json") or {}
    rest_p = _read_json("lab_adaptive_xm_rest5_hunt.progress.json") or {}
    deep_p = _read_json("lab_adaptive_xm_deep_hunt.progress.json") or {}
    invent_st = str(invent_p.get("status") or "")
    cog_st = str(cog_p.get("status") or "")
    rest_st = str(rest_p.get("status") or "")
    deep_st = str(deep_p.get("status") or "")
    if _is_live_running(sol_p):
        return (
            "lab_solution_xm_hunt",
            "lab_solution_xm_hunt.progress.json",
            "lab_solution_xm_hunt.partial.json",
            "lab_solution_xm_hunt.json",
        )
    if _is_live_running(mega_p):
        return (
            "lab_mega_desk_invent_hunt",
            "lab_mega_desk_invent_hunt.progress.json",
            "lab_mega_desk_invent_hunt.partial.json",
            "lab_mega_desk_invent_hunt.json",
        )
    if invent_st == "running":
        return (
            "lab_adaptive_xm_aud_eth_invent_hunt",
            "lab_adaptive_xm_aud_eth_invent_hunt.progress.json",
            "lab_adaptive_xm_aud_eth_invent_hunt.partial.json",
            "lab_adaptive_xm_aud_eth_invent_hunt.json",
        )
    if cog_st == "running":
        return (
            "lab_adaptive_xm_rest5_cog_hunt",
            "lab_adaptive_xm_rest5_cog_hunt.progress.json",
            "lab_adaptive_xm_rest5_cog_hunt.partial.json",
            "lab_adaptive_xm_rest5_cog_hunt.json",
        )
    if rest_st == "running" or (
        rest_st and rest_st != "done" and deep_st == "done" and cog_st != "done"
    ):
        return (
            "lab_adaptive_xm_rest5_hunt",
            "lab_adaptive_xm_rest5_hunt.progress.json",
            "lab_adaptive_xm_rest5_hunt.partial.json",
            "lab_adaptive_xm_rest5_hunt.json",
        )
    candidates = []
    for prefix in (
        "lab_solution_xm_hunt",
        "lab_mega_desk_invent_hunt",
        "lab_adaptive_xm_aud_eth_invent_hunt",
        "lab_adaptive_xm_rest5_cog_hunt",
        "lab_adaptive_xm_rest5_hunt",
        "lab_adaptive_xm_deep_hunt",
    ):
        path = _find(f"{prefix}.json")
        if path and path.exists():
            candidates.append((path.stat().st_mtime, prefix))
        prog = _find(f"{prefix}.progress.json")
        if prog and prog.exists() and prefix not in [c[1] for c in candidates]:
            try:
                st = json.loads(prog.read_text(encoding="utf-8")).get("status")
                if st == "done":
                    candidates.append((prog.stat().st_mtime, prefix))
            except Exception:
                pass
    if candidates:
        candidates.sort(reverse=True)
        prefix = candidates[0][1]
        return (
            prefix,
            f"{prefix}.progress.json",
            f"{prefix}.partial.json",
            f"{prefix}.json",
        )
    return (
        "lab_adaptive_xm_deep_hunt",
        "lab_adaptive_xm_deep_hunt.progress.json",
        "lab_adaptive_xm_deep_hunt.partial.json",
        "lab_adaptive_xm_deep_hunt.json",
    )


def _catalog(
    lean: bool,
    *,
    rest5: bool = False,
    cog: bool = False,
    invent: bool = False,
    mega: bool = False,
) -> list[dict[str, Any]]:
    """Mode order matches hunt sides_cache: baselines then variants."""
    if mega:
        from analysis.lab_mega_desk_invent_catalog import build_mega_catalog

        modes: list[dict[str, Any]] = []
        seen: set[str] = set()
        for vid, vlab, _ in build_mega_catalog(lean=lean, symbol=None):
            if vid in seen:
                continue
            seen.add(vid)
            modes.append(
                {
                    "id": vid,
                    "label": vlab,
                    "family": _family_of(vid),
                    "lean": lean,
                }
            )
        return modes
    if invent:
        from analysis.lab_aud_eth_math_econ_invent_v2 import build_aud_eth_catalog_v2

        modes = []
        seen = set()
        for vid, vlab, _ in build_aud_eth_catalog_v2(lean=lean, symbol=None, densify=1):
            if vid in seen:
                continue
            seen.add(vid)
            modes.append(
                {
                    "id": vid,
                    "label": vlab,
                    "family": _family_of(vid),
                    "lean": lean,
                }
            )
        return modes
    modes = [
        {"id": "static", "label": "static winner sides", "family": "baseline", "lean": lean},
        {"id": "gated", "label": "regime-gated static", "family": "baseline", "lean": lean},
        {"id": "switch", "label": "regime switch alt", "family": "baseline", "lean": lean},
    ]
    if cog:
        from analysis.lab_adaptive_xm_rest5_cog_hunt_bt import _build_cog_variants

        for vid, vlab, _ in _build_cog_variants(lean=lean):
            modes.append(
                {
                    "id": vid,
                    "label": vlab,
                    "family": _family_of(vid),
                    "lean": lean,
                }
            )
        return modes
    if rest5:
        from analysis.lab_adaptive_xm_rest5_hunt_bt import _build_rest_variants

        for vid, vlab, _ in _build_rest_variants(lean=lean):
            modes.append(
                {
                    "id": vid,
                    "label": vlab,
                    "family": _family_of(vid),
                    "lean": lean,
                }
            )
        return modes
    from analysis.lab_adaptive_xm_deep_hunt_bt import _build_variants

    for vid, vlab, _ in _build_variants(lean=lean):
        modes.append(
            {
                "id": vid,
                "label": vlab,
                "family": _family_of(vid),
                "lean": lean,
            }
        )
    return modes


def _grids() -> dict[str, Any]:
    from analysis.lab_adaptive_xm_deep_hunt_bt import (
        GAP_GRID_BY_TF,
        RR_GRID,
        SL_GRID_BY_TF,
        TF_YEARS,
    )

    return {
        "sl_atr": {tf: list(v) for tf, v in SL_GRID_BY_TF.items()},
        "rr": list(RR_GRID),
        "min_gap": {tf: list(v) for tf, v in GAP_GRID_BY_TF.items()},
        "hist_years": dict(TF_YEARS),
        "windows": ["7d", "15d", "30d", "90d", "180d", "1y", "3y", "full"],
        "costs": "XM session-wise spreads on every trade",
    }


def _tail_log(
    n: int = 40,
    *,
    rest5: bool = False,
    cog: bool = False,
    invent: bool = False,
    mega: bool = False,
) -> list[str]:
    if mega:
        names = (
            "lab_mega_desk_invent_hunt.log",
            "lab_mega_desk_invent_hunt.console.log",
        )
    elif invent:
        names = (
            "lab_adaptive_xm_aud_eth_invent_hunt.log",
            "lab_adaptive_xm_aud_eth_invent_hunt.console.log",
        )
    elif cog:
        names = (
            "lab_adaptive_xm_rest5_cog_hunt.log",
            "lab_adaptive_xm_rest5_cog_hunt.console.log",
        )
    elif rest5:
        names = (
            "lab_adaptive_xm_rest5_hunt.log",
            "lab_adaptive_xm_rest5_hunt.console.log",
        )
    else:
        names = (
            "lab_adaptive_xm_deep_hunt.log",
            "lab_adaptive_xm_deep_hunt.console.log",
        )
    for name in names:
        p = _find(name)
        if not p:
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            return [ln for ln in lines if ln.strip()][-n:]
        except Exception:
            continue
    return []


@router.get("/deep")
async def deep_hunt_live(_: dict = Depends(get_current_user)) -> dict[str, Any]:
    source, prog_name, part_name, final_name = _active_hunt_files()
    mega = source.endswith("mega_desk_invent_hunt")
    invent = source.endswith("aud_eth_invent_hunt")
    cog = source.endswith("cog_hunt")
    rest5 = source.endswith("rest5_hunt") and not cog and not invent and not mega
    progress = _read_json(prog_name) or {}
    partial = _read_json(part_name) or {}
    final = _read_json(final_name)
    per_pair = (partial.get("per_pair") or (final or {}).get("per_pair") or {})
    book = list(progress.get("book") or (final or {}).get("book") or BOOK)
    if mega and not progress.get("book"):
        book = ["GOLD", "BTCUSD", "UKOIL", "NAS100", "AUDUSD", "ETHUSD", "USDJPY", "USDCHF"]
    elif invent and not progress.get("book"):
        book = ["AUDUSD", "ETHUSD"]
    elif (rest5 or cog) and not progress.get("book"):
        book = ["AUDUSD", "USDCHF", "ETHUSD", "NAS100", "USDJPY"]

    active_tf = str(progress.get("tf") or "5m")
    lean = active_tf == "1m"
    try:
        catalog = _catalog(lean=lean, rest5=rest5, cog=cog, invent=invent, mega=mega)
        catalog_full = _catalog(lean=False, rest5=rest5, cog=cog, invent=invent, mega=mega)
        catalog_lean = _catalog(lean=True, rest5=rest5, cog=cog, invent=invent, mega=mega)
        grids = _grids()
    except Exception as exc:
        catalog = []
        catalog_full = []
        catalog_lean = []
        grids = {"error": str(exc)}

    mode_i = int(progress.get("mode_i") or 0)
    modes_n = int(progress.get("modes") or 0) or max(len(catalog), 1)
    # progress.mode_i is 1-based after each 10-mode checkpoint (mi+1)
    # Between checkpoints, treat last reported mode_i as scanned count
    scanned = min(mode_i, len(catalog)) if catalog else mode_i
    current_idx = min(max(scanned - 1, 0), len(catalog) - 1) if catalog else 0
    current_mode = catalog[current_idx] if catalog else None
    # Prefer explicit fields if hunt wrote them
    if progress.get("mode_id") and catalog:
        for i, m in enumerate(catalog):
            if m["id"] == progress.get("mode_id"):
                current_idx = i
                current_mode = m
                break

    family_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "scanned": 0})
    for i, m in enumerate(catalog):
        fam = m["family"]
        family_stats[fam]["total"] += 1
        if i < scanned:
            family_stats[fam]["scanned"] += 1

    # Per-mode scan state for nebula particles
    mode_states = []
    for i, m in enumerate(catalog):
        state = "pending"
        if i < scanned - 1:
            state = "scanned"
        elif i == current_idx and str(progress.get("status")) == "running":
            state = "active"
        elif i < scanned:
            state = "scanned"
        mode_states.append({**m, "index": i, "state": state})

    nodes: list[dict[str, Any]] = []
    done_keys: list[str] = []
    winners: list[dict[str, Any]] = []
    family_heat: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"wins": 0, "pass": 0, "soft": 0, "pnl_1y": 0.0, "modes": {}}
    )
    for sym in book:
        row = per_pair.get(sym) or {}
        by_tf = row.get("by_tf") or {}
        tfs_out = []
        for tf in TFS:
            block = by_tf.get(tf) or {}
            done = bool(block.get("done"))
            best = block.get("best") or {}
            top25 = block.get("top25") or []
            if done:
                done_keys.append(f"{sym}:{tf}")
            best_mode = best.get("mode") or (
                str(best.get("id") or "").split("|")[0] or None
            )
            fam = _family_of(str(best_mode or ""))
            horizons = {
                "7d": best.get("pnl_7d"),
                "15d": best.get("pnl_15d"),
                "30d": best.get("pnl_30d"),
                "90d": best.get("pnl_90d"),
                "180d": best.get("pnl_180d"),
                "1y": best.get("pnl_1y"),
            }
            verdicts = {
                "30d": best.get("verdict_30d"),
                "90d": best.get("verdict_90d"),
                "1y": best.get("verdict_1y"),
            }
            entry = {
                "tf": tf,
                "done": done,
                "error": block.get("error"),
                "best_id": best.get("id"),
                "best_mode": best_mode,
                "best_family": fam if best_mode else None,
                "verdict_1y": best.get("verdict_1y"),
                "pnl_1y": best.get("pnl_1y"),
                "exp_r_1y": best.get("exp_r_1y"),
                "oos_1y": best.get("oos_1y"),
                "wf_1y": best.get("wf_1y"),
                "spR_1y": best.get("spR_1y"),
                "n_1y": best.get("n_1y"),
                "sl_atr": best.get("sl_atr"),
                "rr": best.get("rr"),
                "min_gap": best.get("min_gap"),
                "horizons": horizons,
                "verdicts": verdicts,
                "n_combos": block.get("n_combos"),
                "n_bars": block.get("n_bars"),
                "top": [
                    {
                        "id": c.get("id"),
                        "mode": c.get("mode"),
                        "family": _family_of(str(c.get("mode") or "")),
                        "verdict_1y": c.get("verdict_1y"),
                        "pnl_1y": c.get("pnl_1y"),
                        "exp_r_1y": c.get("exp_r_1y"),
                        "oos_1y": c.get("oos_1y"),
                        "sl_atr": c.get("sl_atr"),
                        "rr": c.get("rr"),
                        "min_gap": c.get("min_gap"),
                    }
                    for c in top25[:12]
                ],
            }
            tfs_out.append(entry)
            if done and best_mode:
                winners.append(
                    {
                        "symbol": sym,
                        "tf": tf,
                        "mode": best_mode,
                        "family": fam,
                        "verdict_1y": best.get("verdict_1y"),
                        "pnl_1y": best.get("pnl_1y"),
                        "id": best.get("id"),
                        "horizons": horizons,
                        "verdicts": verdicts,
                        "sl_atr": best.get("sl_atr"),
                        "rr": best.get("rr"),
                        "min_gap": best.get("min_gap"),
                    }
                )
                fh = family_heat[fam]
                fh["wins"] += 1
                fh["pnl_1y"] += float(best.get("pnl_1y") or 0)
                if best.get("verdict_1y") == "PASS":
                    fh["pass"] += 1
                elif best.get("verdict_1y") == "SOFT":
                    fh["soft"] += 1
                fh["modes"][best_mode] = fh["modes"].get(best_mode, 0) + 1
            for c in top25[:8]:
                if c.get("verdict_1y") not in ("PASS", "SOFT"):
                    continue
                cf = _family_of(str(c.get("mode") or ""))
                family_heat[cf]["modes"][str(c.get("mode"))] = (
                    family_heat[cf]["modes"].get(str(c.get("mode")), 0) + 1
                )
        n_done = sum(1 for t in tfs_out if t["done"])
        nodes.append(
            {
                "symbol": sym,
                "done_tfs": n_done,
                "total_tfs": len(TFS),
                "complete": n_done >= len(TFS),
                "tfs": tfs_out,
                "best_overall": row.get("best_overall"),
            }
        )

    max_wins = max((int(v["wins"]) for v in family_heat.values()), default=1) or 1
    adaptive_families: list[dict[str, Any]] = []
    for fam, st in family_stats.items():
        heat = family_heat.get(fam) or {
            "wins": 0,
            "pass": 0,
            "soft": 0,
            "pnl_1y": 0.0,
            "modes": {},
        }
        score = (0.65 * (heat["wins"] / max_wins)) + (
            0.35 * (st["scanned"] / max(1, st["total"]))
        )
        top_modes = sorted(heat["modes"].items(), key=lambda x: -x[1])[:5]
        adaptive_families.append(
            {
                "id": fam,
                "total": st["total"],
                "scanned": st["scanned"],
                "pct": round(100.0 * st["scanned"] / max(1, st["total"]), 1),
                "wins": heat["wins"],
                "pass": heat["pass"],
                "soft": heat["soft"],
                "pnl_1y": round(float(heat["pnl_1y"]), 2),
                "heat": round(score, 3),
                "top_modes": [{"id": m, "n": n} for m, n in top_modes],
            }
        )
    adaptive_families.sort(key=lambda x: (-x["heat"], -x["wins"], -x["total"]))

    cells_done = int(progress.get("cells_done") or partial.get("cells_done") or 0)
    pair_tf_done = len(done_keys)
    pair_tf_total = len(book) * len(TFS)
    active_frac = (mode_i / modes_n) if progress.get("status") == "running" else 0.0
    pct = min(
        100.0,
        round(100.0 * (pair_tf_done + active_frac) / max(1, pair_tf_total), 1),
    )

    status = str(progress.get("status") or ("done" if final else "idle"))
    if final and status != "running":
        status = "done"

    events: list[dict[str, Any]] = []
    for ln in _tail_log(60, rest5=rest5, cog=cog, invent=invent, mega=mega):
        kind = "log"
        if "BEST" in ln:
            kind = "best"
        elif "########" in ln:
            kind = "start"
        elif "modes ·" in ln or "…" in ln:
            kind = "progress"
        elif "sides " in ln and "fail" in ln:
            kind = "error"
        elif "fail" in ln.lower() or "error" in ln.lower():
            kind = "error"
        events.append({"kind": kind, "text": ln.strip()})

    # Cell math for active TF
    sl = (grids.get("sl_atr") or {}).get(active_tf) or []
    gap = (grids.get("min_gap") or {}).get(active_tf) or []
    rr = grids.get("rr") or []
    cells_per_mode = len(sl) * len(rr) * len(gap) if sl and rr and gap else 0

    if mega:
        hunt_label = "Mega desk invent (400–600+ strategies, full book)"
    elif invent:
        hunt_label = "AUD/ETH invent (math+econ + adaptive)"
    elif cog:
        hunt_label = "Rest-5 cognitive (zenith math + bayes/HMM)"
    elif rest5:
        hunt_label = "Rest-5 adaptive (halted pairs)"
    else:
        hunt_label = "Full-book deep hunt"

    return {
        "source": source,
        "status": status,
        "alive": status == "running",
        "progress": progress,
        "book": list(book),
        "live_untouched": True if mega else (["GOLD", "BTCUSD", "UKOIL", "NAS100"] if invent else (["GOLD", "BTCUSD", "UKOIL"] if (rest5 or cog) else [])),
        "hunt_label": hunt_label,
        "tfs": list(TFS),
        "nodes": nodes,
        "done_keys": done_keys,
        "pair_tf_done": pair_tf_done,
        "pair_tf_total": pair_tf_total,
        "cells_done": cells_done,
        "cells_est": CELLS_EST,
        "pct": pct,
        "elapsed_sec": progress.get("elapsed_sec") or partial.get("elapsed_sec"),
        "active_pair": progress.get("pair"),
        "active_tf": active_tf,
        "mode_i": mode_i,
        "modes": modes_n,
        "events": events[-40:],
        "has_final": final is not None,
        "updated_at": progress.get("updated_at") or partial.get("generated_at"),
        # Deep hunt search space
        "search": {
            "layers": SEARCH_LAYERS,
            "grids": grids,
            "catalog_full_n": len(catalog_full),
            "catalog_lean_n": len(catalog_lean),
            "active_catalog": "lean_1m" if lean else "full_intraday",
            "modes": mode_states,
            "families": adaptive_families,
            "current_mode": current_mode,
            "current_mode_index": current_idx,
            "scanned_modes": scanned,
            "cells_per_mode": cells_per_mode,
            "active_sl_grid": sl,
            "active_rr_grid": rr,
            "active_gap_grid": gap,
            "lattice": [
                {"sl_atr": s, "rr": r, "min_gap": g}
                for s in sl
                for r in rr
                for g in gap
            ],
            "testing_now": {
                "pair": progress.get("pair"),
                "tf": active_tf,
                "mode": (current_mode or {}).get("id"),
                "family": (current_mode or {}).get("family"),
                "label": (current_mode or {}).get("label"),
                "mode_progress": f"{mode_i}/{modes_n}",
                "hist_years": (grids.get("hist_years") or {}).get(active_tf),
                "combos_this_tf": modes_n * cells_per_mode,
            },
            "adaptive": {
                "winners": winners,
                "leading_family": (adaptive_families[0]["id"] if adaptive_families else None),
                "insight": _adaptive_insight(adaptive_families, winners),
            },
        },
    }


def _adaptive_insight(
    families: list[dict[str, Any]], winners: list[dict[str, Any]]
) -> str:
    if not winners:
        return "Hunt is still mapping the catalog — no locked pair×TF winners yet."
    lead = families[0] if families else None
    if not lead or not lead.get("wins"):
        return f"{len(winners)} pair×TF locked. Family preference still forming."
    top_modes = lead.get("top_modes") or []
    tip = top_modes[0]["id"] if top_modes else lead["id"]
    return (
        f"Adaptive bias → {lead['id']} ({lead['wins']} wins, "
        f"{lead['pass']} PASS). Emerging process: {tip}."
    )
