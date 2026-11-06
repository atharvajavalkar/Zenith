"""Autopsy self-heal — tip → A/B → soft-apply → promote/rollback, unattended.

Runs after closed-trade learning when the desk is bleeding.
Throttled so MetaAPI / yfinance aren't hammered.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "autopsy_self_heal.json"
_MIN_INTERVAL_SEC = 20 * 60  # at most one full research cycle / 20m
_LOSS_STREAK_TRIGGER = 3
_EFF_TRIGGER = 4  # efficiency ≤ this → heal that symbol


def _load() -> dict[str, Any]:
    if not _STATE_PATH.exists():
        return {"last_run_at": None, "last_result": None, "runs": []}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": None, "last_result": None, "runs": []}


def _save(st: dict[str, Any]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    st["updated_at"] = datetime.now(timezone.utc).isoformat()
    _STATE_PATH.write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")


def _should_run(force: bool = False) -> tuple[bool, str]:
    if force:
        return True, "force"
    st = _load()
    last = st.get("last_run_at")
    if last:
        try:
            ts = datetime.fromisoformat(str(last).replace("Z", "+00:00")).timestamp()
            if time.time() - ts < _MIN_INTERVAL_SEC:
                return False, "throttled"
        except Exception:
            pass
    return True, "interval_ok"


def _bleeding_symbols(days: int = 3) -> list[str]:
    """Symbols that need healing from recent closes."""
    try:
        from analysis.trade_autopsy_agent import _fetch_closes, diagnose, dashboard_pairs
        from config import INSTRUMENT_UNIVERSE, is_autopsy_excluded

        rows = _fetch_closes(days, 200)
        autopsies = [diagnose(r) for r in rows]
        pairs = dashboard_pairs(autopsies, prefer_symbols=tuple(INSTRUMENT_UNIVERSE))
        out = []
        for p in pairs:
            sym = str(p.get("symbol") or "").upper()
            if is_autopsy_excluded(sym):
                continue
            if int(p.get("n") or 0) < 2:
                continue
            losses = int(p.get("losses") or 0)
            wins = int(p.get("wins") or 0)
            eff = int(p.get("efficiency") or 5)
            if eff <= _EFF_TRIGGER or losses >= wins + 2:
                out.append(sym)
        return out
    except Exception as exc:
        logger.warning("self-heal bleed scan failed: %s", exc)
        return []


def _emergency_soft(symbol: str, *, reason: str) -> dict[str, Any]:
    """Immediate soft tighten without waiting for full A/B (desk on fire)."""
    from config import is_autopsy_excluded
    from analysis.autopsy_soft import apply_soft

    if is_autopsy_excluded(symbol):
        return {
            "symbol": str(symbol).upper(),
            "skipped": "autopsy_excluded",
            "kind": "emergency_soft",
        }

    patch = {
        "mc_floor_delta": 2.5,
        "aggression": -0.06,
        "risk_scale": 0.85,
    }
    cur = apply_soft(
        symbol,
        patch,
        proposal_id=f"emergency-{symbol.lower()}",
        note=f"self-heal emergency: {reason}",
    )
    return {"symbol": symbol, "patch": patch, "overlay": cur, "kind": "emergency_soft"}


def run_self_heal(
    *,
    force: bool = False,
    days: int = 14,
    auto_apply: bool = True,
    max_apply: int = 3,
) -> dict[str, Any]:
    """Full cycle: research weak pairs → validate-apply best proposals → emergency soft."""
    ok, why = _should_run(force=force)
    if not ok:
        return {"ok": False, "skipped": why}

    bleeding = _bleeding_symbols(days=min(days, 5))
    result: dict[str, Any] = {
        "ok": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "bleeding": bleeding,
        "proposals": [],
        "applied": [],
        "emergency": [],
        "errors": [],
    }
    if not bleeding and not force:
        result["ok"] = False
        result["skipped"] = "no_bleeding_symbols"
        st = _load()
        st["last_run_at"] = result["started_at"]
        st["last_result"] = result
        _save(st)
        return result

    # 1) Research
    try:
        from analysis.autopsy_improve_agent import run_improve, apply_proposal, get_proposals

        payload = run_improve(
            days=days,
            symbols=bleeding or None,
            max_pairs=min(6, max(3, len(bleeding) or 3)),
            max_exps_per_pair=2,
            mc_paths=300,
            lookback_days=90,
        )
        props = [
            p
            for p in (payload.get("proposals") or [])
            if p.get("status") == "proposed" and p.get("apply_safe")
        ]
        result["proposals"] = [
            {
                "id": p.get("id"),
                "symbol": p.get("symbol"),
                "label": p.get("label"),
                "confidence": p.get("confidence"),
            }
            for p in props
        ]
    except Exception as exc:
        logger.exception("self-heal research failed: %s", exc)
        result["errors"].append(f"research:{exc}")
        props = []
        try:
            from analysis.autopsy_improve_agent import get_proposals

            props = [
                p
                for p in (get_proposals().get("proposals") or [])
                if p.get("status") == "proposed" and p.get("apply_safe")
            ]
        except Exception:
            props = []

    # 2) Auto apply+validate (promote or rollback)
    if auto_apply:
        applied_syms: set[str] = set()
        from config import is_autopsy_excluded

        for p in props:
            if len(result["applied"]) >= max_apply:
                break
            sym = str(p.get("symbol") or "").upper()
            if not sym or sym in applied_syms:
                continue
            if is_autopsy_excluded(sym):
                continue
            # Prefer bleeding symbols
            if bleeding and sym not in bleeding:
                continue
            try:
                from analysis.autopsy_improve_agent import apply_proposal

                out = apply_proposal(str(p["id"]))
                result["applied"].append(
                    {
                        "id": p.get("id"),
                        "symbol": sym,
                        "decision": out.get("decision"),
                        "reason": out.get("reason"),
                        "label": p.get("label"),
                    }
                )
                applied_syms.add(sym)
                logger.info(
                    "self-heal apply %s → %s (%s)",
                    sym,
                    out.get("decision"),
                    out.get("reason"),
                )
            except Exception as exc:
                result["errors"].append(f"apply:{sym}:{exc}")
                logger.warning("self-heal apply failed %s: %s", sym, exc)

        # 3) Emergency soft for bleeding symbols with no successful promote
        healed = {
            a["symbol"]
            for a in result["applied"]
            if a.get("decision") in ("promoted", "promoted_soft", "soft_retained")
        }
        for sym in bleeding:
            if sym in healed:
                continue
            try:
                em = _emergency_soft(sym, reason="bleeding_no_ab_win")
                result["emergency"].append(em)
                logger.info("self-heal emergency soft %s", sym)
            except Exception as exc:
                result["errors"].append(f"emergency:{sym}:{exc}")

    # 4) Full stochastic battery after every heal (healed + bleeding pairs)
    try:
        from analysis.post_heal_battery import run_post_heal_battery

        bat_syms = sorted(
            set(bleeding)
            | {str(a.get("symbol") or "").upper() for a in result["applied"] if a.get("symbol")}
            | {
                str(e.get("symbol") or "").upper()
                for e in result["emergency"]
                if e.get("symbol")
            }
        )
        bat_syms = [s for s in bat_syms if s]
        if force and not bat_syms:
            from config import INSTRUMENT_UNIVERSE

            bat_syms = list(INSTRUMENT_UNIVERSE)
        if bat_syms:
            logger.info("post-heal full battery for %s", bat_syms)
            battery = run_post_heal_battery(bat_syms, lookback_days=182, mc_paths=800)
            result["battery"] = {
                "symbols": bat_syms,
                "summary": battery.get("summary"),
                "results": battery.get("results"),
                "ran_at": battery.get("ran_at"),
            }
            # Flag DEAD after heal — keep soft but surface clearly
            dead = [
                r["symbol"]
                for r in (battery.get("results") or [])
                if r.get("verdict") == "DEAD"
            ]
            if dead:
                result["battery_dead"] = dead
                logger.warning("post-heal battery DEAD: %s", dead)
    except Exception as exc:
        result["errors"].append(f"battery:{exc}")
        logger.exception("post-heal battery failed: %s", exc)

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    st = _load()
    st["last_run_at"] = result["finished_at"]
    st["last_result"] = {
        "bleeding": result["bleeding"],
        "n_proposals": len(result["proposals"]),
        "n_applied": len(result["applied"]),
        "n_emergency": len(result["emergency"]),
        "battery_summary": (result.get("battery") or {}).get("summary"),
        "battery_dead": result.get("battery_dead"),
        "errors": result["errors"],
        "finished_at": result["finished_at"],
    }
    hist = list(st.get("runs") or [])
    hist.insert(0, st["last_result"])
    st["runs"] = hist[:30]
    _save(st)
    return result


def maybe_self_heal_after_close(closed_row: Optional[dict] = None) -> None:
    """Cheap trigger from learning_loop — only kick a full run when bleeding."""
    try:
        from config import AUTOPSY_SELF_HEAL_ENABLED

        if not AUTOPSY_SELF_HEAL_ENABLED:
            return
    except Exception:
        return
    try:
        pnl = float((closed_row or {}).get("pnl") or 0)
    except Exception:
        pnl = 0.0
    # Always consider on a loss; winners don't need heal
    if closed_row is not None and pnl >= -1.0:
        # still check streak via adaptive
        try:
            from analysis.adaptive_guard import get_state

            st = get_state()
            if int(st.get("loss_streak") or 0) < _LOSS_STREAK_TRIGGER:
                return
        except Exception:
            return
    ok, why = _should_run(force=False)
    if not ok:
        logger.debug("self-heal skip: %s", why)
        return
    bleeding = _bleeding_symbols(days=3)
    if not bleeding:
        logger.debug("self-heal: no bleeding symbols")
        return
    logger.info("self-heal triggered for %s", bleeding)
    try:
        run_self_heal(force=False, days=14, auto_apply=True, max_apply=3)
    except Exception as exc:
        logger.exception("self-heal run failed: %s", exc)


def status() -> dict[str, Any]:
    st = _load()
    try:
        from analysis.post_heal_battery import latest_post_heal_battery

        st["latest_battery"] = latest_post_heal_battery()
    except Exception:
        st["latest_battery"] = None
    return st
