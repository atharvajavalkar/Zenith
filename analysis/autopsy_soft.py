"""Soft overlays proposed/applied by the autopsy improve agent.

Hard-capped extras layered into instrument_rl.overlays_for.
Never rewrites .env or brains — only soft gate/size knobs.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PATH = Path(__file__).resolve().parents[1] / "data" / "autopsy_soft_overlays.json"

# Hard caps — cannot dominate closed-trade RL + candle soft budget
_MC = (-3.0, 4.0)
_ML = (-0.02, 0.03)
_AGG = (-0.10, 0.08)
_RISK = (0.80, 1.10)
_RR = (1.5, 3.0)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _empty() -> dict[str, Any]:
    return {
        "updated_at": None,
        "symbols": {},
        "history": [],
    }


def load() -> dict[str, Any]:
    with _LOCK:
        if not _PATH.exists():
            return _empty()
        try:
            raw = json.loads(_PATH.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return _empty()
            raw.setdefault("symbols", {})
            raw.setdefault("history", [])
            return raw
        except Exception as exc:
            logger.warning("autopsy_soft load failed: %s", exc)
            return _empty()


def save(state: dict[str, Any]) -> None:
    with _LOCK:
        state = dict(state)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def overlays_for(symbol: Optional[str] = None) -> dict[str, Any]:
    """Consumable soft deltas for one symbol (or empty)."""
    if not symbol:
        return {}
    st = load()
    row = (st.get("symbols") or {}).get(str(symbol).upper()) or {}
    if not row.get("active"):
        return {}
    return {
        "mc_floor_delta": _clip(float(row.get("mc_floor_delta") or 0), *_MC),
        "min_ml_delta": _clip(float(row.get("min_ml_delta") or 0), *_ML),
        "aggression": _clip(float(row.get("aggression") or 0), *_AGG),
        "risk_scale": _clip(float(row.get("risk_scale") or 1.0), *_RISK),
        "rr_prefer": _clip(float(row.get("rr_prefer") or 2.0), *_RR)
        if row.get("rr_prefer") is not None
        else None,
        "source": "autopsy_soft",
        "proposal_id": row.get("proposal_id"),
        "note": row.get("note"),
    }


def apply_soft(
    symbol: str,
    patch: dict[str, Any],
    *,
    proposal_id: Optional[str] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """Merge capped soft knobs for a symbol and mark active."""
    sym = str(symbol).upper().strip()
    st = load()
    cur = dict((st.get("symbols") or {}).get(sym) or {})
    if "mc_floor_delta" in patch:
        cur["mc_floor_delta"] = _clip(
            float(cur.get("mc_floor_delta") or 0) + float(patch["mc_floor_delta"]),
            *_MC,
        )
    if "min_ml_delta" in patch:
        cur["min_ml_delta"] = _clip(
            float(cur.get("min_ml_delta") or 0) + float(patch["min_ml_delta"]),
            *_ML,
        )
    if "aggression" in patch:
        cur["aggression"] = _clip(
            float(cur.get("aggression") or 0) + float(patch["aggression"]),
            *_AGG,
        )
    if "risk_scale" in patch:
        # absolute or multiplicative — treat as absolute target when |x-1| style
        rs = float(patch["risk_scale"])
        if 0.5 <= rs <= 1.5:
            cur["risk_scale"] = _clip(rs, *_RISK)
        else:
            cur["risk_scale"] = _clip(float(cur.get("risk_scale") or 1.0) * rs, *_RISK)
    if "rr_prefer" in patch:
        cur["rr_prefer"] = _clip(float(patch["rr_prefer"]), *_RR)
    cur["active"] = True
    cur["proposal_id"] = proposal_id
    cur["note"] = note or cur.get("note")
    cur["applied_at"] = datetime.now(timezone.utc).isoformat()
    st.setdefault("symbols", {})[sym] = cur
    hist = list(st.get("history") or [])
    hist.insert(
        0,
        {
            "at": cur["applied_at"],
            "symbol": sym,
            "proposal_id": proposal_id,
            "patch": patch,
            "note": note,
        },
    )
    st["history"] = hist[:40]
    save(st)
    return cur


def clear_symbol(symbol: str) -> None:
    sym = str(symbol).upper().strip()
    st = load()
    if sym in (st.get("symbols") or {}):
        st["symbols"][sym]["active"] = False
        save(st)


def clear_all(*, reason: str = "desk_session_reset") -> dict[str, Any]:
    """Deactivate every soft overlay (research restarts with new-brain closes)."""
    from datetime import datetime, timezone

    st = load()
    syms = st.get("symbols") or {}
    cleared = []
    for sym, row in list(syms.items()):
        if not row:
            continue
        row = dict(row)
        row["active"] = False
        row["cleared_reason"] = reason
        row["cleared_at"] = datetime.now(timezone.utc).isoformat()
        syms[sym] = row
        cleared.append(sym)
    st["symbols"] = syms
    st["updated_at"] = datetime.now(timezone.utc).isoformat()
    st["cleared_reason"] = reason
    save(st)
    return {"cleared": cleared, "reason": reason}


def snapshot(symbol: str) -> dict[str, Any]:
    """Full row copy for rollback (may be inactive / missing)."""
    sym = str(symbol).upper().strip()
    row = (load().get("symbols") or {}).get(sym)
    if not row:
        return {"symbol": sym, "active": False}
    return dict(row)


def restore(symbol: str, snap: Optional[dict[str, Any]]) -> None:
    """Restore prior soft row, or deactivate if snap was empty."""
    sym = str(symbol).upper().strip()
    st = load()
    if not snap or not snap.get("active"):
        if sym in (st.get("symbols") or {}):
            st["symbols"][sym] = dict(snap) if snap else {"active": False}
            st["symbols"][sym]["active"] = False
            save(st)
        return
    st.setdefault("symbols", {})[sym] = dict(snap)
    save(st)


def replace_symbol(symbol: str, row: dict[str, Any]) -> dict[str, Any]:
    """Overwrite symbol soft row (used when applying a clean trial patch)."""
    sym = str(symbol).upper().strip()
    st = load()
    cur = dict(row)
    cur["active"] = bool(cur.get("active", True))
    st.setdefault("symbols", {})[sym] = cur
    save(st)
    return cur
