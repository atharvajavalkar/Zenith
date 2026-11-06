"""Promoted brain knobs from autopsy improve agent (post soft-apply validation).

When a soft-apply clears OOS / walk-forward / Monte Carlo vs the previous
setup, pull/disp (and related) knobs are written here and consumed by live
signal.py. Rollback restores the prior snapshot.
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
_PATH = Path(__file__).resolve().parents[1] / "data" / "autopsy_brain_promoted.json"

_PULL = (0.55, 1.35)
_DISP = (0.40, 0.90)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _empty() -> dict[str, Any]:
    return {"updated_at": None, "symbols": {}, "history": []}


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
            logger.warning("autopsy_brain load failed: %s", exc)
            return _empty()


def save(state: dict[str, Any]) -> None:
    with _LOCK:
        state = dict(state)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get(symbol: Optional[str] = None) -> dict[str, Any]:
    """Active promoted knobs for a symbol, or {}."""
    if not symbol:
        return {}
    row = (load().get("symbols") or {}).get(str(symbol).upper()) or {}
    if not row.get("active"):
        return {}
    out: dict[str, Any] = {"source": "autopsy_brain", "proposal_id": row.get("proposal_id")}
    if row.get("pull_atr") is not None:
        out["pull_atr"] = _clip(float(row["pull_atr"]), *_PULL)
    if row.get("disp_atr") is not None:
        out["disp_atr"] = _clip(float(row["disp_atr"]), *_DISP)
    if row.get("sl_widen") is not None:
        out["sl_widen"] = max(1.0, min(1.35, float(row["sl_widen"])))
    return out


def snapshot(symbol: str) -> dict[str, Any]:
    """Copy of current row (for rollback), including inactive."""
    sym = str(symbol).upper().strip()
    row = (load().get("symbols") or {}).get(sym)
    if not row:
        return {"symbol": sym, "active": False}
    return dict(row)


def restore(symbol: str, snap: Optional[dict[str, Any]]) -> None:
    """Restore a prior snapshot (or clear if snap empty/inactive)."""
    sym = str(symbol).upper().strip()
    st = load()
    if not snap or not snap.get("active"):
        if sym in (st.get("symbols") or {}):
            st["symbols"][sym]["active"] = False
            save(st)
        return
    st.setdefault("symbols", {})[sym] = dict(snap)
    save(st)


def clear_all(*, reason: str = "desk_session_reset") -> dict[str, Any]:
    """Deactivate all promoted brain knobs — research restarts on new-brain closes."""
    st = load()
    cleared = []
    for sym, row in list((st.get("symbols") or {}).items()):
        if not row:
            continue
        row = dict(row)
        row["active"] = False
        row["cleared_reason"] = reason
        row["cleared_at"] = datetime.now(timezone.utc).isoformat()
        st["symbols"][sym] = row
        cleared.append(sym)
    st["cleared_reason"] = reason
    save(st)
    return {"cleared": cleared, "reason": reason}


def promote(
    symbol: str,
    knobs: dict[str, Any],
    *,
    proposal_id: Optional[str] = None,
    note: Optional[str] = None,
    previous: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write validated knobs as the new live brain baseline for this symbol."""
    sym = str(symbol).upper().strip()
    st = load()
    cur: dict[str, Any] = {
        "active": True,
        "proposal_id": proposal_id,
        "note": note,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "previous": previous or snapshot(sym),
    }
    if knobs.get("pull_atr") is not None:
        cur["pull_atr"] = _clip(float(knobs["pull_atr"]), *_PULL)
    if knobs.get("disp_atr") is not None:
        cur["disp_atr"] = _clip(float(knobs["disp_atr"]), *_DISP)
    if knobs.get("sl_widen") is not None:
        cur["sl_widen"] = max(1.0, min(1.35, float(knobs["sl_widen"])))
    st.setdefault("symbols", {})[sym] = cur
    hist = list(st.get("history") or [])
    hist.insert(
        0,
        {
            "at": cur["promoted_at"],
            "symbol": sym,
            "proposal_id": proposal_id,
            "knobs": {k: cur[k] for k in ("pull_atr", "disp_atr", "sl_widen") if k in cur},
            "note": note,
        },
    )
    st["history"] = hist[:40]
    save(st)

    # Seed candle_edge so soft learning starts from the new baseline
    try:
        from analysis.candle_edge import seed_zenith_knobs

        seed_zenith_knobs(
            sym,
            pull_atr=cur.get("pull_atr"),
            disp_atr=cur.get("disp_atr"),
        )
    except Exception as exc:
        logger.debug("candle seed skipped: %s", exc)

    return cur


def defaults_for_brain(brain: str) -> dict[str, float]:
    b = (brain or "").lower()
    if b in ("ema_20_50_tight", "ema_tight"):
        return {"pull_atr": 0.75, "disp_atr": 0.70}
    if b.startswith("ema_"):
        return {"pull_atr": 1.0, "disp_atr": 0.55}
    return {}


def live_ema_knobs(symbol: str, brain: str) -> dict[str, float]:
    """Defaults ← promoted overrides for live signal path."""
    base = defaults_for_brain(brain)
    promo = get(symbol)
    out = dict(base)
    if promo.get("pull_atr") is not None:
        out["pull_atr"] = float(promo["pull_atr"])
    if promo.get("disp_atr") is not None:
        out["disp_atr"] = float(promo["disp_atr"])
    return out
