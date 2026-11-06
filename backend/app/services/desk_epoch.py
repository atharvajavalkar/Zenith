"""Desk session epoch — soft-reset for W/BE/L and Zenith session P&L."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOCK = threading.Lock()
_EPOCH_PATH = Path(__file__).resolve().parents[3] / "data" / "desk_stats_epoch.json"
# fallback if package layout differs
if not _EPOCH_PATH.parent.exists():
    _EPOCH_PATH = Path("/app/data/desk_stats_epoch.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_epoch() -> Optional[datetime]:
    with _LOCK:
        try:
            if not _EPOCH_PATH.exists():
                return None
            raw = json.loads(_EPOCH_PATH.read_text(encoding="utf-8"))
            ts = raw.get("reset_at")
            if not ts:
                return None
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return None


def get_epoch_iso() -> Optional[str]:
    ep = get_epoch()
    return ep.isoformat() if ep else None


def reset_desk_stats(*, reason: str = "manual") -> dict[str, Any]:
    """Start a fresh desk session — counters / session P&L begin here."""
    balance_at_reset = None
    try:
        from data.broker import broker_account_summary

        acct = broker_account_summary(force=True)
        if acct and acct.get("balance") is not None:
            balance_at_reset = float(acct["balance"])
    except Exception:
        balance_at_reset = None

    payload = {
        "reset_at": _now_iso(),
        "reason": reason,
        "balance_at_reset": balance_at_reset,
    }
    with _LOCK:
        _EPOCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        _EPOCH_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    guard_state = None
    try:
        from analysis.adaptive_guard import reset_state

        guard_state = reset_state()
    except Exception:
        try:
            from analysis import adaptive_guard as ag

            with ag._lock:
                ag._state.update(
                    {
                        "mode": "normal",
                        "realized_pnl": 0.0,
                        "peak_pnl": 0.0,
                        "recent_count": 0,
                        "rolling_win_rate": None,
                        "loss_streak": 0,
                        "blocked_states": [],
                        "blocked_hours": [],
                        "last_trade_pnl": None,
                        "updates": 0,
                        "message": "Desk reset — awaiting closed trades",
                        "equity_mode": "ok",
                        "equity_reason": "Equity OK",
                        "equity_allow": True,
                    }
                )
                guard_state = dict(ag._state)
        except Exception:
            pass

    research_cleared = {}
    try:
        from analysis.autopsy_soft import clear_all as clear_soft

        research_cleared["soft"] = clear_soft(reason=f"desk_reset:{reason}")
    except Exception:
        pass
    try:
        from analysis.autopsy_brain import clear_all as clear_brain

        research_cleared["brain"] = clear_brain(reason=f"desk_reset:{reason}")
    except Exception:
        pass

    return {
        "reset_at": payload["reset_at"],
        "reason": reason,
        "guard": guard_state,
        "balance_at_reset": balance_at_reset,
        "research_cleared": research_cleared,
        "message": "Won / BE / Lost reset — counting from this moment",
    }


def row_after_epoch(created_at: Any, log_time: Optional[str] = None) -> bool:
    """True if the trade should count toward the current desk session."""
    ep = get_epoch()
    if ep is None:
        return True

    def _as_dt(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(text[:26], fmt.replace("%z", "") if "%z" not in fmt else fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                continue
        return None

    for candidate in (created_at, log_time):
        dt = _as_dt(candidate)
        if dt is not None:
            return dt >= ep
    # Unknown timestamp with an active epoch → exclude (old history)
    return False


def filter_rows_since_epoch(rows: list[Any]) -> list[Any]:
    """Keep only closed-trade dicts/objects that fall after desk reset."""
    if get_epoch() is None:
        return list(rows)
    out: list[Any] = []
    for r in rows:
        if isinstance(r, dict):
            if row_after_epoch(r.get("created_at"), r.get("log_time")):
                out.append(r)
            continue
        if row_after_epoch(getattr(r, "created_at", None), getattr(r, "log_time", None)):
            out.append(r)
    return out
