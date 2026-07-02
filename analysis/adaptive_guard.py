"""Per-trade adaptive guard — tightens filters when PnL deteriorates."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from typing import Any, Optional

from analysis.trade_outcome import is_breakeven_trade, is_losing_trade, is_winning_trade

_lock = threading.Lock()
_state: dict[str, Any] = {
    "mode": "normal",
    "realized_pnl": 0.0,
    "recent_count": 0,
    "rolling_win_rate": None,
    "loss_streak": 0,
    "mc_floor": 50.0,
    "min_ml_prob": 0.50,
    "blocked_states": [],
    "blocked_hours": [],
    "last_trade_pnl": None,
    "updates": 0,
    "message": "Awaiting closed trades",
}


def _parse_hour(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            dt = datetime.strptime(value.replace("+00:00", ""), fmt.replace("%z", ""))
            return dt.hour
        except ValueError:
            continue
    match = re.search(r"\b(\d{2}):\d{2}:\d{2}\b", value)
    return int(match.group(1)) if match else None


def _parse_features(row: dict) -> dict:
    feats = row.get("features") or {}
    if isinstance(feats, str):
        try:
            feats = json.loads(feats)
        except json.JSONDecodeError:
            feats = {}
    return feats


def rebuild(closed: list[dict], loss_pnl_protect: float = -150.0, mc_floor_base: float = 50.0) -> dict:
    """Rebuild guard state from all closed trades (call after every trade close)."""
    global _state
    rows = [
        r
        for r in closed
        if r.get("status") == "closed" and r.get("signal") in ("BUY", "SELL")
    ]
    rows.sort(key=lambda r: r.get("created_at") or r.get("id") or 0)

    realized = round(sum(float(r.get("pnl") or 0) for r in rows), 2)
    recent = rows[-30:]
    wins_recent = sum(1 for r in recent if is_winning_trade(r))
    decisive_recent = [r for r in recent if not is_breakeven_trade(r)]
    rolling_wr = wins_recent / len(decisive_recent) if decisive_recent else None

    loss_streak = 0
    for r in reversed(rows):
        if is_breakeven_trade(r):
            break
        if is_losing_trade(r):
            loss_streak += 1
        elif is_winning_trade(r):
            break

    mode = "normal"
    if realized <= loss_pnl_protect or (rolling_wr is not None and rolling_wr < 0.35 and len(recent) >= 8):
        mode = "protect"
    elif loss_streak >= 2 or (rolling_wr is not None and rolling_wr < 0.50 and len(recent) >= 6):
        mode = "cautious"

    mc_floor = mc_floor_base
    min_ml_prob = 0.50
    if mode == "cautious":
        mc_floor += 6.0
        min_ml_prob = 0.56
    elif mode == "protect":
        mc_floor += 12.0
        min_ml_prob = 0.63

    state_wr: dict[str, list[float]] = {}
    hour_wr: dict[int, list[float]] = {}
    for r in rows[-40:]:
        feats = _parse_features(r)
        st = feats.get("STATE")
        hr = _parse_hour(r.get("candle_time") or r.get("log_time"))
        win = 1.0 if is_winning_trade(r) else 0.0
        if is_breakeven_trade(r):
            continue
        if st:
            state_wr.setdefault(st, []).append(win)
        if hr is not None:
            hour_wr.setdefault(int(hr), []).append(win)

    blocked_states = [
        st
        for st, outcomes in state_wr.items()
        if len(outcomes) >= 3 and sum(outcomes) / len(outcomes) < 0.28
    ]
    blocked_hours = [
        hr
        for hr, outcomes in hour_wr.items()
        if len(outcomes) >= 4 and sum(outcomes) / len(outcomes) < 0.30
    ]

    last_pnl = rows[-1].get("pnl") if rows else None
    messages = {
        "normal": f"Learning active — realized PnL ${realized:.2f}",
        "cautious": f"Cautious mode — loss streak {loss_streak}, rolling WR {rolling_wr:.0%}" if rolling_wr else f"Cautious — loss streak {loss_streak}",
        "protect": f"Protect mode — realized PnL ${realized:.2f} (target recovery)",
    }

    new_state = {
        "mode": mode,
        "realized_pnl": realized,
        "recent_count": len(recent),
        "rolling_win_rate": round(rolling_wr, 4) if rolling_wr is not None else None,
        "loss_streak": loss_streak,
        "mc_floor": mc_floor,
        "min_ml_prob": min_ml_prob,
        "blocked_states": blocked_states,
        "blocked_hours": blocked_hours,
        "last_trade_pnl": last_pnl,
        "updates": _state.get("updates", 0) + 1,
        "message": messages.get(mode, messages["normal"]),
    }
    with _lock:
        _state = new_state
    return new_state


def get_state() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def check(features: dict[str, Any]) -> tuple[bool, str]:
    """Adaptive block rules from latest closed-trade learning."""
    with _lock:
        st = _state

    state = features.get("STATE")
    if state and state in st.get("blocked_states", []):
        return False, f"Adaptive block — {state} underperforming in recent trades"

    hour = features.get("hour_utc")
    if hour is not None and int(hour) in st.get("blocked_hours", []):
        return False, f"Adaptive block — hour {int(hour):02d} UTC weak in recent history"

    if st.get("mode") == "protect":
        mc = features.get("markov_mc_prob") or features.get("mc_prob")
        if mc is not None and mc < st["mc_floor"]:
            return False, f"Protect mode — MC {mc:.0f}% below {st['mc_floor']:.0f}%"

        margin = features.get("score_margin")
        if margin is not None and margin < 1.0:
            return False, "Protect mode — score margin too tight (<1.0)"

    return True, "Adaptive OK"


def mc_floor() -> float:
    with _lock:
        return float(_state.get("mc_floor", 50.0))


def min_ml_prob() -> float:
    with _lock:
        return float(_state.get("min_ml_prob", 0.50))
