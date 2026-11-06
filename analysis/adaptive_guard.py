"""Per-trade adaptive guard — tightens filters when PnL deteriorates."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from analysis.equity_protect import evaluate_equity_gates, update_peak
from analysis.trade_outcome import is_breakeven_trade, is_losing_trade, is_winning_trade

# runtime_config provides live-tweakable params without a restart
try:
    import runtime_config as _rc
    _USE_RC = True
except ImportError:
    _USE_RC = False


def _rcget(key: str, fallback):
    """Get a runtime config value, falling back gracefully if unavailable."""
    if _USE_RC:
        return _rc.get(key, fallback)
    return fallback


def _cfg(name: str, default):
    try:
        from config import (
            BANK_TARGET_PCT,
            DAILY_LOSS_LIMIT_PCT,
            HARVEST_PCT,
            MAX_TRADES_PER_DAY,
            PEAK_DD_PROTECT_PCT,
            START_CAPITAL_USD,
        )
        mapping = {
            "start_capital_usd": START_CAPITAL_USD,
            "peak_dd_protect_pct": PEAK_DD_PROTECT_PCT,
            "daily_loss_limit_pct": DAILY_LOSS_LIMIT_PCT,
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "harvest_pct": HARVEST_PCT,
            "bank_target_pct": BANK_TARGET_PCT,
        }
        return mapping.get(name, default)
    except Exception:
        return default


_lock = threading.Lock()
_state: dict[str, Any] = {
    "mode": "normal",
    "realized_pnl": 0.0,
    "peak_pnl": 0.0,
    "recent_count": 0,
    "rolling_win_rate": None,
    "loss_streak": 0,
    "mc_floor": 42.0,
    "min_ml_prob": 0.42,
    "blocked_states": [],
    "blocked_hours": [],
    "last_trade_pnl": None,
    "updates": 0,
    "message": "Awaiting closed trades",
    "equity_mode": "ok",
    "equity_reason": "Equity OK",
    "equity_allow": True,
    "start_capital": 1000.0,
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


def _parse_day(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = str(value)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


def _parse_features(row: dict) -> dict:
    feats = row.get("features") or {}
    if isinstance(feats, str):
        try:
            feats = json.loads(feats)
        except json.JSONDecodeError:
            feats = {}
    return feats


def rebuild(
    closed: list[dict],
    loss_pnl_protect: float = -150.0,
    mc_floor_base: float = 50.0,
    start_capital: Optional[float] = None,
    as_of_day: Optional[str] = None,
) -> dict:
    """Rebuild guard state from all closed trades (call after every trade close).

    as_of_day: YYYY-MM-DD for backtest parity (defaults to UTC today for live).
    start_capital: overrides START_CAPITAL_USD when provided (backtest capital input).
    """
    global _state

    _loss_pnl = _rcget("loss_pnl_protect", loss_pnl_protect)
    _mc_base = _rcget("mc_floor_base", mc_floor_base)
    _streak_thr = _rcget("loss_streak_cautious", 2)

    rows = [
        r
        for r in closed
        if r.get("status") == "closed" and r.get("signal") in ("BUY", "SELL")
    ]
    rows.sort(key=lambda r: r.get("created_at") or r.get("id") or r.get("exit_time") or 0)

    realized = 0.0
    peak = 0.0
    for r in rows:
        realized += float(r.get("pnl") or 0)
        peak = update_peak(peak, realized)
    realized = round(realized, 2)
    peak = round(peak, 2)

    today = as_of_day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    day_rows = [
        r
        for r in rows
        if _parse_day(str(r.get("exit_time") or r.get("candle_time") or r.get("created_at") or ""))
        == today
    ]
    day_pnl = round(sum(float(r.get("pnl") or 0) for r in day_rows), 2)
    trades_today = len(day_rows)

    # Prefer live XM balance for % gates — START_CAPITAL_USD is only a cold-start fallback.
    # (Bug: $1k start + −$99 day PnL tripped "4% daily loss" while live capital was ~$9k.)
    start_cap = float(
        start_capital
        if start_capital is not None
        else _rcget("start_capital_usd", _cfg("start_capital_usd", 1000.0))
    )
    if start_capital is None:
        try:
            from strategy.dynamic_lots import fetch_account_capital

            live = float(fetch_account_capital())
            if live >= 500:
                start_cap = live
        except Exception:
            pass
    eq = evaluate_equity_gates(
        start_capital=start_cap,
        realized_pnl=realized,
        peak_pnl=peak,
        day_pnl=day_pnl,
        trades_today=trades_today,
        peak_dd_pct=float(_rcget("peak_dd_protect_pct", _cfg("peak_dd_protect_pct", 0.15))),
        daily_loss_pct=float(_rcget("daily_loss_limit_pct", _cfg("daily_loss_limit_pct", 0.04))),
        max_trades_per_day=int(_rcget("max_trades_per_day", _cfg("max_trades_per_day", 6))),
        harvest_pct=float(_rcget("harvest_pct", _cfg("harvest_pct", 0.50))),
        bank_target_pct=float(_rcget("bank_target_pct", _cfg("bank_target_pct", 1.00))),
    )

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

    if eq["mode"] in ("bank", "protect", "daily"):
        mode = "protect"
    elif eq["mode"] == "harvest":
        mode = "cautious"
    else:
        # Skip day_pnl protect when daily-loss gate is disabled (testing).
        _daily_pct = float(
            _rcget("daily_loss_limit_pct", _cfg("daily_loss_limit_pct", 0.0))
        )
        if (_daily_pct > 0 and day_pnl <= _loss_pnl) or (
            rolling_wr is not None and rolling_wr < 0.35 and len(recent) >= 8
        ):
            mode = "protect"
        elif loss_streak >= _streak_thr or (
            rolling_wr is not None and rolling_wr < 0.50 and len(recent) >= 6
        ):
            mode = "cautious"
        else:
            mode = "normal"

    mc_floor = float(_mc_base) + float(eq.get("mc_bump") or 0)
    min_ml_prob = _rcget("ml_prob_normal", 0.36)
    if mode == "cautious":
        # Mild bump only — RL learns from fills, not from a frozen desk
        mc_floor += _rcget("mc_cautious_offset", 3.0)
        min_ml_prob = _rcget("ml_prob_cautious", 0.38)
    elif mode == "protect":
        mc_floor += _rcget("mc_protect_offset", 4.0)
        min_ml_prob = _rcget("ml_prob_protect", 0.40)
    # Never let adaptive floors drift into "nothing trades" territory
    mc_floor = min(float(mc_floor), 42.0)
    min_ml_prob = min(float(min_ml_prob), 0.42)

    state_wr: dict[str, list[float]] = {}
    hour_wr: dict[int, list[float]] = {}
    for r in rows[-40:]:
        feats = _parse_features(r)
        st = feats.get("STATE")
        hr = _parse_hour(r.get("candle_time") or r.get("log_time") or str(r.get("exit_time") or ""))
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
        # Stricter — 3 losses at 28% WR was freezing all MTF entries after a bad scalp run
        if len(outcomes) >= 6 and sum(outcomes) / len(outcomes) < 0.22
    ]
    blocked_hours = [
        hr
        for hr, outcomes in hour_wr.items()
        if len(outcomes) >= 6 and sum(outcomes) / len(outcomes) < 0.25
    ]

    last_pnl = rows[-1].get("pnl") if rows else None
    if eq["mode"] in ("bank", "protect", "daily"):
        message = eq["reason"]
    elif mode == "cautious" and eq["mode"] == "harvest":
        message = eq["reason"]
    elif mode == "cautious":
        message = (
            f"Cautious mode — loss streak {loss_streak}, rolling WR {rolling_wr:.0%}"
            if rolling_wr
            else f"Cautious — loss streak {loss_streak}"
        )
    elif mode == "protect":
        message = f"Protect mode — realized PnL ${realized:.2f}"
    else:
        message = f"Learning active — realized PnL ${realized:.2f}"

    new_state = {
        "mode": mode,
        "realized_pnl": realized,
        "peak_pnl": peak,
        "recent_count": len(recent),
        "rolling_win_rate": round(rolling_wr, 4) if rolling_wr is not None else None,
        "loss_streak": loss_streak,
        "mc_floor": mc_floor,
        "min_ml_prob": min_ml_prob,
        "blocked_states": blocked_states,
        "blocked_hours": blocked_hours,
        "last_trade_pnl": last_pnl,
        "updates": _state.get("updates", 0) + 1,
        "message": message,
        "equity_mode": eq["mode"],
        "equity_reason": eq["reason"],
        "equity_allow": eq["allow"],
        "start_capital": start_cap,
        "day_pnl": day_pnl,
        "trades_today": trades_today,
        "progress_pct": eq.get("progress_pct"),
    }
    with _lock:
        _state = new_state
    return new_state


def clear_entry_blocks() -> dict[str, Any]:
    """Clear state/hour hard-blocks (keeps equity protect). Used when MTF swings were frozen."""
    global _state
    with _lock:
        _state["blocked_states"] = []
        _state["blocked_hours"] = []
        _state["message"] = "Entry blocks cleared — MTF swing scan active"
        return dict(_state)


def get_state() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def reset_state() -> dict[str, Any]:
    """Hard-reset adaptive guard after a desk / session wipe."""
    global _state
    with _lock:
        start = float(_rcget("start_capital_usd", _cfg("start_capital_usd", 1000.0)))
        _state = {
            "mode": "normal",
            "realized_pnl": 0.0,
            "peak_pnl": 0.0,
            "recent_count": 0,
            "rolling_win_rate": None,
            "loss_streak": 0,
            "mc_floor": float(_rcget("mc_floor_base", 42.0)),
            "min_ml_prob": float(_rcget("ml_prob_normal", 0.42)),
            "blocked_states": [],
            "blocked_hours": [],
            "last_trade_pnl": None,
            "updates": 0,
            "message": "Desk reset — awaiting closed trades",
            "equity_mode": "ok",
            "equity_reason": "Equity OK",
            "equity_allow": True,
            "start_capital": start,
        }
        return dict(_state)


def check(features: dict[str, Any]) -> tuple[bool, str]:
    """Adaptive block rules from latest closed-trade learning."""
    with _lock:
        st = dict(_state)

    if st.get("equity_allow") is False:
        return False, st.get("equity_reason") or "Equity protect — entries paused"

    # Swing / higher-TF independent setups must not inherit scalp-state blacklists
    exec_tf = str(features.get("exec_tf") or features.get("EXEC_TF") or "").lower()
    is_swing = exec_tf in ("15m", "1h", "4h") or bool(features.get("mtf_independent"))

    state = features.get("STATE")
    if state and state in st.get("blocked_states", []) and not is_swing:
        return False, f"Adaptive block — {state} underperforming in recent trades"

    hour = features.get("hour_utc")
    if hour is not None and int(hour) in st.get("blocked_hours", []) and not is_swing:
        return False, f"Adaptive block — hour {int(hour):02d} UTC weak in recent history"

    try:
        from config import MARKOV_MC_GATE_ENABLED
    except Exception:
        MARKOV_MC_GATE_ENABLED = False

    if st.get("mode") == "protect":
        if MARKOV_MC_GATE_ENABLED:
            mc = features.get("markov_mc_prob") or features.get("mc_prob")
            # Protect still prefers quality but must not freeze the desk — RL needs fills
            floor = float(st.get("mc_floor") or 38)
            if is_swing:
                floor = max(32.0, floor - 14.0)
            else:
                floor = max(34.0, floor - 12.0)
            # Float-safe compare + 3pp grace for live setups
            if mc is not None and float(mc) + 3.0 + 1e-6 < floor:
                return False, f"Protect mode — MC {float(mc):.0f}% below {floor:.0f}%"

        margin = features.get("score_margin")
        if margin is not None and margin < 0.25 and not is_swing:
            return False, "Protect mode — score margin too tight (<0.25)"

    if MARKOV_MC_GATE_ENABLED and st.get("equity_mode") == "harvest":
        mc = features.get("markov_mc_prob") or features.get("mc_prob")
        floor = max(36.0, float(st.get("mc_floor") or 42) - 8.0)
        if mc is not None and float(mc) + 2.0 + 1e-6 < floor:
            return False, f"Harvest mode — MC {float(mc):.0f}% below {floor:.0f}%"

    return True, "Adaptive OK"


def mc_floor() -> float:
    with _lock:
        return float(_state.get("mc_floor", 50.0))


def min_ml_prob() -> float:
    with _lock:
        return float(_state.get("min_ml_prob", 0.50))
