"""Hyper-scalp reinforcement learner — per-symbol, online, fully separate.

Learns from the outcomes of hyper trades (time-stop / TP / SL exits) and tunes
the EMA-scalp knobs per instrument. Grounded in common EMA-scalp practice:

  • Reward = expectancy (mean R) of the last N trades  (quant-signals / AMRC).
  • Circuit breaker — cut size + cool the symbol after a losing streak.
  • Regime awareness — widen the flat/braided filter when a symbol keeps losing
    (it's chopping), tighten it when it's winning (clean trend).
  • Exploration ~5% so it keeps probing instead of freezing a symbol out forever.

Never touches analysis/instrument_rl.py (the Zenith desk RL).
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "hyper_rl_state.json"

# Tuning bounds
_LOT_SCALE = (0.30, 1.60)
_NEAR = (0.30, 1.00)
_FLAT = (0.06, 0.60)   # min EMA ribbon spread (in ATR) to allow a trade
_TP = (0.25, 1.20)
_SL = (0.25, 1.20)

_RECENT_MAX = 50
_LOSS_STREAK_TRIP = 3     # consecutive losses → circuit breaker
_COOLDOWN_SEC = 40.0      # block a tripped symbol this long
_EXPLORE = 0.05           # 5% of the time, allow a blocked symbol through

_DEFAULT_SYM = {
    "n": 0,
    "wins": 0,
    "losses": 0,
    "pnl_r_sum": 0.0,
    "streak": 0,          # >0 win streak, <0 loss streak
    "lot_scale": 1.0,
    "near_atr": 0.55,
    "flat_atr": 0.12,
    "tp_atr": 0.45,
    "sl_atr": 0.40,
    "blocked_until": 0.0,
    "expectancy_r": 0.0,
    "win_rate": 0.0,
    "last_r": 0.0,
    "updated": 0.0,
}


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _load() -> dict[str, Any]:
    if not _STATE_PATH.exists():
        return {"symbols": {}}
    try:
        raw = json.loads(_STATE_PATH.read_text())
        if not isinstance(raw, dict):
            return {"symbols": {}}
        raw.setdefault("symbols", {})
        return raw
    except Exception:
        return {"symbols": {}}


def _save(state: dict[str, Any]) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
    except Exception as exc:
        logger.debug("hyper_rl save failed: %s", exc)


_state: dict[str, Any] = _load()
_recent: dict[str, deque] = {}


def _sym_state(symbol: str) -> dict[str, Any]:
    s = str(symbol).upper()
    syms = _state.setdefault("symbols", {})
    if s not in syms:
        syms[s] = dict(_DEFAULT_SYM)
    else:
        # backfill any new keys
        for k, v in _DEFAULT_SYM.items():
            syms[s].setdefault(k, v)
    return syms[s]


def overlay_for(symbol: str) -> dict[str, Any]:
    """Return per-symbol tuning + a `block` flag (circuit breaker)."""
    with _LOCK:
        st = dict(_sym_state(symbol))
    now = time.time()
    blocked = now < float(st.get("blocked_until") or 0)
    if blocked and random.random() < _EXPLORE:
        blocked = False  # occasional exploration probe
    return {
        "symbol": str(symbol).upper(),
        "block": blocked,
        "lot_scale": _clip(st.get("lot_scale", 1.0), *_LOT_SCALE),
        "near_atr": _clip(st.get("near_atr", 0.55), *_NEAR),
        "flat_atr": _clip(st.get("flat_atr", 0.12), *_FLAT),
        "tp_atr": _clip(st.get("tp_atr", 0.45), *_TP),
        "sl_atr": _clip(st.get("sl_atr", 0.40), *_SL),
        "expectancy_r": round(float(st.get("expectancy_r") or 0.0), 3),
        "win_rate": round(float(st.get("win_rate") or 0.0), 3),
        "streak": int(st.get("streak") or 0),
        "n": int(st.get("n") or 0),
    }


def record_outcome(
    symbol: str,
    *,
    r: float,
    pnl_usd: Optional[float] = None,
    reason: str = "",
) -> dict[str, Any]:
    """Feed a closed hyper trade. `r` = PnL in R multiples (reward/risk units)."""
    s = str(symbol).upper()
    with _LOCK:
        st = _sym_state(s)
        dq = _recent.get(s)
        if dq is None:
            dq = deque(maxlen=_RECENT_MAX)
            _recent[s] = dq
        r = float(r)
        dq.append(r)

        st["n"] = int(st.get("n") or 0) + 1
        st["pnl_r_sum"] = float(st.get("pnl_r_sum") or 0.0) + r
        st["last_r"] = round(r, 3)
        win = r > 0
        if win:
            st["wins"] = int(st.get("wins") or 0) + 1
            st["streak"] = max(1, int(st.get("streak") or 0) + 1)
        else:
            st["losses"] = int(st.get("losses") or 0) + 1
            st["streak"] = min(-1, int(st.get("streak") or 0) - 1)

        recent = list(dq)
        st["expectancy_r"] = round(sum(recent) / len(recent), 4)
        wr = sum(1 for x in recent if x > 0) / len(recent)
        st["win_rate"] = round(wr, 4)
        streak = int(st["streak"])
        exp = float(st["expectancy_r"])

        # ── Circuit breaker: losing streak → cut size + cool + get pickier ──
        if streak <= -_LOSS_STREAK_TRIP:
            st["blocked_until"] = time.time() + _COOLDOWN_SEC
            st["lot_scale"] = _clip(float(st["lot_scale"]) * 0.75, *_LOT_SCALE)
            st["flat_atr"] = _clip(float(st["flat_atr"]) + 0.05, *_FLAT)
            st["near_atr"] = _clip(float(st["near_atr"]) - 0.05, *_NEAR)  # tighter pullback
        # ── Confidence: winning + positive expectancy → scale up, loosen ──
        elif streak >= 3 and exp > 0:
            st["lot_scale"] = _clip(float(st["lot_scale"]) * 1.08, *_LOT_SCALE)
            st["flat_atr"] = _clip(float(st["flat_atr"]) - 0.02, *_FLAT)

        # ── Exit tuning from realized win-rate (needs a small sample) ──
        if len(recent) >= 12:
            if wr < 0.40:
                # too many stop-outs → give a touch more room, take profit sooner
                st["sl_atr"] = _clip(float(st["sl_atr"]) + 0.03, *_SL)
                st["tp_atr"] = _clip(float(st["tp_atr"]) - 0.02, *_TP)
            elif wr > 0.60 and exp > 0:
                # winning cleanly → let TP breathe a bit
                st["tp_atr"] = _clip(float(st["tp_atr"]) + 0.03, *_TP)

        st["updated"] = time.time()
        _save(_state)
        out = dict(st)
    logger.info(
        "HYPER RL %s r=%+.2f streak=%d wr=%.0f%% E[R]=%+.2f lot×%.2f%s",
        s, r, out["streak"], out["win_rate"] * 100, out["expectancy_r"],
        out["lot_scale"], " BLOCK" if r <= 0 and out["streak"] <= -_LOSS_STREAK_TRIP else "",
    )
    return out


def snapshot() -> dict[str, Any]:
    now = time.time()
    with _LOCK:
        syms = {k: dict(v) for k, v in _state.get("symbols", {}).items()}
    rows = []
    for s, st in syms.items():
        if not int(st.get("n") or 0):
            continue
        rows.append(
            {
                "symbol": s,
                "n": int(st.get("n") or 0),
                "win_rate": round(float(st.get("win_rate") or 0) * 100, 1),
                "expectancy_r": round(float(st.get("expectancy_r") or 0), 3),
                "streak": int(st.get("streak") or 0),
                "lot_scale": round(float(st.get("lot_scale") or 1), 2),
                "blocked": now < float(st.get("blocked_until") or 0),
                "last_r": round(float(st.get("last_r") or 0), 2),
            }
        )
    rows.sort(key=lambda r: r["n"], reverse=True)
    return {"symbols": rows}
