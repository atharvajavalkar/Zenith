"""Hyper-scalp — sub-minute momentum entries across the multi-market desk.

Designed for machine speed: ~1s cycle, larger lots, tight TP/SL, hard time-stop.
Not HFT; still MetaAPI-bound — but far faster than a human can punch.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _hyper_armed_allowed() -> bool:
    """Fund mode: Hyper overlay stays parked unless HYPER_SCALP_ARMED=true."""
    return os.getenv("HYPER_SCALP_ARMED", "false").lower() in ("1", "true", "yes")


def _start_standby() -> bool:
    return os.getenv("HYPER_SCALP_START_STANDBY", "true").lower() in ("1", "true", "yes")

_LOCK = threading.Lock()
_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "hyper_scalp_state.json"

# Bump when the strategy's tuning semantics change so persisted state migrates.
_HYPER_VER = 11

# Tuning fields reset to defaults on a version bump (session/risk fields kept)
_TUNING_KEYS = (
    "max_hold_sec", "cycle_sec", "max_open", "cooldown_sec", "target_tpm",
    "lot_mult", "ema_tf", "bias_tf", "risk_pct", "one_per_symbol",
    "time_stop_enabled", "disp_atr", "pull_max_atr", "tp_r", "tp1_r",
    "tp1_frac", "cost_mult", "min_range_bps", "swing_lookback", "min_rr",
)

_DEFAULT: dict[str, Any] = {
    "enabled": False,
    "ver": _HYPER_VER,
    "strategy": "zenith_momentum",  # ZENITH ENGINE on 1m — only OOS-green setup
    "lot_mult": 1.0,
    "time_stop_enabled": 0,
    "max_hold_sec": 1500,
    "cycle_sec": 1.0,
    "max_open": 2,
    "cooldown_sec": 45.0,
    "target_tpm": 4,
    "one_per_symbol": 1,
    "risk_pct": 10.0,
    "start_equity": None,
    "start_balance": None,
    "equity_floor": None,
    "last_equity": None,
    "last_balance": None,
    "last_floating": None,
    "realized_pnl": 0.0,
    "session_pnl": 0.0,
    "halted": False,
    "halt_reason": None,
    "ema_tf": "1m",
    "bias_tf": "5m",
    # Zenith momentum knobs — backtest-validated defaults; don't over-tune
    "disp_atr": 0.65,
    "pull_max_atr": 0.85,
    "tp_r": 1.5,
    "tp1_r": 1.0,
    "tp1_frac": 0.5,
    "cost_mult": 3.5,     # hard XM spread gate — TP must clear 3.5× round-trip cost
    "min_range_bps": 1.0,
    "swing_lookback": 10,
    "min_rr": 1.20,
    "punches": 0,
    "closes": 0,
    "last_punch": None,
    "last_error": None,
    "feed": [],
    "open": {},
}


def _load() -> dict[str, Any]:
    if not _STATE_PATH.exists():
        return dict(_DEFAULT)
    try:
        import json

        raw = json.loads(_STATE_PATH.read_text())
        raw = raw if isinstance(raw, dict) else {}
        out = dict(_DEFAULT)
        out.update(raw)
        # Migrate tuning to new defaults on a version bump (keep session/risk).
        # Read the *raw* saved version so a fresh _DEFAULT ver doesn't mask it.
        if int(raw.get("ver") or 0) < _HYPER_VER:
            for k in _TUNING_KEYS:
                out[k] = _DEFAULT[k]
            out["ver"] = _HYPER_VER
            out["strategy"] = _DEFAULT["strategy"]
            out["halted"] = False
            out["halt_reason"] = None
        # Always pin strategy label for this engine version
        out["strategy"] = _DEFAULT["strategy"]
        out["feed"] = list(out.get("feed") or [])[-40:]
        out["open"] = dict(out.get("open") or {})
        return out
    except Exception:
        return dict(_DEFAULT)


def _save(st: dict[str, Any]) -> None:
    try:
        import json

        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        slim = dict(st)
        slim["feed"] = list(slim.get("feed") or [])[-40:]
        _STATE_PATH.write_text(json.dumps(slim, indent=2, default=str))
    except Exception as exc:
        logger.debug("hyper_scalp save failed: %s", exc)


_state: dict[str, Any] = _load()
# Zenith-only desk: force Hyper into standby on process start unless explicitly armed.
if _start_standby() or not _hyper_armed_allowed():
    if _state.get("enabled"):
        _state["enabled"] = False
        _state["last_error"] = None
        _push_feed(_state, "STANDBY", "Hyper parked — Zenith-only core book")
        _save(_state)
_tick_hist: dict[str, deque] = {}
_last_punch_ts: dict[str, float] = {}
_punch_log: deque = deque(maxlen=400)  # timestamps of successful punches (for TPM)


def get_state() -> dict[str, Any]:
    with _LOCK:
        return dict(_state)


def _current_equity() -> Optional[float]:
    acct = _account_snapshot()
    if acct.get("equity") is not None:
        return float(acct["equity"])
    if acct.get("balance") is not None:
        return float(acct["balance"])
    return None


def _account_snapshot() -> dict[str, Optional[float]]:
    """Live XM balance / equity / floating (paper capital fallback)."""
    out: dict[str, Optional[float]] = {
        "balance": None,
        "equity": None,
        "floating": None,
    }
    try:
        import runtime_config
        from data.broker import broker_client

        if runtime_config.is_broker_mode() and broker_client.configured:
            info = broker_client.account_information()
            bal = info.get("balance")
            eq = info.get("equity")
            if bal is not None:
                out["balance"] = float(bal)
            if eq is not None:
                out["equity"] = float(eq)
            profit = info.get("profit")
            if profit is not None:
                out["floating"] = float(profit)
            elif out["balance"] is not None and out["equity"] is not None:
                out["floating"] = float(out["equity"]) - float(out["balance"])
            return out
    except Exception as exc:
        logger.debug("hyper account snapshot failed: %s", exc)
    try:
        from strategy.dynamic_lots import fetch_account_capital

        bal = float(fetch_account_capital())
        out["balance"] = bal
        out["equity"] = bal
        out["floating"] = 0.0
    except Exception:
        pass
    return out


def _estimate_trade_pnl_usd(meta: dict[str, Any], exit_price: Optional[float]) -> float:
    """Approximate closed PnL in account $ from entry/exit/lot."""
    try:
        from strategy.instrument_meta import contract_units_per_lot

        sym = str(meta.get("symbol") or "GOLD")
        entry = float(meta.get("entry") or 0)
        lot = float(meta.get("lot") or 0)
        direction = str(meta.get("signal") or "").upper()
        px = float(exit_price) if exit_price is not None else entry
        if entry <= 0 or lot <= 0:
            return 0.0
        units = float(contract_units_per_lot(sym)) * lot
        pts = (px - entry) if direction == "BUY" else (entry - px)
        return round(pts * units, 2)
    except Exception:
        return 0.0


def _rebaseline_from_account(acct: dict[str, Optional[float]], *, feed_kind: str = "RESET") -> None:
    """Zero Hyper PnL and pin risk baselines to the live XM snapshot."""
    eq = acct.get("equity")
    bal = acct.get("balance")
    _state["realized_pnl"] = 0.0
    _state["session_pnl"] = 0.0
    _state["last_floating"] = acct.get("floating")
    if bal is not None:
        _state["start_balance"] = round(float(bal), 2)
        _state["last_balance"] = round(float(bal), 2)
    if eq is not None:
        risk = float(_state.get("risk_pct") or 20.0)
        _state["start_equity"] = round(float(eq), 2)
        _state["last_equity"] = round(float(eq), 2)
        _state["equity_floor"] = round(float(eq) * (1.0 - risk / 100.0), 2)
        _push_feed(
            _state,
            feed_kind,
            (
                f"PnL zeroed · capital ${float(bal or eq):,.0f} · equity ${float(eq):,.0f} "
                f"· stop at ${_state['equity_floor']:,.0f} (−{risk:.0f}%)"
            ),
        )
    else:
        _state["start_equity"] = None
        _state["equity_floor"] = None
        _push_feed(_state, feed_kind, "PnL zeroed · no live equity yet")


def _apply_xm_cash_flow(acct: dict[str, Optional[float]]) -> None:
    """If XM balance jumps from deposit/withdrawal, shift risk baselines — never Session PnL.

    Session PnL is Hyper realized only (see snapshot). Cash transfers must not look like
    trading profit; they also must not instantly trip / inflate the capital stop.
    """
    bal = acct.get("balance")
    prev = _state.get("last_balance")
    if bal is None or prev is None:
        return
    delta = float(bal) - float(prev)
    if abs(delta) < 1.0:
        return
    # Closed-trade balance ticks are usually modest; deposits/withdrawals are large.
    thresh = max(75.0, abs(float(prev)) * 0.015)
    if abs(delta) < thresh:
        return
    if _state.get("start_equity") is not None:
        _state["start_equity"] = round(float(_state["start_equity"]) + delta, 2)
    if _state.get("start_balance") is not None:
        _state["start_balance"] = round(float(_state["start_balance"]) + delta, 2)
    start = _state.get("start_equity")
    if start is not None:
        risk = float(_state.get("risk_pct") or 10.0)
        _state["equity_floor"] = round(float(start) * (1.0 - risk / 100.0), 2)
    _push_feed(
        _state,
        "REBASE",
        f"XM cash flow {delta:+,.0f} · risk baseline shifted (Session PnL unchanged)",
    )


def reset_session_pnl() -> dict[str, Any]:
    """UI / API: zero Session + Realised and re-pin equity floor to live XM."""
    acct = _account_snapshot()
    with _LOCK:
        _rebaseline_from_account(acct, feed_kind="RESET")
        _save(_state)
        return dict(_state)


def set_enabled(enabled: bool) -> dict[str, Any]:
    with _LOCK:
        if enabled and not _hyper_armed_allowed():
            _state["enabled"] = False
            _state["last_error"] = (
                "Hyper parked (Zenith-only). Set HYPER_SCALP_ARMED=true to arm."
            )
            _push_feed(
                _state,
                "STANDBY",
                "Arm blocked — HYPER_SCALP_ARMED=false (Zenith-only desk)",
            )
            _save(_state)
            return dict(_state)
        _state["enabled"] = bool(enabled)
        _state["last_error"] = None
        if enabled:
            acct = _account_snapshot()
            _state["halted"] = False
            _state["halt_reason"] = None
            _rebaseline_from_account(acct, feed_kind="ARMED")
        else:
            _push_feed(_state, "STANDBY", "Hyper scalp on standby")
        _save(_state)
        return dict(_state)


def risk_guard_ok() -> bool:
    """True → safe to keep trading. On breach, auto-disarms and halts."""
    with _LOCK:
        if not _state.get("enabled"):
            return False
        floor = _state.get("equity_floor")
        start = _state.get("start_equity")
    acct = _account_snapshot()
    eq = acct.get("equity")
    if eq is None:
        eq = acct.get("balance")
    if eq is None and floor is None:
        return True
    with _LOCK:
        _apply_xm_cash_flow(acct)
        if acct.get("balance") is not None:
            _state["last_balance"] = round(float(acct["balance"]), 2)
        if eq is not None:
            _state["last_equity"] = round(float(eq), 2)
            _state["last_equity_ts"] = time.time()
        # Session = Hyper closed PnL only — XM deposits never inflate it
        _state["session_pnl"] = round(float(_state.get("realized_pnl") or 0.0), 2)
        if acct.get("floating") is not None:
            _state["last_floating"] = round(float(acct["floating"]), 2)
        floor = _state.get("equity_floor")
        start = _state.get("start_equity")
        _save(_state)
    if floor is None or eq is None:
        return True
    if float(eq) <= float(floor):
        with _LOCK:
            _state["enabled"] = False
            _state["halted"] = True
            dd = 0.0
            if start:
                dd = (float(start) - float(eq)) / float(start) * 100.0
            _state["halt_reason"] = (
                f"Risk stop hit — equity ${float(eq):,.0f} ≤ floor ${float(floor):,.0f} "
                f"(-{dd:.1f}% of ${float(start or 0):,.0f})"
            )
            _push_feed(_state, "HALT", _state["halt_reason"], equity=round(float(eq), 2))
            _save(_state)
        logger.warning("HYPER RISK STOP: %s", _state.get("halt_reason"))
        return False
    return True


def update_settings(**kwargs: Any) -> dict[str, Any]:
    int_keys = {"max_open", "target_tpm", "swing_lookback"}
    str_keys = {"ema_tf", "bias_tf"}
    with _LOCK:
        for key in (
            "lot_mult",
            "max_hold_sec",
            "cycle_sec",
            "max_open",
            "cooldown_sec",
            "target_tpm",
            "one_per_symbol",
            "time_stop_enabled",
            "risk_pct",
            "ema_tf",
            "bias_tf",
            "disp_atr",
            "pull_max_atr",
            "tp_r",
            "tp1_r",
            "tp1_frac",
            "cost_mult",
            "min_range_bps",
            "swing_lookback",
            "min_rr",
        ):
            if key in kwargs and kwargs[key] is not None:
                try:
                    if key in str_keys:
                        _state[key] = str(kwargs[key])
                    elif key in int_keys or key in ("one_per_symbol", "time_stop_enabled"):
                        _state[key] = int(kwargs[key])
                    else:
                        _state[key] = float(kwargs[key])
                except Exception:
                    pass
        if "risk_pct" in kwargs and kwargs["risk_pct"] is not None:
            start = _state.get("start_equity")
            if start:
                risk = float(_state.get("risk_pct") or 20.0)
                _state["equity_floor"] = round(float(start) * (1.0 - risk / 100.0), 2)
        _save(_state)
        return dict(_state)


def _push_feed(st: dict[str, Any], kind: str, detail: str, **extra: Any) -> None:
    row = {
        "ts": time.time(),
        "kind": kind,
        "detail": detail,
        **extra,
    }
    feed = list(st.get("feed") or [])
    feed.append(row)
    st["feed"] = feed[-40:]


def _bar_seconds() -> int:
    tf = str(_state.get("ema_tf") or "10s").lower().strip()
    if tf.endswith("s") and tf[:-1].isdigit():
        return max(1, int(tf[:-1]))
    return 10


def record_tick(symbol: str, price: float) -> None:
    s = str(symbol).upper()
    if s not in _tick_hist:
        _tick_hist[s] = deque(maxlen=40)
    _tick_hist[s].append((time.time(), float(price)))
    # Feed the live seconds-candle builder (only meaningful for '10s' style tf)
    try:
        from strategy.hyper_ema import feed_tick

        feed_tick(s, float(price), _bar_seconds())
    except Exception:
        pass


def _momentum(symbol: str) -> Optional[float]:
    hist = _tick_hist.get(str(symbol).upper())
    if not hist or len(hist) < 4:
        return None
    # ~last 1s vs ~4s ago (whatever samples we have)
    _, p_now = hist[-1]
    _, p_then = hist[0]
    if p_then <= 0:
        return None
    return (p_now - p_then) / p_then


def _atr_proxy(symbol: str, price: float) -> float:
    """Rough ATR from tick range; fallback to class default."""
    hist = _tick_hist.get(str(symbol).upper())
    if hist and len(hist) >= 5:
        prices = [p for _, p in hist]
        rng = max(prices) - min(prices)
        if rng > 0:
            return max(rng * 1.8, price * 0.00015)
    from strategy.instrument_meta import asset_class

    cls = asset_class(symbol)
    defaults = {
        "gold": 1.2,
        "silver": 0.04,
        "energy": 0.08,
        "index": 8.0,
        "fx": price * 0.0002,
        "crypto": price * 0.0015,
    }
    return float(defaults.get(cls, price * 0.0004))


def build_hyper_signal(symbol: str, price: float) -> Optional[dict[str, Any]]:
    """Build a hyper order from the ZENITH ENGINE on 1m (trend pullback).

    Kept entirely separate from strategy/signal.py (the Zenith desk itself);
    same setup, faster cadence, TP1 partial + breakeven managed by the worker.
    """
    with _LOCK:
        st = dict(_state)
    if not st.get("enabled"):
        return None

    record_tick(symbol, price)

    now = time.time()
    last = float(_last_punch_ts.get(symbol, 0))
    if now - last < float(st.get("cooldown_sec") or 45):
        return None

    with _LOCK:
        opens = dict(_state.get("open") or {})
        if len(opens) >= int(st.get("max_open") or 2):
            return None
        if int(st.get("one_per_symbol") or 1):
            if any(str(m.get("symbol") or "").upper() == str(symbol).upper() for m in opens.values()):
                return None

    from strategy.hyper_rl import overlay_for

    ov = overlay_for(symbol)
    if ov.get("block"):
        return None

    # Continuous candle walk-forward knobs (every new 1m) — history-safe
    candle_ov: dict[str, Any] = {}
    try:
        from analysis.candle_edge import hyper_overlays_for

        candle_ov = hyper_overlays_for(symbol) or {}
        if candle_ov.get("block"):
            return None
    except Exception:
        candle_ov = {}

    from strategy.hyper_momentum import blocks_correlated, evaluate_momentum

    with _LOCK:
        open_metas = list((_state.get("open") or {}).values())

    disp = float(candle_ov.get("disp_atr") or st.get("disp_atr", 0.65))
    pull = float(candle_ov.get("pull_max_atr") or st.get("pull_max_atr", 0.85))
    cost = float(candle_ov.get("cost_mult") or st.get("cost_mult", 3.5))
    tp_r = float(candle_ov.get("tp_r") or st.get("tp_r", 1.5))

    decision = evaluate_momentum(
        symbol,
        price=float(price),
        settings={
            "entry_tf": st.get("ema_tf", "1m"),
            "bias_tf": st.get("bias_tf", "5m"),
            "disp_atr": disp,
            "pull_max_atr": pull,
            "tp_r": tp_r,
            "tp1_r": st.get("tp1_r", 1.0),
            "tp1_frac": st.get("tp1_frac", 0.5),
            "cost_mult": cost,
            "min_range_bps": st.get("min_range_bps", 1.0),
            "swing_lookback": st.get("swing_lookback", 10),
        },
    )
    if not decision:
        return None

    corr_block = blocks_correlated(symbol, str(decision["signal"]), open_metas)
    if corr_block:
        return None

    direction = decision["signal"]
    entry = float(price)
    sl = float(decision["sl"])
    tp = float(decision["tp"])
    tp1_px = float(decision["tp1"])
    sl_d = abs(entry - sl)
    tp_d = abs(tp - entry)

    from strategy.instrument_meta import min_stop_distance, price_digits

    # Broker min-stop floor only
    hard_floor = min_stop_distance(symbol) * 1.05
    if sl_d < hard_floor:
        if direction == "BUY":
            sl = entry - hard_floor
        else:
            sl = entry + hard_floor
        sl_d = hard_floor
    if tp_d < hard_floor * float(st.get("min_rr") or 1.15):
        need = hard_floor * float(st.get("min_rr") or 1.15)
        if direction == "BUY":
            tp = entry + need
        else:
            tp = entry - need
        tp_d = need

    dig = price_digits(symbol)
    entry_r = round(entry, dig)
    sl_r = round(sl, dig)
    tp_r_px = round(tp, dig)
    tp1_r_px = round(tp1_px, dig)

    from strategy.dynamic_lots import sizing_snapshot

    snap = sizing_snapshot(
        symbol,
        entry=entry_r,
        sl=sl_r,
        tp=tp_r_px,
        rr=round(tp_d / max(sl_d, 1e-9), 2),
    )
    mult = float(st.get("lot_mult") or 1.0)
    # Candle-edge + Hyper RL lot scales (dampen oversize)
    try:
        mult *= float(candle_ov.get("lot_scale") or 1.0)
        mult *= float(ov.get("lot_scale") or 1.0)
    except Exception:
        pass
    mult = max(0.25, min(2.5, mult))
    lot = max(0.01, round(float(snap["lot_size"]) * mult, 2))

    tag = f"HS{int(now) % 100000}_{symbol[:4]}"
    return {
        "SIGNAL": direction,
        "INSTRUMENT": symbol,
        "SYMBOL": symbol,
        "ENTRY": entry_r,
        "SL": sl_r,
        "TP": tp_r_px,
        "TP1": tp1_r_px,
        "TP1_FRAC": float(decision.get("tp1_frac") or 0.5),
        "RR": round(tp_d / max(sl_d, 1e-9), 2),
        "EXEC_TF": str(decision.get("tf") or "1m"),
        "MTF_INDEPENDENT": True,
        "LOT_SIZE": lot,
        "desk_lot": float(snap["lot_size"]),
        "expected_profit_usd": snap.get("expected_profit_usd"),
        "MC_PROB": 55.0,
        "markov_mc_prob": 55.0,
        "CANDLE_TIME": tag,
        "HYPER_SCALP": True,
        "hyper_mode": decision.get("mode"),
        "bias": decision.get("bias"),
        "disp_atr": decision.get("disp_atr"),
        "pull_atr": decision.get("pull_atr"),
        "max_hold_sec": float(st.get("max_hold_sec") or 1500),
        "REASON": (
            f"ZENITH-1m {direction} disp={decision.get('disp_atr')}xATR "
            f"pull={decision.get('pull_atr')}xATR score={decision.get('score')} "
            f"edge/cost={decision.get('edge_vs_cost')} lot={lot}"
        ),
        "score": decision.get("score"),
        "edge_vs_cost": decision.get("edge_vs_cost"),
    }


def mark_punched(symbol: str, sig: dict[str, Any], result: dict[str, Any]) -> None:
    with _LOCK:
        _last_punch_ts[symbol] = time.time()
        if result.get("status") in ("ok", "filled", "executed", "open") or result.get("order_id") or result.get(
            "positionId"
        ):
            _state["punches"] = int(_state.get("punches") or 0) + 1
            tag = str(sig.get("CANDLE_TIME") or "")
            _state.setdefault("open", {})[tag] = {
                "symbol": symbol,
                "signal": sig.get("SIGNAL"),
                "entry": sig.get("ENTRY"),
                "sl": sig.get("SL"),
                "tp": sig.get("TP"),
                "tp1": sig.get("TP1"),
                "tp1_frac": sig.get("TP1_FRAC"),
                "partial_done": False,
                "be_moved": False,
                "lot": sig.get("LOT_SIZE"),
                "opened_at": time.time(),
                "max_hold_sec": sig.get("max_hold_sec"),
                "order_id": result.get("order_id") or result.get("positionId"),
                "mode": result.get("mode"),
                "risk_dist": abs(float(sig.get("ENTRY") or 0) - float(sig.get("SL") or 0)),
            }
            _state["last_punch"] = {
                "symbol": symbol,
                "signal": sig.get("SIGNAL"),
                "lot": sig.get("LOT_SIZE"),
                "ts": time.time(),
                "mode": sig.get("hyper_mode"),
                "dist_atr": sig.get("dist_atr"),
            }
            _punch_log.append(time.time())
            _push_feed(
                _state,
                "PUNCH",
                f"{symbol} {sig.get('SIGNAL')} {sig.get('LOT_SIZE')} lot",
                symbol=symbol,
                signal=sig.get("SIGNAL"),
                lot=sig.get("LOT_SIZE"),
            )
        else:
            _push_feed(
                _state,
                "SKIP",
                f"{symbol}: {result.get('message') or result.get('status')}",
                symbol=symbol,
            )
        _state["last_error"] = None if result.get("status") != "error" else result.get("message")
        _save(_state)


def open_positions() -> list[dict[str, Any]]:
    """All currently-open Hyper positions as [{tag, **meta}] (for exit checks)."""
    with _LOCK:
        opens = dict(_state.get("open") or {})
    return [{"tag": tag, **meta} for tag, meta in opens.items()]


def expired_opens(now: Optional[float] = None) -> list[dict[str, Any]]:
    """Hyper positions past their safety hold cap.

    With time_stop_enabled=0 (default) exits are TP/SL driven; this only returns
    positions that exceeded max_hold_sec, which acts purely as a safety net so a
    trade can never hang open forever.
    """
    t = float(now or time.time())
    with _LOCK:
        opens = dict(_state.get("open") or {})
        time_stop = int(_state.get("time_stop_enabled") or 0)
        hard_cap = float(_state.get("max_hold_sec") or 900)
    out = []
    for tag, meta in opens.items():
        opened = float(meta.get("opened_at") or 0)
        if not opened:
            continue
        hold = float(meta.get("max_hold_sec") or hard_cap)
        # If the forced timer is off, only fire at the hard safety cap.
        limit = hold if time_stop else max(hold, hard_cap)
        if t - opened >= limit:
            out.append({"tag": tag, **meta})
    return out


def clear_open(
    tag: str,
    *,
    reason: str = "time_stop",
    exit_price: Optional[float] = None,
) -> None:
    with _LOCK:
        opens = dict(_state.get("open") or {})
        meta = opens.pop(tag, None)
        _state["open"] = opens
        pnl_usd = 0.0
        if meta:
            pnl_usd = _estimate_trade_pnl_usd(meta, exit_price)
            _state["closes"] = int(_state.get("closes") or 0) + 1
            _state["realized_pnl"] = round(
                float(_state.get("realized_pnl") or 0.0) + pnl_usd, 2
            )
            _state["session_pnl"] = round(float(_state["realized_pnl"]), 2)
            sign = "+" if pnl_usd >= 0 else ""
            _push_feed(
                _state,
                "EXIT",
                f"{meta.get('symbol')} {reason} {sign}${pnl_usd:.2f}",
                symbol=meta.get("symbol"),
                reason=reason,
                pnl=pnl_usd,
            )
        _save(_state)

    # Feed the hyper RL with the realized R (outside the state lock)
    if not meta:
        return
    try:
        from strategy.hyper_rl import record_outcome

        entry = float(meta.get("entry") or 0)
        risk = float(meta.get("risk_dist") or 0)
        direction = str(meta.get("signal") or "").upper()
        px = float(exit_price) if exit_price is not None else entry
        if risk <= 0:
            r = 0.0
        else:
            pnl_pts = (px - entry) if direction == "BUY" else (entry - px)
            r = pnl_pts / risk
        record_outcome(str(meta.get("symbol") or ""), r=r, reason=reason)
    except Exception as exc:
        logger.debug("hyper RL record failed: %s", exc)


def mark_partial(tag: str, *, exit_price: float, closed_lot: float) -> None:
    """Book the TP1 partial: realize PnL on the closed slice, move SL to BE.

    Keeps the remaining lot running to the full TP with a risk-free stop —
    same partial + breakeven discipline the Zenith desk uses.
    """
    with _LOCK:
        opens = dict(_state.get("open") or {})
        meta = opens.get(tag)
        if not meta or meta.get("partial_done"):
            return
        part_meta = dict(meta)
        part_meta["lot"] = float(closed_lot)
        pnl_usd = _estimate_trade_pnl_usd(part_meta, exit_price)
        remaining = max(0.0, round(float(meta.get("lot") or 0) - float(closed_lot), 2))
        meta["lot"] = remaining
        meta["partial_done"] = True
        meta["be_moved"] = True
        meta["sl"] = meta.get("entry")  # breakeven — worst case is scratch on the runner
        opens[tag] = meta
        _state["open"] = opens
        _state["realized_pnl"] = round(float(_state.get("realized_pnl") or 0.0) + pnl_usd, 2)
        _state["session_pnl"] = round(float(_state["realized_pnl"]), 2)
        sign = "+" if pnl_usd >= 0 else ""
        _push_feed(
            _state,
            "TP1",
            f"{meta.get('symbol')} partial {closed_lot} lot {sign}${pnl_usd:.2f} · SL → BE",
            symbol=meta.get("symbol"),
            pnl=pnl_usd,
        )
        _save(_state)


def snapshot() -> dict[str, Any]:
    # Refresh live capital/equity whenever the UI polls (even in standby)
    acct = _account_snapshot()
    with _LOCK:
        _apply_xm_cash_flow(acct)
        if acct.get("balance") is not None:
            _state["last_balance"] = round(float(acct["balance"]), 2)
        if acct.get("equity") is not None:
            _state["last_equity"] = round(float(acct["equity"]), 2)
            _state["last_equity_ts"] = time.time()
        # Hyper Session PnL = closed Hyper trades only (not XM equity − start)
        _state["session_pnl"] = round(float(_state.get("realized_pnl") or 0.0), 2)
        if acct.get("floating") is not None:
            _state["last_floating"] = round(float(acct["floating"]), 2)
        st = dict(_state)
    st["tick_symbols"] = {k: len(v) for k, v in _tick_hist.items()}
    st["open_count"] = len(st.get("open") or {})
    # Live trades-per-minute from the dedicated punch log (last 60s)
    now = time.time()
    st["tpm"] = sum(1 for t in list(_punch_log) if now - float(t) <= 60.0)
    # Aliases for the UI
    st["capital"] = st.get("last_balance")
    st["equity"] = st.get("last_equity")
    st["floating_pnl"] = st.get("last_floating")
    start = st.get("start_equity")
    eq = st.get("last_equity")
    if start and eq is not None:
        st["drawdown_pct"] = round((float(start) - float(eq)) / float(start) * 100.0, 2)
        st["risk_used_pct"] = round(
            max(0.0, (float(start) - float(eq))) / float(start) * 100.0, 2
        )
    else:
        st["drawdown_pct"] = None
        st["risk_used_pct"] = None
    try:
        from strategy.hyper_rl import snapshot as rl_snap

        st["rl"] = rl_snap()
    except Exception:
        st["rl"] = {"symbols": []}
    try:
        from analysis.candle_edge import snapshot as edge_snap

        st["candle_edge"] = edge_snap()
    except Exception:
        st["candle_edge"] = {"symbols": []}
    try:
        from strategy.hyper_vp import snapshot as vp_snap

        st["micro_vp"] = vp_snap()
    except Exception:
        st["micro_vp"] = {"profiles": []}
    try:
        from strategy.hyper_flow import bias_snapshot
        from strategy.hyper_momentum import board_snapshot as momentum_snap

        st["bias_5m"] = bias_snapshot()
        board = momentum_snap()
        st["momentum"] = board
        st["flow"] = {
            "flows": [
                {
                    "symbol": r.get("symbol"),
                    "status": r.get("status"),
                    "signal": r.get("signal"),
                    "bias_5m": r.get("bias"),
                    "disp_atr": r.get("disp_atr"),
                    "pull_atr": r.get("pull_atr"),
                    "skip": r.get("skip"),
                    "score": r.get("score"),
                    "edge_vs_cost": r.get("edge_vs_cost"),
                    "updated_at": r.get("updated_at"),
                }
                for r in (board.get("symbols") or [])
            ]
        }
    except Exception:
        st["bias_5m"] = {"symbols": []}
        st["momentum"] = {"symbols": []}
        st["flow"] = {"flows": []}
    return st
