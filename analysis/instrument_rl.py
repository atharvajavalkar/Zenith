"""Per-instrument reinforcement learning — contextual bandit over algo profiles.

Each symbol (GOLD, EURUSD, GBPUSD, USDJPY, BTCUSD, …) keeps its own:
  • closed-trade reward history (R-multiple + USD)
  • discrete policy arms: tight / balanced / aggressive
  • continuous knobs that overlay adaptive MC/ML floors and lot aggression

Reward on close: R = pnl / risk_usd (fallback from SL distance × contract).
Arms selected via UCB1 so under-explored profiles still get trials.
Policy persists to data/instrument_policies.json and is rebuilt from DB on boot.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Optional

from config import INSTRUMENT_UNIVERSE

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PATH = Path(__file__).resolve().parents[1] / "data" / "instrument_policies.json"

ARMS = ("tight", "balanced", "aggressive")

# Profile → algo knobs (tight after losses: smaller size + better RR — not unreachable MC)
ARM_POLICY: dict[str, dict[str, float]] = {
    "tight": {
        "mc_floor_delta": 1.0,   # learn mode — barely lift MC
        "min_ml_delta": 0.02,
        "aggression": 0.28,
        "risk_scale": 0.75,
        "rr_prefer": 3.0,
        "partial_boost": 1.25,
        "trail_delta": 0.10,
    },
    "balanced": {
        "mc_floor_delta": 0.0,
        "min_ml_delta": 0.0,
        "aggression": 0.55,
        "risk_scale": 1.0,
        "rr_prefer": 3.0,        # multi-instrument default: stretch beyond 1:2
        "partial_boost": 1.0,
        "trail_delta": 0.0,
    },
    "aggressive": {
        "mc_floor_delta": -2.0,
        "min_ml_delta": -0.02,
        "aggression": 0.78,
        "risk_scale": 1.08,
        "rr_prefer": 2.0,
        "partial_boost": 0.90,
        "trail_delta": -0.05,
    },
}

_UCB_C = 0.85  # less explore-while-bleeding
_MIN_TRADES_EXPLOIT = 3
_R_CLAMP = 5.0


def _blank_arm() -> dict[str, float]:
    return {"n": 0, "sum_r": 0.0, "sum_pnl": 0.0}


def _blank_symbol(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "n_trades": 0,
        "wins": 0,
        "losses": 0,
        "be": 0,
        "sum_r": 0.0,
        "sum_pnl": 0.0,
        "expectancy_r": 0.0,
        "win_rate": None,
        "loss_streak": 0,
        "active_arm": "balanced",
        "arms": {a: _blank_arm() for a in ARMS},
        "policy": dict(ARM_POLICY["balanced"]),
        "last_reward_r": None,
        "last_update": None,
        "message": "Awaiting closed trades",
    }


_state: dict[str, Any] = {
    "updated_at": None,
    "instruments": {s: _blank_symbol(s) for s in INSTRUMENT_UNIVERSE},
    "total_updates": 0,
}


def _ensure_symbol(sym: str) -> dict[str, Any]:
    s = str(sym or "GOLD").upper().strip()
    if s not in _state["instruments"]:
        _state["instruments"][s] = _blank_symbol(s)
    return _state["instruments"][s]


def _load_disk() -> None:
    global _state
    try:
        if not _PATH.exists():
            return
        raw = json.loads(_PATH.read_text(encoding="utf-8"))
        instruments = raw.get("instruments") or {}
        merged = {s: _blank_symbol(s) for s in INSTRUMENT_UNIVERSE}
        for k, v in instruments.items():
            row = _blank_symbol(str(k).upper())
            row.update(v or {})
            if "arms" not in row or not isinstance(row["arms"], dict):
                row["arms"] = {a: _blank_arm() for a in ARMS}
            for a in ARMS:
                row["arms"].setdefault(a, _blank_arm())
            row["policy"] = dict(ARM_POLICY.get(row.get("active_arm") or "balanced", ARM_POLICY["balanced"]))
            # keep continuous nudges if present
            if isinstance(v.get("policy"), dict):
                row["policy"].update({kk: float(vv) for kk, vv in v["policy"].items() if kk in row["policy"]})
            merged[str(k).upper()] = row
        _state = {
            "updated_at": raw.get("updated_at"),
            "instruments": merged,
            "total_updates": int(raw.get("total_updates") or 0),
        }
    except Exception as exc:
        logger.warning("Instrument RL load failed: %s", exc)


def _save_disk() -> None:
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": _state.get("updated_at"),
            "total_updates": _state.get("total_updates"),
            "instruments": _state.get("instruments"),
        }
        _PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Instrument RL save failed: %s", exc)


def _trade_reward(row: dict) -> tuple[float, float]:
    """Return (R-multiple, pnl_usd) — clamped so FX/crypto bugs can't poison UCB."""
    from analysis.trade_outcome import is_breakeven_trade, is_winning_trade
    from strategy.position_sizing import contract_units_per_lot

    pnl = float(row.get("pnl") or 0.0)
    entry = row.get("entry")
    sl = row.get("sl")
    lot = float(row.get("lot_size") or 0.1)
    sym = str(row.get("symbol") or "GOLD").upper()

    feats = row.get("features") or {}
    if isinstance(feats, str):
        try:
            feats = json.loads(feats)
        except Exception:
            feats = {}

    risk_usd = None
    if isinstance(feats, dict) and feats.get("risk_usd"):
        try:
            risk_usd = float(feats["risk_usd"])
        except Exception:
            risk_usd = None
    if (risk_usd is None or risk_usd < 1.0) and entry is not None and sl is not None:
        risk_pts = abs(float(entry) - float(sl))
        # Ignore near-zero / inverted stops that make R explode
        if risk_pts > 1e-6:
            risk_usd = risk_pts * contract_units_per_lot(sym) * max(lot, 0.01)

    if risk_usd and risk_usd >= 5.0:
        r_mult = pnl / risk_usd
    elif is_breakeven_trade(row):
        r_mult = 0.0
    elif is_winning_trade(row) or pnl > 0:
        r_mult = 1.5 if pnl >= 0 else -1.0
    else:
        r_mult = -1.0

    r_mult = max(-_R_CLAMP, min(_R_CLAMP, float(r_mult)))
    return r_mult, float(pnl)


def _select_arm(row: dict[str, Any]) -> str:
    """UCB1 with hard protect: loss streaks / red WR → tight."""
    losses = int(row.get("losses") or 0)
    wins = int(row.get("wins") or 0)
    streak = int(row.get("loss_streak") or 0)
    decisive = wins + losses
    wr = (wins / decisive) if decisive else None
    exp_r = float(row.get("expectancy_r") or 0.0)

    # After bleeding — stop exploring aggressive
    if streak >= 2 or (decisive >= 3 and wr is not None and wr < 0.35) or exp_r < -0.35:
        return "tight"
    if decisive >= 5 and wr is not None and wr < 0.45 and exp_r < 0:
        return "tight"

    arms = row["arms"]
    total_n = sum(int(arms[a]["n"]) for a in ARMS) or 1
    best_a = "balanced"
    best_score = -1e18
    for a in ARMS:
        # Don't cold-start into aggressive while already losing
        if a == "aggressive" and decisive >= 2 and (wr or 1) < 0.5:
            continue
        n = int(arms[a]["n"])
        if n <= 0:
            if a == "tight" and decisive >= 1 and (wr or 1) < 0.5:
                return "tight"
            if a != "aggressive":
                return a
            continue
        mean = float(arms[a]["sum_r"]) / n
        bonus = _UCB_C * math.sqrt(math.log(total_n + 1) / n)
        score = mean + bonus
        if score > best_score:
            best_score = score
            best_a = a
    return best_a


def _apply_arm_policy(row: dict[str, Any]) -> None:
    arm = row.get("active_arm") or "balanced"
    # Bleeding symbols never keep an aggressive-easing base
    losses = int(row.get("losses") or 0)
    wins = int(row.get("wins") or 0)
    streak = int(row.get("loss_streak") or 0)
    decisive = wins + losses
    wr = (wins / decisive) if decisive else None
    exp_r = float(row.get("expectancy_r") or 0.0)
    if streak >= 2 or (decisive >= 3 and wr is not None and wr < 0.40):
        arm = "tight"
        row["active_arm"] = "tight"

    base = dict(ARM_POLICY.get(arm, ARM_POLICY["balanced"]))
    n = int(row.get("n_trades") or 0)

    if n >= _MIN_TRADES_EXPLOIT and wr is not None:
        if exp_r < -0.10 or wr < 0.45 or streak >= 2:
            # Quality via size + RR preference — keep MC lift small so setups still pass
            bump = 0.5 + min(0.5, float(streak) * 0.25)
            base["mc_floor_delta"] = min(1.0, max(0.0, base["mc_floor_delta"] + bump))
            base["min_ml_delta"] = min(0.02, base["min_ml_delta"] + 0.005 * min(streak, 3))
            base["aggression"] = max(0.18, base["aggression"] - 0.12)
            base["risk_scale"] = max(0.60, base["risk_scale"] - 0.12)
            base["rr_prefer"] = 4.0 if streak >= 3 or wr < 0.30 else 3.0
            base["partial_boost"] = min(1.45, base["partial_boost"] + 0.15)
            base["trail_delta"] = min(0.20, base["trail_delta"] + 0.08)
            row["message"] = (
                f"tight learn · WR {wr:.0%} · E[R]={exp_r:+.2f} · "
                f"streak {streak} · size×{base['risk_scale']:.2f} · prefer 1:{int(base['rr_prefer'])}"
            )
        elif exp_r > 0.35 and wr >= 0.55 and streak == 0:
            base["mc_floor_delta"] = max(-6.0, base["mc_floor_delta"] - 2.0)
            base["aggression"] = min(0.88, base["aggression"] + 0.05)
            base["risk_scale"] = min(1.15, base["risk_scale"] + 0.04)
            base["rr_prefer"] = max(2.0, base["rr_prefer"] - 0)  # keep ≥2; still allow 3/4 via EV
            row["message"] = f"{arm} ease · WR {wr:.0%} · E[R]={exp_r:+.2f}"
        else:
            # Mild default: still prefer ≥1:3 so we take runners + partials
            base["rr_prefer"] = max(3.0, float(base.get("rr_prefer") or 3.0))
            row["message"] = (
                f"{arm} · WR {wr:.0%} · E[R]={exp_r:+.2f} · prefer 1:{int(base['rr_prefer'])}"
            )
    else:
        base["rr_prefer"] = max(3.0, float(base.get("rr_prefer") or 3.0))
        row["message"] = f"{arm} · collecting ({n}) · prefer 1:{int(base['rr_prefer'])}"

    row["policy"] = base


def rebuild_from_trades(
    closed_rows: list[dict], *, ignore_epoch: bool = False
) -> dict[str, Any]:
    """Full rebuild from closed trades.

    ignore_epoch=False (default): only learn from closes after desk reset
    (new-brain session). Pass True only for research that needs full history.
    """
    from analysis.trade_outcome import is_breakeven_trade, is_losing_trade, is_winning_trade

    try:
        from app.services.desk_epoch import row_after_epoch
    except Exception:

        def row_after_epoch(*_a, **_k):  # type: ignore
            return True

    with _LOCK:
        _load_disk()  # keep structure
        fresh = {s: _blank_symbol(s) for s in INSTRUMENT_UNIVERSE}
        # Chronological so UCB sequence makes sense
        ordered = sorted(
            closed_rows,
            key=lambda r: str(r.get("created_at") or r.get("log_time") or r.get("id") or ""),
        )
        updates = 0
        for r in ordered:
            if r.get("status") != "closed" or r.get("signal") not in ("BUY", "SELL"):
                continue
            if not ignore_epoch and not row_after_epoch(r.get("created_at"), r.get("log_time")):
                continue
            # Skip ghost planted rows / zero risk junk
            if abs(float(r.get("pnl") or 0)) < 0.01 and not r.get("oanda_transaction_id"):
                continue
            sym = str(r.get("symbol") or "GOLD").upper().strip()
            if sym not in fresh:
                fresh[sym] = _blank_symbol(sym)
            row = fresh[sym]
            r_mult, pnl = _trade_reward(r)

            # Credit the arm that was active when trade was opened if tagged, else current
            feats = r.get("features") or {}
            if isinstance(feats, str):
                try:
                    feats = json.loads(feats)
                except Exception:
                    feats = {}
            arm = str((feats or {}).get("rl_arm") or row.get("active_arm") or "balanced")
            if arm not in ARMS:
                arm = "balanced"

            row["arms"][arm]["n"] = int(row["arms"][arm]["n"]) + 1
            row["arms"][arm]["sum_r"] = float(row["arms"][arm]["sum_r"]) + r_mult
            row["arms"][arm]["sum_pnl"] = float(row["arms"][arm]["sum_pnl"]) + pnl

            row["n_trades"] += 1
            row["sum_r"] += r_mult
            row["sum_pnl"] += pnl
            row["last_reward_r"] = round(r_mult, 3)
            if is_breakeven_trade(r):
                row["be"] += 1
                # BE doesn't break a loss streak but doesn't grow it
            elif is_winning_trade(r):
                row["wins"] += 1
                row["loss_streak"] = 0
            elif is_losing_trade(r):
                row["losses"] += 1
                row["loss_streak"] = int(row.get("loss_streak") or 0) + 1

            decisive = row["wins"] + row["losses"]
            row["win_rate"] = round(row["wins"] / decisive, 4) if decisive else None
            row["expectancy_r"] = round(row["sum_r"] / row["n_trades"], 4) if row["n_trades"] else 0.0

            # Re-select arm after credit
            row["active_arm"] = _select_arm(row)
            _apply_arm_policy(row)
            updates += 1

        # Ensure universe keys exist + apply policy even with 0 trades
        for s in INSTRUMENT_UNIVERSE:
            fresh.setdefault(s, _blank_symbol(s))
            if fresh[s]["n_trades"] == 0:
                fresh[s]["active_arm"] = "balanced"
                _apply_arm_policy(fresh[s])

        _state["instruments"] = fresh
        _state["total_updates"] = updates
        _state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _save_disk()
        # Snapshot under lock — do not call get_state() here (would deadlock).
        return {
            "updated_at": _state.get("updated_at"),
            "total_updates": _state.get("total_updates"),
            "instruments": {k: dict(v) for k, v in _state["instruments"].items()},
        }


def on_trade_closed(row: dict) -> dict[str, Any]:
    """Online update for a single close (called from learning loop)."""
    with _LOCK:
        if not _state.get("instruments"):
            _load_disk()
        sym = str(row.get("symbol") or "GOLD").upper().strip()
        instrument = _ensure_symbol(sym)
        r_mult, pnl = _trade_reward(row)

        feats = row.get("features") or {}
        if isinstance(feats, str):
            try:
                feats = json.loads(feats)
            except Exception:
                feats = {}
        arm = str((feats or {}).get("rl_arm") or instrument.get("active_arm") or "balanced")
        if arm not in ARMS:
            arm = "balanced"

        instrument["arms"][arm]["n"] = int(instrument["arms"][arm]["n"]) + 1
        instrument["arms"][arm]["sum_r"] += r_mult
        instrument["arms"][arm]["sum_pnl"] += pnl
        instrument["n_trades"] += 1
        instrument["sum_r"] += r_mult
        instrument["sum_pnl"] += pnl
        instrument["last_reward_r"] = round(r_mult, 3)

        from analysis.trade_outcome import is_breakeven_trade, is_losing_trade, is_winning_trade

        if is_breakeven_trade(row):
            instrument["be"] += 1
        elif is_winning_trade(row):
            instrument["wins"] += 1
            instrument["loss_streak"] = 0
        elif is_losing_trade(row):
            instrument["losses"] += 1
            instrument["loss_streak"] = int(instrument.get("loss_streak") or 0) + 1

        decisive = instrument["wins"] + instrument["losses"]
        instrument["win_rate"] = (
            round(instrument["wins"] / decisive, 4) if decisive else None
        )
        instrument["expectancy_r"] = round(
            instrument["sum_r"] / instrument["n_trades"], 4
        )
        instrument["active_arm"] = _select_arm(instrument)
        _apply_arm_policy(instrument)
        instrument["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _state["total_updates"] = int(_state.get("total_updates") or 0) + 1
        _state["updated_at"] = instrument["last_update"]
        _save_disk()
        logger.info(
            "RL %s: reward=%+.2fR arm→%s agg=%.2f mcΔ=%+.0f (%s)",
            sym,
            r_mult,
            instrument["active_arm"],
            instrument["policy"]["aggression"],
            instrument["policy"]["mc_floor_delta"],
            instrument["message"],
        )
        return dict(instrument)


def get_policy(symbol: Optional[str] = None) -> dict[str, float]:
    with _LOCK:
        if not _state.get("instruments"):
            _load_disk()
        if not symbol:
            return dict(ARM_POLICY["balanced"])
        row = _ensure_symbol(symbol)
        return dict(row.get("policy") or ARM_POLICY["balanced"])


def get_active_arm(symbol: Optional[str] = None) -> str:
    with _LOCK:
        if not _state.get("instruments"):
            _load_disk()
        if not symbol:
            return "balanced"
        return str(_ensure_symbol(symbol).get("active_arm") or "balanced")


def overlays_for(symbol: Optional[str] = None) -> dict[str, Any]:
    """Consumable overlays for gates + sizing (board + minute candle + corr)."""
    pol = get_policy(symbol)
    mc_delta = float(pol.get("mc_floor_delta") or 0)
    ml_delta = float(pol.get("min_ml_delta") or 0)
    aggression = float(pol.get("aggression") or 0.55)
    board = {}
    minute = {}
    if symbol:
        try:
            from analysis.opportunity_learn import board_soft_deltas

            board = board_soft_deltas(str(symbol))
            mc_delta += float(board.get("mc_floor_delta") or 0)
            aggression = max(
                0.15, min(0.95, aggression + float(board.get("aggression") or 0))
            )
        except Exception:
            board = {}
        try:
            from analysis.minute_learn import minute_overlays_for

            minute = minute_overlays_for(str(symbol))
            mc_delta += float(minute.get("mc_floor_delta") or 0)
            ml_delta += float(minute.get("min_ml_delta") or 0)
            aggression = max(
                0.15, min(0.95, aggression + float(minute.get("aggression") or 0))
            )
        except Exception:
            minute = {}

    # Candle retrospection (live-brain paper) — soft only; never wipes trade RL
    # Hard budget so minute+candle cannot dominate closed-trade arm policy.
    candle: dict[str, Any] = {}
    risk_scale = float(pol.get("risk_scale") or 1.0)
    if symbol:
        try:
            from analysis.candle_edge import zenith_overlays_for

            candle = zenith_overlays_for(str(symbol))
            mc_delta += max(-2.5, min(2.5, float(candle.get("mc_floor_delta") or 0)))
            ml_delta += max(-0.03, min(0.04, float(candle.get("min_ml_delta") or 0)))
            aggression = max(
                0.15,
                min(0.95, aggression + max(-0.12, min(0.12, float(candle.get("aggression") or 0)))),
            )
            # Candle risk nudge is narrow; close-RL risk_scale stays primary
            c_rs = float(candle.get("risk_scale") or 1.0)
            risk_scale = max(0.70, min(1.25, risk_scale * max(0.85, min(1.12, c_rs))))
        except Exception:
            candle = {}
    # Global soft-source budget (minute + board + candle vs arm base)
    arm_mc = float(pol.get("mc_floor_delta") or 0)
    soft_mc = mc_delta - arm_mc
    soft_mc = max(-4.0, min(4.0, soft_mc))
    mc_delta = arm_mc + soft_mc

    # Autopsy improve agent — soft only, hard-capped (never rewrites brains/.env)
    autopsy: dict[str, Any] = {}
    if symbol:
        try:
            from analysis.autopsy_soft import overlays_for as autopsy_overlays_for

            autopsy = autopsy_overlays_for(str(symbol))
            if autopsy:
                mc_delta += max(-3.0, min(4.0, float(autopsy.get("mc_floor_delta") or 0)))
                ml_delta += max(-0.02, min(0.03, float(autopsy.get("min_ml_delta") or 0)))
                aggression = max(
                    0.15,
                    min(
                        0.95,
                        aggression + max(-0.10, min(0.08, float(autopsy.get("aggression") or 0))),
                    ),
                )
                a_rs = float(autopsy.get("risk_scale") or 1.0)
                risk_scale = max(0.70, min(1.25, risk_scale * max(0.80, min(1.10, a_rs))))
        except Exception:
            autopsy = {}

    # Adaptive v3 size throttle (rolling 7d) — soft only
    if symbol:
        try:
            from strategy.adaptive_v3_runtime import risk_scale_for as _v3_risk

            v3_rs = float(_v3_risk(str(symbol)))
            risk_scale = max(0.50, min(1.25, risk_scale * max(0.55, min(1.10, v3_rs))))
        except Exception:
            pass

    arm = get_active_arm(symbol)
    # Prefer research SYMBOL_RR map; autopsy soft can still override
    try:
        from config import symbol_rr

        rr_prefer = float(symbol_rr(symbol, default=float(pol.get("rr_prefer") or 3.0)))
    except Exception:
        rr_prefer = float(pol.get("rr_prefer") or 3.0)
    if autopsy.get("rr_prefer") is not None:
        rr_prefer = float(autopsy["rr_prefer"])
    # Structure-brain symbols: don't force 1:3 grid when structure TP is primary.
    # Skip clamp when SYMBOL_RR already pinned a research target.
    try:
        from analysis.structure_targets import structure_enabled_for
        from config import SYMBOL_RR

        sym_u = str(symbol or "").upper()
        if structure_enabled_for(symbol) and sym_u not in SYMBOL_RR:
            rr_prefer = min(rr_prefer, 2.0) if arm != "tight" else min(rr_prefer, 3.0)
    except Exception:
        pass

    return {
        "symbol": str(symbol or "").upper() or None,
        "arm": arm,
        "mc_floor_delta": mc_delta,
        "min_ml_delta": ml_delta,
        "aggression": aggression,
        "risk_scale": risk_scale,
        "rr_prefer": rr_prefer,
        "partial_boost": float(pol.get("partial_boost") or 1.0),
        "trail_delta": float(pol.get("trail_delta") or 0.0),
        "board_seen": board.get("board_seen"),
        "board_entered": board.get("board_entered"),
        "structure_score": minute.get("structure_score"),
        "vp_score": minute.get("vp_score"),
        "candle_exp_r": candle.get("candle_exp_r"),
        "candle_wr": candle.get("candle_wr"),
        "candle_n": candle.get("candle_n"),
        "candle_message": candle.get("candle_message"),
        "candle_brain": candle.get("candle_brain"),
        "autopsy_soft": bool(autopsy),
        "autopsy_proposal_id": autopsy.get("proposal_id"),
    }


def mc_floor_for(symbol: Optional[str] = None, base: Optional[float] = None) -> float:
    """RL-tuned MC floor — soft enough that real setups punch and RL can learn.

    Band ~36–42. Tight only nudges +1; quality comes from risk_scale + rr_prefer.
    ETH gets an extra floor bump (desk mastery C) to cut spam losers.
    """
    from analysis.adaptive_guard import mc_floor as global_mc

    b = float(base) if base is not None else float(global_mc())
    ov = overlays_for(symbol)
    delta = float(ov.get("mc_floor_delta") or 0)
    arm = str(ov.get("arm") or get_active_arm(symbol) or "balanced")
    sym_u = str(symbol or "").upper().strip()
    if sym_u in ("ETHUSD", "ETH"):
        try:
            from config import ETH_MC_FLOOR_DELTA

            delta += float(ETH_MC_FLOOR_DELTA or 0)
        except Exception:
            delta += 8.0

    if arm == "aggressive":
        return max(32.0, min(46.0, b + max(-4.0, min(8.0, delta))))
    if arm == "tight":
        # Still tradeable — tiny nudge only (+ ETH delta still applies)
        return max(34.0, min(48.0, b + min(10.0, max(0.0, delta))))
    return max(34.0, min(48.0, b + min(10.0, max(-1.0, delta))))


def min_ml_for(symbol: Optional[str] = None, base: Optional[float] = None) -> float:
    """Soft ML hint only — hard blocks are skipped when MC clears (see evaluate_trade)."""
    from analysis.adaptive_guard import min_ml_prob as global_ml

    b = float(base) if base is not None else float(global_ml())
    ov = overlays_for(symbol)
    delta = float(ov.get("min_ml_delta") or 0)
    arm = str(ov.get("arm") or get_active_arm(symbol) or "balanced")
    if arm == "tight":
        return max(0.26, min(0.36, b + min(delta, 0.02)))
    return max(0.26, min(0.34, b + min(delta, 0.01)))


def aggression_for(symbol: Optional[str] = None) -> float:
    """Blend adaptive protect mode with per-symbol RL aggression."""
    try:
        from analysis.adaptive_guard import get_state

        mode = str((get_state() or {}).get("mode") or "")
        if mode in ("protect", "bank", "daily"):
            return min(0.25, overlays_for(symbol)["aggression"])
        if mode in ("seek", "aggressive"):
            return max(0.70, overlays_for(symbol)["aggression"])
    except Exception:
        pass
    return overlays_for(symbol)["aggression"]


def get_state() -> dict[str, Any]:
    with _LOCK:
        if not _state.get("instruments"):
            _load_disk()
        return {
            "updated_at": _state.get("updated_at"),
            "total_updates": _state.get("total_updates"),
            "instruments": {
                k: {
                    **v,
                    "policy": dict(v.get("policy") or {}),
                    "arms": {
                        a: {
                            "n": int(v["arms"][a]["n"]),
                            "mean_r": round(
                                float(v["arms"][a]["sum_r"]) / max(int(v["arms"][a]["n"]), 1),
                                3,
                            )
                            if int(v["arms"][a]["n"])
                            else None,
                            "sum_pnl": round(float(v["arms"][a]["sum_pnl"]), 2),
                        }
                        for a in ARMS
                    },
                    "sum_pnl": round(float(v.get("sum_pnl") or 0), 2),
                    "sum_r": round(float(v.get("sum_r") or 0), 3),
                }
                for k, v in _state["instruments"].items()
            },
        }


# Load on import
_load_disk()
