"""Continuous candle-edge — retrospection for soft prospection.

Two-track learning (history-safe):
  1. Closed-trade RL (instrument_rl) = truth — never wiped here
  2. This module = paper walk-forward on LIVE brains at each exec-TF close
       GOLD → fib_liq 15m
       Book → ema_20_50 5m
     Soft overlays only, hard-capped so they cannot override close RL

On each new 1m candle for the universe:
  • If a new exec TF bar just closed → paper-sim that brain signal forward
  • Accumulate rolling E[R] / WR → retune soft MC / aggression / risk_scale
  • Periodic EMA micro-grid on OOS slice (book only)

Hyper paper path kept for state compat but does not feed live (Hyper parked).

State: data/candle_edge_state.json
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PATH = Path(__file__).resolve().parents[1] / "data" / "candle_edge_state.json"

_SPREAD_BPS = {
    "GOLD": 1.8,
    "USOIL": 2.5,
    "UKOIL": 2.5,
    "SILVER": 3.5,
    "USDCHF": 1.2,
    "USDJPY": 1.2,
    "AUDUSD": 1.2,
    "NAS100": 1.5,
    "BTCUSD": 2.0,
    "ETHUSD": 2.5,
}
_RECENT_MAX = 80
_GRID_EVERY = 12  # exec-TF credits between micro grid searches
_MIN_BARS = 80

# Soft overlay hard caps — close-trade RL owns the rest
_SOFT_MC = (-2.5, 2.5)
_SOFT_ML = (-0.03, 0.04)
_SOFT_AGG = (-0.12, 0.12)
_SOFT_RISK = (0.85, 1.12)

_EMA_PULL = (0.70, 1.30)
_EMA_DISP = (0.40, 0.85)

_DEFAULT_SYM = {
    "last_1m": None,
    "last_exec_ts": None,
    "bars_seen": 0,
    "grid_tick": 0,
    "zenith": {
        "n": 0,
        "wins": 0,
        "exp_r": 0.0,
        "wr": 0.0,
        "sum_r": 0.0,
        "pull_atr": 1.0,
        "disp_atr": 0.55,
        "tp_r": 2.0,
        "mc_floor_delta": 0.0,
        "min_ml_delta": 0.0,
        "aggression": 0.0,
        "risk_scale": 1.0,
        "message": "warming",
        "brain": "ema_20_50",
    },
    "hyper": {
        "n": 0,
        "wins": 0,
        "exp_r": 0.0,
        "wr": 0.0,
        "sum_r": 0.0,
        "disp_atr": 0.65,
        "pull_max_atr": 0.85,
        "cost_mult": 3.5,
        "tp_r": 1.5,
        "lot_scale": 1.0,
        "block": False,
        "message": "parked",
    },
}

_STATE: dict[str, Any] = {"updated_at": None, "symbols": {}}
_recent: dict[str, dict[str, deque]] = {}


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _load() -> None:
    global _STATE
    if not _PATH.exists():
        return
    try:
        raw = json.loads(_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            _STATE = raw
            _STATE.setdefault("symbols", {})
    except Exception:
        pass


def _save() -> None:
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(_STATE, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.debug("candle_edge save: %s", exc)


_load()


def _is_gold(symbol: str) -> bool:
    return str(symbol).upper().strip() in ("GOLD", "XAUUSD", "XAU")


def _exec_rule(symbol: str) -> str:
    return "15min" if _is_gold(symbol) else "5min"


def _sym(symbol: str) -> dict[str, Any]:
    s = str(symbol).upper()
    syms = _STATE.setdefault("symbols", {})
    if s not in syms:
        import copy

        syms[s] = copy.deepcopy(_DEFAULT_SYM)
        syms[s]["symbol"] = s
        syms[s]["zenith"]["brain"] = "fib_liq" if _is_gold(s) else "ema_20_50"
    for k, v in _DEFAULT_SYM.items():
        if k not in syms[s]:
            import copy

            syms[s][k] = copy.deepcopy(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                syms[s][k].setdefault(kk, vv)
    return syms[s]


def _recent_dq(symbol: str, engine: str) -> deque:
    s = str(symbol).upper()
    bucket = _recent.setdefault(s, {})
    if engine not in bucket:
        bucket[engine] = deque(maxlen=_RECENT_MAX)
    return bucket[engine]


def _spread(symbol: str, px: float) -> float:
    return abs(float(px)) * float(_SPREAD_BPS.get(str(symbol).upper(), 2.0)) / 1e4


def _to_htf(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum() if "volume" in df.columns else c * 0
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()


def _prep_brain_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    c = out["close"].astype(float)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()
    out["EMA20"] = c.ewm(span=20, adjust=False).mean()
    out["EMA50"] = c.ewm(span=50, adjust=False).mean()
    # fib brain uses lowercase atr optionally
    out["atr"] = out["ATR"]
    return out


def _sim_bar(
    df: pd.DataFrame,
    i: int,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    max_hold: int,
) -> Optional[float]:
    risk = abs(entry - sl)
    if risk <= 1e-12:
        return None
    n = len(df)
    for j in range(i, min(i + max_hold, n)):
        hi = float(df["high"].iloc[j])
        lo = float(df["low"].iloc[j])
        if direction == "BUY":
            if lo <= sl:
                return -1.0
            if hi >= tp:
                return (tp - entry) / risk
        else:
            if hi >= sl:
                return -1.0
            if lo <= tp:
                return (entry - tp) / risk
    last = float(df["close"].iloc[min(i + max_hold, n) - 1])
    return ((last - entry) if direction == "BUY" else (entry - last)) / risk


def _brain_hit(
    symbol: str,
    htf: pd.DataFrame,
    *,
    pull_atr: float,
    disp_atr: float,
) -> Optional[dict[str, Any]]:
    if _is_gold(symbol):
        from strategy.fib_liq_brain import evaluate_fib_liq_sweep

        return evaluate_fib_liq_sweep(htf)
    from strategy.ema_pull_brain import evaluate_ema_2050_pull

    return evaluate_ema_2050_pull(htf, pull_atr=pull_atr, disp_atr=disp_atr)


def evaluate_live_brain_r(
    symbol: str,
    df_1m: pd.DataFrame,
    *,
    pull_atr: float = 1.0,
    disp_atr: float = 0.55,
    tp_r: float = 2.0,
    max_hold_1m: int = 48,
    asof_1m_i: Optional[int] = None,
) -> tuple[Optional[float], Optional[str]]:
    """Paper R for live brain on latest closed exec-TF bar.

    Returns (r_multiple_or_None, exec_bar_timestamp).
    Uses only data through the closed 1m bar (excludes forming bar by default).
    """
    s = str(symbol).upper()
    if df_1m is None or len(df_1m) < _MIN_BARS:
        return None, None
    end = len(df_1m) - 1 if asof_1m_i is None else int(asof_1m_i)
    if end < 80:
        return None, None
    # Closed history only
    hist = df_1m.iloc[: end + 1]
    rule = _exec_rule(s)
    htf = _prep_brain_df(_to_htf(hist, rule))
    min_bars = 80 if _is_gold(s) else 60
    if len(htf) < min_bars:
        return None, None

    hit = _brain_hit(s, htf, pull_atr=pull_atr, disp_atr=disp_atr)
    if not hit:
        return None, str(htf.index[-1])

    side = hit["signal"]
    sl = float(hit["sl"])
    exec_ts = htf.index[-1]
    delta_min = 15 if rule.startswith("15") else 5
    delta = pd.Timedelta(minutes=delta_min)
    bar_end = exec_ts + delta

    # Detect native bar size (1m broker vs 5m hist fallback)
    if len(hist) >= 3:
        med_sec = float((hist.index[1:] - hist.index[:-1]).median().total_seconds())
    else:
        med_sec = 60.0
    coarse = med_sec >= 180  # ≥3m → already near exec TF

    if coarse:
        # Rows are already ~exec TF: enter at next row open after signal bar
        pos = int(htf.index.get_loc(exec_ts))
        if isinstance(pos, slice):
            pos = int(pos.start or 0)
        # Map exec bar timestamp onto df_1m (same index when hist fallback)
        if exec_ts not in df_1m.index:
            return None, str(exec_ts)
        entry_i = int(df_1m.index.get_loc(exec_ts))
        if isinstance(entry_i, slice):
            entry_i = int(entry_i.start or 0)
        entry_i = min(entry_i + 1, len(df_1m) - 1)
        if entry_i <= 0:
            return None, str(exec_ts)
    else:
        in_bar = hist.index[(hist.index >= exec_ts) & (hist.index < bar_end)]
        # Need most of the bucket filled
        need = max(2, delta_min - 1)
        if len(in_bar) < need:
            return None, str(exec_ts)
        after = df_1m.index[df_1m.index >= bar_end]
        if len(after) == 0:
            return None, str(exec_ts)
        entry_t = after[0]
        entry_i = int(df_1m.index.get_loc(entry_t))
        if isinstance(entry_i, slice):
            entry_i = int(entry_i.start or 0)

    entry = float(df_1m["open"].iloc[entry_i])
    if side == "BUY":
        risk2 = entry - sl
        if risk2 <= 0:
            return None, str(exec_ts)
        tp = entry + tp_r * risk2
        sl_e = sl
    else:
        risk2 = sl - entry
        if risk2 <= 0:
            return None, str(exec_ts)
        tp = entry - tp_r * risk2
        sl_e = sl

    spread = _spread(s, entry)
    hold = max_hold_1m if not coarse else max(12, max_hold_1m // 5)
    r = _sim_bar(df_1m, entry_i, side, entry, sl_e, tp, hold)
    if r is None:
        return None, str(exec_ts)
    adj = float(r) - spread / max(abs(entry - sl_e), 1e-12)
    return adj, str(exec_ts)


def _credit(symbol: str, engine: str, r: float, row: dict[str, Any]) -> None:
    dq = _recent_dq(symbol, engine)
    dq.append(float(r))
    recent = list(dq)
    row["n"] = int(row.get("n") or 0) + 1
    row["sum_r"] = float(row.get("sum_r") or 0.0) + float(r)
    if r > 0:
        row["wins"] = int(row.get("wins") or 0) + 1
    row["exp_r"] = round(sum(recent) / len(recent), 4)
    row["wr"] = round(sum(1 for x in recent if x > 0) / len(recent), 4)


def _retune_zenith(z: dict[str, Any]) -> None:
    """Soft retune only — hard-capped; never touches closed-trade RL state."""
    exp = float(z.get("exp_r") or 0.0)
    wr = float(z.get("wr") or 0.0)
    n = int(z.get("n") or 0)
    if n < 6:
        z["message"] = f"collecting ({n})"
        return
    if exp < -0.05 or wr < 0.42:
        z["mc_floor_delta"] = _clip(float(z.get("mc_floor_delta") or 0) + 0.5, *_SOFT_MC)
        z["min_ml_delta"] = _clip(float(z.get("min_ml_delta") or 0) + 0.01, *_SOFT_ML)
        z["aggression"] = _clip(float(z.get("aggression") or 0) - 0.04, *_SOFT_AGG)
        z["risk_scale"] = _clip(float(z.get("risk_scale") or 1.0) * 0.96, *_SOFT_RISK)
        if not _is_gold(z.get("symbol") or ""):
            z["disp_atr"] = _clip(float(z.get("disp_atr") or 0.55) + 0.02, *_EMA_DISP)
            z["pull_atr"] = _clip(float(z.get("pull_atr") or 1.0) - 0.03, *_EMA_PULL)
        z["message"] = f"tighten · E[R]={exp:+.3f} WR={wr:.0%}"
    elif exp > 0.08 and wr > 0.52:
        z["mc_floor_delta"] = _clip(float(z.get("mc_floor_delta") or 0) - 0.35, *_SOFT_MC)
        z["aggression"] = _clip(float(z.get("aggression") or 0) + 0.03, *_SOFT_AGG)
        z["risk_scale"] = _clip(float(z.get("risk_scale") or 1.0) * 1.02, *_SOFT_RISK)
        z["message"] = f"lean in · E[R]={exp:+.3f} WR={wr:.0%}"
    else:
        # Mean-revert soft knobs slowly toward neutral
        z["mc_floor_delta"] = _clip(float(z.get("mc_floor_delta") or 0) * 0.92, *_SOFT_MC)
        z["aggression"] = _clip(float(z.get("aggression") or 0) * 0.92, *_SOFT_AGG)
        z["message"] = f"hold · E[R]={exp:+.3f} WR={wr:.0%}"


def _micro_grid_ema(symbol: str, df_1m: pd.DataFrame) -> dict[str, float]:
    """OOS micro-grid for EMA knobs (book only). GOLD fib has no knobs here."""
    if _is_gold(symbol):
        return {}
    n = len(df_1m)
    if n < 160:
        return {}
    # Sample exec bars via stepping 1m index near 5m closes
    i0 = max(90, n - 280)
    split = i0 + int((n - i0) * 0.70)
    best = None
    best_sc = -1e9
    grid = [
        (p, d)
        for p in (0.85, 1.0, 1.15)
        for d in (0.45, 0.55, 0.70)
    ]
    for pull, disp in grid:
        rs_oos: list[float] = []
        last_exec = None
        for i in range(split, n - 3, 5):  # ~every 5m
            r, ets = evaluate_live_brain_r(
                symbol,
                df_1m,
                pull_atr=pull,
                disp_atr=disp,
                tp_r=2.0,
                asof_1m_i=i,
            )
            if ets is None or ets == last_exec:
                continue
            last_exec = ets
            if r is None:
                continue
            rs_oos.append(r)
        if len(rs_oos) < 3:
            continue
        exp = float(np.mean(rs_oos))
        sc = exp * math.log1p(len(rs_oos))
        if sc > best_sc:
            best_sc = sc
            best = {
                "pull_atr": pull,
                "disp_atr": disp,
                "oos_n": len(rs_oos),
                "oos_exp": round(exp, 4),
            }
    return best or {}


def _fetch_hist_5m(symbol: str, days: int = 21) -> Optional[pd.DataFrame]:
    """Deep 5m history for paper retrospection when broker 1m is thin (weekends)."""
    try:
        from datetime import datetime, timedelta, timezone

        from data.historical import fetch_symbol_ohlcv

        end = datetime.now(timezone.utc).date()
        start = (end - timedelta(days=days)).isoformat()
        raw = fetch_symbol_ohlcv(symbol, start, end.isoformat(), interval="5m")
        if raw is None or len(raw) < 80:
            return None
        raw = raw.rename(columns=str.lower)
        out = raw[["open", "high", "low", "close"]].copy()
        out["volume"] = raw["volume"] if "volume" in raw.columns else 0.0
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        return out
    except Exception as exc:
        logger.debug("candle_edge hist fallback %s: %s", symbol, exc)
        return None


def _fetch_1m(symbol: str, bars: int = 900) -> Optional[pd.DataFrame]:
    """Prefer broker 1m when deep enough; else 5m hist for bootstrap depth."""
    df = None
    try:
        from data.mt5_client import mt5_client

        if mt5_client.configured and not mt5_client._in_backoff():  # noqa: SLF001
            df = mt5_client.fetch_candles("1m", bars=min(bars, 1000), symbol=symbol)
    except Exception:
        pass
    if df is None or len(df) < _MIN_BARS:
        try:
            from data.market_data import fetch_ohlcv

            df = fetch_ohlcv("1m", bars, symbol=symbol)
        except Exception as exc:
            logger.debug("candle_edge fetch 1m %s: %s", symbol, exc)

    # Thin/weekend 1m → use 5m history so paper RL can still learn
    if df is None or len(df) < 400:
        deep = _fetch_hist_5m(symbol, days=21)
        if deep is not None and (df is None or len(deep) > len(df)):
            return deep
    if df is not None and len(df) >= _MIN_BARS:
        return df
    return _fetch_hist_5m(symbol, days=21)


def _bootstrap_symbol(symbol: str, df: pd.DataFrame) -> None:
    """Seed paper expectancy from recent exec-TF closes — no trade DB writes."""
    s = str(symbol).upper()
    n = len(df)
    start = max(100, n - 400)
    # Step by ~one exec bar (1 if series is already 5m hist)
    if n >= 3:
        med_sec = float((df.index[1:] - df.index[:-1]).median().total_seconds())
    else:
        med_sec = 60.0
    if med_sec >= 180:
        step = 3 if _is_gold(s) else 1
    else:
        step = 15 if _is_gold(s) else 5
    last_exec = None
    with _LOCK:
        z = _sym(s)["zenith"]
        pull = float(z.get("pull_atr") or 1.0)
        disp = float(z.get("disp_atr") or 0.55)
        z["symbol"] = s

    for i in range(start, n - 3, step):
        r, ets = evaluate_live_brain_r(
            s, df, pull_atr=pull, disp_atr=disp, tp_r=2.0, asof_1m_i=i
        )
        if ets is None or ets == last_exec:
            continue
        last_exec = ets
        if r is None:
            continue
        with _LOCK:
            _credit(s, "zenith", r, _sym(s)["zenith"])

    with _LOCK:
        row = _sym(s)
        row["zenith"]["symbol"] = s
        _retune_zenith(row["zenith"])
        if not _is_gold(s):
            try:
                zg = _micro_grid_ema(s, df)
                if zg:
                    row["zenith"]["pull_atr"] = zg["pull_atr"]
                    row["zenith"]["disp_atr"] = zg["disp_atr"]
            except Exception:
                pass
        row["bars_seen"] = max(int(row.get("bars_seen") or 0), 1)
        row["last_exec_ts"] = last_exec
        row["zenith"]["message"] = f"bootstrapped · {row['zenith'].get('message')}"
        row["hyper"]["message"] = "parked"
        _save()


def on_new_candle(symbol: str, *, candle_key: Optional[str] = None) -> dict[str, Any]:
    """Call once per new 1m candle. Credits live-brain paper trade on new exec TF close."""
    s = str(symbol).upper()
    with _LOCK:
        row = _sym(s)
        if candle_key and candle_key == row.get("last_1m"):
            return {"symbol": s, "skipped": "same_candle"}
        if candle_key:
            row["last_1m"] = candle_key
        need_bootstrap = int(row.get("bars_seen") or 0) == 0 and int(row["zenith"].get("n") or 0) == 0

    df = _fetch_1m(s, 360)
    if df is None or len(df) < _MIN_BARS:
        with _LOCK:
            row = _sym(s)
            row["zenith"]["message"] = "warming candles"
            _STATE["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _save()
        return {"symbol": s, "skipped": "candles"}

    if need_bootstrap:
        try:
            _bootstrap_symbol(s, df)
        except Exception as exc:
            logger.debug("candle_edge bootstrap %s: %s", s, exc)

    with _LOCK:
        z = _sym(s)["zenith"]
        pull = float(z.get("pull_atr") or 1.0)
        disp = float(z.get("disp_atr") or 0.55)

    # Evaluate on last fully closed 1m (iloc[-2])
    r, ets = evaluate_live_brain_r(
        s,
        df,
        pull_atr=pull,
        disp_atr=disp,
        tp_r=2.0,
        asof_1m_i=len(df) - 2,
    )

    with _LOCK:
        row = _sym(s)
        row["bars_seen"] = int(row.get("bars_seen") or 0) + 1
        row["zenith"]["brain"] = "fib_liq" if _is_gold(s) else "ema_20_50"
        row["zenith"]["symbol"] = s
        credited = False
        # Only one credit per exec TF bar
        if ets and ets != row.get("last_exec_ts"):
            row["last_exec_ts"] = ets
            if r is not None:
                _credit(s, "zenith", r, row["zenith"])
                _retune_zenith(row["zenith"])
                credited = True
                row["grid_tick"] = int(row.get("grid_tick") or 0) + 1

        if credited and row["grid_tick"] >= _GRID_EVERY and not _is_gold(s):
            row["grid_tick"] = 0
            try:
                zg = _micro_grid_ema(s, df)
                if zg:
                    row["zenith"]["pull_atr"] = zg["pull_atr"]
                    row["zenith"]["disp_atr"] = zg["disp_atr"]
                    row["zenith"]["message"] = (
                        f"grid OOS E[R]={zg.get('oos_exp'):+.3f} n={zg.get('oos_n')} · "
                        f"{row['zenith'].get('message', '')}"
                    )
            except Exception as exc:
                logger.debug("candle_edge grid %s: %s", s, exc)

        _STATE["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _save()
        out = {
            "symbol": s,
            "brain": row["zenith"].get("brain"),
            "zenith_r": r if credited else None,
            "exec_ts": ets,
            "credited": credited,
            "zenith": dict(row["zenith"]),
        }
    return out


def run_candle_edge(
    symbols: Optional[list[str]] = None,
    *,
    candle_key: Optional[str] = None,
) -> dict[str, Any]:
    """Universe pass — call from signal_worker on each new 1m candle."""
    try:
        from config import INSTRUMENT_UNIVERSE

        universe = list(symbols or INSTRUMENT_UNIVERSE)
    except Exception:
        universe = list(symbols or ["GOLD", "USDCHF", "BTCUSD"])
    results = []
    for sym in universe:
        try:
            results.append(on_new_candle(sym, candle_key=candle_key))
        except Exception as exc:
            logger.debug("candle_edge %s: %s", sym, exc)
            results.append({"symbol": sym, "error": str(exc)})
    return {"updated_at": _STATE.get("updated_at"), "results": results}


def zenith_overlays_for(symbol: Optional[str] = None) -> dict[str, Any]:
    """Soft deltas for instrument_rl — dampened until enough paper samples."""
    if not symbol:
        return {}
    with _LOCK:
        row = _STATE.get("symbols", {}).get(str(symbol).upper())
        if not row:
            return {}
        z = dict(row.get("zenith") or {})
    n = int(z.get("n") or 0)
    pull = float(z.get("pull_atr") or 1.0)
    disp = float(z.get("disp_atr") or 0.55)
    # Warmup: no gate/size influence until enough paper fills (knobs still exposed)
    if n < 6:
        return {
            "mc_floor_delta": 0.0,
            "min_ml_delta": 0.0,
            "aggression": 0.0,
            "risk_scale": 1.0,
            "candle_exp_r": float(z.get("exp_r") or 0.0),
            "candle_wr": float(z.get("wr") or 0.0),
            "candle_n": n,
            "candle_message": z.get("message") or "collecting",
            "candle_brain": z.get("brain"),
            "pull_atr": pull,
            "disp_atr": disp,
        }
    # Confidence scales 6→20 samples
    conf = min(1.0, (n - 5) / 15.0)
    return {
        "mc_floor_delta": _clip(float(z.get("mc_floor_delta") or 0.0) * conf, *_SOFT_MC),
        "min_ml_delta": _clip(float(z.get("min_ml_delta") or 0.0) * conf, *_SOFT_ML),
        "aggression": _clip(float(z.get("aggression") or 0.0) * conf, *_SOFT_AGG),
        "risk_scale": _clip(
            1.0 + (float(z.get("risk_scale") or 1.0) - 1.0) * conf,
            *_SOFT_RISK,
        ),
        "candle_exp_r": float(z.get("exp_r") or 0.0),
        "candle_wr": float(z.get("wr") or 0.0),
        "candle_n": n,
        "candle_message": z.get("message"),
        "candle_brain": z.get("brain"),
        "pull_atr": float(z.get("pull_atr") or 1.0),
        "disp_atr": float(z.get("disp_atr") or 0.55),
    }


def hyper_overlays_for(symbol: Optional[str] = None) -> dict[str, Any]:
    """Hyper parked — return neutral knobs (compat for any leftover callers)."""
    if not symbol:
        return {}
    return {
        "disp_atr": 0.65,
        "pull_max_atr": 0.85,
        "cost_mult": 3.5,
        "tp_r": 1.5,
        "lot_scale": 1.0,
        "block": False,
        "candle_exp_r": 0.0,
        "candle_wr": 0.0,
        "candle_n": 0,
        "candle_message": "parked",
    }


def snapshot() -> dict[str, Any]:
    with _LOCK:
        return {
            "updated_at": _STATE.get("updated_at"),
            "mode": "live_brain_retrospection",
            "symbols": [
                {
                    "symbol": k,
                    "bars_seen": v.get("bars_seen"),
                    "last_exec_ts": v.get("last_exec_ts"),
                    "zenith": dict(v.get("zenith") or {}),
                    "hyper": {"message": "parked"},
                }
                for k, v in sorted((_STATE.get("symbols") or {}).items())
            ],
        }


def seed_zenith_knobs(
    symbol: str,
    *,
    pull_atr: Optional[float] = None,
    disp_atr: Optional[float] = None,
) -> None:
    """Seed EMA knobs after autopsy promote (does not wipe paper stats)."""
    from datetime import datetime, timezone

    with _LOCK:
        row = _sym(symbol)
        z = row.setdefault("zenith", {})
        if pull_atr is not None:
            z["pull_atr"] = _clip(float(pull_atr), *_EMA_PULL)
        if disp_atr is not None:
            z["disp_atr"] = _clip(float(disp_atr), *_EMA_DISP)
        z["message"] = "autopsy_promoted"
        _STATE["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save()


# ── Legacy name kept for any external imports ───────────────────────────────
def evaluate_bar(*_a, **_k):  # noqa: ANN001
    """Deprecated — use evaluate_live_brain_r. Returns None."""
    return None
