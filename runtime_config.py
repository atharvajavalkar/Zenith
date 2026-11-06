"""
Runtime-mutable algorithm parameters.

Initialised from config/.env at startup.
Updated in-memory via PATCH /api/algo-settings — no Docker restart needed.
Settings revert to .env defaults on container restart (permanent changes go in .env).
"""
from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_settings: dict[str, Any] = {}
_defaults: dict[str, Any] = {}
_initialized = False


def _load_defaults() -> dict[str, Any]:
    from config import (
        ALLOW_ALL_SESSIONS,
        BANK_TARGET_PCT,
        BIAS_GATE_ENABLED,
        BLOCK_OPPOSITE_OPEN,
        CALIBRATION_TARGET_WIN_RATE,
        DAILY_LOSS_LIMIT_PCT,
        DIRECTION_LOCK_5M,
        DYNAMIC_LOTS_ENABLED,
        HARVEST_PCT,
        LOSS_PNL_PROTECT,
        LOSS_STREAK_REDUCE_AT,
        MAX_RISK_PCT_PER_TRADE,
        MAX_SL_ATR,
        MAX_TRADES_PER_DAY,
        MC_FLOOR_BASE,
        MIN_BIAS_CONFIDENCE,
        MIN_SMC_SCORE,
        MIN_TRAIN_SAMPLES,
        MINUTE_SL_ATR,
        ML_GATE_ENABLED,
        PEAK_DD_PROTECT_PCT,
        START_CAPITAL_USD,
        TARGET_PROFIT_USD,
        TRADE_LOT_SIZE,
        TRAIL_PROFIT_LOCK,
        TRADE_MODE,
        VALID_TRADE_MODES,
    )

    mode = str(TRADE_MODE).lower().strip()
    if mode not in VALID_TRADE_MODES:
        mode = "paper"

    return {
        # ── MC / Markov ───────────────────────────────────────────────────────
        "mc_floor_base":        float(MC_FLOOR_BASE),
        "mc_cautious_offset":   6.0,
        "mc_protect_offset":    12.0,

        # ── Equity protect (live XM + backtest parity) ─────────────────────────
        "start_capital_usd":    float(START_CAPITAL_USD),
        "peak_dd_protect_pct":  float(PEAK_DD_PROTECT_PCT),
        "daily_loss_limit_pct": float(DAILY_LOSS_LIMIT_PCT),
        "max_trades_per_day":   int(MAX_TRADES_PER_DAY),
        "harvest_pct":          float(HARVEST_PCT),
        "bank_target_pct":      float(BANK_TARGET_PCT),

        # ── ML Gate ───────────────────────────────────────────────────────────
        "ml_gate_enabled":      bool(ML_GATE_ENABLED),
        "ml_prob_normal":       0.42,
        "ml_prob_cautious":     0.50,
        "ml_prob_protect":      0.58,
        "calibration_win_rate": float(CALIBRATION_TARGET_WIN_RATE),
        "min_train_samples":    int(MIN_TRAIN_SAMPLES),

        # ── Adaptive mode triggers ────────────────────────────────────────────
        "loss_pnl_protect":     float(LOSS_PNL_PROTECT),
        "loss_streak_cautious": int(LOSS_STREAK_REDUCE_AT),

        # ── Risk management ───────────────────────────────────────────────────
        "sl_atr_mult":          float(MINUTE_SL_ATR),
        "max_sl_atr":           float(MAX_SL_ATR),
        "trail_profit_lock":    float(TRAIL_PROFIT_LOCK),
        "min_smc_score":        float(MIN_SMC_SCORE),

        # ── Execution ─────────────────────────────────────────────────────────
        "trade_mode":           mode,
        "trade_lot_size":       float(TRADE_LOT_SIZE),
        "dynamic_lots_enabled": bool(DYNAMIC_LOTS_ENABLED),
        "target_profit_usd":    float(TARGET_PROFIT_USD),
        "max_risk_pct_per_trade": float(MAX_RISK_PCT_PER_TRADE),
        "allow_all_sessions":   bool(ALLOW_ALL_SESSIONS),
        "block_opposite_open":  bool(BLOCK_OPPOSITE_OPEN),
        "direction_lock_5m":    bool(DIRECTION_LOCK_5M),

        # ── Bias gate ─────────────────────────────────────────────────────────
        "bias_gate_enabled":    bool(BIAS_GATE_ENABLED),
        "min_bias_confidence":  float(MIN_BIAS_CONFIDENCE),
    }


def _ensure_init() -> None:
    global _initialized, _settings, _defaults
    if not _initialized:
        with _lock:
            if not _initialized:
                _defaults = _load_defaults()
                _settings = dict(_defaults)
                _initialized = True
                try:
                    from data.mt5_client import mt5_client

                    mt5_client.apply_trade_mode(str(_settings.get("trade_mode") or "paper"))
                except Exception:
                    pass


def get_all() -> dict[str, Any]:
    _ensure_init()
    with _lock:
        return dict(_settings)


def get(key: str, default: Any = None) -> Any:
    _ensure_init()
    with _lock:
        return _settings.get(key, default)


def get_defaults() -> dict[str, Any]:
    _ensure_init()
    with _lock:
        return dict(_defaults)


def update(data: dict[str, Any]) -> dict[str, Any]:
    """Merge data into live settings. Unknown keys are silently ignored."""
    _ensure_init()
    with _lock:
        for k, v in data.items():
            if k not in _settings:
                continue
            orig = _settings[k]
            try:
                if isinstance(orig, bool):
                    _settings[k] = bool(v)
                elif isinstance(orig, int):
                    _settings[k] = int(v)
                elif isinstance(orig, str):
                    _settings[k] = str(v)
                else:
                    _settings[k] = float(v)
            except (TypeError, ValueError):
                continue
    return get_all()


def get_trade_mode() -> str:
    from config import VALID_TRADE_MODES

    mode = str(get("trade_mode", "paper")).lower().strip()
    return mode if mode in VALID_TRADE_MODES else "paper"


def is_broker_mode(mode: str | None = None) -> bool:
    """Orders go to XM MetaAPI (demo or live account)."""
    m = mode if mode is not None else get_trade_mode()
    return m in ("demo", "live")


def set_trade_mode(mode: str) -> str:
    """Set paper / demo / live trading mode at runtime."""
    from config import VALID_TRADE_MODES

    normalized = str(mode).lower().strip()
    if normalized not in VALID_TRADE_MODES:
        raise ValueError("trade_mode must be 'paper', 'demo', or 'live'")
    _ensure_init()
    with _lock:
        _settings["trade_mode"] = normalized
    try:
        from data.mt5_client import mt5_client

        mt5_client.apply_trade_mode(normalized)
    except Exception:
        pass
    try:
        from data.market_data import on_mode_change

        on_mode_change(normalized)
    except Exception:
        pass
    return normalized


def reset() -> dict[str, Any]:
    """Restore all settings to .env / startup defaults."""
    _ensure_init()
    with _lock:
        _settings.clear()
        _settings.update(_defaults)
    return get_all()
