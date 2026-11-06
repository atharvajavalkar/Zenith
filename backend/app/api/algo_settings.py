"""API router — runtime algorithm parameter control panel."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from app.core.security import get_current_user

router = APIRouter(prefix="/api/algo-settings", tags=["algo-settings"])

# ── Parameter metadata for the UI ─────────────────────────────────────────────
PARAM_META: dict[str, dict] = {
    # MC / Markov
    "mc_floor_base": {
        "label": "MC Floor Base (%)", "group": "MC / Markov", "type": "float",
        "min": 15, "max": 70, "step": 1,
        "description": "Minimum Monte Carlo success probability to allow a trade (normal mode)",
    },
    "mc_cautious_offset": {
        "label": "Cautious Mode Extra (%)", "group": "MC / Markov", "type": "float",
        "min": 0, "max": 25, "step": 1,
        "description": "Additional MC % added on top of base when in cautious mode",
    },
    "mc_protect_offset": {
        "label": "Protect Mode Extra (%)", "group": "MC / Markov", "type": "float",
        "min": 0, "max": 40, "step": 1,
        "description": "Additional MC % added on top of base when in protect mode",
    },

    # ML Gate
    "ml_gate_enabled": {
        "label": "ML Gate", "group": "ML Gate", "type": "bool",
        "description": "Enable machine-learning win-probability gate",
    },
    "ml_prob_normal": {
        "label": "ML Min Prob — Normal", "group": "ML Gate", "type": "float",
        "min": 0.30, "max": 0.80, "step": 0.01,
        "description": "Minimum ML win probability required in normal mode",
    },
    "ml_prob_cautious": {
        "label": "ML Min Prob — Cautious", "group": "ML Gate", "type": "float",
        "min": 0.30, "max": 0.85, "step": 0.01,
        "description": "Minimum ML win probability required in cautious mode",
    },
    "ml_prob_protect": {
        "label": "ML Min Prob — Protect", "group": "ML Gate", "type": "float",
        "min": 0.40, "max": 0.90, "step": 0.01,
        "description": "Minimum ML win probability required in protect mode",
    },
    "calibration_win_rate": {
        "label": "Calibration WR Target", "group": "ML Gate", "type": "float",
        "min": 0.40, "max": 0.75, "step": 0.01,
        "description": "Target win rate for ML threshold calibration (lower = more trades allowed)",
    },
    "min_train_samples": {
        "label": "Min Train Samples", "group": "ML Gate", "type": "int",
        "min": 5, "max": 100, "step": 1,
        "description": "Closed trades needed before the ML model is trained",
    },

    # Adaptive mode triggers
    "loss_pnl_protect": {
        "label": "Loss P&L → Protect ($)", "group": "Adaptive Triggers", "type": "float",
        "min": -1000, "max": -10, "step": 5,
        "description": "Cumulative realized loss that triggers protect mode",
    },
    "loss_streak_cautious": {
        "label": "Loss Streak → Cautious", "group": "Adaptive Triggers", "type": "int",
        "min": 1, "max": 15, "step": 1,
        "description": "Consecutive losses before the algo switches to cautious mode",
    },

    # Risk management
    "sl_atr_mult": {
        "label": "SL ATR Multiplier", "group": "Risk Management", "type": "float",
        "min": 0.3, "max": 3.0, "step": 0.1,
        "description": "Stop-loss distance in ATR multiples (smaller = tighter SL)",
    },
    "max_sl_atr": {
        "label": "Max SL ATR Cap", "group": "Risk Management", "type": "float",
        "min": 0.5, "max": 4.0, "step": 0.1,
        "description": "Hard cap on SL distance — trades with wider SL are rejected",
    },
    "trail_profit_lock": {
        "label": "Trail Profit Lock", "group": "Risk Management", "type": "float",
        "min": 0.10, "max": 0.95, "step": 0.05,
        "description": "Fraction of max favorable move locked when trailing SL (0.5 = lock 50%)",
    },
    "min_smc_score": {
        "label": "Min SMC Score", "group": "Risk Management", "type": "float",
        "min": 0.5, "max": 5.0, "step": 0.5,
        "description": "Minimum Smart Money Concept structure confirmation score",
    },

    # Execution
    "trade_lot_size": {
        "label": "Fallback Lot Size", "group": "Execution", "type": "float",
        "min": 0.01, "max": 10.0, "step": 0.01,
        "description": "Used only when dynamic lots are off",
    },
    "dynamic_lots_enabled": {
        "label": "Dynamic Lots", "group": "Execution", "type": "bool",
        "description": "Size each trade for target TP profit; SL follows setup RR",
    },
    "target_profit_usd": {
        "label": "Target Profit ($)", "group": "Execution", "type": "float",
        "min": 50, "max": 5000, "step": 50,
        "description": "Aim for at least this USD profit at TP on each trade",
    },
    "max_risk_pct_per_trade": {
        "label": "Max Risk % / Trade", "group": "Execution", "type": "float",
        "min": 0.005, "max": 0.10, "step": 0.005,
        "description": "Cap SL loss as a fraction of account (safety ceiling)",
    },
    "allow_all_sessions": {
        "label": "Allow All Sessions", "group": "Execution", "type": "bool",
        "description": "If off, only trade during NY/London session windows",
    },
    "block_opposite_open": {
        "label": "Block Opposite Open", "group": "Execution", "type": "bool",
        "description": "Reject new trade if an open position exists in the opposite direction",
    },
    "direction_lock_5m": {
        "label": "Direction Lock 5M", "group": "Execution", "type": "bool",
        "description": "Force trade direction to match the 5M trend direction",
    },

    # Bias gate
    "bias_gate_enabled": {
        "label": "Bias Gate", "group": "Bias Gate", "type": "bool",
        "description": "Enable AI analyst directional bias gate",
    },
    "min_bias_confidence": {
        "label": "Min Bias Confidence (%)", "group": "Bias Gate", "type": "float",
        "min": 50, "max": 95, "step": 5,
        "description": "AI bias confidence required to block opposing trades",
    },
}


@router.get("")
async def get_algo_settings(_: dict = Depends(get_current_user)):
    """Return current runtime settings, startup defaults, and UI metadata."""
    import runtime_config
    from analysis.adaptive_guard import get_state
    from config import (
        HTF_ENTRY_5M_EXIT,
        SYMBOL_ENTRY_BRAIN,
        SYMBOL_RR,
        SYMBOL_SCAN_TFS,
    )

    return {
        "settings": runtime_config.get_all(),
        "defaults": runtime_config.get_defaults(),
        "meta":     PARAM_META,
        "guard":    get_state(),          # live adaptive guard state for context
        "symbol_rr": {k: float(v) for k, v in SYMBOL_RR.items()},
        "symbol_brains": dict(SYMBOL_ENTRY_BRAIN),
        "symbol_scan_tfs": {k: list(v) for k, v in SYMBOL_SCAN_TFS.items()},
        "htf_entry_5m_exit": list(HTF_ENTRY_5M_EXIT),
    }


@router.patch("")
async def update_algo_settings(
    data: dict[str, Any] = Body(...),
    _: dict = Depends(get_current_user),
):
    """Update one or more runtime settings. Returns the full updated settings."""
    import runtime_config
    updated = runtime_config.update(data)
    return {"settings": updated}


@router.post("/reset")
async def reset_algo_settings(_: dict = Depends(get_current_user)):
    """Reset all settings to startup/.env defaults."""
    import runtime_config
    return {"settings": runtime_config.reset()}
