"""Trading algorithm configuration constants."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

SYMBOL = os.getenv("SYMBOL", "XAUUSD")
EXCHANGE = os.getenv("EXCHANGE", "XM")

BUFFER = 0.20
# Extra stop distance beyond the buffered zone edge (fraction of zone width).
SL_ZONE_EXTRA = float(os.getenv("SL_ZONE_EXTRA", "0.20"))
RR_CHOICES = [2]
MINUTE_RR = int(os.getenv("MINUTE_RR", "2"))
# Try all three RR profiles; algo picks the best expectancy.
MINUTE_RR_CHOICES = [2, 3, 4]
MINUTE_SL_ATR = float(os.getenv("MINUTE_SL_ATR", "1.0"))
MAX_SL_ATR = float(os.getenv("MAX_SL_ATR", "1.5"))
# Close this fraction of the position at TP1 (1R); runner trails to TP2 (2R).
TP1_PARTIAL_FRACTION = float(os.getenv("TP1_PARTIAL_FRACTION", "0.50"))

# --- Multi-TP booking fractions (fraction of FULL original position booked at each TP) ---
# 1:2  — 50% at TP1 (1R), runner to TP2 (2R)
# 1:3  — 33% at TP1, 33% at TP2, ~34% at TP3
# 1:4  — 30% at TP1, 25% at TP2, 25% at TP3, 20% at TP4
RR_TP_FRACS: dict[int, list[float]] = {
    2: [0.50],
    3: [0.333, 0.333],
    4: [0.30, 0.25, 0.25],
}
# After TP1: trail SL to lock this fraction of max favorable move (0.5 = 50% profit lock).
TRAIL_PROFIT_LOCK = float(os.getenv("TRAIL_PROFIT_LOCK", "0.50"))
# Trailing: breakeven once this fraction of entry→TP distance is reached (0.5 = 50%).
BREAKEVEN_PROGRESS = float(os.getenv("BREAKEVEN_PROGRESS", "0.50"))
# Legacy percent trail (unused when TRAIL_PROFIT_LOCK is active).
TRAIL_PERCENT = float(os.getenv("TRAIL_PERCENT", "0.0025"))

NORMAL_MIN_MC = 50
SCALP_MIN_MC = 45

BARS_1M = 500
BARS_3M = 500
BARS_5M = 500
BARS_15M = 300
BARS_1H = 300
BARS_4H = 300

MC_SIMULATIONS = 500
MC_STEPS = 12

POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
ERROR_RETRY_SECONDS = 60
TICK_INTERVAL_SECONDS = float(os.getenv("TICK_INTERVAL_SECONDS", "0.1"))
PRICE_FETCH_SECONDS = float(os.getenv("PRICE_FETCH_SECONDS", "1"))

# TradingView data feed (anonymous mode works — no login needed)
TV_USE_LOGIN = os.getenv("TV_USE_LOGIN", "false").lower() in ("1", "true", "yes")
TV_USERNAME = os.getenv("TV_USERNAME", "") if TV_USE_LOGIN else ""
TV_PASSWORD = os.getenv("TV_PASSWORD", "") if TV_USE_LOGIN else ""
FETCH_RETRIES = int(os.getenv("FETCH_RETRIES", "4"))
FETCH_DELAY_SECONDS = float(os.getenv("FETCH_DELAY_SECONDS", "2.0"))
FETCH_BETWEEN_TF_DELAY = float(os.getenv("FETCH_BETWEEN_TF_DELAY", "1.5"))

# Fallback symbol/exchange pairs if primary fails (XM first, then common fallbacks)
FETCH_SOURCES = [
    (SYMBOL, EXCHANGE),
    ("XAUUSD", "XM"),
    ("XAUUSD", "OANDA"),
    ("XAUUSD", "FX"),
    ("GOLD", "TVC"),
    ("XAUUSD", "FOREXCOM"),
    ("XAUUSD", "ICMARKETS"),
]

# Broker (XM via MT5 for live — paper mode needs no credentials)
# MT5 execution is wired when BROKER=xm and TRADE_MODE=live
BROKER = os.getenv("BROKER", "xm")
MT5_LOGIN = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")          # e.g. XMGlobal-MT5 2
MT5_PATH = os.getenv("MT5_PATH", "")              # e.g. C:\Program Files\XM Global MT5\terminal64.exe
MT5_SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSD")   # confirm in MT5 Market Watch
MT5_BRIDGE_URL = os.getenv("MT5_BRIDGE_URL", "")  # optional HTTP bridge when MT5 runs on VPS

# Legacy OANDA fields kept for fallback candle fetch (leave empty to disable)
OANDA_API_TOKEN = os.getenv("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
OANDA_INSTRUMENT = os.getenv("OANDA_INSTRUMENT", "XAU_USD")

# paper = simulated trades (no broker account needed)
# live  = real orders via XM/MT5 (or OANDA if BROKER=oanda)
TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()
AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
TRADE_UNITS = int(os.getenv("TRADE_UNITS", "0"))  # 0 = derive from lot size
TRADE_LOT_SIZE = float(os.getenv("TRADE_LOT_SIZE", os.getenv("LOT_SIZE", "0.1")))
# OANDA XAU_USD: 1 standard lot = 100 troy oz → 0.1 lot = 10 units.
UNITS_PER_LOT = float(os.getenv("UNITS_PER_LOT", "100"))
MARGIN_LEVERAGE = float(os.getenv("MARGIN_LEVERAGE", "1000"))

if TRADE_UNITS <= 0:
    TRADE_UNITS = int(round(TRADE_LOT_SIZE * UNITS_PER_LOT))

# Execution cadence: 5M production (3M/5M zones) or 1M test mode (1M/5M zones)
TRADE_5M_MODE = os.getenv("TRADE_5M_MODE", "true").lower() in ("1", "true", "yes")
MAX_TRADES_PER_5M = int(os.getenv("MAX_TRADES_PER_5M", "3"))
MAX_TRADES_PER_MINUTE = int(os.getenv("MAX_TRADES_PER_MINUTE", "5"))
TRADE_BUCKET_SECONDS = 300 if TRADE_5M_MODE else 60
MAX_TRADES_PER_BUCKET = MAX_TRADES_PER_5M if TRADE_5M_MODE else MAX_TRADES_PER_MINUTE
_trade_interval = os.getenv("TRADE_INTERVAL_SECONDS")
TRADE_INTERVAL_SECONDS = (
    float(_trade_interval)
    if _trade_interval
    else TRADE_BUCKET_SECONDS / max(MAX_TRADES_PER_BUCKET, 1)
)
BIAS_GATE_ENABLED = os.getenv("BIAS_GATE_ENABLED", "false").lower() in ("1", "true", "yes")
MIN_BIAS_CONFIDENCE = float(os.getenv("MIN_BIAS_CONFIDENCE", "70"))
DIRECTION_LOCK_5M = os.getenv(
    "DIRECTION_LOCK_5M", "true" if TRADE_5M_MODE else "false"
).lower() in ("1", "true", "yes")
BLOCK_OPPOSITE_OPEN = os.getenv("BLOCK_OPPOSITE_OPEN", "true").lower() in ("1", "true", "yes")

# SMC confirmation layer from xauusd_smc_monte_carlo_algo.ipynb
SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "3"))
SMC_LOOKBACK = int(os.getenv("SMC_LOOKBACK", "80"))
FVG_LOOKBACK = int(os.getenv("FVG_LOOKBACK", "25"))
DISPLACEMENT_ATR_MULT = float(os.getenv("DISPLACEMENT_ATR_MULT", "0.65"))
MIN_SMC_SCORE = float(os.getenv("MIN_SMC_SCORE", "2.0"))
CONSOLIDATION_MIN_SMC_SCORE = float(os.getenv("CONSOLIDATION_MIN_SMC_SCORE", "2.5"))
STRONG_SMC_MIN_MC = float(os.getenv("STRONG_SMC_MIN_MC", "25"))

# Keeps the bot useful outside NY windows. Set false to enforce session filter.
ALLOW_ALL_SESSIONS = os.getenv("ALLOW_ALL_SESSIONS", "true").lower() in (
    "1",
    "true",
    "yes",
)

ML_GATE_ENABLED = os.getenv("ML_GATE_ENABLED", "true").lower() in ("1", "true", "yes")
CALIBRATION_TARGET_WIN_RATE = float(os.getenv("CALIBRATION_TARGET_WIN_RATE", "0.67"))
MIN_TRAIN_SAMPLES = int(os.getenv("MIN_TRAIN_SAMPLES", "12"))
ADAPTIVE_LEARNING = os.getenv("ADAPTIVE_LEARNING", "true").lower() in ("1", "true", "yes")
LOSS_PNL_PROTECT = float(os.getenv("LOSS_PNL_PROTECT", "-150"))
LOSS_STREAK_REDUCE_AT = int(os.getenv("LOSS_STREAK_REDUCE_AT", "3"))
MC_FLOOR_BASE = float(os.getenv("MC_FLOOR_BASE", "50"))

# If a strong SMC setup points past the nearest zone target, still allow 2R.
ALLOW_SMC_STRUCTURE_OVERRIDE = os.getenv("ALLOW_SMC_STRUCTURE_OVERRIDE", "true").lower() in (
    "1",
    "true",
    "yes",
)

STATE_PARAMS = {
    "LAMINAR_BULL": {"drift": 0.18, "vol": 0.90},
    "LAMINAR_BEAR": {"drift": -0.18, "vol": 0.90},
    "TURBULENT_BULL": {"drift": 0.08, "vol": 1.60},
    "TURBULENT_BEAR": {"drift": -0.08, "vol": 1.60},
    "TRANSITION_BULL": {"drift": 0.10, "vol": 1.20},
    "TRANSITION_BEAR": {"drift": -0.10, "vol": 1.20},
    "CHOP": {"drift": 0.00, "vol": 1.10},
}
