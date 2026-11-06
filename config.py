"""Trading algorithm configuration constants."""

import os
from pathlib import Path
from typing import Optional

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
# After TP1: trail SL to lock this fraction of max favorable move (0.70 = keep more runner).
TRAIL_PROFIT_LOCK = float(os.getenv("TRAIL_PROFIT_LOCK", "0.70"))
# Trailing: breakeven once this fraction of entry→TP distance is reached.
BREAKEVEN_PROGRESS = float(os.getenv("BREAKEVEN_PROGRESS", "0.35"))
# Legacy percent trail (unused when TRAIL_PROFIT_LOCK is active).
TRAIL_PERCENT = float(os.getenv("TRAIL_PERCENT", "0.0025"))

NORMAL_MIN_MC = 42
SCALP_MIN_MC = 40

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
PRICE_FETCH_SECONDS = float(os.getenv("PRICE_FETCH_SECONDS", "2"))
# Live MetaAPI tick poll (signal worker); keep ≥ price cache TTL
TICK_INTERVAL_LIVE_SECONDS = float(os.getenv("TICK_INTERVAL_LIVE_SECONDS", "1.0"))
# Live signal candle refresh — MetaAPI candles are cached; no need for 5s full refetch
POLL_INTERVAL_LIVE_SECONDS = int(os.getenv("POLL_INTERVAL_LIVE_SECONDS", "20"))

# TradingView data feed (anonymous mode works — no login needed)
TV_USE_LOGIN = os.getenv("TV_USE_LOGIN", "false").lower() in ("1", "true", "yes")
TV_USERNAME = os.getenv("TV_USERNAME", "") if TV_USE_LOGIN else ""
TV_PASSWORD = os.getenv("TV_PASSWORD", "") if TV_USE_LOGIN else ""
FETCH_RETRIES = int(os.getenv("FETCH_RETRIES", "4"))
FETCH_DELAY_SECONDS = float(os.getenv("FETCH_DELAY_SECONDS", "2.0"))
FETCH_BETWEEN_TF_DELAY = float(os.getenv("FETCH_BETWEEN_TF_DELAY", "1.5"))

# Price feed order: XM first (matches live broker), then other TradingView exchanges.
# OANDA/FX are fallbacks only — never preferred over XM for signal/price data.
_FETCH_SOURCES_RAW = [
    (SYMBOL, EXCHANGE),
    ("XAUUSD", "XM"),
    ("XAUUSD", "OANDA"),
    ("XAUUSD", "FX"),
    ("GOLD", "TVC"),
    ("XAUUSD", "FOREXCOM"),
    ("XAUUSD", "ICMARKETS"),
]
FETCH_SOURCES = []
_seen_sources = set()
for _pair in _FETCH_SOURCES_RAW:
    if _pair not in _seen_sources:
        _seen_sources.add(_pair)
        FETCH_SOURCES.append(_pair)
del _FETCH_SOURCES_RAW, _seen_sources, _pair

# Broker (XM via MT5 for live — paper mode needs no credentials)
# MT5 execution is wired when BROKER=xm and TRADE_MODE=live
BROKER = os.getenv("BROKER", "xm")
MT5_LOGIN = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")          # e.g. XMGlobal-MT5 2
MT5_PATH = os.getenv("MT5_PATH", "")              # e.g. C:\Program Files\XM Global MT5\terminal64.exe
MT5_SYMBOL = os.getenv("MT5_SYMBOL", "GOLD")   # XM demo often uses "GOLD" — confirm in MT5 Market Watch
MT5_BRIDGE_URL = os.getenv("MT5_BRIDGE_URL", "")  # optional HTTP bridge when MT5 runs on VPS

# MetaAPI cloud bridge — connects XM MT5 to a REST API callable from Mac/Docker.
# Sign up at app.metaapi.cloud, connect XM demo + live MT5 accounts, then paste:
METAAPI_TOKEN = os.getenv("METAAPI_TOKEN", "")  # API access token (shared)
# Legacy single-account id (used as demo fallback if DEMO is empty)
METAAPI_ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID", "")
METAAPI_ACCOUNT_ID_DEMO = os.getenv("METAAPI_ACCOUNT_ID_DEMO", "") or METAAPI_ACCOUNT_ID
METAAPI_ACCOUNT_ID_LIVE = os.getenv("METAAPI_ACCOUNT_ID_LIVE", "")
# Region-specific client API host (copy from your MetaAPI account page)
METAAPI_BASE_URL = os.getenv(
    "METAAPI_BASE_URL",
    "https://mt-client-api-v1.new-york.agiliumtrade.ai",
)
# Historical candles use a separate MetaAPI host (market-data client).
# Default: derive from METAAPI_BASE_URL by swapping the subdomain.
_default_md = METAAPI_BASE_URL.replace(
    "mt-client-api-v1", "mt-market-data-client-api-v1"
)
METAAPI_MARKET_DATA_URL = os.getenv("METAAPI_MARKET_DATA_URL", _default_md)
# Order comment tag sent to the broker (appears in MT5 trade history)
MT5_ORDER_COMMENT = os.getenv("MT5_ORDER_COMMENT", "zenith-algo")

# Legacy OANDA fields kept for fallback candle fetch (leave empty to disable)
OANDA_API_TOKEN = os.getenv("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
OANDA_INSTRUMENT = os.getenv("OANDA_INSTRUMENT", "XAU_USD")

# paper = simulated (no broker)
# demo  = XM demo account via MetaAPI (METAAPI_ACCOUNT_ID_DEMO)
# live  = XM real account via MetaAPI (METAAPI_ACCOUNT_ID_LIVE)
_raw_trade_mode = os.getenv("TRADE_MODE", "paper").lower().strip()
if _raw_trade_mode in ("paper", "demo", "live"):
    TRADE_MODE = _raw_trade_mode
elif _raw_trade_mode in ("1", "true", "yes"):
    TRADE_MODE = "live"
else:
    TRADE_MODE = "paper"

VALID_TRADE_MODES = ("paper", "demo", "live")


def is_broker_trade_mode(mode: Optional[str] = None) -> bool:
    """True when orders go to XM via MetaAPI (demo or live)."""
    m = (mode if mode is not None else TRADE_MODE) or "paper"
    return str(m).lower().strip() in ("demo", "live")


def metaapi_account_id_for_mode(mode: str) -> str:
    """Resolve MetaAPI account UUID for demo/live. Never fall live → demo."""
    m = str(mode or "").lower().strip()
    if m == "demo":
        return (METAAPI_ACCOUNT_ID_DEMO or METAAPI_ACCOUNT_ID or "").strip()
    if m == "live":
        return (METAAPI_ACCOUNT_ID_LIVE or "").strip()
    return ""
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
# Size so TP reward ≈ this USD; SL loss follows setup RR (1:2 → half).
TARGET_PROFIT_USD = float(os.getenv("TARGET_PROFIT_USD", "300"))
MAX_RISK_PCT_PER_TRADE = float(os.getenv("MAX_RISK_PCT_PER_TRADE", "0.02"))
# Hard ceiling on planned SL $ — kills index/crypto target-profit blow-ups
MAX_RISK_USD_PER_TRADE = float(os.getenv("MAX_RISK_USD_PER_TRADE", "200"))
DYNAMIC_LOTS_ENABLED = os.getenv("DYNAMIC_LOTS_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)

if TRADE_UNITS <= 0:
    TRADE_UNITS = int(round(TRADE_LOT_SIZE * UNITS_PER_LOT))

# Execution cadence: 5M production — TF-hop fills up to 10 legs per 5M candle
TRADE_5M_MODE = os.getenv("TRADE_5M_MODE", "true").lower() in ("1", "true", "yes")
MAX_TRADES_PER_5M = int(os.getenv("MAX_TRADES_PER_5M", "10"))
MAX_TRADES_PER_MINUTE = int(os.getenv("MAX_TRADES_PER_MINUTE", "10"))
TRADE_BUCKET_SECONDS = 300 if TRADE_5M_MODE else 60
MAX_TRADES_PER_BUCKET = MAX_TRADES_PER_5M if TRADE_5M_MODE else MAX_TRADES_PER_MINUTE
_trade_interval = os.getenv("TRADE_INTERVAL_SECONDS")
TRADE_INTERVAL_SECONDS = (
    float(_trade_interval)
    if _trade_interval
    else TRADE_BUCKET_SECONDS / max(MAX_TRADES_PER_BUCKET, 1)
)
# TF-hop stacking (legacy correlated clones) — prefer independent MTF scan
TF_HOP_ENABLED = os.getenv("TF_HOP_ENABLED", "false").lower() in ("1", "true", "yes")
TF_HOP_MAX_LEGS = int(os.getenv("TF_HOP_MAX_LEGS", "4"))
# Independent multi-TF scanner: each TF finds its own setup & hold
MTF_SCAN_ENABLED = os.getenv("MTF_SCAN_ENABLED", "true").lower() in ("1", "true", "yes")
MTF_MAX_CONCURRENT = int(os.getenv("MTF_MAX_CONCURRENT", "8"))
# Which TFs to scan (comma list). Backtest without 1m/3m still scans 5m/15m/1h/4h.
MTF_SCAN_TFS = tuple(
    t.strip()
    for t in os.getenv("MTF_SCAN_TFS", "1m,3m,5m,15m,1h,4h").split(",")
    if t.strip()
)
# Multi-instrument desk — same core algo across XM symbols
MULTI_INSTRUMENT_ENABLED = os.getenv("MULTI_INSTRUMENT_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
INSTRUMENT_UNIVERSE = tuple(
    s.strip().upper()
    for s in os.getenv(
        "INSTRUMENT_UNIVERSE",
        # GOLD fib sleeve + FX/index + crypto
        "GOLD,USDCHF,USDJPY,NAS100,UKOIL,AUDUSD,BTCUSD,ETHUSD",
    ).split(",")
    if s.strip()
)

# Hard halt — XM-cost walk-forward found no edge (see data/xm_cost_finetune_v2.json)
TRADING_HALT_SYMBOLS = frozenset(
    s.strip().upper()
    for s in os.getenv("TRADING_HALT_SYMBOLS", "BTCUSD,AUDUSD,USDCHF").split(",")
    if s.strip()
)


def is_trading_halted(symbol: Optional[str]) -> bool:
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    return sym in TRADING_HALT_SYMBOLS


def _parse_allocation(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in raw.split(","):
        if ":" not in part:
            continue
        sym, w = part.split(":", 1)
        try:
            out[sym.strip().upper()] = float(w)
        except ValueError:
            continue
    return out


# Desk risk allocation — NAS/crypto sized for live capital (not starved at 5%).
SYMBOL_ALLOCATION: dict[str, float] = _parse_allocation(
    os.getenv(
        "SYMBOL_ALLOCATION",
        "GOLD:0.10,USDCHF:0.16,USDJPY:0.12,NAS100:0.15,UKOIL:0.15,AUDUSD:0.16,BTCUSD:0.08,ETHUSD:0.08",
    )
)
# Soft correlation caps — allow concurrent markets; limit identical macro stacks
MAX_PER_CORR_GROUP = int(os.getenv("MAX_PER_CORR_GROUP", "2"))
# Risk-on meta-cap (NAS + crypto stack) — separate from FX/energy groups
MAX_PER_RISK_ON_GROUP = int(os.getenv("MAX_PER_RISK_ON_GROUP", "2"))
MAX_CONCURRENT_INSTRUMENT_TRADES = int(os.getenv("MAX_CONCURRENT_INSTRUMENT_TRADES", "6"))
# ETH overtrade brake — 5m bars between new ETH entries (book default is ENTRY_COOLDOWN_BARS)
ETH_ENTRY_COOLDOWN_BARS = int(os.getenv("ETH_ENTRY_COOLDOWN_BARS", "12"))
# Extra MC points required for ETH (on top of RL floor)
ETH_MC_FLOOR_DELTA = float(os.getenv("ETH_MC_FLOOR_DELTA", "8.0"))
BIAS_GATE_ENABLED = os.getenv("BIAS_GATE_ENABLED", "false").lower() in ("1", "true", "yes")
MIN_BIAS_CONFIDENCE = float(os.getenv("MIN_BIAS_CONFIDENCE", "70"))
# World-news layer (Forex Factory + Yahoo country desks) blended into overall bias.
# Physics (TF structure + intermarket) stays primary; news is an additive variable.
NEWS_BIAS_WEIGHT = float(os.getenv("NEWS_BIAS_WEIGHT", "0.18"))
# Soft MC nudge from world-news alignment (percentage points, capped).
NEWS_MC_ADJUST_MAX = float(os.getenv("NEWS_MC_ADJUST_MAX", "3.0"))
# Session volume profile (POC / VAH / VAL) — soft range + bias layer for MC/ML.
PROFILE_BIAS_WEIGHT = float(os.getenv("PROFILE_BIAS_WEIGHT", "0.14"))
PROFILE_MC_ADJUST_MAX = float(os.getenv("PROFILE_MC_ADJUST_MAX", "2.5"))
PROFILE_LOOKBACK_BARS = int(os.getenv("PROFILE_LOOKBACK_BARS", "288"))
DIRECTION_LOCK_5M = os.getenv(
    "DIRECTION_LOCK_5M", "true" if TRADE_5M_MODE else "false"
).lower() in ("1", "true", "yes")
BLOCK_OPPOSITE_OPEN = os.getenv("BLOCK_OPPOSITE_OPEN", "true").lower() in ("1", "true", "yes")

# SMC confirmation layer from xauusd_smc_monte_carlo_algo.ipynb
SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "3"))
SMC_LOOKBACK = int(os.getenv("SMC_LOOKBACK", "80"))
FVG_LOOKBACK = int(os.getenv("FVG_LOOKBACK", "25"))
DISPLACEMENT_ATR_MULT = float(os.getenv("DISPLACEMENT_ATR_MULT", "0.65"))
MIN_SMC_SCORE = float(os.getenv("MIN_SMC_SCORE", "2.5"))
CONSOLIDATION_MIN_SMC_SCORE = float(os.getenv("CONSOLIDATION_MIN_SMC_SCORE", "3.0"))
# Entry quality floors — cut chop / late-range bleed
ENTRY_MIN_SCORE = float(os.getenv("ENTRY_MIN_SCORE", "7.5"))
ENTRY_SCORE_MARGIN_ZONE = float(os.getenv("ENTRY_SCORE_MARGIN_ZONE", "1.25"))
ENTRY_SCORE_MARGIN_SETUP = float(os.getenv("ENTRY_SCORE_MARGIN_SETUP", "2.25"))
# Bars to wait after a close before next entry (0 = allow TF-hop stacks immediately)
ENTRY_COOLDOWN_BARS = int(os.getenv("ENTRY_COOLDOWN_BARS", "1"))
STRONG_SMC_MIN_MC = float(os.getenv("STRONG_SMC_MIN_MC", "25"))
# AVP / volume profile hard gate (POC–VAH–VAL location)
AVP_HARD_GATE = os.getenv("AVP_HARD_GATE", "true").lower() in ("1", "true", "yes")
ATR_RATIO_MIN = float(os.getenv("ATR_RATIO_MIN", "0.65"))
ATR_RATIO_MAX = float(os.getenv("ATR_RATIO_MAX", "2.00"))
# Capital protect — stop the post-run-up giveback (candle ~3401 class failures)
START_CAPITAL_USD = float(os.getenv("START_CAPITAL_USD", "1000"))
PEAK_DD_PROTECT_PCT = float(os.getenv("PEAK_DD_PROTECT_PCT", "0.15"))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "40"))
HARVEST_PCT = float(os.getenv("HARVEST_PCT", "0.50"))
# Bank when realized PnL ≥ this fraction of capital (1.0 = +100%)
BANK_TARGET_PCT = float(os.getenv("BANK_TARGET_PCT", "1.00"))

# Keeps the bot useful outside NY windows. Set false to enforce session filter.
ALLOW_ALL_SESSIONS = os.getenv("ALLOW_ALL_SESSIONS", "true").lower() in (
    "1",
    "true",
    "yes",
)

ML_GATE_ENABLED = os.getenv("ML_GATE_ENABLED", "true").lower() in ("1", "true", "yes")
# Coinbase book gate for BTCUSD/ETHUSD (L2 view of L3 engine). Fail-open if fetch dies.
CRYPTO_L3_GATE_ENABLED = os.getenv("CRYPTO_L3_GATE_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
CRYPTO_L3_FAIL_OPEN = os.getenv("CRYPTO_L3_FAIL_OPEN", "true").lower() in (
    "1",
    "true",
    "yes",
)
CRYPTO_L3_MAX_SPREAD_BPS = float(os.getenv("CRYPTO_L3_MAX_SPREAD_BPS", "8"))
CRYPTO_L3_MIN_TOP_NOTIONAL_USD = float(
    os.getenv("CRYPTO_L3_MIN_TOP_NOTIONAL_USD", "5000")
)
CRYPTO_L3_MIN_RISK_SPREAD_MULT = float(
    os.getenv("CRYPTO_L3_MIN_RISK_SPREAD_MULT", "4")
)
CRYPTO_L3_MAX_ADVERSE_IMBALANCE = float(
    os.getenv("CRYPTO_L3_MAX_ADVERSE_IMBALANCE", "0.55")
)
CRYPTO_L3_DEPTH_LEVELS = int(os.getenv("CRYPTO_L3_DEPTH_LEVELS", "10"))
# Firm hunt winners (Fourier/dist + widened SL) for BTCUSD/ETHUSD BB path
CRYPTO_FIRM_FORMULA_ENABLED = os.getenv(
    "CRYPTO_FIRM_FORMULA_ENABLED", "true"
).lower() in ("1", "true", "yes")
# Firm FX hunt winners (AUDUSD 3m / USDCHF 1m BB2.5 + Fourier HF)
FX_FIRM_FORMULA_ENABLED = os.getenv(
    "FX_FIRM_FORMULA_ENABLED", "true"
).lower() in ("1", "true", "yes")
# USDJPY opp-to-GOLD liquidity + corr + Fourier (lab winner)
USDJPY_OPP_GOLD_FIRM_ENABLED = os.getenv(
    "USDJPY_OPP_GOLD_FIRM_ENABLED", "true"
).lower() in ("1", "true", "yes")
# Crypto with-trend: BB only when 15m+1h agree (pullbacks with trend, not fade)
CRYPTO_TREND_SYNC_ENABLED = os.getenv(
    "CRYPTO_TREND_SYNC_ENABLED", "true"
).lower() in ("1", "true", "yes")
CRYPTO_TREND_SYNC_SYMBOLS = tuple(
    s.strip().upper()
    for s in os.getenv("CRYPTO_TREND_SYNC_SYMBOLS", "ETHUSD,BTCUSD").split(",")
    if s.strip()
)
CRYPTO_TREND_SYNC_MODE = (
    os.getenv("CRYPTO_TREND_SYNC_MODE", "ema") or "ema"
).strip().lower()  # ema | candle
# Lowered from 0.67 — multi-TP (1:3/1:4) is profitable at 52%+ WR due to asymmetric payoff
CALIBRATION_TARGET_WIN_RATE = float(os.getenv("CALIBRATION_TARGET_WIN_RATE", "0.52"))
MIN_TRAIN_SAMPLES = int(os.getenv("MIN_TRAIN_SAMPLES", "12"))
ADAPTIVE_LEARNING = os.getenv("ADAPTIVE_LEARNING", "true").lower() in ("1", "true", "yes")
LOSS_PNL_PROTECT = float(os.getenv("LOSS_PNL_PROTECT", "-120"))
LOSS_STREAK_REDUCE_AT = int(os.getenv("LOSS_STREAK_REDUCE_AT", "2"))
# Floor 42 — late-range chop after early winners needs fewer, better trades.
MC_FLOOR_BASE = float(os.getenv("MC_FLOOR_BASE", "42"))
# Soft penalty scale for stretching RR when final hit-rate is soft (was hard-block +5/+10).
MC_RR_PREMIUM = float(os.getenv("MC_RR_PREMIUM", "2.0"))
# Live Markov / path-MC floor hard-blocks. Off by default — Dukascopy scalp/agentic
# hunts that wired SYMBOL_ENTRY_BRAIN did NOT use per-entry Markov MC gates
# (only bootstrap MC on closed-trade R for SURVIVES). Keep false to match BT.
MARKOV_MC_GATE_ENABLED = os.getenv("MARKOV_MC_GATE_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
# Autopsy / self-heal: do not research, propose, or soft-apply for these symbols
# (fresh PA brains wired — leave them alone). Comma-separated.
# Empty string = exclude none (new-brain sessions). Missing env → legacy default.
_AUTOPSY_EXCLUDE_ENV = os.getenv("AUTOPSY_EXCLUDE_SYMBOLS")
_AUTOPSY_EXCLUDE_RAW = (
    "GOLD,USDCHF" if _AUTOPSY_EXCLUDE_ENV is None else _AUTOPSY_EXCLUDE_ENV
)
AUTOPSY_EXCLUDE_SYMBOLS = frozenset(
    s.strip().upper()
    for s in _AUTOPSY_EXCLUDE_RAW.split(",")
    if s.strip()
)
# Off by default — soft-apply / brain promote can tweak live baselines.
# Trade autopsy board stays read-only diagnostics.
AUTOPSY_SELF_HEAL_ENABLED = os.getenv("AUTOPSY_SELF_HEAL_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
AUTOPSY_IMPROVE_ENABLED = os.getenv("AUTOPSY_IMPROVE_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)


def is_autopsy_excluded(symbol: Optional[str]) -> bool:
    """True → skip self-heal / improve-agent proposals for this symbol."""
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    return sym in AUTOPSY_EXCLUDE_SYMBOLS

# If a strong SMC setup points past the nearest zone target, still allow 2R.
ALLOW_SMC_STRUCTURE_OVERRIDE = os.getenv("ALLOW_SMC_STRUCTURE_OVERRIDE", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Entry brain: zenith (zones/SMC/physics) | ema_20_50_pull (research winner)
ENTRY_BRAIN = (os.getenv("ENTRY_BRAIN", "ema_20_50_pull") or "ema_20_50_pull").strip().lower()
# GOLD-only brain (legacy; SYMBOL_ENTRY_BRAIN wins when set)
GOLD_ENTRY_BRAIN = (
    os.getenv("GOLD_ENTRY_BRAIN", "pa_liq_engulf") or "pa_liq_engulf"
).strip().lower()
GOLD_FIB_LEVELS = tuple(
    float(x.strip())
    for x in os.getenv("GOLD_FIB_LEVELS", "0.50,0.618").split(",")
    if x.strip()
)
# Restrict fib-brain MTF scan to these TFs (comma-separated). Research: 15m; user: also 1h.
GOLD_FIB_EXEC_TF = (os.getenv("GOLD_FIB_EXEC_TF", "15m,1h") or "15m,1h").strip().lower()
FIB_EXEC_TF = (os.getenv("FIB_EXEC_TF", GOLD_FIB_EXEC_TF) or GOLD_FIB_EXEC_TF).strip().lower()


def fib_exec_tfs() -> tuple[str, ...]:
    """Parse FIB_EXEC_TF / GOLD_FIB_EXEC_TF into a TF tuple (e.g. 15m,1h)."""
    raw = FIB_EXEC_TF or "15m"
    tfs = tuple(t.strip().lower() for t in str(raw).split(",") if t.strip())
    return tfs or ("15m",)


# Per-symbol scan TFs (entry). Research: NAS/UKOIL/AUD = HTF entry;
# USDCHF = pure 5m. Format: SYM:tf1|tf2,SYM:tf
# Example: NAS100:15m|1h|4h,USDCHF:5m
# Intraday hunt winners (1m/3m/5m/15m) — primary exec TF per pair
_DEFAULT_SYMBOL_SCAN_TFS = (
    "GOLD:15m,"
    "USDCHF:1m,"
    "USDJPY:15m,"
    "NAS100:15m,"
    "UKOIL:15m,"
    "AUDUSD:3m,"
    "BTCUSD:3m,"
    "ETHUSD:1m"
)


def _parse_symbol_scan_tfs(raw: str) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        sym, tfs_s = part.split(":", 1)
        sym_u = sym.strip().upper()
        tfs = tuple(
            t.strip().lower()
            for t in tfs_s.replace("|", ",").split(",")
            if t.strip()
        )
        if sym_u and tfs:
            out[sym_u] = tfs
    return out


SYMBOL_SCAN_TFS = _parse_symbol_scan_tfs(
    os.getenv("SYMBOL_SCAN_TFS", _DEFAULT_SYMBOL_SCAN_TFS)
)

# HTF entry + 5m manage/exit path (research A/B winners)
# Off by default after intraday-lab rewire (pure scan-TF entry)
HTF_ENTRY_5M_EXIT = tuple(
    s.strip().upper()
    for s in os.getenv("HTF_ENTRY_5M_EXIT", "").split(",")
    if s.strip()
)


def symbol_scan_tfs(symbol: Optional[str]) -> Optional[tuple[str, ...]]:
    """Per-symbol MTF entry filter, or None → desk default / fib rules."""
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    return SYMBOL_SCAN_TFS.get(sym)


def uses_htf_entry_5m_exit(symbol: Optional[str]) -> bool:
    sym = str(symbol or "").upper().strip()
    return sym in HTF_ENTRY_5M_EXIT

# Per-symbol entry brains. Format: SYM:brain,SYM:brain
# Intraday lab winners (1m/3m/5m/15m hunt) + legacy: fib | ema_* | sma | breakout
_DEFAULT_SYMBOL_BRAINS = (
    "GOLD:cta_10_40,"
    "USDCHF:rsi,"
    "USDJPY:bb_1.5,"
    "NAS100:ema_20_50_tight,"
    "UKOIL:st_3,"
    "AUDUSD:ema_20_50_pull,"
    "BTCUSD:fib_liq_sweep,"
    "ETHUSD:bb_2"
)


def _parse_symbol_entry_brains(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        sym, brain = part.split(":", 1)
        sym_u = sym.strip().upper()
        brain_l = brain.strip().lower()
        if sym_u and brain_l:
            out[sym_u] = brain_l
    return out


SYMBOL_ENTRY_BRAIN = _parse_symbol_entry_brains(
    os.getenv("SYMBOL_ENTRY_BRAIN", _DEFAULT_SYMBOL_BRAINS)
)


def symbol_entry_brain(symbol: Optional[str]) -> str:
    """Resolve entry brain for a symbol (SYMBOL map → GOLD legacy → book default)."""
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    if sym in SYMBOL_ENTRY_BRAIN:
        return SYMBOL_ENTRY_BRAIN[sym]
    if sym == "GOLD":
        return GOLD_ENTRY_BRAIN
    return ENTRY_BRAIN


def is_fib_entry_brain(brain: Optional[str]) -> bool:
    return str(brain or "").strip().lower() in ("fib_liq_sweep", "fib_liq")


# PA liq-engulf: per-symbol liquidity TF + bias pair (entry defaults 1m).
# USDCHF 2m-horizon trial: liq=15m · bias=5m+1m · RR=4
_DEFAULT_SYMBOL_PA_LIQ_TF = "GOLD:3m,USDCHF:15m,USDJPY:15m"
_DEFAULT_SYMBOL_PA_BIAS_TF = "GOLD:15m+5m,USDCHF:5m+1m,USDJPY:15m+5m"
_PA_TF_OK = frozenset({"1m", "3m", "5m", "15m"})


def _parse_symbol_pa_liq_tf(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        sym, tf = part.split(":", 1)
        sym_u = sym.strip().upper()
        tf_l = tf.strip().lower()
        if sym_u and tf_l in _PA_TF_OK:
            out[sym_u] = tf_l
    return out


def _parse_symbol_pa_bias_tf(raw: str) -> dict[str, tuple[str, str]]:
    """USDCHF:5m+1m → {'USDCHF': ('5m', '1m')} (higher TF first)."""
    out: dict[str, tuple[str, str]] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        sym, pair = part.split(":", 1)
        sym_u = sym.strip().upper()
        bits = [b.strip().lower() for b in pair.replace("/", "+").split("+") if b.strip()]
        if sym_u and len(bits) >= 2 and bits[0] in _PA_TF_OK and bits[1] in _PA_TF_OK:
            a, b = bits[0], bits[1]
            # higher TF → bias_a (matches lab combo_iter)
            mins = {"1m": 1, "3m": 3, "5m": 5, "15m": 15}
            if mins[a] < mins[b]:
                a, b = b, a
            out[sym_u] = (a, b)
    return out


SYMBOL_PA_LIQ_TF = _parse_symbol_pa_liq_tf(
    os.getenv("SYMBOL_PA_LIQ_TF", _DEFAULT_SYMBOL_PA_LIQ_TF)
)
SYMBOL_PA_BIAS_TF = _parse_symbol_pa_bias_tf(
    os.getenv("SYMBOL_PA_BIAS_TF", _DEFAULT_SYMBOL_PA_BIAS_TF)
)


def symbol_pa_liq_tf(symbol: Optional[str], default: str = "3m") -> str:
    """Liquidity TF for pa_liq_engulf."""
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    return SYMBOL_PA_LIQ_TF.get(sym) or default


def symbol_pa_bias_tfs(
    symbol: Optional[str], default: tuple[str, str] = ("15m", "5m")
) -> tuple[str, str]:
    """Dual bias TFs for pa_liq_engulf (higher TF first)."""
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    return SYMBOL_PA_BIAS_TF.get(sym) or default


def is_forced_2r_brain(brain: Optional[str]) -> bool:
    """True for research brains that pin TP to SYMBOL_RR (not structure grid)."""
    b = str(brain or "").strip().lower()
    if b in ("ema_tight",):
        b = "ema_20_50_tight"
    if b in ("ema_pull",):
        b = "ema_20_50_pull"
    return b in (
        "fib_liq_sweep",
        "fib_liq",
        "ema_20_50_pull",
        "ema_20_50_tight",
        "sma_20_50_cross",
        "breakout_20",
        "don_100",
        "donchian_100",
        "pa_liq_engulf",
        "pa_liq_1h25_engulf5",
        "gold_pa_liq",
        "pa_opp_gold",
        "usdjpy_opp_gold",
        "opp_gold_liq",
        "cta_10_40",
        "cta_10/40",
        "cta_50_200",
        "cta_50/200",
        "rsi",
        "rsi_mr",
        "bb_1.5",
        "bb_1_5",
        "bb_2",
        "bb_2.0",
        "bb_2.5",
        "bb_2_5",
        "roc_120",
        "st_3",
        "supertrend_3",
        "supertrend3",
        "volmom",
        "vol_scaled_mom",
        "vol_mom",
        "session_mix",
        "adaptive_session_mix",
        "circadian_kalman_z",
        "circadian_kalman",
        "adaptive_circadian_kalman_z",
        "entropy_tod",
        "adaptive_entropy_tod",
        "omni",
        "omni_horizon",
        "omnihorizon",
    )


# Per-symbol fixed RR from OOS/WF/MC sweep (Jan–Jul 2026, 1h proxy).
# Format: SYM:rr,SYM:rr  — supports floats (AUDUSD 1.5, USDCHF 2.4).
_DEFAULT_SYMBOL_RR = (
    "GOLD:4,"
    "USDCHF:2.4,"
    "USDJPY:4,"
    "NAS100:4,"
    "UKOIL:3,"
    "AUDUSD:1.5,"
    "BTCUSD:3,"
    "ETHUSD:4"
)


def _parse_symbol_rr(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        sym, rr_s = part.split(":", 1)
        sym_u = sym.strip().upper()
        try:
            rr_v = float(rr_s.strip())
        except Exception:
            continue
        if sym_u and rr_v > 0:
            out[sym_u] = rr_v
    return out


SYMBOL_RR = _parse_symbol_rr(os.getenv("SYMBOL_RR", _DEFAULT_SYMBOL_RR))


def symbol_rr(symbol: Optional[str], default: float = 2.0) -> float:
    """Research-backed RR multiple for a symbol (reward / risk)."""
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    if sym in SYMBOL_RR:
        return float(SYMBOL_RR[sym])
    return float(default)


def rr_profile_int(rr: float) -> int:
    """Nearest staged partial profile (2/3/4) for a float RR."""
    choices = list(MINUTE_RR_CHOICES) or [2, 3, 4]
    return int(min(choices, key=lambda x: abs(float(x) - float(rr))))

# Structure brain — preferred TP (+ optional soft entry) for OOS-green symbols.
# Mode: prefer = use structure TP when RR in band & MC ok; hybrid = only if EV≈grid; off = grid only.
STRUCTURE_TP_ENABLED = os.getenv("STRUCTURE_TP_ENABLED", "true").lower() in ("1", "true", "yes")
STRUCTURE_TP_MODE = (os.getenv("STRUCTURE_TP_MODE", "prefer") or "prefer").strip().lower()
STRUCTURE_TP_SYMBOLS = tuple(
    s.strip().upper()
    for s in os.getenv(
        "STRUCTURE_TP_SYMBOLS",
        "USDCHF,USDJPY,NAS100,UKOIL,AUDUSD,BTCUSD,ETHUSD",
    ).split(",")
    if s.strip()
)
STRUCTURE_TP_MIN_RR = float(os.getenv("STRUCTURE_TP_MIN_RR", "1.5"))
STRUCTURE_TP_MAX_RR = float(os.getenv("STRUCTURE_TP_MAX_RR", "4.0"))
# Physics filter on soft structure entries: require laminar + Graetz alignment.
PHYSICS_FILTER_ENABLED = os.getenv("PHYSICS_FILTER_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Soft entry when structure setup + physics filter clear but score is near the floor.
STRUCTURE_SOFT_ENTRY = os.getenv("STRUCTURE_SOFT_ENTRY", "false").lower() in (
    "1",
    "true",
    "yes",
)
STRUCTURE_ENTRY_MIN_SCORE = float(os.getenv("STRUCTURE_ENTRY_MIN_SCORE", "6.5"))

# Noise / loose-entry cuts (post structure-brain OOS)
# Require a real impulse candle (or SMC displacement) — no weak taps.
REQUIRE_DISPLACEMENT_ENTRY = os.getenv("REQUIRE_DISPLACEMENT_ENTRY", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Location AND pullback together (not OR spam) on structure symbols.
REQUIRE_ZONE_AND_PULLBACK = os.getenv("REQUIRE_ZONE_AND_PULLBACK", "true").lower() in (
    "1",
    "true",
    "yes",
)
# GOLD: laminar+Graetz hard block. USOIL: soft only (stricter score if misaligned).
PHYSICS_HARD_SYMBOLS = tuple(
    s.strip().upper()
    for s in os.getenv("PHYSICS_HARD_SYMBOLS", "").split(",")
    if s.strip()
)
PHYSICS_SOFT_SYMBOLS = tuple(
    s.strip().upper()
    for s in os.getenv(
        "PHYSICS_SOFT_SYMBOLS",
        "USDCHF,USDJPY,NAS100,UKOIL,AUDUSD,BTCUSD,ETHUSD",
    ).split(",")
    if s.strip()
)
PHYSICS_SOFT_SCORE_EXTRA = float(os.getenv("PHYSICS_SOFT_SCORE_EXTRA", "1.0"))
# Cap concurrent Zenith opens per symbol (1 = one GOLD + one USOIL max).
MAX_OPEN_PER_SYMBOL = int(os.getenv("MAX_OPEN_PER_SYMBOL", "1"))

STATE_PARAMS = {
    "LAMINAR_BULL": {"drift": 0.18, "vol": 0.90},
    "LAMINAR_BEAR": {"drift": -0.18, "vol": 0.90},
    "TURBULENT_BULL": {"drift": 0.08, "vol": 1.60},
    "TURBULENT_BEAR": {"drift": -0.08, "vol": 1.60},
    "TRANSITION_BULL": {"drift": 0.10, "vol": 1.20},
    "TRANSITION_BEAR": {"drift": -0.10, "vol": 1.20},
    "CHOP": {"drift": 0.00, "vol": 1.10},
}
