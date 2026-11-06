"""Canonical XM desk instrument metadata — asset class, sessions, digits, contracts.

Universe names match XM MetaAPI Market Watch when possible:
  GOLD, SILVER, USOIL, US30, EURUSD, GBPUSD, USDJPY, BTCUSD, …
"""

from __future__ import annotations

from typing import Callable, Optional

from config import UNITS_PER_LOT

# Soft entry (MC/ML) — metals & CFD commodities should not freeze auto after losses
_SOFT_ENTRY = frozenset({
    "GOLD", "XAUUSD", "SILVER", "XAGUSD",
    "USOIL", "UKOIL", "XTIUSD", "XBRUSD", "OIL", "OILCASH", "BRENT",
    "US30", "US30CASH", "US500", "US500CASH", "NAS100", "USTEC", "SPX500", "US100", "US100CASH",
})

_METALS = frozenset({"GOLD", "XAUUSD", "SILVER", "XAGUSD"})
_ENERGY = frozenset({"USOIL", "UKOIL", "XTIUSD", "XBRUSD", "OIL", "OILCASH", "BRENT"})
_INDICES = frozenset({
    "US30", "US30CASH", "US500", "US500CASH", "NAS100", "USTEC", "SPX500",
    "US100", "US100CASH",
})
_CRYPTO = frozenset({"BTCUSD", "BTC", "ETHUSD", "ETH"})


def norm(symbol: Optional[str] = None) -> str:
    return str(symbol or "GOLD").upper().strip()


def asset_class(symbol: Optional[str] = None) -> str:
    s = norm(symbol)
    # Strip Cash / # suffixes used by XM cash CFDs
    base = s.replace("CASH", "").rstrip("#")
    if s in ("GOLD", "XAUUSD") or base in ("GOLD", "XAUUSD"):
        return "gold"
    if s in ("SILVER", "XAGUSD") or base in ("SILVER", "XAGUSD"):
        return "silver"
    if s in _CRYPTO or s.startswith("BTC") or s.startswith("ETH"):
        return "crypto"
    if s in _INDICES or base in ("US30", "US500", "NAS100", "USTEC", "SPX500", "US100"):
        return "index"
    if s in _ENERGY or base in ("USOIL", "UKOIL", "OIL", "BRENT", "XTIUSD", "XBRUSD"):
        return "energy"
    if len(s) == 6 and s.isalpha():
        return "fx"
    return "gold"


def bias_asset_class(symbol: Optional[str] = None) -> str:
    """Coarser class for macro / World desk filters."""
    cls = asset_class(symbol)
    if cls in ("gold", "silver"):
        return "metals"
    if cls == "index":
        return "indices"
    return cls


def is_metal(symbol: Optional[str] = None) -> bool:
    return norm(symbol) in _METALS


def is_energy(symbol: Optional[str] = None) -> bool:
    return norm(symbol) in _ENERGY


def is_index(symbol: Optional[str] = None) -> bool:
    return norm(symbol) in _INDICES


def soft_entry_gates(symbol: Optional[str] = None) -> bool:
    """True → don't stack RL MC/ML walls; skip broken ML when desk MC clears."""
    return norm(symbol) in _SOFT_ENTRY


def price_digits(symbol: Optional[str] = None) -> int:
    s = norm(symbol)
    if s in ("SILVER", "XAGUSD"):
        return 3
    if s in ("GOLD", "XAUUSD") or s in _CRYPTO or s.startswith("BTC") or s.startswith("ETH"):
        return 2
    if s in _ENERGY:
        return 2
    if s in _INDICES:
        return 1
    if s.endswith("JPY") or "JPY" in s:
        return 3
    if len(s) == 6 and s.isalpha():
        return 5
    return 2


def contract_units_per_lot(symbol: Optional[str] = None) -> float:
    """XM CFD contract size per 1.00 lot (typical cash symbols)."""
    s = norm(symbol)
    if s in ("GOLD", "XAUUSD"):
        return float(UNITS_PER_LOT)  # usually 100 oz
    if s in ("SILVER", "XAGUSD"):
        return 5000.0
    if s in _ENERGY:
        return 100.0  # barrels
    if s in _INDICES:
        return 1.0  # $1 / point per lot on XM cash
    if s in _CRYPTO or s.startswith("BTC") or s.startswith("ETH"):
        return 1.0
    if len(s) == 6 and s.isalpha():
        return 100_000.0
    return float(UNITS_PER_LOT)


def min_stop_distance(symbol: Optional[str] = None) -> float:
    s = norm(symbol)
    if s in ("BTCUSD", "BTC") or s.startswith("BTC"):
        return 150.0
    if s in ("GOLD", "XAUUSD"):
        return 5.0
    if s in ("SILVER", "XAGUSD"):
        return 0.08
    if s in _ENERGY:
        return 0.25
    if s in _INDICES:
        return 15.0
    if len(s) == 6 and s.isalpha():
        return 0.0015
    return 0.12


def uses_gold_session(symbol: Optional[str] = None) -> bool:
    """Metals, oil, and US indices on XM roughly share gold CFD hours."""
    s = norm(symbol)
    return s in _METALS or s in _ENERGY or s in _INDICES


def instrument_tradable(symbol: Optional[str] = None) -> bool:
    from datetime import datetime, timezone

    from analysis.market_hours import is_gold_market_open

    s = norm(symbol)
    if uses_gold_session(s):
        return is_gold_market_open()
    if s in _CRYPTO or s.startswith("BTC") or s.startswith("ETH"):
        return True
    # Live: wall clock. Backtest: injected bar time (else FX looks "closed" all weekend).
    try:
        from analysis.session import get_backtest_bar_time

        now = get_backtest_bar_time() or datetime.now(timezone.utc)
    except Exception:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    if now.weekday() == 5:
        return False
    if now.weekday() == 6 and now.hour < 22:
        return False
    if now.weekday() == 4 and now.hour >= 21:
        return False
    return True


def get_desk_market_status(now=None) -> dict:
    """Desk-level hours: open if any universe symbol is tradable (crypto weekends)."""
    from analysis.market_hours import get_gold_market_status
    from config import INSTRUMENT_UNIVERSE, MULTI_INSTRUMENT_ENABLED

    gold = get_gold_market_status(now)
    universe = list(INSTRUMENT_UNIVERSE) if MULTI_INSTRUMENT_ENABLED else ["GOLD"]
    open_syms = [s for s in universe if instrument_tradable(s)]
    crypto_open = [
        s
        for s in open_syms
        if asset_class(s) == "crypto" or s.startswith("BTC") or s.startswith("ETH")
    ]

    if open_syms:
        if not gold["is_open"] and crypto_open:
            try:
                from config import symbol_entry_brain

                brain_bits = [
                    f"{s}:{symbol_entry_brain(s)}" for s in crypto_open
                ]
                brain_txt = " · ".join(brain_bits)
            except Exception:
                brain_txt = ", ".join(crypto_open)
            return {
                "is_open": True,
                "status": "crypto_session",
                "label": "Crypto open · FX/metals closed",
                "message": (
                    "Crypto market is open 24/7 — "
                    f"{brain_txt}. "
                    "FX, metals, energy, and indices resume with XM session."
                ),
                "next_open": gold.get("next_open"),
                "next_open_iso": gold.get("next_open_iso"),
                "next_open_display": gold.get("next_open_display"),
                "server_time_display": gold.get("server_time_display"),
                "open_symbols": open_syms,
                "crypto_symbols": crypto_open,
            }
        out = dict(gold)
        out["is_open"] = True
        out["open_symbols"] = open_syms
        out["crypto_symbols"] = crypto_open
        return out

    out = dict(gold)
    out["is_open"] = False
    out["open_symbols"] = []
    out["crypto_symbols"] = []
    return out


def tv_fetch_sources(symbol: Optional[str] = None) -> list[tuple[str, str]]:
    """TradingView (symbol, exchange) candidates for paper / history fallback."""
    s = norm(symbol)
    if s in ("GOLD", "XAUUSD"):
        return []  # use primary gold fetcher
    if s in ("SILVER", "XAGUSD"):
        return [
            ("SILVER", "XM"),
            ("XAGUSD", "XM"),
            ("XAGUSD", "OANDA"),
            ("SILVER", "TVC"),
            ("XAGUSD", "FOREXCOM"),
        ]
    if s in ("USOIL", "XTIUSD", "OIL", "OILCASH"):
        return [
            ("OIL", "XM"),
            ("OILCash", "XM"),
            ("USOIL", "XM"),
            ("USOIL", "FOREXCOM"),
            ("USOIL", "ICMARKETS"),
            ("WTIUSD", "TVC"),
            ("CL1!", "NYMEX"),
        ]
    if s in ("UKOIL", "XBRUSD", "BRENT"):
        return [
            ("BRENT", "XM"),
            ("UKOIL", "XM"),
            ("UKOIL", "FOREXCOM"),
            ("UKOIL", "ICMARKETS"),
        ]
    if s in ("US30", "US30CASH"):
        return [
            ("US30Cash", "XM"),
            ("US30", "XM"),
            ("US30", "FOREXCOM"),
            ("US30", "ICMARKETS"),
            ("DJI", "TVC"),
            ("YM1!", "CBOT"),
        ]
    if s in ("US500", "SPX500"):
        return [
            ("US500Cash", "XM"),
            (s, "XM"),
            ("SPX", "TVC"),
            (s, "FOREXCOM"),
        ]
    if s in ("NAS100", "USTEC", "US100"):
        return [
            ("US100Cash", "XM"),
            ("NAS100", "XM"),
            ("NDX", "TVC"),
            ("NAS100", "FOREXCOM"),
        ]
    if s in ("BTCUSD", "BTC"):
        return [
            ("BTCUSD", "XM"),
            ("BTCUSD", "BINANCE"),
            ("BTCUSD", "BITSTAMP"),
            ("BTCUSD", "COINBASE"),
        ]
    if s in ("ETHUSD", "ETH"):
        return [
            ("ETHUSD", "XM"),
            ("ETHUSD", "BINANCE"),
            ("ETHUSD", "BITSTAMP"),
            ("ETHUSD", "COINBASE"),
        ]
    return [
        (s, "XM"),
        (s, "FX_IDC"),
        (s, "OANDA"),
        (s, "FOREXCOM"),
        (s, "ICMARKETS"),
    ]


# Desk name → ordered XM MetaAPI candidates (first that exists on the account wins)
_BROKER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "GOLD": ("GOLD", "XAUUSD"),
    "XAUUSD": ("GOLD", "XAUUSD"),
    "SILVER": ("SILVER", "XAGUSD"),
    "XAGUSD": ("SILVER", "XAGUSD"),
    "USOIL": ("OIL", "OILCash", "OILCash#", "USOIL", "XTIUSD"),
    "OIL": ("OIL", "OILCash", "OILCash#"),
    "UKOIL": ("BRENT", "UKOIL", "XBRUSD", "BRENTCash"),
    "BRENT": ("BRENT", "UKOIL", "XBRUSD", "BRENTCash"),
    "US30": ("US30Cash", "US30Cash#", "US30", "DJ30"),
    "US500": ("US500Cash", "US500Cash#", "US500", "SPX500"),
    "NAS100": ("US100Cash", "NAS100", "USTEC", "US100"),
    "BTCUSD": ("BTCUSD", "BTC"),
    "ETHUSD": ("ETHUSD", "ETH"),
    "EURUSD": ("EURUSD",),
    "GBPUSD": ("GBPUSD",),
    "USDJPY": ("USDJPY",),
    "USDCHF": ("USDCHF",),
    "AUDUSD": ("AUDUSD",),
}

_resolved_broker: dict[str, str] = {}


def broker_candidates(symbol: Optional[str] = None) -> tuple[str, ...]:
    s = norm(symbol)
    hit = _BROKER_CANDIDATES.get(s)
    if hit:
        return hit
    # Already a wire name (OILCash / US30Cash) — map back to desk candidates.
    # Critical: do NOT return uppercased OILCASH; XM symbol names are case-sensitive.
    for _desk, cands in _BROKER_CANDIDATES.items():
        if s in {c.upper() for c in cands}:
            return cands
    raw = str(symbol or "").strip()
    return (raw,) if raw else (s,)


def resolve_broker_symbol(
    symbol: Optional[str] = None,
    *,
    probe: Optional[Callable[[str], bool]] = None,
) -> str:
    """Map desk symbol (USOIL/US30) to the live XM Market Watch name.

    `probe(sym) -> bool` should return True when the symbol exists on MetaAPI.
    Without probe, returns the first candidate (best XM default).
    """
    raw = str(symbol or "").strip()
    s = norm(raw)
    if s in _resolved_broker:
        return _resolved_broker[s]
    # Also hit cache if caller already passed a wire name
    if raw and raw in _resolved_broker.values():
        return raw

    cands = broker_candidates(raw or s)
    if probe is None:
        wire = cands[0]
        _resolved_broker[s] = wire
        if raw:
            _resolved_broker[norm(raw)] = wire
        return wire
    for c in cands:
        try:
            if probe(c):
                _resolved_broker[s] = c
                return c
        except Exception:
            continue
    wire = cands[0]
    _resolved_broker[s] = wire
    return wire


def clear_broker_resolution_cache() -> None:
    _resolved_broker.clear()


def desk_symbol_from_broker(broker_sym: Optional[str] = None) -> str:
    """Reverse map OIL → USOIL for RL / UI when XM returns native names."""
    b = norm(broker_sym)
    for desk, cands in _BROKER_CANDIDATES.items():
        # XM wire names are mixed-case (US100Cash, BRENTCash); norm() is UPPER —
        # compare case-insensitively or NAS100/UKOIL closes drop out of Realised.
        cand_u = {c.upper() for c in cands}
        if b == desk or b in cand_u:
            # Prefer universe-facing names
            if desk in ("OIL",):
                return "USOIL"
            if desk in ("BRENT",):
                return "UKOIL"
            return desk
    return b
