"""Light correlation groups for multi-instrument risk.

Same algo may fire on EURUSD + GBPUSD + USDJPY together — that's allowed up to
MAX_PER_CORR_GROUP. Blocks only identical-group over-stacking.

Also enforces a risk-on meta-cap (NAS + crypto) via MAX_PER_RISK_ON_GROUP so
index + BTC + ETH cannot all stack at once (desk mastery E).
"""

from __future__ import annotations

from typing import Any, Optional

from config import MAX_PER_CORR_GROUP

try:
    from config import MAX_PER_RISK_ON_GROUP
except Exception:  # pragma: no cover
    MAX_PER_RISK_ON_GROUP = 2

# Static Phase-1 groups (expand later; keep core algo untouched)
_GROUP_MEMBERS: dict[str, tuple[str, ...]] = {
    "metals": ("GOLD", "XAUUSD", "SILVER", "XAGUSD"),
    "fx_usd": ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"),
    "us_indices": (
        "US30", "US30CASH", "US500", "US500CASH", "NAS100", "USTEC", "SPX500",
        "US100", "US100CASH",
    ),
    "energy": ("USOIL", "UKOIL", "XTIUSD", "XBRUSD", "OIL", "OILCASH", "BRENT"),
    "crypto": ("BTCUSD", "BTC", "ETHUSD", "ETH"),
}

# Cross-group stack that behaves as one risk factor
_RISK_ON_MEMBERS: frozenset[str] = frozenset(
    {
        "NAS100", "USTEC", "US100", "US100CASH", "US30", "US30CASH",
        "US500", "US500CASH", "SPX500",
        "BTCUSD", "BTC", "ETHUSD", "ETH",
    }
)

_SYMBOL_TO_GROUP: dict[str, str] = {}
for _g, _syms in _GROUP_MEMBERS.items():
    for _s in _syms:
        _SYMBOL_TO_GROUP[_s.upper()] = _g


def correlation_group(symbol: str) -> str:
    s = str(symbol or "").upper().strip()
    return _SYMBOL_TO_GROUP.get(s, f"solo:{s}")


def is_risk_on(symbol: str) -> bool:
    return str(symbol or "").upper().strip() in _RISK_ON_MEMBERS


def open_symbols_from_rows(rows: list[Any]) -> list[str]:
    """Extract broker/local symbols from open trade rows or position dicts."""
    out: list[str] = []
    for r in rows:
        if isinstance(r, dict):
            sym = r.get("symbol") or r.get("SYMBOL")
            if not sym:
                feats = r.get("features") or {}
                if isinstance(feats, dict):
                    sym = feats.get("symbol") or feats.get("SYMBOL")
            if sym:
                out.append(str(sym).upper())
            continue
        sym = getattr(r, "symbol", None)
        if not sym:
            feats = getattr(r, "features", None) or {}
            if isinstance(feats, str):
                try:
                    import json

                    feats = json.loads(feats)
                except Exception:
                    feats = {}
            if isinstance(feats, dict):
                sym = feats.get("symbol") or feats.get("SYMBOL")
        if sym:
            out.append(str(sym).upper())
    return out


def can_open_symbol(
    symbol: str,
    open_symbols: list[str],
    *,
    max_per_group: Optional[int] = None,
    max_risk_on: Optional[int] = None,
) -> tuple[bool, str]:
    """
    Allow concurrent markets. Block when primary corr group OR risk-on meta is full.
    """
    cap = int(max_per_group if max_per_group is not None else MAX_PER_CORR_GROUP)
    risk_cap = int(
        max_risk_on if max_risk_on is not None else MAX_PER_RISK_ON_GROUP
    )
    sym = str(symbol or "").upper().strip()
    if not sym:
        return False, "Missing symbol"
    group = correlation_group(sym)
    peers = [s for s in open_symbols if correlation_group(s) == group]
    # Already holding this exact symbol counts; still allow other TFs via exec_tf elsewhere
    if len(peers) >= cap and sym not in peers:
        return (
            False,
            f"Correlation cap — {group} already has {len(peers)}/{cap} symbols open",
        )

    # Risk-on meta: NAS + BTC + ETH share one stack budget
    if is_risk_on(sym):
        risk_peers = [s for s in open_symbols if is_risk_on(s)]
        if len(risk_peers) >= risk_cap and sym not in risk_peers:
            return (
                False,
                f"Risk-on cap — already {len(risk_peers)}/{risk_cap} "
                f"(NAS/crypto stack) open",
            )
    return True, "OK"
