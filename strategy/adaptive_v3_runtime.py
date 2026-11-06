"""Adaptive v3 live efficiency layer (no dual-horizon GOLD/JPY primary swap).

· Night sit-out (FX/oil Asia deep night)
· Horizon router (short alt book when rolling 7d < −2R)
· Size throttle (risk_scale from rolling 7d)
· Monthly rescore of short-alt table from dual-horizon file
· 1m densify handled in signal path (scan 1m+5m)
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_ROOT = Path(__file__).resolve().parents[1]
_STATE = _ROOT / "data" / "adaptive_v3_runtime.json"
_DUAL = _ROOT / "data" / "lab_adaptive_dual_horizon_policy.json"
_RISK_USD = 200.0

# Primary brains stay as wired in .env (absolute 1y). Short alts for horizon router.
DEFAULT_SHORT_ALT = {
    "ETHUSD": "circadian_kalman_z",
    "BTCUSD": "session_mix",
    "GOLD": "session_mix",
    "NAS100": "session_mix",
    "USDJPY": "session_mix",
    "AUDUSD": "session_mix",
    "USDCHF": "circadian_kalman_z",
    "UKOIL": "circadian_kalman_z",
}

NIGHT_SITOUT = frozenset({"AUDUSD", "USDCHF", "USDJPY", "UKOIL"})
# XM Off-session (21–24 UTC) — GOLD/UKOIL improve under Off sit-out in fine-tune v2
OFF_SESSION_SITOUT = frozenset({"GOLD", "UKOIL"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _empty() -> dict[str, Any]:
    return {
        "updated_at": None,
        "last_rescore_at": None,
        "short_alt": dict(DEFAULT_SHORT_ALT),
        "per_symbol": {},
        "note": "adaptive v3 runtime — primary brains from .env",
    }


def load_state() -> dict[str, Any]:
    with _LOCK:
        if not _STATE.exists():
            return _empty()
        try:
            raw = json.loads(_STATE.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return _empty()
            raw.setdefault("short_alt", dict(DEFAULT_SHORT_ALT))
            raw.setdefault("per_symbol", {})
            return raw
        except Exception as exc:
            logger.warning("adaptive_v3_runtime load: %s", exc)
            return _empty()


def save_state(st: dict[str, Any]) -> None:
    with _LOCK:
        st = dict(st)
        st["updated_at"] = _now().isoformat()
        _STATE.parent.mkdir(parents=True, exist_ok=True)
        _STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def maybe_monthly_rescore() -> dict[str, Any]:
    """Refresh short-alt map from dual-horizon file at most once / 30d."""
    st = load_state()
    last = st.get("last_rescore_at")
    due = True
    if last:
        try:
            dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
            due = (_now() - dt) >= timedelta(days=30)
        except Exception:
            due = True
    if not due:
        return st
    alts = dict(DEFAULT_SHORT_ALT)
    if _DUAL.exists():
        try:
            dual = json.loads(_DUAL.read_text(encoding="utf-8"))
            per = dual.get("per_pair") or {}
            for sym, row in per.items():
                if row.get("error"):
                    continue
                # short-horizon best mode among live adaptive brains only
                bs = (row.get("best_short") or {}).get("mode")
                if bs in ("session_mix", "circadian_kalman_z", "entropy_tod"):
                    alts[str(sym).upper()] = bs
        except Exception as exc:
            logger.warning("monthly rescore dual read: %s", exc)
    st["short_alt"] = alts
    st["last_rescore_at"] = _now().isoformat()
    save_state(st)
    return st


_pnl_cache: dict[str, tuple[float, float]] = {}  # sym -> (monotonic_ts, pnl)


def _rolling_7d_pnl(symbol: str) -> Optional[float]:
    """Sum closed PnL for symbol in last 7 days (history kept; not wiped)."""
    import os
    import time

    now_m = time.monotonic()
    cached = _pnl_cache.get(symbol)
    if cached and (now_m - cached[0]) < 60.0:
        return cached[1]

    pnl = 0.0
    try:
        import asyncio

        import asyncpg

        url = os.getenv("DATABASE_URL", "postgresql://algo:algo@postgres:5432/algo")
        if "postgres:5432" in url and not Path("/.dockerenv").exists():
            url = url.replace("@postgres:5432", "@127.0.0.1:5432")
        cut = _now() - timedelta(days=7)

        async def _fetch() -> list:
            conn = await asyncpg.connect(url)
            try:
                return await conn.fetch(
                    """
                    SELECT pnl, features FROM trade_records
                    WHERE UPPER(symbol) = $1
                      AND (status = 'closed' OR pnl_status LIKE 'closed%')
                      AND created_at >= $2
                    """,
                    symbol.upper(),
                    cut,
                )
            finally:
                await conn.close()

        try:
            rows = asyncio.get_event_loop().run_until_complete(_fetch())
        except RuntimeError:
            rows = asyncio.run(_fetch())
        for r in rows:
            try:
                feats = r["features"]
                if isinstance(feats, str):
                    feats = json.loads(feats)
                if isinstance(feats, dict) and feats.get("pnl_source") == "xm_deal":
                    if feats.get("xm_deal_pnl") is not None:
                        pnl += float(feats["xm_deal_pnl"])
                        continue
                pnl += float(r["pnl"] or 0)
            except Exception:
                continue
    except Exception as exc:
        logger.debug("rolling_7d_pnl %s: %s", symbol, exc)
        pnl = 0.0

    _pnl_cache[symbol] = (now_m, pnl)
    return pnl


def night_sitout(symbol: Optional[str], hour_utc: float) -> bool:
    sym = str(symbol or "").upper()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    if sym not in NIGHT_SITOUT:
        return False
    return hour_utc < 5 or hour_utc >= 22


def off_session_sitout(symbol: Optional[str], hour_utc: float) -> bool:
    """Sit out 21–24 UTC for pairs where Off-session XM spread kills E[R]."""
    sym = str(symbol or "").upper()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    if sym not in OFF_SESSION_SITOUT:
        return False
    return float(hour_utc) >= 21.0 or float(hour_utc) < 0.0


def trading_halted(symbol: Optional[str]) -> bool:
    """Hard halt — no XM-cost WF edge (BTC/AUD/CHF in fine-tune v2)."""
    try:
        from config import is_trading_halted

        return bool(is_trading_halted(symbol))
    except Exception:
        return False


def should_sitout(symbol: Optional[str], hour_utc: float) -> bool:
    if trading_halted(symbol):
        return True
    if night_sitout(symbol, hour_utc):
        return True
    if off_session_sitout(symbol, hour_utc):
        return True
    return False


def resolve_brain(symbol: Optional[str], primary: str) -> tuple[str, dict[str, Any]]:
    """Horizon router: use short alt when rolling 7d PnL < −2R."""
    maybe_monthly_rescore()
    sym = str(symbol or "").upper()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    primary = str(primary or "").strip().lower()
    st = load_state()
    pnl7 = _rolling_7d_pnl(sym)
    if pnl7 is None:
        pnl7 = 0.0
    thr = -2.0 * _RISK_USD
    alt = (st.get("short_alt") or DEFAULT_SHORT_ALT).get(sym) or primary
    alt = str(alt).strip().lower()
    use_alt = pnl7 < thr and alt in (
        "session_mix",
        "circadian_kalman_z",
        "entropy_tod",
    )
    active = alt if use_alt else primary
    meta = {
        "primary": primary,
        "active": active,
        "short_alt": alt,
        "horizon_switched": use_alt,
        "pnl_7d": round(pnl7, 2),
        "switch_thr": thr,
        "risk_scale": risk_scale_for_pnl(pnl7),
    }
    # Persist lightly (at most ~once/min per symbol via cache in risk path)
    per = st.setdefault("per_symbol", {})
    prev = per.get(sym) or {}
    if (
        prev.get("active") != active
        or prev.get("horizon_switched") != use_alt
        or abs(float(prev.get("pnl_7d") or 0) - pnl7) > 1.0
    ):
        per[sym] = {**meta, "at": _now().isoformat()}
        save_state(st)
    return active, meta


def risk_scale_for_pnl(pnl7: float) -> float:
    """Size throttle from rolling 7d PnL."""
    if pnl7 <= -2.0 * _RISK_USD:
        return 0.55
    if pnl7 < 0:
        return 0.75
    if pnl7 >= 2.0 * _RISK_USD:
        return 1.05
    return 1.0


def risk_scale_for(symbol: Optional[str]) -> float:
    sym = str(symbol or "").upper()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    st = load_state()
    row = (st.get("per_symbol") or {}).get(sym) or {}
    if row.get("risk_scale") is not None:
        return float(row["risk_scale"])
    pnl7 = _rolling_7d_pnl(sym) or 0.0
    return risk_scale_for_pnl(pnl7)
