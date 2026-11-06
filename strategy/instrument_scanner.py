"""Multi-instrument scanner — same core algo across XM universe.

Does not change strategy/signal.py logic. Fetches per-symbol frames,
runs independent MTF scan, tags SYMBOL for execution + correlation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config import (
    BARS_1H,
    BARS_1M,
    BARS_3M,
    BARS_4H,
    BARS_5M,
    BARS_15M,
    INSTRUMENT_UNIVERSE,
    MTF_SCAN_ENABLED,
    MTF_SCAN_TFS,
    MULTI_INSTRUMENT_ENABLED,
)
from indicators.prepare import prepare_df

logger = logging.getLogger(__name__)


def instrument_tradable(symbol: str) -> bool:
    """Metals / oil / US indices → XM gold hours; FX Mon–Fri; crypto 24/7."""
    from strategy.instrument_meta import instrument_tradable as _tradable

    return _tradable(symbol)

def fetch_frames_for_symbol(symbol: str) -> dict[str, Any]:
    """OHLCV frames for one XM symbol (MetaAPI in demo/live, TV in paper)."""
    from data.market_data import fetch_ohlcv
    from tvDatafeed import Interval

    sym = str(symbol).upper().strip()
    specs = [
        ("1m", Interval.in_1_minute, BARS_1M),
        ("3m", Interval.in_3_minute, BARS_3M),
        ("5m", Interval.in_5_minute, BARS_5M),
        ("15m", Interval.in_15_minute, BARS_15M),
        ("1h", Interval.in_1_hour, BARS_1H),
        ("4h", Interval.in_4_hour, BARS_4H),
    ]
    frames: dict[str, Any] = {}
    for key, interval, bars in specs:
        try:
            raw = fetch_ohlcv(interval, bars, symbol=sym)
            frames[key] = prepare_df(raw.copy()) if raw is not None and not raw.empty else None
        except Exception as exc:
            logger.debug("Frames %s %s failed: %s", sym, key, exc)
            frames[key] = None
    return frames


def resolve_entry_scan_tfs(symbol: str) -> tuple[str, ...]:
    """Per-symbol exec TFs from SYMBOL_SCAN_TFS / fib map (live + backtest parity)."""
    sym = str(symbol or "").upper().strip()
    if sym in ("XAUUSD", "XAU"):
        sym = "GOLD"
    try:
        from config import (
            fib_exec_tfs,
            is_fib_entry_brain,
            symbol_entry_brain,
            symbol_scan_tfs,
        )

        brain = symbol_entry_brain(sym)
        mapped = symbol_scan_tfs(sym)
        if is_fib_entry_brain(brain):
            return tuple(mapped or fib_exec_tfs() or MTF_SCAN_TFS)
        if mapped:
            return tuple(mapped)
    except Exception:
        pass
    if sym in ("GOLD", "XAUUSD", "XAU", "USDJPY"):
        return ("15m", "1h")
    if sym == "USDCHF":
        return ("5m",)
    if sym in ("NAS100", "UKOIL", "AUDUSD"):
        return ("15m", "1h", "4h")
    return tuple(MTF_SCAN_TFS)


def scan_symbol(symbol: str, *, only_tfs: Optional[tuple] = None) -> list[dict]:
    """Run MTF (or single 5m) scan for one symbol. Returns BUY/SELL payloads."""
    from strategy.mtf_scanner import scan_all_timeframes
    from strategy.signal import _compute_signal

    sym = str(symbol).upper().strip()
    if not instrument_tradable(sym):
        return []

    frames = fetch_frames_for_symbol(sym)
    hits: list[dict] = []

    if only_tfs is None:
        only_tfs = resolve_entry_scan_tfs(sym)

    if MTF_SCAN_ENABLED:
        tfs = only_tfs or MTF_SCAN_TFS
        hits = scan_all_timeframes(frames, only_tfs=tuple(tfs), symbol=sym)
        # Tag HTF-entry / 5m-manage research path
        try:
            from config import uses_htf_entry_5m_exit

            if uses_htf_entry_5m_exit(sym):
                for h in hits:
                    h["ENTRY_TF"] = h.get("EXEC_TF")
                    h["MANAGE_TF"] = "5m"
                    h["EXIT_PATH"] = "htf_entry_5m_exit"
                    zone = str(h.get("ZONE_TFS") or "")
                    if "5m manage" not in zone.lower():
                        h["ZONE_TFS"] = (zone + " · 5m manage").strip(" ·")
        except Exception:
            pass
    else:
        f5 = frames.get("5m")
        if f5 is not None and len(f5) >= 40:
            try:
                sig = _compute_signal(
                    df_1m=frames.get("1m") if frames.get("1m") is not None else f5,
                    df_3m=frames.get("3m") if frames.get("3m") is not None else f5,
                    df_5m=f5,
                    df_15m=frames.get("15m") if frames.get("15m") is not None else f5,
                    df_1h=frames.get("1h") if frames.get("1h") is not None else f5,
                    df_4h=frames.get("4h") if frames.get("4h") is not None else f5,
                    symbol=sym,
                )
                if sig.get("SIGNAL") in ("BUY", "SELL"):
                    sig["EXEC_TF"] = "5m"
                    hits = [sig]
            except Exception as exc:
                logger.debug("Signal %s failed: %s", sym, exc)

    out: list[dict] = []
    for h in hits:
        row = dict(h)
        row["SYMBOL"] = sym
        row["INSTRUMENT"] = sym
        row["MTF_INDEPENDENT"] = bool(row.get("MTF_INDEPENDENT") or MTF_SCAN_ENABLED)
        out.append(row)
    return out


def scan_universe(
    symbols: Optional[tuple[str, ...]] = None,
    *,
    only_tfs: Optional[tuple] = None,
) -> list[dict]:
    """
    Scan all configured instruments. Returns every valid setup
    (multiple markets + TFs) for concurrent execution.
    """
    if not MULTI_INSTRUMENT_ENABLED:
        # Fallback: primary MT5 symbol only
        from config import MT5_SYMBOL

        return scan_symbol(MT5_SYMBOL or "GOLD", only_tfs=only_tfs)

    universe = symbols or INSTRUMENT_UNIVERSE
    prefer = ("4h", "1h", "15m", "5m", "3m", "1m")
    rank = {tf: i for i, tf in enumerate(prefer)}
    all_hits: list[dict] = []

    for sym in universe:
        try:
            hits = scan_symbol(sym, only_tfs=only_tfs)
            all_hits.extend(hits)
        except Exception as exc:
            logger.warning("Universe scan %s failed: %s", sym, exc)

    all_hits.sort(
        key=lambda h: (
            rank.get(str(h.get("EXEC_TF") or ""), 99),
            -float(h.get("SUCCESS_PERCENT") or h.get("MARKOV_MC_PROB") or 0),
        )
    )
    return all_hits


def opportunity_summary(hits: list[dict]) -> list[dict]:
    """Compact rows for the dashboard opportunity board."""
    rows = []
    for h in hits:
        # Gate MC = Markov when present (same as trade_learning / auto punch).
        # Path MC (SUCCESS_PERCENT) can look "good" while Markov fails the floor.
        markov = h.get("MARKOV_MC_PROB")
        path_mc = h.get("SUCCESS_PERCENT")
        gate_mc = markov if markov is not None else path_mc
        rows.append(
            {
                "symbol": h.get("SYMBOL") or h.get("INSTRUMENT"),
                "exec_tf": h.get("EXEC_TF"),
                "signal": h.get("SIGNAL"),
                "entry": h.get("ENTRY"),
                "sl": h.get("SL"),
                "tp": h.get("TP"),
                "rr": h.get("RR"),
                "mc": gate_mc,
                "markov_mc": markov,
                "path_mc": path_mc,
                "state": h.get("STATE"),
                "buy_score": h.get("BUY_SCORE"),
                "sell_score": h.get("SELL_SCORE"),
                "reason": (h.get("REASON") or "")[:160],
            }
        )
    return rows
