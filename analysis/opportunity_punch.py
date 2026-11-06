"""Annotate opportunity-board rows with auto-punch readiness.

Board = brain scan hits (visible setups).
Auto = same hits after desk gates (open book, adaptive, MC floor, corr).

MC check must match analysis.trade_learning.evaluate_trade:
  markov_mc_prob → else path mc, with 3pp grace vs symbol floor.
"""
from __future__ import annotations

from typing import Any, Optional


def _open_desk_symbols() -> set[str]:
    out: set[str] = set()
    try:
        from data.broker import broker_client
        from strategy.instrument_meta import desk_symbol_from_broker, norm

        for p in broker_client.get_open_positions() or []:
            raw = p.get("symbol") or p.get("symbolName") or ""
            try:
                desk = desk_symbol_from_broker(raw) or norm(raw)
            except Exception:
                desk = norm(raw)
            d = str(desk).upper().replace("CASH", "")
            if "BRENT" in d or d in ("XBRUSD", "OIL"):
                d = "UKOIL"
            if d in ("US100", "USTEC") or "US100" in str(raw).upper():
                d = "NAS100"
            if d:
                out.add(d)
    except Exception:
        pass
    return out


def _gate_mc(row: dict) -> tuple[Optional[float], bool]:
    """Return (mc_for_gate, is_markov) — same preference as trade_learning."""
    for key in ("markov_mc", "markov_mc_prob", "mc"):
        raw = row.get(key)
        if raw is None:
            continue
        try:
            return float(raw), key.startswith("markov")
        except Exception:
            continue
    path = row.get("path_mc")
    if path is not None:
        try:
            return float(path), False
        except Exception:
            pass
    return None, False


def annotate_opportunities(rows: list[dict]) -> list[dict]:
    """Add punchable / block_reason using the same gates as signal_worker auto."""
    from analysis.adaptive_guard import check, get_state
    from analysis.instrument_rl import mc_floor_for
    from strategy.trade_executor import can_trade

    open_syms = _open_desk_symbols()
    desk_ok, desk_reason = can_trade()
    adaptive = get_state()

    annotated = []
    for r in rows:
        row = dict(r)
        sym = str(row.get("symbol") or "").upper()
        tf = str(row.get("exec_tf") or "")
        sig = str(row.get("signal") or "")
        mc_f, is_markov = _gate_mc(row)

        punchable = True
        reason = "Ready for auto-punch"

        if sig not in ("BUY", "SELL"):
            punchable, reason = False, "No trade signal"
        elif not desk_ok:
            punchable, reason = False, desk_reason or "Auto-trade off"
        elif adaptive.get("equity_allow") is False:
            punchable, reason = False, str(adaptive.get("equity_reason") or "Equity gate")
        elif sym in open_syms:
            punchable, reason = False, f"Already open on XM ({sym})"
        else:
            try:
                from config import MARKOV_MC_GATE_ENABLED
            except Exception:
                MARKOV_MC_GATE_ENABLED = False
            floor = float(mc_floor_for(sym))
            # Match trade_learning: Markov floor only when gate enabled (BT hunt did not)
            if (
                MARKOV_MC_GATE_ENABLED
                and mc_f is not None
                and float(mc_f) + 3.0 + 1e-6 < floor
            ):
                label = "Markov MC" if is_markov else "MC"
                punchable, reason = (
                    False,
                    f"{label} {mc_f:.0f}% below floor {floor:.0f}%",
                )
            else:
                ok, a_reason = check(
                    {
                        "mc_prob": mc_f,
                        "markov_mc_prob": mc_f if is_markov else None,
                        "exec_tf": tf,
                        "EXEC_TF": tf,
                        "STATE": row.get("state"),
                        "symbol": sym,
                        "mtf_independent": tf in ("15m", "1h", "4h"),
                    }
                )
                if not ok:
                    punchable, reason = False, a_reason

        row["punchable"] = punchable
        row["block_reason"] = None if punchable else reason
        row["status"] = "ready" if punchable else "blocked"
        row["mc_floor"] = round(float(mc_floor_for(sym)), 1)
        # Display the gate MC so the board number matches Auto
        if mc_f is not None:
            row["mc"] = mc_f
        annotated.append(row)
    return annotated
