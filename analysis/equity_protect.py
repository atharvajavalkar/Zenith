"""Equity / peak-drawdown / daily loss / profit-bank gates.

Shared by backtest and live so XM execution follows the same capital rules.
Designed to stop the classic pattern: early run-up → overtrade chop → giveback
(e.g. bleed after candle ~3401 once the first stretch from ~1667 has already paid).
"""

from __future__ import annotations

from typing import Any, Optional


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def peak_drawdown(peak_pnl: float, realized_pnl: float) -> float:
    """Absolute giveback from peak realized PnL (never negative)."""
    return max(0.0, _f(peak_pnl) - _f(realized_pnl))


def evaluate_equity_gates(
    *,
    start_capital: float,
    realized_pnl: float,
    peak_pnl: float,
    day_pnl: float = 0.0,
    trades_today: int = 0,
    peak_dd_pct: float = 0.15,
    daily_loss_pct: float = 0.04,
    max_trades_per_day: int = 6,
    harvest_pct: float = 0.50,
    bank_target_pct: float = 1.00,
) -> dict[str, Any]:
    """
    Return gate decision + MC bump for harvest / protect.

    Modes:
      bank   — hit monthly-style target; no new entries (lock the double)
      protect — peak drawdown too large; block new entries
      harvest — already +harvest_pct; allow only if caller raises MC
      daily   — daily loss or trade-count cap
      ok
    """
    cap = max(_f(start_capital, 1000.0), 1.0)
    pnl = _f(realized_pnl)
    peak = max(_f(peak_pnl), pnl)
    dd = peak_drawdown(peak, pnl)
    day = _f(day_pnl)

    bank_target = cap * bank_target_pct
    harvest_line = cap * harvest_pct
    peak_dd_limit = cap * peak_dd_pct
    daily_limit = -(cap * daily_loss_pct) if daily_loss_pct > 0 else None

    out: dict[str, Any] = {
        "allow": True,
        "mode": "ok",
        "reason": "Equity OK",
        "mc_bump": 0.0,
        "peak_pnl": peak,
        "drawdown": round(dd, 2),
        "realized_pnl": round(pnl, 2),
        "progress_pct": round(100.0 * pnl / cap, 1),
    }

    if pnl >= bank_target:
        out.update(
            allow=False,
            mode="bank",
            reason=f"Banked target — +{out['progress_pct']:.0f}% of capital (lock profits)",
            mc_bump=25.0,
        )
        return out

    # peak_dd_pct <= 0 disables this gate (algo testing / demo desks)
    if peak_dd_pct > 0 and dd >= peak_dd_limit and peak > 0:
        out.update(
            allow=False,
            mode="protect",
            reason=(
                f"Peak drawdown ${dd:.0f} ≥ {peak_dd_pct:.0%} of capital "
                f"— pause entries to stop giveback"
            ),
            mc_bump=15.0,
        )
        return out

    # daily_loss_pct <= 0 disables this gate (algo testing / paper desks)
    if daily_limit is not None and day <= daily_limit:
        out.update(
            allow=False,
            mode="daily",
            reason=f"Daily loss ${day:.0f} hit {daily_loss_pct:.0%} capital limit",
            mc_bump=10.0,
        )
        return out

    if max_trades_per_day > 0 and trades_today >= max_trades_per_day:
        out.update(
            allow=False,
            mode="daily",
            reason=f"Daily trade cap ({trades_today}/{max_trades_per_day})",
            mc_bump=5.0,
        )
        return out

    if pnl >= harvest_line:
        out.update(
            allow=True,
            mode="harvest",
            reason=f"Harvest mode — +{out['progress_pct']:.0f}% already; require stronger setups",
            mc_bump=8.0,
        )
        return out

    return out


def update_peak(peak_pnl: Optional[float], realized_pnl: float) -> float:
    return max(_f(peak_pnl), _f(realized_pnl))
