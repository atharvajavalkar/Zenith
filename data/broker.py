"""
Broker selector — returns the active execution client based on BROKER config.

  BROKER=xm     → MetaAPI (XM MetaTrader 5)   [default]
  BROKER=oanda  → OANDA v20 REST

Both clients expose the same interface:
  .configured, .place_market_order(), .has_open_position(),
  .get_open_positions(), .close_all_positions()

so strategy/trade_executor and the trades API can stay broker-agnostic.
"""
from __future__ import annotations

import logging

from config import BROKER

logger = logging.getLogger(__name__)


def _select_broker():
    name = (BROKER or "xm").lower().strip()
    if name == "oanda":
        from data.oanda import oanda_client
        return oanda_client, "oanda"
    # default → XM via MetaAPI
    from data.mt5_client import mt5_client
    return mt5_client, "xm"


broker_client, broker_name = _select_broker()


def broker_account_summary(*, force: bool = False) -> dict | None:
    """Live broker account snapshot (balance, equity, P&L) when supported."""
    if not broker_client.configured:
        return None
    if force:
        clearer = getattr(broker_client, "clear_pnl_caches", None) or getattr(
            broker_client, "clear_caches", None
        )
        if clearer:
            try:
                clearer()
            except Exception:
                pass
    if hasattr(broker_client, "account_summary"):
        try:
            return broker_client.account_summary()
        except Exception as exc:
            logger.warning("broker_account_summary failed: %s", exc)
    return None


def order_id_from_response(response: dict) -> str | None:
    """Extract XM/OANDA position id — prefer positionId over order ticket."""
    if not isinstance(response, dict):
        return None
    return (
        response.get("positionId")      # MetaAPI position (required for manage/close)
        or response.get("id")           # MetaAPI sometimes
        or response.get("orderId")      # MetaAPI order ticket (fallback)
        or (response.get("orderFillTransaction") or {}).get("id")    # OANDA
        or (response.get("orderCreateTransaction") or {}).get("id")  # OANDA
    )
