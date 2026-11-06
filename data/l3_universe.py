"""Zenith universe → L3 / depth venue map.

True L3 (MBO with order_id) only exists on exchange matching engines.
XM CFD DOM is aggregated L2 at best. Futures/FX full history needs Databento.
"""
from __future__ import annotations

from typing import Any

ZENITH_UNIVERSE = (
    "GOLD",
    "USDCHF",
    "USDJPY",
    "NAS100",
    "UKOIL",
    "AUDUSD",
    "BTCUSD",
    "ETHUSD",
)

# Proxy venues for research (not the XM fill venue)
L3_MAP: dict[str, dict[str, Any]] = {
    "BTCUSD": {
        "true_l3": True,
        "primary": "coinbase",
        "product": "BTC-USD",
        "databento": None,
        "tardis": {"exchange": "coinbase", "symbol": "BTC-USD"},
        "note": "Coinbase Exchange L3 book (order_id per resting order)",
    },
    "ETHUSD": {
        "true_l3": True,
        "primary": "coinbase",
        "product": "ETH-USD",
        "databento": None,
        "tardis": {"exchange": "coinbase", "symbol": "ETH-USD"},
        "note": "Coinbase Exchange L3 book (order_id per resting order)",
    },
    "GOLD": {
        "true_l3": False,  # needs Databento CME GC MBO for true L3
        "primary": "xm_dom",
        "product": "GOLD",
        "databento": {"dataset": "GLBX.MDP3", "symbol": "GC.c.0", "schema": "mbo"},
        "tardis": None,
        "note": "XM DOM now; CME GC MBO via DATABENTO_API_KEY for true order chain",
    },
    "NAS100": {
        "true_l3": False,
        "primary": "xm_dom",
        "product": "NAS100",
        "databento": {"dataset": "GLBX.MDP3", "symbol": "NQ.c.0", "schema": "mbo"},
        "tardis": None,
        "note": "XM DOM now; CME NQ MBO via DATABENTO_API_KEY for true order chain",
    },
    "UKOIL": {
        "true_l3": False,
        "primary": "xm_dom",
        "product": "UKOIL",
        "databento": {"dataset": "GLBX.MDP3", "symbol": "CL.c.0", "schema": "mbo"},
        "tardis": None,
        "note": "XM DOM now; NYMEX CL (or ICE Brent) MBO via Databento for true L3",
    },
    "USDCHF": {
        "true_l3": False,
        "primary": "xm_dom",
        "product": "USDCHF",
        "databento": {"dataset": "DBFX.PLUS", "symbol": "USD/CHF", "schema": "mbp-10"},
        "tardis": None,
        "note": "Spot FX has no single exchange L3; XM DOM + Databento FX L2 when keyed",
    },
    "USDJPY": {
        "true_l3": False,
        "primary": "xm_dom",
        "product": "USDJPY",
        "databento": {"dataset": "DBFX.PLUS", "symbol": "USD/JPY", "schema": "mbp-10"},
        "tardis": None,
        "note": "Spot FX aggregated L2 at best; XM DOM primary without keys",
    },
    "AUDUSD": {
        "true_l3": False,
        "primary": "xm_dom",
        "product": "AUDUSD",
        "databento": {"dataset": "DBFX.PLUS", "symbol": "AUD/USD", "schema": "mbp-10"},
        "tardis": None,
        "note": "Spot FX aggregated L2 at best; XM DOM primary without keys",
    },
}
