"""OANDA v20 REST API client for trade execution."""

import logging
from typing import Any, Optional

import requests

from config import (
    OANDA_ACCOUNT_ID,
    OANDA_API_TOKEN,
    OANDA_ENVIRONMENT,
    OANDA_INSTRUMENT,
    TRADE_UNITS,
)

logger = logging.getLogger(__name__)

_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

_GRANULARITY = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "1h": "H1",
    "4h": "H4",
}


class OandaClient:
    def __init__(
        self,
        token: Optional[str] = None,
        account_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> None:
        self.token = token or OANDA_API_TOKEN
        self.account_id = account_id or OANDA_ACCOUNT_ID
        self.environment = (environment or OANDA_ENVIRONMENT).lower()
        self.base_url = _HOSTS.get(self.environment, _HOSTS["practice"])
        self.instrument = OANDA_INSTRUMENT

    @property
    def configured(self) -> bool:
        return bool(self.token and self.account_id)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _request(
        self, method: str, path: str, json_body: Optional[dict] = None
    ) -> dict:
        url = f"{self.base_url}{path}"
        resp = requests.request(
            method, url, headers=self._headers(), json=json_body, timeout=30
        )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json()
            except Exception:
                pass
            raise RuntimeError(f"OANDA {resp.status_code}: {detail}")
        if resp.text:
            return resp.json()
        return {}

    def get_open_positions(self) -> list:
        data = self._request("GET", f"/v3/accounts/{self.account_id}/openPositions")
        return data.get("positions", [])

    def fetch_candles(self, granularity: str, count: int = 300):
        """Fetch OHLCV from OANDA as fallback when TradingView fails."""
        import pandas as pd

        gran = _GRANULARITY.get(granularity, granularity)
        data = self._request(
            "GET",
            f"/v3/instruments/{self.instrument}/candles"
            f"?granularity={gran}&count={count}&price=M",
        )
        rows = []
        for c in data.get("candles", []):
            if not c.get("complete", True):
                continue
            mid = c["mid"]
            rows.append(
                {
                    "datetime": pd.Timestamp(c["time"]),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(c.get("volume", 0)),
                }
            )
        if not rows:
            raise ValueError("OANDA returned no candles")
        df = pd.DataFrame(rows).set_index("datetime")
        return df[["open", "high", "low", "close", "volume"]]

    def has_open_position(self, direction: str) -> bool:
        return self.count_positions(direction) > 0

    def count_positions(self, direction: Optional[str] = None) -> int:
        count = 0
        for pos in self.get_open_positions():
            if pos.get("instrument") != self.instrument:
                continue
            long_units = float(pos.get("long", {}).get("units", 0))
            short_units = float(pos.get("short", {}).get("units", 0))
            if direction == "BUY" and long_units > 0:
                count += 1
            elif direction == "SELL" and short_units < 0:
                count += 1
            elif direction is None and (long_units > 0 or short_units < 0):
                count += 1
        return count

    def place_market_order(
        self,
        direction: str,
        sl: float,
        tp: float,
        units: Optional[int] = None,
        client_tag: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self.configured:
            raise RuntimeError(
                "OANDA not configured. Set OANDA_API_TOKEN and OANDA_ACCOUNT_ID in .env"
            )

        qty = units if units is not None else TRADE_UNITS
        signed_units = qty if direction == "BUY" else -qty

        order_body = {
            "order": {
                "type": "MARKET",
                "instrument": self.instrument,
                "units": str(signed_units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"timeInForce": "GTC", "price": f"{sl:.2f}"},
                "takeProfitOnFill": {"timeInForce": "GTC", "price": f"{tp:.2f}"},
            }
        }
        if client_tag:
            order_body["order"]["clientExtensions"] = {
                "tag": client_tag[:20],
                "comment": "algo-signal",
            }

        logger.info(
            "Placing OANDA %s %s units=%s SL=%.2f TP=%.2f env=%s",
            direction,
            self.instrument,
            signed_units,
            sl,
            tp,
            self.environment,
        )
        return self._request(
            "POST",
            f"/v3/accounts/{self.account_id}/orders",
            json_body=order_body,
        )

    def close_all_positions(self) -> dict[str, Any]:
        if not self.configured:
            raise RuntimeError(
                "OANDA not configured. Set OANDA_API_TOKEN and OANDA_ACCOUNT_ID in .env"
            )
        logger.info("Closing all OANDA positions for %s", self.instrument)
        return self._request(
            "PUT",
            f"/v3/accounts/{self.account_id}/positions/{self.instrument}/close",
            json_body={"longUnits": "ALL", "shortUnits": "ALL"},
        )


oanda_client = OandaClient()
