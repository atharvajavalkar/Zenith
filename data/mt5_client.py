"""
XM (MetaTrader 5) execution via MetaAPI cloud REST bridge.

Caches reads and backs off on HTTP 429 so the dashboard cannot burn the
free-tier quota (TooManyRequestsError).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

from config import (
    METAAPI_ACCOUNT_ID,
    METAAPI_ACCOUNT_ID_DEMO,
    METAAPI_ACCOUNT_ID_LIVE,
    METAAPI_BASE_URL,
    METAAPI_MARKET_DATA_URL,
    METAAPI_TOKEN,
    MT5_ORDER_COMMENT,
    MT5_SYMBOL,
    TRADE_LOT_SIZE,
    TRADE_MODE,
    metaapi_account_id_for_mode,
)

logger = logging.getLogger(__name__)

_DONE_CODES = {10008, 10009}  # placed / done

# Read-cache TTLs (seconds)
_TTL_PRICE = 2.5
_TTL_ACCOUNT = 10.0
_TTL_POSITIONS = 6.0
_TTL_REALIZED = 90.0
_TTL_SPEC = 300.0
_TTL_CANDLES = {
    "1m": 30.0,
    "3m": 50.0,
    "5m": 70.0,
    "15m": 150.0,
    "1h": 300.0,
    "4h": 600.0,
}


class MetaApiClient:
    """Thin REST wrapper around the MetaAPI MetaTrader Client API."""

    def __init__(
        self,
        token: Optional[str] = None,
        account_id: Optional[str] = None,
        base_url: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> None:
        self.token = token or METAAPI_TOKEN
        # Start on whatever TRADE_MODE points at (demo/live); paper keeps demo id for quick switch
        initial_mode = str(TRADE_MODE).lower().strip() if not account_id else ""
        if account_id:
            self.account_id = account_id
        else:
            self.account_id = metaapi_account_id_for_mode(
                initial_mode if initial_mode in ("demo", "live") else "demo"
            ) or METAAPI_ACCOUNT_ID_DEMO or METAAPI_ACCOUNT_ID
        self.base_url = (base_url or METAAPI_BASE_URL).rstrip("/")
        self.market_data_url = (METAAPI_MARKET_DATA_URL or self.base_url).rstrip("/")
        self.symbol = symbol or MT5_SYMBOL
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, Any]] = {}
        self._backoff_until = 0.0
        # True when the last _request/_market_data_request returned expired cache
        # (429 / backoff). Callers must not re-cache derived values as "fresh".
        self._last_served_stale = False
        self._active_mode = (
            initial_mode if initial_mode in ("demo", "live") else "demo"
        )

    @property
    def configured(self) -> bool:
        return bool(self.token and self.account_id and self.base_url)

    def auth_status(self) -> dict[str, Any]:
        """Local token check. Does not call MetaAPI."""
        token = (self.token or "").strip()
        if not token:
            return {
                "ok": False,
                "state": "missing",
                "message": "METAAPI_TOKEN is empty. Paste a token from app.metaapi.cloud.",
            }
        try:
            import base64
            import json as _json

            part = token.split(".")[1]
            part += "=" * (-len(part) % 4)
            claims = _json.loads(base64.urlsafe_b64decode(part))
            exp = int(claims.get("exp") or 0)
        except Exception:
            return {
                "ok": False,
                "state": "invalid",
                "message": "METAAPI_TOKEN is not a valid token.",
            }
        if exp and time.time() >= exp:
            when = datetime.fromtimestamp(exp, timezone.utc).strftime("%b %d, %Y")
            return {
                "ok": False,
                "state": "expired",
                "expired_at": datetime.fromtimestamp(exp, timezone.utc).isoformat(),
                "message": (
                    f"MetaAPI token expired {when}. "
                    "Generate a new token at app.metaapi.cloud and replace METAAPI_TOKEN."
                ),
            }
        return {"ok": True, "state": "ok", "message": "MetaAPI token is current"}

    def account_ids_available(self) -> dict[str, bool]:
        return {
            "demo": bool(METAAPI_ACCOUNT_ID_DEMO or METAAPI_ACCOUNT_ID),
            "live": bool(METAAPI_ACCOUNT_ID_LIVE),
        }

    def clear_caches(self) -> None:
        with self._lock:
            self._cache.clear()
            self._backoff_until = 0.0
            self._last_served_stale = False

    def clear_pnl_caches(self) -> None:
        """Drop account / positions / realised caches only (keep candle cache).

        Used by Sync XM and dashboard summary so rate-limited market-data
        caches are not wiped (which would amplify MetaAPI 429s).
        """
        with self._lock:
            for key in list(self._cache):
                if (
                    key == "account-information"
                    or key.startswith("positions")
                    or key.startswith("realized_pnl")
                ):
                    self._cache.pop(key, None)
            self._last_served_stale = False

    def apply_trade_mode(self, mode: str) -> str:
        """
        Point this client at the MetaAPI account for demo/live.
        Paper keeps the last broker account id (for reconnect) but clears caches.
        """
        m = str(mode or "").lower().strip()
        if m == "paper":
            self.clear_caches()
            return self.account_id or ""
        new_id = metaapi_account_id_for_mode(m)
        if m == "live" and not new_id:
            logger.warning(
                "TRADE_MODE=live but METAAPI_ACCOUNT_ID_LIVE is empty — broker not configured for live"
            )
        if new_id != self.account_id:
            logger.info(
                "MetaAPI account switch %s → %s (%s…)",
                self._active_mode,
                m,
                (new_id[:8] + "…") if new_id else "none",
            )
            self.account_id = new_id
            self.clear_caches()
        self._active_mode = m
        return self.account_id or ""

    def configured_for(self, mode: str) -> bool:
        if not self.token or not self.base_url:
            return False
        return bool(metaapi_account_id_for_mode(mode))

    def _headers(self) -> dict:
        return {"auth-token": self.token, "Content-Type": "application/json"}

    def _url(self, path: str) -> str:
        return f"{self.base_url}/users/current/accounts/{self.account_id}{path}"

    def _market_data_url_path(self, path: str) -> str:
        return f"{self.market_data_url}/users/current/accounts/{self.account_id}{path}"

    def _cache_get(self, key: str) -> Any:
        with self._lock:
            hit = self._cache.get(key)
            if not hit:
                return None
            expires, value = hit
            if time.time() >= expires:
                return None
            return value

    def _cache_set(self, key: str, value: Any, ttl: float) -> None:
        with self._lock:
            self._cache[key] = (time.time() + max(ttl, 0.5), value)

    def _cache_get_stale(self, key: str) -> Any:
        with self._lock:
            hit = self._cache.get(key)
            return hit[1] if hit else None

    def _in_backoff(self) -> bool:
        return time.time() < self._backoff_until

    def _trip_backoff(self, detail: Any) -> None:
        wait = 65.0
        if isinstance(detail, dict):
            retry_at = (detail.get("metadata") or {}).get("recommendedRetryTime")
            if retry_at:
                try:
                    dt = datetime.fromisoformat(str(retry_at).replace("Z", "+00:00"))
                    wait = max(15.0, (dt - datetime.now(timezone.utc)).total_seconds() + 3.0)
                except Exception:
                    pass
        with self._lock:
            self._backoff_until = max(self._backoff_until, time.time() + wait)
        logger.warning(
            "MetaAPI 429 rate limit — pausing requests for %.0fs (serving cache when possible)",
            wait,
        )

    def _handle_error_response(
        self,
        resp: requests.Response,
        *,
        cache_key: Optional[str],
        label: str,
    ) -> Any:
        detail: Any = resp.text
        try:
            detail = resp.json()
        except Exception:
            pass
        if resp.status_code == 429:
            self._trip_backoff(detail)
            if cache_key:
                stale = self._cache_get_stale(cache_key)
                if stale is not None:
                    logger.warning("Serving stale %s cache after MetaAPI 429", cache_key)
                    self._last_served_stale = True
                    return stale
            raise RuntimeError(f"MetaAPI 429: {detail}")
        raise RuntimeError(f"MetaAPI {label}{resp.status_code}: {detail}")

    def _request(
        self,
        method: str,
        path: str,
        json_body: Optional[dict] = None,
        *,
        cache_key: Optional[str] = None,
        cache_ttl: float = 0,
    ) -> Any:
        if not self.configured:
            raise RuntimeError(
                "MetaAPI not configured — set METAAPI_TOKEN and METAAPI_ACCOUNT_ID in .env"
            )
        auth = self.auth_status()
        if not auth.get("ok"):
            raise RuntimeError(auth.get("message") or "MetaAPI token rejected")

        self._last_served_stale = False
        if cache_key and method.upper() == "GET":
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        if self._in_backoff() and method.upper() == "GET":
            if cache_key:
                stale = self._cache_get_stale(cache_key)
                if stale is not None:
                    self._last_served_stale = True
                    return stale
            raise RuntimeError("MetaAPI in rate-limit backoff — retry shortly")

        # Writes during backoff: still try (orders matter) but log
        if self._in_backoff() and method.upper() != "GET":
            logger.warning("MetaAPI write during rate-limit backoff — attempting anyway")

        resp = requests.request(
            method, self._url(path), headers=self._headers(), json=json_body, timeout=30
        )
        if resp.status_code >= 400:
            return self._handle_error_response(resp, cache_key=cache_key, label="")

        data: Any = resp.json() if resp.text else {}
        if cache_key and method.upper() == "GET" and cache_ttl > 0:
            self._cache_set(cache_key, data, cache_ttl)
        self._last_served_stale = False
        return data

    def _market_data_request(self, path: str, *, cache_key: str, cache_ttl: float) -> Any:
        if not self.configured:
            raise RuntimeError(
                "MetaAPI not configured — set METAAPI_TOKEN and METAAPI_ACCOUNT_ID in .env"
            )
        auth = self.auth_status()
        if not auth.get("ok"):
            raise RuntimeError(auth.get("message") or "MetaAPI token rejected")
        self._last_served_stale = False
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        if self._in_backoff():
            stale = self._cache_get_stale(cache_key)
            if stale is not None:
                self._last_served_stale = True
                return stale
            raise RuntimeError("MetaAPI in rate-limit backoff — retry shortly")

        resp = requests.get(
            self._market_data_url_path(path),
            headers=self._headers(),
            timeout=45,
        )
        if resp.status_code >= 400:
            return self._handle_error_response(
                resp, cache_key=cache_key, label="market-data "
            )
        data = resp.json() if resp.text else {}
        self._cache_set(cache_key, data, cache_ttl)
        self._last_served_stale = False
        return data

    def _live_lot(self, lot_size: Optional[float]) -> float:
        if lot_size is not None:
            raw = float(lot_size)
        else:
            try:
                import runtime_config

                raw = float(runtime_config.get("trade_lot_size", TRADE_LOT_SIZE))
            except Exception:
                raw = float(TRADE_LOT_SIZE)
        try:
            return self.round_volume(raw)
        except Exception:
            return max(0.01, round(raw, 2))

    # ── account / positions ─────────────────────────────────────────────────────
    def account_information(self) -> dict:
        return (
            self._request(
                "GET",
                "/account-information",
                cache_key="account-information",
                cache_ttl=_TTL_ACCOUNT,
            )
            or {}
        )

    def get_open_positions(self, *, force: bool = False) -> list:
        """Live XM open positions. force=True bypasses cache (sync / after fill)."""
        if force:
            with self._lock:
                for key in list(self._cache):
                    if key.startswith("positions"):
                        self._cache.pop(key, None)
        data = self._request(
            "GET",
            "/positions",
            cache_key="positions",
            cache_ttl=0.5 if force else _TTL_POSITIONS,
        )
        return data if isinstance(data, list) else []

    def _sym(self, symbol: Optional[str] = None) -> str:
        raw = (symbol or self.symbol or "").strip()
        if not raw:
            return ""
        try:
            from strategy.instrument_meta import resolve_broker_symbol

            def _probe(cand: str) -> bool:
                try:
                    self._request(
                        "GET",
                        f"/symbols/{cand}/specification",
                        cache_key=f"spec:{cand}",
                        cache_ttl=_TTL_SPEC,
                    )
                    return True
                except Exception:
                    return False

            return resolve_broker_symbol(raw, probe=_probe)
        except Exception:
            # Preserve original casing — XM rejects OILCASH / US30CASH
            return raw.strip() or raw.upper()

    def _price_digits(self, symbol: Optional[str] = None) -> int:
        """XM price decimals — prefer specification, else instrument_meta."""
        sym = self._sym(symbol).upper()
        try:
            from strategy.instrument_meta import price_digits

            return price_digits(sym)
        except Exception:
            pass
        try:
            spec = self._request(
                "GET",
                f"/symbols/{sym}/specification",
                cache_key=f"spec:{sym}",
                cache_ttl=_TTL_SPEC,
            )
            digits = spec.get("digits")
            if digits is not None:
                return int(digits)
        except Exception:
            pass
        return 2

    def _round_price(self, price: float, symbol: Optional[str] = None) -> float:
        return round(float(price), self._price_digits(symbol))

    def _symbol_positions(self, symbol: Optional[str] = None) -> list:
        wire = self._sym(symbol)
        desk = str(symbol or self.symbol or "").upper().strip()
        aliases = {wire, desk}
        try:
            from strategy.instrument_meta import broker_candidates, desk_symbol_from_broker

            aliases.update(broker_candidates(desk))
            aliases.add(desk_symbol_from_broker(wire))
        except Exception:
            pass
        aliases = {a.upper() for a in aliases if a}
        return [
            p
            for p in self.get_open_positions()
            if str(p.get("symbol") or "").upper().strip() in aliases
        ]

    def count_positions(
        self, direction: Optional[str] = None, symbol: Optional[str] = None
    ) -> int:
        count = 0
        for pos in self._symbol_positions(symbol):
            if direction is None:
                count += 1
                continue
            want = "POSITION_TYPE_BUY" if direction == "BUY" else "POSITION_TYPE_SELL"
            if pos.get("type") == want:
                count += 1
        return count

    def has_open_position(self, direction: str, symbol: Optional[str] = None) -> bool:
        return self.count_positions(direction, symbol=symbol) > 0

    def _load_desk_epoch(self) -> tuple[Optional[datetime], Optional[float]]:
        """Return (reset_at, balance_at_reset) from desk_stats_epoch.json."""
        try:
            import json
            from pathlib import Path

            for cand in (
                Path("/app/data/desk_stats_epoch.json"),
                Path(__file__).resolve().parents[1] / "data" / "desk_stats_epoch.json",
            ):
                if not cand.exists():
                    continue
                raw = json.loads(cand.read_text(encoding="utf-8"))
                ts = raw.get("reset_at")
                epoch_since = None
                if ts:
                    epoch_since = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                bal = raw.get("balance_at_reset")
                bal_f = float(bal) if bal is not None else None
                return epoch_since, bal_f
        except Exception:
            pass
        return None, None

    def account_summary(self) -> dict[str, Any]:
        info = self.account_information()
        # Multi-instrument: float across every open XM symbol
        try:
            from config import MULTI_INSTRUMENT_ENABLED

            multi = MULTI_INSTRUMENT_ENABLED
        except Exception:
            multi = False
        positions = self.get_open_positions() if multi else self._symbol_positions()
        floating = round(sum(float(p.get("profit") or 0) for p in positions), 2)
        balance = info.get("balance")
        try:
            balance_f = float(balance) if balance is not None else None
        except Exception:
            balance_f = None

        # Desk session Realised must match Capital: balance − balance_at_reset
        # (deal sums filtered by universe previously drifted from XM balance).
        epoch_since, balance_at_reset = self._load_desk_epoch()
        deals_pnl = None
        deposits = 0.0
        if epoch_since is not None:
            deals_pnl, deposits = self.realized_pnl_breakdown(
                since=epoch_since, all_symbols=True
            )
        else:
            deals_pnl, deposits = self.realized_pnl_breakdown(days=1, all_symbols=True)

        history_pnl: Optional[float] = None
        realized_source = "xm_deals"
        if (
            balance_f is not None
            and balance_at_reset is not None
            and abs(float(deposits or 0)) < 0.01
        ):
            history_pnl = round(balance_f - float(balance_at_reset), 2)
            realized_source = "xm_balance_delta"
        elif deals_pnl is not None:
            history_pnl = deals_pnl
            realized_source = "xm_deals"

        lifetime_pnl = self.realized_pnl(days=30, all_symbols=True)
        return {
            "symbol": self.symbol,
            "balance": balance,
            "equity": info.get("equity"),
            "profit": info.get("profit"),
            "margin": info.get("margin"),
            "free_margin": info.get("freeMargin"),
            "currency": info.get("currency", "USD"),
            "server": info.get("server"),
            "open_positions": len(positions),
            "floating_pnl": floating,
            "realized_pnl": history_pnl,
            "history_pnl": history_pnl,
            "deals_pnl": deals_pnl,
            "balance_at_reset": balance_at_reset,
            "realized_source": realized_source,
            "lifetime_pnl_30d": lifetime_pnl,
            "realized_since": epoch_since.isoformat() if epoch_since else None,
        }

    def realized_pnl(
        self,
        days: int = 1,
        *,
        since: Optional[datetime] = None,
        all_symbols: bool = False,
    ) -> Optional[float]:
        """Sum closed-trade P&L from MetaAPI history deals."""
        total, _deposits = self.realized_pnl_breakdown(
            days=days, since=since, all_symbols=all_symbols
        )
        return total

    def realized_pnl_breakdown(
        self,
        days: int = 1,
        *,
        since: Optional[datetime] = None,
        all_symbols: bool = False,
    ) -> tuple[Optional[float], float]:
        """
        Sum closed-trade P&L from MetaAPI history deals.

        Returns (trade_pnl, deposit_credit_sum). Deposit/credit rows are excluded
        from trade_pnl but returned separately so callers can decide whether
        balance−start is safe to use as Realised.

        all_symbols=True → full account (matches Capital). False → desk universe.
        """
        if since is not None:
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
            scope = "all" if all_symbols else "uni"
            cache_key = f"realized_pnl:v3:{scope}:since:{since.strftime('%Y%m%d%H%M')}"
        else:
            scope = "all" if all_symbols else "uni"
            cache_key = f"realized_pnl:v3:{scope}:{int(days)}"
        cached = self._cache_get(cache_key)
        if isinstance(cached, (list, tuple)) and len(cached) == 2:
            return cached[0], float(cached[1] or 0)
        if isinstance(cached, (int, float)):
            return float(cached), 0.0
        # During rate-limit backoff, never re-derive from stale deal history and
        # re-cache as fresh — that froze Realised at $0 after closed trades.
        if self._in_backoff():
            stale = self._cache_get_stale(cache_key)
            if isinstance(stale, (list, tuple)) and len(stale) == 2:
                return stale[0], float(stale[1] or 0)
            if isinstance(stale, (int, float)):
                return float(stale), 0.0
            return None, 0.0
        try:
            end = datetime.now(timezone.utc)
            start = since if since is not None else (end - timedelta(days=max(1, int(days))))
            start_s = start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            end_s = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            path = f"/history-deals/time/{start_s}/{end_s}?limit=1000"
            deals = self._request(
                "GET",
                path,
                cache_key=f"{cache_key}:raw",
                cache_ttl=min(_TTL_REALIZED, 20.0),
            )
            if self._last_served_stale:
                stale = self._cache_get_stale(cache_key)
                if isinstance(stale, (list, tuple)) and len(stale) == 2:
                    return stale[0], float(stale[1] or 0)
                if isinstance(stale, (int, float)):
                    return float(stale), 0.0
                if not isinstance(deals, list):
                    return None, 0.0
            elif not isinstance(deals, list):
                return None, 0.0
            total = 0.0
            deposits = 0.0
            try:
                from config import MULTI_INSTRUMENT_ENABLED, INSTRUMENT_UNIVERSE

                multi = MULTI_INSTRUMENT_ENABLED
                universe = {str(s).upper() for s in INSTRUMENT_UNIVERSE}
            except Exception:
                multi = False
                universe = {self.symbol}

            _NON_TRADE = (
                "DEAL_TYPE_BALANCE",
                "DEAL_TYPE_CREDIT",
                "DEAL_TYPE_CHARGE",
                "DEAL_TYPE_CORRECTION",
                "DEAL_TYPE_BONUS",
                "DEAL_TYPE_COMMISSION",
            )
            for deal in deals:
                dtype = str(deal.get("type") or "").upper()
                p = float(deal.get("profit") or 0)
                c = float(deal.get("commission") or 0)
                s = float(deal.get("swap") or 0)
                if dtype in _NON_TRADE or "BALANCE" in dtype or "CREDIT" in dtype:
                    deposits += p
                    continue
                dsym = str(deal.get("symbol") or "").upper().strip()
                if not dsym:
                    deposits += p  # non-trading cash rows
                    continue
                # Map XM wire names (US100Cash, BRENT, …) → desk (NAS100, UKOIL)
                try:
                    from strategy.instrument_meta import desk_symbol_from_broker

                    desk = desk_symbol_from_broker(dsym)
                except Exception:
                    desk = dsym
                if not all_symbols:
                    if multi:
                        if universe and desk not in universe and dsym not in universe:
                            continue
                    elif desk != str(self.symbol).upper() and dsym != str(self.symbol).upper():
                        continue
                total += p + c + s
            result = round(total, 2)
            deposits = round(deposits, 2)
            if not self._last_served_stale:
                self._cache_set(cache_key, (result, deposits), min(_TTL_REALIZED, 20.0))
            return result, deposits
        except Exception as exc:
            logger.warning("MetaAPI realized_pnl failed: %s", exc)
            stale = self._cache_get_stale(cache_key)
            if isinstance(stale, (list, tuple)) and len(stale) == 2:
                return stale[0], float(stale[1] or 0)
            if isinstance(stale, (int, float)):
                return float(stale), 0.0
            return None, 0.0

    # ── order placement ─────────────────────────────────────────────────────────
    @contextmanager
    def using_symbol(self, symbol: Optional[str]):
        """Temporarily route candles/orders to another XM symbol (broker wire name).

        Holds the client RLock for the whole block so parallel Hyper punches cannot
        stomp self.symbol (which caused OILCash/EURUSD mixups).
        """
        raw = (symbol or "").strip()
        if not raw:
            yield self
            return
        wire = self._sym(raw) or raw
        with self._lock:
            if wire == self.symbol:
                yield self
                return
            prev = self.symbol
            self.symbol = wire
            try:
                yield self
            finally:
                self.symbol = prev

    def _stops_min_distance(self) -> float:
        try:
            spec = self._request(
                "GET",
                f"/symbols/{self.symbol}/specification",
                cache_key=f"spec:{self.symbol}",
                cache_ttl=_TTL_SPEC,
            )
            points = float(spec.get("points") or spec.get("point") or 0.01)
            stops_level = float(spec.get("stopsLevel") or 0)
            return max(stops_level * points, points)
        except Exception:
            return 0.01

    def rebase_stops_to_live(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        tp1: Optional[float] = None,
        tp2: Optional[float] = None,
        tp3: Optional[float] = None,
    ) -> dict[str, Any]:
        quote = self.current_price()
        # OILCash / US30Cash often return empty current-price when asked by wire
        # name alone — retry desk aliases, then synthetic candle mark.
        if not quote or quote.get("bid") is None or quote.get("ask") is None:
            for alias in (self.symbol, "USOIL", "OIL", "US30", "US30Cash", "OILCash"):
                if not alias:
                    continue
                try:
                    quote = self.current_price(symbol=alias)
                except Exception:
                    quote = None
                if quote and quote.get("bid") is not None and quote.get("ask") is not None:
                    break
        if not quote or quote.get("bid") is None or quote.get("ask") is None:
            quote = self._synthetic_quote(self._sym(self.symbol) or self.symbol)
        if not quote or quote.get("bid") is None or quote.get("ask") is None:
            raise RuntimeError(
                f"Cannot place {self.symbol} order — MetaAPI live price unavailable"
            )

        bid = float(quote["bid"])
        ask = float(quote["ask"])
        # 1.4× buffer over the raw stops level — very tight (hyper-scalp) stops
        # otherwise get rejected as TRADE_RETCODE_INVALID_STOPS at the moment of fill.
        min_dist = self._stops_min_distance() * 1.4
        entry_f = float(entry)
        offset = (bid if direction == "SELL" else ask) - entry_f
        digits = self._price_digits()
        rnd = lambda x: round(float(x), digits)  # noqa: E731

        def shift(level: Optional[float]) -> Optional[float]:
            if level is None:
                return None
            return rnd(float(level) + offset)

        fill = rnd(bid if direction == "SELL" else ask)
        new_sl = shift(sl)
        new_tp = shift(tp)
        new_tp1 = shift(tp1)
        new_tp2 = shift(tp2)
        new_tp3 = shift(tp3)

        if direction == "BUY":
            ceiling_sl = rnd(bid - min_dist)
            floor_tp = rnd(ask + min_dist)
            if new_sl is None or new_sl > ceiling_sl:
                new_sl = ceiling_sl
            if new_tp is None or new_tp < floor_tp:
                new_tp = floor_tp
        else:
            floor_sl = rnd(ask + min_dist)
            ceiling_tp = rnd(bid - min_dist)
            if new_sl is None or new_sl < floor_sl:
                new_sl = floor_sl
            if new_tp is None or new_tp > ceiling_tp:
                new_tp = ceiling_tp

        warn_thresh = 0.05 if digits <= 2 else (0.05 if digits <= 3 else 0.0005)
        if abs(offset) >= warn_thresh:
            logger.warning(
                "Rebased %s stops to live %s: signal entry %.*f → fill≈%.*f "
                "(bid=%.*f ask=%.*f) SL %.*f→%.*f TP %.*f→%.*f",
                direction,
                self.symbol,
                digits,
                entry_f,
                digits,
                fill,
                digits,
                bid,
                digits,
                ask,
                digits,
                float(sl),
                digits,
                new_sl,
                digits,
                float(tp),
                digits,
                new_tp,
            )

        return {
            "entry": fill,
            "sl": new_sl,
            "tp": new_tp,
            "tp1": new_tp1,
            "tp2": new_tp2,
            "tp3": new_tp3,
            "bid": bid,
            "ask": ask,
            "offset": rnd(offset),
        }

    def place_market_order(
        self,
        direction: str,
        sl: float,
        tp: float,
        units: Optional[int] = None,
        lot_size: Optional[float] = None,
        client_tag: Optional[str] = None,
        entry: Optional[float] = None,
        tp1: Optional[float] = None,
        tp2: Optional[float] = None,
        tp3: Optional[float] = None,
    ) -> dict[str, Any]:
        volume = self._live_lot(lot_size)
        action = "ORDER_TYPE_BUY" if direction == "BUY" else "ORDER_TYPE_SELL"

        rebased = None
        if entry is not None:
            rebased = self.rebase_stops_to_live(
                direction,
                float(entry),
                float(sl),
                float(tp),
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
            )
            sl = rebased["sl"]
            tp = rebased["tp"]
        else:
            quote = self.current_price()
            if quote and quote.get("bid") is not None and quote.get("ask") is not None:
                bid, ask = float(quote["bid"]), float(quote["ask"])
                min_dist = self._stops_min_distance()
                if direction == "BUY" and (
                    float(sl) >= bid - min_dist or float(tp) <= ask + min_dist
                ):
                    raise RuntimeError(
                        f"Invalid BUY stops vs live {self.symbol} bid={bid} ask={ask} "
                        f"SL={sl} TP={tp} — pass entry to rebase"
                    )
                if direction == "SELL" and (
                    float(sl) <= ask + min_dist or float(tp) >= bid - min_dist
                ):
                    raise RuntimeError(
                        f"Invalid SELL stops vs live {self.symbol} bid={bid} ask={ask} "
                        f"SL={sl} TP={tp} — pass entry to rebase"
                    )

        comment = MT5_ORDER_COMMENT
        if client_tag:
            safe = re.sub(r"[^A-Za-z0-9_\-]", "", str(client_tag))[:16]
            if safe:
                comment = f"{MT5_ORDER_COMMENT[:14]}-{safe}"[:31]

        digits = self._price_digits()
        body = {
            "actionType": action,
            "symbol": self.symbol,
            "volume": volume,
            "stopLoss": self._round_price(sl),
            "takeProfit": self._round_price(tp),
            "fillingModes": ["ORDER_FILLING_IOC"],
            "comment": comment[:31],
        }

        logger.info(
            "Placing MetaAPI %s %s vol=%s SL=%.*f TP=%.*f",
            direction,
            self.symbol,
            volume,
            digits,
            float(sl),
            digits,
            float(tp),
        )
        resp = self._request("POST", "/trade", json_body=body)
        # Invalidate position caches after a trade
        with self._lock:
            for key in list(self._cache):
                if key.startswith("positions") or key.startswith("account"):
                    self._cache.pop(key, None)

        code = resp.get("numericCode")
        string_code = resp.get("stringCode", "")
        if code is not None and code not in _DONE_CODES:
            raise RuntimeError(
                f"MetaAPI order rejected: {string_code or code} — {resp.get('message', '')}"
            )

        resp = dict(resp) if isinstance(resp, dict) else {"raw": resp}
        # Prefer the live position id (Modify/Close need this — not the order ticket)
        pos_id = self._resolve_position_id_after_order(
            symbol=self.symbol,
            volume=volume,
            comment=comment[:31],
            direction=direction,
            resp=resp,
        )
        if pos_id:
            resp["positionId"] = pos_id
            resp["orderId"] = resp.get("orderId") or pos_id
            # Guarantee stops landed with correct digits (FX used to truncate to 2dp)
            try:
                self.modify_position(pos_id, stop_loss=float(sl), take_profit=float(tp))
            except Exception as exc:
                logger.warning(
                    "Post-fill stop attach failed pos=%s SL=%s TP=%s: %s",
                    pos_id,
                    sl,
                    tp,
                    exc,
                )
        if rebased:
            resp["_rebased"] = rebased
        return resp

    def _resolve_position_id_after_order(
        self,
        *,
        symbol: str,
        volume: float,
        comment: str,
        direction: str,
        resp: dict,
    ) -> Optional[str]:
        """Map a just-placed market order to its XM position id."""
        # Immediate hints from MetaAPI payload
        for key in ("positionId", "positionid", "id"):
            raw = resp.get(key)
            if raw is not None and str(raw).strip():
                return str(raw).strip()

        want_type = "POSITION_TYPE_BUY" if direction == "BUY" else "POSITION_TYPE_SELL"
        comment_key = (comment or "")[:20]

        for _attempt in range(4):
            time.sleep(0.35)
            try:
                with self._lock:
                    for key in list(self._cache):
                        if key.startswith("positions"):
                            self._cache.pop(key, None)
                positions = self.get_open_positions()
            except Exception:
                continue

            # Exact comment match first
            for p in positions:
                if str(p.get("symbol") or "") != symbol:
                    continue
                if p.get("type") != want_type:
                    continue
                pc = str(p.get("comment") or p.get("brokerComment") or "")
                if comment_key and comment_key in pc:
                    return str(p.get("id") or p.get("positionId") or "")

            # Fallback: newest matching symbol/side/volume
            candidates = [
                p
                for p in positions
                if str(p.get("symbol") or "") == symbol
                and p.get("type") == want_type
                and abs(float(p.get("volume") or 0) - float(volume)) < 0.001
            ]
            if candidates:
                candidates.sort(
                    key=lambda p: str(p.get("time") or p.get("brokerTime") or ""),
                    reverse=True,
                )
                pid = candidates[0].get("id") or candidates[0].get("positionId")
                if pid:
                    return str(pid)
        return None

    def close_all_positions(self) -> dict[str, Any]:
        """Close every open symbol on the account (multi-instrument safe)."""
        try:
            positions = self.get_open_positions()
            symbols = sorted(
                {str(p.get("symbol")).strip() for p in positions if p.get("symbol")}
            )
        except Exception:
            symbols = []
        if not symbols:
            symbols = [self.symbol]
        results = []
        for sym in symbols:
            logger.info("Closing all MetaAPI positions for %s", sym)
            results.append(
                self._request(
                    "POST",
                    "/trade",
                    json_body={"actionType": "POSITIONS_CLOSE_SYMBOL", "symbol": sym},
                )
            )
        return {"closed_symbols": symbols, "results": results}

    def _check_trade_response(self, resp: dict, action: str) -> dict[str, Any]:
        code = resp.get("numericCode")
        string_code = resp.get("stringCode", "")
        ok_codes = _DONE_CODES | {10025}
        if code is not None and code not in ok_codes:
            raise RuntimeError(
                f"MetaAPI {action} rejected: {string_code or code} — {resp.get('message', '')}"
            )
        return resp

    def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "actionType": "POSITION_MODIFY",
            "positionId": str(position_id),
        }
        # Digits from open position symbol when possible
        pos_sym = None
        try:
            for p in self.get_open_positions():
                if str(p.get("id") or p.get("positionId") or "") == str(position_id):
                    pos_sym = p.get("symbol")
                    break
        except Exception:
            pass
        if stop_loss is not None:
            body["stopLoss"] = self._round_price(stop_loss, pos_sym)
        if take_profit is not None:
            body["takeProfit"] = self._round_price(take_profit, pos_sym)
        logger.info(
            "MetaAPI modify pos=%s SL=%s TP=%s",
            position_id,
            body.get("stopLoss"),
            body.get("takeProfit"),
        )
        return self._check_trade_response(
            self._request("POST", "/trade", json_body=body),
            "POSITION_MODIFY",
        )

    def partial_close(self, position_id: str, volume: float) -> dict[str, Any]:
        vol = round(float(volume), 2)
        if vol <= 0:
            raise ValueError(f"Partial close volume must be > 0, got {vol}")
        body = {
            "actionType": "POSITION_PARTIAL",
            "positionId": str(position_id),
            "volume": vol,
        }
        logger.info("MetaAPI partial close pos=%s vol=%s", position_id, vol)
        return self._check_trade_response(
            self._request("POST", "/trade", json_body=body),
            "POSITION_PARTIAL",
        )

    def close_position(self, position_id: str) -> dict[str, Any]:
        body = {
            "actionType": "POSITION_CLOSE_ID",
            "positionId": str(position_id),
        }
        logger.info("MetaAPI close position %s", position_id)
        return self._check_trade_response(
            self._request("POST", "/trade", json_body=body),
            "POSITION_CLOSE_ID",
        )

    def history_deals_since(
        self, since: Optional[datetime] = None, *, days: int = 90
    ) -> list[dict[str, Any]]:
        """Raw MetaAPI deals from `since` (or last `days`) — profit/commission/swap."""
        try:
            end = datetime.now(timezone.utc)
            if since is not None:
                start = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
            else:
                start = end - timedelta(days=max(1, int(days)))
            start_s = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            end_s = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            path = f"/history-deals/time/{start_s}/{end_s}?limit=1000"
            deals = self._request(
                "GET",
                path,
                cache_key=f"deals:bulk:{start_s[:13]}",
                cache_ttl=20.0,
            )
            return deals if isinstance(deals, list) else []
        except Exception as exc:
            logger.debug("history_deals_since failed: %s", exc)
            return []

    def closed_pnl_map_since(
        self, since: Optional[datetime] = None, *, days: int = 90
    ) -> dict[str, dict[str, Any]]:
        """positionId → deal cash + entry/exit for reconcile / import."""
        deals = self.history_deals_since(since, days=days)
        out: dict[str, dict[str, Any]] = {}
        for deal in deals:
            if str(deal.get("type") or "") == "DEAL_TYPE_BALANCE":
                continue
            pid = str(deal.get("positionId") or "").strip()
            if not pid:
                continue
            bucket = out.setdefault(
                pid,
                {
                    "pnl": 0.0,
                    "exit_price": None,
                    "entry_price": None,
                    "volume": None,
                    "side": None,
                    "symbol": None,
                    "time": None,
                    "outs": 0,
                },
            )
            bucket["pnl"] += float(deal.get("profit") or 0)
            bucket["pnl"] += float(deal.get("commission") or 0)
            bucket["pnl"] += float(deal.get("swap") or 0)
            entry = str(deal.get("entryType") or "").upper()
            dtype = str(deal.get("type") or "").upper()
            if "IN" in entry and deal.get("price") is not None:
                bucket["entry_price"] = float(deal["price"])
                if deal.get("volume") is not None:
                    bucket["volume"] = float(deal["volume"])
                # Opening deal type is the position side
                if "BUY" in dtype:
                    bucket["side"] = "BUY"
                elif "SELL" in dtype:
                    bucket["side"] = "SELL"
            if deal.get("price") is not None and "OUT" in entry:
                bucket["exit_price"] = float(deal["price"])
                bucket["outs"] += 1
                if deal.get("time"):
                    bucket["time"] = deal.get("time")
            if deal.get("symbol"):
                bucket["symbol"] = deal.get("symbol")
        cleaned: dict[str, dict[str, Any]] = {}
        for pid, b in out.items():
            if b["outs"] == 0 and abs(float(b["pnl"])) < 1e-12:
                continue
            cleaned[pid] = {
                "pnl": round(float(b["pnl"]), 2),
                "exit_price": b["exit_price"],
                "entry_price": b["entry_price"],
                "volume": b["volume"],
                "side": b["side"],
                "symbol": b["symbol"],
                "time": b["time"],
            }
        return cleaned

    def position_closed_pnl(
        self,
        position_id: str,
        days: int = 90,
        *,
        since: Optional[datetime] = None,
    ) -> Optional[dict[str, Any]]:
        """Sum deals for a position id — true XM realised P&L (incl. swap/commission)."""
        pid = str(position_id or "").strip()
        if not pid:
            return None
        try:
            if since is not None:
                m = self.closed_pnl_map_since(since)
            else:
                m = self.closed_pnl_map_since(days=days)
            hit = m.get(pid)
            if hit is None and days < 90 and since is None:
                # widen once for older desk positions
                hit = self.closed_pnl_map_since(days=90).get(pid)
            return hit
        except Exception as exc:
            logger.debug("position_closed_pnl failed %s: %s", pid, exc)
            return None

    def position_exists(self, position_id: str) -> bool:
        for pos in self.get_open_positions():
            pid = str(pos.get("id") or pos.get("positionId") or "")
            if pid == str(position_id):
                return True
        return False

    def symbol_volume_step(self) -> float:
        try:
            spec = self._request(
                "GET",
                f"/symbols/{self.symbol}/specification",
                cache_key=f"spec:{self.symbol}",
                cache_ttl=_TTL_SPEC,
            )
            step = spec.get("volumeStep") or spec.get("minVolume") or 0.01
            return max(float(step), 0.01)
        except Exception as exc:
            logger.debug("symbol volume step fallback 0.01: %s", exc)
            return 0.01

    def fetch_candles(self, timeframe: str, bars: int = 500, symbol: Optional[str] = None):
        """Historical OHLCV — cached per symbol+timeframe to avoid 429s."""
        import pandas as pd

        sym = self._sym(symbol)
        tf = str(timeframe).lower().strip()
        limit = max(1, min(int(bars), 1000))
        cache_key = f"candles:{sym}:{tf}:{limit}"
        ttl = _TTL_CANDLES.get(tf, 60.0)

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached.copy()

        path = (
            f"/historical-market-data/symbols/{sym}"
            f"/timeframes/{tf}/candles?limit={limit}"
        )
        raw = self._market_data_request(path, cache_key=cache_key + ":raw", cache_ttl=ttl)
        if not isinstance(raw, list) or not raw:
            stale = self._cache_get_stale(cache_key)
            if stale is not None:
                return stale.copy()
            raise ValueError(f"MetaAPI returned no {tf} candles for {sym}")

        rows = []
        for c in raw:
            t = c.get("time")
            if not t:
                continue
            rows.append(
                {
                    "time": t,
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c.get("tickVolume") or c.get("volume") or 0),
                }
            )
        if not rows:
            raise ValueError(f"MetaAPI {sym} {tf} candles empty after parse")

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out = df[["open", "high", "low", "close", "volume"]]
        self._cache_set(cache_key, out, ttl)
        logger.info("MetaAPI candles OK: %s %s (%s bars, cache %.0fs)", sym, tf, len(out), ttl)
        return out.copy()

    def current_book(
        self,
        symbol: Optional[str] = None,
        *,
        keep_subscription: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Broker DOM (aggregated depth) via MetaAPI current-book.

        XM returns price/volume levels — not venue order_ids (not true L3 MBO).
        """
        sym = self._sym(symbol)
        q = "keepSubscription=true" if keep_subscription else "keepSubscription=false"
        path = f"/symbols/{sym}/current-book?{q}"
        cache_key = f"book:{sym}"
        try:
            data = self._request(
                "GET",
                path,
                cache_key=cache_key,
                cache_ttl=1.5,
            )
            if isinstance(data, dict) and data.get("book") is not None:
                return data
        except Exception as exc:
            logger.debug("MetaAPI current-book client failed (%s): %s", sym, exc)
        try:
            data = self._market_data_request(
                path,
                cache_key=f"md:{cache_key}",
                cache_ttl=1.5,
            )
            if isinstance(data, dict) and data.get("book") is not None:
                return data
        except Exception as exc:
            logger.debug("MetaAPI current-book market-data failed (%s): %s", sym, exc)
        return None

    def current_price(self, symbol: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Live bid/ask — cached ~2.5s per symbol."""
        sym = self._sym(symbol)
        try:
            data = self._request(
                "GET",
                f"/symbols/{sym}/current-price",
                cache_key=f"price:{sym}",
                cache_ttl=_TTL_PRICE,
            )
            if not isinstance(data, dict):
                return self._synthetic_quote(sym)
            bid = data.get("bid")
            ask = data.get("ask")
            if bid is None and ask is None:
                return self._synthetic_quote(sym)
            bid_f = float(bid) if bid is not None else float(ask)
            ask_f = float(ask) if ask is not None else float(bid)
            mid = self._round_price((bid_f + ask_f) / 2.0, sym)
            payload = {
                "symbol": data.get("symbol") or sym,
                "bid": bid_f,
                "ask": ask_f,
                "price": mid,
                "time": data.get("time") or data.get("brokerTime"),
            }
            self._cache_set(f"price:{sym}", payload, _TTL_PRICE)
            return payload
        except Exception as exc:
            logger.debug("MetaAPI current_price failed (%s): %s", sym, exc)
            stale = self._cache_get_stale(f"price:{sym}")
            if isinstance(stale, dict):
                bid = stale.get("bid")
                ask = stale.get("ask")
                if bid is not None or ask is not None:
                    bid_f = float(bid) if bid is not None else float(ask)
                    ask_f = float(ask) if ask is not None else float(bid)
                    return {
                        "symbol": stale.get("symbol") or sym,
                        "bid": bid_f,
                        "ask": ask_f,
                        "price": self._round_price((bid_f + ask_f) / 2.0, sym),
                        "time": stale.get("time") or stale.get("brokerTime"),
                        "stale": True,
                    }
            return self._synthetic_quote(sym)

    def _synthetic_quote(self, sym: str) -> Optional[dict[str, Any]]:
        """Fallback bid/ask from the last candle close when live price is empty.

        Some XM CFDs (OILCash, US30Cash) don't return REST current-price even
        though candles work. Synthesize a mark from the latest 1m close and a
        spec-derived spread so orders can still be placed / rebased.
        """
        try:
            df = self.fetch_candles("1m", bars=2, symbol=sym)
            if df is None or df.empty:
                return None
            close = float(df["close"].iloc[-1])
        except Exception as exc:
            logger.debug("synthetic quote candle fetch failed (%s): %s", sym, exc)
            return None
        try:
            spec = self._request(
                "GET",
                f"/symbols/{sym}/specification",
                cache_key=f"spec:{sym}",
                cache_ttl=_TTL_SPEC,
            )
            points = float(spec.get("points") or spec.get("point") or 0.0)
        except Exception:
            points = 0.0
        spread = points * 6 if points > 0 else close * 0.0002
        bid_f = close - spread / 2.0
        ask_f = close + spread / 2.0
        mid = self._round_price((bid_f + ask_f) / 2.0, sym)
        payload = {
            "symbol": sym,
            "bid": bid_f,
            "ask": ask_f,
            "price": mid,
            "time": None,
            "synthetic": True,
        }
        logger.info("MetaAPI current_price empty for %s — using synthetic mark %.5f", sym, mid)
        return payload

    def round_volume(self, volume: float) -> float:
        step = self.symbol_volume_step()
        rounded = round(round(float(volume) / step) * step, 2)
        return max(rounded, step)


mt5_client = MetaApiClient()
