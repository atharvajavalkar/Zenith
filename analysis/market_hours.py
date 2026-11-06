"""XM GOLD (XAU) market hours.

XM server time is GMT+2 in winter and GMT+3 in summer, aligned with US DST
(same schedule as New York close charts). Gold session:

  Mon–Thu  01:05 – 23:55  (server)
  Friday   01:05 – 23:50  (server)
  Weekend  closed until Monday 01:05

Daily maintenance gap (~23:55/23:50 → 01:05) is also treated as closed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

_NY = ZoneInfo("America/New_York")

OPEN_HM = 1 + 5 / 60.0       # 01:05
CLOSE_WEEKDAY = 23 + 55 / 60.0  # 23:55 Mon–Thu
CLOSE_FRIDAY = 23 + 50 / 60.0   # 23:50 Friday


def xm_server_now(now: Optional[datetime] = None) -> datetime:
    """Current (or provided) instant as XM server wall-clock (aware)."""
    if now is None:
        try:
            from analysis.session import get_backtest_bar_time

            override = get_backtest_bar_time()
            if override is not None:
                now = override
        except Exception:
            now = None
    if now is None:
        utc = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        utc = now.replace(tzinfo=timezone.utc)
    else:
        utc = now.astimezone(timezone.utc)
    ny = utc.astimezone(_NY)
    offset_h = 2 + (1 if bool(ny.dst()) else 0)
    return utc.astimezone(timezone(timedelta(hours=offset_h)))


def _hm(dt: datetime) -> float:
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def _at(srv: datetime, days: int, hour: int = 1, minute: int = 5) -> datetime:
    d = (srv + timedelta(days=days)).date()
    return datetime(d.year, d.month, d.day, hour, minute, 0, tzinfo=srv.tzinfo)


def next_gold_open(now: Optional[datetime] = None) -> datetime:
    """Next XM GOLD open in server timezone."""
    srv = xm_server_now(now)
    hm = _hm(srv)
    wd = srv.weekday()  # Mon=0 … Sun=6

    # Still before today's open on a weekday → today 01:05
    if wd <= 4 and hm < OPEN_HM:
        return _at(srv, 0)

    # After Friday close → Monday 01:05
    if wd == 4 and hm >= CLOSE_FRIDAY:
        return _at(srv, 3)
    if wd == 5:  # Saturday
        return _at(srv, 2)
    if wd == 6:  # Sunday
        return _at(srv, 1)

    # Mon–Thu after close (daily break) → next day 01:05
    close = CLOSE_FRIDAY if wd == 4 else CLOSE_WEEKDAY
    if wd <= 4 and hm >= close:
        return _at(srv, 1)

    # Currently open — next open is tomorrow (or Monday after Friday session)
    if wd == 4:
        return _at(srv, 3)
    return _at(srv, 1)


def is_gold_market_open(now: Optional[datetime] = None) -> bool:
    return bool(get_gold_market_status(now)["is_open"])


def get_gold_market_status(now: Optional[datetime] = None) -> dict[str, Any]:
    """
    Returns is_open, status ('open'|'daily_break'|'weekend'), message,
    next_open, next_open_iso, next_open_display, server_time_display.
    """
    srv = xm_server_now(now)
    wd = srv.weekday()
    hm = _hm(srv)
    off = int(srv.utcoffset().total_seconds() // 3600) if srv.utcoffset() else 2
    server_disp = srv.strftime("%a %H:%M") + f" (XM GMT+{off})"

    def _closed(status: str, label: str) -> dict[str, Any]:
        nxt = next_gold_open(now)
        ny_open = nxt.astimezone(_NY)
        return {
            "is_open": False,
            "status": status,
            "label": label,
            "message": (
                "XM gold is closed. "
                f"Trading resumes {nxt.strftime('%A %H:%M')} XM server "
                f"(~{ny_open.strftime('%a %I:%M %p')} New York)."
            ),
            "next_open": nxt,
            "next_open_iso": nxt.astimezone(timezone.utc).isoformat(),
            "next_open_display": nxt.strftime("%a %H:%M XM"),
            "server_time_display": server_disp,
        }

    if wd >= 5:
        return _closed("weekend", "Weekend — market closed")

    close_hm = CLOSE_FRIDAY if wd == 4 else CLOSE_WEEKDAY
    if hm < OPEN_HM or hm >= close_hm:
        if wd == 4 and hm >= close_hm:
            return _closed("weekend", "Weekend — market closed")
        return _closed("daily_break", "Daily break — market closed")

    return {
        "is_open": True,
        "status": "open",
        "label": "Market open",
        "message": "XM gold market is open",
        "next_open": None,
        "next_open_iso": None,
        "next_open_display": None,
        "server_time_display": server_disp,
    }
