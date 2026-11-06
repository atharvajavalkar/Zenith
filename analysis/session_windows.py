"""Session windows for FRVP (fixed-range volume profile) from prior sessions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

# Named FX/metal sessions (exchange-local hours)
# Asia: Tokyo 09:00–15:00 JST ≈ 00:00–06:00 UTC
# London: 08:00–16:00 London
# New York: 09:30–16:00 America/New_York


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def current_session_name(now: Optional[datetime] = None) -> str:
    """Coarse session label overlapping NY/London/Asia."""
    ref = _as_utc(now or datetime.now(timezone.utc))
    ny = ref.astimezone(ZoneInfo("America/New_York"))
    ldn = ref.astimezone(ZoneInfo("Europe/London"))
    h_ny = ny.hour + ny.minute / 60.0
    h_ldn = ldn.hour + ldn.minute / 60.0
    if 9.5 <= h_ny < 16.0:
        return "NY"
    if 8.0 <= h_ldn < 16.0:
        return "LONDON"
    # Asia (approximate UTC 0–6 and late Tokyo)
    h_utc = ref.hour + ref.minute / 60.0
    if 0.0 <= h_utc < 7.0 or 23.0 <= h_utc:
        return "ASIA"
    return "OFF"


def previous_session_window(
    now: Optional[datetime] = None,
) -> Tuple[datetime, datetime, str]:
    """
    Return (start_utc, end_utc, session_name) of the last *completed* major session.

    Prefer NY → London → Asia walk-back from now.
    """
    ref = _as_utc(now or datetime.now(timezone.utc))
    candidates: list[Tuple[datetime, datetime, str]] = []

    # Build last 3 calendar days of each session
    for day_back in range(0, 4):
        day = (ref - timedelta(days=day_back)).date()

        # NY 09:30–16:00 America/New_York
        ny_tz = ZoneInfo("America/New_York")
        ny_start = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ny_tz).astimezone(
            timezone.utc
        )
        ny_end = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ny_tz).astimezone(
            timezone.utc
        )
        candidates.append((ny_start, ny_end, "NY"))

        # London 08:00–16:00
        ldn_tz = ZoneInfo("Europe/London")
        ldn_start = datetime(day.year, day.month, day.day, 8, 0, tzinfo=ldn_tz).astimezone(
            timezone.utc
        )
        ldn_end = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ldn_tz).astimezone(
            timezone.utc
        )
        candidates.append((ldn_start, ldn_end, "LONDON"))

        # Asia (Tokyo cash) 00:00–06:00 UTC proxy
        asia_start = datetime(day.year, day.month, day.day, 0, 0, tzinfo=timezone.utc)
        asia_end = datetime(day.year, day.month, day.day, 6, 0, tzinfo=timezone.utc)
        candidates.append((asia_start, asia_end, "ASIA"))

    # Last completed (end <= now), prefer most recent end
    done = [c for c in candidates if c[1] <= ref]
    if not done:
        # Fallback: prior calendar day UTC
        end = ref.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(hours=24)
        return start, end, "PRIOR_DAY"
    done.sort(key=lambda x: x[1], reverse=True)
    return done[0]


def prior_day_window(now: Optional[datetime] = None) -> Tuple[datetime, datetime, str]:
    """Previous full UTC day — backup FRVP range."""
    ref = _as_utc(now or datetime.now(timezone.utc))
    end = ref.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=1)
    return start, end, "PRIOR_UTC_DAY"
