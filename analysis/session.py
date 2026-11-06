"""Trading session detection with NY/IST time windows."""

from contextvars import ContextVar
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from typing import Optional, Tuple

# Set this to a UTC datetime during backtesting so session checks use bar time
_backtest_bar_time: ContextVar[Optional[datetime]] = ContextVar(
    "_backtest_bar_time", default=None
)


def set_backtest_bar_time(dt: Optional[datetime]) -> None:
    """Inject historical bar time so session checks work correctly in backtest."""
    _backtest_bar_time.set(dt)


def get_backtest_bar_time() -> Optional[datetime]:
    """UTC bar time during backtest replay; None in live."""
    return _backtest_bar_time.get()


def get_ny_time_and_session() -> Tuple[datetime, float, str]:
    override = _backtest_bar_time.get()
    if override is not None:
        # Backtest mode — use the bar's UTC timestamp
        ref = override if override.tzinfo else override.replace(tzinfo=timezone.utc)
        ny = ref.astimezone(ZoneInfo("America/New_York"))
        ist = ref.astimezone(ZoneInfo("Asia/Kolkata"))
    else:
        ny = datetime.now(ZoneInfo("America/New_York"))
        ist = datetime.now(ZoneInfo("Asia/Kolkata"))

    h_ny = ny.hour + ny.minute / 60
    h_ist = ist.hour + ist.minute / 60

    if 18.5 <= h_ist <= 20:
        return ny, 1.35, "SCALP_WINDOW"

    if 7 <= h_ny < 11:
        return ny, 1.25, "NY_POWER"
    if 11 <= h_ny < 13:
        return ny, 0.90, "NY_MIDDAY"
    if 13 <= h_ny < 16:
        return ny, 1.05, "NY_AFTERNOON"

    return ny, 0.0, "NO_TRADE_SESSION"
