"""Trading session detection with NY/IST time windows."""

from datetime import datetime
from zoneinfo import ZoneInfo


from typing import Tuple


def get_ny_time_and_session() -> Tuple[datetime, float, str]:
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
