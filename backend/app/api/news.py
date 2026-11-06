"""Multi-market world news map API."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi import APIRouter, Depends

_ALGO_ROOT = Path(__file__).resolve().parents[3]
if str(_ALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGO_ROOT))

from data.gold_news import build_world_desk_news  # noqa: E402

from app.core.security import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/news", tags=["news"])


@router.get("/world")
@router.get("/gold-world")
async def world_desk_news(_: User = Depends(get_current_user)):
    """
    Multi-market world desk: metals, FX majors, crypto (+ oil colour).
    Sources: Forex Factory calendar + Yahoo Finance.
    """
    return await asyncio.to_thread(build_world_desk_news)
