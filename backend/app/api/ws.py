"""WebSocket endpoint for live signal updates."""

import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from jose import JWTError
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.core.security import decode_token
from app.models.user import User
from app.services.market_cache import market_cache
from app.services.ws_manager import ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


async def _authenticate_ws(token: str) -> User:
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise ValueError("Invalid token type")
        user_id = payload.get("user_id")
    except (JWTError, ValueError) as exc:
        raise ValueError("Invalid token") from exc

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise ValueError("User not found")
        return user


@router.websocket("/ws/signals")
async def signals_ws(websocket: WebSocket, token: str = Query(...)):
    try:
        await _authenticate_ws(token)
    except ValueError:
        await websocket.close(code=4401)
        return

    await ws_manager.connect(websocket)
    try:
        latest = await market_cache.get_signal()
        if latest:
            await websocket.send_json({"type": "signal_update", "signal": latest})

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WebSocket closed: %s", exc)
    finally:
        await ws_manager.disconnect(websocket)
