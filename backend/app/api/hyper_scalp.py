"""Hyper-scalp control API."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/hyper-scalp", tags=["hyper-scalp"])


class ToggleBody(BaseModel):
    enabled: bool


class SettingsBody(BaseModel):
    lot_mult: Optional[float] = Field(default=None, ge=0.25, le=3.0)
    max_hold_sec: Optional[float] = Field(default=None, ge=3.0, le=1800.0)
    cycle_sec: Optional[float] = Field(default=None, ge=0.2, le=5.0)
    max_open: Optional[int] = Field(default=None, ge=1, le=20)
    cooldown_sec: Optional[float] = Field(default=None, ge=0.2, le=60.0)
    target_tpm: Optional[int] = Field(default=None, ge=1, le=60)
    one_per_symbol: Optional[int] = Field(default=None, ge=0, le=1)
    risk_pct: Optional[float] = Field(default=None, ge=1.0, le=50.0)
    # Zenith momentum on 1m — trend pullback with partials
    ema_tf: Optional[str] = Field(default=None)  # entry TF (1m)
    bias_tf: Optional[str] = Field(default=None)  # structure TF (5m)
    time_stop_enabled: Optional[int] = Field(default=None, ge=0, le=1)
    disp_atr: Optional[float] = Field(default=None, ge=0.3, le=2.0)
    pull_max_atr: Optional[float] = Field(default=None, ge=0.3, le=2.0)
    tp_r: Optional[float] = Field(default=None, ge=1.0, le=4.0)
    tp1_r: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    tp1_frac: Optional[float] = Field(default=None, ge=0.2, le=0.8)
    cost_mult: Optional[float] = Field(default=None, ge=1.0, le=8.0)
    min_range_bps: Optional[float] = Field(default=None, ge=0.3, le=10.0)
    swing_lookback: Optional[int] = Field(default=None, ge=3, le=40)
    min_rr: Optional[float] = Field(default=None, ge=0.8, le=4.0)


@router.get("/status")
async def status(_: User = Depends(get_current_user)) -> dict[str, Any]:
    from strategy.hyper_scalp import snapshot
    from app.services.hyper_scalp_worker import hyper_scalp_worker

    snap = snapshot()
    snap["worker_alive"] = bool(
        hyper_scalp_worker._task and not hyper_scalp_worker._task.done()  # noqa: SLF001
    )
    return snap


@router.post("/toggle")
async def toggle(body: ToggleBody, _: User = Depends(get_current_user)) -> dict[str, Any]:
    from strategy.hyper_scalp import set_enabled

    return set_enabled(body.enabled)


@router.post("/settings")
async def settings(body: SettingsBody, _: User = Depends(get_current_user)) -> dict[str, Any]:
    from strategy.hyper_scalp import update_settings

    return update_settings(**body.model_dump(exclude_none=True))


@router.post("/reset-pnl")
async def reset_pnl(_: User = Depends(get_current_user)) -> dict[str, Any]:
    from strategy.hyper_scalp import reset_session_pnl

    return reset_session_pnl()
