"""Strategy Lab API — standard / max-hunt / quant-max / million-per-pair."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.security import get_current_user

router = APIRouter(prefix="/api/strategy-lab", tags=["strategy-lab"])

_running = False
_last_error: Optional[str] = None
_run_mode: Optional[str] = None


@router.get("")
async def get_strategy_lab(_: dict = Depends(get_current_user)) -> dict[str, Any]:
    from analysis.strategy_lab import STRATEGIES, latest_strategy_lab
    from pathlib import Path
    import json

    data = latest_strategy_lab()
    progress = None
    prog_path = Path(__file__).resolve().parents[3] / "data" / "strategy_lab_progress.json"
    # parents: api -> app -> backend -> Algo root? 
    # File is backend/app/api/strategy_lab.py -> parents[3] = Algo root if structure is Algo/backend/app/api
    for candidate in (
        Path("/app/data/strategy_lab_progress.json"),
        Path(__file__).resolve().parents[3] / "data" / "strategy_lab_progress.json",
        Path(__file__).resolve().parents[2] / "data" / "strategy_lab_progress.json",
    ):
        if candidate.exists():
            try:
                progress = json.loads(candidate.read_text())
                break
            except Exception:
                pass

    return {
        "has_results": data is not None,
        "running": _running,
        "run_mode": _run_mode,
        "last_error": _last_error,
        "catalog": (data or {}).get("strategies") or STRATEGIES,
        "results": data,
        "progress": progress,
    }


@router.get("/live-pnl")
async def live_pnl_curves(
    _: dict = Depends(get_current_user),
    rebuild: bool = Query(False, description="Force rebuild equity curves from hunt winners"),
) -> dict[str, Any]:
    """1y Strategy Lab hunt equity race (gated winners) — not live demo trades."""
    from pathlib import Path

    out_path = Path("/app/data/strategy_lab_equity_1y.json")
    if not out_path.exists():
        out_path = Path(__file__).resolve().parents[3] / "data" / "strategy_lab_equity_1y.json"

    def _load_or_build() -> dict[str, Any]:
        from analysis.build_lab_equity_curves import build

        return build(force=bool(rebuild))

    try:
        if rebuild or not out_path.exists():
            payload = await asyncio.to_thread(_load_or_build)
        else:
            import json

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            if not payload.get("by_symbol"):
                payload = await asyncio.to_thread(_load_or_build)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hunt equity curves failed: {exc}") from exc

    # Shape expected by LivePnlRaceChart
    return {
        "since_epoch": False,
        "source": payload.get("source") or "duka_scalp_1y_hunt_winners",
        "mode": payload.get("mode"),
        "risk_usd": payload.get("risk_usd"),
        "window_days": payload.get("window_days"),
        "window_start": payload.get("window_start"),
        "window_end": payload.get("window_end"),
        "desk_reset_at": payload.get("window_start"),
        "n_trades": payload.get("n_trades"),
        "n_points": payload.get("n_points"),
        "symbols": payload.get("symbols") or [],
        "times": payload.get("times") or [],
        "desk": payload.get("desk") or [],
        "by_symbol": payload.get("by_symbol") or {},
        "finals": payload.get("finals") or {},
        "desk_final": payload.get("desk_final"),
        "per_pair": payload.get("per_pair") or {},
        "note": payload.get("note"),
        "generated_at": payload.get("generated_at"),
    }


@router.post("/run")
async def run_strategy_lab_endpoint(
    _: dict = Depends(get_current_user),
    years: float = Query(15.0, ge=2.0, le=20.0),
) -> dict[str, Any]:
    global _running, _last_error, _run_mode
    if _running:
        raise HTTPException(status_code=409, detail="Strategy Lab already running")
    from analysis.strategy_lab import run_strategy_lab as _run

    _running = True
    _run_mode = "standard"
    _last_error = None
    try:
        payload = await asyncio.to_thread(_run, years=years)
        return {"ok": True, "mode": "standard", "summary": payload.get("summary"), "best_by_pair": payload.get("best_by_pair")}
    except Exception as exc:
        _last_error = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _running = False
        _run_mode = None


@router.post("/run-max")
async def run_strategy_lab_max_endpoint(
    _: dict = Depends(get_current_user),
    years: float = Query(15.0, ge=2.0, le=20.0),
) -> dict[str, Any]:
    global _running, _last_error, _run_mode
    if _running:
        raise HTTPException(status_code=409, detail="Strategy Lab already running")
    from analysis.strategy_lab_hunt import run_strategy_lab_max as _run_max

    _running = True
    _run_mode = "max_hunt"
    _last_error = None
    try:
        payload = await asyncio.to_thread(_run_max, years=years)
        return {"ok": True, "mode": "max_hunt", "summary": payload.get("summary"), "hunt": payload.get("hunt")}
    except Exception as exc:
        _last_error = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _running = False
        _run_mode = None


@router.post("/run-quant-max")
async def run_quant_max_endpoint(
    _: dict = Depends(get_current_user),
    years: float = Query(15.0, ge=2.0, le=20.0),
) -> dict[str, Any]:
    """Multi-TF · SURVIVES-only · max candles available · live progress file."""
    global _running, _last_error, _run_mode
    if _running:
        raise HTTPException(status_code=409, detail="Strategy Lab already running")
    from analysis.strategy_lab_quant_max import run_quant_max as _run_q

    _running = True
    _run_mode = "quant_max"
    _last_error = None
    try:
        payload = await asyncio.to_thread(_run_q, years=years)
        return {
            "ok": True,
            "mode": "quant_max_survives_only",
            "summary": payload.get("summary"),
            "hunt": payload.get("hunt"),
            "limitations": payload.get("limitations"),
            "best_by_pair": payload.get("best_by_pair"),
        }
    except Exception as exc:
        _last_error = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _running = False
        _run_mode = None


@router.post("/run-million-per-pair")
async def run_million_per_pair_endpoint(
    _: dict = Depends(get_current_user),
    years: float = Query(15.0, ge=2.0, le=20.0),
    target: int = Query(1_000_000, ge=100_000, le=5_000_000),
) -> dict[str, Any]:
    """All retail+hedge strategies · accumulate ≥target closed trades per desk pair."""
    global _running, _last_error, _run_mode
    if _running:
        raise HTTPException(status_code=409, detail="Strategy Lab already running")
    from analysis.strategy_lab_million_pair import run_million_per_pair as _run_m

    _running = True
    _run_mode = "million_per_pair"
    _last_error = None
    try:
        payload = await asyncio.to_thread(_run_m, years=years, target=target)
        return {
            "ok": True,
            "mode": "million_per_pair_all_strategies",
            "summary": payload.get("summary"),
            "hunt": payload.get("hunt"),
            "best_by_pair": payload.get("best_by_pair"),
        }
    except Exception as exc:
        _last_error = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _running = False
        _run_mode = None
