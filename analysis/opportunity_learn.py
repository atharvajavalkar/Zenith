"""Log opportunity-board exposures for per-instrument RL."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

_LOCK = threading.Lock()
_PATH = Path(__file__).resolve().parents[1] / "data" / "opportunity_log.jsonl"


def record_opportunity(
    *,
    symbol: str,
    exec_tf: str,
    signal: str,
    action: str,
    mc: Optional[float] = None,
    detail: Optional[str] = None,
) -> None:
    """action: seen | entered | skipped | blocked"""
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "symbol": str(symbol).upper(),
        "exec_tf": exec_tf,
        "signal": signal,
        "action": action,
        "mc": mc,
        "detail": detail,
    }
    with _LOCK:
        try:
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            with _PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception:
            pass


def _parse_ts(ts: str) -> float:
    try:
        # 2026-07-14T01:02:03Z
        return time.mktime(time.strptime(ts.replace("Z", ""), "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return 0.0


def recent_board_stats(symbol: Optional[str] = None, hours: float = 24.0) -> dict[str, Any]:
    """
    Aggregate recent board activity for RL soft features.
    Returns per-symbol counts + avg MC for seen/entered.
    """
    cutoff = time.time() - max(1.0, hours) * 3600.0
    counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"seen": 0, "entered": 0, "blocked": 0, "skipped": 0, "mc_sum": 0.0, "mc_n": 0}
    )
    with _LOCK:
        if not _PATH.exists():
            return {"hours": hours, "instruments": {}}
        try:
            lines = _PATH.read_text(encoding="utf-8").splitlines()[-4000:]
        except Exception:
            return {"hours": hours, "instruments": {}}

    for line in lines:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if _parse_ts(str(row.get("ts") or "")) < cutoff:
            continue
        sym = str(row.get("symbol") or "").upper()
        if not sym:
            continue
        if symbol and sym != str(symbol).upper():
            continue
        action = str(row.get("action") or "")
        bucket = counts[sym]
        if action in bucket:
            bucket[action] += 1
        mc = row.get("mc")
        if mc is not None:
            try:
                bucket["mc_sum"] += float(mc)
                bucket["mc_n"] += 1
            except Exception:
                pass

    out: dict[str, Any] = {}
    for sym, b in counts.items():
        avg_mc = (b["mc_sum"] / b["mc_n"]) if b["mc_n"] else None
        out[sym] = {
            "seen": b["seen"],
            "entered": b["entered"],
            "blocked": b["blocked"],
            "skipped": b["skipped"],
            "avg_mc": round(avg_mc, 1) if avg_mc is not None else None,
            "fill_rate": round(b["entered"] / b["seen"], 3) if b["seen"] else None,
        }
    return {"hours": hours, "instruments": out}


def board_soft_deltas(symbol: str, hours: float = 24.0) -> dict[str, float]:
    """
    Soft RL nudges from opportunity exposure (not hard rewards).
    High-quality recurring setups → slightly ease MC; chronic blocks → tighten.
    """
    stats = recent_board_stats(symbol, hours=hours)["instruments"].get(
        str(symbol).upper(), {}
    )
    seen = int(stats.get("seen") or 0)
    entered = int(stats.get("entered") or 0)
    blocked = int(stats.get("blocked") or 0)
    avg_mc = stats.get("avg_mc")
    mc_delta = 0.0
    aggression = 0.0
    if seen >= 8 and avg_mc is not None and avg_mc >= 55 and blocked <= entered:
        mc_delta = -1.5
        aggression = 0.03
    elif blocked >= 5 and entered == 0 and seen >= 5:
        mc_delta = 2.0
        aggression = -0.04
    return {
        "mc_floor_delta": mc_delta,
        "aggression": aggression,
        "board_seen": float(seen),
        "board_entered": float(entered),
    }
