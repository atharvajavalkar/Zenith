#!/usr/bin/env python3
"""DUKA SCALP / INTRADAY HUNT — past 1 year · same deep-gate process as live brains.

READ-ONLY for live: does NOT change SYMBOL_ENTRY_BRAIN / .env.

Same machinery as strategy_lab_duka_deep_hunt:
  • Full retail + hedge catalog
  • RR sweep (1.5 / 2 / 2.4 / 3 / 4)
  • Battery: full / OOS70 / 4-fold WF / spread+skip torture / MC
  • Per-bar walk on every loaded candle (no stride)

Scoped for frequency:
  • Window: last 365d of Dukascopy local hist
  • TFs: 1m · 3m · 5m · 15m (intraday / scalp — no 1h swing)
  • Deep gate: SURVIVES on 1y + positive OOS on last 90d (n≥20) + WF≥2

Outputs:
  data/lab_duka_scalp_1y_hunt.json
  data/lab_duka_scalp_1y_hunt.log
  data/strategy_lab_progress.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from analysis._edge_survive_bt import BOOK, RISK_USD  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import (  # noqa: E402
    HEAVY,
    _battery,
    _catalog,
    _passes_deep_gate as _passes_deep_gate_5y,  # noqa: F401 — ref only
    _rank_score,
)
from analysis.strategy_lab_intraday_hunt import _prog  # noqa: E402
from config import symbol_rr  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_duka_scalp_1y_hunt.json"
LOG = Path(ROOT) / "data" / "lab_duka_scalp_1y_hunt.log"
LOCK = Path(ROOT) / "data" / "lab_duka_scalp_1y_hunt.lock"
PROGRESS = Path(ROOT) / "data" / "strategy_lab_progress.json"

# Intraday / scalp only
TFS = ("1m", "3m", "5m", "15m")
RRS = (1.5, 2.0, 2.4, 3.0, 4.0)
WINDOW_DAYS = 365
RECENT_DAYS = 90  # gate confirmation window (replaces 5y in decade hunt)
# Heavy collectors: skip 1m/3m (same spirit as decade hunt skipping 5m/15m for HEAVY)
HEAVY_OK_TF = frozenset({"5m", "15m"})


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _load_frames_1y(sym: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    from analysis._edge_survive_bt import _prep
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    clear_hist_cache()
    frames: dict[str, pd.DataFrame] = {}
    start_s = start.isoformat()
    end_s = end.isoformat()
    for tf in TFS:
        if not has_local_hist(sym, tf):
            # try resample path via 1m
            if tf != "1m" and not has_local_hist(sym, "1m"):
                continue
        try:
            df = load_local_ohlcv(sym, tf=tf, start=start_s, end=end_s)
        except Exception as exc:
            _log(f"  load {sym} {tf} FAIL: {exc}")
            continue
        if df is None or len(df) < 400:
            continue
        need = [c for c in ("open", "high", "low", "close") if c in df.columns]
        frames[tf] = _prep(df[need].dropna())
        # trim exactly to window (load_local keeps 5d warmup)
        frames[tf] = frames[tf][frames[tf].index >= start]
        if len(frames[tf]) < 400:
            del frames[tf]
            continue
        _log(
            f"  load {sym} {tf}: {len(frames[tf]):,} bars "
            f"{frames[tf].index[0].date()}→{frames[tf].index[-1].date()}"
        )
    return frames


def _slice_days(df: pd.DataFrame, days: int, end: pd.Timestamp) -> pd.DataFrame:
    start = end - pd.Timedelta(days=days)
    return df[df.index >= start]


def _passes_scalp_gate(full_bat: dict, bat_recent: dict) -> bool:
    """SURVIVE 1y span + positive OOS on last 90d + enough sample + WF≥2."""
    if full_bat.get("verdict") != "SURVIVES":
        return False
    o = bat_recent.get("oos") or {}
    if float(o.get("exp_r") or 0) <= 0:
        return False
    if int((bat_recent.get("full") or {}).get("n") or 0) < 20:
        return False
    if int(full_bat.get("wf_green") or 0) < 2:
        return False
    # Prefer frequency for scalp: need enough 1y trades
    if int((full_bat.get("full") or {}).get("n") or 0) < 40:
        return False
    return True


def main() -> int:
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip() or "0")
            if pid and Path(f"/proc/{pid}").exists():
                _log(f"another scalp hunt running (pid {pid}) — exit")
                return 2
        except Exception:
            pass
    LOCK.write_text(str(os.getpid()))
    if LOG.exists():
        LOG.write_text("")

    t0 = time.time()
    catalog = _catalog()
    end = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=1), tz="UTC")
    start = end - pd.Timedelta(days=WINDOW_DAYS)

    _log(
        "DUKA SCALP/INTRADAY HUNT · past 1y · Dukascopy · "
        f"${RISK_USD}/R · TFs={TFS}\n"
        f"Window: {start.date()} → {end.date()} · recent gate={RECENT_DAYS}d\n"
        "Same catalog/RR/battery as deep hunt that produced live brains.\n"
        "LIVE BRAINS NOT MODIFIED\n"
    )
    _prog(
        status="running",
        phase="duka_scalp_1y_hunt",
        pct=1,
        message=(
            f"Scalp 1y hunt · {len(BOOK)} pairs · {len(catalog)} strategies · TFs {TFS}"
        ),
        candles_total=0,
    )

    per_pair: dict[str, Any] = {}
    survivors_book: dict[str, Any] = {}
    cells_total = 0
    survives_cells = 0
    candles_total = 0

    for pi, sym in enumerate(BOOK):
        rr_live = float(symbol_rr(sym, default=2.0))
        _log(f"\n######## {sym} · live RR={rr_live} ########")
        try:
            frames = _load_frames_1y(sym, start, end)
        except Exception as exc:
            _log(f"  LOAD FAIL {sym}: {exc}")
            per_pair[sym] = {"error": str(exc)}
            continue
        if not frames:
            per_pair[sym] = {"error": "no_local_hist"}
            continue
        candles_total += sum(len(df) for df in frames.values())

        scored: list[dict] = []
        for sid, family, label, fn in catalog:
            for tf in TFS:
                if tf not in frames:
                    continue
                if sid in HEAVY and tf not in HEAVY_OK_TF:
                    continue
                df_all = frames[tf]
                cells_total += 1
                _log(f"  … {sid}@{tf} ({len(df_all):,} bars)")
                _prog(
                    status="running",
                    phase="duka_scalp_1y_hunt",
                    pct=int(5 + 90 * (pi + 0.2) / len(BOOK)),
                    message=f"[{sym}] {sid}@{tf}",
                    symbol=sym,
                    candles_total=candles_total,
                )
                try:
                    entries = fn(df_all)
                    if not entries or len(entries) < 30:
                        continue
                    df_r = _slice_days(df_all, RECENT_DAYS, end)
                    try:
                        entries_r = fn(df_r) if len(df_r) >= 200 else []
                    except Exception:
                        entries_r = []
                    best_cell = None
                    for rr in RRS:
                        bat = _battery(df_all, entries, rr)
                        if len(df_r) < 200 or len(entries_r) < 15:
                            bat_r = {
                                "verdict": "DEAD",
                                "oos": {"exp_r": 0},
                                "full": {"n": 0},
                                "wf_green": 0,
                            }
                        else:
                            bat_r = _battery(df_r, entries_r, rr)
                        gate = _passes_scalp_gate(bat, bat_r)
                        cell = {
                            "strategy": sid,
                            "family": family,
                            "label": label,
                            "tf": tf,
                            "rr": rr,
                            "full_span": bat,
                            "last_90d": {
                                "verdict": bat_r.get("verdict"),
                                "full": bat_r.get("full"),
                                "oos": bat_r.get("oos"),
                                "wf_green": bat_r.get("wf_green"),
                            },
                            "deep_gate": gate,
                            "score": round(
                                _rank_score(bat, family) + (2.5 if gate else 0), 3
                            ),
                        }
                        # scalp bonus: more trades (frequency)
                        n_tr = int((bat.get("full") or {}).get("n") or 0)
                        cell["score"] = round(cell["score"] + min(2.0, n_tr / 200.0), 3)
                        if best_cell is None or cell["score"] > best_cell["score"]:
                            best_cell = cell
                    if best_cell:
                        scored.append(best_cell)
                        if best_cell["full_span"].get("verdict") == "SURVIVES":
                            survives_cells += 1
                        if best_cell["deep_gate"]:
                            _log(
                                f"  ✓ GATE {sid:12} {tf:3} RR={best_cell['rr']} "
                                f"n={best_cell['full_span']['full']['n']} "
                                f"OOS_E[R]={best_cell['full_span']['oos']['exp_r']:+.3f} "
                                f"90dOOS={best_cell['last_90d']['oos'].get('exp_r', 0):+.3f} "
                                f"{best_cell['full_span']['verdict']}"
                            )
                except Exception as exc:
                    _log(f"  skip {sid}@{tf}: {exc}")

            _prog(
                status="running",
                phase="duka_scalp_1y_hunt",
                pct=int(5 + 90 * (pi + 0.5) / len(BOOK)),
                message=(
                    f"[{sym}] scored {len(scored)} cells · "
                    f"survives_cells={survives_cells}"
                ),
                symbol=sym,
                candles_total=candles_total,
            )

        scored.sort(key=lambda r: float(r.get("score") or -1e9), reverse=True)
        gated = [r for r in scored if r.get("deep_gate")]
        pool = gated if gated else [
            r for r in scored if r["full_span"].get("verdict") == "SURVIVES"
        ]
        if not pool:
            pool = [r for r in scored if r["full_span"].get("verdict") != "DEAD"]
        winner = pool[0] if pool else None

        top_hedge = next(
            (r for r in scored if r["family"] == "hedge" and r.get("deep_gate")), None
        )
        if top_hedge is None:
            top_hedge = next(
                (
                    r
                    for r in scored
                    if r["family"] == "hedge"
                    and r["full_span"].get("verdict") == "SURVIVES"
                ),
                None,
            )
        top_retail = next(
            (r for r in scored if r["family"] == "retail" and r.get("deep_gate")), None
        )
        # Highest-frequency gated (or survives) cell
        by_n = sorted(
            [r for r in (gated or scored) if r["full_span"].get("full")],
            key=lambda r: int(r["full_span"]["full"].get("n") or 0),
            reverse=True,
        )
        top_freq = by_n[0] if by_n else None

        per_pair[sym] = {
            "rr_live": rr_live,
            "cells": len(scored),
            "gated_count": len(gated),
            "winner": winner,
            "top_hedge": top_hedge,
            "top_retail": top_retail,
            "top_freq": top_freq,
            "top10": scored[:10],
            "gated": gated[:8],
        }
        if winner and winner.get("deep_gate"):
            survivors_book[sym] = {
                "strategy": winner["strategy"],
                "family": winner["family"],
                "tf": winner["tf"],
                "rr": winner["rr"],
                "label": winner["label"],
                "verdict": winner["full_span"]["verdict"],
                "oos_exp_r": winner["full_span"]["oos"]["exp_r"],
                "full_pnl": winner["full_span"]["full"]["pnl"],
                "n": winner["full_span"]["full"]["n"],
                "score": winner["score"],
            }
            _log(
                f"  ★ {sym} WINNER {winner['strategy']}@{winner['tf']} "
                f"RR={winner['rr']} n={winner['full_span']['full']['n']} "
                f"score={winner['score']} "
                f"PnL=${winner['full_span']['full']['pnl']:,.0f}"
            )
        else:
            _log(f"  ✗ {sym}: no scalp-gate survivor (best non-dead in top10)")

        OUT.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "status": "running",
                    "survivors_book": survivors_book,
                    "per_pair": {
                        k: {
                            "gated_count": v.get("gated_count"),
                            "winner": v.get("winner"),
                            "top_freq": v.get("top_freq"),
                        }
                        for k, v in per_pair.items()
                    },
                    "candles_total": candles_total,
                    "elapsed_s": round(time.time() - t0, 1),
                },
                indent=2,
                default=str,
            )
        )

    n_gate = len(survivors_book)
    proposed_map = ",".join(
        f"{s}:{survivors_book[s]['strategy']}" for s in BOOK if s in survivors_book
    )
    proposed_tf = ",".join(
        f"{s}:{survivors_book[s]['tf']}" for s in BOOK if s in survivors_book
    )
    proposed_rr = ",".join(
        f"{s}:{survivors_book[s]['rr']}" for s in BOOK if s in survivors_book
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "duka_scalp_1y_hunt_readonly",
        "source": "dukascopy_local_hist",
        "window_days": WINDOW_DAYS,
        "recent_gate_days": RECENT_DAYS,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "risk_usd": RISK_USD,
        "tfs": list(TFS),
        "strategies_tested": len(catalog),
        "cells_total": cells_total,
        "survives_cells": survives_cells,
        "candles_total": candles_total,
        "deep_gate_survivors": n_gate,
        "survivors_book": survivors_book,
        "proposed_SYMBOL_ENTRY_BRAIN": proposed_map,
        "proposed_SYMBOL_SCAN_TFS": proposed_tf,
        "proposed_SYMBOL_RR": proposed_rr,
        "per_pair": per_pair,
        "limits": {
            "live_unchanged": True,
            "note": (
                "1y intraday/scalp hunt. Same OOS/WF/torture/MC battery as decade deep hunt. "
                "Gate uses last-90d OOS instead of 5y. Heavy brains skip 1m/3m."
            ),
        },
        "summary": {
            "pairs_with_deep_gate": n_gate,
            "desk_success_rate_pct": round(100.0 * n_gate / max(len(BOOK), 1), 2),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _prog(
        status="done",
        phase="duka_scalp_1y_hunt",
        pct=100,
        message=(
            f"Scalp 1y hunt done · {n_gate}/{len(BOOK)} gated · "
            f"{survives_cells} SURVIVES cells · {payload['elapsed_s']}s"
        ),
        candles_total=candles_total,
    )
    _log(
        f"\n→ Scalp-gate survivors: {n_gate}/{len(BOOK)}\n"
        f"→ Proposed map (NOT applied): {proposed_map or '(none)'}\n"
        f"→ Proposed TFs: {proposed_tf or '(none)'}\n"
        f"→ Wrote {OUT} in {payload['elapsed_s']}s"
    )
    try:
        LOCK.unlink(missing_ok=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        try:
            LOCK.unlink(missing_ok=True)
        except Exception:
            pass
