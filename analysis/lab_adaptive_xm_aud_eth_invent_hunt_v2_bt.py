#!/usr/bin/env python3
"""AUD/ETH invent hunt v2 — dense catalog until criteria met.

Criteria (both pairs):
  · 1y PnL ≥ 15% of account (default $10k → $1,500) at RISK_USD=$100/R
  · verdict_30d & verdict_15d in {PASS, SOFT}
  · verdict_1y ≠ FAIL
  · XM session spreads ON

Research-only. Live desk untouched.
Supports densify rounds 0→2 via AUD_ETH_DENSIFY env.
Resume-safe. Nebula: lab_adaptive_xm_aud_eth_invent_hunt.* (same prefix for UI).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.lab_adaptive_minute_day_v3_bt import _calendar  # noqa: E402
from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    _classify_regime,
    _sides_snr,
    _window_trades,
)
from analysis.lab_adaptive_xm_deep_hunt_bt import (  # noqa: E402
    TF_YEARS,
    WINDOWS,
    _entries,
    _load,
    _simulate,
    _summarize,
)
from analysis.lab_aud_eth_math_econ_invent_v2 import build_aud_eth_catalog_v2  # noqa: E402
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL  # noqa: E402

BOOK = ("AUDUSD", "ETHUSD")
# Intraday first (faster signal on short windows), scalp last
TFS = ("15m", "5m", "3m", "1m")

# Reuse invent hunt filenames so Nebula keeps following
OUT = Path(ROOT) / "data" / "lab_adaptive_xm_aud_eth_invent_hunt.json"
LOG = Path(ROOT) / "data" / "lab_adaptive_xm_aud_eth_invent_hunt.log"
PROGRESS = Path(ROOT) / "data" / "lab_adaptive_xm_aud_eth_invent_hunt.progress.json"
PARTIAL = Path(ROOT) / "data" / "lab_adaptive_xm_aud_eth_invent_hunt.partial.json"
ROUND_TAG = Path(ROOT) / "data" / "lab_adaptive_xm_aud_eth_invent_hunt.round.json"

ACCOUNT_USD = float(os.getenv("AUD_ETH_ACCOUNT_USD", "10000"))
MIN_RETURN = float(os.getenv("AUD_ETH_MIN_RETURN", "0.15"))
MIN_PNL_1Y = ACCOUNT_USD * MIN_RETURN
DENSIFY = int(os.getenv("AUD_ETH_DENSIFY", "0"))

# densify=0 leaner grids; densify≥1 wider
if DENSIFY <= 0:
    RR_GRID_V2 = (2.0, 3.0, 4.0)
    SL_GRID_FX_V2 = {
        "1m": (2.5, 3.0),
        "3m": (2.5, 3.0, 3.5),
        "5m": (2.0, 2.5, 3.0, 3.5),
        "15m": (2.0, 2.5, 3.0, 3.5),
    }
    SL_GRID_ETH_V2 = {
        "1m": (2.0, 2.5),
        "3m": (2.0, 2.5, 3.0),
        "5m": (2.0, 2.5, 3.0),
        "15m": (2.0, 2.5, 3.0),
    }
    GAP_GRID_V2 = {
        "1m": (3, 5),
        "3m": (2, 3),
        "5m": (2, 3, 5),
        "15m": (2, 3),
    }
else:
    RR_GRID_V2 = (2.0, 2.5, 3.0, 4.0)
    SL_GRID_FX_V2 = {
        "1m": (2.5, 3.0, 3.5),
        "3m": (2.5, 3.0, 3.5),
        "5m": (2.0, 2.5, 3.0, 3.5),
        "15m": (2.0, 2.5, 3.0, 3.5),
    }
    SL_GRID_ETH_V2 = {
        "1m": (2.0, 2.5, 3.0),
        "3m": (2.0, 2.5, 3.0),
        "5m": (1.5, 2.0, 2.5, 3.0),
        "15m": (1.5, 2.0, 2.5, 3.0),
    }
    GAP_GRID_V2 = {
        "1m": (3, 5, 8),
        "3m": (2, 3, 5),
        "5m": (2, 3, 5),
        "15m": (2, 3, 5),
    }


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    p["hunt"] = "aud_eth_invent"
    p["book"] = list(BOOK)
    p["densify"] = DENSIFY
    p["min_pnl_1y"] = MIN_PNL_1Y
    p["min_return"] = MIN_RETURN
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _save_partial(payload: dict) -> None:
    PARTIAL.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _n_green(wins: dict[str, dict]) -> int:
    n = 0
    for k in ("7d", "15d", "30d", "90d", "180d", "1y"):
        try:
            if float((wins.get(k) or {}).get("pnl") or 0) > 0:
                n += 1
        except Exception:
            pass
    return n


def _vrank(v: Any) -> float:
    if v == "PASS":
        return 2.0
    if v == "SOFT":
        return 1.0
    return 0.0


def _cell_score(wins: dict[str, dict]) -> float:
    def _q(wname: str, w: float) -> float:
        sm = wins.get(wname) or {}
        return w * (
            _vrank(sm.get("verdict")) * 1000.0
            + float(sm.get("pnl") or 0) * 0.02
            + float(sm.get("exp_r") or 0) * 50.0
            + float(sm.get("oos_exp_r") or 0) * 40.0
            + float(sm.get("wf_green") or 0) * 20.0
        )

    pnl1 = float((wins.get("1y") or {}).get("pnl") or 0)
    ret_bonus = 50_000.0 if pnl1 >= MIN_PNL_1Y else (pnl1 / max(MIN_PNL_1Y, 1.0)) * 10_000.0
    return (
        _q("30d", 120.0)
        + _q("15d", 80.0)
        + _q("7d", 40.0)
        + _n_green(wins) * 3_000.0
        + ret_bonus
        + _q("1y", 2.0)
        + _q("90d", 0.5)
    )


def _meets(row: dict) -> bool:
    return (
        row.get("verdict_30d") in ("PASS", "SOFT")
        and row.get("verdict_15d") in ("PASS", "SOFT")
        and row.get("verdict_1y") != "FAIL"
        and float(row.get("pnl_1y") or 0) >= MIN_PNL_1Y
    )


def _top_row(c: dict) -> dict:
    w = c["windows"]
    row = {
        "id": c["id"],
        "mode": c["mode"],
        "sl_atr": c["sl_atr"],
        "rr": c["rr"],
        "min_gap": c["min_gap"],
        "n_green": _n_green(w),
        "all_six": _n_green(w) >= 6,
        "verdict_1y": w["1y"]["verdict"],
        "verdict_30d": w["30d"]["verdict"],
        "verdict_90d": w["90d"]["verdict"],
        "verdict_7d": w["7d"]["verdict"],
        "verdict_15d": w["15d"]["verdict"],
        "pnl_1y": w["1y"]["pnl"],
        "exp_r_1y": w["1y"]["exp_r"],
        "oos_1y": w["1y"]["oos_exp_r"],
        "wf_1y": w["1y"]["wf_green"],
        "n_1y": w["1y"]["n"],
        "spR_1y": w["1y"]["avg_spread_r"],
        "pnl_180d": w["180d"]["pnl"],
        "pnl_90d": w["90d"]["pnl"],
        "pnl_30d": w["30d"]["pnl"],
        "pnl_15d": w["15d"]["pnl"],
        "pnl_7d": w["7d"]["pnl"],
        "return_1y": round(float(w["1y"]["pnl"]) / ACCOUNT_USD, 4),
    }
    row["wireable"] = _meets(row)
    row["_score"] = c.get("_score")
    return row


def _better(cand: dict, cur: dict | None) -> bool:
    if cur is None:
        return True
    cw, ow = int(bool(cand.get("wireable"))), int(bool(cur.get("wireable")))
    if cw != ow:
        return cw > ow
    for k in ("verdict_30d", "verdict_15d", "verdict_7d"):
        if _vrank(cand.get(k)) != _vrank(cur.get(k)):
            return _vrank(cand.get(k)) > _vrank(cur.get(k))
    if float(cand.get("pnl_1y") or 0) != float(cur.get("pnl_1y") or 0):
        return float(cand.get("pnl_1y") or 0) > float(cur.get("pnl_1y") or 0)
    return (cand.get("n_green") or 0) > (cur.get("n_green") or 0)


def main() -> int:
    # Fresh log section for this round
    with LOG.open("a") as f:
        f.write(f"\n\n===== V2 ROUND densify={DENSIFY} @ {datetime.now(timezone.utc).isoformat()} =====\n")
    t0 = time.time()

    # Resume key includes densify round so new catalog re-runs
    resume_tag = f"v2_d{DENSIFY}"
    done_keys: set[str] = set()
    per_pair: dict[str, Any] = {}
    if PARTIAL.exists():
        try:
            prev = json.loads(PARTIAL.read_text(encoding="utf-8"))
            if prev.get("resume_tag") == resume_tag:
                per_pair = prev.get("per_pair") or {}
                for sym, row in per_pair.items():
                    for tf, block in (row.get("by_tf") or {}).items():
                        if block.get("done"):
                            done_keys.add(f"{sym}:{tf}")
                _log(f"RESUME densify={DENSIFY} · {len(done_keys)} pair×tf done")
            else:
                _log(f"NEW ROUND densify={DENSIFY} (prior tag={prev.get('resume_tag')})")
        except Exception:
            pass

    est = 0
    for tf in TFS:
        for sym in BOOK:
            nv = len(
                build_aud_eth_catalog_v2(lean=(tf == "1m"), symbol=sym, densify=DENSIFY)
            )
            slg = SL_GRID_ETH_V2[tf] if sym == "ETHUSD" else SL_GRID_FX_V2[tf]
            est += nv * len(slg) * len(RR_GRID_V2) * len(GAP_GRID_V2[tf])

    _log(
        f"=== AUD/ETH INVENT v2 densify={DENSIFY} ===\n"
        f"book={BOOK}\n"
        f"criteria: return≥{MIN_RETURN:.0%} (${MIN_PNL_1Y:.0f}/y on ${ACCOUNT_USD:.0f}) "
        f"+ 30d/15d PASS|SOFT + 1y≠FAIL · XM spreads ON\n"
        f"live untouched · est_cells≈{est:,}\n"
    )
    _progress({"status": "running", "pct": 0, "cells_est": est, "round": resume_tag})

    cells_done = 0
    for pi, sym in enumerate(BOOK):
        per_pair.setdefault(
            sym,
            {
                "symbol": sym,
                "family": "math_econ_invent_v2",
                "by_tf": {},
                "best_overall": None,
                "best_wireable": None,
            },
        )
        for tf in TFS:
            key = f"{sym}:{tf}"
            if key in done_keys:
                _log(f"skip {key} (resume)")
                continue
            years = TF_YEARS[tf]
            lean = tf == "1m"
            variants = build_aud_eth_catalog_v2(lean=lean, symbol=sym, densify=DENSIFY)
            _log(
                f"\n######## V2 {sym} @ {tf} · {years}y · {len(variants)} modes · d={DENSIFY} ########"
            )
            _progress(
                {
                    "status": "running",
                    "pct": round(
                        100
                        * (pi * len(TFS) + list(TFS).index(tf))
                        / (len(BOOK) * len(TFS)),
                        1,
                    ),
                    "pair": sym,
                    "tf": tf,
                    "cells_done": cells_done,
                    "cells_est": est,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "modes": len(variants),
                    "round": resume_tag,
                }
            )
            try:
                df = _load(sym, tf, years)
            except Exception as exc:
                _log(f"  LOAD FAIL: {exc}")
                per_pair[sym]["by_tf"][tf] = {"error": str(exc), "done": True}
                _save_partial(
                    {
                        "resume_tag": resume_tag,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "elapsed_sec": round(time.time() - t0, 1),
                        "cells_done": cells_done,
                        "per_pair": per_pair,
                        "hunt": "aud_eth_invent",
                        "densify": DENSIFY,
                    }
                )
                continue

            cal = _calendar(df)
            static = _sides_snr(df, 0.40)
            regime = _classify_regime(df)
            ctx = {
                "sym": sym,
                "cal": cal,
                "static": static,
                "regime": regime,
                "v2_preferred_sides": static,
                "family": "snr",
            }
            live_sl = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))
            sl_grid = SL_GRID_ETH_V2[tf] if sym == "ETHUSD" else SL_GRID_FX_V2[tf]
            gap_grid = GAP_GRID_V2[tf]

            mode_ids: list[tuple[str, str, Any]] = []
            for vid, vlab, vfn in variants:
                try:
                    mode_ids.append((vid, vlab, vfn(df, ctx)))
                except Exception as exc:
                    _log(f"  sides {vid} fail: {exc}")

            results: list[dict] = []
            for mi, (mode, _desc, sides) in enumerate(mode_ids):
                try:
                    sides_arr = np.asarray(sides, dtype=int)
                except Exception as exc:
                    _log(f"  skip {mode}: {exc}")
                    continue
                if int((sides_arr != 0).sum()) < 20:
                    continue
                for sl_atr in sl_grid:
                    for gap in gap_grid:
                        entries = _entries(
                            df, sides_arr, sym, sl_atr=float(sl_atr), min_gap=int(gap)
                        )
                        for rr in RR_GRID_V2:
                            trades = _simulate(df, entries, float(rr), sym)
                            wins = {}
                            for wname, days in WINDOWS:
                                tw = (
                                    trades
                                    if days is None
                                    else _window_trades(df, trades, days)
                                )
                                wins[wname] = _summarize(tw)
                            cell = {
                                "id": f"{mode}|sl{sl_atr}|rr{rr}|g{gap}",
                                "mode": mode,
                                "sl_atr": sl_atr,
                                "rr": rr,
                                "min_gap": gap,
                                "tf": tf,
                                "windows": wins,
                                "n_entries": len(entries),
                            }
                            cell["_score"] = _cell_score(wins)
                            results.append(cell)
                            cells_done += 1
                if (mi + 1) % 6 == 0 or mi == 0:
                    _log(
                        f"  … {mi+1}/{len(mode_ids)} modes · cells={cells_done} · "
                        f"{time.time()-t0:.0f}s · {mode}"
                    )
                    _progress(
                        {
                            "status": "running",
                            "pair": sym,
                            "tf": tf,
                            "mode_i": mi + 1,
                            "modes": len(mode_ids),
                            "mode_id": mode,
                            "mode_family": str(mode).split("_")[0],
                            "cells_done": cells_done,
                            "cells_est": est,
                            "elapsed_sec": round(time.time() - t0, 1),
                            "pct": round(
                                100
                                * (
                                    pi * len(TFS)
                                    + list(TFS).index(tf)
                                    + (mi + 1) / max(len(mode_ids), 1)
                                )
                                / (len(BOOK) * len(TFS)),
                                1,
                            ),
                            "round": resume_tag,
                        }
                    )

            results.sort(key=lambda c: c["_score"], reverse=True)
            top = [_top_row(c) for c in results[:50]]
            best = top[0] if top else None
            wireables = [t for t in top if t.get("wireable")]
            per_pair[sym]["by_tf"][tf] = {
                "done": True,
                "n_bars": len(df),
                "span": [str(df.index[0]), str(df.index[-1])],
                "n_combos": len(results),
                "best": best,
                "top25": top[:25],
                "wireable": wireables[:15],
                "live_sl_ref": live_sl,
            }
            if best and _better(best, per_pair[sym].get("best_overall")):
                per_pair[sym]["best_overall"] = {**best, "tf": tf}
            if wireables:
                w0 = wireables[0]
                if _better(w0, per_pair[sym].get("best_wireable")):
                    per_pair[sym]["best_wireable"] = {**w0, "tf": tf}

            _log(
                f"  BEST {tf}: {best['id'] if best else None} "
                f"30d={best.get('verdict_30d') if best else None} "
                f"15d={best.get('verdict_15d') if best else None} "
                f"ret={best.get('return_1y') if best else None} "
                f"pnl1y=${best.get('pnl_1y') if best else 0:,.0f} "
                f"wireable_in_top={len(wireables)}"
            )
            _save_partial(
                {
                    "resume_tag": resume_tag,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "cells_done": cells_done,
                    "per_pair": per_pair,
                    "hunt": "aud_eth_invent",
                    "densify": DENSIFY,
                    "min_pnl_1y": MIN_PNL_1Y,
                }
            )

    recs = []
    both_ok = True
    for sym in BOOK:
        row = per_pair.get(sym) or {}
        pick = row.get("best_wireable") or row.get("best_overall")
        if pick:
            recs.append({"symbol": sym, **pick})
        if not (pick and pick.get("wireable")):
            both_ok = False

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": f"aud_eth_invent_v2_d{DENSIFY}",
        "mode": "math_econ_invent_v2",
        "research_only": True,
        "recommend_live": False,
        "criteria_met": both_ok,
        "min_return": MIN_RETURN,
        "account_usd": ACCOUNT_USD,
        "min_pnl_1y": MIN_PNL_1Y,
        "densify": DENSIFY,
        "elapsed_sec": round(time.time() - t0, 1),
        "cells_done": cells_done,
        "tfs": list(TFS),
        "book": list(BOOK),
        "live_untouched": True,
        "xm_spreads": True,
        "per_pair": per_pair,
        "recommendations": recs,
        "wireable_count": sum(1 for r in recs if r.get("wireable")),
        "note": (
            f"v2 densify={DENSIFY}. Need ≥{MIN_RETURN:.0%} return "
            f"(${MIN_PNL_1Y:.0f}/y) + short verdicts. Live untouched."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    ROUND_TAG.write_text(
        json.dumps(
            {
                "densify": DENSIFY,
                "criteria_met": both_ok,
                "wireable_count": payload["wireable_count"],
                "finished_at": payload["generated_at"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _progress(
        {
            "status": "done" if both_ok else "done_need_densify",
            "pct": 100,
            "cells_done": cells_done,
            "cells_est": est,
            "elapsed_sec": round(time.time() - t0, 1),
            "wireable_count": payload["wireable_count"],
            "criteria_met": both_ok,
            "round": resume_tag,
            "next_densify": None if both_ok else DENSIFY + 1,
        }
    )
    _log(
        f"\n=== V2 DONE densify={DENSIFY} · cells={cells_done} · "
        f"{time.time()-t0:.0f}s · wireable={payload['wireable_count']}/2 · "
        f"criteria_met={both_ok} ==="
    )
    for r in recs:
        _log(
            f"  {r['symbol']}: {r.get('id')} tf={r.get('tf')} "
            f"wire={r.get('wireable')} ret={r.get('return_1y')} "
            f"30d={r.get('verdict_30d')} 15d={r.get('verdict_15d')} "
            f"1y={r.get('verdict_1y')} pnl=${r.get('pnl_1y')}"
        )
    return 0 if both_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
