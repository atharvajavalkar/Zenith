#!/usr/bin/env python3
"""AUDUSD + ETHUSD invent hunt — math/econ equations + adaptive routers.

Research-only. XM session spreads ON. Short-verdict-first scoring.
Does NOT wire live / touch .env.

Outputs:
  data/lab_adaptive_xm_aud_eth_invent_hunt.{json,partial.json,progress.json,log}
"""
from __future__ import annotations

import json
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
from analysis.lab_adaptive_regime_router_bt import _window_trades  # noqa: E402
from analysis.lab_adaptive_xm_deep_hunt_bt import (  # noqa: E402
    GAP_GRID_BY_TF,
    RR_GRID,
    SL_GRID_BY_TF,
    TF_YEARS,
    TFS,
    WINDOWS,
    _entries,
    _load,
    _simulate,
    _summarize,
)
from analysis.lab_adaptive_xm_rest5_hunt_bt import FX_SYMS, SL_GRID_FX  # noqa: E402
from analysis.lab_aud_eth_math_econ_invent import (  # noqa: E402
    build_aud_eth_catalog,
    equations_md,
)
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL  # noqa: E402

BOOK = ("AUDUSD", "ETHUSD")

OUT = Path(ROOT) / "data" / "lab_adaptive_xm_aud_eth_invent_hunt.json"
LOG = Path(ROOT) / "data" / "lab_adaptive_xm_aud_eth_invent_hunt.log"
PROGRESS = Path(ROOT) / "data" / "lab_adaptive_xm_aud_eth_invent_hunt.progress.json"
PARTIAL = Path(ROOT) / "data" / "lab_adaptive_xm_aud_eth_invent_hunt.partial.json"
EQ_MD = Path(ROOT) / "data" / "lab_adaptive_xm_aud_eth_invent_equations.md"


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    p["hunt"] = "aud_eth_invent"
    p["book"] = list(BOOK)
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _save_partial(payload: dict) -> None:
    PARTIAL.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _n_green(wins: dict[str, dict]) -> int:
    n = 0
    for k in ("7d", "15d", "30d", "90d", "180d", "1y"):
        w = wins.get(k) or {}
        try:
            if float(w.get("pnl") or 0) > 0:
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
    """Short-verdict-first (30d/15d/7d), then n_green, then 1y quality."""

    def _q(wname: str, w: float) -> float:
        sm = wins.get(wname) or {}
        return w * (
            _vrank(sm.get("verdict")) * 1000.0
            + float(sm.get("pnl") or 0) * 0.01
            + float(sm.get("exp_r") or 0) * 50.0
            + float(sm.get("oos_exp_r") or 0) * 40.0
            + float(sm.get("wf_green") or 0) * 20.0
        )

    ng = _n_green(wins)
    return (
        _q("30d", 100.0)
        + _q("15d", 60.0)
        + _q("7d", 40.0)
        + ng * 5_000.0
        + _q("1y", 1.0)
        + _q("90d", 0.4)
    )


def _wireable(row: dict) -> bool:
    return (
        row.get("verdict_30d") in ("PASS", "SOFT")
        and row.get("verdict_15d") in ("PASS", "SOFT")
        and row.get("verdict_1y") != "FAIL"
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
    }
    row["wireable"] = _wireable(row)
    row["_score"] = c.get("_score")
    return row


def _better(cand: dict, cur: dict | None) -> bool:
    if cur is None:
        return True
    # prefer wireable, then short verdicts, then score/pnl
    cw = 1 if cand.get("wireable") else 0
    ow = 1 if cur.get("wireable") else 0
    if cw != ow:
        return cw > ow
    for k in ("verdict_30d", "verdict_15d", "verdict_7d"):
        if _vrank(cand.get(k)) != _vrank(cur.get(k)):
            return _vrank(cand.get(k)) > _vrank(cur.get(k))
    if (cand.get("n_green") or 0) != (cur.get("n_green") or 0):
        return (cand.get("n_green") or 0) > (cur.get("n_green") or 0)
    return float(cand.get("pnl_1y") or 0) > float(cur.get("pnl_1y") or 0)


def main() -> int:
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    EQ_MD.write_text(equations_md(), encoding="utf-8")
    t0 = time.time()

    done_keys: set[str] = set()
    per_pair: dict[str, Any] = {}
    if PARTIAL.exists():
        try:
            prev = json.loads(PARTIAL.read_text(encoding="utf-8"))
            per_pair = prev.get("per_pair") or {}
            for sym, row in per_pair.items():
                for tf, block in (row.get("by_tf") or {}).items():
                    if block.get("done"):
                        done_keys.add(f"{sym}:{tf}")
            _log(f"RESUME · {len(done_keys)} pair×tf already done")
        except Exception:
            pass

    # estimate cells
    est = 0
    for tf in TFS:
        for sym in BOOK:
            nv = len(build_aud_eth_catalog(lean=(tf == "1m"), symbol=sym))
            slg = SL_GRID_FX[tf] if sym in FX_SYMS else SL_GRID_BY_TF[tf]
            # ETH uses crypto SL grid
            if sym == "ETHUSD":
                slg = SL_GRID_BY_TF[tf]
            est += nv * len(slg) * len(RR_GRID) * len(GAP_GRID_BY_TF[tf])

    _log(
        "=== AUD/ETH math–econ INVENT hunt (Nebula) ===\n"
        f"book={BOOK}\n"
        "live untouched: GOLD BTCUSD UKOIL NAS100 + all .env brains\n"
        "layers: pair econ equations + dynamic Hurst/vol/Bayes routers\n"
        f"est_cells≈{est:,} · score=short-verdict-first · XM spreads ON\n"
    )
    _progress({"status": "running", "pct": 0, "cells_est": est})

    cells_done = 0
    for pi, sym in enumerate(BOOK):
        per_pair.setdefault(
            sym,
            {
                "symbol": sym,
                "family": "math_econ_invent",
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
            variants = build_aud_eth_catalog(lean=lean, symbol=sym)
            _log(f"\n######## INVENT {sym} @ {tf} · {years}y · {len(variants)} modes ########")
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
                }
            )
            try:
                df = _load(sym, tf, years)
            except Exception as exc:
                _log(f"  LOAD FAIL: {exc}")
                per_pair[sym]["by_tf"][tf] = {"error": str(exc), "done": True}
                _save_partial(
                    {
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "elapsed_sec": round(time.time() - t0, 1),
                        "cells_done": cells_done,
                        "per_pair": per_pair,
                        "hunt": "aud_eth_invent",
                    }
                )
                continue

            cal = _calendar(df)
            ctx = {"sym": sym, "cal": cal}
            live_sl = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))
            sl_grid = SL_GRID_FX[tf] if sym in FX_SYMS else SL_GRID_BY_TF[tf]
            if sym == "ETHUSD":
                sl_grid = SL_GRID_BY_TF[tf]
            gap_grid = GAP_GRID_BY_TF[tf]

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
                        for rr in RR_GRID:
                            trades = _simulate(df, entries, float(rr), sym)
                            wins = {}
                            for wname, days in WINDOWS:
                                if days is None:
                                    tw = trades
                                else:
                                    tw = _window_trades(df, trades, days)
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
                if (mi + 1) % 4 == 0 or mi == 0:
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
                            "mode_family": (
                                "adaptive"
                                if str(mode).startswith("dyn_")
                                else ("aud_econ" if mode.startswith("aud_") else "eth_econ")
                            ),
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
                        }
                    )

            results.sort(key=lambda c: c["_score"], reverse=True)
            top = [_top_row(c) for c in results[:40]]
            best = top[0] if top else None
            wireables = [t for t in top if t.get("wireable")]
            all_six = [t for t in top if t.get("all_six")]
            per_pair[sym]["by_tf"][tf] = {
                "done": True,
                "n_bars": len(df),
                "span": [str(df.index[0]), str(df.index[-1])],
                "n_combos": len(results),
                "best": best,
                "top25": top[:25],
                "wireable": wireables[:10],
                "all_six": all_six[:10],
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
                f"green={best.get('n_green') if best else 0}/6 "
                f"wireable_in_top={len(wireables)} "
                f"1y ${best['pnl_1y'] if best else 0:,.0f} spR={best.get('spR_1y') if best else 0}"
            )
            _save_partial(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "cells_done": cells_done,
                    "per_pair": per_pair,
                    "hunt": "aud_eth_invent",
                }
            )

    # recommendations
    recs = []
    for sym in BOOK:
        row = per_pair.get(sym) or {}
        pick = row.get("best_wireable") or row.get("best_overall")
        if pick:
            recs.append({"symbol": sym, **pick})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "aud_eth_invent_v1",
        "mode": "math_econ_invent",
        "research_only": True,
        "recommend_live": False,
        "elapsed_sec": round(time.time() - t0, 1),
        "cells_done": cells_done,
        "tfs": list(TFS),
        "book": list(BOOK),
        "live_untouched": True,
        "score": "short_verdict_first",
        "xm_spreads": True,
        "per_pair": per_pair,
        "recommendations": recs,
        "wireable_count": sum(1 for r in recs if r.get("wireable")),
        "note": (
            "AUD/ETH invented math+econ equations + adaptive routers. "
            "XM session spreads on. Short-verdict-first. Live desk untouched."
        ),
        "equations_path": str(EQ_MD.name),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress(
        {
            "status": "done",
            "pct": 100,
            "cells_done": cells_done,
            "cells_est": est,
            "elapsed_sec": round(time.time() - t0, 1),
            "wireable_count": payload["wireable_count"],
        }
    )
    _log(
        f"\n=== DONE · {cells_done} cells · {time.time()-t0:.0f}s · "
        f"wireable={payload['wireable_count']} → {OUT.name} ==="
    )
    for r in recs:
        _log(
            f"  {r['symbol']}: {r.get('id')} tf={r.get('tf')} "
            f"wireable={r.get('wireable')} "
            f"30d={r.get('verdict_30d')} 15d={r.get('verdict_15d')} 1y={r.get('verdict_1y')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
