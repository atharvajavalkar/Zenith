#!/usr/bin/env python3
"""Full-desk MEGA invent hunt — 400–600+ strategies × all pairs.

Book: GOLD BTCUSD UKOIL NAS100 AUDUSD ETHUSD USDJPY USDCHF
Criteria: ≥15% return ($1500/y on $10k) + 30d/15d PASS|SOFT + 1y≠FAIL
XM session spreads ON. Research-only — does NOT wire live / .env.

Outputs:
  data/lab_mega_desk_invent_hunt.{json,partial.json,progress.json,log}
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
from analysis.lab_mega_desk_invent_catalog import BOOK, build_mega_catalog  # noqa: E402
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_mega_desk_invent_hunt.json"
LOG = Path(ROOT) / "data" / "lab_mega_desk_invent_hunt.log"
PROGRESS = Path(ROOT) / "data" / "lab_mega_desk_invent_hunt.progress.json"
PARTIAL = Path(ROOT) / "data" / "lab_mega_desk_invent_hunt.partial.json"

ACCOUNT_USD = float(os.getenv("MEGA_ACCOUNT_USD", "10000"))
MIN_RETURN = float(os.getenv("MEGA_MIN_RETURN", "0.15"))
MIN_PNL_1Y = ACCOUNT_USD * MIN_RETURN

# Intraday first; scalp last
TFS = ("15m", "5m", "3m", "1m")

FX = frozenset({"AUDUSD", "USDCHF", "USDJPY"})
CRYPTO = frozenset({"BTCUSD", "ETHUSD"})

RR_GRID = (2.0, 3.0, 4.0)
SL_FX = {
    "1m": (2.5, 3.0),
    "3m": (2.5, 3.0, 3.5),
    "5m": (2.0, 2.5, 3.0, 3.5),
    "15m": (2.0, 2.5, 3.0, 3.5),
}
SL_CRYPTO = {
    "1m": (2.0, 2.5),
    "3m": (2.0, 2.5, 3.0),
    "5m": (2.0, 2.5, 3.0),
    "15m": (2.0, 2.5, 3.0),
}
SL_DEFAULT = {
    "1m": (2.0, 2.5),
    "3m": (2.0, 2.5, 3.0),
    "5m": (1.5, 2.0, 2.5, 3.0),
    "15m": (1.5, 2.0, 2.5, 3.0),
}
GAP = {
    "1m": (3, 5),
    "3m": (2, 3),
    "5m": (2, 3, 5),
    "15m": (2, 3),
}


def _sl_grid(sym: str, tf: str):
    if sym in FX:
        return SL_FX[tf]
    if sym in CRYPTO:
        return SL_CRYPTO[tf]
    return SL_DEFAULT[tf]


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    p["hunt"] = "mega_desk_invent"
    p["book"] = list(BOOK)
    p["min_pnl_1y"] = MIN_PNL_1Y
    p["min_return"] = MIN_RETURN
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _save_partial(payload: dict) -> None:
    PARTIAL.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _n_green(wins: dict) -> int:
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


def _cell_score(wins: dict) -> float:
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
    if int(bool(cand.get("wireable"))) != int(bool(cur.get("wireable"))):
        return bool(cand.get("wireable"))
    for k in ("verdict_30d", "verdict_15d", "verdict_7d"):
        if _vrank(cand.get(k)) != _vrank(cur.get(k)):
            return _vrank(cand.get(k)) > _vrank(cur.get(k))
    return float(cand.get("pnl_1y") or 0) > float(cur.get("pnl_1y") or 0)


def main() -> int:
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    with LOG.open("a") as f:
        f.write(f"\n===== MEGA DESK INVENT @ {datetime.now(timezone.utc).isoformat()} =====\n")
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
            _log(f"RESUME · {len(done_keys)} pair×tf done")
        except Exception:
            pass

    n_full = len(build_mega_catalog(lean=False, symbol="GOLD"))
    n_lean = len(build_mega_catalog(lean=True, symbol="GOLD"))
    est = 0
    for tf in TFS:
        for sym in BOOK:
            nv = len(build_mega_catalog(lean=(tf == "1m"), symbol=sym))
            est += nv * len(_sl_grid(sym, tf)) * len(RR_GRID) * len(GAP[tf])

    _log(
        "=== MEGA DESK INVENT (Nebula) ===\n"
        f"book={BOOK}\n"
        f"modes≈{n_full} full / {n_lean} lean1m\n"
        f"criteria: ≥{MIN_RETURN:.0%} (${MIN_PNL_1Y:.0f}/y) + 30d/15d PASS|SOFT + 1y≠FAIL\n"
        f"XM spreads ON · research_only · live .env UNTOUCHED\n"
        f"est_cells≈{est:,}\n"
    )
    _progress({"status": "running", "pct": 0, "cells_est": est, "modes": n_full})

    cells_done = 0
    for pi, sym in enumerate(BOOK):
        per_pair.setdefault(
            sym,
            {
                "symbol": sym,
                "family": "mega_invent",
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
            variants = build_mega_catalog(lean=lean, symbol=sym)
            _log(
                f"\n######## MEGA {sym} @ {tf} · {years}y · {len(variants)} modes ########"
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
                        "hunt": "mega_desk_invent",
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
            sl_grid = _sl_grid(sym, tf)
            gap_grid = GAP[tf]
            live_sl = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))

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
                except Exception:
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
                            wins = {
                                wname: _summarize(
                                    trades
                                    if days is None
                                    else _window_trades(df, trades, days)
                                )
                                for wname, days in WINDOWS
                            }
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
                if (mi + 1) % 25 == 0 or mi == 0:
                    _log(
                        f"  … {mi+1}/{len(mode_ids)} · cells={cells_done} · "
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
                        }
                    )

            results.sort(key=lambda c: c["_score"], reverse=True)
            top = [_top_row(c) for c in results[:60]]
            best = top[0] if top else None
            wireables = [t for t in top if t.get("wireable")]
            per_pair[sym]["by_tf"][tf] = {
                "done": True,
                "n_bars": len(df),
                "span": [str(df.index[0]), str(df.index[-1])],
                "n_combos": len(results),
                "best": best,
                "top25": top[:25],
                "wireable": wireables[:20],
                "live_sl_ref": live_sl,
            }
            if best and _better(best, per_pair[sym].get("best_overall")):
                per_pair[sym]["best_overall"] = {**best, "tf": tf}
            if wireables and _better(wireables[0], per_pair[sym].get("best_wireable")):
                per_pair[sym]["best_wireable"] = {**wireables[0], "tf": tf}

            _log(
                f"  BEST {tf}: {best['id'] if best else None} "
                f"ret={best.get('return_1y') if best else None} "
                f"30d={best.get('verdict_30d') if best else None} "
                f"15d={best.get('verdict_15d') if best else None} "
                f"wireable_in_top={len(wireables)}"
            )
            _save_partial(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "cells_done": cells_done,
                    "per_pair": per_pair,
                    "hunt": "mega_desk_invent",
                    "min_pnl_1y": MIN_PNL_1Y,
                }
            )

    recs = []
    n_ok = 0
    for sym in BOOK:
        pick = (per_pair.get(sym) or {}).get("best_wireable") or (
            per_pair.get(sym) or {}
        ).get("best_overall")
        if pick:
            recs.append({"symbol": sym, **pick})
            if pick.get("wireable"):
                n_ok += 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "mega_desk_invent_v1",
        "research_only": True,
        "recommend_live": False,
        "criteria_met": n_ok == len(BOOK),
        "wireable_count": n_ok,
        "min_return": MIN_RETURN,
        "account_usd": ACCOUNT_USD,
        "min_pnl_1y": MIN_PNL_1Y,
        "elapsed_sec": round(time.time() - t0, 1),
        "cells_done": cells_done,
        "tfs": list(TFS),
        "book": list(BOOK),
        "live_untouched": True,
        "xm_spreads": True,
        "n_modes_full": n_full,
        "per_pair": per_pair,
        "recommendations": recs,
        "note": (
            "Mega invent 400–600+ strategies across full desk. "
            "Research-only — no .env / live brain writes."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress(
        {
            "status": "done",
            "pct": 100,
            "cells_done": cells_done,
            "cells_est": est,
            "elapsed_sec": round(time.time() - t0, 1),
            "wireable_count": n_ok,
            "criteria_met": n_ok == len(BOOK),
        }
    )
    _log(f"\n=== MEGA DONE · cells={cells_done} · wireable={n_ok}/{len(BOOK)} ===")
    for r in recs:
        _log(
            f"  {r['symbol']}: wire={r.get('wireable')} ret={r.get('return_1y')} "
            f"30d={r.get('verdict_30d')} 15d={r.get('verdict_15d')} {r.get('id')}"
        )
    return 0 if n_ok == len(BOOK) else 2


if __name__ == "__main__":
    raise SystemExit(main())
