#!/usr/bin/env python3
"""XM-cost fine-tune v2 — fix WF indexing + hunt survivable configs.

Grid per pair (live TF):
  brains × risk_spread_mult × max_spread_r × off_sitout
Score: 1y/3y windows + OOS70 + WF×4 (absolute bar indices) + MC
Picks best PASS/SOFT under real XM session spreads.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import _prep, monte_carlo  # noqa: E402
from analysis.lab_zenith_math_invention_deep_bt import (  # noqa: E402
    _hurst_rs,
    _perm_entropy,
)
from analysis.xm_cost_wired_battery import (  # noqa: E402
    BOOK,
    RISK_USD,
    WINDOWS,
    _apply_night_sitout,
    _calendar,
    _enrich_light,
    _entries_live_sl,
    _load,
    _sides_for_brain,
    _summarize,
    _window_trades,
)
from analysis.xm_session_costs import session_name, spread_price  # noqa: E402
from config import symbol_entry_brain, symbol_rr, symbol_scan_tfs  # noqa: E402
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL, _min_sl_px  # noqa: E402

OUT = Path(ROOT) / "data" / "xm_cost_finetune_v2.json"
LOG = Path(ROOT) / "data" / "xm_cost_finetune_v2.log"
PROGRESS = Path(ROOT) / "data" / "xm_cost_finetune_v2.progress.json"

BRAINS = ("session_mix", "circadian_kalman_z", "entropy_tod", "volmom")
RISK_MULTS = (1.0, 5.0, 8.0, 10.0)  # min risk = max(live_sl, mult × XM spread)
MAX_SPR = (None, 0.12, 0.18)  # skip if spread_r would exceed
OFF_SIT = (False, True)


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _entries_spread_aware(
    df: pd.DataFrame,
    sides: np.ndarray,
    sym: str,
    *,
    risk_mult: float,
    min_gap: int = 3,
) -> list[dict]:
    """Live ATR/floor SL, then raise risk to ≥ risk_mult × session spread."""
    atr = df["atr"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    hours = pd.DatetimeIndex(df.index).hour.to_numpy(dtype=float)
    sl_atr = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))
    floor = float(_min_sl_px(sym))
    n = len(df)
    out: list[dict] = []
    busy = -1
    for i in range(40, n - 2):
        if i < busy:
            continue
        s = int(sides[i])
        if s == 0:
            continue
        a = float(atr[i] or 0)
        if a <= 0 or not np.isfinite(a):
            continue
        entry = float(c[i])
        sp = spread_price(sym, hour_utc=float(hours[i]))
        risk = max(sl_atr * a, floor, float(risk_mult) * sp)
        if risk <= 0:
            continue
        if s > 0:
            sl, side = entry - risk, "BUY"
        else:
            sl, side = entry + risk, "SELL"
        out.append(
            {
                "ei": i,
                "side": side,
                "entry": entry,
                "sl": sl,
                "risk": risk,
                "spread_px": sp,
            }
        )
        busy = i + min_gap
    return out


def _simulate(
    df: pd.DataFrame,
    entries: list[dict],
    rr: float,
    sym: str,
    *,
    max_spread_r: Optional[float],
    off_sit: bool,
) -> list[dict]:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    hours = pd.DatetimeIndex(df.index).hour.to_numpy(dtype=float)
    n = len(df)
    trades: list[dict] = []
    busy = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < busy or ei >= n - 1:
            continue
        hour = float(hours[ei])
        sess = session_name(hour)
        if off_sit and sess == "Off" and sym in (
            "AUDUSD",
            "USDCHF",
            "USDJPY",
            "UKOIL",
            "GOLD",
            "ETHUSD",
            "BTCUSD",
        ):
            continue
        risk = float(e["risk"])
        if risk <= 0:
            continue
        sp = float(e.get("spread_px") or spread_price(sym, hour_utc=hour))
        spread_r = sp / risk
        if max_spread_r is not None and spread_r > float(max_spread_r):
            continue
        side = e["side"]
        entry = float(e["entry"])
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk
        exit_r = None
        exit_i = ei
        if side == "BUY":
            for j in range(ei + 1, n):
                if low[j] <= sl or high[j] >= tp:
                    exit_r = -1.0 if low[j] <= sl else float(rr)
                    exit_i = j
                    break
        else:
            for j in range(ei + 1, n):
                if high[j] >= sl or low[j] <= tp:
                    exit_r = -1.0 if high[j] >= sl else float(rr)
                    exit_i = j
                    break
        if exit_r is None:
            continue
        r_net = float(exit_r) - float(spread_r)
        trades.append(
            {
                "i": ei,
                "exit_i": exit_i,
                "side": side,
                "session": sess,
                "r_gross": round(float(exit_r), 4),
                "spread_r": round(float(spread_r), 4),
                "r": round(r_net, 4),
                "pnl": round(r_net * RISK_USD, 2),
            }
        )
        busy = exit_i + 2
    return trades


def _i0_for_days(df: pd.DataFrame, days: int) -> int:
    end = df.index[-1]
    start = end - pd.Timedelta(days=days)
    return int(np.searchsorted(np.asarray(df.index), start, side="left"))


def _wf_folds(trades: list[dict], i0: int, i1: int) -> list[dict]:
    """Anchored WF folds inside [i0, i1) using absolute trade indices."""
    n = max(i1 - i0, 1)
    folds = []
    for k in range(4):
        a = i0 + int(n * (0.40 + 0.15 * k))
        b = i0 + int(n * (0.55 + 0.15 * k)) - 1
        if b <= a:
            continue
        sub = [t for t in trades if a <= int(t["i"]) <= b]
        folds.append({"fold": k + 1, "a": a, "b": b, **_summarize(sub)})
    return folds


def _battery(trades: list[dict], i0: int, i1: int) -> dict:
    if not trades:
        return {"verdict": "NO_TRADES", "wf_green": 0, "oos": {}, "wf": [], "mc": {}}
    n = max(i1 - i0, 1)
    split = i0 + int(n * 0.70)
    oos = _summarize([t for t in trades if int(t["i"]) >= split])
    wf = _wf_folds(trades, i0, i1)
    wf_green = sum(1 for f in wf if float(f.get("exp_r") or 0) > 0 and int(f.get("n") or 0) >= 5)
    full = _summarize(trades)
    mc = {}
    try:
        mc_raw = monte_carlo(
            [{"r": float(t["r"]), "pnl": float(t["pnl"])} for t in trades]
        )
        mc = {
            "prob_profit": (mc_raw or {}).get("prob_profit"),
            "p05_pnl": (mc_raw or {}).get("p05_pnl"),
            "p50_pnl": (mc_raw or {}).get("p50_pnl"),
        }
    except Exception:
        pass
    er = float(full.get("exp_r") or 0)
    oos_er = float(oos.get("exp_r") or 0)
    ntr = int(full.get("n") or 0)
    ok = er > 0 and oos_er > 0 and wf_green >= 3 and ntr >= 20
    soft = er > 0 and wf_green >= 2 and ntr >= 12
    verd = "PASS" if ok else ("SOFT" if soft else "FAIL")
    return {
        "verdict": verd,
        "wf_green": wf_green,
        "full": full,
        "oos": oos,
        "wf": wf,
        "mc": mc,
    }


def _score(cell: dict) -> tuple:
    """Rank: PASS > SOFT > FAIL, then 1y pnl, 1y E[R], 3y E[R], n."""
    v = cell["battery_1y"]["verdict"]
    rank = 2 if v == "PASS" else (1 if v == "SOFT" else 0)
    return (
        rank,
        float(cell["windows"]["1y"]["pnl"]),
        float(cell["windows"]["1y"]["exp_r"]),
        float(cell["windows"]["3y"]["exp_r"]),
        int(cell["windows"]["1y"]["n"]),
    )


def main() -> int:
    t0 = time.time()
    LOG.write_text("", encoding="utf-8")
    _log(
        f"=== XM fine-tune v2 · ${RISK_USD}/R · brains={BRAINS} · "
        f"risk_mult={RISK_MULTS} · max_spr={MAX_SPR} · off={OFF_SIT} ==="
    )
    _progress({"status": "running", "pct": 0})

    per_pair: dict[str, Any] = {}
    desk_live = {w[0]: 0.0 for w in WINDOWS}
    desk_best = {w[0]: 0.0 for w in WINDOWS}
    recs = []

    for pi, sym in enumerate(BOOK):
        live_brain = symbol_entry_brain(sym)
        rr = float(symbol_rr(sym) or 2)
        tfs = symbol_scan_tfs(sym) or ["5m"]
        tf = tfs[0]
        _log(f"\n[{pi+1}/{len(BOOK)}] {sym} live={live_brain} tf={tf} RR={rr}")
        _progress(
            {
                "status": "running",
                "pct": round(100 * pi / len(BOOK), 1),
                "pair": sym,
                "elapsed_sec": round(time.time() - t0, 1),
            }
        )
        try:
            df = _load(sym, tf)
        except Exception as exc:
            _log(f"  LOAD FAIL {exc}")
            per_pair[sym] = {"error": str(exc)}
            continue
        cal = _calendar(df)
        i0_1y = _i0_for_days(df, 365)
        i0_3y = 0
        i1 = len(df)

        # precompute sides per brain
        sides_map: dict[str, np.ndarray] = {}
        for brain in BRAINS:
            try:
                sides_map[brain] = _sides_for_brain(sym, brain, df, cal)
            except Exception as exc:
                _log(f"  sides {brain} fail: {exc}")

        cells = []
        total = len(BRAINS) * len(RISK_MULTS) * len(MAX_SPR) * len(OFF_SIT)
        done = 0
        for brain, rmult, max_spr, off in product(BRAINS, RISK_MULTS, MAX_SPR, OFF_SIT):
            done += 1
            if brain not in sides_map:
                continue
            # volmom on 5m for FX is OK to test; skip absurd off×crypto noise lightly
            entries = _entries_spread_aware(
                df, sides_map[brain], sym, risk_mult=float(rmult)
            )
            trades = _simulate(
                df,
                entries,
                rr,
                sym,
                max_spread_r=max_spr,
                off_sit=bool(off),
            )
            wins = {
                wname: _summarize(_window_trades(df, trades, days))
                for wname, days in WINDOWS
            }
            t_1y = _window_trades(df, trades, 365)
            t_3y = trades
            bat_1y = _battery(t_1y, i0_1y, i1)
            bat_3y = _battery(t_3y, i0_3y, i1)
            cell = {
                "brain": brain,
                "risk_mult": rmult,
                "max_spread_r": max_spr,
                "off_sitout": off,
                "n_entries": len(entries),
                "windows": wins,
                "battery_1y": bat_1y,
                "battery_3y": bat_3y,
                "is_live_brain": brain == live_brain,
            }
            cells.append(cell)
            if done % 24 == 0:
                _log(f"  … {done}/{total} configs")

        cells.sort(key=_score, reverse=True)
        best = cells[0]
        # best with live brain only
        live_cells = [c for c in cells if c["is_live_brain"]]
        live_cells.sort(key=_score, reverse=True)
        best_live = live_cells[0] if live_cells else best

        # baseline: live brain, mult=1, no max, no off
        baseline = next(
            (
                c
                for c in cells
                if c["is_live_brain"]
                and c["risk_mult"] == 1.0
                and c["max_spread_r"] is None
                and c["off_sitout"] is False
            ),
            best_live,
        )

        for wname, _ in WINDOWS:
            desk_live[wname] = round(
                desk_live[wname] + float(baseline["windows"][wname]["pnl"]), 2
            )
            desk_best[wname] = round(
                desk_best[wname] + float(best["windows"][wname]["pnl"]), 2
            )

        top5 = [
            {
                "brain": c["brain"],
                "risk_mult": c["risk_mult"],
                "max_spread_r": c["max_spread_r"],
                "off_sitout": c["off_sitout"],
                "verdict_1y": c["battery_1y"]["verdict"],
                "wf_1y": c["battery_1y"]["wf_green"],
                "pnl_1y": c["windows"]["1y"]["pnl"],
                "exp_r_1y": c["windows"]["1y"]["exp_r"],
                "n_1y": c["windows"]["1y"]["n"],
                "avg_spr_1y": c["windows"]["1y"]["avg_spread_r"],
            }
            for c in cells[:5]
        ]
        rec = {
            "symbol": sym,
            "live_brain": live_brain,
            "tf": tf,
            "rr": rr,
            "baseline_1y": {
                "pnl": baseline["windows"]["1y"]["pnl"],
                "exp_r": baseline["windows"]["1y"]["exp_r"],
                "verdict": baseline["battery_1y"]["verdict"],
                "wf": baseline["battery_1y"]["wf_green"],
            },
            "best": top5[0],
            "best_live_settings": {
                "brain": best_live["brain"],
                "risk_mult": best_live["risk_mult"],
                "max_spread_r": best_live["max_spread_r"],
                "off_sitout": best_live["off_sitout"],
                "verdict_1y": best_live["battery_1y"]["verdict"],
                "wf_1y": best_live["battery_1y"]["wf_green"],
                "pnl_1y": best_live["windows"]["1y"]["pnl"],
                "exp_r_1y": best_live["windows"]["1y"]["exp_r"],
                "n_1y": best_live["windows"]["1y"]["n"],
            },
            "top5": top5,
        }
        recs.append(rec)
        per_pair[sym] = {
            **rec,
            "n_bars": len(df),
            "n_configs": len(cells),
            "pass_count": sum(
                1 for c in cells if c["battery_1y"]["verdict"] == "PASS"
            ),
            "soft_count": sum(
                1 for c in cells if c["battery_1y"]["verdict"] == "SOFT"
            ),
        }
        b = top5[0]
        _log(
            f"  BASE 1y ${baseline['windows']['1y']['pnl']:,.0f} "
            f"{baseline['battery_1y']['verdict']} wf={baseline['battery_1y']['wf_green']}/4"
        )
        _log(
            f"  BEST  {b['brain']} mult={b['risk_mult']} maxSp={b['max_spread_r']} "
            f"off={b['off_sitout']} → {b['verdict_1y']} wf={b['wf_1y']}/4 "
            f"1y ${b['pnl_1y']:,.0f} E[R]={b['exp_r_1y']:+.3f} n={b['n_1y']} "
            f"spR={b['avg_spr_1y']:.3f}"
        )
        bl = rec["best_live_settings"]
        _log(
            f"  LIVE+ {bl['brain']} mult={bl['risk_mult']} maxSp={bl['max_spread_r']} "
            f"off={bl['off_sitout']} → {bl['verdict_1y']} wf={bl['wf_1y']}/4 "
            f"1y ${bl['pnl_1y']:,.0f} E[R]={bl['exp_r_1y']:+.3f}"
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "risk_usd": RISK_USD,
        "grid": {
            "brains": list(BRAINS),
            "risk_mults": list(RISK_MULTS),
            "max_spread_r": list(MAX_SPR),
            "off_sitout": list(OFF_SIT),
        },
        "elapsed_sec": round(time.time() - t0, 1),
        "desk_baseline_live": desk_live,
        "desk_best_any": desk_best,
        "recommendations": recs,
        "per_pair": per_pair,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress(
        {
            "status": "done",
            "pct": 100,
            "elapsed_sec": round(time.time() - t0, 1),
            "desk_base_1y": desk_live["1y"],
            "desk_best_1y": desk_best["1y"],
            "out": str(OUT),
        }
    )
    _log("\n=== DESK ===")
    _log(
        "BASE live: "
        + " · ".join(f"{w} ${desk_live[w]:,.0f}" for w, _ in WINDOWS)
    )
    _log(
        "BEST any:  "
        + " · ".join(f"{w} ${desk_best[w]:,.0f}" for w, _ in WINDOWS)
    )
    _log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
