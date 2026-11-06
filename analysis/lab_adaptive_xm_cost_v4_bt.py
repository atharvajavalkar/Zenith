#!/usr/bin/env python3
"""RESEARCH — Adaptive XM-cost v4: re-hunt minute/day brains on Dukascopy WITH XM spreads.

Same catalog / process as lab_adaptive_minute_day_v3_bt.py, but:
  · every trade pays measured XM session spread (analysis/xm_session_costs.py)
  · live-safe SL: max(ATR×mult, pip floor, N×session spread)
  · absolute winner = best 1y under XM costs with WF/OOS gates
  · PA/OB not included

Does NOT auto-wire live (.env stays paused unless user confirms).
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.lab_adaptive_minute_day_v3_bt import (  # noqa: E402
    _calendar,
    _catalog as _v3_catalog,
    eq_circadian_kalman_z,
    eq_entropy_tod,
    eq_session_mix,
)
from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    BOOK,
    TF,
    WINDOWS,
    _classify_regime,
    _load,
    _sides_bimod,
    _sides_kalman,
    _sides_perm,
    _sides_snr,
    _static_sides_for_winner,
    _window_trades,
)
from analysis.lab_adaptive_regime_v2_bt import (  # noqa: E402
    _family,
    _gated_sides,
    _switch_sides,
)
from analysis.xm_session_costs import (  # noqa: E402
    SPREAD_RISK_MULT,
    min_risk_with_spread,
    session_name,
    spread_price,
)
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL, _min_sl_px  # noqa: E402

HUNT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"
V2 = Path(ROOT) / "data" / "lab_adaptive_regime_v2.json"
OUT = Path(ROOT) / "data" / "lab_adaptive_xm_cost_v4.json"
EQ = Path(ROOT) / "data" / "lab_adaptive_xm_cost_v4_equations.md"
LOG = Path(ROOT) / "data" / "lab_adaptive_xm_cost_v4.log"
PROGRESS = Path(ROOT) / "data" / "lab_adaptive_xm_cost_v4.progress.json"

CAP_YEARS = 3.0
RISK_USD = 100.0  # match live MAX_RISK_USD_PER_TRADE
SideFn = Callable[[pd.DataFrame, dict[str, Any]], np.ndarray]


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


# ── Extra XM-aware equations ────────────────────────────────────────────────


def eq_session_mix_no_off(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """session_mix but flat in Off (21–24 UTC) — XM FX/oil spread blowout."""
    sides = eq_session_mix(df, ctx)
    hour = ctx["cal"]["hour"]
    out = sides.copy()
    out[(hour >= 21) | (hour < 0)] = 0
    return out


def eq_circadian_strict(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Live-like higher Kalman |z| thr (2.2 base)."""
    cal = ctx["cal"]
    kz = df["kalman_z"].to_numpy()
    zthr = 2.2 + 0.5 * cal["sin_tod"] + 0.25 * (cal["dow"] / 7.0)
    out = np.zeros(len(df), dtype=int)
    out[kz <= -zthr] = 1
    out[kz >= zthr] = -1
    return out


def eq_entropy_strict(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Live-like entropy ceiling + SNR floor."""
    cal = ctx["cal"]
    pe = df["perm_ent"].to_numpy()
    ds = df["drift_sign"].to_numpy()
    snr = df["snr"].to_numpy()
    pmax = 0.72 + 0.08 * cal["cos_tod"] + 0.03 * cal["sin_week"]
    ok = (pe <= pmax) & (ds != 0) & (snr >= 0.22)
    out = np.zeros(len(df), dtype=int)
    out[ok] = ds[ok].astype(int)
    return out


def eq_liq_sessions_only(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """static only London+Overlap+NY (08–21 UTC)."""
    cal = ctx["cal"]
    s = ctx["static"]
    out = np.zeros(len(df), dtype=int)
    m = (cal["hour"] >= 8) & (cal["hour"] < 21)
    out[m] = s[m]
    return out


def eq_volmom(df: pd.DataFrame, ctx: dict[str, Any]) -> np.ndarray:
    """Vol-scaled momentum zero-cross (NAS lab winner family)."""
    ret = df["close"].pct_change(20)
    vol = df["close"].pct_change().rolling(20).std().replace(0, np.nan)
    score = (ret / vol).fillna(0).to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=int)
    for i in range(40, len(df) - 1):
        a, b = score[i - 1], score[i]
        if a <= 0 < b and b > 0.5:
            out[i] = 1
        elif a >= 0 > b and b < -0.5:
            out[i] = -1
    return out


def _catalog() -> list[tuple[str, str, SideFn]]:
    base = list(_v3_catalog())
    extra = [
        ("session_mix_no_off", "session_mix + flat Off session", eq_session_mix_no_off),
        ("circadian_strict", "Kalman |z| thr live-strict 2.2", eq_circadian_strict),
        ("entropy_strict", "entropy_tod + SNR≥0.22 live-strict", eq_entropy_strict),
        ("liq_sessions_only", "static only 08–21 UTC", eq_liq_sessions_only),
        ("volmom", "Vol-scaled momentum zero-cross", eq_volmom),
    ]
    return base + extra


def _sl_atr_for(sym: str, research_sl: float) -> float:
    """Prefer live SL table; never thinner than research winner."""
    live = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))
    return max(live, float(research_sl or 1.2), 2.0)


def _entries_xm(
    df: pd.DataFrame,
    sides: np.ndarray,
    sym: str,
    *,
    sl_atr: float,
    min_gap: int = 3,
) -> list[dict]:
    atr = df["atr"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    hours = pd.DatetimeIndex(df.index).hour.to_numpy(dtype=float)
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
        hour = float(hours[i])
        atr_risk = float(sl_atr) * a
        risk = min_risk_with_spread(sym, atr_risk, hour_utc=hour, floor_px=floor)
        if risk <= 0:
            continue
        # skip if spread would eat > 0.25R even after widen (soft gate)
        sp = spread_price(sym, hour_utc=hour)
        if sp > 0 and risk < 4.0 * sp:
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
                "hour": hour,
            }
        )
        busy = i + min_gap
    return out


def _trades_xm(df: pd.DataFrame, entries: list[dict], rr: float, sym: str) -> list[dict]:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    trades: list[dict] = []
    busy = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < busy or ei >= n - 1:
            continue
        risk = float(e["risk"])
        if risk <= 0:
            continue
        sp = float(e.get("spread_px") or spread_price(sym, hour_utc=e.get("hour")))
        spread_r = sp / risk
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
        hour = float(e.get("hour") or 0)
        trades.append(
            {
                "i": ei,
                "exit_i": exit_i,
                "side": side,
                "session": session_name(hour),
                "r_gross": round(float(exit_r), 4),
                "spread_r": round(float(spread_r), 4),
                "r": round(r_net, 4),
                "pnl": round(r_net * RISK_USD, 2),
            }
        )
        busy = exit_i + 2
    return trades


def _summarize(trades: list[dict], n_bars: int) -> dict:
    if not trades:
        return {
            "n": 0,
            "exp_r": 0.0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "avg_spread_r": 0.0,
            "wf_green": 0,
            "oos_exp_r": 0.0,
            "mc": None,
        }
    n = len(trades)
    rs = [float(t["r"]) for t in trades]
    pnl = sum(float(t["pnl"]) for t in trades)
    exp_r = sum(rs) / n
    wr = sum(1 for r in rs if r > 0) / n
    avg_sp = sum(float(t.get("spread_r") or 0) for t in trades) / n
    # WF on trade indices relative to bar span of this window
    idxs = [int(t["i"]) for t in trades]
    i0, i1 = min(idxs), max(idxs) + 1
    span = max(i1 - i0, 1)
    wf_green = 0
    for k in range(4):
        a = i0 + int(span * (0.40 + 0.15 * k))
        b = i0 + int(span * (0.55 + 0.15 * k)) - 1
        sub = [t for t in trades if a <= int(t["i"]) <= b]
        if len(sub) >= 5 and (sum(float(t["r"]) for t in sub) / len(sub)) > 0:
            wf_green += 1
    split = i0 + int(span * 0.70)
    oos = [t for t in trades if int(t["i"]) >= split]
    oos_er = (
        sum(float(t["r"]) for t in oos) / len(oos) if oos else 0.0
    )
    return {
        "n": n,
        "exp_r": round(exp_r, 4),
        "win_rate": round(wr, 4),
        "pnl": round(pnl, 2),
        "avg_spread_r": round(avg_sp, 4),
        "wf_green": wf_green,
        "oos_exp_r": round(oos_er, 4),
        "n_bars": n_bars,
    }


def _verdict(sm: dict) -> str:
    n = int(sm.get("n") or 0)
    er = float(sm.get("exp_r") or 0)
    oos = float(sm.get("oos_exp_r") or 0)
    wf = int(sm.get("wf_green") or 0)
    if n >= 20 and er > 0 and oos > 0 and wf >= 3:
        return "PASS"
    if n >= 12 and er > 0 and wf >= 2:
        return "SOFT"
    return "FAIL"


def _rank_key(sm: dict) -> tuple:
    v = _verdict(sm)
    rank = 2 if v == "PASS" else (1 if v == "SOFT" else 0)
    return (
        rank,
        float(sm.get("pnl") or 0),
        float(sm.get("exp_r") or 0),
        float(sm.get("oos_exp_r") or 0),
        int(sm.get("wf_green") or 0),
        int(sm.get("n") or 0),
    )


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    t0 = time.time()
    hunt = json.loads(HUNT.read_text(encoding="utf-8")) if HUNT.exists() else {}
    per_hunt = hunt.get("per_pair") or {}
    v2 = json.loads(V2.read_text(encoding="utf-8")) if V2.exists() else {}
    v2_pref = {
        s: (v2.get("per_pair") or {}).get(s, {}).get("best_mode_1y") or "static"
        for s in BOOK
    }
    formulas = _catalog()
    _log(
        "=== ADAPTIVE XM-COST v4 · Dukascopy + XM session spreads ===\n"
        f"TF={TF} cap={CAP_YEARS}y · risk=${RISK_USD}/R · formulas={len(formulas)}\n"
        f"spread mults={SPREAD_RISK_MULT}\n"
        "PA/OB on hold · LIVE not auto-wired\n"
    )
    _progress({"status": "running", "pct": 1, "message": "start"})

    desk: dict[str, dict[str, float]] = {}
    per_pair: dict[str, Any] = {}
    eq_lines = [
        "# Adaptive XM-cost v4",
        "",
        "Dukascopy hist + measured XM session spreads per trade.",
        "Live SL floors + N×spread min risk. Absolute winner = best 1y under XM costs.",
        "",
    ]

    for pi, sym in enumerate(BOOK):
        _progress(
            {
                "status": "running",
                "pct": int(2 + 95 * pi / len(BOOK)),
                "message": f"v4 [{sym}]",
                "symbol": sym,
                "elapsed_sec": round(time.time() - t0, 1),
            }
        )
        _log(f"\n######## {sym} ########")
        try:
            df = _load(sym, TF, CAP_YEARS)
        except Exception as exc:
            _log(f"  FAIL load: {exc}")
            per_pair[sym] = {"error": str(exc)}
            continue

        w = (per_hunt.get(sym) or {}).get("winner") or {}
        wid = str(w.get("id") or "")
        fam = _family(wid)
        rr = float(w.get("rr") or 2.4)
        # Prefer live SYMBOL_RR when present
        try:
            from config import symbol_rr

            rr = float(symbol_rr(sym) or rr)
        except Exception:
            pass
        research_sl = float(w.get("sl_atr") or 1.2)
        sl = _sl_atr_for(sym, research_sl)

        static, label, _ = _static_sides_for_winner(df, wid)
        regime = _classify_regime(df)
        alt_trend = (
            _sides_bimod(df, 0.55)
            if sym in ("ETHUSD", "BTCUSD")
            else _sides_perm(df, 0.80)
            if sym == "GOLD"
            else _sides_snr(df, 0.40)
        )
        alt_fade = _sides_kalman(df, 2.2)
        v2_sides = {
            "static": static,
            "gated": _gated_sides(static, regime, fam),
            "switch": _switch_sides(static, regime, fam, alt_trend, alt_fade),
        }
        pref_mode = v2_pref.get(sym) or "static"
        v2_sides["v2_preferred"] = v2_sides.get(pref_mode, static)

        cal = _calendar(df)
        ctx: dict[str, Any] = {
            "sym": sym,
            "static": static,
            "regime": regime,
            "cal": cal,
            "v2_preferred_sides": v2_sides["v2_preferred"],
            "family": fam,
        }

        all_sides: dict[str, np.ndarray] = dict(v2_sides)
        for fid, _flabel, fn in formulas:
            try:
                all_sides[fid] = fn(df, ctx)
            except Exception as exc:
                _log(f"  {fid} FAIL: {exc}")
                all_sides[fid] = np.zeros(len(df), dtype=int)

        trades: dict[str, list] = {}
        for mode, sides in all_sides.items():
            ent = _entries_xm(df, sides, sym, sl_atr=sl)
            trades[mode] = _trades_xm(df, ent, rr, sym)

        win_out: dict[str, dict] = {}
        for wname, days in WINDOWS:
            end = df.index[-1]
            start = end - pd.Timedelta(days=days)
            n_bars = int(np.sum(np.asarray(df.index >= start)))
            block = {}
            for mode in all_sides:
                tw = _window_trades(df, trades[mode], days)
                sm = _summarize(tw, n_bars)
                sm["verdict"] = _verdict(sm)
                block[mode] = sm
                desk.setdefault(mode, {w[0]: 0.0 for w in WINDOWS})
                desk[mode][wname] = round(
                    desk[mode][wname] + float(sm["pnl"]), 2
                )
            win_out[wname] = block

        modes_1y = win_out["1y"]
        ranked = sorted(modes_1y.keys(), key=lambda m: _rank_key(modes_1y[m]), reverse=True)
        # Prefer PASS/SOFT; if all FAIL, still pick best but flag no_wire
        abs_win = ranked[0]
        abs_v = _verdict(modes_1y[abs_win])
        wireable = abs_v in ("PASS", "SOFT")

        v2_best_pnl = max(
            float(modes_1y[m]["pnl"]) for m in ("static", "gated", "switch")
        )
        beat_v2 = float(modes_1y[abs_win]["pnl"]) > v2_best_pnl + 1e-6

        top8 = [
            {
                "id": m,
                "pnl_1y": modes_1y[m]["pnl"],
                "exp_r": modes_1y[m]["exp_r"],
                "oos_exp_r": modes_1y[m]["oos_exp_r"],
                "n": modes_1y[m]["n"],
                "wr": modes_1y[m]["win_rate"],
                "wf_green": modes_1y[m]["wf_green"],
                "avg_spread_r": modes_1y[m]["avg_spread_r"],
                "verdict": modes_1y[m]["verdict"],
            }
            for m in ranked[:8]
        ]
        _log(
            f"  ABS={abs_win} {abs_v} ${modes_1y[abs_win]['pnl']:,.0f} "
            f"E[R]={modes_1y[abs_win]['exp_r']:+.3f} wf={modes_1y[abs_win]['wf_green']}/4 "
            f"spR={modes_1y[abs_win]['avg_spread_r']:.3f} wireable={wireable}"
        )
        _log(
            "  top: "
            + ", ".join(
                f"{t['id']}={t['verdict']}${t['pnl_1y']:,.0f}" for t in top8[:5]
            )
        )

        per_pair[sym] = {
            "symbol": sym,
            "family": fam,
            "static_id": wid,
            "static_label": label,
            "rr": rr,
            "sl_atr": sl,
            "spread_risk_mult": float(SPREAD_RISK_MULT.get(sym, 5.0)),
            "v2_preferred": pref_mode,
            "absolute_winner": abs_win,
            "absolute_verdict": abs_v,
            "wireable": wireable,
            "beat_v2": beat_v2,
            "top8_1y": top8,
            "windows": {
                wname: {
                    m: win_out[wname][m]
                    for m in (
                        "static",
                        "gated",
                        "switch",
                        "v2_preferred",
                        abs_win,
                        *[t["id"] for t in top8[:3]],
                    )
                }
                for wname, _ in WINDOWS
            },
            "all_1y": {m: modes_1y[m] for m in ranked},
        }
        eq_lines.append(f"## {sym}")
        eq_lines.append(
            f"- absolute: `{abs_win}` ({abs_v}) wireable={wireable} "
            f"1y ${modes_1y[abs_win]['pnl']:,.0f} E[R]={modes_1y[abs_win]['exp_r']}"
        )
        eq_lines.append(
            f"- SL={sl}×ATR · RR={rr} · spread_mult={SPREAD_RISK_MULT.get(sym)}"
        )
        eq_lines.append("")

    desk_abs = {w[0]: 0.0 for w in WINDOWS}
    desk_wireable = {w[0]: 0.0 for w in WINDOWS}
    for sym, row in per_pair.items():
        if row.get("error"):
            continue
        aw = row["absolute_winner"]
        for wname, _ in WINDOWS:
            pnl = float(row["windows"][wname].get(aw, {}).get("pnl") or 0)
            desk_abs[wname] = round(desk_abs[wname] + pnl, 2)
            if row.get("wireable"):
                desk_wireable[wname] = round(desk_wireable[wname] + pnl, 2)

    wire_n = sum(1 for r in per_pair.values() if r.get("wireable"))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "recommend_live": False,
        "version": "adaptive_xm_cost_v4",
        "tf": TF,
        "risk_usd": RISK_USD,
        "cap_years": CAP_YEARS,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": (
            "Dukascopy adaptive re-hunt with XM session spreads charged per trade. "
            "PA/OB excluded. Absolute winner ranked by PASS/SOFT then 1y PnL."
        ),
        "spreads": "analysis/xm_session_costs.py",
        "formulas": [{"id": a, "label": b} for a, b, _ in formulas],
        "desk_modes": {
            m: desk[m]
            for m in desk
            if m in ("static", "gated", "switch", "v2_preferred")
            or m in {f[0] for f in formulas}
        },
        "desk_absolute": desk_abs,
        "desk_wireable_only": desk_wireable,
        "pairs_wireable": wire_n,
        "per_pair": per_pair,
        "charts": {
            "desk": [
                {
                    "window": w,
                    "absolute": desk_abs[w],
                    "wireable": desk_wireable[w],
                    "static": (desk.get("static") or {}).get(w),
                    "session_mix": (desk.get("session_mix") or {}).get(w),
                    "circadian_kalman_z": (desk.get("circadian_kalman_z") or {}).get(w),
                    "entropy_tod": (desk.get("entropy_tod") or {}).get(w),
                    "volmom": (desk.get("volmom") or {}).get(w),
                }
                for w, _ in WINDOWS
            ],
            "pair_1y": [
                {
                    "symbol": s,
                    "absolute": per_pair[s]["absolute_winner"],
                    "verdict": per_pair[s]["absolute_verdict"],
                    "wireable": per_pair[s]["wireable"],
                    "pnl": per_pair[s]["all_1y"][per_pair[s]["absolute_winner"]]["pnl"],
                    "exp_r": per_pair[s]["all_1y"][per_pair[s]["absolute_winner"]][
                        "exp_r"
                    ],
                    "wf": per_pair[s]["all_1y"][per_pair[s]["absolute_winner"]][
                        "wf_green"
                    ],
                    "avg_spread_r": per_pair[s]["all_1y"][
                        per_pair[s]["absolute_winner"]
                    ]["avg_spread_r"],
                }
                for s in BOOK
                if s in per_pair and not per_pair[s].get("error")
            ],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    EQ.write_text("\n".join(eq_lines) + "\n", encoding="utf-8")
    _progress(
        {
            "status": "done",
            "pct": 100,
            "elapsed_sec": round(time.time() - t0, 1),
            "desk_abs_1y": desk_abs["1y"],
            "desk_wireable_1y": desk_wireable["1y"],
            "pairs_wireable": wire_n,
            "out": str(OUT),
        }
    )
    _log(
        f"\nDESK abs 1y ${desk_abs['1y']:,.0f} · wireable-only ${desk_wireable['1y']:,.0f} "
        f"· wireable {wire_n}/8 · wrote {OUT}"
    )
    for s in BOOK:
        r = per_pair.get(s) or {}
        if r.get("error"):
            continue
        aw = r["absolute_winner"]
        a = r["all_1y"][aw]
        _log(
            f"  REC {s:<8} {aw:<22} {r['absolute_verdict']:<5} "
            f"wf={a['wf_green']}/4 1y ${a['pnl']:,.0f} E[R]={a['exp_r']:+.3f} "
            f"spR={a['avg_spread_r']:.3f} wire={r['wireable']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
