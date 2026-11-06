#!/usr/bin/env python3
"""RESEARCH ONLY — USDCHF PA TF grid across horizons + breakeven timing.

Windows: 2m / 6m / 12m / 2y (trailing from Dukascopy hist end).
Process: liq sweep → dual-TF EMA bias → engulf
TFs: {1m,3m,5m,15m} · liq ∈ {3m,5m,15m} · entry ≤ liq · RR ∈ {2,2.4,3,4}

For each cell, equity path reports:
  - first time cum PnL > 0 (first profit)
  - first recovery to ≥ 0 after going red (breakeven after DD)
  - first trade from which equity never goes red again (locked BE+)
  - calendar days from window start for each milestone

Does NOT change live wiring.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD, _prep  # noqa: E402
from analysis.lab_agentic_1y_battery import _battery_from_trades  # noqa: E402
from analysis.lab_crypto_pa_tf_grid_bt import _rank_key, _slim  # noqa: E402
from analysis.lab_gold_pa_liq_engulf_bt import RRS, _resample_ohlc  # noqa: E402
from analysis.lab_gold_pa_tf_grid_bt import (  # noqa: E402
    LIQ_TFS,
    TFS,
    _collect_entries,
    _combo_iter,
    _fast_sweeps,
)
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402
from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_usdchf_pa_horizons_tf_bt.json"
LOG = Path(ROOT) / "data" / "lab_usdchf_pa_horizons_tf_bt.log"

SYM = "USDCHF"
WINDOWS = (
    ("2m", 61),
    ("6m", 183),
    ("12m", 365),
    ("2y", 730),
)
WIRED_BB = {"brain": "bb_2.5", "tf": "1m", "rr": 4.0, "bb_mult": 2.5}
# Focus / trial PA stack (overridden per-symbol runners)
LIVE_PA = {
    "liq_tf": "3m",
    "bias_a": "15m",
    "bias_b": "5m",
    "entry_tf": "1m",
    "rr": 2.0,
}
# Long-horizon efficient reference (always reported)
EFFICIENT_PA = {
    "liq_tf": "3m",
    "bias_a": "15m",
    "bias_b": "5m",
    "entry_tf": "1m",
    "rr": 2.0,
}


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _parse_t(s: Any) -> Optional[pd.Timestamp]:
    if s is None or s == "":
        return None
    try:
        ts = pd.Timestamp(s)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts
    except Exception:
        return None


def _days_from(start: pd.Timestamp, t: Optional[pd.Timestamp]) -> Optional[float]:
    if t is None or start is None:
        return None
    return round((t - start).total_seconds() / 86400.0, 2)


def _equity_milestones(
    trades: list[dict], window_start: pd.Timestamp
) -> dict[str, Any]:
    """Walk cum PnL; emphasize BE recovery then profit lock."""
    if not trades:
        return {
            "n": 0,
            "final_pnl": 0.0,
            "max_dd_usd": 0.0,
            "went_red": False,
            "first_profit": None,
            "be_after_dd": None,
            "locked_be": None,
        }

    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    path: list[float] = []
    times: list[Optional[pd.Timestamp]] = []
    went_red = False
    first_profit = None
    be_after_dd = None
    waiting_be = False

    for i, t in enumerate(trades):
        cum += float(t.get("pnl") or 0)
        path.append(cum)
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        ts = _parse_t(t.get("exit_t") or t.get("entry_t"))
        times.append(ts)
        if cum < -1e-9:
            went_red = True
            waiting_be = True
        if first_profit is None and cum > 1e-9:
            first_profit = {
                "trade_n": i + 1,
                "cum_pnl": round(cum, 2),
                "at": str(ts)[:19] if ts is not None else None,
                "days_from_start": _days_from(window_start, ts),
            }
        if waiting_be and cum >= -1e-9 and be_after_dd is None:
            be_after_dd = {
                "trade_n": i + 1,
                "cum_pnl": round(cum, 2),
                "at": str(ts)[:19] if ts is not None else None,
                "days_from_start": _days_from(window_start, ts),
            }
            waiting_be = False

    locked_be = None
    for i, c in enumerate(path):
        if min(path[i:]) >= -1e-9:
            ts = times[i]
            locked_be = {
                "trade_n": i + 1,
                "cum_pnl": round(c, 2),
                "at": str(ts)[:19] if ts is not None else None,
                "days_from_start": _days_from(window_start, ts),
                "note": "from this trade onward equity never < 0",
            }
            break

    # profit_after_be: first time cum > 0 after be_after_dd (or first_profit if never red)
    profit_after_be = None
    if be_after_dd is not None:
        start_i = int(be_after_dd["trade_n"])  # 1-based; look from next
        for i in range(start_i, len(path)):
            if path[i] > 1e-9:
                ts = times[i]
                profit_after_be = {
                    "trade_n": i + 1,
                    "cum_pnl": round(path[i], 2),
                    "at": str(ts)[:19] if ts is not None else None,
                    "days_from_start": _days_from(window_start, ts),
                }
                break
    elif first_profit is not None:
        profit_after_be = {**first_profit, "note": "never went red — first green is profit"}

    return {
        "n": len(trades),
        "final_pnl": round(cum, 2),
        "max_dd_usd": round(max_dd, 2),
        "went_red": went_red,
        "first_profit": first_profit,
        "be_after_dd": be_after_dd,
        "profit_after_be": profit_after_be,
        "locked_be": locked_be,
    }


def _simulate_with_exit(
    df: pd.DataFrame, entries: list, rr: float
) -> list[dict]:
    """Like _simulate_fast but stamps exit_t for equity calendar days."""
    from analysis._edge_survive_bt import RISK_USD as R

    high = df["high"].to_numpy(dtype=float, copy=False)
    low = df["low"].to_numpy(dtype=float, copy=False)
    idx = df.index
    n = len(df)
    trades: list[dict] = []
    busy_until = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < 0 or ei >= n or ei < busy_until:
            continue
        side = e["side"]
        entry = float(e["entry"])
        sl0 = float(e["sl"])
        risk = abs(entry - sl0)
        if risk <= 0:
            continue
        if side == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk
        exit_r = None
        exit_i = ei
        if side == "BUY":
            for j in range(ei + 1, n):
                hit_sl = low[j] <= sl
                hit_tp = high[j] >= tp
                if hit_sl or hit_tp:
                    exit_r = -1.0 if hit_sl else float(rr)
                    exit_i = j
                    break
        else:
            for j in range(ei + 1, n):
                hit_sl = high[j] >= sl
                hit_tp = low[j] <= tp
                if hit_sl or hit_tp:
                    exit_r = -1.0 if hit_sl else float(rr)
                    exit_i = j
                    break
        if exit_r is None:
            continue
        et = e.get("entry_t")
        if not et:
            et = str(idx[ei])[:19]
        trades.append(
            {
                "i": ei,
                "entry_t": str(et)[:19],
                "exit_t": str(idx[exit_i])[:19],
                "side": side,
                "r": round(exit_r, 4),
                "pnl": round(exit_r * R, 2),
            }
        )
        busy_until = exit_i + 2
    return trades


def _load_frames(window_days: int) -> tuple[dict[str, pd.DataFrame], pd.Timestamp, pd.Timestamp]:
    clear_hist_cache()
    # Anchor to hist end (local Dukascopy may lag wall clock)
    probe = None
    for tf in ("1m", "5m"):
        if has_local_hist(SYM, tf):
            probe = load_local_ohlcv(SYM, tf=tf)
            break
    if probe is None or len(probe) < 400:
        raise RuntimeError(f"{SYM}: no hist")
    data_end = probe.index[-1]
    if getattr(data_end, "tzinfo", None) is not None:
        data_end = data_end.tz_localize(None)
    start = data_end - pd.Timedelta(days=window_days)
    # Warmup for EMAs / swings
    load_start = start - pd.Timedelta(days=14)

    frames: dict[str, pd.DataFrame] = {}
    for tf in TFS:
        if not has_local_hist(SYM, tf):
            continue
        try:
            df = load_local_ohlcv(
                SYM, tf=tf, start=load_start.isoformat(), end=data_end.isoformat()
            )
        except Exception as exc:
            _log(f"  load {SYM} {tf} FAIL: {exc}")
            continue
        if df is None or len(df) < 200:
            continue
        need = [c for c in ("open", "high", "low", "close") if c in df.columns]
        df = _prep(df[need].dropna())
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        frames[tf] = df

    for tf, rule in (("3m", "3min"), ("5m", "5min"), ("15m", "15min")):
        if tf not in frames:
            src = frames.get("1m") or frames.get("5m")
            if src is None:
                raise RuntimeError(f"{SYM}: need 1m or 5m")
            frames[tf] = _resample_ohlc(src, rule)

    # Trim to window for trading (keep warmup inside each frame for indicators —
    # _prep already computed emas on load slice; slice trading window after)
    trimmed: dict[str, pd.DataFrame] = {}
    for tf, df in frames.items():
        # Keep a small warmup inside the frame used for sweeps/bias
        warm = start - pd.Timedelta(days=7)
        trimmed[tf] = df[df.index >= warm]
        _log(
            f"  load {SYM} {tf}: {len(trimmed[tf]):,} bars "
            f"{trimmed[tf].index[0]}→{trimmed[tf].index[-1]}"
        )
    return trimmed, start, data_end


def _pack_cell(
    name: str,
    df: pd.DataFrame,
    entries: list,
    rr: float,
    meta: dict,
    window_start: pd.Timestamp,
) -> dict[str, Any]:
    # Restrict entries to window
    ents = []
    for e in entries:
        et = _parse_t(e.get("entry_t"))
        if et is not None and et < window_start:
            continue
        ents.append(e)
    tr = _simulate_with_exit(df, ents, rr)
    # Also filter sims whose entry bar is before window
    bat = _battery_from_trades(tr, len(df))
    full = bat.get("full") or {}
    tests = bat.get("tests") or {}
    eq = _equity_milestones(tr, window_start)
    return {
        "name": name,
        **meta,
        "rr": rr,
        "entries_raw": len(ents),
        "trades": full.get("n", 0),
        "pnl": full.get("pnl", 0.0),
        "exp_r": full.get("exp_r", 0.0),
        "win_rate": full.get("win_rate", 0.0),
        "verdict": bat.get("verdict"),
        "wf_green": bat.get("wf_green"),
        "wf_n": bat.get("wf_n"),
        "mc_prob_profit": (bat.get("monte_carlo") or {}).get("prob_profit"),
        "equity": eq,
        "tests": {
            k: {
                "n": (tests.get(k) or {}).get("n"),
                "pnl": (tests.get(k) or {}).get("pnl"),
                "exp_r": (tests.get(k) or {}).get("exp_r"),
            }
            for k in (
                "1_full",
                "2_oos70",
                "3_wf_fold_1",
                "4_wf_fold_2",
                "5_wf_fold_3",
                "6_wf_fold_4",
                "7_torture_spread",
                "8_torture_skip",
            )
        },
    }


def _run_window(label: str, days: int) -> dict[str, Any]:
    t0 = time.time()
    _log(f"\n{'='*64}\n===== WINDOW {label} ({days}d) =====")
    frames, start, end = _load_frames(days)
    _log(f"  trade window: {start.date()} → {end.date()}")

    sweeps_cache: dict[str, list] = {}
    for liq in LIQ_TFS:
        t1 = time.time()
        sweeps_cache[liq] = _fast_sweeps(frames[liq])
        # drop sweeps before window
        sweeps_cache[liq] = [
            s for s in sweeps_cache[liq] if _parse_t(s.get("t")) is None or _parse_t(s["t"]) >= start
        ]
        _log(f"  sweeps {liq}: {len(sweeps_cache[liq])} ({time.time()-t1:.1f}s)")

    combos = _combo_iter()
    cells: list[dict[str, Any]] = []
    for ci, cfg in enumerate(combos, 1):
        liq, ea, eb, ent = cfg["liq_tf"], cfg["bias_a"], cfg["bias_b"], cfg["entry_tf"]
        tag = f"liq={liq}|bias={ea}+{eb}|entry={ent}"
        t1 = time.time()
        entries = _collect_entries(
            frames[ent], sweeps_cache[liq], frames[ea], frames[eb], ent
        )
        _log(f"[{ci}/{len(combos)}] {tag}: entries={len(entries)} ({time.time()-t1:.1f}s)")
        meta = {"liq_tf": liq, "bias_a": ea, "bias_b": eb, "entry_tf": ent}
        for rr in RRS:
            cell = _pack_cell(tag, frames[ent], entries, rr, meta, start)
            cells.append(cell)

    ranked = sorted(cells, key=_rank_key, reverse=True)
    survives = [c for c in ranked if c.get("verdict") == "SURVIVES"]
    top10 = [_slim_eq(c) for c in ranked[:10]]

    def _match(spec: dict) -> Optional[dict]:
        return next(
            (
                c
                for c in cells
                if c.get("liq_tf") == spec["liq_tf"]
                and c.get("bias_a") == spec["bias_a"]
                and c.get("bias_b") == spec["bias_b"]
                and c.get("entry_tf") == spec["entry_tf"]
                and abs(float(c["rr"]) - float(spec["rr"])) < 1e-9
            ),
            None,
        )

    live = _match(LIVE_PA)
    efficient = _match(EFFICIENT_PA)

    bb_cells: list[dict] = []
    if "1m" in frames:
        try:
            bb_cells = _bb_entries_pack(frames["1m"], start)
        except Exception as exc:
            _log(f"  bb baseline skip: {exc}")

    _log_window_summary(label, live, top10[0] if top10 else None, bb_cells)
    if efficient and efficient is not live:
        eq = efficient.get("equity") or {}
        _log(
            f"EFFICIENT PA liq=3m|bias=15m+5m|entry=1m RR=2: "
            f"n={efficient.get('trades')} pnl=${efficient.get('pnl'):,.0f} "
            f"{efficient.get('verdict')} | BE={_fmt_ms(eq.get('be_after_dd'))} "
            f"→ profit={_fmt_ms(eq.get('profit_after_be'))} "
            f"| locked={_fmt_ms(eq.get('locked_be'))}"
        )

    return {
        "label": label,
        "days": days,
        "start": str(start)[:19],
        "end": str(end)[:19],
        "combos": len(combos),
        "cells": len(cells),
        "survives_n": len(survives),
        "best": top10[0] if top10 else None,
        "top10": top10,
        "live_pa": _slim_eq(live) if live else None,
        "efficient_pa": _slim_eq(efficient) if efficient else None,
        "wired_bb": [_slim_eq(c) for c in bb_cells],
        "elapsed_s": round(time.time() - t0, 1),
    }


def _slim_eq(c: Optional[dict]) -> Optional[dict]:
    if not c:
        return None
    base = _slim(c)
    base["equity"] = c.get("equity")
    base["is_wired_bb"] = c.get("is_wired_bb")
    return base


def _bb_entries_pack(df_1m: pd.DataFrame, window_start: pd.Timestamp) -> list[dict]:
    """Wired bb_2.5 baseline via same _bb_std helper as 1y grid."""
    entries = _bb_std(df_1m, float(WIRED_BB["bb_mult"]))
    _log(f"  BB entries: {len(entries)}")
    out = []
    for rr in sorted(set(RRS) | {float(WIRED_BB["rr"])}):
        cell = _pack_cell(
            "wired_bb_2.5",
            df_1m,
            entries,
            float(rr),
            {
                "liq_tf": None,
                "bias_a": None,
                "bias_b": None,
                "entry_tf": "1m",
                "brain": "bb_2.5",
            },
            window_start,
        )
        cell["is_wired_bb"] = abs(float(rr) - float(WIRED_BB["rr"])) < 1e-9
        out.append(cell)
        eq = cell.get("equity") or {}
        tag = "WIRED" if cell["is_wired_bb"] else ""
        _log(
            f"  bb_2.5 RR={rr} {tag}: n={cell['trades']} pnl=${cell['pnl']:,.0f} "
            f"{cell['verdict']} | BE={_fmt_ms(eq.get('be_after_dd'))} "
            f"profit={_fmt_ms(eq.get('profit_after_be'))} "
            f"locked={_fmt_ms(eq.get('locked_be'))}"
        )
    return out


def _fmt_ms(m: Optional[dict]) -> str:
    if not m:
        return "—"
    d = m.get("days_from_start")
    return f"t{m.get('trade_n')}@{m.get('at','?')[:10]}({d}d)" if d is not None else f"t{m.get('trade_n')}"


def _log_window_summary(
    label: str,
    live: Optional[dict],
    best: Optional[dict],
    bb_cells: list,
) -> None:
    _log(f"\n--- {label} SUMMARY ---")
    if live:
        eq = live.get("equity") or {}
        _log(
            f"LIVE PA liq=3m|bias=15m+5m|entry=1m RR=2: "
            f"n={live.get('trades')} pnl=${live.get('pnl'):,.0f} "
            f"E[R]={live.get('exp_r')} {live.get('verdict')}"
        )
        _log(f"  went_red={eq.get('went_red')} maxDD=${eq.get('max_dd_usd')}")
        _log(f"  first_profit: {_fmt_ms(eq.get('first_profit'))}")
        _log(f"  BE after DD:  {_fmt_ms(eq.get('be_after_dd'))}")
        _log(f"  profit after BE: {_fmt_ms(eq.get('profit_after_be'))}")
        _log(f"  locked BE+:   {_fmt_ms(eq.get('locked_be'))}")
    if best:
        _log(
            f"BEST PA: {best.get('name')} RR={best.get('rr')} "
            f"pnl=${best.get('pnl'):,.0f} {best.get('verdict')}"
        )
        eq = best.get("equity") or {}
        _log(
            f"  BE after DD: {_fmt_ms(eq.get('be_after_dd'))} | "
            f"profit after BE: {_fmt_ms(eq.get('profit_after_be'))} | "
            f"locked: {_fmt_ms(eq.get('locked_be'))}"
        )


def main() -> int:
    t0 = time.time()
    try:
        LOG.write_text("")
    except Exception:
        pass
    _log("=== USDCHF PA horizons × TF grid · Dukascopy · RESEARCH ===")
    _log(f"Windows: {', '.join(f'{a}({b}d)' for a,b in WINDOWS)}")
    _log(f"Risk ${RISK_USD}/R · Live PA NOT changed")

    windows_out = []
    for label, days in WINDOWS:
        windows_out.append(_run_window(label, days))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "symbol": SYM,
        "risk_usd": RISK_USD,
        "live_pa": LIVE_PA,
        "process": "liq_sweep → dual EMA bias → engulf",
        "tf_universe": list(TFS),
        "windows": windows_out,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _log(f"\nWrote {OUT} ({payload['elapsed_s']}s)")

    # Final cross-horizon table for live PA
    _log("\n======== LIVE PA CROSS-HORIZON ========")
    for w in windows_out:
        lp = w.get("live_pa") or {}
        eq = lp.get("equity") or {}
        _log(
            f"{w['label']:>4}: pnl=${lp.get('pnl', 0):>10,.0f} n={lp.get('trades')} "
            f"{lp.get('verdict')} | BE={_fmt_ms(eq.get('be_after_dd'))} "
            f"→ profit={_fmt_ms(eq.get('profit_after_be'))} "
            f"| locked={_fmt_ms(eq.get('locked_be'))} maxDD=${eq.get('max_dd_usd')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
