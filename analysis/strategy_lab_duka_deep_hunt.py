#!/usr/bin/env python3
"""DEEP Strategy Lab on Dukascopy local hist — find brains that SURVIVE hard tests.

READ-ONLY for live: does NOT change SYMBOL_ENTRY_BRAIN / .env.

Scope:
  • All retail + hedge strategies from the intraday catalog
  • TFs: 15m, 5m, 1h (resampled) — max Dukascopy depth (~10y on disk)
  • Per cell: full / OOS70 / 4-fold WF / spread+skip torture / MC
  • Survival gate: SURVIVES on full-span battery AND OOS E[R]>0 on last 5y window
  • Hedge strategies ranked with a small bonus (CTA/Donchian/TSM/BB/Z/RSI…)

Limits (honest):
  • Intraday M5 does NOT go back 125y — Dukascopy FX/metals ~2003+; this repo
    currently has ~2016→2026 ingested. Use max available.
  • Free Dukascopy bi5 candles are NOT L2/L3 order book. L2/L3 needs paid
    feeds (Databento / exchange / broker DOM). Hunt uses OHLCV + spread torture.

Outputs:
  data/lab_duka_deep_hunt.json
  data/lab_duka_deep_hunt.log
  data/strategy_lab_progress.json  (UI-compatible progress)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis._edge_survive_bt import BOOK, RISK_USD, _prep, _stats, monte_carlo, verdict  # noqa: E402
from analysis._duka_brains_horizons_bt import _collect_fib_fast  # noqa: E402
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_breakout,
    _collect_ema,
    _collect_sma,
)
from analysis.strategy_lab import (  # noqa: E402
    _collect_adx_ema,
    _collect_atr_vol_break,
    _collect_cta,
    _collect_engulfing,
    _collect_ichimoku_tk,
    _collect_inside_break,
    _collect_liq_sweep,
    _collect_macd,
    _collect_mom_roc,
    _collect_ob_fvg,
    _collect_price_action,
    _collect_psar,
    _collect_rsi,
    _collect_supertrend,
    _collect_trendline,
    _collect_tsmom,
    _collect_vol_scaled_mom,
)
from analysis.strategy_lab_hunt import _bb_std, _keltner_mult, _stoch_band, _zscore_thr  # noqa: E402
from analysis.strategy_lab_intraday_hunt import _catalog as _catalog_raw, _prog  # noqa: E402
from config import symbol_rr  # noqa: E402


def _catalog():
    """Same catalog, but fib uses multi-year-safe collector."""
    out = []
    for sid, family, label, fn in _catalog_raw():
        if sid == "fib_liq":
            out.append((sid, family, label, _collect_fib_fast))
        else:
            out.append((sid, family, label, fn))
    return out

OUT = Path(ROOT) / "data" / "lab_duka_deep_hunt.json"
LOG = Path(ROOT) / "data" / "lab_duka_deep_hunt.log"
LOCK = Path(ROOT) / "data" / "lab_duka_deep_hunt.lock"
DETAIL_UI = Path(ROOT) / "data" / "strategy_lab.json"  # refresh lab board when done

TFS = ("15m", "5m", "1h")
RRS = (1.5, 2.0, 2.4, 3.0, 4.0)
# Heavy collectors: only run on 1h for decade-scale hist (15m/5m too slow)
HEAVY = {
    "fib_liq",
    "pa",
    "liq",
    "tl",
    "obfvg",
    "ichi",
    "ema_pull",
    "ema_tight",
    "adx",
    "psar",
    "engulf",
    "inside",
    "macd",
}


def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _simulate_fast(
    df: pd.DataFrame,
    entries: list,
    rr: float,
    *,
    start_i: int = 0,
    end_i: Optional[int] = None,
) -> list[dict]:
    end_i = end_i if end_i is not None else len(df) - 1
    high = df["high"].to_numpy(dtype=float, copy=False)
    low = df["low"].to_numpy(dtype=float, copy=False)
    n = len(df)
    trades: list[dict] = []
    busy_until = -1
    for e in entries:
        ei = int(e["ei"])
        if ei < start_i or ei > end_i or ei < busy_until:
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
        j_end = min(end_i, n - 1)
        if side == "BUY":
            for j in range(ei + 1, j_end + 1):
                hit_sl = low[j] <= sl
                hit_tp = high[j] >= tp
                if hit_sl or hit_tp:
                    exit_r = -1.0 if hit_sl else float(rr)
                    exit_i = j
                    break
        else:
            for j in range(ei + 1, j_end + 1):
                hit_sl = high[j] >= sl
                hit_tp = low[j] <= tp
                if hit_sl or hit_tp:
                    exit_r = -1.0 if hit_sl else float(rr)
                    exit_i = j
                    break
        if exit_r is None:
            continue
        trades.append(
            {
                "i": ei,
                "entry_t": str(e.get("entry_t", ""))[:16],
                "side": side,
                "r": round(exit_r, 4),
                "pnl": round(exit_r * RISK_USD, 2),
            }
        )
        busy_until = exit_i + 2
    return trades


def _battery(df: pd.DataFrame, entries: list, rr: float) -> dict[str, Any]:
    n = len(df)
    if n < 250 or not entries:
        return {
            "bars": n,
            "entries": len(entries) if entries else 0,
            "full": {"n": 0, "pnl": 0.0, "exp_r": 0.0, "win_rate": 0.0},
            "oos": {"n": 0, "pnl": 0.0, "exp_r": 0.0, "win_rate": 0.0},
            "monte_carlo": {},
            "verdict": "DEAD",
            "wf_green": 0,
            "wf_n": 0,
        }

    def gen(start_i=0, end_i=None, **kw):
        spread_r = float(kw.get("spread_r") or 0)
        skip_p = float(kw.get("skip_p") or 0)
        rng = kw.get("rng") or random.Random(0)
        tr = _simulate_fast(df, entries, rr, start_i=start_i, end_i=end_i)
        if spread_r > 0:
            tr = [
                {
                    **t,
                    "r": round(float(t["r"]) - spread_r, 4),
                    "pnl": round((float(t["r"]) - spread_r) * RISK_USD, 2),
                }
                for t in tr
            ]
        if skip_p > 0 and tr:
            tr = [t for t in tr if rng.random() >= skip_p]
        return tr

    full_tr = gen()
    full = _stats(full_tr)
    split = int(n * 0.70)
    oos = _stats(gen(start_i=split))
    wf = []
    for k in range(4):
        ts = int(n * (0.40 + 0.15 * k))
        te = int(n * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 80:
            continue
        wf.append(
            {
                "fold": k + 1,
                "train": _stats(gen(end_i=ts - 1)),
                "oos": _stats(gen(start_i=ts, end_i=te)),
            }
        )
    torture = {
        "spread_0.15R": _stats(gen(spread_r=0.15)),
        "skip_30pct": _stats(gen(skip_p=0.30, rng=random.Random(7))),
    }
    mc = monte_carlo(full_tr)
    v = verdict(full, oos, wf, torture, mc)
    return {
        "bars": n,
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "entries": len(entries),
        "full": full,
        "oos": oos,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": sum(1 for f in wf if f["oos"]["exp_r"] > 0),
        "wf_n": len(wf),
        "torture": {
            k: {"n": x["n"], "pnl": x["pnl"], "exp_r": x["exp_r"]} for k, x in torture.items()
        },
    }


def _load_frames(sym: str) -> dict[str, pd.DataFrame]:
    from data.hist_local import clear_hist_cache, has_local_hist, load_local_ohlcv

    clear_hist_cache()
    frames: dict[str, pd.DataFrame] = {}
    for tf in ("5m", "15m"):
        if not has_local_hist(sym, tf):
            continue
        df = load_local_ohlcv(sym, tf=tf)
        if df is None or len(df) < 400:
            continue
        need = [c for c in ("open", "high", "low", "close") if c in df.columns]
        frames[tf] = _prep(df[need].dropna())
        _log(
            f"  load {sym} {tf}: {len(frames[tf]):,} bars "
            f"{frames[tf].index[0].date()}→{frames[tf].index[-1].date()}"
        )
    if "5m" in frames:
        h = (
            frames["5m"]
            .resample("1h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )
        if len(h) >= 300:
            frames["1h"] = _prep(h)
            _log(
                f"  load {sym} 1h: {len(frames['1h']):,} bars (from 5m) "
                f"{frames['1h'].index[0].date()}→{frames['1h'].index[-1].date()}"
            )
    return frames


def _slice_years(df: pd.DataFrame, years: float, end: pd.Timestamp) -> pd.DataFrame:
    start = end - pd.Timedelta(days=int(years * 365.25))
    return df[df.index >= start]


def _rank_score(bat: dict, family: str) -> float:
    if bat.get("verdict") == "DEAD":
        return -1e9
    oos = bat.get("oos") or {}
    full = bat.get("full") or {}
    mc = bat.get("monte_carlo") or {}
    s = float(oos.get("exp_r") or 0) * 4.0
    s += float(full.get("exp_r") or 0) * 1.5
    s += 0.02 * float(mc.get("prob_profit") or 0)
    s += 0.35 * float(bat.get("wf_green") or 0)
    if bat.get("verdict") == "SURVIVES":
        s += 3.0
    elif bat.get("verdict") == "FRAGILE":
        s += 0.8
    if family == "hedge":
        s += 0.4  # slight preference — user asked hedge-fund book
    # torture: don't reward if spread kills edge
    t = (bat.get("torture") or {}).get("spread_0.15R") or {}
    if float(t.get("exp_r") or 0) <= 0:
        s -= 1.5
    return s


def _passes_deep_gate(full_bat: dict, bat_5y: dict) -> bool:
    """Must SURVIVE full span and keep positive OOS on last 5y."""
    if full_bat.get("verdict") != "SURVIVES":
        return False
    o5 = bat_5y.get("oos") or {}
    if float(o5.get("exp_r") or 0) <= 0:
        return False
    if int((bat_5y.get("full") or {}).get("n") or 0) < 40:
        return False
    if int(full_bat.get("wf_green") or 0) < 2:
        return False
    return True


def main() -> int:
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip() or "0")
            if pid and Path(f"/proc/{pid}").exists():
                _log(f"another deep hunt running (pid {pid}) — exit")
                return 2
        except Exception:
            pass
    LOCK.write_text(str(os.getpid()))
    if LOG.exists():
        LOG.write_text("")

    t0 = time.time()
    catalog = _catalog()
    end = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=1), tz="UTC")

    _log(
        "DUKA DEEP HUNT · all retail+hedge · Dukascopy hist · "
        f"${RISK_USD}/R · TFs={TFS}\n"
        "LIVE BRAINS NOT MODIFIED\n"
        "Note: max hist ≈10y on disk (not 125y). No L2/L3 on free Dukascopy candles.\n"
    )
    _prog(
        status="running",
        phase="duka_deep_hunt",
        pct=1,
        message=f"Duka deep hunt · {len(BOOK)} pairs · {len(catalog)} strategies · TFs {TFS}",
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
            frames = _load_frames(sym)
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
                if sid in HEAVY and tf != "1h":
                    continue
                if tf == "5m" and family == "retail" and sid not in ("sma_cross", "st_2", "st_3", "kelt_1", "kelt_1.5", "kelt_2", "stoch_20", "stoch_30"):
                    continue
                df_all = frames[tf]
                cells_total += 1
                _log(f"  … {sid}@{tf} ({len(df_all):,} bars)")
                _prog(
                    status="running",
                    phase="duka_deep_hunt",
                    pct=int(5 + 90 * (pi + 0.2) / len(BOOK)),
                    message=f"[{sym}] {sid}@{tf}",
                    symbol=sym,
                    candles_total=candles_total,
                )
                try:
                    # Collect once on full span
                    entries = fn(df_all)
                    if not entries or len(entries) < 30:
                        continue
                    # RR sweep — keep best by deep gate then score
                    df5 = _slice_years(df_all, 5.0, end)
                    try:
                        entries5 = fn(df5) if len(df5) >= 300 else []
                    except Exception:
                        entries5 = []
                    best_cell = None
                    for rr in RRS:
                        bat = _battery(df_all, entries, rr)
                        if len(df5) < 300 or len(entries5) < 20:
                            bat5 = {
                                "verdict": "DEAD",
                                "oos": {"exp_r": 0},
                                "full": {"n": 0},
                                "wf_green": 0,
                            }
                        else:
                            bat5 = _battery(df5, entries5, rr)
                        gate = _passes_deep_gate(bat, bat5)
                        cell = {
                            "strategy": sid,
                            "family": family,
                            "label": label,
                            "tf": tf,
                            "rr": rr,
                            "full_span": bat,
                            "last_5y": {
                                "verdict": bat5.get("verdict"),
                                "full": bat5.get("full"),
                                "oos": bat5.get("oos"),
                                "wf_green": bat5.get("wf_green"),
                            },
                            "deep_gate": gate,
                            "score": round(_rank_score(bat, family) + (2.5 if gate else 0), 3),
                        }
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
                                f"5yOOS={best_cell['last_5y']['oos'].get('exp_r', 0):+.3f} "
                                f"{best_cell['full_span']['verdict']}"
                            )
                except Exception as exc:
                    _log(f"  skip {sid}@{tf}: {exc}")

            _prog(
                status="running",
                phase="duka_deep_hunt",
                pct=int(5 + 90 * (pi + 0.5) / len(BOOK)),
                message=f"[{sym}] scored {len(scored)} cells · survives_cells={survives_cells}",
                symbol=sym,
                candles_total=candles_total,
            )

        scored.sort(key=lambda r: float(r.get("score") or -1e9), reverse=True)
        gated = [r for r in scored if r.get("deep_gate")]
        pool = gated if gated else [r for r in scored if r["full_span"].get("verdict") == "SURVIVES"]
        if not pool:
            pool = [r for r in scored if r["full_span"].get("verdict") != "DEAD"]
        winner = pool[0] if pool else None

        # Top hedge + top retail for the pair
        top_hedge = next((r for r in scored if r["family"] == "hedge" and r.get("deep_gate")), None)
        if top_hedge is None:
            top_hedge = next(
                (r for r in scored if r["family"] == "hedge" and r["full_span"].get("verdict") == "SURVIVES"),
                None,
            )
        top_retail = next((r for r in scored if r["family"] == "retail" and r.get("deep_gate")), None)

        per_pair[sym] = {
            "rr_live": rr_live,
            "cells": len(scored),
            "gated_count": len(gated),
            "winner": winner,
            "top_hedge": top_hedge,
            "top_retail": top_retail,
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
                "score": winner["score"],
            }
            _log(
                f"  ★ {sym} WINNER {winner['strategy']}@{winner['tf']} "
                f"RR={winner['rr']} score={winner['score']} "
                f"PnL=${winner['full_span']['full']['pnl']:,.0f}"
            )
        else:
            _log(f"  ✗ {sym}: no deep-gate survivor (best non-dead kept in top10)")

        # checkpoint
        payload_ckpt = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "survivors_book": survivors_book,
            "per_pair": {k: {
                "gated_count": v.get("gated_count"),
                "winner": v.get("winner"),
                "top_hedge": v.get("top_hedge"),
            } for k, v in per_pair.items()},
        }
        OUT.write_text(json.dumps(payload_ckpt, indent=2, default=str))

    n_gate = len(survivors_book)
    proposed_map = ",".join(
        f"{s}:{survivors_book[s]['strategy']}" for s in BOOK if s in survivors_book
    )
    proposed_tf = ",".join(
        f"{s}:{survivors_book[s]['tf']}" for s in BOOK if s in survivors_book
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "duka_deep_hunt_readonly",
        "source": "dukascopy_local_hist",
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
        "per_pair": per_pair,
        "limits": {
            "hist_years_available": "~10 (2016→2026 on disk; Dukascopy FX/metals from ~2003 possible)",
            "hist_years_requested_125": False,
            "l2_l3": (
                "Not available on free Dukascopy M1/M5 candles. "
                "Need Databento/exchange/broker DOM for true L2/L3. "
                "This hunt uses OHLCV + 0.15R spread torture as stress proxy."
            ),
            "live_unchanged": True,
        },
        "summary": {
            "pairs_with_deep_gate": n_gate,
            "desk_success_rate_pct": round(100.0 * n_gate / max(len(BOOK), 1), 2),
            "note": (
                "Winners listed under survivors_book. "
                "NOT wired live — review then ask to apply."
            ),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    # Mirror a slim board into strategy_lab.json for the UI (research only)
    try:
        board = {
            "generated_at": payload["generated_at"],
            "mode": "duka_deep_hunt",
            "summary": {
                "pairs_with_survives": n_gate,
                "strategies_tested": len(catalog),
                "cells": cells_total,
                "survives_cells": survives_cells,
                "desk_success_rate_pct": payload["summary"]["desk_success_rate_pct"],
                "candles_total": candles_total,
            },
            "winners": survivors_book,
            "risk_usd": RISK_USD,
            "note": "Dukascopy deep hunt — live brains NOT changed",
        }
        DETAIL_UI.write_text(json.dumps(board, indent=2, default=str))
    except Exception:
        pass

    _prog(
        status="done",
        phase="duka_deep_hunt",
        pct=100,
        message=(
            f"Duka deep hunt done · {n_gate}/{len(BOOK)} deep-gate pairs · "
            f"{survives_cells} SURVIVES cells · {payload['elapsed_s']}s"
        ),
        candles_total=candles_total,
    )
    _log(
        f"\n→ Deep-gate survivors: {n_gate}/{len(BOOK)}\n"
        f"→ Proposed map (NOT applied): {proposed_map or '(none)'}\n"
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
