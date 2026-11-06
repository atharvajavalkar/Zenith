#!/usr/bin/env python3
"""RESEARCH ONLY — Schwab densify v2 crowns vs CURRENT live brains.

Live map (from .env):
  GOLD/BTCUSD/UKOIL → omni @ SYMBOL_SCAN_TFS
  NAS100 → session_mix @ 5m

Schwab: locked per-pair crowns from lab_schwab_tally_densify_v2.json (1y best).
Same $200/R, XM spreads, 1y + full horizons, 8-test battery.

Does NOT touch Nebula / live / .env.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis._edge_survive_bt import RISK_USD, _stats, monte_carlo, verdict  # noqa: E402
from analysis.lab_adaptive_xm_deep_hunt_bt import (  # noqa: E402
    _entries,
    _simulate_fast as _barrier_sim,
)
from analysis.lab_omnihorizon_adaptive_bt import _omni_route  # noqa: E402
from analysis.xm_session_costs import spread_price  # noqa: E402
from config import (  # noqa: E402
    INSTRUMENT_UNIVERSE,
    symbol_entry_brain,
    symbol_rr,
    symbol_scan_tfs,
)
from data.hist_local import clear_hist_cache, load_local_ohlcv  # noqa: E402
from strategy.omni_live import (  # noqa: E402
    OMNI_LIVE,
    _feature_frame,
    _sides_circadian,
    _sides_session_mix,
    _sides_static,
)
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL  # noqa: E402

# Reuse schwab sim pieces (keyword-only _simulate_fast — do not shadow barrier sim)
from analysis.lab_schwab_tally_densify_v2_bt import (  # noqa: E402
    LOOKBACK_SETS,  # noqa: F401 — ensure module loads
    _atr_arr,
    _base_size,
    _load_daily,
    _simulate_fast as _schwab_sim,
    _tally_from_cache,
    _vote_cache,
    SL_ATR as SCHWAB_SL_ATR,
)

SCHWAB_V2 = Path(ROOT) / "data" / "lab_schwab_tally_densify_v2.json"
OUT = Path(ROOT) / "data" / "lab_schwab_vs_live_brains.json"
LOG = Path(ROOT) / "data" / "lab_schwab_vs_live_brains.log"
MD = Path(ROOT) / "data" / "lab_schwab_vs_live_brains.md"

LIVE_BOOK = [s for s in ("GOLD", "BTCUSD", "UKOIL", "NAS100") if s in set(INSTRUMENT_UNIVERSE)]
HORIZONS = {"1y": 365, "full": None}


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _battery(full_tr: list[dict], n_bars: int) -> dict[str, Any]:
    if not full_tr or n_bars < 80:
        empty = _stats([])
        return {
            "full": empty,
            "oos": empty,
            "monte_carlo": {},
            "verdict": "DEAD",
            "wf_green": 0,
            "wf_n": 0,
            "tests": {},
            "entries": 0,
        }

    def by_range(start_i: int = 0, end_i: Optional[int] = None) -> list[dict]:
        end_i = n_bars - 1 if end_i is None else end_i
        return [t for t in full_tr if start_i <= int(t["i"]) <= end_i]

    full = _stats(full_tr)
    split = int(n_bars * 0.70)
    oos = _stats(by_range(start_i=split))
    wf, wf_details = [], {}
    for k in range(4):
        ts = int(n_bars * (0.40 + 0.15 * k))
        te = int(n_bars * (0.55 + 0.15 * k)) - 1
        if te <= ts or ts < 60:
            continue
        fold = _stats(by_range(start_i=ts, end_i=te))
        wf.append({"fold": k + 1, "oos": fold})
        wf_details[f"wf_fold_{k+1}"] = fold
    spread_tr = [
        {
            **t,
            "r": round(float(t["r"]) - 0.15, 4),
            "pnl": round((float(t["r"]) - 0.15) * RISK_USD, 2),
        }
        for t in full_tr
    ]
    rng = random.Random(7)
    skip_tr = [t for t in full_tr if rng.random() >= 0.30]
    torture = {"spread_0.15R": _stats(spread_tr), "skip_30pct": _stats(skip_tr)}
    mc = monte_carlo(full_tr, paths=1000)
    v = verdict(full, oos, wf, torture, mc)
    tests = {
        "1_full": full,
        "2_oos70": oos,
        "3_wf_fold_1": wf_details.get("wf_fold_1") or {"n": 0, "exp_r": 0},
        "4_wf_fold_2": wf_details.get("wf_fold_2") or {"n": 0, "exp_r": 0},
        "5_wf_fold_3": wf_details.get("wf_fold_3") or {"n": 0, "exp_r": 0},
        "6_wf_fold_4": wf_details.get("wf_fold_4") or {"n": 0, "exp_r": 0},
        "7_torture_spread": torture["spread_0.15R"],
        "8_torture_skip": torture["skip_30pct"],
        "mc_prob_profit": mc.get("prob_profit"),
    }
    return {
        "full": full,
        "oos": oos,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": sum(1 for f in wf if f["oos"]["exp_r"] > 0),
        "wf_n": len(wf),
        "tests": tests,
        "entries": len(full_tr),
    }


def _slice_daily(daily: pd.DataFrame, days: Optional[int]) -> tuple[pd.DataFrame, int]:
    if days is None:
        return daily, 0
    end = daily.index[-1]
    cut = end - pd.Timedelta(days=int(days))
    warm = cut - pd.Timedelta(days=150)
    frame = daily[daily.index >= warm]
    cut_i = 0
    for i, ts in enumerate(frame.index):
        if ts >= cut:
            cut_i = i
            break
    return frame, cut_i


def _schwab_trades(daily: pd.DataFrame, crown: dict, cut_i: int) -> list[dict]:
    lb = tuple(crown["lookbacks"])
    c = daily["close"].to_numpy(dtype=float)
    o = daily["open"].to_numpy(dtype=float)
    h = daily["high"].to_numpy(dtype=float)
    l = daily["low"].to_numpy(dtype=float)
    atr = _atr_arr(daily)
    votes = _vote_cache(c, max(lb))
    tally = _tally_from_cache(votes, lb)
    try:
        sp = float(spread_price(crown.get("symbol") or "GOLD", hour_utc=12.0))
    except Exception:
        sp = 0.0
    med_a = float(np.nanmedian(atr[np.isfinite(atr)])) if np.isfinite(atr).any() else 1.0
    spread_r = max(0.02, sp / med_a) if med_a > 0 and sp > 0 else 0.05
    trades = _schwab_sim(
        o=o,
        h=h,
        l=l,
        c=c,
        atr=atr,
        tally=tally,
        lookbacks=lb,
        size_mode=str(crown["size_mode"]),
        lot_mult=float(crown["lot_mult"]),
        tp_r=crown.get("tp_r"),
        hold_days=int(crown["hold_days"]),
        spread_r_default=spread_r,
    )
    if cut_i > 0:
        trades = [t for t in trades if int(t["i"]) >= cut_i]
    return trades


def _prep_intraday(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    if "atr" not in d.columns:
        c = d["close"].astype(float)
        h = d["high"].astype(float)
        l = d["low"].astype(float)
        prev = c.shift(1)
        tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        d["atr"] = tr.rolling(14, min_periods=5).mean()
    return d.dropna()


def _live_omni_sides(df: pd.DataFrame, sym: str, tf: str) -> tuple[np.ndarray, dict]:
    cfg = OMNI_LIVE.get((sym, tf))
    if cfg is None:
        raise KeyError(f"no OMNI_LIVE for {sym}/{tf}")
    feat = _feature_frame(df)
    if feat is None:
        raise RuntimeError("feature frame failed")
    # align length to df
    ret = feat["ret"].to_numpy(dtype=float)
    brain_sides = {
        "session_mix": _sides_session_mix(feat, 0.22, 0.35),
        "session_mix_strict": _sides_session_mix(feat, 0.32, 0.42),
        "circadian": _sides_circadian(feat, 1.7, 0.6),
        "static": _sides_static(feat),
    }
    sides, meta = _omni_route(
        brain_sides,
        ret,
        look_fast=int(cfg["look_fast"]),
        look_mid=int(cfg["look_mid"]),
        look_slow=int(cfg["look_slow"]),
        edge_min=float(cfg["edge_min"]),
        hysteresis=int(cfg["hysteresis"]),
        min_agree=int(cfg["min_agree"]),
    )
    return sides, {"omni_cfg": cfg, **meta}


def _live_session_mix_sides(df: pd.DataFrame) -> np.ndarray:
    """Match live evaluate_session_mix thresholds (0.32 / 0.42)."""
    feat = _feature_frame(df)
    if feat is None:
        raise RuntimeError("feature frame failed")
    return _sides_session_mix(feat, 0.32, 0.42)


def _trades_from_sides(
    df: pd.DataFrame, sides: np.ndarray, sym: str, *, sl_atr: float, rr: float, gap: int
) -> list[dict]:
    entries = _entries(df, sides, sym, sl_atr=sl_atr, min_gap=gap)
    raw = _barrier_sim(df, entries, float(rr), sym)
    out = []
    for t in raw:
        r = float(t.get("r") or 0)
        out.append(
            {
                "i": int(t.get("i") or t.get("ei") or 0),
                "r": round(r, 4),
                "pnl": round(float(t.get("pnl") or r * RISK_USD), 2),
                "side": t.get("side"),
            }
        )
    return out


def _filter_horizon_trades(trades: list[dict], n_bars: int, days: Optional[int]) -> list[dict]:
    if days is None:
        return trades
    # approx bars: use last fraction of series by calendar share
    # For intraday, cut by index fraction matching days/years of hist
    # Caller passes pre-sliced df — trades already on that frame.
    return trades


def _load_crowns() -> dict[str, dict]:
    raw = json.loads(SCHWAB_V2.read_text(encoding="utf-8"))
    crowns = {}
    for row in raw.get("summary", {}).get("best_per_pair_1y") or []:
        sym = row["symbol"]
        crowns[sym] = {
            "symbol": sym,
            "lookbacks": row["lookbacks"],
            "size_mode": row["mode"],
            "lot_mult": row["lot"],
            "tp_r": row["tp_r"],
            "hold_days": row["hold"],
            "id": row.get("id"),
            "v2_exp_r": row.get("exp_r"),
            "v2_verdict": row.get("verdict"),
        }
    return crowns


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    clear_hist_cache()
    crowns = _load_crowns()
    _log("=== Schwab v2 crowns vs LIVE brains · Dukascopy · research ===")
    _log(f"live_book={LIVE_BOOK}")
    _log(f"started {datetime.now(timezone.utc).isoformat()}")
    t0 = time.time()

    results: dict[str, Any] = {
        "meta": {
            "risk_usd": RISK_USD,
            "live_book": LIVE_BOOK,
            "schwab_source": str(SCHWAB_V2.name),
            "nebula": "untouched",
            "live_env": "untouched",
            "started": datetime.now(timezone.utc).isoformat(),
        },
        "symbols": {},
        "head_to_head": [],
    }

    for sym in LIVE_BOOK:
        brain = symbol_entry_brain(sym)
        tfs = list(symbol_scan_tfs(sym) or ("5m",))
        rr = float(symbol_rr(sym, default=3.0))
        crown = crowns.get(sym)
        _log(f"\n######## {sym} live={brain}@{tfs} rr={rr} ########")
        sym_out: dict[str, Any] = {
            "live_brain": brain,
            "live_tfs": tfs,
            "live_rr": rr,
            "schwab_crown": crown,
            "horizons": {},
        }

        # --- Schwab daily ---
        daily = _load_daily(sym)
        if daily is None or crown is None:
            _log(f"  SCHWAB skip (daily/crown missing)")
            schwab_by_h = {}
        else:
            schwab_by_h = {}
            for hname, days in HORIZONS.items():
                frame, cut_i = _slice_daily(daily, days)
                trades = _schwab_trades(frame, {**crown, "symbol": sym}, cut_i)
                bat = _battery(trades, len(frame))
                f = bat["full"]
                _log(
                    f"  SCHWAB {hname:4} {bat['verdict']:8} n={f['n']} "
                    f"E[R]={f['exp_r']:+.3f} pnl=${f['pnl']:+.0f} "
                    f"OOS={bat['oos']['exp_r']:+.3f} WF={bat['wf_green']}/{bat['wf_n']} "
                    f"MC={bat['monte_carlo'].get('prob_profit')}%"
                )
                schwab_by_h[hname] = {
                    "verdict": bat["verdict"],
                    "full": bat["full"],
                    "oos": bat["oos"],
                    "wf_green": bat["wf_green"],
                    "wf_n": bat["wf_n"],
                    "mc": bat["monte_carlo"],
                    "tests": {
                        k: (bat["tests"].get(k) if not isinstance(bat["tests"].get(k), dict)
                            else {"n": bat["tests"][k].get("n"), "exp_r": bat["tests"][k].get("exp_r"), "pnl": bat["tests"][k].get("pnl")})
                        for k in bat["tests"]
                    },
                    "crown_id": crown.get("id"),
                }

        # --- Live brain (best TF among scan TFs by 1y E[R]) ---
        live_by_h: dict[str, Any] = {}
        live_tf_detail: dict[str, Any] = {}
        for tf in tfs:
            df0 = load_local_ohlcv(sym, tf=tf)
            if df0 is None or len(df0) < 500:
                _log(f"  LIVE {tf} no hist")
                continue
            df0 = _prep_intraday(df0)
            try:
                if brain in ("omni", "omni_horizon", "omnihorizon"):
                    if (sym, tf) not in OMNI_LIVE:
                        _log(f"  LIVE {tf} no OMNI_LIVE cfg — skip")
                        continue
                    sides, meta = _live_omni_sides(df0, sym, tf)
                    sl_atr = float(meta["omni_cfg"]["sl_atr"])
                    gap = 2
                else:
                    # session_mix (NAS)
                    sides = _live_session_mix_sides(df0)
                    sl_atr = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))
                    gap = 2
                    meta = {"brain": "session_mix", "thr": "0.32/0.42"}
            except Exception as exc:
                _log(f"  LIVE {tf} sides fail: {exc}")
                continue

            # full-hist sides once (do not reuse 1y-window sides on full df)
            sides_full = sides
            tf_horizons = {}
            for hname, days in HORIZONS.items():
                if days is None:
                    df_use = df0
                    sides_use = sides_full
                    cut_i = 0
                else:
                    end = df0.index[-1]
                    cut = end - pd.Timedelta(days=int(days))
                    warm = cut - pd.Timedelta(days=20)
                    use = df0[df0.index >= warm]
                    try:
                        if brain in ("omni", "omni_horizon", "omnihorizon"):
                            sides_use, _ = _live_omni_sides(use, sym, tf)
                        else:
                            sides_use = _live_session_mix_sides(use)
                    except Exception as exc:
                        _log(f"  LIVE {tf}/{hname} fail: {exc}")
                        continue
                    cut_i = 0
                    for i, ts in enumerate(use.index):
                        if ts >= cut:
                            cut_i = i
                            break
                    df_use = use

                # align lengths defensively
                n = min(len(df_use), len(sides_use))
                df_use = df_use.iloc[:n]
                sides_use = np.asarray(sides_use)[:n]

                trades = _trades_from_sides(
                    df_use, sides_use, sym, sl_atr=sl_atr, rr=rr, gap=gap
                )
                if cut_i > 0:
                    trades = [t for t in trades if int(t["i"]) >= cut_i]
                bat = _battery(trades, len(df_use))
                f = bat["full"]
                _log(
                    f"  LIVE  {tf}/{hname:4} {bat['verdict']:8} n={f['n']} "
                    f"E[R]={f['exp_r']:+.3f} pnl=${f['pnl']:+.0f} "
                    f"OOS={bat['oos']['exp_r']:+.3f} WF={bat['wf_green']}/{bat['wf_n']} "
                    f"MC={bat['monte_carlo'].get('prob_profit')}%"
                )
                tf_horizons[hname] = {
                    "verdict": bat["verdict"],
                    "full": bat["full"],
                    "oos": bat["oos"],
                    "wf_green": bat["wf_green"],
                    "wf_n": bat["wf_n"],
                    "mc": bat["monte_carlo"],
                    "sl_atr": sl_atr,
                    "rr": rr,
                    "meta": {k: meta[k] for k in meta if k != "usage"} if isinstance(meta, dict) else meta,
                }
            live_tf_detail[tf] = tf_horizons

        # pick best live TF on 1y exp_r
        best_tf = None
        best_1y = None
        for tf, hz in live_tf_detail.items():
            h1 = hz.get("1y")
            if not h1:
                continue
            if best_1y is None or float((h1["full"] or {}).get("exp_r") or -9) > float(
                (best_1y["full"] or {}).get("exp_r") or -9
            ):
                best_tf, best_1y = tf, h1

        for hname in HORIZONS:
            live_h = None
            if best_tf and best_tf in live_tf_detail:
                live_h = live_tf_detail[best_tf].get(hname)
            schwab_h = schwab_by_h.get(hname)
            sym_out["horizons"][hname] = {
                "schwab": schwab_h,
                "live": live_h,
                "live_tf": best_tf,
            }
            # head to head
            if schwab_h and live_h:
                se = float((schwab_h["full"] or {}).get("exp_r") or 0)
                le = float((live_h["full"] or {}).get("exp_r") or 0)
                winner = "schwab" if se > le else ("live" if le > se else "tie")
                results["head_to_head"].append(
                    {
                        "symbol": sym,
                        "horizon": hname,
                        "winner": winner,
                        "schwab_exp_r": se,
                        "live_exp_r": le,
                        "schwab_verdict": schwab_h.get("verdict"),
                        "live_verdict": live_h.get("verdict"),
                        "live_tf": best_tf,
                        "live_brain": brain,
                        "schwab_pnl": (schwab_h["full"] or {}).get("pnl"),
                        "live_pnl": (live_h["full"] or {}).get("pnl"),
                    }
                )

        sym_out["live_by_tf"] = live_tf_detail
        sym_out["live_best_tf_1y"] = best_tf
        results["symbols"][sym] = sym_out

    # summary
    h1 = [x for x in results["head_to_head"] if x["horizon"] == "1y"]
    schwab_wins = sum(1 for x in h1 if x["winner"] == "schwab")
    live_wins = sum(1 for x in h1 if x["winner"] == "live")
    ties = sum(1 for x in h1 if x["winner"] == "tie")
    results["summary"] = {
        "1y_schwab_wins": schwab_wins,
        "1y_live_wins": live_wins,
        "1y_ties": ties,
        "recommendation": (
            "SCHWAB_SLEEVE"
            if schwab_wins > live_wins
            else ("KEEP_LIVE" if live_wins > schwab_wins else "MIXED")
        ),
        "note": (
            f"On 1y E[R]: Schwab {schwab_wins} · Live {live_wins} · tie {ties}. "
            "Research only — not wired."
        ),
        "elapsed_s": round(time.time() - t0, 1),
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    results["meta"]["finished"] = results["summary"]["finished"]

    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_md(results)
    _log(f"\n=== SUMMARY === {results['summary']}")
    _log(f"wrote {OUT}")
    _log(f"wrote {MD}")


def _write_md(results: dict[str, Any]) -> None:
    lines = [
        "# Schwab v2 crowns vs live brains",
        "",
        f"- **{results['summary']['recommendation']}**: {results['summary']['note']}",
        f"- Live book: {LIVE_BOOK}",
        f"- Risk ${RISK_USD}/R | Nebula/live untouched",
        "",
        "## 1y head-to-head",
        "",
        "| Pair | Winner | Schwab E[R] / verdict / PnL | Live brain@TF E[R] / verdict / PnL |",
        "|---|---|---|---|",
    ]
    for x in results["head_to_head"]:
        if x["horizon"] != "1y":
            continue
        lines.append(
            "| {sym} | **{win}** | {ser} / {sv} / ${sp} | {lb}@{tf} {ler} / {lv} / ${lp} |".format(
                sym=x["symbol"],
                win=x["winner"],
                ser=x["schwab_exp_r"],
                sv=x["schwab_verdict"],
                sp=x["schwab_pnl"],
                lb=x["live_brain"],
                tf=x["live_tf"],
                ler=x["live_exp_r"],
                lv=x["live_verdict"],
                lp=x["live_pnl"],
            )
        )
    lines += ["", "## Full horizon", ""]
    for x in results["head_to_head"]:
        if x["horizon"] != "full":
            continue
        lines.append(
            "- **{sym}**: winner={win} schwab E[R]={ser} ({sv}) | live E[R]={ler} ({lv}) @{tf}".format(
                sym=x["symbol"],
                win=x["winner"],
                ser=x["schwab_exp_r"],
                sv=x["schwab_verdict"],
                ler=x["live_exp_r"],
                lv=x["live_verdict"],
                tf=x["live_tf"],
            )
        )
    MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
