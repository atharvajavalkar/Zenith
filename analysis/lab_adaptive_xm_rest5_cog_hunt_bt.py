#!/usr/bin/env python3
"""REST-5 COGNITIVE deep hunt — all deeper analytical / dynamic brains.

Book: AUDUSD · USDCHF · ETHUSD · NAS100 · USDJPY
Live untouched: GOLD · BTCUSD · UKOIL

Catalog layers:
  1. Zenith invented-math formulas (OU, Kalman, Higuchi, Allan, CUSUM, …)
  2. Cognitive routers: Bayesian edge meta, HMM-ish trend/fade book,
     multi-math omni vote, information-flow gates
  3. Cross-asset (USDJPY↔GOLD corr / opp) where hist allows

Scores by n_green(7d→1y) first. XM session spreads. Resume-safe.
Does NOT wire live.

Outputs:
  data/lab_adaptive_xm_rest5_cog_hunt.{json,progress.json,partial.json,log}
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.lab_adaptive_minute_day_v3_bt import _calendar  # noqa: E402
from analysis.lab_adaptive_regime_router_bt import (  # noqa: E402
    _classify_regime,
    _sides_kalman,
    _sides_snr,
    _static_sides_for_winner,
    _window_trades,
)
from analysis.lab_adaptive_regime_v2_bt import _family, _gated_sides, _switch_sides  # noqa: E402
from analysis.lab_adaptive_xm_deep_hunt_bt import (  # noqa: E402
    GAP_GRID_BY_TF,
    RISK_USD,
    RR_GRID,
    SL_GRID_BY_TF,
    TF_YEARS,
    WINDOWS,
    _entries,
    _load,
    _simulate,
    _summarize,
    mk_circadian,
    mk_entropy,
    mk_rolling_edge_gate,
    mk_session_mix,
)
from analysis.lab_adaptive_xm_rest5_hunt_bt import (  # noqa: E402
    FX_SYMS,
    REST_BOOK,
    SL_GRID_FX,
    _cell_score,
    _n_green,
    _top_row,
    mk_dual_agree,
    mk_off_flat,
    mk_omni_pool,
)
from analysis.lab_omnihorizon_adaptive_bt import _omni_route  # noqa: E402
from analysis.lab_zenith_math_invention_deep_bt import _catalog as _zenith_catalog  # noqa: E402
from strategy.adaptive_v3_brains import SL_ATR_BY_SYMBOL  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_adaptive_xm_rest5_cog_hunt.json"
LOG = Path(ROOT) / "data" / "lab_adaptive_xm_rest5_cog_hunt.log"
PROGRESS = Path(ROOT) / "data" / "lab_adaptive_xm_rest5_cog_hunt.progress.json"
PARTIAL = Path(ROOT) / "data" / "lab_adaptive_xm_rest5_cog_hunt.partial.json"
HUNT = Path(ROOT) / "data" / "lab_zenith_math_invention_deep.json"

# Full hist TFs + lean 1m
TFS = ("5m", "15m", "3m", "1m")

SideFn = Callable[[pd.DataFrame, dict[str, Any]], np.ndarray]


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def _progress(p: dict) -> None:
    p = dict(p)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    p["hunt"] = "rest5_cog"
    p["book"] = list(REST_BOOK)
    PROGRESS.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")


def _save_partial(payload: dict) -> None:
    PARTIAL.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _wrap_zenith(fid: str, fn) -> SideFn:
    def side_fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sides, _eq = fn(df)
        return np.asarray(sides, dtype=int)

    side_fn.__name__ = f"zenith_{fid}"
    return side_fn


def mk_bayes_edge_meta(pool: dict[str, SideFn], look: int = 400, prior: float = 0.52) -> SideFn:
    """Online Bayesian-ish: posterior = rolling hit-rate; trade best brain if > prior."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        ret = df["close"].pct_change().fillna(0).to_numpy(dtype=float)
        nxt = np.roll(ret, -1)
        nxt[-1] = 0.0
        names = list(pool.keys())
        mats = []
        side_map = {}
        for nm, sfn in pool.items():
            s = np.asarray(sfn(df, ctx), dtype=int)
            side_map[nm] = s
            hit = np.full(len(df), np.nan)
            m = s != 0
            hit[m] = ((np.sign(s[m]) * np.sign(nxt[m])) > 0).astype(float)
            post = (
                pd.Series(hit)
                .rolling(look, min_periods=max(40, look // 5))
                .mean()
                .fillna(0.5)
                .to_numpy()
            )
            mats.append(post)
        edge = np.column_stack(mats)
        best = np.argmax(edge, axis=1)
        out = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            j = int(best[i])
            if edge[i, j] < prior:
                continue
            out[i] = int(side_map[names[j]][i])
        return out

    return fn


def mk_hmm_trend_fade() -> SideFn:
    """2-state proxy: regime + SNR → trend book vs Kalman fade book."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        reg = ctx["regime"]
        snr = df["snr"].to_numpy(dtype=float)
        trend = ctx["static"].copy()
        fade = _sides_kalman(df, 2.0)
        # soft HMM emission: high SNR + trend regime → trend; else fade
        p_trend = 0.55 * (reg == 1).astype(float) + 0.45 * np.clip(snr / 0.8, 0, 1)
        out = np.zeros(len(df), dtype=int)
        use_t = p_trend >= 0.55
        out[use_t] = trend[use_t]
        out[~use_t] = fade[~use_t]
        hour = ctx["cal"]["hour"]
        out[hour >= 21] = 0
        return out

    return fn


def mk_info_flow_gate(base: SideFn, *, pe_max: float = 0.78, lf_min: float = 0.55) -> SideFn:
    """Only take base when information is ordered (low perm-ent) + LF dominant."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        sides = base(df, ctx)
        pe = df["perm_ent"].to_numpy() if "perm_ent" in df.columns else np.full(len(df), 0.5)
        lf = df["fourier_lf"].to_numpy() if "fourier_lf" in df.columns else np.ones(len(df))
        out = sides.copy()
        out[(pe > pe_max) | (lf < lf_min)] = 0
        return out

    return fn


def mk_math_omni_vote(look_fast: int = 100, edge_min: float = 0.52, min_agree: int = 2) -> SideFn:
    """Omni-route over a diverse zenith math pool."""

    zcat = {fid: fn for fid, _lab, fn in _zenith_catalog()}
    pick = [
        "snr_drift_0.40",
        "hurst_trend_0.52",
        "ou_snap_5_40",
        "kalman_fade_2.0",
        "perm_order_0.80",
        "cusum_0.015",
        "spec_ent_0.50",
        "bimod_0.55",
    ]
    pick = [p for p in pick if p in zcat]

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        brain_sides = {}
        for pid in pick:
            s, _ = zcat[pid](df)
            brain_sides[pid] = np.asarray(s, dtype=int)
        # also add session_mix / circadian as cognitive anchors
        brain_sides["session_mix"] = mk_session_mix(0.28, 0.38)(df, ctx)
        brain_sides["circadian"] = mk_circadian(2.0, 0.6)(df, ctx)
        ret = df["close"].pct_change().fillna(0).to_numpy(dtype=float)
        sides, _meta = _omni_route(
            brain_sides,
            ret,
            look_fast=look_fast,
            look_mid=look_fast * 2,
            look_slow=look_fast * 5,
            edge_min=edge_min,
            hysteresis=20,
            min_agree=min_agree,
        )
        return sides

    return fn


def mk_usdjpy_gold_corr(*, look: int = 120, thr: float = 0.15) -> SideFn:
    """USDJPY fades/trends only when rolling corr vs GOLD agrees with book."""

    def fn(df: pd.DataFrame, ctx: dict) -> np.ndarray:
        g = ctx.get("gold_close")
        if g is None or len(g) != len(df):
            return np.zeros(len(df), dtype=int)
        jpy = df["close"].pct_change()
        gold = pd.Series(g, index=df.index).pct_change()
        corr = jpy.rolling(look, min_periods=look // 2).corr(gold).fillna(0).to_numpy()
        # classic: risk-off gold up → JPY often bid (USDJPY down) → sell USDJPY when corr strongly neg with gold up
        kz = df["kalman_z"].to_numpy()
        ds = df["drift_sign"].to_numpy()
        out = np.zeros(len(df), dtype=int)
        # fade when |corr| high and kalman extreme
        m = np.abs(corr) >= thr
        out[m & (kz <= -1.5)] = 1
        out[m & (kz >= 1.5)] = -1
        # trend when corr mild and SNR drift
        snr = _sides_snr(df, 0.40)
        mild = np.abs(corr) < thr
        out[mild] = snr[mild]
        hour = ctx["cal"]["hour"]
        out[(hour < 7) | (hour >= 21)] = 0
        return out

    return fn


def _build_cog_variants(*, lean: bool = False) -> list[tuple[str, str, SideFn]]:
    out: list[tuple[str, str, SideFn]] = []

    # ── Zenith invented math (all) ──────────────────────────────────────────
    for fid, lab, fn in _zenith_catalog():
        out.append((f"math_{fid}", f"zenith:{lab}", _wrap_zenith(fid, fn)))
        if lean and len(out) >= 18:
            break

    # ── Cognitive routers ───────────────────────────────────────────────────
    sm = mk_session_mix(0.28, 0.38)
    cir = mk_circadian(2.0, 0.6)
    ent = mk_entropy(0.72, 0.22)
    out.append(("cog_hmm_trend_fade", "hmm-ish trend/fade", mk_hmm_trend_fade()))
    out.append(
        (
            "cog_bayes_sm_cir_ent",
            "bayes edge meta",
            mk_bayes_edge_meta({"sm": sm, "cir": cir, "ent": ent}, look=400, prior=0.52),
        )
    )
    out.append(("cog_info_sm", "info-flow gate", mk_info_flow_gate(sm)))
    out.append(("cog_info_cir", "info-flow gate", mk_info_flow_gate(cir, pe_max=0.82, lf_min=0.50)))
    out.append(("cog_agree_sm_cir", "dual agree", mk_dual_agree(sm, cir)))
    out.append(("cog_offflat_bayes", "offflat+bayes", mk_off_flat(
        mk_bayes_edge_meta({"sm": sm, "cir": cir, "snr": mk_entropy(0.72, 0.0)}, 350, 0.50)
    )))
    out.append(("cog_math_omni_f100", "math omni vote", mk_math_omni_vote(100, 0.52, 2)))
    if not lean:
        out.append(("cog_math_omni_f80", "math omni vote", mk_math_omni_vote(80, 0.50, 2)))
        out.append(("cog_math_omni_f120", "math omni vote", mk_math_omni_vote(120, 0.55, 2)))
        out.append(
            (
                "cog_omni_wide",
                "omni wide pool",
                mk_omni_pool(
                    look_fast=100,
                    look_mid=240,
                    look_slow=600,
                    edge_min=0.52,
                    hysteresis=25,
                    min_agree=2,
                    pool="wide",
                ),
            )
        )
        out.append(
            (
                "cog_edge_hmm",
                "edgegated hmm",
                mk_rolling_edge_gate(mk_hmm_trend_fade(), 500, 0.02),
            )
        )
        # bayes over a math subset
        z = {fid: _wrap_zenith(fid, fn) for fid, _l, fn in _zenith_catalog()}
        math_pool = {
            k: z[k]
            for k in (
                "kalman_fade_2.0",
                "ou_snap_5_40",
                "snr_drift_0.40",
                "perm_order_0.80",
                "cusum_0.015",
            )
            if k in z
        }
        if math_pool:
            out.append(
                (
                    "cog_bayes_math5",
                    "bayes over math5",
                    mk_bayes_edge_meta(math_pool, look=450, prior=0.53),
                )
            )

    # Cross-asset JPY
    out.append(("cog_usdjpy_gold_corr", "USDJPY↔GOLD corr", mk_usdjpy_gold_corr()))

    return out


def _load_gold_aligned(df: pd.DataFrame, tf: str, years: float) -> Optional[np.ndarray]:
    try:
        g = _load("GOLD", tf, years)
    except Exception:
        return None
    # align on intersection index
    joined = df[["close"]].join(g[["close"]].rename(columns={"close": "g"}), how="left")
    # ffill limited
    joined["g"] = joined["g"].ffill()
    if joined["g"].isna().mean() > 0.3:
        return None
    return joined["g"].ffill().bfill().to_numpy(dtype=float)


def main() -> int:
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    t0 = time.time()
    hunt = json.loads(HUNT.read_text(encoding="utf-8")) if HUNT.exists() else {}
    per_hunt = hunt.get("per_pair") or {}
    variants_full = _build_cog_variants(lean=False)
    variants_lean = _build_cog_variants(lean=True)

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

    est = 0
    for tf in TFS:
        nv = len(variants_lean if tf == "1m" else variants_full) + 3
        for sym in REST_BOOK:
            slg = SL_GRID_FX[tf] if sym in FX_SYMS else SL_GRID_BY_TF[tf]
            est += nv * len(slg) * len(RR_GRID) * len(GAP_GRID_BY_TF[tf])

    _log(
        f"=== REST-5 COGNITIVE deep hunt ===\n"
        f"book={REST_BOOK}\n"
        f"live untouched: GOLD BTCUSD UKOIL\n"
        f"layers: zenith-math + bayes/HMM/info/omni-math + JPY↔GOLD\n"
        f"full_variants={len(variants_full)} lean_1m={len(variants_lean)}\n"
        f"est_cells≈{est:,} · n_green-first · XM spreads ON\n"
    )
    _progress({"status": "running", "pct": 0, "cells_est": est, "modes": len(variants_full)})

    cells_done = 0
    for pi, sym in enumerate(REST_BOOK):
        wmeta = (per_hunt.get(sym) or {}).get("winner") or {}
        wid = str(wmeta.get("id") or "")
        fam = _family(wid)
        per_pair.setdefault(
            sym,
            {"symbol": sym, "static_id": wid, "family": fam, "by_tf": {}, "best_overall": None},
        )

        for tf in TFS:
            key = f"{sym}:{tf}"
            if key in done_keys:
                _log(f"skip {key} (resume)")
                continue
            years = TF_YEARS[tf]
            _log(f"\n######## COG {sym} @ {tf} · {years}y ########")
            _progress(
                {
                    "status": "running",
                    "pct": round(
                        100 * (pi * len(TFS) + list(TFS).index(tf)) / (len(REST_BOOK) * len(TFS)),
                        1,
                    ),
                    "pair": sym,
                    "tf": tf,
                    "cells_done": cells_done,
                    "cells_est": est,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "modes": len(variants_lean if tf == "1m" else variants_full),
                }
            )
            try:
                df = _load(sym, tf, years)
            except Exception as exc:
                _log(f"  LOAD FAIL: {exc}")
                per_pair[sym]["by_tf"][tf] = {"error": str(exc), "done": True}
                _save_partial({"per_pair": per_pair, "elapsed_sec": round(time.time() - t0, 1)})
                continue

            static, label, _ = _static_sides_for_winner(df, wid)
            regime = _classify_regime(df)
            cal = _calendar(df)
            gold_close = None
            if sym == "USDJPY":
                gold_close = _load_gold_aligned(df, tf, years)
            ctx = {
                "sym": sym,
                "static": static,
                "regime": regime,
                "cal": cal,
                "family": fam,
                "label": label,
                "gold_close": gold_close,
            }
            live_sl = float(SL_ATR_BY_SYMBOL.get(sym, 2.0))
            variants = variants_lean if tf == "1m" else variants_full
            alt_trend = _sides_snr(df, 0.40)
            alt_fade = _sides_kalman(df, 2.2)
            mode_ids: list[tuple[str, str, Any]] = [
                ("static", "static", static),
                ("gated", "gated", _gated_sides(static, regime, fam)),
                ("switch", "switch", _switch_sides(static, regime, fam, alt_trend, alt_fade)),
            ]
            for vid, vlab, vfn in variants:
                # skip JPY-gold brain on non-JPY
                if vid == "cog_usdjpy_gold_corr" and sym != "USDJPY":
                    continue
                try:
                    mode_ids.append((vid, vlab, vfn(df, ctx)))
                except Exception as exc:
                    _log(f"  sides {vid} fail: {exc}")

            sl_grid = SL_GRID_FX[tf] if sym in FX_SYMS else SL_GRID_BY_TF[tf]
            gap_grid = GAP_GRID_BY_TF[tf]
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
                            "mode_family": (
                                "zenith_math"
                                if str(mode).startswith("math_")
                                else str(mode).split("_")[0]
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
                                / (len(REST_BOOK) * len(TFS)),
                                1,
                            ),
                        }
                    )

            results.sort(key=lambda c: c["_score"], reverse=True)
            top = [_top_row(c) for c in results[:30]]
            best = top[0] if top else None
            all_six = [t for t in top if t.get("all_six")]
            per_pair[sym]["by_tf"][tf] = {
                "done": True,
                "n_bars": len(df),
                "span": [str(df.index[0]), str(df.index[-1])],
                "n_combos": len(results),
                "best": best,
                "top25": top[:25],
                "all_six": all_six[:10],
                "live_sl_ref": live_sl,
            }
            cand = best
            cur = per_pair[sym].get("best_overall")
            if cand and (
                cur is None
                or (cand.get("n_green") or 0) > (cur.get("n_green") or 0)
                or (
                    (cand.get("n_green") or 0) == (cur.get("n_green") or 0)
                    and float(cand.get("pnl_1y") or 0) > float(cur.get("pnl_1y") or 0)
                )
            ):
                per_pair[sym]["best_overall"] = {**cand, "tf": tf}

            _log(
                f"  BEST {tf}: {best['id'] if best else None} "
                f"green={best.get('n_green') if best else 0}/6 "
                f"1y ${best['pnl_1y'] if best else 0:,.0f} "
                f"all_six_in_top={len(all_six)}"
            )
            _save_partial(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "cells_done": cells_done,
                    "per_pair": per_pair,
                    "hunt": "rest5_cog",
                }
            )

    survivors = []
    recs = []
    for sym in REST_BOOK:
        row = per_pair.get(sym) or {}
        b = row.get("best_overall")
        if b:
            recs.append({"symbol": sym, **b})
        for tf, block in (row.get("by_tf") or {}).items():
            for a in block.get("all_six") or []:
                survivors.append({"symbol": sym, "tf": tf, **a})
    survivors.sort(key=lambda x: (-int(x.get("n_green") or 0), -float(x.get("pnl_1y") or 0)))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "adaptive_xm_rest5_cog_hunt",
        "mode": "adaptive_xm_rest5_cog_hunt",
        "research_only": True,
        "recommend_live": bool(survivors),
        "risk_usd": RISK_USD,
        "elapsed_sec": round(time.time() - t0, 1),
        "cells_done": cells_done,
        "tfs": list(TFS),
        "book": list(REST_BOOK),
        "live_untouched": ["GOLD", "BTCUSD", "UKOIL"],
        "n_variants": len(variants_full),
        "windows": [w for w, _ in WINDOWS],
        "survivors_all_six": survivors[:30],
        "recommendations": recs,
        "per_pair": per_pair,
        "note": (
            "Cognitive deep hunt — zenith math + bayes/HMM/info/math-omni + "
            "USDJPY↔GOLD. Live GOLD/BTC/UKOIL untouched."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _save_partial(payload)
    _progress(
        {
            "status": "done",
            "pct": 100,
            "cells_done": cells_done,
            "cells_est": est,
            "elapsed_sec": round(time.time() - t0, 1),
            "survivors": len(survivors),
        }
    )
    _log(f"\n=== COG DONE · {time.time()-t0:.0f}s · cells={cells_done} · survivors={len(survivors)} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
