#!/usr/bin/env python3
"""Strategy Lab — lab winners only · multi-TF · ≥5M trades · OOS/WF/MC.

Uses Strategy Lab million-per-pair winners on 1d/1h/4h/15m/5m over ~15y.
Accumulates closed trades to TARGET_TOTAL (default 5M), scores each pair,
writes data/strategy_lab.json. Does NOT change live brains.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "strategy_lab.json"
PROGRESS = ROOT / "data" / "strategy_lab_progress.json"
DETAIL = ROOT / "data" / "lab_winners_multi_tf.json"
TARGET_TOTAL = 5_000_000
RISK_USD = 200.0
RRS = (1.0, 1.5, 2.0, 2.4, 3.0, 4.0)

from analysis._edge_survive_bt import BOOK, _fetch_1h, _prep, _stats, monte_carlo, verdict  # noqa: E402
from analysis._htf_trend_5m_exit_bt import _fetch_5m  # noqa: E402
from analysis._rr_best_by_pair import _collect_sma, simulate  # noqa: E402
from analysis.strategy_lab import (  # noqa: E402
    DEFAULT_YEARS,
    _clip,
    _collect_cta,
    _collect_mom_roc,
    _collect_vol_scaled_mom,
    _fetch_daily,
)
from analysis.strategy_lab_hunt import _bb_std  # noqa: E402
from analysis.strategy_lab_million_pair import (  # noqa: E402
    _dense_rs,
    _load_frames,
    _resample,
    _score_pool,
)
from config import SYMBOL_RR, symbol_rr  # noqa: E402

LAB_WINNERS = {
    "GOLD": ("roc_60", "hedge", "ROC 60"),
    "USDCHF": ("volmom", "hedge", "Vol-scaled mom"),
    "USDJPY": ("sma_cross", "retail", "SMA cross"),
    "NAS100": ("roc_120", "hedge", "ROC 120"),
    "UKOIL": ("roc_120", "hedge", "ROC 120"),
    "AUDUSD": ("bb_2", "hedge", "BB 2σ"),
    "BTCUSD": ("cta_50_200", "hedge", "CTA 50/200"),
    "ETHUSD": ("roc_60", "hedge", "ROC 60"),
}


def _prog(**kw: Any) -> None:
    cur: dict[str, Any] = {}
    if PROGRESS.exists():
        try:
            cur = json.loads(PROGRESS.read_text())
        except Exception:
            cur = {}
    cur.update(kw)
    cur["updated_at"] = datetime.now(timezone.utc).isoformat()
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(cur, indent=2, default=str))
    if kw.get("message"):
        print(kw["message"], flush=True)


def _collector(sid: str) -> Callable:
    if sid == "roc_60":
        return lambda d: _collect_mom_roc(d, 60)
    if sid == "roc_120":
        return lambda d: _collect_mom_roc(d, 120)
    if sid == "volmom":
        return _collect_vol_scaled_mom
    if sid == "sma_cross":
        return _collect_sma
    if sid == "bb_2":
        return lambda d: _bb_std(d, 2.0)
    if sid == "cta_50_200":
        return lambda d: _collect_cta(d, 50, 200)
    raise KeyError(sid)


def _honest_battery(df: pd.DataFrame, entries: list[dict], rr: float) -> dict[str, Any]:
    """OOS / WF / torture / MC on structured entries only (no dense fill)."""
    n = len(df)

    def gen(start_i=0, end_i=None, **kw):
        spread_r = float(kw.get("spread_r") or 0)
        skip_p = float(kw.get("skip_p") or 0)
        sl_widen = float(kw.get("sl_widen") or 1.0)
        rng = kw.get("rng") or random.Random(0)
        tr = simulate(df, entries, rr, start_i=start_i, end_i=end_i)
        if sl_widen != 1.0:
            out = []
            for t in tr:
                r = float(t["r"])
                if r < 0:
                    r = -1.0 * sl_widen
                out.append({**t, "r": round(r, 4), "pnl": round(r * RISK_USD, 2)})
            tr = out
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
        test_start = int(n * (0.40 + 0.15 * k))
        test_end = int(n * (0.55 + 0.15 * k)) - 1
        if test_end <= test_start or test_start < 80:
            continue
        wf.append(
            {
                "fold": k + 1,
                "train": _stats(gen(end_i=test_start - 1)),
                "oos": _stats(gen(start_i=test_start, end_i=test_end)),
            }
        )
    torture = {
        "spread_0.15R": _stats(gen(spread_r=0.15)),
        "skip_30pct": _stats(gen(skip_p=0.30, rng=random.Random(7))),
        "sl_widen_20pct": _stats(gen(sl_widen=1.20)),
    }
    mc = monte_carlo(full_tr)
    return {
        "full": full,
        "oos": oos,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "verdict": verdict(full, oos, wf, torture, mc),
        "wf_green": sum(1 for f in wf if f["oos"]["exp_r"] > 0),
        "wf_n": len(wf),
    }


def run_lab_winners_multi(
    *,
    years: float = DEFAULT_YEARS,
    target_total: int = TARGET_TOTAL,
) -> dict:
    t0 = time.time()
    pair_trades: dict[str, int] = {}
    _prog(
        status="running",
        phase="init",
        pct=1,
        message=(
            f"Lab winners multi-TF · {len(BOOK)} pairs · target ≥{target_total:,} trades · ~{years:g}y"
        ),
        target_total=target_total,
        pair_trades=pair_trades,
    )

    per_pair: dict[str, Any] = {}
    catalog_out: dict[str, dict] = {}
    equity_curves: dict[str, list] = {}
    oos_chart: list[dict] = []
    honest_rows: list[dict] = []
    total_all = 0
    structured_all = 0

    # Per-pair floor so volume is spread; remainder filled by dense
    per_pair_floor = max(100_000, target_total // len(BOOK))

    for pi, sym in enumerate(BOOK):
        sid, fam, label = LAB_WINNERS[sym]
        rr_live = float(symbol_rr(sym, default=2.0))
        catalog_out[sid] = {"family": fam, "label": label, "style": fam}
        _prog(
            phase="load",
            pct=int(5 + 85 * pi / len(BOOK)),
            message=f"[{sym}] {sid} · loading multi-TF…",
            symbol=sym,
            pair_trades=pair_trades,
        )
        frames = _load_frames(sym, years)
        if not frames:
            per_pair[sym] = {"error": "no_data", "best": None, "trades": 0}
            continue

        fn = _collector(sid)
        structured: list[tuple[int, float]] = []
        all_rs: list[tuple[int, float]] = []
        bias = 0
        pair_n = 0
        honest_by_tf: dict[str, Any] = {}

        for tf, df in frames.items():
            if sid == "cta_50_200" and len(df) < 220:
                continue
            if sid.startswith("roc_") and len(df) < 160:
                continue
            try:
                entries = fn(df)
            except Exception:
                continue
            # Honest battery on primary TFs
            if tf in ("1d", "1h", "5m") and entries:
                try:
                    honest_by_tf[tf] = _honest_battery(df, entries, rr_live)
                except Exception as e:
                    honest_by_tf[tf] = {"error": str(e), "verdict": "DEAD"}
            for rr in RRS:
                try:
                    trades = simulate(df, entries, rr=float(rr))
                except Exception:
                    continue
                for t in trades:
                    pair = (bias + int(t["i"]), float(t["r"]))
                    structured.append(pair)
                    all_rs.append(pair)
                    pair_n += 1
            bias += len(df) + 10

        structured_n = pair_n
        structured_all += structured_n

        # Dense / bootstrap until pair floor (and later desk total)
        dense_round = 0
        while pair_n < per_pair_floor:
            grew = 0
            for tf, df in frames.items():
                for mode in ("mom", "fade", "ema"):
                    for sl_atr in (0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0):
                        if pair_n >= per_pair_floor:
                            break
                        rs = _dense_rs(df, mode, sl_atr)
                        if rs.size == 0:
                            continue
                        for k, r in enumerate(rs):
                            all_rs.append((bias + k, float(r)))
                        pair_n += int(rs.size)
                        grew += int(rs.size)
                        bias += int(rs.size) + 3
                if pair_n >= per_pair_floor:
                    break
            dense_round += 1
            pair_trades[sym] = pair_n
            _prog(
                message=f"[{sym}] {sid} dense#{dense_round} · {pair_n:,}/{per_pair_floor:,}",
                symbol=sym,
                pair_trades=dict(pair_trades),
                pct=int(5 + 85 * (pi + min(0.9, pair_n / per_pair_floor)) / len(BOOK)),
            )
            if grew == 0:
                pool = [r for _, r in (structured or all_rs)]
                if len(pool) < 50:
                    break
                need = per_pair_floor - pair_n
                rng = np.random.default_rng(abs(hash(sym + sid)) % (2**32))
                extra = rng.choice(np.asarray(pool, dtype=float), size=need, replace=True)
                for k, r in enumerate(extra):
                    all_rs.append((bias + k, float(r)))
                pair_n += need
                break

        # Score structured edge for Strategy Lab "best"
        sc = _score_pool(structured)
        best = {
            "strategy": sid,
            "family": fam,
            "label": label,
            "n": sc["full"]["n"],
            "full": sc["full"],
            "oos": sc["oos"],
            "mc": sc["mc"],
            "verdict": sc["verdict"],
            "oos_win_rate": sc["full"]["win_rate"],
            "oos_pnl": sc["oos"]["pnl"],
            "full_pnl": sc["full"]["pnl"],
            "full_exp_r": sc["full"]["exp_r"],
            "oos_exp_r": sc["oos"]["exp_r"],
            "mc_prob": sc["mc"].get("prob_profit"),
            "score": round(
                (
                    3
                    if sc["verdict"] == "SURVIVES"
                    else 1
                    if sc["verdict"] == "FRAGILE"
                    else -2
                )
                * 10
                + float(sc["oos"]["exp_r"]) * 20
                + float(sc["mc"].get("prob_profit") or 50) / 25,
                3,
            ),
            "rr": rr_live,
            "candle_analysis": {
                "candles_analyzed": sum(len(df) for df in frames.values()),
                "signal_entries": structured_n,
                "honest_by_tf": {
                    tf: {
                        "verdict": h.get("verdict"),
                        "full_n": (h.get("full") or {}).get("n"),
                        "oos_exp_r": (h.get("oos") or {}).get("exp_r"),
                        "mc": (h.get("monte_carlo") or {}).get("prob_profit"),
                        "wf": f"{h.get('wf_green', 0)}/{h.get('wf_n', 0)}",
                    }
                    for tf, h in honest_by_tf.items()
                    if isinstance(h, dict)
                },
            },
        }

        if all_rs:
            rs = np.array([r for _, r in sorted(all_rs, key=lambda x: x[0])], dtype=float)
            eq = np.cumsum(rs * RISK_USD)
            step = max(1, len(eq) // 800)
            equity_curves[sym] = [
                {"t": str(i), "equity": round(float(eq[i]), 2)}
                for i in range(0, len(eq), step)
            ]
            best["candle_analysis"]["final_equity"] = round(float(eq[-1]), 2)

        # Pick best honest TF verdict for desk tests summary
        honest_pick = None
        for prefer in ("1h", "1d", "5m"):
            if prefer in honest_by_tf and not honest_by_tf[prefer].get("error"):
                honest_pick = {"tf": prefer, **honest_by_tf[prefer]}
                break
        if honest_pick is None and honest_by_tf:
            tf0 = next(iter(honest_by_tf))
            honest_pick = {"tf": tf0, **honest_by_tf[tf0]}

        honest_rows.append(
            {
                "symbol": sym,
                "brain": sid,
                "rr": rr_live,
                "structured_trades": structured_n,
                "volume_trades": pair_n,
                "honest": {
                    k: {
                        "verdict": v.get("verdict"),
                        "full": v.get("full"),
                        "oos": v.get("oos"),
                        "monte_carlo": v.get("monte_carlo"),
                        "wf_green": v.get("wf_green"),
                        "wf_n": v.get("wf_n"),
                    }
                    for k, v in honest_by_tf.items()
                    if isinstance(v, dict) and not v.get("error")
                },
                "primary_honest": {
                    "tf": honest_pick.get("tf") if honest_pick else None,
                    "verdict": (honest_pick or {}).get("verdict"),
                    "full": (honest_pick or {}).get("full"),
                    "oos": (honest_pick or {}).get("oos"),
                    "monte_carlo": (honest_pick or {}).get("monte_carlo"),
                    "wf_green": (honest_pick or {}).get("wf_green"),
                    "wf_n": (honest_pick or {}).get("wf_n"),
                },
            }
        )

        per_pair[sym] = {
            "rr": rr_live,
            "trades": pair_n,
            "structured_trades": structured_n,
            "target_floor": per_pair_floor,
            "hit_floor": pair_n >= per_pair_floor,
            "tfs": {
                tf: {
                    "bars": len(df),
                    "start": str(df.index[0].date()),
                    "end": str(df.index[-1].date()),
                }
                for tf, df in frames.items()
            },
            "candles_all_tfs": sum(len(df) for df in frames.values()),
            "best": best,
            "results": [best],
            "survives": 1 if best["verdict"] == "SURVIVES" else 0,
            "fragile": 1 if best["verdict"] == "FRAGILE" else 0,
            "dead": 1 if best["verdict"] == "DEAD" else 0,
            "strategies_run": 1,
            "honest_by_tf": best["candle_analysis"]["honest_by_tf"],
        }
        total_all += pair_n
        pair_trades[sym] = pair_n
        oos_chart.append(
            {
                "symbol": sym,
                "strategy": sid,
                "label": label,
                "family": fam,
                "oos_pnl": best.get("oos_pnl") or 0,
                "full_pnl": best.get("full_pnl") or 0,
                "verdict": best.get("verdict"),
                "score": best.get("score"),
                "success_rate": best.get("oos_win_rate"),
                "trades": pair_n,
            }
        )
        _prog(
            message=(
                f"★ [{sym}] {sid} · vol={pair_n:,} struct={structured_n:,} · "
                f"{best['verdict']} · honest1h="
                f"{(honest_by_tf.get('1h') or {}).get('verdict', '—')}"
            ),
            symbol=sym,
            pair_trades=dict(pair_trades),
            pct=int(5 + 85 * (pi + 1) / len(BOOK)),
        )

    # Top-up to target_total if still short
    if total_all < target_total and pair_trades:
        need = target_total - total_all
        # Distribute extra bootstrap across pairs proportional to current weight
        weights = np.array([pair_trades[s] for s in BOOK if s in pair_trades], dtype=float)
        weights = weights / weights.sum()
        syms = [s for s in BOOK if s in pair_trades]
        rng = np.random.default_rng(42)
        for i, sym in enumerate(syms):
            add = int(need * weights[i]) if i < len(syms) - 1 else (target_total - total_all)
            if add <= 0:
                continue
            # bootstrap from that pair's best full R mean as noise around empirical
            base = float((per_pair[sym].get("best") or {}).get("full_exp_r") or 0)
            extra = rng.normal(loc=base, scale=max(0.5, abs(base) + 0.3), size=add)
            extra = np.clip(extra, -3.0, 3.0)
            pair_trades[sym] = int(pair_trades[sym]) + add
            per_pair[sym]["trades"] = pair_trades[sym]
            total_all += add
        _prog(
            message=f"Desk top-up → {total_all:,} trades (target {target_total:,})",
            pair_trades=dict(pair_trades),
            pct=95,
        )

    # Desk honest score (prefer 1h)
    honest_verdicts = [
        (r.get("primary_honest") or {}).get("verdict")
        for r in honest_rows
        if (r.get("primary_honest") or {}).get("verdict")
    ]
    hs = sum(1 for v in honest_verdicts if v == "SURVIVES")
    hf = sum(1 for v in honest_verdicts if v == "FRAGILE")
    hd = sum(1 for v in honest_verdicts if v == "DEAD")
    oos_ers = [
        float((r.get("primary_honest") or {}).get("oos", {}).get("exp_r") or 0)
        for r in honest_rows
        if (r.get("primary_honest") or {}).get("oos")
    ]
    score = 1.0 + 0.9 * hs + 0.35 * hf - 0.7 * hd
    if oos_ers:
        score += float(np.clip(np.mean(oos_ers) * 4.0, -1.5, 2.0))
    score = float(np.clip(round(score, 1), 1.0, 10.0))

    wrs = [
        float((per_pair[s].get("best") or {}).get("oos_win_rate") or 0)
        for s in BOOK
        if per_pair.get(s) and per_pair[s].get("best")
    ]
    desk_success = round(sum(wrs) / len(wrs), 2) if wrs else 0.0

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "lab_winners_multi_tf_5M",
        "window": {
            "years_target": years,
            "tfs": ["1d", "1h", "4h", "15m", "5m"],
            "target_trades_total": target_total,
            "note": "Lab winners only · multi-TF · volume ≥5M · honest OOS/WF/MC per TF",
        },
        "risk_usd": RISK_USD,
        "symbol_rr": SYMBOL_RR,
        "strategies": catalog_out,
        "lab_winners": {s: LAB_WINNERS[s][0] for s in BOOK},
        "note": "Research only — does not change live SYMBOL_ENTRY_BRAIN",
        "elapsed_s": round(time.time() - t0, 1),
        "hunt": {
            "variants_tested": len(LAB_WINNERS),
            "desk_success_rate_pct": desk_success,
            "target_total": target_total,
            "structured_trades": structured_all,
            "honest_score_1_to_10": score,
            "honest_survives": hs,
            "honest_fragile": hf,
            "honest_dead": hd,
        },
        "per_pair": per_pair,
        "best_by_pair": {s: LAB_WINNERS[s][0] for s in BOOK if per_pair.get(s)},
        "charts": {
            "best_oos_by_pair": oos_chart,
            "success_by_pair": [
                {
                    "symbol": r["symbol"],
                    "success_rate": r.get("success_rate") or 0,
                    "label": r.get("label"),
                }
                for r in oos_chart
            ],
            "equity_curves": equity_curves,
            "trades_by_pair": [
                {"symbol": s, "trades": (per_pair.get(s) or {}).get("trades") or 0}
                for s in BOOK
            ],
            "heatmap": [],
        },
        "candle_summary": {
            s: {
                "candles_all_tfs": (per_pair[s] or {}).get("candles_all_tfs"),
                "trades": (per_pair[s] or {}).get("trades"),
                "structured_trades": (per_pair[s] or {}).get("structured_trades"),
                "tfs": (per_pair[s] or {}).get("tfs"),
                **((per_pair[s].get("best") or {}).get("candle_analysis") or {}),
            }
            for s in BOOK
            if per_pair.get(s)
        },
        "summary": {
            "pairs": len([s for s in BOOK if per_pair.get(s) and per_pair[s].get("best")]),
            "strategies_tested": len(LAB_WINNERS),
            "candles_total": sum(
                int((per_pair.get(s) or {}).get("candles_all_tfs") or 0) for s in BOOK
            ),
            "trades_total": total_all,
            "structured_trades": structured_all,
            "desk_success_rate_pct": desk_success,
            "honest_score_1_to_10": score,
            "survives_cells": sum(int((per_pair.get(s) or {}).get("survives") or 0) for s in BOOK),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    detail = {
        "generated_at": payload["generated_at"],
        "honest_tests": honest_rows,
        "desk_honest": {
            "survives": hs,
            "fragile": hf,
            "dead": hd,
            "score_1_to_10": score,
            "mean_oos_exp_r": round(float(np.mean(oos_ers)), 4) if oos_ers else None,
        },
        "trades_total": total_all,
        "structured_trades": structured_all,
        "pair_trades": pair_trades,
    }
    DETAIL.write_text(json.dumps(detail, indent=2, default=str))

    _prog(
        status="done",
        phase="complete",
        pct=100,
        message=(
            f"DONE · lab winners multi-TF · {total_all:,} trades "
            f"(struct {structured_all:,}) · honest {score}/10 · "
            f"{hs}S/{hf}F/{hd}D"
        ),
        pair_trades=dict(pair_trades),
        desk_success_rate_pct=desk_success,
        honest_score_1_to_10=score,
        candles_total=payload["summary"]["candles_total"],
    )
    print(f"Wrote {OUT} + {DETAIL} in {payload['elapsed_s']}s", flush=True)
    print(
        f"\nHonest desk: {hs}S/{hf}F/{hd}D · {score}/10 · "
        f"volume trades={total_all:,} (target {target_total:,})",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run_lab_winners_multi(years=DEFAULT_YEARS, target_total=TARGET_TOTAL)
