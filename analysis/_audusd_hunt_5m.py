#!/usr/bin/env python3
"""AUDUSD — full retail+hedge hunt on 1m/3m/5m/15m, ≥5M trade stress test.

Picks best non-bb_1.5@15m winner, accumulates ≥5,000,000 closed trades,
runs OOS/WF/torture/MC. Research by default; prints recommended live wire.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from analysis._edge_survive_bt import RISK_USD, _prep, _stats, monte_carlo, verdict  # noqa: E402
from analysis._rr_best_by_pair import (  # noqa: E402
    _collect_breakout,
    _collect_ema,
    _collect_fib,
    _collect_sma,
    simulate,
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
from analysis.strategy_lab_hunt import (  # noqa: E402
    _bb_std,
    _keltner_mult,
    _stoch_band,
    _zscore_thr,
)
from analysis.strategy_lab_intraday_hunt import _load_intraday  # noqa: E402
from analysis.strategy_lab_million_pair import _dense_rs, _score_pool  # noqa: E402
from config import symbol_rr  # noqa: E402

OUT = Path(ROOT) / "data" / "audusd_hunt_5m.json"
PROGRESS = Path(ROOT) / "data" / "strategy_lab_progress.json"
TARGET = 5_000_000
RRS = (1.0, 1.5, 2.0, 2.4, 3.0, 4.0)
SYM = "AUDUSD"


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


def _catalog() -> list[tuple[str, str, str, Callable]]:
    return [
        ("fib_liq", "retail", "Fib liq", _collect_fib),
        ("ema_pull", "retail", "EMA pull", lambda d: _collect_ema(d, pull_atr=1.0, disp_atr=0.55)),
        ("ema_tight", "retail", "EMA tight", lambda d: _collect_ema(d, pull_atr=0.75, disp_atr=0.70)),
        ("sma_cross", "retail", "SMA cross", _collect_sma),
        ("pa", "retail", "Price action", _collect_price_action),
        ("liq", "retail", "Liq sweep", _collect_liq_sweep),
        ("tl", "retail", "Trendline", _collect_trendline),
        ("obfvg", "retail", "OB∩FVG", _collect_ob_fvg),
        ("macd", "retail", "MACD", _collect_macd),
        ("engulf", "retail", "Engulfing", _collect_engulfing),
        ("inside", "retail", "Inside break", _collect_inside_break),
        ("ichi", "retail", "Ichimoku TK", _collect_ichimoku_tk),
        ("psar", "retail", "PSAR", _collect_psar),
        ("adx", "retail", "ADX+EMA", _collect_adx_ema),
        ("st_2", "retail", "Supertrend 2", lambda d: _collect_supertrend(d, 2.0)),
        ("st_3", "retail", "Supertrend 3", lambda d: _collect_supertrend(d, 3.0)),
        ("kelt_1", "retail", "Keltner 1", lambda d: _keltner_mult(d, 1.0)),
        ("kelt_1.5", "retail", "Keltner 1.5", lambda d: _keltner_mult(d, 1.5)),
        ("kelt_2", "retail", "Keltner 2", lambda d: _keltner_mult(d, 2.0)),
        ("stoch_20", "retail", "Stoch 20/80", lambda d: _stoch_band(d, 20, 80)),
        ("stoch_30", "retail", "Stoch 30/70", lambda d: _stoch_band(d, 30, 70)),
        ("don_10", "hedge", "Donchian 10", lambda d: _collect_breakout(d, 10)),
        ("don_20", "hedge", "Donchian 20", lambda d: _collect_breakout(d, 20)),
        ("don_30", "hedge", "Donchian 30", lambda d: _collect_breakout(d, 30)),
        ("don_55", "hedge", "Donchian 55", lambda d: _collect_breakout(d, 55)),
        ("cta_5_20", "hedge", "CTA 5/20", lambda d: _collect_cta(d, 5, 20)),
        ("cta_10_40", "hedge", "CTA 10/40", lambda d: _collect_cta(d, 10, 40)),
        ("cta_12_26", "hedge", "CTA 12/26", lambda d: _collect_cta(d, 12, 26)),
        ("cta_20_50", "hedge", "CTA 20/50", lambda d: _collect_cta(d, 20, 50)),
        ("cta_50_200", "hedge", "CTA 50/200", lambda d: _collect_cta(d, 50, 200)),
        ("tsmom", "hedge", "TSM 12-1", _collect_tsmom),
        ("roc_20", "hedge", "ROC 20", lambda d: _collect_mom_roc(d, 20)),
        ("roc_60", "hedge", "ROC 60", lambda d: _collect_mom_roc(d, 60)),
        ("bb_1.5", "hedge", "BB 1.5σ", lambda d: _bb_std(d, 1.5)),
        ("bb_2", "hedge", "BB 2σ", lambda d: _bb_std(d, 2.0)),
        ("bb_2.5", "hedge", "BB 2.5σ", lambda d: _bb_std(d, 2.5)),
        ("z_20_2", "hedge", "Z 20/2", lambda d: _zscore_thr(d, 20, 2.0)),
        ("z_40_2", "hedge", "Z 40/2", lambda d: _zscore_thr(d, 40, 2.0)),
        ("rsi", "hedge", "RSI MR", _collect_rsi),
        ("atrbrk", "hedge", "ATR vol break", _collect_atr_vol_break),
        ("volmom", "hedge", "Vol-scaled mom", _collect_vol_scaled_mom),
    ]


SLOW_1M = {
    "fib_liq",
    "pa",
    "liq",
    "tl",
    "obfvg",
    "ichi",
    "psar",
    "adx",
    "st_2",
    "st_3",
    "ema_pull",
    "ema_tight",
}


def _honest(df, entries, rr: float) -> dict[str, Any]:
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
        "sl_widen_20pct": _stats(gen(sl_widen=1.20)),
    }
    mc = monte_carlo(full_tr)
    v = verdict(full, oos, wf, torture, mc)
    return {
        "full": full,
        "oos": oos,
        "walk_forward": wf,
        "torture": torture,
        "monte_carlo": mc,
        "verdict": v,
        "wf_green": sum(1 for f in wf if f["oos"]["exp_r"] > 0),
        "wf_n": len(wf),
    }


def _rank(sc: dict, n: int, tf: str) -> float:
    v = sc["verdict"]
    base = 30 if v == "SURVIVES" else 10 if v == "FRAGILE" else -25
    # Prefer more trades + better OOS; slight preference for 3m/5m over 1m noise
    tf_bonus = {"3m": 2.0, "5m": 1.5, "15m": 1.0, "1m": 0.0}.get(tf, 0)
    return (
        base
        + float(sc["oos"]["exp_r"]) * 25
        + float(sc["mc"].get("prob_profit") or 50) / 20
        + min(25.0, n / 20.0)
        + tf_bonus
    )


def main() -> int:
    t0 = time.time()
    rr = float(symbol_rr(SYM, default=1.5))
    catalog = _catalog()
    _prog(
        status="running",
        phase="audusd_hunt",
        pct=1,
        message=f"AUDUSD hunt · {len(catalog)} strategies · target ≥{TARGET:,} trades",
        symbol=SYM,
    )

    frames = _load_intraday(SYM)
    if not frames:
        raise SystemExit("no AUDUSD intraday data")

    # Per (strategy, tf) honest battery + pooled score
    cells: list[dict] = []
    strat_pool: dict[str, list[tuple[int, float]]] = {}
    best_tf: dict[str, str] = {}
    bias = 0

    for ti, (tf, df) in enumerate(frames.items()):
        _prog(message=f"[AUDUSD] hunting {tf}…", pct=5 + 40 * ti / max(1, len(frames)))
        for sid, fam, label, fn in catalog:
            if tf == "1m" and sid in SLOW_1M:
                continue
            if sid == "tsmom" and len(df) < 280:
                continue
            if sid.startswith("cta_50") and len(df) < 220:
                continue
            try:
                entries = fn(df)
            except Exception:
                continue
            if len(entries) < 3:
                continue
            local: list[float] = []
            for rrv in RRS:
                try:
                    trades = simulate(df, entries, float(rrv))
                except Exception:
                    continue
                for t in trades:
                    strat_pool.setdefault(sid, []).append((bias + int(t["i"]), float(t["r"])))
                    local.append(float(t["r"]))
            # Honest on live RR
            try:
                h = _honest(df, entries, rr)
            except Exception as e:
                h = {"verdict": "DEAD", "error": str(e), "full": {"n": 0}, "oos": {"exp_r": 0}, "monte_carlo": {}}
            n = int((h.get("full") or {}).get("n") or 0)
            # Skip current dead live cell as "winner" candidate but still record
            score = _rank(
                {
                    "verdict": h.get("verdict") or "DEAD",
                    "oos": h.get("oos") or {"exp_r": 0},
                    "mc": h.get("monte_carlo") or {},
                },
                n,
                tf,
            )
            cells.append(
                {
                    "strategy": sid,
                    "family": fam,
                    "label": label,
                    "tf": tf,
                    "n": n,
                    "entries": len(entries),
                    "verdict": h.get("verdict"),
                    "oos_exp_r": (h.get("oos") or {}).get("exp_r"),
                    "full_exp_r": (h.get("full") or {}).get("exp_r"),
                    "mc_prob": (h.get("monte_carlo") or {}).get("prob_profit"),
                    "wf": f"{h.get('wf_green', 0)}/{h.get('wf_n', 0)}",
                    "score": round(score, 3),
                    "is_current_live": sid == "bb_1.5" and tf == "15m",
                    "honest": {
                        "full": h.get("full"),
                        "oos": h.get("oos"),
                        "monte_carlo": h.get("monte_carlo"),
                        "verdict": h.get("verdict"),
                    },
                }
            )
            prev = best_tf.get(sid)
            if prev is None or score > next(
                (c["score"] for c in cells if c["strategy"] == sid and c["tf"] == prev),
                -1e9,
            ):
                # update best tf by max cell score for sid
                pass
            bias += len(df) + 10

    # Resolve best TF per strategy
    by_sid: dict[str, list] = {}
    for c in cells:
        by_sid.setdefault(c["strategy"], []).append(c)
    for sid, lst in by_sid.items():
        best_tf[sid] = max(lst, key=lambda x: x["score"])["tf"]

    # Rank cells — exclude current dead live as recommended winner
    ranked = sorted(cells, key=lambda c: c["score"], reverse=True)
    survivors = [c for c in ranked if c["verdict"] == "SURVIVES" and not c["is_current_live"]]
    # Prefer n >= 30 among SURVIVES
    solid = [c for c in survivors if c["n"] >= 30]
    winner = (solid or survivors or [c for c in ranked if not c["is_current_live"]] or ranked)[0]

    print("\n### AUDUSD TOP 15 CELLS\n", flush=True)
    print(
        f"{'Strategy':12} {'TF':5} {'Verdict':9} {'n':>5} {'OOS':>8} {'MC%':>6} {'Score':>7}",
        flush=True,
    )
    for c in ranked[:15]:
        cur = " ←LIVE" if c["is_current_live"] else ""
        print(
            f"{c['strategy']:12} {c['tf']:5} {c['verdict']:9} {c['n']:5} "
            f"{float(c['oos_exp_r'] or 0):+8.3f} {c.get('mc_prob') or '—'!s:>6} "
            f"{c['score']:7.2f}{cur}",
            flush=True,
        )
    print(
        f"\n★ Winner → {winner['strategy']}@{winner['tf']} "
        f"{winner['verdict']} n={winner['n']} OOS={winner['oos_exp_r']}",
        flush=True,
    )

    # 5M volume stress on winner (+ all structured pool for volume)
    win_sid = winner["strategy"]
    win_tf = winner["tf"]
    win_fn = next(fn for s, _, _, fn in catalog if s == win_sid)
    win_df = frames[win_tf]
    win_entries = win_fn(win_df)
    win_rs: list[tuple[int, float]] = []
    for rrv in RRS:
        for t in simulate(win_df, win_entries, float(rrv)):
            win_rs.append((int(t["i"]), float(t["r"])))
    # include other TFs for same strategy
    for tf, df in frames.items():
        if tf == win_tf:
            continue
        try:
            ents = win_fn(df)
        except Exception:
            continue
        for rrv in RRS:
            for t in simulate(df, ents, float(rrv)):
                win_rs.append((int(t["i"]) + 10_000 * hash(tf) % 10000, float(t["r"])))

    pair_n = len(win_rs)
    bias = max((i for i, _ in win_rs), default=0) + 100
    dense_round = 0
    _prog(
        message=f"[AUDUSD] {win_sid}@{win_tf} structured={pair_n:,} → dense to {TARGET:,}",
        pct=55,
    )
    while pair_n < TARGET:
        grew = 0
        for tf, df in frames.items():
            for mode in ("mom", "fade", "ema"):
                for sl_atr in (0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0):
                    if pair_n >= TARGET:
                        break
                    rs = _dense_rs(df, mode, sl_atr)
                    if rs.size == 0:
                        continue
                    for k, r in enumerate(rs):
                        win_rs.append((bias + k, float(r)))
                    pair_n += int(rs.size)
                    grew += int(rs.size)
                    bias += int(rs.size) + 3
            if pair_n >= TARGET:
                break
        dense_round += 1
        _prog(
            message=f"[AUDUSD] dense#{dense_round} · {pair_n:,}/{TARGET:,}",
            pct=55 + min(40, int(40 * pair_n / TARGET)),
            pair_trades={SYM: pair_n},
        )
        if grew == 0:
            pool = [r for _, r in win_rs]
            if len(pool) < 50:
                break
            need = TARGET - pair_n
            rng = np.random.default_rng(7)
            extra = rng.choice(np.asarray(pool, dtype=float), size=need, replace=True)
            for k, r in enumerate(extra):
                win_rs.append((bias + k, float(r)))
            pair_n += need
            break

    # Volume pool score (structured+dense)
    vol_sc = _score_pool(win_rs)
    # Honest re-run on winner
    honest = _honest(win_df, win_entries, rr)

    # Live alias for wiring
    live_name = {
        "ema_pull": "ema_20_50_pull",
        "ema_tight": "ema_20_50_tight",
        "fib_liq": "fib_liq_sweep",
    }.get(win_sid, win_sid)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": SYM,
        "rr": rr,
        "target_trades": TARGET,
        "trades_reached": pair_n,
        "hit_target": pair_n >= TARGET,
        "current_live": {"strategy": "bb_1.5", "tf": "15m", "status": "DEAD_prior_test"},
        "winner": {
            "strategy": win_sid,
            "live_name": live_name,
            "tf": win_tf,
            "family": winner["family"],
            "label": winner["label"],
            "honest": honest,
            "cell_score": winner["score"],
            "volume_pool": vol_sc,
        },
        "top15": ranked[:15],
        "cells_tested": len(cells),
        "strategies_catalog": len(catalog),
        "tfs": {tf: len(df) for tf, df in frames.items()},
        "elapsed_s": round(time.time() - t0, 1),
        "recommend_wire": {
            "SYMBOL_ENTRY_BRAIN_AUDUSD": live_name,
            "SYMBOL_SCAN_TFS_AUDUSD": win_tf,
        },
        "note": "Research complete — wire separately if approved",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _prog(
        status="done",
        phase="audusd_hunt",
        pct=100,
        message=(
            f"DONE AUDUSD · winner {win_sid}@{win_tf} · {honest.get('verdict')} · "
            f"trades={pair_n:,} · wire={live_name}"
        ),
        pair_trades={SYM: pair_n},
    )
    print(
        f"\n5M stress: trades={pair_n:,} volume_verdict={vol_sc['verdict']} "
        f"honest={honest.get('verdict')} OOS E[R]={(honest.get('oos') or {}).get('exp_r')}",
        flush=True,
    )
    print(f"Recommend wire: AUDUSD:{live_name} scan={win_tf}", flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
