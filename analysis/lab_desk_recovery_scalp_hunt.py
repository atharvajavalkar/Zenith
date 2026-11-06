#!/usr/bin/env python3
"""DESK RECOVERY — scalper / intraday brain hunt (do not stop early).

Goal: find better scalp/intraday brains to flip the live book green.
  • TFs: 1m · 3m · 5m · 15m only (no swing)
  • Full Duka catalog + RR sweep + 8-test battery + MC
  • Priority order: ETH → BTC → GOLD → NAS → JPY → AUD → CHF → OIL
  • Prefer FREQUENCY (scalp punches) among firm survivors
  • Compare each winner vs current live brain
  • Densify math overlays on leaders until MIN_RUNTIME (default 8h)
  • Write Obsidian vault notes as each pair finishes

Does NOT auto-wire .env (report + Obsidian only).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis._edge_survive_bt import RISK_USD  # noqa: E402
from analysis.strategy_lab_duka_deep_hunt import (  # noqa: E402
    HEAVY,
    _battery,
    _catalog,
    _rank_score,
)
from analysis.strategy_lab_duka_scalp_1y_hunt import (  # noqa: E402
    RECENT_DAYS,
    RRS,
    TFS,
    WINDOW_DAYS,
    _load_frames_1y,
    _passes_scalp_gate,
    _slice_days,
)
from analysis.strategy_lab_intraday_hunt import _prog  # noqa: E402
from config import SYMBOL_ENTRY_BRAIN, symbol_entry_brain, symbol_rr  # noqa: E402

OUT = Path(ROOT) / "data" / "lab_desk_recovery_scalp_hunt.json"
LOG = Path(ROOT) / "data" / "lab_desk_recovery_scalp_hunt.log"
LOCK = Path(ROOT) / "data" / "lab_desk_recovery_scalp_hunt.lock"
PROGRESS = Path(ROOT) / "data" / "lab_desk_recovery_scalp_hunt.progress.json"
CHECKPOINT = Path(ROOT) / "data" / "lab_desk_recovery_scalp_hunt.checkpoint.json"
# Prefer mounted data/ so host Obsidian vault can open the folder
OBSIDIAN = Path(ROOT) / "data" / "ZenithRecovery"

# Bleed-first recovery order (scalper desk)
RECOVERY_ORDER = (
    "ETHUSD",
    "BTCUSD",
    "GOLD",
    "NAS100",
    "USDJPY",
    "AUDUSD",
    "USDCHF",
    "UKOIL",
)

# Heavy collectors on 5m melt the clock (fib_liq@5m burned ~19h). Scalp hunt: 15m only.
HEAVY_OK_TF = frozenset({"15m"})

# Keep running densify until this floor (user: even 10h)
MIN_RUNTIME_SEC = int(os.getenv("RECOVERY_HUNT_MIN_SEC", str(8 * 3600)))
MAX_RUNTIME_SEC = int(os.getenv("RECOVERY_HUNT_MAX_SEC", str(10 * 3600)))


def _log(msg: str) -> None:
    # File first — if stdout pipe fills, print must not block progress writes
    try:
        with LOG.open("a") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _progress(payload: dict) -> None:
    try:
        PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _prog(
            status=payload.get("status", "running"),
            phase="desk_recovery_scalp",
            pct=int(payload.get("pct") or 0),
            message=str(payload.get("message") or ""),
            symbol=payload.get("symbol"),
        )
    except Exception:
        pass


def _obs_write(rel: str, body: str) -> None:
    path = OBSIDIAN / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _init_obsidian() -> None:
    _obs_write(
        "00_Recovery_Board.md",
        f"""# Zenith Desk Recovery Board

Started: {datetime.now(timezone.utc).isoformat()}
Mode: **scalper / intraday** (1m·3m·5m·15m)
Goal: firm survivors that can punch often and flip the book green.

## Rules
- No auto-wire until reviewed
- Firm = SURVIVES 1y + recent 90d OOS E[R]>0 + WF≥2 + n≥40
- Prefer higher trade count (scalp frequency)
- Must beat or clearly rival live brain on score

## Priority
1. ETHUSD 2. BTCUSD 3. GOLD 4. NAS100 5. USDJPY 6. AUDUSD 7. USDCHF 8. UKOIL

## Status
_See per-pair notes as hunt completes._
""",
    )
    _obs_write(
        "Templates/Hunt_Result.md",
        """# {{symbol}} — hunt result

- Live brain: 
- Winner: 
- TF / RR: 
- n / E[R] / OOS / WF / MC: 
- vs live: 
- Wire?: 
""",
    )


def _slim_bat(bat: dict) -> dict:
    full = bat.get("full") or {}
    oos = bat.get("oos") or {}
    return {
        "verdict": bat.get("verdict"),
        "n": full.get("n"),
        "pnl": full.get("pnl"),
        "exp_r": full.get("exp_r"),
        "win_rate": full.get("win_rate"),
        "oos_exp_r": oos.get("exp_r"),
        "wf_green": bat.get("wf_green"),
        "mc": (bat.get("monte_carlo") or {}).get("prob_profit"),
    }


def _rolling_fourier_hf(ret: np.ndarray, win: int = 64, stride: int = 8) -> np.ndarray:
    n = len(ret)
    out = np.full(n, np.nan, dtype=float)
    for i in range(win, n, stride):
        seg = ret[i - win : i]
        if not np.isfinite(seg).all():
            continue
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg)) ** 2
        tot = float(spec.sum())
        if tot <= 0:
            continue
        cut = max(2, len(spec) // 2)
        out[i] = float(spec[cut:].sum() / tot)
    return pd.Series(out).ffill().fillna(0.5).to_numpy(dtype=float)


def _filter_entries_math(
    df: pd.DataFrame, entries: list[dict], *, hf_max: Optional[float], min_eff: Optional[float]
) -> list[dict]:
    if not entries:
        return []
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fh = _rolling_fourier_hf(ret.to_numpy(), 64, 8)
    rng = (h - l).replace(0, np.nan)
    eff = ((c - o).abs() / rng).clip(0, 1).fillna(0).to_numpy()
    out = []
    for e in entries:
        i = int(e["ei"])
        if i < 0 or i >= len(df):
            continue
        if hf_max is not None and fh[i] > hf_max:
            continue
        if min_eff is not None and eff[i] < min_eff:
            continue
        out.append(e)
    return out


def _math_gate_grid() -> list[dict[str, Any]]:
    return [
        {"tag": "hf0.55", "hf_max": 0.55, "min_eff": None},
        {"tag": "hf0.50", "hf_max": 0.50, "min_eff": None},
        {"tag": "hf0.45", "hf_max": 0.45, "min_eff": None},
        {"tag": "hf0.50_e0.25", "hf_max": 0.50, "min_eff": 0.25},
        {"tag": "hf0.45_e0.30", "hf_max": 0.45, "min_eff": 0.30},
        {"tag": "e0.35", "hf_max": None, "min_eff": 0.35},
    ]


def _score_scalp(bat: dict, family: str, gate: bool) -> float:
    n = int((bat.get("full") or {}).get("n") or 0)
    return round(_rank_score(bat, family) + (2.5 if gate else 0) + min(3.0, n / 150.0), 3)


def _save_checkpoint(per_pair: dict[str, Any], t0: float) -> None:
    try:
        CHECKPOINT.write_text(
            json.dumps(
                {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "per_pair": per_pair,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT.exists():
        return {}
    try:
        raw = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        return dict(raw.get("per_pair") or {})
    except Exception:
        return {}


def _winner_from_obsidian(sym: str) -> Optional[dict[str, Any]]:
    """Resume helper: parse a completed Pairs/{sym}.md into a minimal winner row."""
    path = OBSIDIAN / "Pairs" / f"{sym}.md"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    import re

    m_w = re.search(r"\*\*Hunt winner:\*\* `([^`]+)`", text)
    m_tf = re.search(r"\*\*TF / RR:\*\* (\S+) / ([\d.]+)", text)
    m_1y = re.search(
        r"\*\*1y:\*\* n=(\d+) E\[R\]=([\d.]+) OOS=([\d.]+) WF=(\d+) MC=([\d.]+)",
        text,
    )
    m_gate = re.search(r"\*\*Deep gate:\*\* (True|False)", text)
    m_score = re.search(r"\*\*Score:\*\* ([\d.]+)", text)
    m_live = re.search(r"\*\*Live brain:\*\* `([^`]+)` RR=([\d.]+)", text)
    if not (m_w and m_tf and m_score):
        return None
    full = {
        "n": int(m_1y.group(1)) if m_1y else None,
        "exp_r": float(m_1y.group(2)) if m_1y else None,
        "oos_exp_r": float(m_1y.group(3)) if m_1y else None,
        "wf_green": int(m_1y.group(4)) if m_1y else None,
        "mc": float(m_1y.group(5)) if m_1y else None,
        "verdict": "SURVIVES",
    }
    return {
        "symbol": sym,
        "live_brain": m_live.group(1) if m_live else symbol_entry_brain(sym),
        "live_rr": float(m_live.group(2)) if m_live else float(symbol_rr(sym, default=2.0)),
        "n_scored": 0,
        "n_math": 0,
        "n_gated": 1 if (m_gate and m_gate.group(1) == "True") else 0,
        "resumed": True,
        "winner": {
            "strategy": m_w.group(1),
            "family": "resumed",
            "label": "resumed from Obsidian",
            "tf": m_tf.group(1),
            "rr": float(m_tf.group(2)),
            "math": None,
            "full": full,
            "last_90d": {},
            "deep_gate": bool(m_gate and m_gate.group(1) == "True"),
            "score": float(m_score.group(1)),
        },
        "top5": [],
        "gated": [],
    }


def _run_symbol(
    sym: str,
    catalog: list,
    start: pd.Timestamp,
    end: pd.Timestamp,
    t0: float,
    pi: int,
    n_sym: int,
) -> dict[str, Any]:
    rr_live = float(symbol_rr(sym, default=2.0))
    live_brain = symbol_entry_brain(sym)
    _log(f"\n######## {sym} · live={live_brain} RR={rr_live} ########")
    frames = _load_frames_1y(sym, start, end)
    if not frames:
        return {"symbol": sym, "error": "no_local_hist", "live_brain": live_brain}

    scored: list[dict] = []
    for sid, family, label, fn in catalog:
        if time.time() - t0 > MAX_RUNTIME_SEC:
            _log(f"  MAX_RUNTIME inside {sym} — stopping catalog early")
            break
        for tf in TFS:
            if time.time() - t0 > MAX_RUNTIME_SEC:
                break
            if tf not in frames:
                continue
            if sid in HEAVY and tf not in HEAVY_OK_TF:
                continue
            df_all = frames[tf]
            _log(f"  … {sid}@{tf} ({len(df_all):,} bars)")
            _progress(
                {
                    "status": "running",
                    "pct": int(5 + 70 * (pi + 0.5) / n_sym),
                    "message": f"[{sym}] {sid}@{tf} scored={len(scored)}",
                    "symbol": sym,
                    "elapsed_sec": round(time.time() - t0, 1),
                }
            )
            try:
                entries = fn(df_all)
                if not entries or len(entries) < 30:
                    continue
                df_r = _slice_days(df_all, RECENT_DAYS, end)
                try:
                    entries_r = fn(df_r) if len(df_r) >= 200 else []
                except Exception:
                    entries_r = []
                best_cell = None
                for rr in RRS:
                    bat = _battery(df_all, entries, rr)
                    if len(df_r) < 200 or len(entries_r) < 15:
                        bat_r = {
                            "verdict": "DEAD",
                            "oos": {"exp_r": 0},
                            "full": {"n": 0},
                            "wf_green": 0,
                        }
                    else:
                        bat_r = _battery(df_r, entries_r, rr)
                    gate = _passes_scalp_gate(bat, bat_r)
                    cell = {
                        "strategy": sid,
                        "family": family,
                        "label": label,
                        "tf": tf,
                        "rr": rr,
                        "math": None,
                        "full": _slim_bat(bat),
                        "last_90d": {
                            "verdict": bat_r.get("verdict"),
                            "n": (bat_r.get("full") or {}).get("n"),
                            "oos_exp_r": (bat_r.get("oos") or {}).get("exp_r"),
                            "wf_green": bat_r.get("wf_green"),
                        },
                        "deep_gate": gate,
                        "score": _score_scalp(bat, family, gate),
                    }
                    if best_cell is None or cell["score"] > best_cell["score"]:
                        best_cell = cell
                if best_cell:
                    scored.append(best_cell)
                    if best_cell["deep_gate"]:
                        _log(
                            f"  ✓ GATE {sid:12} {tf:3} RR={best_cell['rr']} "
                            f"n={best_cell['full']['n']} "
                            f"E[R]={best_cell['full']['exp_r']} "
                            f"90dOOS={best_cell['last_90d'].get('oos_exp_r')}"
                        )
            except Exception as exc:
                _log(f"  skip {sid}@{tf}: {exc}")
            _progress(
                {
                    "status": "running",
                    "pct": int(5 + 70 * (pi + 0.5) / n_sym),
                    "message": f"[{sym}] {sid}@{tf} scored={len(scored)}",
                    "symbol": sym,
                    "elapsed_sec": round(time.time() - t0, 1),
                }
            )

    scored.sort(key=lambda r: float(r.get("score") or -1e9), reverse=True)
    gated = [r for r in scored if r.get("deep_gate")]
    winner = (gated or scored or [None])[0]

    # densify math on top bare winners (scalp recovery edge)
    math_cells: list[dict] = []
    leaders = (gated or scored)[:8]
    for lead in leaders:
        sid = lead["strategy"]
        tf = lead["tf"]
        if tf not in frames:
            continue
        # find collector
        fn = None
        family = lead["family"]
        label = lead["label"]
        for a, b, c, f in catalog:
            if a == sid:
                fn = f
                family, label = b, c
                break
        if fn is None:
            continue
        try:
            base_ents = fn(frames[tf])
        except Exception:
            continue
        df_all = frames[tf]
        df_r = _slice_days(df_all, RECENT_DAYS, end)
        for g in _math_gate_grid():
            ents = _filter_entries_math(
                df_all, base_ents, hf_max=g["hf_max"], min_eff=g["min_eff"]
            )
            if len(ents) < 30:
                continue
            try:
                ents_r = _filter_entries_math(
                    df_r, fn(df_r) if len(df_r) >= 200 else [], hf_max=g["hf_max"], min_eff=g["min_eff"]
                )
            except Exception:
                ents_r = []
            for rr in RRS:
                bat = _battery(df_all, ents, rr)
                bat_r = (
                    _battery(df_r, ents_r, rr)
                    if len(df_r) >= 200 and len(ents_r) >= 15
                    else {
                        "verdict": "DEAD",
                        "oos": {"exp_r": 0},
                        "full": {"n": 0},
                        "wf_green": 0,
                    }
                )
                gate = _passes_scalp_gate(bat, bat_r)
                cell = {
                    "strategy": f"{sid}+{g['tag']}",
                    "family": family,
                    "label": f"{label} · math {g['tag']}",
                    "tf": tf,
                    "rr": rr,
                    "math": g,
                    "full": _slim_bat(bat),
                    "last_90d": {
                        "verdict": bat_r.get("verdict"),
                        "n": (bat_r.get("full") or {}).get("n"),
                        "oos_exp_r": (bat_r.get("oos") or {}).get("exp_r"),
                        "wf_green": bat_r.get("wf_green"),
                    },
                    "deep_gate": gate,
                    "score": _score_scalp(bat, family, gate) + (0.3 if gate else 0),
                }
                math_cells.append(cell)
                if gate:
                    _log(
                        f"  ✓ MATH {cell['strategy'][:20]:20} {tf} RR={rr} "
                        f"n={cell['full']['n']} E[R]={cell['full']['exp_r']}"
                    )

    all_cells = scored + math_cells
    all_cells.sort(key=lambda r: float(r.get("score") or -1e9), reverse=True)
    gated_all = [r for r in all_cells if r.get("deep_gate")]
    best = (gated_all or all_cells or [None])[0]

    # Obsidian note
    if best:
        body = f"""# {sym} — recovery scalp hunt

- **Live brain:** `{live_brain}` RR={rr_live}
- **Hunt winner:** `{best.get('strategy')}` · {best.get('label')}
- **TF / RR:** {best.get('tf')} / {best.get('rr')}
- **Math:** {best.get('math')}
- **1y:** n={best['full'].get('n')} E[R]={best['full'].get('exp_r')} OOS={best['full'].get('oos_exp_r')} WF={best['full'].get('wf_green')} MC={best['full'].get('mc')} · {best['full'].get('verdict')}
- **90d OOS E[R]:** {best['last_90d'].get('oos_exp_r')} (n={best['last_90d'].get('n')})
- **Deep gate:** {best.get('deep_gate')}
- **Score:** {best.get('score')}
- **Wire?:** {'CANDIDATE — review' if best.get('deep_gate') else 'NO (failed firm scalp gate)'}

## Top 5
"""
        for i, c in enumerate(all_cells[:5], 1):
            body += (
                f"{i}. `{c['strategy']}` @{c['tf']} RR={c['rr']} "
                f"n={c['full'].get('n')} E[R]={c['full'].get('exp_r')} "
                f"gate={c['deep_gate']} score={c['score']}\n"
            )
        body += f"\n_Updated {datetime.now(timezone.utc).isoformat()}_\n"
        _obs_write(f"Pairs/{sym}.md", body)

    return {
        "symbol": sym,
        "live_brain": live_brain,
        "live_rr": rr_live,
        "n_scored": len(scored),
        "n_math": len(math_cells),
        "n_gated": len(gated_all),
        "winner": best,
        "top5": all_cells[:5],
        "gated": gated_all[:10],
    }


def main() -> int:
    my_pid = os.getpid()
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip() or "0")
            if pid and pid != my_pid and Path(f"/proc/{pid}").exists():
                _log(f"another recovery hunt running (pid {pid}) — exit")
                return 2
        except Exception:
            pass
    LOCK.write_text(str(my_pid))
    try:
        # Append separator on resume; full wipe only if empty/new
        if not LOG.exists() or LOG.stat().st_size == 0:
            LOG.write_text("")
        else:
            with LOG.open("a") as f:
                f.write(f"\n\n===== RESTART pid={my_pid} {datetime.now(timezone.utc).isoformat()} =====\n")
    except Exception:
        pass

    t0 = time.time()
    _init_obsidian()
    catalog = _catalog()
    end = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=1), tz="UTC")
    # hist may lag — anchor later if needed inside load
    start = end - pd.Timedelta(days=WINDOW_DAYS)

    _log(
        "=== DESK RECOVERY SCALP HUNT ===\n"
        f"Scalper/intraday TFs={TFS} · RR={list(RRS)}\n"
        f"Window {start.date()}→{end.date()} · recent gate {RECENT_DAYS}d\n"
        f"Order={RECOVERY_ORDER}\n"
        f"min_runtime={MIN_RUNTIME_SEC}s max={MAX_RUNTIME_SEC}s\n"
        f"catalog={len(catalog)} · heavy_ok_tf={sorted(HEAVY_OK_TF)} · Obsidian={OBSIDIAN}\n"
        "LIVE .env NOT auto-wired · resume via checkpoint/Obsidian\n"
    )
    _progress(
        {
            "status": "running",
            "pct": 1,
            "message": f"Recovery scalp · {len(RECOVERY_ORDER)} pairs · {len(catalog)} strats",
            "elapsed_sec": 0,
        }
    )

    per_pair: dict[str, Any] = _load_checkpoint()
    for pi, sym in enumerate(RECOVERY_ORDER):
        if time.time() - t0 > MAX_RUNTIME_SEC:
            _log("MAX_RUNTIME hit — stopping cleanly")
            break
        # Resume: keep finished pairs (checkpoint or Obsidian note)
        existing = per_pair.get(sym)
        if existing and existing.get("winner") and not existing.get("error"):
            _log(f"\n######## {sym} · RESUME checkpoint winner={existing['winner'].get('strategy')} ########")
            continue
        obs = _winner_from_obsidian(sym)
        if obs and obs.get("winner"):
            _log(
                f"\n######## {sym} · RESUME Obsidian winner={obs['winner'].get('strategy')} ########"
            )
            per_pair[sym] = obs
            _save_checkpoint(per_pair, t0)
            continue
        try:
            per_pair[sym] = _run_symbol(
                sym, catalog, start, end, t0, pi, len(RECOVERY_ORDER)
            )
            _save_checkpoint(per_pair, t0)
        except Exception as exc:
            _log(f"  FAIL {sym}: {exc}")
            per_pair[sym] = {"symbol": sym, "error": str(exc)}
            _obs_write(
                f"Pairs/{sym}.md",
                f"# {sym}\n\nERROR: {exc}\n\n_{datetime.now(timezone.utc).isoformat()}_\n",
            )
            _save_checkpoint(per_pair, t0)

    # Densify rounds on gated leaders until min runtime
    densify_round = 0
    while time.time() - t0 < MIN_RUNTIME_SEC:
        densify_round += 1
        _log(
            f"\n##### DENSIFY round {densify_round} "
            f"(elapsed={time.time()-t0:.0f}s / {MIN_RUNTIME_SEC}s) #####"
        )
        # re-run math densify on symbols that have winners
        for pi, sym in enumerate(RECOVERY_ORDER):
            if time.time() - t0 >= MIN_RUNTIME_SEC:
                break
            if time.time() - t0 > MAX_RUNTIME_SEC:
                break
            row = per_pair.get(sym) or {}
            if row.get("error") or not row.get("winner"):
                continue
            # Extra RR/math already done once; expand HF grid slightly
            try:
                frames = _load_frames_1y(sym, start, end)
            except Exception:
                continue
            if not frames:
                continue
            w = row["winner"]
            sid_base = str(w.get("strategy") or "").split("+")[0]
            tf = w.get("tf")
            if tf not in frames:
                continue
            fn = next((f for a, b, c, f in catalog if a == sid_base), None)
            if fn is None:
                continue
            try:
                base = fn(frames[tf])
            except Exception:
                continue
            extra_gates = [
                {"tag": f"D{densify_round}_hf{h:g}", "hf_max": h, "min_eff": e}
                for h in (0.42, 0.47, 0.52, 0.58)
                for e in (None, 0.22, 0.28)
            ]
            df_all = frames[tf]
            df_r = _slice_days(df_all, RECENT_DAYS, end)
            improved = False
            for g in extra_gates:
                ents = _filter_entries_math(
                    df_all, base, hf_max=g["hf_max"], min_eff=g["min_eff"]
                )
                if len(ents) < 40:
                    continue
                for rr in (2.0, 2.4, 3.0, 4.0):
                    bat = _battery(df_all, ents, rr)
                    try:
                        ents_r = _filter_entries_math(
                            df_r,
                            fn(df_r) if len(df_r) >= 200 else [],
                            hf_max=g["hf_max"],
                            min_eff=g["min_eff"],
                        )
                    except Exception:
                        ents_r = []
                    bat_r = (
                        _battery(df_r, ents_r, rr)
                        if len(ents_r) >= 15
                        else {
                            "verdict": "DEAD",
                            "oos": {"exp_r": 0},
                            "full": {"n": 0},
                            "wf_green": 0,
                        }
                    )
                    gate = _passes_scalp_gate(bat, bat_r)
                    cell = {
                        "strategy": f"{sid_base}+{g['tag']}",
                        "family": w.get("family"),
                        "label": f"densify {g['tag']}",
                        "tf": tf,
                        "rr": rr,
                        "math": g,
                        "full": _slim_bat(bat),
                        "last_90d": {
                            "verdict": bat_r.get("verdict"),
                            "n": (bat_r.get("full") or {}).get("n"),
                            "oos_exp_r": (bat_r.get("oos") or {}).get("exp_r"),
                            "wf_green": bat_r.get("wf_green"),
                        },
                        "deep_gate": gate,
                        "score": _score_scalp(bat, str(w.get("family") or "x"), gate),
                    }
                    if gate and cell["score"] > float(w.get("score") or -1e9):
                        row["winner"] = cell
                        row.setdefault("gated", []).insert(0, cell)
                        improved = True
                        _log(
                            f"  ↑ {sym} improved → {cell['strategy']} "
                            f"RR={rr} n={cell['full']['n']} score={cell['score']}"
                        )
            if improved:
                per_pair[sym] = row
                # refresh obsidian
                best = row["winner"]
                _obs_write(
                    f"Pairs/{sym}.md",
                    f"""# {sym} — recovery scalp hunt (densify r{densify_round})

- **Live:** `{row.get('live_brain')}` RR={row.get('live_rr')}
- **Winner:** `{best.get('strategy')}` @{best.get('tf')} RR={best.get('rr')}
- **1y:** n={best['full'].get('n')} E[R]={best['full'].get('exp_r')} OOS={best['full'].get('oos_exp_r')} MC={best['full'].get('mc')}
- **Gate:** {best.get('deep_gate')} score={best.get('score')}
- **Wire?:** CANDIDATE — review

_{datetime.now(timezone.utc).isoformat()}_
""",
                )
            _progress(
                {
                    "status": "running",
                    "pct": min(99, int(75 + 20 * densify_round / 10)),
                    "message": f"densify r{densify_round} [{sym}]",
                    "symbol": sym,
                    "elapsed_sec": round(time.time() - t0, 1),
                }
            )
        if densify_round >= 24:
            _log("densify round cap")
            break

    # Board summary
    lines = [
        f"# Zenith Desk Recovery Board",
        f"",
        f"Finished: {datetime.now(timezone.utc).isoformat()}",
        f"Elapsed: {(time.time()-t0)/3600:.2f}h",
        f"Mode: scalper / intraday",
        f"",
        f"| Pair | Live | Winner | TF | RR | n | E[R] | Gate |",
        f"|---|---|---|---|---:|---:|---:|---|",
    ]
    wire_list = []
    for sym in RECOVERY_ORDER:
        row = per_pair.get(sym) or {}
        w = row.get("winner") or {}
        full = w.get("full") or {}
        lines.append(
            f"| {sym} | `{row.get('live_brain','?')}` | `{w.get('strategy','—')}` | "
            f"{w.get('tf','—')} | {w.get('rr','—')} | {full.get('n','—')} | "
            f"{full.get('exp_r','—')} | {w.get('deep_gate', False)} |"
        )
        if w.get("deep_gate"):
            wire_list.append(sym)
    lines.append("")
    lines.append("## Firm scalp candidates (deep_gate)")
    for sym in wire_list:
        w = (per_pair.get(sym) or {}).get("winner") or {}
        lines.append(
            f"- **{sym}**: `{w.get('strategy')}` @{w.get('tf')} RR={w.get('rr')} "
            f"n={w.get('full',{}).get('n')} E[R]={w.get('full',{}).get('exp_r')}"
        )
    if not wire_list:
        lines.append("- _None cleared firm scalp gate_")
    lines.append("")
    lines.append("_Do not wire until reviewed. LIVE .env unchanged by this hunt._")
    _obs_write("00_Recovery_Board.md", "\n".join(lines) + "\n")

    elapsed = round(time.time() - t0, 1)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "scalper_intraday": True,
        "elapsed_sec": elapsed,
        "min_runtime_sec": MIN_RUNTIME_SEC,
        "max_runtime_sec": MAX_RUNTIME_SEC,
        "densify_rounds": densify_round,
        "recovery_order": list(RECOVERY_ORDER),
        "obsidian": str(OBSIDIAN),
        "live_brains": dict(SYMBOL_ENTRY_BRAIN or {}),
        "per_pair": per_pair,
        "firm_candidates": [
            {
                "symbol": s,
                "winner": (per_pair.get(s) or {}).get("winner"),
            }
            for s in wire_list
        ],
        "recommend_live": False,
        "note": "Review Obsidian board then wire selectively — not auto-applied",
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _progress(
        {
            "status": "done",
            "pct": 100,
            "message": f"done {elapsed/3600:.2f}h · firm={len(wire_list)}",
            "elapsed_sec": elapsed,
        }
    )
    _log(
        f"\nWrote {OUT}\nObsidian {OBSIDIAN}/00_Recovery_Board.md\n"
        f"Firm candidates: {wire_list or 'none'}\n"
        f"Elapsed {elapsed}s ({elapsed/3600:.2f}h)"
    )
    try:
        LOCK.unlink(missing_ok=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
