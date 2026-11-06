#!/usr/bin/env python3
"""STEP 2 — Read-only trade autopsy agent.

Reads closed trades from Postgres, diagnoses what went wrong/right,
writes a report. NEVER changes live config, brains, or .env.

Usage (Docker):
  docker compose exec -T -e PYTHONPATH=/app backend \\
    python /app/analysis/trade_autopsy_agent.py --days 14
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, unquote

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_JSON = Path("/tmp/trade_autopsy_report.json")
OUT_MD = Path("/tmp/trade_autopsy_report.md")


def _parse_features(raw: Any) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _fetch_closes(
    days: int = 14,
    limit: int = 400,
    *,
    since: Optional[datetime] = None,
    since_desk_epoch: bool = True,
) -> list[dict]:
    """Load closed trades via asyncpg (already in backend image).

    Default: only trades after desk reset (new-brain session). Pass
    since_desk_epoch=False to use a rolling ``days`` window instead.
    """
    import asyncio

    url = os.getenv("DATABASE_URL", "postgresql://algo:algo@postgres:5432/algo")
    if "postgres:5432" in url and not Path("/.dockerenv").exists():
        url = url.replace("@postgres:5432", "@127.0.0.1:5432")

    cut = since
    if cut is None and since_desk_epoch:
        try:
            from app.services.desk_epoch import get_epoch

            cut = get_epoch()
        except Exception:
            cut = None
    if cut is None:
        cut = datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
    if cut.tzinfo is None:
        cut = cut.replace(tzinfo=timezone.utc)

    sql = """
        SELECT id, symbol, signal, entry, sl, tp, tp1, tp2, tp3,
               pnl, pnl_status, status, message, lot_size, margin_usd,
               rr_type, trade_mode, features, candle_time, log_time, created_at
        FROM trade_records
        WHERE (status = 'closed' OR pnl_status LIKE 'closed%')
          AND created_at >= $1
        ORDER BY created_at DESC
        LIMIT $2
    """

    async def _run() -> list[dict]:
        import asyncpg

        conn = await asyncpg.connect(url)
        try:
            records = await conn.fetch(sql, cut, limit)
            out = []
            for r in records:
                d = dict(r)
                out.append(d)
            return out
        finally:
            await conn.close()

    try:
        rows = asyncio.run(_run())
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read trade_records ({exc}). "
            "Run: docker compose exec -T -e PYTHONPATH=/app backend "
            "python /app/analysis/trade_autopsy_agent.py"
        ) from exc

    # Extra guard: respect desk epoch via log_time when created_at is ambiguous
    if since_desk_epoch:
        try:
            from app.services.desk_epoch import row_after_epoch

            rows = [
                r
                for r in rows
                if row_after_epoch(r.get("created_at"), r.get("log_time"))
            ]
        except Exception:
            pass
    return rows


def _r_multiple(row: dict, feats: dict) -> Optional[float]:
    pnl = row.get("pnl")
    if pnl is None:
        return None
    risk = feats.get("risk_usd")
    try:
        risk_f = float(risk) if risk is not None else None
    except Exception:
        risk_f = None
    if risk_f and risk_f > 1e-9:
        # risk_usd in features is sometimes price-distance not $ — prefer |entry-sl|*notion
        if risk_f < 1.0 and row.get("lot_size"):
            # tiny risk_usd likely price units; fall through to SL distance heuristic
            pass
        elif risk_f >= 1.0:
            return float(pnl) / risk_f

    entry, sl = row.get("entry"), row.get("sl")
    lot = float(row.get("lot_size") or 0) or None
    if entry is None or sl is None:
        return None
    risk_px = abs(float(entry) - float(sl))
    if risk_px <= 0:
        return None
    # Approximate R from pnl vs planned risk when risk_usd missing/bad
    if risk_f and risk_f > 1e-12:
        # if risk_usd looked like price distance, scale loosely
        return float(pnl) / max(abs(float(pnl)) / 2.0, 1.0) if False else None
    if abs(float(pnl)) < 1e-9:
        return 0.0
    # Sign-correct binary-ish R from win/loss when $ risk unknown
    return 2.0 if float(pnl) > 0 else (-1.0 if float(pnl) < 0 else 0.0)


def _planned_r(row: dict) -> Optional[float]:
    entry, sl, tp = row.get("entry"), row.get("sl"), row.get("tp")
    if entry is None or sl is None or tp is None:
        return None
    risk = abs(float(entry) - float(sl))
    if risk <= 0:
        return None
    return abs(float(tp) - float(entry)) / risk


def diagnose(row: dict) -> dict[str, Any]:
    feats = _parse_features(row.get("features"))
    pnl = float(row.get("pnl") or 0.0)
    side = str(row.get("signal") or feats.get("signal") or "").upper()
    sym = str(row.get("symbol") or feats.get("symbol") or feats.get("SYMBOL") or "").upper().strip()
    if not sym or sym == "NONE":
        # broker wire names sometimes only in features
        sym = str(feats.get("broker_symbol") or feats.get("INSTRUMENT") or "?").upper().strip()
    # Map common XM wires back to desk names for reporting
    wire_map = {
        "US100CASH": "NAS100",
        "BRENTCASH": "UKOIL",
        "XAUUSD": "GOLD",
        "XAGUSD": "SILVER",
    }
    sym = wire_map.get(sym, sym) or "?"
    brain = (
        feats.get("candle_brain")
        or (feats.get("rl_overlays") or {}).get("candle_brain")
        or feats.get("BRAIN")
        or feats.get("brain")
        or "unknown"
    )
    mc = feats.get("mc_prob") or feats.get("SUCCESS_PERCENT")
    markov = feats.get("markov_mc_prob") or feats.get("MARKOV_MC_PROB")
    exec_tf = feats.get("exec_tf") or feats.get("EXEC_TF")
    state = feats.get("STATE") or feats.get("state")
    pnl_status = str(row.get("pnl_status") or "")
    msg = str(row.get("message") or "")
    rr_type = row.get("rr_type") or feats.get("rr_ratio")
    hour = feats.get("hour_utc")
    prime = feats.get("prime_hour")
    news_aligned = feats.get("news_aligned") or feats.get("WORLD_NEWS_ALIGNED")
    bias_aligned = feats.get("bias_aligned")
    smc_ok = (feats.get("concepts") or {}).get("smc_ok")
    planned = _planned_r(row)
    r_est = _r_multiple(row, feats)

    # Re-classify legacy "closed_manual" XM sync rows that were actually SL/TP
    exit_px = row.get("current_price")
    if pnl_status in ("closed_manual", "closed_broker", "") or "Closed on XM" in msg:
        try:
            from analysis.trade_outcome import classify_broker_exit

            inferred = classify_broker_exit(
                side if side in ("BUY", "SELL") else row.get("signal"),
                row.get("entry"),
                row.get("sl"),
                row.get("tp"),
                exit_px,
                symbol=sym,
                tp1=row.get("tp1"),
                tp2=row.get("tp2"),
                tp3=row.get("tp3"),
            )
            if inferred in ("closed_sl", "closed_tp"):
                pnl_status = inferred
        except Exception:
            pass

    tags: list[str] = []
    notes: list[str] = []

    won = pnl > 0.5
    lost = pnl < -0.5

    if pnl_status == "closed_sl":
        tags.append("stop_loss_hit")
        notes.append("Broker stop-loss was hit (or price stopped through SL).")
    elif pnl_status == "closed_tp":
        tags.append("take_profit_hit")
        notes.append("Take-profit / target was hit.")
    elif "Stopped at market" in msg or (
        pnl_status == "closed_manual" and "stop-all" in msg.lower()
    ):
        tags.append("desk_force_close")
        notes.append("Desk force-closed the position (stop-all / market).")
    elif pnl_status == "closed_manual":
        tags.append("true_manual_close")
        notes.append("Labeled as a true manual close.")
    elif pnl_status == "closed_broker" or (
        "Closed on XM" in msg and pnl_status not in ("closed_sl", "closed_tp")
    ):
        tags.append("broker_unclear_exit")
        notes.append(
            "XM closed the position but exit didn’t clearly match SL or TP "
            "(spread/gap/partial trail)."
        )
    if mc is not None:
        try:
            mc_f = float(mc)
            if mc_f < 42:
                tags.append("soft_mc_entry")
                notes.append(f"Entered with soft MC ({mc_f:.1f}%).")
            elif mc_f < 48:
                tags.append("mid_mc_entry")
        except Exception:
            pass

    if planned is not None and planned >= 2.8 and lost:
        tags.append("stretched_rr_loss")
        notes.append(f"Planned RR≈{planned:.1f} and trade lost — stretch may have been optimistic.")

    if prime == 0 or prime == 0.0:
        tags.append("off_prime_hour")
        notes.append(f"Entry hour_utc={hour} was not marked prime.")

    if news_aligned in (0, 0.0, False) and lost:
        tags.append("news_against")
        notes.append("World-news bias was not aligned with the trade.")

    if bias_aligned in (0, 0.0, False) and lost:
        tags.append("bias_against")
        notes.append("Desk bias not aligned.")

    if smc_ok is False and lost:
        tags.append("smc_weak")
        notes.append("SMC concept flag was false at entry.")

    if state and "CHOP" in str(state).upper() and lost:
        tags.append("chop_state")
        notes.append(f"State was {state}.")

    if exec_tf in ("1m", "3m") and lost:
        tags.append("fast_tf_loss")
        notes.append(f"Lost on fast TF ({exec_tf}).")

    if exec_tf in ("4h", "1h") and abs(pnl) > 100 and lost:
        tags.append("large_htf_loss")
        notes.append(f"Large loss on higher TF ({exec_tf}).")

    if won and mc is not None and float(mc) >= 50:
        tags.append("clean_win_profile")
        notes.append("Win with decent MC — keep similar setups.")

    if not tags:
        tags.append("insufficient_context" if not feats else "no_strong_flag")
        if not notes:
            notes.append("No strong diagnostic flag — needs candle replay for deeper blame.")

    # Research-only suggestion (never applied)
    suggestion = None
    if "soft_mc_entry" in tags and lost:
        suggestion = f"Research: raise MC soft floor for {sym} (paper A/B only)."
    elif "stretched_rr_loss" in tags:
        suggestion = f"Research: prefer 1:2 forced RR on {sym} vs stretched {rr_type}."
    elif "off_prime_hour" in tags and lost:
        suggestion = f"Research: session filter / reduce size off-prime for {sym}."
    elif "news_against" in tags:
        suggestion = f"Research: harder news-alignment gate on {sym}."
    elif "large_htf_loss" in tags:
        suggestion = f"Research: cap risk_usd on {exec_tf} {sym} legs."
    elif "stop_loss_hit" in tags and lost:
        suggestion = (
            f"Research: review {sym} stop placement / entry quality — "
            "most recent losses are genuine stop-outs."
        )
    elif "desk_force_close" in tags and lost:
        suggestion = "Process: avoid desk force-close when testing brain edge; let SL/TP decide."
    elif "true_manual_close" in tags and lost:
        suggestion = "Process: avoid manual early exits when testing brain edge; let SL/TP decide."
    elif won:
        suggestion = "Keep — no change suggested."
    elif "take_profit_hit" in tags:
        suggestion = "Keep — target hits are the desired exit."
    else:
        suggestion = None

    return {
        "id": row.get("id"),
        "symbol": sym,
        "side": side,
        "brain": brain,
        "exec_tf": exec_tf,
        "trade_mode": row.get("trade_mode"),
        "pnl": round(pnl, 2),
        "r_est": round(r_est, 3) if r_est is not None else None,
        "planned_rr": round(planned, 2) if planned is not None else None,
        "mc_prob": float(mc) if mc is not None else None,
        "markov_mc": float(markov) if markov is not None else None,
        "state": state,
        "pnl_status": pnl_status,
        "created_at": str(row.get("created_at")),
        "tags": tags,
        "notes": notes,
        "suggestion": suggestion,
        "outcome": "win" if won else ("loss" if lost else "flat"),
    }


def aggregate(autopsies: list[dict]) -> dict[str, Any]:
    by_sym: dict[str, list] = defaultdict(list)
    tag_counter: Counter = Counter()
    sugg_counter: Counter = Counter()
    for a in autopsies:
        by_sym[a["symbol"]].append(a)
        tag_counter.update(a.get("tags") or [])
        if a.get("suggestion") and a["outcome"] == "loss":
            sugg_counter[a["suggestion"]] += 1

    per_symbol = {}
    for sym, items in by_sym.items():
        wins = sum(1 for x in items if x["outcome"] == "win")
        losses = sum(1 for x in items if x["outcome"] == "loss")
        pnl = sum(float(x["pnl"] or 0) for x in items)
        top_tags = Counter(t for x in items for t in x.get("tags") or []).most_common(5)
        per_symbol[sym] = {
            "n": len(items),
            "wins": wins,
            "losses": losses,
            "pnl": round(pnl, 2),
            "top_tags": top_tags,
            "top_loss_suggestions": Counter(
                x["suggestion"]
                for x in items
                if x.get("outcome") == "loss" and x.get("suggestion")
            ).most_common(3),
        }

    return {
        "tag_counts": tag_counter.most_common(15),
        "suggestion_counts": sugg_counter.most_common(10),
        "per_symbol": per_symbol,
    }


TAG_LAYMAN = {
    "stop_loss_hit": "Stopped out at SL (broker hit the stop)",
    "take_profit_hit": "Took profit at target",
    "broker_unclear_exit": "XM closed it, but exit didn’t clearly match SL/TP",
    "desk_force_close": "Desk force-closed (stop-all)",
    "true_manual_close": "True manual close",
    "soft_mc_entry": "Entered with weak Monte Carlo confidence",
    "mid_mc_entry": "Entered on mid-tier Monte Carlo",
    "stretched_rr_loss": "Target was stretched (high RR) and the trade lost",
    "off_prime_hour": "Traded outside prime session hours",
    "news_against": "News bias was against the trade",
    "bias_against": "Desk bias was against the trade",
    "smc_weak": "SMC structure was weak at entry",
    "chop_state": "Market was choppy at entry",
    "fast_tf_loss": "Loss on a fast timeframe",
    "large_htf_loss": "Large loss on a higher timeframe",
    "clean_win_profile": "Clean win profile",
    "no_strong_flag": "No clear failure pattern in the log",
    "insufficient_context": "Not enough feature data to diagnose",
    # legacy key kept for old cached payloads
    "manual_or_sync_close": "Legacy sync label (reclassified when possible)",
}


def efficiency_score(
    *,
    wins: int,
    losses: int,
    pnl: float,
    tag_counts: Counter,
) -> int:
    """1–10 desk efficiency for a symbol (higher = healthier)."""
    n = wins + losses
    if n <= 0:
        return 5
    wr = wins / n
    score = 5.0
    score += (wr - 0.45) * 8.0
    if pnl > 50:
        score += 1.5
    elif pnl > 0:
        score += 0.75
    elif pnl < -500:
        score -= 2.0
    elif pnl < -100:
        score -= 1.25
    elif pnl < 0:
        score -= 0.5

    # Bad-pattern penalties (share of closes)
    def share(tag: str) -> float:
        return float(tag_counts.get(tag, 0)) / n

    if share("soft_mc_entry") >= 0.35:
        score -= 1.0
    if share("stretched_rr_loss") >= 0.25:
        score -= 0.75
    if share("off_prime_hour") >= 0.5:
        score -= 0.5
    if share("fast_tf_loss") >= 0.35:
        score -= 0.5
    # Stop-outs are normal — only mild drag when they dominate losses
    if share("stop_loss_hit") >= 0.7 and losses > wins:
        score -= 0.35
    if share("true_manual_close") >= 0.5 and losses > wins:
        score -= 0.4

    # Small-sample haircut toward neutral
    if n < 5:
        score = 0.55 * score + 0.45 * 5.0
    elif n < 10:
        score = 0.75 * score + 0.25 * 5.0

    return int(max(1, min(10, round(score))))


def dashboard_pairs(
    autopsies: list[dict],
    *,
    prefer_symbols: Optional[tuple[str, ...]] = None,
) -> list[dict[str, Any]]:
    """Per-pair board rows: issues + efficiency meter score.

    When prefer_symbols is set (live INSTRUMENT_UNIVERSE), only those pairs
    are shown — drops legacy book names like SILVER / EURUSD / GBPUSD.
    """
    agg = aggregate(autopsies)
    allow = {s.upper() for s in (prefer_symbols or ()) if s}
    rows: list[dict[str, Any]] = []
    for sym, st in agg["per_symbol"].items():
        if sym in ("?", ""):
            continue
        if allow and sym.upper() not in allow:
            continue
        tag_counts = Counter({t: c for t, c in st["top_tags"]})
        score = efficiency_score(
            wins=int(st["wins"]),
            losses=int(st["losses"]),
            pnl=float(st["pnl"]),
            tag_counts=tag_counts,
        )
        # Prefer loss-linked issues; skip clean_win; prioritize real exits over legacy manual
        prefer = (
            "stop_loss_hit",
            "take_profit_hit",
            "soft_mc_entry",
            "stretched_rr_loss",
            "off_prime_hour",
            "fast_tf_loss",
            "large_htf_loss",
            "news_against",
            "bias_against",
            "smc_weak",
            "broker_unclear_exit",
            "desk_force_close",
            "true_manual_close",
        )
        ranked = sorted(
            st["top_tags"],
            key=lambda tc: (
                prefer.index(tc[0]) if tc[0] in prefer else 50,
                -tc[1],
            ),
        )
        issues = []
        for tag, cnt in ranked:
            if tag in ("clean_win_profile", "manual_or_sync_close"):
                continue
            label = TAG_LAYMAN.get(tag, tag.replace("_", " "))
            issues.append({"tag": tag, "label": label, "count": cnt})
            if len(issues) >= 3:
                break
        if not issues and st["wins"] > st["losses"]:
            issues = [
                {
                    "tag": "ok",
                    "label": "No major red flags in recent closes",
                    "count": st["n"],
                }
            ]
        tip = None
        if st["top_loss_suggestions"]:
            tip = st["top_loss_suggestions"][0][0]
        rows.append(
            {
                "symbol": sym,
                "n": st["n"],
                "wins": st["wins"],
                "losses": st["losses"],
                "pnl": st["pnl"],
                "efficiency": score,
                "issues": issues,
                "tip": tip,
            }
        )

    # Keep universe order; append any extras only if no allow-list
    rank = {s.upper(): i for i, s in enumerate(prefer_symbols or ())}
    rows.sort(key=lambda r: (rank.get(r["symbol"].upper(), 100), -r["efficiency"], r["symbol"]))

    # Ensure every live universe symbol appears (even with zero closes)
    if allow:
        have = {r["symbol"].upper() for r in rows}
        for sym in prefer_symbols or ():
            su = sym.upper()
            if su in have:
                continue
            rows.append(
                {
                    "symbol": su,
                    "n": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0.0,
                    "efficiency": 5,
                    "no_data": True,
                    "issues": [
                        {
                            "tag": "no_closes",
                            "label": "No closed trades in this window yet",
                            "count": 0,
                        }
                    ],
                    "tip": None,
                }
            )
        rows.sort(
            key=lambda r: (rank.get(r["symbol"].upper(), 100), -r["efficiency"], r["symbol"])
        )
    return rows


def build_dashboard_payload(
    rows: list[dict],
    *,
    days: int = 14,
    prefer_symbols: Optional[tuple[str, ...]] = None,
    desk_reset_at: Optional[str] = None,
    since_desk_reset: bool = False,
) -> dict[str, Any]:
    from analysis.trade_outcome import (
        is_breakeven_trade,
        is_losing_trade,
        is_winning_trade,
    )

    autopsies = [diagnose(r) for r in rows]
    pairs = dashboard_pairs(autopsies, prefer_symbols=prefer_symbols)

    # Lightweight W/BE/L for the board header (same classifiers as TradeStatsBar)
    class _Row:
        def __init__(self, d: dict):
            self.signal = d.get("signal")
            self.status = d.get("status") or "closed"
            self.pnl = d.get("pnl")
            self.pnl_status = d.get("pnl_status")
            self.message = d.get("message")
            self.entry = d.get("entry")
            self.sl = d.get("sl")
            self.symbol = d.get("symbol")
            self.lot_size = d.get("lot_size")
            self.features = d.get("features")

    wins = losses = be = 0
    for r in rows:
        if str(r.get("signal") or "") not in ("BUY", "SELL"):
            continue
        wrap = _Row(r)
        if is_winning_trade(wrap):
            wins += 1
        elif is_breakeven_trade(wrap):
            be += 1
        elif is_losing_trade(wrap):
            losses += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "n": len(autopsies),
        "wins": wins,
        "breakevens": be,
        "losses": losses,
        "mode": "read_only",
        "since_desk_reset": bool(since_desk_reset),
        "desk_reset_at": desk_reset_at,
        "pairs": pairs,
        "desk_efficiency": (
            int(round(sum(p["efficiency"] for p in pairs) / len(pairs)))
            if pairs
            else 5
        ),
    }


def render_md(payload: dict) -> str:
    lines = []
    lines.append("# Trade Autopsy Report (read-only)")
    lines.append("")
    lines.append(f"- Window: last **{payload['days']}** days")
    lines.append(f"- Closed trades scored: **{payload['n']}**")
    lines.append(f"- Generated: `{payload['generated_at']}`")
    lines.append("- **No live changes applied** — suggestions are research-only.")
    lines.append("")
    lines.append("## Desk patterns")
    lines.append("")
    lines.append("| Tag | Count |")
    lines.append("|---|---:|")
    for tag, n in payload["aggregate"]["tag_counts"]:
        lines.append(f"| `{tag}` | {n} |")
    lines.append("")
    lines.append("## Per symbol")
    lines.append("")
    lines.append("| Symbol | n | W | L | PnL $ | Top tags |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for sym, st in sorted(payload["aggregate"]["per_symbol"].items()):
        tags = ", ".join(f"{t}({c})" for t, c in st["top_tags"][:3]) or "—"
        lines.append(
            f"| {sym} | {st['n']} | {st['wins']} | {st['losses']} | {st['pnl']} | {tags} |"
        )
    lines.append("")
    lines.append("## Research suggestions (do not auto-apply)")
    lines.append("")
    if not payload["aggregate"]["suggestion_counts"]:
        lines.append("_No loss-linked suggestions._")
    else:
        for s, n in payload["aggregate"]["suggestion_counts"]:
            lines.append(f"- ({n}×) {s}")
    lines.append("")
    lines.append("## Recent trade autopsies")
    lines.append("")
    for a in payload["autopsies"][:40]:
        lines.append(
            f"### #{a['id']} {a['symbol']} {a['side']} · {a['outcome']} · "
            f"${a['pnl']} · TF={a.get('exec_tf')} · MC={a.get('mc_prob')}"
        )
        lines.append(f"- Tags: {', '.join(a['tags'])}")
        for n in a["notes"]:
            lines.append(f"- {n}")
        if a.get("suggestion"):
            lines.append(f"- Suggestion: _{a['suggestion']}_")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only closed-trade autopsy agent")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    print(
        f"Trade autopsy agent (READ-ONLY) days={args.days} limit={args.limit}",
        flush=True,
    )
    rows = _fetch_closes(args.days, args.limit)
    print(f"Loaded {len(rows)} closed trades", flush=True)

    autopsies = [diagnose(r) for r in rows]
    agg = aggregate(autopsies)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": args.days,
        "n": len(autopsies),
        "mode": "read_only",
        "live_changes": False,
        "aggregate": agg,
        "autopsies": autopsies,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    OUT_MD.write_text(render_md(payload))
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Wrote {OUT_MD}", flush=True)

    print("\n### TOP LOSS SUGGESTIONS (research only)\n", flush=True)
    for s, n in agg["suggestion_counts"][:8]:
        print(f"  ({n}×) {s}", flush=True)
    print("\n### PER SYMBOL\n", flush=True)
    for sym, st in sorted(agg["per_symbol"].items()):
        print(
            f"  {sym:8s} n={st['n']:3d} W{st['wins']}/L{st['losses']} "
            f"pnl=${st['pnl']:+.0f}  tags={st['top_tags'][:3]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
