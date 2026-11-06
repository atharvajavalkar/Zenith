"""Post self-heal validation — full stochastic pack per healed/bleeding pair.

Runs OOS / WF / MC / Lévy / Malliavin / Brownian / Itô / Martingale / HJM
after a heal cycle so soft/brain changes are stress-checked immediately.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_OUT = Path(__file__).resolve().parents[1] / "data" / "post_heal_battery.json"


def run_post_heal_battery(
    symbols: Optional[list[str]] = None,
    *,
    lookback_days: int = 182,
    mc_paths: int = 800,
) -> dict[str, Any]:
    """Battery for given symbols (default: full desk universe)."""
    from analysis._edge_survive_bt import (
        BOOK,
        BRAINS,
        _fetch_1h,
        _prep,
        _stats,
        monte_carlo,
        verdict,
    )
    from analysis._rr_best_by_pair import collect_entries, simulate
    from analysis.levy_malliavin import levy_malliavin_pack
    from analysis.stochastic_pack import stochastic_pack
    from config import INSTRUMENT_UNIVERSE, symbol_rr

    want = [str(s).upper() for s in (symbols or list(INSTRUMENT_UNIVERSE or BOOK))]
    want = [s for s in want if s]
    if not want:
        want = list(BOOK)

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end - timedelta(days=lookback_days)).isoformat()
    end_s = end.isoformat()

    results: list[dict[str, Any]] = []
    for sym in want:
        try:
            raw = _fetch_1h(sym, start, end_s)
            df = _prep(raw)
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_s).replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if len(df) < 200:
                results.append(
                    {"symbol": sym, "error": f"thin_bars:{len(df)}", "verdict": "DEAD"}
                )
                continue

            rr = float(symbol_rr(sym, default=2.0))
            brain = BRAINS.get(sym, "?")
            n = len(df)
            split = int(n * 0.70)
            entries = collect_entries(sym, df)
            full_tr = simulate(df, entries, rr)
            full = _stats(full_tr)
            oos = _stats(simulate(df, entries, rr, start_i=split))

            wf = []
            for k in range(4):
                test_start = int(n * (0.40 + 0.15 * k))
                test_end = int(n * (0.55 + 0.15 * k)) - 1
                if test_end <= test_start or test_start < 80:
                    continue
                wf.append(
                    {
                        "fold": k + 1,
                        "oos": _stats(
                            simulate(
                                df, entries, rr, start_i=test_start, end_i=test_end
                            )
                        ),
                    }
                )
            torture = {
                "spread_0.15R": _stats(
                    [
                        {
                            **t,
                            "r": round(t["r"] - 0.15, 4),
                            "pnl": round((t["r"] - 0.15) * 200, 2),
                        }
                        for t in full_tr
                    ]
                )
            }
            mc = monte_carlo(full_tr, paths=mc_paths)
            lm = levy_malliavin_pack(full_tr)
            stoch = stochastic_pack(df, full_tr)
            v = verdict(full, oos, wf, torture, mc)
            wf_g = sum(1 for f in wf if f["oos"]["exp_r"] > 0)
            row = {
                "symbol": sym,
                "brain": brain,
                "rr_label": f"1:{rr:g}",
                "full_er": full.get("exp_r"),
                "oos_er": oos.get("exp_r"),
                "n": full.get("n"),
                "wf_green": f"{wf_g}/{len(wf)}",
                "mc_profit": mc.get("prob_profit"),
                "levy_profit": lm["levy"].get("prob_profit"),
                "brownian_profit": stoch["brownian"].get("prob_profit"),
                "ito": stoch["ito"].get("verdict"),
                "martingale": stoch["martingale"].get("verdict"),
                "hjm": stoch["hjm"].get("verdict"),
                "malliavin": lm["malliavin"].get("fragility"),
                "verdict": v,
            }
            results.append(row)
            logger.info(
                "post-heal battery %s %s E[R]=%s OOS=%s MC=%s Lévy=%s BM=%s %s",
                sym,
                row["rr_label"],
                row["full_er"],
                row["oos_er"],
                row["mc_profit"],
                row["levy_profit"],
                row["brownian_profit"],
                v,
            )
        except Exception as exc:
            logger.warning("post-heal battery %s failed: %s", sym, exc)
            results.append({"symbol": sym, "error": str(exc), "verdict": "DEAD"})

    payload = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start, "end": end_s},
        "symbols": want,
        "results": results,
        "summary": {
            "survives": sum(1 for r in results if r.get("verdict") == "SURVIVES"),
            "fragile": sum(1 for r in results if r.get("verdict") == "FRAGILE"),
            "dead": sum(1 for r in results if r.get("verdict") == "DEAD"),
        },
    }
    try:
        _OUT.parent.mkdir(parents=True, exist_ok=True)
        _OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("post-heal battery write failed: %s", exc)
    return payload


def latest_post_heal_battery() -> dict[str, Any]:
    if not _OUT.exists():
        return {"results": [], "note": "no_run_yet"}
    try:
        return json.loads(_OUT.read_text(encoding="utf-8"))
    except Exception:
        return {"results": [], "note": "corrupt"}
