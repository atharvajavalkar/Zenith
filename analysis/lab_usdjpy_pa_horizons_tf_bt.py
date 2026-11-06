#!/usr/bin/env python3
"""RESEARCH ONLY — USDJPY PA TF×horizon battery (same process as USDCHF).

Focus / trial cell = CHF 2m-horizon winner:
  liq=15m · bias=5m+1m · entry=1m · RR=4

Also reports efficient stack (liq=3m|bias=15m+5m|entry=1m RR=2)
and wired bb_2.5 baseline.

Does NOT change live SYMBOL_ENTRY_BRAIN.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import analysis.lab_usdchf_pa_horizons_tf_bt as base  # noqa: E402

base.SYM = "USDJPY"
base.OUT = Path(ROOT) / "data" / "lab_usdjpy_pa_horizons_tf_bt.json"
base.LOG = Path(ROOT) / "data" / "lab_usdjpy_pa_horizons_tf_bt.log"
base.LIVE_PA = {
    "liq_tf": "15m",
    "bias_a": "5m",
    "bias_b": "1m",
    "entry_tf": "1m",
    "rr": 4.0,
}
base.WIRED_BB = {"brain": "bb_2.5", "tf": "1m", "rr": 4.0, "bb_mult": 2.5}


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
