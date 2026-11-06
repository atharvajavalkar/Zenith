"""Stop running duka horizons battery processes (module form only)."""
from __future__ import annotations

import glob
import os
import sys


def main() -> int:
    pids: list[int] = []
    for path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            cmd = open(path, "rb").read().replace(b"\0", b" ").decode()
        except Exception:
            continue
        if "-m analysis._duka_brains_horizons_bt" in cmd:
            pids.append(int(path.split("/")[2]))
    print("targets", pids, flush=True)
    for p in pids:
        try:
            os.kill(p, 9)
            print("killed", p, flush=True)
        except Exception as exc:
            print("fail", p, exc, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
