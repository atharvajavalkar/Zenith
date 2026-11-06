"""Stop duplicate duka deep hunt processes; keep none or report."""
from __future__ import annotations

import glob
import os


def main() -> None:
    pids = []
    for path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            cmd = open(path, "rb").read().replace(b"\0", b" ").decode()
        except Exception:
            continue
        if "-m analysis.strategy_lab_duka_deep_hunt" in cmd:
            pids.append(int(path.split("/")[2]))
    print("targets", pids, flush=True)
    for p in pids:
        try:
            os.kill(p, 9)
            print("killed", p, flush=True)
        except Exception as e:
            print("fail", p, e, flush=True)


if __name__ == "__main__":
    main()
