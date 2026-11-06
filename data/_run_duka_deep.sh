#!/bin/bash
set -e
cd /app
LOCK=/app/data/lab_duka_deep_hunt.lock
LOG=/app/data/lab_duka_deep_hunt.log
# single-instance via mkdir
if ! mkdir "$LOCK.dir" 2>/dev/null; then
  echo "already running" >> "$LOG"
  exit 0
fi
trap 'rmdir "$LOCK.dir" 2>/dev/null || true' EXIT
: > "$LOG"
exec python -u -m analysis.strategy_lab_duka_deep_hunt >> "$LOG" 2>&1
