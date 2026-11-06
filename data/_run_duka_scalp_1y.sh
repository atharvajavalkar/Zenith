#!/bin/bash
set -e
cd /app
LOCKDIR=/app/data/lab_duka_scalp_1y_hunt.lock.dir
LOG=/app/data/lab_duka_scalp_1y_hunt.log
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "already running" >> "$LOG"
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
: > "$LOG"
exec python -u -m analysis.strategy_lab_duka_scalp_1y_hunt >> "$LOG" 2>&1
