#!/usr/bin/env bash
# Keeps the REST-5 Nebula hunt running (AUD/CHF/ETH/NAS/JPY).
# Does not touch live GOLD/BTCUSD/UKOIL trading.
# Resume-safe via partial JSON.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/data/lab_adaptive_xm_rest5_hunt.console.log"
WATCH="$ROOT/data/lab_adaptive_xm_rest5_hunt.watchdog.log"
CONTAINER="${ALGO_BACKEND_CONTAINER:-algo-backend}"

is_running() {
  docker top "$CONTAINER" 2>/dev/null | grep -F 'lab_adaptive_xm_rest5_hunt_bt' | grep -q python
}

start_hunt() {
  if is_running; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start skipped — already running" >>"$WATCH"
    return 0
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting detached REST-5 hunt" >>"$WATCH"
  docker exec -d -w /app \
    -e PYTHONPATH=/app:/app/backend \
    -e PYTHONUNBUFFERED=1 \
    "$CONTAINER" \
    sh -c "nice -n 10 python -u -m analysis.lab_adaptive_xm_rest5_hunt_bt >> /app/data/lab_adaptive_xm_rest5_hunt.console.log 2>&1"
}

mkdir -p "$ROOT/data"
touch "$LOG" "$WATCH"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) rest5 watchdog up · polling every 60s" >>"$WATCH"

if ! is_running; then
  start_hunt
else
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) hunt already running — leave it" >>"$WATCH"
fi

while true; do
  sleep 60
  if ! is_running; then
    if [[ -f "$ROOT/data/lab_adaptive_xm_rest5_hunt.json" ]]; then
      st=$(python3 -c "import json;print(json.load(open('$ROOT/data/lab_adaptive_xm_rest5_hunt.progress.json')).get('status',''))" 2>/dev/null || true)
      if [[ "$st" == "done" || "$st" == "complete" ]]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) rest5 hunt complete — watchdog exit" >>"$WATCH"
        exit 0
      fi
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) hunt missing — resume restart" >>"$WATCH"
    start_hunt
  fi
done
