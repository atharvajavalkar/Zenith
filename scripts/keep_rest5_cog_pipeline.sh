#!/usr/bin/env bash
# Pipeline: one Rest-5 + Cognitive deep hunt for AUD/CHF/ETH/NAS/JPY until BOTH done.
# Live GOLD/BTCUSD/UKOIL never touched.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${ALGO_BACKEND_CONTAINER:-algo-backend}"
WATCH="$ROOT/data/lab_adaptive_xm_rest5_pipeline.watchdog.log"
REST5_PROG="$ROOT/data/lab_adaptive_xm_rest5_hunt.progress.json"
COG_PROG="$ROOT/data/lab_adaptive_xm_rest5_cog_hunt.progress.json"

mkdir -p "$ROOT/data"
touch "$WATCH"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >>"$WATCH"; }

rest5_running() {
  docker top "$CONTAINER" 2>/dev/null | grep 'lab_adaptive_xm_rest5_hunt_bt' | grep -v cog | grep -q python
}
cog_running() {
  docker top "$CONTAINER" 2>/dev/null | grep -F 'lab_adaptive_xm_rest5_cog_hunt_bt' | grep -q python
}
prog_done() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  python3 -c "import json;print(json.load(open('$f')).get('status',''))" 2>/dev/null | grep -qx done
}

dedupe_rest5() {
  docker exec "$CONTAINER" python -c "
import os
def dedupe(substr, exclude=None):
  found=[]
  for pid in os.listdir('/proc'):
    if not pid.isdigit(): continue
    try:
      cmd=open(f'/proc/{pid}/cmdline','rb').read().replace(b'\\x00',b' ').decode('utf-8','replace')
    except Exception:
      continue
    if substr in cmd and 'python' in cmd and (not exclude or exclude not in cmd):
      found.append(int(pid))
  found=sorted(set(found))
  print(substr, found)
  if len(found)>1:
    keep=found[-1]
    for p in found[:-1]:
      try:
        os.kill(p,9); print('killed',p)
      except Exception as e:
        print('fail',p,e)
    print('kept',keep)
dedupe('lab_adaptive_xm_rest5_hunt_bt', 'cog_hunt')
dedupe('lab_adaptive_xm_rest5_cog_hunt_bt')
" >>"$WATCH" 2>&1 || true
}

start_rest5() {
  if rest5_running || prog_done "$REST5_PROG"; then return 0; fi
  log "starting rest5 hunt"
  docker exec -d -w /app -e PYTHONPATH=/app:/app/backend -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    "$CONTAINER" \
    sh -c "nice -n 10 python -u -m analysis.lab_adaptive_xm_rest5_hunt_bt >> /app/data/lab_adaptive_xm_rest5_hunt.console.log 2>&1"
}

start_cog() {
  if cog_running || prog_done "$COG_PROG"; then return 0; fi
  log "starting COGNITIVE deep hunt (zenith+bayes+HMM+math-omni)"
  docker exec -d -w /app -e PYTHONPATH=/app:/app/backend -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    "$CONTAINER" \
    sh -c "nice -n 10 python -u -m analysis.lab_adaptive_xm_rest5_cog_hunt_bt >> /app/data/lab_adaptive_xm_rest5_cog_hunt.console.log 2>&1"
}

log "pipeline watchdog up — rest5 + cog in parallel until both done"
dedupe_rest5
start_rest5
start_cog

while true; do
  sleep 60
  dedupe_rest5

  r_done=0
  c_done=0
  prog_done "$REST5_PROG" && r_done=1
  prog_done "$COG_PROG" && c_done=1

  if [[ "$r_done" -eq 1 && "$c_done" -eq 1 ]]; then
    log "BOTH hunts complete — pipeline exit"
    exit 0
  fi

  if [[ "$r_done" -eq 0 ]] && ! rest5_running; then
    log "rest5 missing — resume"
    start_rest5
  fi
  if [[ "$c_done" -eq 0 ]] && ! cog_running; then
    log "cog missing — resume"
    start_cog
  fi
done
