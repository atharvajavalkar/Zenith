#!/usr/bin/env bash
# Keep densifying AUD/ETH invent v2 until both pairs meet criteria (or densify>2).
# Research-only — does not touch live .env / brains.
set -euo pipefail
cd /app
export PYTHONPATH=/app:/app/backend
export PYTHONUNBUFFERED=1
LOG=/app/data/lab_adaptive_xm_aud_eth_invent_pipeline.log
ROUND=/app/data/lab_adaptive_xm_aud_eth_invent_hunt.round.json

echo "=== aud_eth invent pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

for D in 0 1 2; do
  echo "--- densify=$D ---" | tee -a "$LOG"
  export AUD_ETH_DENSIFY=$D
  # clear partial so new round is clean (progress tag changes anyway)
  rm -f /app/data/lab_adaptive_xm_aud_eth_invent_hunt.partial.json
  set +e
  python -u -m analysis.lab_adaptive_xm_aud_eth_invent_hunt_v2_bt \
    >> /app/data/lab_adaptive_xm_aud_eth_invent_hunt.console.log 2>&1
  rc=$?
  set -e
  echo "densify=$D exit=$rc" | tee -a "$LOG"
  if [[ -f "$ROUND" ]]; then
    cat "$ROUND" | tee -a "$LOG"
    met=$(python -c "import json;print(json.load(open('$ROUND')).get('criteria_met'))")
    if [[ "$met" == "True" ]]; then
      echo "CRITERIA MET at densify=$D" | tee -a "$LOG"
      exit 0
    fi
  fi
done

echo "Pipeline finished densify 0..2 without full criteria" | tee -a "$LOG"
exit 2
