#!/usr/bin/env bash
# Deploy Zenith to an always-on VPS (Docker).
#
# Usage:
#   ./scripts/deploy_vps.sh user@YOUR_SERVER_IP
#
# First time on a blank Ubuntu 22.04/24.04 box:
#   curl -fsSL https://get.docker.com | sh
#   sudo usermod -aG docker $USER   # then re-login
#
# This copies code + .env, builds, and starts with docker-compose.prod.yml.
# MetaAPI is cloud — no MT5 Windows box required.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${1:-}"
REMOTE_DIR="${REMOTE_DIR:-/opt/zenith}"

if [[ -z "$HOST" ]]; then
  echo "Usage: $0 user@server-ip"
  echo "Example: $0 root@143.198.x.x"
  exit 1
fi

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Missing $ROOT/.env — copy from .env.example and fill MetaAPI + secrets first."
  exit 1
fi

# Patch CORS / Vite URLs to the server if still localhost
SERVER_IP="$(echo "$HOST" | sed 's/.*@//')"
if ! grep -q "VITE_API_URL=http://${SERVER_IP}:8000" "$ROOT/.env" 2>/dev/null; then
  echo "NOTE: Set these in .env before or after deploy for the dashboard to work remotely:"
  echo "  CORS_ORIGINS=http://${SERVER_IP}:3000"
  echo "  VITE_API_URL=http://${SERVER_IP}:8000"
  echo "  VITE_WS_URL=ws://${SERVER_IP}:8000"
  echo ""
fi

echo "→ Syncing to ${HOST}:${REMOTE_DIR}"
ssh "$HOST" "mkdir -p ${REMOTE_DIR}"
rsync -az --delete \
  --exclude '.git' \
  --exclude 'algo-env' \
  --exclude 'frontend/node_modules' \
  --exclude 'frontend/dist' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'data/*.db' \
  "$ROOT/" "${HOST}:${REMOTE_DIR}/"

# Always ship local .env (secrets stay off git)
scp -q "$ROOT/.env" "${HOST}:${REMOTE_DIR}/.env"

echo "→ Building & starting (prod)"
ssh "$HOST" "cd ${REMOTE_DIR} && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build"

echo ""
echo "Zenith should be up on the VPS:"
echo "  Dashboard:  http://${SERVER_IP}:3000"
echo "  API:        http://${SERVER_IP}:8000"
echo "  Logs:       ssh ${HOST} 'cd ${REMOTE_DIR} && docker compose logs -f backend'"
echo ""
ssh "$HOST" "cd ${REMOTE_DIR} && docker compose -f docker-compose.yml -f docker-compose.prod.yml ps"
