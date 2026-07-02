#!/usr/bin/env bash
# Start Zenith (rebuild + run). Run from anywhere:
#   bash /Users/atharvajavalkar/Desktop/Algo/Algo/start.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Open Docker Desktop first, then run this script again."
  exit 1
fi

echo "Building and starting Zenith from $ROOT ..."
docker compose up -d --build

echo ""
echo "Waiting for backend..."
for _ in $(seq 1 30); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo ""
echo "Zenith is up:"
echo "  Dashboard:  http://localhost:3000/dashboard"
echo "  Login:      http://localhost:3000/login"
echo "  API:        http://localhost:8000"
echo "  API docs:   http://localhost:8000/docs"
echo ""
docker compose ps
