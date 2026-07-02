#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT"
export PATH="$ROOT/algo-env/bin:$PATH"

cd "$ROOT/backend"

if [ ! -f "$ROOT/.env" ]; then
  echo "Warning: .env not found. Copy .env.example to .env"
fi

echo "Starting backend at http://localhost:8000"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
