#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$ROOT/algo-env/bin:$PATH"

if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js not found. Install it in algo-env:"
  echo "  conda install -p \"$ROOT/algo-env\" nodejs -y"
  exit 1
fi

cd "$ROOT/frontend"

if [ ! -d node_modules ]; then
  echo "Installing frontend dependencies..."
  npm install
fi

echo "Starting frontend at http://localhost:5173"
npm run dev
