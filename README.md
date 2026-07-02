# XAUUSD Monte Carlo Algo — Web Dashboard

Live trading signal dashboard with multi-user auth, candlestick charts, and signal history.

## Project Structure

```
Algo/
├── config.py, data/, indicators/, analysis/, strategy/  # Algo core
├── main.py                    # CLI runner (optional)
├── backend/                   # FastAPI API + signal worker
├── frontend/                  # React + Vite dashboard
├── docker-compose.yml
└── .env.example
```

## Quick Start (Local Dev)

### 1. Backend

```bash
conda activate algo-env
pip install -r backend/requirements.txt

export PYTHONPATH="$(pwd)"
cd backend
uvicorn app.main:app --reload --port 8000
```

Uses SQLite by default (`algo.db` in backend/). Override with `DATABASE_URL` in `.env`.

### 2. Frontend

**Option A — Local dev (port 5173)**

Node.js must be available. It's installed in `algo-env` via conda. Use the start script:

```bash
bash scripts/start-frontend.sh
```

Or manually:

```bash
export PATH="$(pwd)/algo-env/bin:$PATH"
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

**Option B — Docker (port 3000)**

If you ran `docker compose up`, the frontend is already served at http://localhost:3000 (not 5173).

## Docker (Production-like)

```bash
cp .env.example .env
# Edit JWT_SECRET in .env

docker compose up -d --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/auth/register` | No | Create account |
| POST | `/api/auth/login` | No | Get JWT tokens |
| POST | `/api/auth/refresh` | Refresh token | Rotate access token |
| GET | `/api/auth/me` | Bearer | Current user |
| GET | `/api/signals/latest` | Bearer | Live signal |
| GET | `/api/signals/history` | Bearer | Past signals |
| GET | `/api/chart/candles?tf=5m` | Bearer | OHLCV candles |
| WS | `/ws/signals?token=...` | JWT query | Live signal push |

## Environment Variables

See `.env.example` for all options.

## Architecture

- **Signal worker** runs inside the FastAPI process, polling every 20s via `run_signal_cycle()`
- **Market cache** holds latest signal + candles in memory
- **WebSocket** broadcasts new signals when a 5M candle closes
- **PostgreSQL** stores users and signal history snapshots
