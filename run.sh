#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "Missing .env. Run ./setup.sh first, then edit .env with your Roboflow settings."
  exit 1
fi

./backend/run.sh &
BACKEND_PID=$!

./frontend/run.sh &
FRONTEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Backend:  http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:8081"
echo "Press Ctrl+C to stop both."

wait

