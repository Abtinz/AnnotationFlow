#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if [ -f ../.env ]; then
  set -a
  . ../.env
  set +a
fi

.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
