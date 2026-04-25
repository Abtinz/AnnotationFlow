#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if [ -f ../.env ]; then
  set -a
  . ../.env
  set +a
fi

uv run uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000
