#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add your Roboflow API key before running real jobs."
fi

./backend/setup.sh
./frontend/setup.sh

echo "Setup complete."

