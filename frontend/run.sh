#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
npm run dev -- --host 127.0.0.1 --port 8081

