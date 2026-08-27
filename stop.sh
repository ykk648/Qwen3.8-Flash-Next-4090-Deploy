#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PID_FILE="$ROOT/llama-server.pid"
SERVICE=qwen3.8-flash-next.service

if systemctl --user is-active --quiet "$SERVICE"; then
  systemctl --user stop "$SERVICE"
  rm -f "$PID_FILE"
  echo "Stopped $SERVICE."
  exit 0
fi

[[ -f "$PID_FILE" ]] || { echo "llama-server is stopped."; exit 0; }
pid=$(cat "$PID_FILE")
kill "$pid" 2>/dev/null || true
for _ in $(seq 1 120); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
kill -0 "$pid" 2>/dev/null && { echo "llama-server did not stop." >&2; exit 1; }
rm -f "$PID_FILE"
echo "Stopped llama-server."
