#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE=${1:-2gpu}
PID_FILE="$ROOT/llama-server.pid"
LOG_FILE="$ROOT/llama-server.log"
SERVICE=qwen3.8-flash-next.service

if [[ "$PROFILE" == 2gpu ]] && systemctl --user cat "$SERVICE" >/dev/null 2>&1; then
  systemctl --user start "$SERVICE"
  echo "Started $SERVICE."
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "llama-server is already running."
  exit 0
fi
nohup "$ROOT/launch.sh" "$PROFILE" >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "Started $PROFILE with PID $(cat "$PID_FILE")."
