#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SERVICE=qwen3.8-flash-next.service

if systemctl --user is-active --quiet "$SERVICE"; then
  echo "status=running service=$SERVICE"
elif [[ -f "$ROOT/llama-server.pid" ]] && kill -0 "$(cat "$ROOT/llama-server.pid")" 2>/dev/null; then
  echo "status=running pid=$(cat "$ROOT/llama-server.pid")"
else
  echo "status=stopped"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
