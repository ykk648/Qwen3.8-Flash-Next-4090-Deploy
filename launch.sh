#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE=${1:-4gpu}
MODEL=${MODEL:-$ROOT/models/Qwen3.8-Flash-Next-UD-IQ4_XS-ms/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
PORT=${PORT:-8001}
CONTEXT=${CONTEXT:-131072}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
SPLIT_MODE=${SPLIT_MODE:-layer}
API_KEY_FILE=${API_KEY_FILE:-$ROOT/.api-key}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$ROOT/qwen3.8-flash-next-codex.jinja}

case "$PROFILE" in
  3gpu)
    export CUDA_VISIBLE_DEVICES=0,1,2
    tensor_split=1,1,1
    ;;
  4gpu)
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    tensor_split=1,1,1,1
    ;;
  8gpu)
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    tensor_split=1,1,1,1,1,1,1,1
    ;;
  *) echo "Usage: $0 {3gpu|4gpu|8gpu}" >&2; exit 2 ;;
esac

args=(
  -m "$MODEL"
  -ngl 999
  --split-mode "$SPLIT_MODE"
  --tensor-split "$tensor_split"
  --ctx-size "$CONTEXT"
  --batch-size "$BATCH_SIZE"
  --ubatch-size "$UBATCH_SIZE"
  --flash-attn on
  --parallel 1
  --host 0.0.0.0
  --port "$PORT"
  --jinja
  --chat-template-file "$CHAT_TEMPLATE"
  --reasoning-format deepseek
  --alias qwen3.8-flash-next
  --cache-prompt
  --metrics
)
[[ -s "$API_KEY_FILE" ]] && args+=(--api-key-file "$API_KEY_FILE")

exec "$ROOT/llama.cpp/build/bin/llama-server" "${args[@]}"
