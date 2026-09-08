#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE=${1:-3gpu}
MODEL=${MODEL:-$ROOT/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
MTP_MODEL=${MTP_MODEL:-$ROOT/models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
PORT=${PORT:-8001}
HOST=${HOST:-127.0.0.1}
CONTEXT=${CONTEXT:-262144}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
SPLIT_MODE=${SPLIT_MODE:-layer}
TENSOR_SPLIT=${TENSOR_SPLIT:-}
N_GPU_LAYERS=${N_GPU_LAYERS:-}
SPEC_MTP=${SPEC_MTP:-1}
DRAFT_N=${DRAFT_N:-4}
DRAFT_CACHE_TYPE=${DRAFT_CACHE_TYPE:-}
API_KEY_FILE=${API_KEY_FILE:-$ROOT/.api-key}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$ROOT/qwen3.8-flash-next-codex.jinja}

case "$PROFILE" in
  1gpu)
    default_gpus=4
    default_cpu_set=32-63,96-127
    default_ngl=32
    ;;
  2gpu)
    default_gpus=4,5
    default_cpu_set=32-63,96-127
    default_ngl=999
    ;;
  3gpu)
    default_gpus=4,5,6
    default_cpu_set=32-63,96-127
    default_ngl=999
    ;;
  4gpu)
    default_gpus=4,5,6,7
    default_cpu_set=32-63,96-127
    default_ngl=999
    ;;
  4gpu-numa0)
    default_gpus=0,1,2,3
    default_cpu_set=0-31,64-95
    default_ngl=999
    ;;
  8gpu)
    default_gpus=0,1,2,3,4,5,6,7
    default_cpu_set=
    default_ngl=999
    ;;
  *) echo "Usage: $0 {1gpu|2gpu|3gpu|4gpu|4gpu-numa0|8gpu}" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=${GPUS:-$default_gpus}
CPU_SET=${CPU_SET-$default_cpu_set}
N_GPU_LAYERS=${N_GPU_LAYERS:-$default_ngl}
gpu_count=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
if [[ -n "$TENSOR_SPLIT" ]]; then
  tensor_split=$TENSOR_SPLIT
else
  tensor_split=1
  for ((i = 1; i < gpu_count; i++)); do
    tensor_split+=,1
  done
fi

if [[ "${ALLOW_BUSY_GPUS:-0}" != 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t busy_uuids < <(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null |
      sort -u
  )
  for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
    uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits)
    if printf '%s\n' "${busy_uuids[@]}" | grep -Fxq "$uuid"; then
      echo "Refusing to use busy GPU $gpu. Set ALLOW_BUSY_GPUS=1 to override." >&2
      exit 1
    fi
  done
fi

[[ -x "$ROOT/llama.cpp/build/bin/llama-server" ]] || {
  echo "llama-server not found; run tools/build-llama.sh first." >&2
  exit 1
}
[[ -s "$MODEL" ]] || { echo "Model not found: $MODEL" >&2; exit 1; }

args=(
  -m "$MODEL"
  -ngl "$N_GPU_LAYERS"
  --split-mode "$SPLIT_MODE"
  --tensor-split "$tensor_split"
  --ctx-size "$CONTEXT"
  --batch-size "$BATCH_SIZE"
  --ubatch-size "$UBATCH_SIZE"
  --flash-attn on
  --parallel 1
  --host "$HOST"
  --port "$PORT"
  --jinja
  --chat-template-file "$CHAT_TEMPLATE"
  --reasoning-format deepseek
  --alias qwen3.8-flash-next
  --cache-prompt
  --metrics
)
[[ -s "$API_KEY_FILE" ]] && args+=(--api-key-file "$API_KEY_FILE")
if [[ "$SPEC_MTP" == 1 ]]; then
  [[ -s "$MTP_MODEL" ]] || { echo "MTP model not found: $MTP_MODEL" >&2; exit 1; }
  args+=(
    --spec-type draft-mtp
    --spec-draft-model "$MTP_MODEL"
    --spec-draft-n-max "$DRAFT_N"
    --spec-draft-ngl 999
  )
  if [[ -n "$DRAFT_CACHE_TYPE" ]]; then
    args+=(
      --cache-type-k-draft "$DRAFT_CACHE_TYPE"
      --cache-type-v-draft "$DRAFT_CACHE_TYPE"
    )
  fi
fi

if [[ -n "$CPU_SET" ]]; then
  exec taskset -c "$CPU_SET" "$ROOT/llama.cpp/build/bin/llama-server" "${args[@]}"
fi
exec "$ROOT/llama.cpp/build/bin/llama-server" "${args[@]}"
