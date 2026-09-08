#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY=${REPOSITORY:-unsloth/Qwen3.8-Flash-Next-GGUF}
DOWNLOAD_BASE=${DOWNLOAD_BASE:-https://modelscope.cn/models/$REPOSITORY/resolve/master}
PARALLEL=${PARALLEL:-1}

download() {
  local path=$1
  local size=$2
  "$ROOT/tools/range-download.sh" \
    "$DOWNLOAD_BASE/$path" \
    "$ROOT/models/$path" \
    "$size" \
    "$PARALLEL"
}

download UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf 10946624
download UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf 49835229856
download UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf 43836407744
download MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf 2786568256

cd "$ROOT"
sha256sum --check model-checksums.sha256
