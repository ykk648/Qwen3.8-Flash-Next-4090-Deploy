#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY=${REPOSITORY:-unsloth/Qwen3.8-Flash-Next-GGUF}
DOWNLOAD_BASE=${DOWNLOAD_BASE:-https://modelscope.cn/models/$REPOSITORY/resolve/master}
PARALLEL=${PARALLEL:-1}
QUANT=${QUANT:-UD-Q4_K_XL}

download() {
  local path=$1
  local size=$2
  "$ROOT/tools/range-download.sh" \
    "$DOWNLOAD_BASE/$path" \
    "$ROOT/models/$path" \
    "$size" \
    "$PARALLEL"
}

download_iq4_xs() {
  download UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf 10946624
  download UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf 49835229856
  download UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf 43836407744
}

download_q4_k_xl() {
  download UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf 10946624
  download UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf 49859583136
  download UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf 49376141504
  download UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf 12087983520
}

case "$QUANT" in
  UD-Q4_K_XL) download_q4_k_xl ;;
  UD-IQ4_XS) download_iq4_xs ;;
  all)
    download_q4_k_xl
    download_iq4_xs
    ;;
  *)
    echo "QUANT must be UD-Q4_K_XL, UD-IQ4_XS, or all" >&2
    exit 2
    ;;
esac

download MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf 2786568256

cd "$ROOT"
sha256sum --check --ignore-missing model-checksums.sha256
