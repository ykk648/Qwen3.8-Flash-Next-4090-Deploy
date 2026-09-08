#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LLAMA_DIR=${LLAMA_DIR:-$ROOT/llama.cpp}
LLAMA_REPOSITORY=${LLAMA_REPOSITORY:-https://github.com/unslothai/llama.cpp.git}
LLAMA_REF=${LLAMA_REF:-refs/pull/144/head}
LLAMA_COMMIT=${LLAMA_COMMIT:-a9e9c3c5fed8a0bb5cc617532d0d16b8f59c13e0}
CUDA_ARCH=${CUDA_ARCH:-89}

[[ -x "$ROOT/.venv-tools/bin/cmake" ]] || {
  echo "Run tools/bootstrap.sh first." >&2
  exit 1
}

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$LLAMA_REPOSITORY" "$LLAMA_DIR"
fi
if ! git -C "$LLAMA_DIR" cat-file -e "$LLAMA_COMMIT^{commit}" 2>/dev/null; then
  git -C "$LLAMA_DIR" fetch origin "$LLAMA_REF"
fi
git -C "$LLAMA_DIR" checkout --detach "$LLAMA_COMMIT"

"$ROOT/.venv-tools/bin/cmake" \
  -S "$LLAMA_DIR" \
  -B "$LLAMA_DIR/build" \
  -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DCMAKE_BUILD_TYPE=Release
"$ROOT/.venv-tools/bin/cmake" --build "$LLAMA_DIR/build" \
  --parallel "$(nproc)" \
  --target llama-server llama-bench

"$LLAMA_DIR/build/bin/llama-server" --version
