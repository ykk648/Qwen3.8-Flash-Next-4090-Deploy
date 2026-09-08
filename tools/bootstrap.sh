#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UV_VERSION=${UV_VERSION:-0.12.10}
PYTHON_VERSION=${PYTHON_VERSION:-3}
PYPI_MIRROR=${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}
PYPI_EXTRA_MIRROR=${PYPI_EXTRA_MIRROR:-https://mirrors.aliyun.com/pypi/simple}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/.cache/uv}

if [[ -x "$ROOT/.tools/local/bin/uv" ]]; then
  UV=$ROOT/.tools/local/bin/uv
elif command -v uv >/dev/null 2>&1; then
  UV=$(readlink -f "$(command -v uv)")
else
  mkdir -p "$ROOT/.tools/local"
  python3 -m pip install \
    --disable-pip-version-check \
    --index-url "$PYPI_MIRROR" \
    --extra-index-url "$PYPI_EXTRA_MIRROR" \
    --target "$ROOT/.tools/local" \
    "uv==$UV_VERSION"
  UV=$ROOT/.tools/local/bin/uv
fi
ln -sfn "$UV" "$ROOT/.tools/uv"
UV=$ROOT/.tools/uv

if [[ ! -x "$ROOT/.venv-tools/bin/python" ]]; then
  "$UV" venv --python "$PYTHON_VERSION" "$ROOT/.venv-tools"
fi

UV_INDEX="$PYPI_MIRROR" \
UV_DEFAULT_INDEX="$PYPI_EXTRA_MIRROR" \
  "$UV" pip install \
    --python "$ROOT/.venv-tools/bin/python" \
    'cmake==4.4.3' \
    'ninja==1.13.2'

printf 'uv=%s\n' "$UV"
printf 'python=%s\n' "$ROOT/.venv-tools/bin/python"
