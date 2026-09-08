#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNIT_NAME=qwen3.8-flash-next.service
UNIT_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
UNIT_PATH=$UNIT_DIR/$UNIT_NAME
SERVICE_HOST=${SERVICE_HOST:-127.0.0.1}

escaped_root=${ROOT//\\/\\\\}
escaped_root=${escaped_root//&/\\&}
escaped_root=${escaped_root//|/\\|}
escaped_host=${SERVICE_HOST//\\/\\\\}
escaped_host=${escaped_host//&/\\&}
escaped_host=${escaped_host//|/\\|}

mkdir -p "$UNIT_DIR"
sed \
  -e "s|@DEPLOY_ROOT@|$escaped_root|g" \
  -e "s|@HOST@|$escaped_host|g" \
  "$ROOT/$UNIT_NAME" >"$UNIT_PATH"
systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"
systemctl --user restart "$UNIT_NAME"
systemctl --user --no-pager --full status "$UNIT_NAME"
