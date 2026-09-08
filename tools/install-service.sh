#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNIT_NAME=qwen3.8-flash-next.service
UNIT_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
UNIT_PATH=$UNIT_DIR/$UNIT_NAME

escaped_root=${ROOT//\\/\\\\}
escaped_root=${escaped_root//&/\\&}
escaped_root=${escaped_root//|/\\|}

mkdir -p "$UNIT_DIR"
sed "s|@DEPLOY_ROOT@|$escaped_root|g" "$ROOT/$UNIT_NAME" >"$UNIT_PATH"
systemctl --user daemon-reload
systemctl --user enable --now "$UNIT_NAME"
systemctl --user --no-pager --full status "$UNIT_NAME"
