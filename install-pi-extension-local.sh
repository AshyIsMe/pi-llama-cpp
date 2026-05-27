#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PI_AGENT_DIR=${PI_CODING_AGENT_DIR:-"$HOME/.pi/agent"}
EXTENSION_DIR="$PI_AGENT_DIR/extensions"
EXTENSION_LINK="$EXTENSION_DIR/pi-llama-cpp"

if [ ! -f "$ROOT/index.ts" ]; then
  echo "error: $ROOT/index.ts not found" >&2
  exit 1
fi

mkdir -p "$EXTENSION_DIR" "$HOME/.pi/llama-cpp"
ln -sfn "$ROOT" "$EXTENSION_LINK"

echo "Installed pi extension package symlink:"
echo "  $EXTENSION_LINK -> $ROOT"
echo
echo "Reload pi with /reload or start pi normally; the extension is auto-discovered."
