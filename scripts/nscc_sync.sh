#!/bin/bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=nscc_common.sh
source "$SCRIPT_DIR/nscc_common.sh"
nscc_require_socket

ssh -S "$NSCC_SOCKET" "$NSCC_TARGET" "mkdir -p '$NSCC_REMOTE_DIR'"
rsync -az --delete \
  -e "ssh -S $NSCC_SOCKET" \
  --exclude .git/ \
  --exclude .env \
  --exclude '.env.*' \
  --exclude .nscc/ \
  --exclude .pytest_cache/ \
  --exclude artifacts/ \
  --exclude data/downloads/ \
  --exclude data/raw/ \
  --exclude data/bonus_1k/ \
  --exclude '*.pyc' \
  "$NSCC_REPO_ROOT/" "$NSCC_TARGET:$NSCC_REMOTE_DIR/"

echo "Synced code and prepared data to $NSCC_TARGET:$NSCC_REMOTE_DIR"
