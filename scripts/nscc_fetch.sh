#!/bin/bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=nscc_common.sh
source "$SCRIPT_DIR/nscc_common.sh"
nscc_require_socket

mkdir -p "$NSCC_REPO_ROOT/artifacts/nscc_logs"
rsync -az -e "ssh -S $NSCC_SOCKET" \
  --exclude 'checkpoints/kuairand_1k_*' \
  "$NSCC_TARGET:$NSCC_REMOTE_DIR/artifacts/" "$NSCC_REPO_ROOT/artifacts/"
rsync -az -e "ssh -S $NSCC_SOCKET" \
  --include '*.o*' --include '*.e*' --exclude '*' \
  "$NSCC_TARGET:$NSCC_REMOTE_DIR/" "$NSCC_REPO_ROOT/artifacts/nscc_logs/"
mkdir -p "$NSCC_REPO_ROOT/artifacts/bonus_1k"
rsync -az -e "ssh -S $NSCC_SOCKET" \
  "$NSCC_TARGET:$NSCC_REMOTE_DIR/data/bonus_1k/prepared/manifest.json" \
  "$NSCC_REPO_ROOT/artifacts/bonus_1k/data_manifest.json" 2>/dev/null || true
echo "Fetched NSCC artifacts and PBS logs."
