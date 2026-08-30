#!/bin/bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=nscc_common.sh
source "$SCRIPT_DIR/nscc_common.sh"
nscc_require_socket

EXPERIMENT_NAME="${1:-}"
if [[ ! "$EXPERIMENT_NAME" =~ ^kuairand_1k_[A-Za-z0-9_-]+$ ]]; then
  echo "Pass a validated KuaiRand-1K experiment name (kuairand_1k_*)." >&2
  exit 2
fi

mkdir -p "$NSCC_REPO_ROOT/artifacts/checkpoints/$EXPERIMENT_NAME"
rsync -az -e "ssh -S $NSCC_SOCKET" \
  "$NSCC_TARGET:$NSCC_REMOTE_DIR/artifacts/checkpoints/$EXPERIMENT_NAME/" \
  "$NSCC_REPO_ROOT/artifacts/checkpoints/$EXPERIMENT_NAME/"
echo "Fetched selected bonus checkpoint: $EXPERIMENT_NAME"
