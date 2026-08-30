#!/bin/bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=nscc_common.sh
source "$SCRIPT_DIR/nscc_common.sh"
nscc_require_socket

if [[ -z "${NSCC_PROJECT_ID:-}" || "$NSCC_PROJECT_ID" == "YOUR_PROJECT_ID" ]]; then
  echo "Set NSCC_PROJECT_ID in cluster/nscc.env before submission." >&2
  exit 2
fi

JOB_ID=$(ssh -S "$NSCC_SOCKET" "$NSCC_TARGET" \
  "cd '$NSCC_REMOTE_DIR' && qsub -P '$NSCC_PROJECT_ID' cluster/nscc_bootstrap.pbs")
echo "$JOB_ID" | tee "$NSCC_REPO_ROOT/.nscc_bootstrap_job"
echo "Submitted scheduled bootstrap job: $JOB_ID"
