#!/bin/bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=nscc_common.sh
source "$SCRIPT_DIR/nscc_common.sh"
nscc_require_socket

JOB_ID="${1:-}"
if [[ -z "$JOB_ID" && -f "$NSCC_REPO_ROOT/.nscc_last_job" ]]; then
  JOB_ID=$(tr -d '[:space:]' < "$NSCC_REPO_ROOT/.nscc_last_job")
fi
if [[ -z "$JOB_ID" ]]; then
  echo "Pass a PBS job ID or submit a job first." >&2
  exit 2
fi
ssh -S "$NSCC_SOCKET" "$NSCC_TARGET" "qstat -f '$JOB_ID'"

