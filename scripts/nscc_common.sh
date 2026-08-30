#!/bin/bash

set -euo pipefail

NSCC_COMMON_SOURCE="${BASH_SOURCE[0]:-$0}"
NSCC_REPO_ROOT=$(cd "$(dirname "$NSCC_COMMON_SOURCE")/.." && pwd)
NSCC_ENV_FILE="${NSCC_ENV_FILE:-$NSCC_REPO_ROOT/cluster/nscc.env}"
if [[ -f "$NSCC_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$NSCC_ENV_FILE"
fi

NSCC_USER="${NSCC_USER:-rishika0}"
NSCC_HOST="${NSCC_HOST:-aspire2antu.nscc.sg}"
NSCC_REMOTE_DIR="${NSCC_REMOTE_DIR:-TikTok-TechJam-2026-NTUWIT}"
NSCC_SOCKET="${NSCC_SOCKET:-${TMPDIR:-/tmp}/nscc-${NSCC_USER}.sock}"
NSCC_TARGET="$NSCC_USER@$NSCC_HOST"

nscc_require_socket() {
  if [[ ! -S "$NSCC_SOCKET" ]]; then
    echo "No authenticated SSH socket at $NSCC_SOCKET" >&2
    echo "Run scripts/nscc_login.sh in your Terminal and enter the password there." >&2
    exit 2
  fi
}
