#!/bin/bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=nscc_common.sh
source "$SCRIPT_DIR/nscc_common.sh"
nscc_require_socket

ssh -S "$NSCC_SOCKET" "$NSCC_TARGET" '
  set -e
  echo "USER=$(whoami)"
  echo "HOST=$(hostname)"
  echo "GROUPS=$(id -Gn)"
  echo "FILESYSTEMS"
  df -h .
  echo "PBS"
  command -v qsub
  qstat -Q 2>/dev/null | head -20 || true
  echo "SINGULARITY"
  module load singularity >/dev/null 2>&1
  singularity --version
'

