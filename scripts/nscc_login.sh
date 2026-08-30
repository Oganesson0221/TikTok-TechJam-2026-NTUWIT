#!/bin/bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=nscc_common.sh
source "$SCRIPT_DIR/nscc_common.sh"

echo "Opening password-authenticated NSCC session for $NSCC_TARGET"
echo "Socket: $NSCC_SOCKET"
echo "Type the password only into the SSH prompt; its characters will be invisible."
exec ssh -M -S "$NSCC_SOCKET" -o ControlPersist=4h "$NSCC_TARGET"

