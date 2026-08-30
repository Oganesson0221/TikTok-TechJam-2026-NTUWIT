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

JOB_KIND="${1:-agent}"
case "$JOB_KIND" in
  agent) PBS_FILE=cluster/nscc_train.pbs ;;
  deepfm) PBS_FILE=cluster/nscc_single_model.pbs ;;
  dcn) PBS_FILE=cluster/nscc_single_model.pbs ;;
  dcn-retry) PBS_FILE=cluster/nscc_train.pbs ;;
  final-sweep) PBS_FILE=cluster/nscc_train.pbs ;;
  hardened-sweep) PBS_FILE=cluster/nscc_train.pbs ;;
  bonus-prepare) PBS_FILE=cluster/nscc_prepare_bonus.pbs ;;
  bonus-sweep) PBS_FILE=cluster/nscc_train.pbs ;;
  pairwise-sweep) PBS_FILE=cluster/nscc_train.pbs ;;
  bonus-side-sweep) PBS_FILE=cluster/nscc_train.pbs ;;
  bonus-export) PBS_FILE=cluster/nscc_export_bonus.pbs ;;
  openai-selected) PBS_FILE=cluster/nscc_train.pbs ;;
  *) echo "Usage: $0 [agent|deepfm|dcn|dcn-retry|final-sweep|hardened-sweep|bonus-prepare|bonus-sweep|bonus-side-sweep|bonus-export|openai-selected|pairwise-sweep]" >&2; exit 2 ;;
esac

VARIABLES=""
EXTRA_QSUB=""
if [[ "$JOB_KIND" == "deepfm" ]]; then
  VARIABLES="-v TRAIN_CONFIG=configs/nscc_deepfm_features.json"
elif [[ "$JOB_KIND" == "dcn" ]]; then
  VARIABLES="-v TRAIN_CONFIG=configs/nscc_dcn_features.json"
elif [[ "$JOB_KIND" == "dcn-retry" ]]; then
  VARIABLES="-v AGENT_CONFIG=configs/nscc_dcn_retry_agent.json"
  EXTRA_QSUB="-l walltime=01:00:00"
elif [[ "$JOB_KIND" == "final-sweep" ]]; then
  VARIABLES="-v AGENT_CONFIG=configs/nscc_final_sweep.json"
  EXTRA_QSUB="-l walltime=01:00:00"
elif [[ "$JOB_KIND" == "hardened-sweep" ]]; then
  VARIABLES="-v AGENT_CONFIG=configs/nscc_hardened_sweep.json"
  EXTRA_QSUB="-l walltime=02:00:00"
elif [[ "$JOB_KIND" == "bonus-sweep" ]]; then
  VARIABLES="-v AGENT_CONFIG=configs/nscc_1k_bonus_sweep.json"
  EXTRA_QSUB="-l walltime=03:00:00"
elif [[ "$JOB_KIND" == "pairwise-sweep" ]]; then
  VARIABLES="-v AGENT_CONFIG=configs/nscc_pairwise_sweep.json"
  EXTRA_QSUB="-l walltime=02:00:00"
elif [[ "$JOB_KIND" == "bonus-side-sweep" ]]; then
  VARIABLES="-v AGENT_CONFIG=configs/nscc_1k_side_sweep.json"
  EXTRA_QSUB="-l walltime=03:00:00"
elif [[ "$JOB_KIND" == "bonus-export" ]]; then
  EXPERIMENT_NAME="${2:-}"
  if [[ ! "$EXPERIMENT_NAME" =~ ^kuairand_1k_[A-Za-z0-9_-]+$ ]]; then
    echo "Usage: $0 bonus-export kuairand_1k_EXPERIMENT_NAME" >&2
    exit 2
  fi
  VARIABLES="-v EXPERIMENT_NAME=$EXPERIMENT_NAME"
  EXTRA_QSUB="-l walltime=01:00:00"
elif [[ "$JOB_KIND" == "openai-selected" ]]; then
  VARIABLES="-v AGENT_CONFIG=configs/nscc_openai_selected_seed44.json"
  EXTRA_QSUB="-l walltime=01:00:00"
fi

DEPENDENCY=""
if [[ -f "$NSCC_REPO_ROOT/.nscc_bootstrap_job" ]]; then
  BOOTSTRAP_JOB=$(tr -d '[:space:]' < "$NSCC_REPO_ROOT/.nscc_bootstrap_job")
  if [[ -n "$BOOTSTRAP_JOB" ]]; then
    DEPENDENCY="-W depend=afterok:$BOOTSTRAP_JOB"
  fi
fi
if [[ ( "$JOB_KIND" == "bonus-sweep" || "$JOB_KIND" == "bonus-side-sweep" ) && -f "$NSCC_REPO_ROOT/.nscc_bonus_prepare_job" ]]; then
  BONUS_PREPARE_JOB=$(tr -d '[:space:]' < "$NSCC_REPO_ROOT/.nscc_bonus_prepare_job")
  if [[ -n "$BONUS_PREPARE_JOB" ]]; then
    DEPENDENCY="-W depend=afterok:$BONUS_PREPARE_JOB"
  fi
fi

JOB_ID=$(ssh -S "$NSCC_SOCKET" "$NSCC_TARGET" \
  "cd '$NSCC_REMOTE_DIR' && qsub -P '$NSCC_PROJECT_ID' $DEPENDENCY $EXTRA_QSUB $VARIABLES '$PBS_FILE'")
echo "$JOB_ID" | tee "$NSCC_REPO_ROOT/.nscc_last_job"
if [[ "$JOB_KIND" == "bonus-prepare" ]]; then
  echo "$JOB_ID" > "$NSCC_REPO_ROOT/.nscc_bonus_prepare_job"
fi
echo "Submitted $JOB_KIND job: $JOB_ID"
if [[ -n "$DEPENDENCY" ]]; then
  echo "Job dependency: $DEPENDENCY"
fi
