#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AZURE_SCRIPT="${SCRIPT_DIR}/verl_training_azure.sh"
STANDARD_SCRIPT="${SCRIPT_DIR}/verl_training_standard.sh"

is_azure="0"
if [ "${USE_AZURE_VERL_TRAINING:-0}" = "1" ]; then
    is_azure="1"
elif [ -n "${AZUREML_RUN_ID:-}" ] || [ -n "${AZ_BATCHAI_JOB_NAME:-}" ]; then
    is_azure="1"
fi

if [ "$is_azure" = "1" ]; then
    TARGET_SCRIPT="$AZURE_SCRIPT"
    echo "[verl_training] using Azure script: $TARGET_SCRIPT"
else
    TARGET_SCRIPT="$STANDARD_SCRIPT"
    echo "[verl_training] using standard script: $TARGET_SCRIPT"
fi

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "[verl_training] target script not found: $TARGET_SCRIPT"
    exit 1
fi

bash "$TARGET_SCRIPT" "$@"
