#!/bin/bash
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
# export RAY_DEBUG=1
ulimit -c 0

export WANDB_ENTITY=${WANDB_ENTITY:-"sample-efficient-rlvr"} # team (allow override)
export EXPERIMENT=${1:-"experiment"}
CONFIG_NAME=${2:-"ppo_trainer"}
export TASK=${3:-"datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"}
export PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}

# Some cluster launchers export ROCm device visibility variables even on CUDA nodes.
# This conflicts with Ray's CUDA_VISIBLE_DEVICES handling and crashes worker startup.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# removes the first three arguments from the command line
if [ "$#" -ge 3 ]; then
    shift 3
else
    echo "Usage: $0 <experiment_name> <config_name> <data_path>"
    echo "Example: $0 test ppo_trainer datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"
    exit 1
fi

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG_NAME"
echo "Task: $TASK"
echo "Project root: $PROJECT_ROOT"
echo "Arguments: $@"

python -m verl.trainer.main_ppo --config-name $CONFIG_NAME "$@"
