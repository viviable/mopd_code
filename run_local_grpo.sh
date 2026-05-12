#!/bin/bash

# Usage: ./run_local_grpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="baseline_grpo"

# Default to ToolUse dataset
# DATA_PATH="datasets/tooluse"
# DATA_PATH="datasets/sciknoweval/biology"
# DATA_PATH="datasets/sciknoweval/chemistry"
# DATA_PATH="datasets/sciknoweval/physics"
# DATA_PATH="datasets/sciknoweval/material"
DATA_PATH="datasets/lcb_v6"

# Hyperparameters (from experiments/run_baseline_grpo_all.sh)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
MINI_BATCH_SIZE=32
LR=1e-5
MODEL_PATH="Qwen/Qwen3-4B"
SEED=42

# Local-safe default: 1 GPU unless user explicitly pins devices.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ "${CUDA_VISIBLE_DEVICES}" = "-1" ]; then
        VISIBLE_GPUS=0
    else
        VISIBLE_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^\s*$/d' | wc -l)
    fi
else
    VISIBLE_GPUS=1
fi

if [ "${VISIBLE_GPUS}" -lt 1 ]; then
    echo "No visible GPUs detected. This script requires at least 1 GPU."
    exit 1
fi

# Allow overriding from shell, but clamp to visible devices to avoid vLLM init assertion.
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$VISIBLE_GPUS}
if [ "${N_GPUS_PER_NODE}" -gt "${VISIBLE_GPUS}" ]; then
    echo "N_GPUS_PER_NODE (${N_GPUS_PER_NODE}) > visible GPUs (${VISIBLE_GPUS}); clamping to ${VISIBLE_GPUS}."
    export N_GPUS_PER_NODE=${VISIBLE_GPUS}
fi

ROLLOUT_TP_SIZE=1
if [ "${ROLLOUT_TP_SIZE}" -gt "${VISIBLE_GPUS}" ]; then
    echo "ROLLOUT_TP_SIZE (${ROLLOUT_TP_SIZE}) > visible GPUs (${VISIBLE_GPUS}); clamping to ${VISIBLE_GPUS}."
    ROLLOUT_TP_SIZE=${VISIBLE_GPUS}
fi

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_grpo"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

resolve_cuda_toolkit_for_torch() {
    if ! command -v python >/dev/null 2>&1; then
        return 0
    fi

    local torch_cuda_version=""
    torch_cuda_version=$(python - <<'PY' 2>/dev/null
import torch
print(torch.version.cuda or "")
PY
)

    if [ -z "$torch_cuda_version" ]; then
        return 0
    fi

    local preferred_cuda_home="/usr/local/cuda-${torch_cuda_version}"
    if [ ! -d "$preferred_cuda_home" ]; then
        echo "PyTorch expects CUDA ${torch_cuda_version}, but ${preferred_cuda_home} is not installed. Leaving current CUDA toolkit unchanged."
        return 0
    fi

    export CUDA_HOME="$preferred_cuda_home"
    export PATH="$CUDA_HOME/bin:$PATH"
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
    fi

    echo "Resolved CUDA toolkit from PyTorch: torch.version.cuda=${torch_cuda_version}, CUDA_HOME=${CUDA_HOME}"
}

resolve_cuda_toolkit_for_torch

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}
export WANDB_ENTITY="safety"

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="${DATA_PATH##*/}-GRPO-mbs${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${MODEL_NAME}-${SUFFIX}"
RUN_TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="$PROJECT_ROOT/logs/local_runs"
LOG_FILE="$LOG_DIR/${EXP_NAME}-${RUN_TS}.log"

mkdir -p "$LOG_DIR"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.max_prompt_length=2048 \
data.seed=$SEED \
trainer.group_name=GRPO-local \
trainer.project_name=sdpo_base \
trainer.logger=[console,wandb] \
trainer.val_before_train=True \
trainer.test_freq=5 \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.actor.data_loader_seed=$SEED \
actor_rollout_ref.actor.fsdp_config.seed=$SEED \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
critic.data_loader_seed=$SEED \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.use_remove_padding=False \
+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
+critic.model.override_config.attn_implementation=flash_attention_2 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=8"

echo "----------------------------------------------------------------"
echo "Starting Local GRPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Seed: $SEED"
echo "Log file: $LOG_FILE"
echo "Resolved GPUs: visible=${VISIBLE_GPUS}, trainer.n_gpus_per_node=${N_GPUS_PER_NODE}, rollout.tp=${ROLLOUT_TP_SIZE}"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
exit $EXIT_CODE
