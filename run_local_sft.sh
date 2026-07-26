#!/bin/bash

# Usage: ./run_local_sft.sh [experiment_name_suffix]
#
# Verified-success SFT baseline for Table 5 (filter-then-SFT / rejection-sampling /
# STaR). Same rollout budget as GRPO/MOPD (n rollouts per prompt); supervised only
# on the self-generated rollouts that pass verification (reward >= threshold). No
# preference pairs, no reference model.

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="verified_sft"

DATA_PATH="datasets/lcb_v6"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8          # rollouts per prompt (n); matches GRPO/MOPD
MINI_BATCH_SIZE=32
LR=${LR:-1e-6}
SEED=${SEED:-42}
SFT_SUCCESS_THRESHOLD=${SFT_SUCCESS_THRESHOLD:-1.0}
SFT_LENGTH_NORMALIZE=${SFT_LENGTH_NORMALIZE:-True}
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-2048}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-4096}
ROLLOUT_CONTEXT_HEADROOM=${ROLLOUT_CONTEXT_HEADROOM:-2048}
MIN_ROLLOUT_MAX_MODEL_LEN=$((ROLLOUT_PROMPT_LENGTH + ROLLOUT_RESPONSE_LENGTH + ROLLOUT_CONTEXT_HEADROOM))
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$MIN_ROLLOUT_MAX_MODEL_LEN}
if [ "${ROLLOUT_MAX_MODEL_LEN}" -lt "${MIN_ROLLOUT_MAX_MODEL_LEN}" ]; then
    echo "Bumping ROLLOUT_MAX_MODEL_LEN to ${MIN_ROLLOUT_MAX_MODEL_LEN}."
    ROLLOUT_MAX_MODEL_LEN=${MIN_ROLLOUT_MAX_MODEL_LEN}
fi

MAX_ACTOR_CKPT_TO_KEEP=2
MAX_CRITIC_CKPT_TO_KEEP=2

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

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$VISIBLE_GPUS}
if [ "${N_GPUS_PER_NODE}" -gt "${VISIBLE_GPUS}" ]; then
    echo "N_GPUS_PER_NODE (${N_GPUS_PER_NODE}) > visible GPUs (${VISIBLE_GPUS}); clamping to ${VISIBLE_GPUS}."
    export N_GPUS_PER_NODE=${VISIBLE_GPUS}
fi

ROLLOUT_TP_SIZE=1
if [ "${ROLLOUT_TP_SIZE}" -gt "${VISIBLE_GPUS}" ]; then
    ROLLOUT_TP_SIZE=${VISIBLE_GPUS}
fi

SUFFIX=${1:-"local_sft"}

# =============================================================================
# SETUP
# =============================================================================

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

export USER=${USER:-$(whoami)}
export WANDB_ENTITY="safety"

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="${DATA_PATH##*/}-VerifiedSFT-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-thr${SFT_SUCCESS_THRESHOLD}-lr${LR}-${MODEL_NAME}-${SUFFIX}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/${EXP_NAME}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-./rollouts/${EXP_NAME}}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-./validation/${EXP_NAME}}"

ARGS=(
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.seed=$SEED"
  "data.max_prompt_length=$ROLLOUT_PROMPT_LENGTH"
  "data.max_response_length=$ROLLOUT_RESPONSE_LENGTH"
  "trainer.group_name=VerifiedSFT-local"
  "trainer.project_name=sdpo_base"
  "trainer.logger=[console,wandb]"
  "trainer.val_before_train=True"
  "trainer.test_freq=5"
  "trainer.save_freq=50"
  "trainer.default_local_dir=$CKPT_DIR"
  "trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"
  "trainer.max_critic_ckpt_to_keep=$MAX_CRITIC_CKPT_TO_KEEP"
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
  "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
  "actor_rollout_ref.rollout.name=$ROLLOUT_BACKEND"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
  "actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.val_kwargs.n=8"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "actor_rollout_ref.model.use_remove_padding=False"
  "+actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
  "+critic.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.actor.data_loader_seed=$SEED"
  "critic.data_loader_seed=$SEED"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
  "actor_rollout_ref.actor.policy_loss.loss_mode=sft"
  "actor_rollout_ref.actor.policy_loss.sft_success_threshold=$SFT_SUCCESS_THRESHOLD"
  "actor_rollout_ref.actor.policy_loss.sft_length_normalize=$SFT_LENGTH_NORMALIZE"
)

if [ "$ATTN_IMPLEMENTATION" = "flash_attention_2" ]; then
  ARGS+=(
    "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
    "critic.model.fsdp_config.model_dtype=bfloat16"
  )
fi

if [ -n "${VAL_DATA_PATH:-}" ]; then
  ARGS+=("data.val_files=['$PROJECT_ROOT/$VAL_DATA_PATH']")
fi
if [ -n "$ROLLOUT_DATA_DIR" ]; then
  ARGS+=("trainer.rollout_data_dir=$ROLLOUT_DATA_DIR")
fi
if [ -n "$VALIDATION_DATA_DIR" ]; then
  ARGS+=("trainer.validation_data_dir=$VALIDATION_DATA_DIR")
fi

echo "----------------------------------------------------------------"
echo "Starting Local Verified-Success SFT Training (filter-then-SFT)"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Rollout backend: $ROLLOUT_BACKEND | n=$ROLLOUT_BATCH_SIZE | success_threshold=$SFT_SUCCESS_THRESHOLD | lr=$LR"
echo "Rollout lengths: prompt=${ROLLOUT_PROMPT_LENGTH}, response=${ROLLOUT_RESPONSE_LENGTH}, max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
echo "Checkpoint dir: $CKPT_DIR"
echo "Resolved GPUs: visible=${VISIBLE_GPUS}, trainer.n_gpus_per_node=${N_GPUS_PER_NODE}, rollout.tp=${ROLLOUT_TP_SIZE}"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" "${ARGS[@]}"
