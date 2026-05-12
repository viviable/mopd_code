#!/bin/bash

# Usage: ./run_local_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

# Default to ToolUse dataset
# DATA_PATH="datasets/tooluse"
DATA_PATH="datasets/sciknoweval/biology"
DATA_PATH="datasets/sciknoweval/biology"

# DATA_PATH="datasets/lcb_v6"
# Optional validation override. Useful when training on one dataset but validating
# on a separate benchmark parquet, e.g. Humaneval+/MBPP+.
# VAL_DATA_PATH="datasets/humanevalplus/humanevalplus/test.parquet"
# VAL_DATA_PATH="datasets/gsm8k/test.parquet"
# VAL_DATA_PATH="datasets/mbppplus/test.parquet"
# VAL_DATA_PATH="${VAL_DATA_PATH:-}"

# Hyperparameters (from experiments/run_sdpo_all.sh)
TRAIN_BATCH_SIZE=4
ROLLOUT_BATCH_SIZE=2
LR=1e-5
SEED=${SEED:-42}
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
TEACHER_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"

DEFAULT_ATTN_IMPLEMENTATION="flash_attention_2"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-$DEFAULT_ATTN_IMPLEMENTATION}"

ROLLOUT_MAX_MODEL_LEN=18944

# TEACHER_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
INCLUDE_ANOTHER_SOLUTION=False
INCLUDE_FAILURE_SOLUTION=False
SUMMARIZE_SOLUTIONS=False
SUMMARY_FROM_ALL=False   # 新增
SUMMARY_K=8

SAVE_FREQ=${SAVE_FREQ:-50}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-2}

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
SUFFIX=${1:-"local_sdpo"}

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

# Bootstrap the canonical GSM8K parquet layout when requested.
if [ "$DATA_PATH" = "datasets/gsm8k" ]; then
    GSM8K_TRAIN_FILE="$PROJECT_ROOT/$DATA_PATH/train.parquet"
    GSM8K_TEST_FILE="$PROJECT_ROOT/$DATA_PATH/test.parquet"

    if [ ! -f "$GSM8K_TRAIN_FILE" ] || [ ! -f "$GSM8K_TEST_FILE" ]; then
        echo "GSM8K parquet files not found under $PROJECT_ROOT/$DATA_PATH. Generating them from openai/gsm8k..."
        mkdir -p "$PROJECT_ROOT/$DATA_PATH"
        python "$PROJECT_ROOT/examples/data_preprocess/gsm8k.py" --local_save_dir "$PROJECT_ROOT/$DATA_PATH"
    fi
fi

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}
export WANDB_ENTITY="safety"

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
DATA_NAME="${DATA_PATH##*/}"
EXP_NAME="107361-0-TS-${DATA_NAME}-${TEACHER_MODEL_PATH}-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/${EXP_NAME}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-./rollouts/${EXP_NAME}}"

ARGS=(
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.seed=$SEED"
  "data.max_prompt_length=2048"
  "trainer.group_name=SDPO-local"
  "trainer.project_name=sdpo_base"
  "trainer.logger=[console,wandb]"
  "trainer.val_before_train=True"
  "trainer.test_freq=5"
  "trainer.save_freq=$SAVE_FREQ"
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
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "+actor_rollout_ref.ref.model.path=$TEACHER_MODEL_PATH"
  "actor_rollout_ref.model.use_remove_padding=False"
  "+actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
  "+critic.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "actor_rollout_ref.actor.data_loader_seed=$SEED"
  "critic.data_loader_seed=$SEED"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
  "algorithm.rollout_correction.rollout_is=token"
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS}"
  "actor_rollout_ref.actor.self_distillation.alpha=$ALPHA"
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0"
  "actor_rollout_ref.actor.self_distillation.include_another_solution=$INCLUDE_ANOTHER_SOLUTION"
  "actor_rollout_ref.actor.self_distillation.include_failure_solution=$INCLUDE_FAILURE_SOLUTION"
  "actor_rollout_ref.actor.self_distillation.summarize_solutions=$SUMMARIZE_SOLUTIONS"
  "actor_rollout_ref.actor.self_distillation.summary_k=$SUMMARY_K"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.rollout.val_kwargs.n=8"
)

# FSDP initializes trainable modules in fp32 by default in this repo. That is
# incompatible with FlashAttention-2, which only supports fp16/bf16.
if [ "$ATTN_IMPLEMENTATION" = "flash_attention_2" ]; then
  ARGS+=(
    "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
    "critic.model.fsdp_config.model_dtype=bfloat16"
  )
fi

if [ -n "$VAL_DATA_PATH" ]; then
  ARGS+=("data.val_files=['$PROJECT_ROOT/$VAL_DATA_PATH']")
fi

if [ -n "$ROLLOUT_DATA_DIR" ]; then
  ARGS+=("trainer.rollout_data_dir=$ROLLOUT_DATA_DIR")
fi

echo "----------------------------------------------------------------"
echo "Starting Local SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Validation data: ${VAL_DATA_PATH:-${DATA_PATH}/test.parquet}"
echo "Model: $MODEL_PATH"
echo "Teacher model: $TEACHER_MODEL_PATH"
echo "Rollout backend: $ROLLOUT_BACKEND"
echo "Attention implementation: $ATTN_IMPLEMENTATION"
echo "Seed: $SEED"
echo "Checkpoint dir: $CKPT_DIR"
echo "Rollout data dir: ${ROLLOUT_DATA_DIR:-disabled}"
echo "Resolved GPUs: visible=${VISIBLE_GPUS}, trainer.n_gpus_per_node=${N_GPUS_PER_NODE}, rollout.tp=${ROLLOUT_TP_SIZE}"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" "${ARGS[@]}"
