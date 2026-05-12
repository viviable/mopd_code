#!/bin/bash

# Usage: ./run_local_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

# Default to ToolUse dataset
# DATA_PATH="datasets/tooluse"
# DATA_PATH="datasets/sciknoweval/biology"
# DATA_PATH="datasets/sciknoweval/chemistry"
# DATA_PATH="datasets/sciknoweval/physics"
# DATA_PATH="datasets/sciknoweval/material"

DATA_PATH="datasets/lcb_v6"
# DATA_PATH="datasets/G-OPD-Training-Data/Eurus"
# DATA_PATH="datasets/gsm8k"
# DATA_PATH="datasets/G-OPD-Training-Data/DeepMath-103K_test_aime2025"
# DATA_PATH="datasets/G-OPD-Training-Data/DeepMath-103K_test_hmmt2025_feb"
# DATA_PATH="datasets/G-OPD-Training-Data/opd-math"
# Optional validation override. Useful when training on one dataset but validating
# on a separate benchmark parquet, e.g. Humaneval+/MBPP+.
# VAL_DATA_PATH="datasets/humanevalplus/humanevalplus/test.parquet"
# VAL_DATA_PATH="datasets/gsm8k/test.parquet"
# VAL_DATA_PATH="datasets/mbppplus/test.parquet"
# VAL_DATA_PATH="${VAL_DATA_PATH:-}"

# Hyperparameters (from experiments/run_sdpo_all.sh)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
SEED=${SEED:-42}
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=${DONTS_REPROMPT_ON_SELF_SUCCESS:-True}
ALPHA=0.5
INCLUDE_PRIMARY_SOLUTION=${INCLUDE_PRIMARY_SOLUTION:-True}
FAILURE_SOLUTION_CONDITION=${FAILURE_SOLUTION_CONDITION:-always}
SUMMARY_SOURCE=${SUMMARY_SOURCE:-success}
# MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
# MODEL_PATH="Qwen/Qwen3-4B"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
# MODEL_PATH="Qwen/Qwen3.5-4B"
# MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"

# Qwen3.5 needs a newer attention stack than Qwen3. Default to SDPA there unless
# the environment explicitly opts back into FlashAttention-2.
DEFAULT_ATTN_IMPLEMENTATION="flash_attention_2"
# if [[ "${MODEL_PATH}" == Qwen/Qwen3.5-* ]]; then
#     DEFAULT_ATTN_IMPLEMENTATION="sdpa"
# fi
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-$DEFAULT_ATTN_IMPLEMENTATION}"

ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-2048}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-4096}
# SDPO prompts can include extra teacher / failure context, so leave some
# headroom beyond prompt+response for the rendered template and auxiliary text.
ROLLOUT_CONTEXT_HEADROOM=${ROLLOUT_CONTEXT_HEADROOM:-2048}
MIN_ROLLOUT_MAX_MODEL_LEN=$((ROLLOUT_PROMPT_LENGTH + ROLLOUT_RESPONSE_LENGTH + ROLLOUT_CONTEXT_HEADROOM))
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$MIN_ROLLOUT_MAX_MODEL_LEN}

if [ "${ROLLOUT_MAX_MODEL_LEN}" -lt "${MIN_ROLLOUT_MAX_MODEL_LEN}" ]; then
    echo "ROLLOUT_MAX_MODEL_LEN (${ROLLOUT_MAX_MODEL_LEN}) is too small for SDPO local defaults."
    echo "Need at least prompt (${ROLLOUT_PROMPT_LENGTH}) + response (${ROLLOUT_RESPONSE_LENGTH}) + headroom (${ROLLOUT_CONTEXT_HEADROOM}) = ${MIN_ROLLOUT_MAX_MODEL_LEN}."
    echo "Bumping ROLLOUT_MAX_MODEL_LEN to ${MIN_ROLLOUT_MAX_MODEL_LEN}."
    ROLLOUT_MAX_MODEL_LEN=${MIN_ROLLOUT_MAX_MODEL_LEN}
fi


INCLUDE_ANOTHER_SOLUTION=${INCLUDE_ANOTHER_SOLUTION:-True}
INCLUDE_FAILURE_SOLUTION=${INCLUDE_FAILURE_SOLUTION:-True}
SUMMARIZE_SOLUTIONS=${SUMMARIZE_SOLUTIONS:-False}
SUMMARY_FROM_ALL=${SUMMARY_FROM_ALL:-False}
SUMMARY_K=${SUMMARY_K:-8}

MAX_ACTOR_CKPT_TO_KEEP=2
MAX_CRITIC_CKPT_TO_KEEP=2

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

# if [ "${ROLLOUT_BACKEND}" = "vllm" ] && [[ "${MODEL_PATH}" == Qwen/Qwen3.5-* ]]; then
#     cat <<'EOF'
# Unsupported local combo detected:
# - rollout backend: vllm
# - model: Qwen3.5

# This repo's async rollout path uses the vLLM V1 engine, and vllm==0.8.5.post1 does not support
# Qwen3.5 architectures here. Use one of these instead:
# - MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 ./run_local_sdpo.sh
# - MODEL_PATH=Qwen/Qwen2.5-3B-Instruct ./run_local_sdpo.sh
# - upgrade vllm to a version that supports Qwen3.5 in this async path
# - switch to sglang if your environment supports it: ROLLOUT_BACKEND=sglang ./run_local_sdpo.sh
# EOF
#     exit 1
# fi

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
TEACHER_CONTEXT_TAG="P${INCLUDE_PRIMARY_SOLUTION}-A${INCLUDE_ANOTHER_SOLUTION}-F${INCLUDE_FAILURE_SOLUTION}-FC${FAILURE_SOLUTION_CONDITION}-S${SUMMARIZE_SOLUTIONS}"
EXP_NAME="Seed2-${TEACHER_CONTEXT_TAG}-${DATA_PATH##*/}-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/${EXP_NAME}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-./rollouts/${EXP_NAME}}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-./validation/${EXP_NAME}}"

ARGS=(
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.seed=$SEED"
  "data.max_prompt_length=$ROLLOUT_PROMPT_LENGTH"
  "data.max_response_length=$ROLLOUT_RESPONSE_LENGTH"
  "trainer.group_name=SDPO-local"
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
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "actor_rollout_ref.model.use_remove_padding=False"
  "+actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
  "+critic.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "actor_rollout_ref.actor.data_loader_seed=$SEED"
  "critic.data_loader_seed=$SEED"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
  "algorithm.rollout_correction.rollout_is=token"
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS}"
  "actor_rollout_ref.actor.self_distillation.alpha=$ALPHA"
  "actor_rollout_ref.actor.self_distillation.include_primary_solution=${INCLUDE_PRIMARY_SOLUTION}"
  "actor_rollout_ref.actor.self_distillation.include_another_solution=$INCLUDE_ANOTHER_SOLUTION"
  "actor_rollout_ref.actor.self_distillation.include_failure_solution=$INCLUDE_FAILURE_SOLUTION"
  "actor_rollout_ref.actor.self_distillation.failure_solution_condition=${FAILURE_SOLUTION_CONDITION}"
  "actor_rollout_ref.actor.self_distillation.summarize_solutions=$SUMMARIZE_SOLUTIONS"
  "actor_rollout_ref.actor.self_distillation.summary_source=${SUMMARY_SOURCE}"
  "actor_rollout_ref.actor.self_distillation.summary_from_all=$SUMMARY_FROM_ALL"
  "actor_rollout_ref.actor.self_distillation.summary_k=$SUMMARY_K"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.rollout.val_kwargs.n=8"
)

# if [[ "${MODEL_PATH}" == Qwen/Qwen3-* ]]; then
#   ARGS+=("++data.apply_chat_template_kwargs.enable_thinking=True")
# fi

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

if [ -n "$VALIDATION_DATA_DIR" ]; then
  ARGS+=("trainer.validation_data_dir=$VALIDATION_DATA_DIR")
fi

echo "----------------------------------------------------------------"
echo "Starting Local SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Validation data: ${VAL_DATA_PATH:-${DATA_PATH}/test.parquet}"
echo "Model: $MODEL_PATH"
echo "Rollout backend: $ROLLOUT_BACKEND"
echo "Attention implementation: $ATTN_IMPLEMENTATION"
echo "Seed: $SEED"
echo "Checkpoint dir: $CKPT_DIR"
echo "Rollout lengths: prompt=${ROLLOUT_PROMPT_LENGTH}, response=${ROLLOUT_RESPONSE_LENGTH}, headroom=${ROLLOUT_CONTEXT_HEADROOM}, max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
echo "Summary from all: $SUMMARY_FROM_ALL"
echo "Teacher context: primary=${INCLUDE_PRIMARY_SOLUTION}, another=${INCLUDE_ANOTHER_SOLUTION}, failure=${INCLUDE_FAILURE_SOLUTION}, failure_condition=${FAILURE_SOLUTION_CONDITION}, summarize=${SUMMARIZE_SOLUTIONS}"
echo "Rollout data dir: ${ROLLOUT_DATA_DIR:-disabled}"
echo "Validation data dir: ${VALIDATION_DATA_DIR:-disabled}"
echo "Resolved GPUs: visible=${VISIBLE_GPUS}, trainer.n_gpus_per_node=${N_GPUS_PER_NODE}, rollout.tp=${ROLLOUT_TP_SIZE}"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" "${ARGS[@]}"
