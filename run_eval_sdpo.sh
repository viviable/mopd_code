#!/bin/bash

# Pure evaluation entrypoint for either:
# 1) an existing SDPO checkpoint directory (.../global_step_N), or
# 2) a plain Hugging Face model path / snapshot.
#
# In case (1), this resumes from one checkpoint, runs validation once, and dumps
# the validation generations to VALIDATION_DATA_DIR as JSONL files.
# In case (2), this evaluates the base model directly without checkpoint resume.
#
# Example:
# CKPT_PATH=./checkpoints/<exp_name>/global_step_50 \
# DATA_PATH=datasets/lcb_v6 \
# VAL_DATA_PATH=datasets/lcb_v6/test.parquet \
# VALIDATION_DATA_DIR=./validation/<exp_name>_eval \
# ./run_eval_sdpo.sh step50_eval

set -euo pipefail

CONFIG_NAME="sdpo"

CKPT_PATH="${CKPT_PATH:-}"
DATA_PATH="${DATA_PATH:-datasets/lcb_v6}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
SEED="${SEED:-42}"
EVAL_N="${EVAL_N:-8}"
EVAL_TRAIN_BATCH_SIZE="${EVAL_TRAIN_BATCH_SIZE:-32}"

ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-2048}"
ROLLOUT_RESPONSE_LENGTH="${ROLLOUT_RESPONSE_LENGTH:-4096}"
ROLLOUT_CONTEXT_HEADROOM="${ROLLOUT_CONTEXT_HEADROOM:-2048}"
MIN_ROLLOUT_MAX_MODEL_LEN=$((ROLLOUT_PROMPT_LENGTH + ROLLOUT_RESPONSE_LENGTH + ROLLOUT_CONTEXT_HEADROOM))
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$MIN_ROLLOUT_MAX_MODEL_LEN}"

DEFAULT_ATTN_IMPLEMENTATION="flash_attention_2"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-$DEFAULT_ATTN_IMPLEMENTATION}"

if [ "${ROLLOUT_MAX_MODEL_LEN}" -lt "${MIN_ROLLOUT_MAX_MODEL_LEN}" ]; then
    echo "ROLLOUT_MAX_MODEL_LEN (${ROLLOUT_MAX_MODEL_LEN}) is too small; bumping to ${MIN_ROLLOUT_MAX_MODEL_LEN}."
    ROLLOUT_MAX_MODEL_LEN=${MIN_ROLLOUT_MAX_MODEL_LEN}
fi

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

export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-$VISIBLE_GPUS}"
if [ "${N_GPUS_PER_NODE}" -gt "${VISIBLE_GPUS}" ]; then
    echo "N_GPUS_PER_NODE (${N_GPUS_PER_NODE}) > visible GPUs (${VISIBLE_GPUS}); clamping to ${VISIBLE_GPUS}."
    export N_GPUS_PER_NODE="${VISIBLE_GPUS}"
fi

ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
if [ "${ROLLOUT_TP_SIZE}" -gt "${VISIBLE_GPUS}" ]; then
    echo "ROLLOUT_TP_SIZE (${ROLLOUT_TP_SIZE}) > visible GPUs (${VISIBLE_GPUS}); clamping to ${VISIBLE_GPUS}."
    ROLLOUT_TP_SIZE="${VISIBLE_GPUS}"
fi

SUFFIX="${1:-eval}"

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"
export WANDB_ENTITY="safety"

if [ -z "$CKPT_PATH" ]; then
    echo "CKPT_PATH must be set to either a checkpoint directory (.../global_step_N) or a model path."
    exit 1
fi

EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/analysis_outputs/sdpo_eval}"
mkdir -p "$EVAL_ROOT"

RESUME_MODE="disable"
MODEL_SOURCE="$MODEL_PATH"
INPUT_BASENAME="$(basename "$CKPT_PATH")"
INPUT_TAG="$(printf '%s' "$CKPT_PATH" | sha1sum | awk '{print $1}' | cut -c1-10)"

if [ ! -e "$CKPT_PATH" ]; then
    echo "Path does not exist: $CKPT_PATH"
    exit 1
fi

CKPT_BASENAME="$(basename "$CKPT_PATH")"
CKPT_PARENT_BASENAME="$(basename "$(dirname "$CKPT_PATH")")"

if [[ "$CKPT_BASENAME" == "actor" ]] || [[ "$CKPT_BASENAME" == "critic" ]]; then
    if [[ "$CKPT_PARENT_BASENAME" == global_step_* ]]; then
        CKPT_PATH="$(dirname "$CKPT_PATH")"
        CKPT_BASENAME="$(basename "$CKPT_PATH")"
        CKPT_PARENT_BASENAME="$(basename "$(dirname "$CKPT_PATH")")"
    fi
fi

if [[ "$CKPT_BASENAME" == global_step_* ]]; then
    if [ ! -d "$CKPT_PATH" ]; then
        echo "Checkpoint directory does not exist: $CKPT_PATH"
        exit 1
    fi
    RESUME_MODE="resume_path"
    MODEL_SOURCE="$MODEL_PATH"
    INPUT_BASENAME="$CKPT_BASENAME"
else
    RESUME_MODE="disable"
    MODEL_SOURCE="$CKPT_PATH"
    if [ -f "$CKPT_PATH/config.json" ] || [ -f "$CKPT_PATH/generation_config.json" ]; then
        INPUT_BASENAME="$(basename "$CKPT_PATH")"
    else
        echo "CKPT_PATH does not look like a global_step checkpoint or a model directory: $CKPT_PATH"
        exit 1
    fi
fi

MODEL_NAME=$(echo "$MODEL_SOURCE" | tr '/.' '--' | tr -s '-')
EXP_NAME="EVAL-${DATA_PATH##*/}-${MODEL_NAME}-${INPUT_BASENAME}-${INPUT_TAG}-${SUFFIX}"

DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$EVAL_ROOT/eval_runs/${EXP_NAME}}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$EVAL_ROOT/eval_rollout/${EXP_NAME}}"

ARGS=(
  "data.train_batch_size=$EVAL_TRAIN_BATCH_SIZE"
  "data.seed=$SEED"
  "data.max_prompt_length=$ROLLOUT_PROMPT_LENGTH"
  "data.max_response_length=$ROLLOUT_RESPONSE_LENGTH"
  "trainer.group_name=SDPO-eval"
  "trainer.project_name=sdpo_eval"
  "trainer.experiment_name=$EXP_NAME"
  "trainer.logger=[console]"
  "trainer.default_local_dir=$DEFAULT_LOCAL_DIR"
  "trainer.resume_mode=$RESUME_MODE"
  "trainer.val_before_train=True"
  "trainer.val_only=True"
  "trainer.test_freq=-1"
  "trainer.save_freq=-1"
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
  "trainer.validation_data_dir=$VALIDATION_DATA_DIR"
  "actor_rollout_ref.rollout.n=$EVAL_N"
  "actor_rollout_ref.rollout.val_kwargs.n=$EVAL_N"
  "actor_rollout_ref.rollout.name=$ROLLOUT_BACKEND"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
  "actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_MODEL_LEN"
  "actor_rollout_ref.model.path=$MODEL_SOURCE"
  "actor_rollout_ref.model.use_remove_padding=False"
  "+actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
  "+critic.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
)

if [ "$RESUME_MODE" = "resume_path" ]; then
  ARGS+=("trainer.resume_from_path=$CKPT_PATH")
fi

if [[ "${MODEL_SOURCE}" == Qwen/Qwen3-* ]] || [[ "${MODEL_SOURCE}" == *Qwen3* ]]; then
  ARGS+=("++data.apply_chat_template_kwargs.enable_thinking=True")
fi

if [ "$ATTN_IMPLEMENTATION" = "flash_attention_2" ]; then
  ARGS+=(
    "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
    "critic.model.fsdp_config.model_dtype=bfloat16"
  )
fi

if [ -n "$VAL_DATA_PATH" ]; then
  ARGS+=("data.val_files=['$PROJECT_ROOT/$VAL_DATA_PATH']")
fi

echo "----------------------------------------------------------------"
echo "Starting SDPO evaluation"
echo "Input path: $CKPT_PATH"
echo "Resume mode: $RESUME_MODE"
echo "Data: $DATA_PATH"
echo "Validation data: ${VAL_DATA_PATH:-${DATA_PATH}/test.parquet}"
echo "Model: $MODEL_SOURCE"
echo "Eval samples per prompt: $EVAL_N"
echo "Validation dump dir: $VALIDATION_DATA_DIR"
echo "Default local dir: $DEFAULT_LOCAL_DIR"
echo "Resolved GPUs: visible=${VISIBLE_GPUS}, trainer.n_gpus_per_node=${N_GPUS_PER_NODE}, rollout.tp=${ROLLOUT_TP_SIZE}"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" "${ARGS[@]}"
