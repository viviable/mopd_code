#!/bin/bash

# Usage: ./run_local_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

# Default to ToolUse dataset
DATA_PATH="datasets/tooluse"

# Hyperparameters (from experiments/run_sdpo_all.sh)
TRAIN_BATCH_SIZE=8
ROLLOUT_BATCH_SIZE=2
LR=1e-5
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
export N_GPUS_PER_NODE=1

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_sdpo"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}
export WANDB_ENTITY="safety"

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="STEP-LOCAL-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.max_prompt_length=512 \
data.max_response_length=512 \
max_model_len=2048 \
trainer.group_name=SDPO-local \
trainer.project_name=sdpo_base \
trainer.logger=[console,wandb] \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
actor_rollout_ref.rollout.max_num_seqs=4 \
actor_rollout_ref.rollout.max_model_len=2048 \
actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.use_remove_padding=False \
+actor_rollout_ref.model.override_config.attn_implementation=sdpa \
+critic.model.override_config.attn_implementation=sdpa \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=8 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.actor.self_distillation.step_level_kl=True \
actor_rollout_ref.actor.self_distillation.step_separator='\\n\\n'"

echo "----------------------------------------------------------------"
echo "Starting Local SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
