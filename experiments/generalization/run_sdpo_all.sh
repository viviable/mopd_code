#!/bin/bash

# Usage: ./run_sdpo_all.sh [--dry-run] [--local] [--skip-install]

DRY_RUN=false
LOCAL_MODE=false
SKIP_INSTALL=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        --local)
            LOCAL_MODE=true
            ;;
        --skip-install)
            SKIP_INSTALL=true
            ;;
    esac
done
if [ "$DRY_RUN" = true ]; then
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi
if [ "$LOCAL_MODE" = true ]; then
    echo "Local mode enabled. Jobs will run directly on current node (no sbatch)."
fi
if [ "$SKIP_INSTALL" = true ]; then
    echo "Skip-install mode enabled. pip install steps will be skipped."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get the directory where this script is located
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

# Base settings
CONFIG_NAME="sdpo"
BASE_JOB_NAME="rlvr"

DATA_PATHS=(
    "datasets/sciknoweval/biology/"
    "datasets/sciknoweval/chemistry/"
    "datasets/sciknoweval/material/"
    "datasets/sciknoweval/physics/"
    "datasets/tooluse"
)

# Fixed Slurm resources
ACCOUNT="a156"
NODES=1
PARTITION="normal"
TIME="12:00:00"
ENV="sdpo"
NTASKS_PER_NODE=1
GPUS_PER_NODE=4
MEM=460000
CPUS_PER_TASK=288

# Sweep Parameters
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
LRS=(1e-5)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)

# 0: forward KL, 0.5: Jensen-Shannon divergence, 1: reverse KL
ALPHAS=(0.5)

MODEL_PATHS=(
    "Qwen/Qwen3-8B"
    "allenai/Olmo-3-7B-Instruct"
)
# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    # Define the environment setup and command execution
    local setup_cmds="pip install word2number latex2sympy2 math-verify[antlr4_9_3]==0.8.0; \
pip install -e $PROJECT_ROOT; \
pip install --upgrade wandb; \
export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH"
    if [ "$SKIP_INSTALL" = true ]; then
        setup_cmds="export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH"
    fi

    local run_cmd="bash $PROJECT_ROOT/training/verl_training.sh $exp_name $CONFIG_NAME $data_path $script_args"

    if [ "$LOCAL_MODE" = true ]; then
        local local_cmd="bash -c '$setup_cmds; $run_cmd'"
        if [ "$DRY_RUN" = true ]; then
            echo "----------------------------------------------------------------"
            echo "Would run locally for: $exp_name"
            echo "$local_cmd"
        else
            echo "Running locally for: $exp_name"
            eval "$local_cmd"
        fi
        return
    fi

    local wrapped_cmd="srun bash -c '$setup_cmds; $run_cmd'"

    local sbatch_cmd=(
        sbatch
        --job-name="$BASE_JOB_NAME"
        --account="$ACCOUNT"
        --nodes="$NODES"
        --partition="$PARTITION"
        --time="$TIME"
        --ntasks-per-node="$NTASKS_PER_NODE"
        --gpus-per-node="$GPUS_PER_NODE"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="/users/$USER/output/SDPO/%j.log"
        --error="/users/$USER/output/SDPO/%j.err"
        --wrap="$wrapped_cmd"
    )

    # Some Slurm versions do not support --environment.
    if sbatch --help 2>&1 | grep -q -- "--environment"; then
        sbatch_cmd+=(--environment="$ENV")
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "----------------------------------------------------------------"
        echo "Would submit job for: $exp_name"
        echo "${sbatch_cmd[@]}"
    else
        echo "Submitting job for: $exp_name"
        "${sbatch_cmd[@]}"
    fi
}

# =============================================================================
# MAIN SWEEP LOOP
# =============================================================================

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for DONTS_REPROMPT_ON_SELF_SUCCESS in "${DONTS_REPROMPT_ON_SELF_SUCCESSS[@]}"; do
                for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                    for ALPHA in "${ALPHAS[@]}"; do
                        for DATA_PATH in "${DATA_PATHS[@]}"; do
                            # 1. Construct the experiment name (must be unique)
                            MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                            EXP_NAME="FINAL-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}"

                            # 2. Construct the arguments string to pass to the training script
                            # Format: key=value key2=value2 ...
                            ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-generalization \
vars.dir=$PROJECT_ROOT \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 "

                            # 3. Submit
                            submit_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
                        done
                    done
                done
            done
        done
    done
done
