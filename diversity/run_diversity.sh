#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Candidate files consumed by the diversity analysis.
SDPO_CANDIDATES="${SDPO_CANDIDATES:-./rollouts/sdpo/candidate_responses.jsonl}"
MOPD_CANDIDATES="${MOPD_CANDIDATES:-./rollouts/mopd/candidate_responses.jsonl}"
BASE_CANDIDATES="${BASE_CANDIDATES:-./rollouts/base/candidate_responses.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis_outputs/diversity_lcbv6}"

# Optional eval + rollout preparation. Leave GENERATE_SDPO=0 to skip this block
# when you already have candidate_responses.jsonl on disk.
GENERATE_SDPO="${GENERATE_SDPO:-0}"
CKPT_PATH="${CKPT_PATH:-}"
DATA_PATH="${DATA_PATH:-datasets/lcb_v6}"
VAL_DATA_PATH="${VAL_DATA_PATH:-datasets/lcb_v6/test.parquet}"
EVAL_N="${EVAL_N:-8}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-}"
ROLLOUT_INPUT_DIR="${ROLLOUT_INPUT_DIR:-}"
SDPO_OUTPUT_DIR="$(dirname "$SDPO_CANDIDATES")"

if [ -z "$VALIDATION_DATA_DIR" ]; then
    VALIDATION_DATA_DIR="$PROJECT_ROOT/analysis_outputs/lcbv6_sdpo_eval_rollout"
fi

    if [ "$GENERATE_SDPO" = "1" ]; then
        if [ -z "$CKPT_PATH" ]; then
        echo "GENERATE_SDPO=1 requires CKPT_PATH to point to a checkpoint directory, actor/ subdir, or model path."
            exit 1
        fi

    if [ ! -d "$CKPT_PATH" ]; then
        echo "Checkpoint/model directory does not exist: $CKPT_PATH"
        exit 1
    fi

    CKPT_NAME="$(basename "$CKPT_PATH")"
    if [ -z "$ROLLOUT_INPUT_DIR" ]; then
        ROLLOUT_INPUT_DIR="$VALIDATION_DATA_DIR"
    fi

    CKPT_PATH="$CKPT_PATH" \
    DATA_PATH="$DATA_PATH" \
    VAL_DATA_PATH="$VAL_DATA_PATH" \
    EVAL_N="$EVAL_N" \
    VALIDATION_DATA_DIR="$VALIDATION_DATA_DIR" \
    bash run_eval_sdpo.sh "${CKPT_NAME}_eval"

    python3 analysis/prepare_rollout_dataset.py \
      --input "$ROLLOUT_INPUT_DIR" \
      --output-dir "$SDPO_OUTPUT_DIR" \
      --max-prompts 300 \
      --max-responses-per-prompt 8
fi

for path in "$SDPO_CANDIDATES" "$MOPD_CANDIDATES" "$BASE_CANDIDATES"; do
    if [ ! -f "$path" ]; then
        echo "Missing candidate file: $path"
        echo "Set the *_CANDIDATES env vars to existing files, or run with GENERATE_SDPO=1."
        exit 1
    fi
done

python3 diversity/compute_metrics.py \
  --run SDPO="$SDPO_CANDIDATES" \
  --run MOPD_V1="$MOPD_CANDIDATES" \
  --run MOPD_V2="$BASE_CANDIDATES" \
  --task code \
  --ks 1 2 4 8 \
  --intersect-prompts \
  --max-prompts 300 \
  --max-responses-per-prompt 8 \
  --output-dir "$OUTPUT_DIR"

python3 diversity/plot_metrics.py \
  --summary "$OUTPUT_DIR/summary.json" \
  --per-prompt "$OUTPUT_DIR/per_prompt.jsonl" \
  --output-dir "$OUTPUT_DIR/plots" \
  --scatter-k 8
