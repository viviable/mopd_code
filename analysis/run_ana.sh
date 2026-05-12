cd "$(dirname "$(dirname "$(realpath "$0")")")"

# Set ROLLOUT_DIR to the path where your training rollouts were saved.
# Example: ROLLOUT_DIR=./rollouts/<exp_name>
ROLLOUT_DIR="${ROLLOUT_DIR:-./rollouts/your_experiment_name}"

python3 analysis/prepare_rollout_dataset.py \
  --input "$ROLLOUT_DIR" \
  --output-dir analysis_outputs/lcbv6_rollout_dataset_v2 \
  --max-prompts 300 \
  --max-responses-per-prompt 8

python3 analysis/build_context_variants.py \
  --candidates analysis_outputs/lcbv6_rollout_dataset_v2/candidate_responses.jsonl \
  --evidence analysis_outputs/lcbv6_rollout_dataset_v2/evidence_items.jsonl \
  --output analysis_outputs/lcbv6_rollout_dataset_v2.jsonl

## if there's no strong success responses, generate strong success responses first
# python3 analysis/generate_strong_success.py \
#   --candidates analysis_outputs/chemistry_rollout_dataset_v2/candidate_responses.jsonl \
#   --output analysis_outputs/chemistry_strong_success.jsonl \
#   --backend local \
#   --model qwen/qwen3.5-9b \
#   --max-prompts 100

# OPENROUTER_API_KEY=... python3 analysis/generate_strong_success.py \
#   --candidates analysis_outputs/chemistry_rollout_dataset_v2/candidate_responses.jsonl \
#   --output analysis_outputs/chemistry_strong_success.jsonl \
#   --backend openrouter \
#   --model openai/gpt-4.1-mini \
#   --max-prompts 100

python3 analysis/build_target_responses.py \
  --candidates analysis_outputs/lcbv6_rollout_dataset_v2/candidate_responses.jsonl \
  --output analysis_outputs/lcbv6_target_responses_v2.jsonl \
  --student-success-per-prompt 1 \
  --student-failure-per-prompt 1 \
  --strong-success-per-prompt 1

## for strong model success responses, build target responses
# python3 analysis/build_target_responses.py \
#   --candidates analysis_outputs/chemistry_rollout_dataset_v2/candidate_responses.jsonl \
#   --strong-success analysis_outputs/chemistry_strong_success.jsonl \
#   --output analysis_outputs/chemistry_target_responses.jsonl

### self rollout score
python3 analysis/score_teacher_contexts.py \
  --variants analysis_outputs/lcbv6_rollout_dataset_v2.jsonl \
  --targets analysis_outputs/lcbv6_rollout_dataset_v2/candidate_responses.jsonl \
  --output analysis_outputs/lcbv6_teacher_scores_self_v2.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --self-target-only \
  --condition-filter base solution another_solution failure_solution solution+failure_solution all_solutions solution+another_solution+failure_solution \
  --max-model-len 8192


# python3 analysis/score_teacher_contexts.py \
#   --variants analysis_outputs/chemistry_context_variants_v2.jsonl \
#   --targets analysis_outputs/chemistry_target_responses—wo-strong.jsonl \
#   --output analysis_outputs/chemistry_teacher_scores—wo-strong.jsonl \
#   --model Qwen/Qwen3-4B-Instruct-2507 \
#   --condition-filter base solution another_solution failure_solution solution+failure_solution all_solutions solution+another_solution+failure_solution \
#   --target-type-filter student_success student_failure \
#   --max-model-len 8192

python3 analysis/compute_teacher_signal_metrics.py \
  --input analysis_outputs/lcbv6_teacher_scores_self_v2.jsonl \
  --output analysis_outputs/lcbv6_teacher_scores_self_v2.json

### plot

python3 analysis/plot_teacher_metrics.py \
  --input analysis_outputs/lcbv6_teacher_scores_self_v2.json \
  --output-dir analysis_outputs/plots_lcbv6_self_rollout_v2 \
  --sample-set effective_only

# python3 analysis/plot_teacher_metrics.py \
#   --input analysis_outputs/teacher_metrics.json \
#   --output-dir analysis_outputs/plots_student_success_paired \
#   --sample-set all_samples \
#   --paired-target-type student_success
