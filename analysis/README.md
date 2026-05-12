# Summary Signal Analysis

This directory contains a concrete offline workflow for testing whether adding more context
to the teacher prompt improves teacher signal quality.

## Goal

Answer two separate questions:

1. Does extra context improve the teacher's ranking signal on a fixed set of candidate responses?
2. If yes, does that stronger signal translate into better training?

The scripts here focus on the first question.

## Files

- `prepare_rollout_dataset.py`
  - Reads rollout JSONL dumps.
  - Groups responses by prompt.
  - Extracts `<summary>...</summary>`.
  - Computes success labels from reward.
  - Buckets prompts by difficulty.
  - Produces:
    - `candidate_responses.jsonl`
    - `evidence_items.jsonl`
    - `dataset_summary.json`
- `build_context_variants.py`
  - Reads `candidate_responses.jsonl` and `evidence_items.jsonl`.
  - Builds condition-specific teacher contexts.
  - Supports both:
    - `base_raw`: plain prompt only
    - `base_reprompt`: same reprompt shell with all optional sections empty
  - Emits `context_variants.jsonl` with assembled prompt text and evidence provenance.
- `build_target_responses.py`
  - Builds `target_responses.jsonl`.
  - Supports student-success, student-failure, and optional strong-success targets.
- `score_teacher_contexts.py`
  - Joins context variants with target responses.
  - Computes teacher conditional logprobs for each `(variant, target)` pair.
  - Emits `teacher_scores.jsonl`.
- `compute_teacher_signal_metrics.py`
  - Reads teacher-scored candidate data.
  - Computes ranking metrics by condition.
  - Produces aggregate tables in JSON.

## Execution Checklist

### Phase 1: Freeze a rollout candidate set

1. Run SDPO with rollout dumping enabled.
   - Confirm the training log prints `Rollout data dir: ...`.
   - Wait until at least one rollout JSONL file is written.

2. Build a fixed offline candidate dataset.
   - Use `prepare_rollout_dataset.py`.
   - Recommended starting point:
     - 300 prompts
     - up to 8 responses per prompt
     - keep only prompts with at least 2 responses
   - This now writes two base tables:
     - candidate responses
     - atomic evidence items

3. Inspect the bucket balance.
   - Easy: many successes
   - Medium: some successes
   - Hard: few or zero successes
   - Do not evaluate only easy prompts.

### Phase 2: Score the same responses under different teacher contexts

4. Build context variants.
   - Use `build_context_variants.py`.
   - Recommended minimum set:
     - `base_raw`
     - `base_reprompt`
     - `solution`
     - `feedback`
     - `summary_success_k2`
     - `solution+feedback`
     - `solution+summary_success_k2`
     - `solution+feedback+summary_all_k2`
     - `failure_only`
     - `random_summary_control`

5. For each condition, score the exact same `(prompt, response)` pairs.
   - Teacher model must be identical across conditions.
   - Only context changes.
   - Save one row per `(condition, prompt_id, response_id)` with:
     - `teacher_score`
     - `teacher_seq_logprob` if available
     - `teacher_avg_token_logprob` if available
     - `teacher_prompt_length` if available
     - `teacher_prompt_truncated` if available

6. Verify control variables.
   - Same candidate responses across conditions
   - Same teacher checkpoint
   - Same tokenization path
   - Same max prompt length

### Phase 3: Compute offline signal metrics

7. Run `compute_teacher_signal_metrics.py`.
   - Main metrics:
     - mean Spearman
     - mean Kendall tau
     - pairwise accuracy
     - success AUC
     - top-1 hit rate

8. Inspect prompt-length confounding.
   - Compare teacher prompt length by condition.
   - Compare truncation rate by condition.
   - If one condition wins only because it avoids truncation, note that explicitly.

9. Check negative control behavior.
   - `random_summary` should not outperform meaningful conditions.
   - If it does, you are likely measuring formatting/length effects rather than information quality.

### Phase 4: Decide whether to run training ablations

10. Only proceed to online training if offline ranking improves.
   - Recommended first training comparison:
     - `base_raw`
     - `base_reprompt`
     - `summary`
     - `key_step`

11. Run at least 3 seeds per condition.

12. Compare:
   - eval accuracy / pass rate
   - reward mean
   - sample efficiency
   - stability
   - teacher prompt truncation rate

## Output Files From `prepare_rollout_dataset.py`

### `candidate_responses.jsonl`

One row per response with:

- `prompt_id`
- `uid`
- `prompt`
- `response_id`
- `response_text`
- `reward`
- `success`
- `summary_text`
- `summary_text_effective`
- `summary_has_tag`
- `summary_is_fallback`
- `feedback_text`
- `has_feedback`
- `source_file`
- `source_line`
- `rollout_step`
- `difficulty_bucket`

### `evidence_items.jsonl`

One row per atomic context item with:

- `evidence_id`
- `prompt_id`
- `uid`
- `source_response_id`
- `evidence_type`
- `role`
- `text`
- `success`
- `reward`

The main evidence types are:

- `response_full`
- `summary`
- `environment_feedback`

### `context_variants.jsonl`

One row per `(condition, prompt_id, response_id)` with:

- `variant_id`
- `condition`
- `prompt_id`
- `response_id`
- `variant_spec`
- `sections_used`
- `evidence_ids`
- `assembled_context_text`

## Required Fields For Teacher-Scored Data

`compute_teacher_signal_metrics.py` expects one row per scored response with:

- `variant_id`
- `condition`
- `prompt_id`
- `response_id`
- `teacher_score`
- `reward`
- `success`

Optional but recommended:

- `difficulty_bucket`
- `teacher_prompt_length`
- `teacher_prompt_truncated`
- `teacher_seq_logprob`
- `teacher_avg_token_logprob`

## Recommended Aggregate Tables

### Table 1: Main ranking results

| condition | mean_spearman | mean_kendall_tau | pairwise_accuracy | success_auc | top1_hit_rate |
|---|---:|---:|---:|---:|---:|

### Table 2: Difficulty buckets

| condition | bucket | mean_spearman | pairwise_accuracy | success_auc | top1_hit_rate |
|---|---|---:|---:|---:|---:|

### Table 3: Prompt length / truncation controls

| condition | mean_prompt_length | truncation_rate | count |
|---|---:|---:|---:|

## Must-Control Confounders

- Fixed candidate responses across conditions
- Fixed teacher model/checkpoint
- Fixed tokenization and scoring logic
- Prompt length and truncation
- Success/failure composition
- Difficulty distribution
- Seed effects for online training
- Random-summary negative control
